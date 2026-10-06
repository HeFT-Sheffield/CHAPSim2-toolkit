import glob
import os
import re
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm

# =====================================================================================================================================================
# XDMF READING CONFIGURATION
# =====================================================================================================================================================

# Set to True to load all variables, False to load only required ones (faster)
LOAD_ALL_VARS = False

# Variables required for computation (base names without prefixes)
_BASE_REQUIRED_VARS = [
    # Velocities
    'u1', 'u2', 'u3',
    # Reynolds stresses
    'uu11', 'uu12', 'uu13', 'uu22', 'uu23', 'uu33',
    # Triple correlations
    'uuu111', 'uuu112', 'uuu113', 'uuu122', 'uuu123', 'uuu133',
    'uuu222', 'uuu223', 'uuu233', 'uuu333',
    # Velocity gradients
    'dudx11', 'dudx12', 'dudx13', 'dudx21', 'dudx22', 'dudx23', 'dudx31', 'dudx32', 'dudx33',
    # Dissipation terms
    'dudu11', 'dudu12', 'dudu13', 'dudu22', 'dudu23', 'dudu33',
    # Pressure terms
    'pr', 'pru1', 'pru2', 'pru3',
    'prdu11', 'prdu12', 'prdu13', 'prdu21', 'prdu22', 'prdu23', 'prdu31', 'prdu32', 'prdu33',
    # Density / volume fraction (variable properties)
    'f',
    'fu1', 'fu2', 'fu3',
    'fuu11', 'fuu12', 'fuu13', 'fuu22', 'fuu23', 'fuu33',
    'fuuu111', 'fuuu112', 'fuuu113', 'fuuu122', 'fuuu123', 'fuuu133',
    'fuuu222', 'fuuu223', 'fuuu233', 'fuuu333',
    'fuh1', 'fuh2', 'fuh3',
    'fuuh11', 'fuuh12', 'fuuh13', 'fuuh22', 'fuuh23', 'fuuh33',
    'fh',
    # Temperature
    'T', 'TT', 'Tu1', 'Tu2', 'Tu3',
    # MHD
    'e',
    'j1', 'j2', 'j3',
    'ej1', 'ej2', 'ej3',
    'jj11', 'jj12', 'jj13', 'jj22', 'jj23', 'jj33',
    'eu1', 'eu2', 'eu3',
    'ju11', 'ju12', 'ju13', 'ju21', 'ju22', 'ju23', 'ju31', 'ju32', 'ju33',
]

# Build full set including prefixed versions (tsp_avg_, t_avg_)
REQUIRED_VARS = set(_BASE_REQUIRED_VARS)
for prefix in ['tsp_avg_', 't_avg_']:
    REQUIRED_VARS.update(f'{prefix}{var}' for var in _BASE_REQUIRED_VARS)

# =====================================================================================================================================================
# =====================================================================================================================================================
# THERMAL PROPERTIES CLASSES
# =====================================================================================================================================================

# The seven liquid-metal property classes used to live here, each one a
# hand-copy of the correlations in CHAPSim2/src/modules.f90. They have moved
# to fluid_properties, which ports the solver's whole model - both the
# correlations and the NIST tables for supercritical water and CO2, which
# this file never had. Keeping a second copy here is what let the two drift.
#
# The names are kept because thermal_BC_calc and older scripts import them.

def _fluid_alias(key, label):
    def make():
        return get_fluid_properties(key)
    make.__name__ = f'Liquid{label}Properties'
    make.__doc__ = (f"Properties of liquid {key}. Deprecated alias for "
                    f"get_fluid_properties({key!r}).")
    return make


LiquidLithiumProperties = _fluid_alias('lithium', 'Lithium')
LiquidPbLiProperties = _fluid_alias('pbli', 'PbLi')
LiquidSodiumProperties = _fluid_alias('sodium', 'Sodium')
LiquidLeadProperties = _fluid_alias('lead', 'Lead')
LiquidBismuthProperties = _fluid_alias('bismuth', 'Bismuth')
LiquidLBEProperties = _fluid_alias('lbe', 'LBE')
LiquidFLiBeProperties = _fluid_alias('flibe', 'FLiBe')


def get_fluid_properties(medium, case_dir=None):
    """Return a thermal properties object for the requested medium name.

    Kept here because the toolkit has always imported it from utils; the
    properties themselves now live in fluid_properties, which ports the
    solver's own model - including the NIST tables for supercritical
    water and CO2, which this never had.

    Pass case_dir for a supercritical case so the table the run used is
    the table the post-processing uses.
    """
    from fluid_properties import get_fluid_properties as _get
    return _get(medium, case_dir=case_dir)


# =====================================================================================================================================================
# INTERACTIVE PROMPTS
# =====================================================================================================================================================

def ask_number(prompt, default=None, cast=float):
    """Read a number from the terminal, re-asking rather than giving up.

    The interactive scripts ask for several numbers in a row, often after
    something slow has already been chosen. A mistyped character used to
    raise ValueError out of the prompt and end the session; this re-asks
    instead. A blank line takes the default, and so does EOF, which is what
    a piped-in list of answers runs out of.
    """
    suffix = '' if default is None else f' [{default}]'
    while True:
        try:
            text = input(f'{prompt}{suffix}: ').strip()
        except EOFError:
            return default
        if not text:
            return default
        try:
            return cast(text)
        except ValueError:
            print(f"  '{text}' is not a number. Try again, "
                  f"or press enter for {default}.")


# CHAPSim2 OUTPUT LAYOUT
# =====================================================================================================================================================
#
# A CHAPSim2 case directory looks like this:
#
#   1_data/      restart/checkpoint binaries, spectra, stats bundles
#   2_visu/
#       xdmf/    .xdmf descriptors
#       data/    field binaries (.bin) and bundle metadata (*_meta_*.dat,
#                *_?profile_*.dat)
#       mesh/    grid coordinate binaries
#   3_monitor/   monitor point files and history logs
#   4_check/     mesh and property check tables
#
# Older CHAPSim2 builds kept every visualisation file flat in 2_visu/ with the
# binaries in 1_data/.  resolve_case_dirs() accepts either and reports where
# each kind of file actually lives, so the rest of the toolkit never has to
# care which layout it was handed.

_SLICE_LABEL_RE = re.compile(r'^([xyz])i(\d+)$')
_SLICE_IN_NAME_RE = re.compile(r'([xyz]i\d+)')
_SLICE_FILE_RE = re.compile(r'_([xyz]i\d+)_(\d+)\.xdmf$')
_XML_DECL_RE = re.compile(r'<\?xml[^?]*\?>')

#: Physics groups CHAPSim2 writes visualisation output for.
VISU_GROUPS = ('flow', 'thermo', 'mhd')


def resolve_case_dirs(path):
    """Locate the standard CHAPSim2 output directories for a case.

    Args:
        path: A case directory, or its ``2_visu`` / ``2_visu/xdmf`` /
            ``2_visu/data`` / ``2_visu/mesh`` / ``1_data`` subdirectory.  Any of
            these identify the same case.

    Returns:
        dict with keys ``case``, ``visu``, ``xdmf``, ``data``, ``mesh``,
        ``raw`` (1_data), ``monitor`` (3_monitor) and ``check`` (4_check).
        Entries always hold a path; ``xdmf``/``data``/``mesh`` collapse onto
        ``2_visu`` itself for the old flat layout.
    """
    base = os.path.normpath(os.path.expanduser(os.path.expandvars(str(path).strip())))

    # Walk up from any of the known subdirectories to the case root.
    name = os.path.basename(base)
    parent = os.path.dirname(base)
    if name in ('xdmf', 'data', 'mesh') and os.path.basename(parent) == '2_visu':
        base = os.path.dirname(parent)
    elif name in ('2_visu', '1_data', '3_monitor', '4_check'):
        base = parent

    visu = os.path.join(base, '2_visu')
    nested = {k: os.path.join(visu, k) for k in ('xdmf', 'data', 'mesh')}
    # The nested layout is in use as soon as 2_visu/xdmf exists; otherwise fall
    # back to the flat layout where everything sat directly in 2_visu.
    if os.path.isdir(nested['xdmf']):
        xdmf_dir, data_dir, mesh_dir = (nested['xdmf'], nested['data'], nested['mesh'])
    else:
        xdmf_dir = data_dir = mesh_dir = visu

    return {
        'case': base,
        'visu': visu,
        'xdmf': xdmf_dir,
        'data': data_dir,
        'mesh': mesh_dir,
        'raw': os.path.join(base, '1_data'),
        'monitor': os.path.join(base, '3_monitor'),
        'check': os.path.join(base, '4_check'),
    }


def xdmf_folder(path):
    """Return the directory holding a case's .xdmf files."""
    return resolve_case_dirs(path)['xdmf']


_BIN_SEARCH_DIRS = {}


def _binary_search_dirs(xdmf_dir):
    """Ordered fallback directories for a binary referenced by an XDMF file."""
    cached = _BIN_SEARCH_DIRS.get(xdmf_dir)
    if cached is not None:
        return cached

    dirs = resolve_case_dirs(xdmf_dir)
    candidates = [xdmf_dir, dirs['data'], dirs['mesh'], dirs['visu'], dirs['raw']]
    # Preserve order, drop duplicates and anything that is not present.
    seen = set()
    ordered = []
    for d in candidates:
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            ordered.append(d)
    _BIN_SEARCH_DIRS[xdmf_dir] = ordered
    return ordered


def resolve_binary_path(ref, xdmf_dir):
    """Resolve a binary reference from an XDMF DataItem to a real file.

    CHAPSim2 writes paths relative to the XDMF file (``../data/x.bin``,
    ``../mesh/y.bin``), so that is tried first.  If the tree has been moved or
    flattened the basename is looked up in the case's other output
    directories.

    Returns:
        str path, or None when no candidate exists.
    """
    if not ref:
        return None

    direct = os.path.normpath(os.path.join(xdmf_dir, ref))
    if os.path.isfile(direct):
        return direct

    basename = os.path.basename(ref)
    for directory in _binary_search_dirs(xdmf_dir):
        candidate = os.path.join(directory, basename)
        if os.path.isfile(candidate):
            return candidate
    return None


# =====================================================================================================================================================
# XML PARSING HELPER
# =====================================================================================================================================================

#: Parsed XDMF documents and grid descriptors, keyed on (path, mtime, size) so
#: an edited or regenerated file is never served stale. Bounded because a long
#: GUI session can walk hundreds of timesteps; XDMF trees are not small.
_XDMF_CACHE_LIMIT = 32
_XDMF_XML_CACHE = {}
_XDMF_GRID_CACHE = {}


def _cache_key(path):
    """Identity of a file's current contents, or None if it is unreadable."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (os.path.abspath(path), stat.st_mtime_ns, stat.st_size)


def _cache_store(cache, key, value):
    """Insert into a bounded cache, evicting the oldest entry when full."""
    if len(cache) >= _XDMF_CACHE_LIMIT:
        cache.pop(next(iter(cache)))
    cache[key] = value
    return value


def _parse_xdmf_xml(xdmf_path):
    """
    Parse an XDMF file, handling files that have been appended
    (multiple root elements).

    When a simulation appends to an existing XDMF file, the result is
    multiple consecutive <Xdmf>...</Xdmf> blocks, which is not valid XML.
    This function wraps them in a synthetic root and returns the *last*
    <Xdmf> element (the most recent write).

    Results are cached on (path, mtime, size) so that the metadata pass and
    the subsequent data pass do not re-read and re-parse the same document.

    Returns:
        ET.Element: The root element to iterate over, or None on failure.
    """
    cache_key = _cache_key(xdmf_path)
    if cache_key is None:
        return None
    if cache_key in _XDMF_XML_CACHE:
        return _XDMF_XML_CACHE[cache_key]

    root = None
    try:
        root = ET.parse(xdmf_path).getroot()
    except ET.ParseError:
        # Likely multiple root elements — wrap in a synthetic root, take the last entry.
        try:
            with open(xdmf_path, 'r') as f:
                xml_content = f.read()
        except IOError as e:
            print(f"Error reading {xdmf_path}: {e}")
            return None

        cleaned = _XML_DECL_RE.sub('', xml_content)
        try:
            wrapper = ET.fromstring(f"<_wrapper>{cleaned}</_wrapper>")
            xdmf_elements = list(wrapper)
            if xdmf_elements:
                print(f"Note: {os.path.basename(xdmf_path)} contains {len(xdmf_elements)} "
                      f"appended entries — using the last one.")
                root = xdmf_elements[-1]
        except ET.ParseError as e:
            print(f"Error parsing {xdmf_path} (even after handling appended entries): {e}")
            return None

    return _cache_store(_XDMF_XML_CACHE, cache_key, root)


# =====================================================================================================================================================
# TEXT DATA UTILITIES
# =====================================================================================================================================================

def case_path(folder_path, case):
    """Build a normalised case path from folder and case inputs."""
    base = os.path.expanduser(os.path.expandvars(str(folder_path).strip()))
    case_name = str(case).strip().strip('/\\')
    if not base:
        return case_name
    return os.path.normpath(os.path.join(base, case_name))

def data_filepath(folder_path, case, quantity, timestep):
    """Path of a legacy per-quantity tsp_avg table (per-field output layout)."""
    return os.path.join(
        case_path(folder_path, case),
        '1_data',
        f'domain1_tsp_avg_{quantity}_{timestep}.dat'
    )

def load_ts_avg_data(data_filepath):
    try:
        return np.loadtxt(data_filepath)
    except OSError:
        print(f'Error loading data for {data_filepath}')
        return None

def get_quantities(thermo_on):
    quantities = ['u1', 'u2', 'u3', 'uu11', 'uu12', 'uu22','uu33','pr']
    if thermo_on:
        quantities.extend(['T', 'Tu2'])
    return quantities


# =====================================================================================================================================================
# ASCII PROFILE BUNDLES (time-and-space averaged statistics)
# =====================================================================================================================================================
#
# With the bundled output layout CHAPSim2 writes every tsp_avg statistic for a
# physics group into one self-describing table:
#
#   2_visu/data/domain1_tsp_avg_flow_yprofile_60.dat
#
#   # CHAPSim2 time-and-space averaged profile
#   # format_version: CHAPSim_profile_ascii_v1
#   # direction: y
#   # coordinate: yc
#   # npoints: 80
#   # columns: index yc tsp_avg_pr tsp_avg_u1 ...
#   <npoints rows of whitespace-separated values>
#
# This replaced the per-quantity 1_data/domain1_tsp_avg_<quantity>_<iter>.dat
# files, which the per-field output layout still writes.

PROFILE_BUNDLE_FORMAT = 'CHAPSim_profile_ascii_v1'


def read_profile_bundle(path):
    """Read a CHAPSim2 ASCII profile bundle.

    Args:
        path: Path to a ``*_?profile_<iter>.dat`` table.

    Returns:
        tuple ``(columns, meta)`` where ``columns`` maps each column name to a
        1-D array and ``meta`` holds the header fields (``direction``,
        ``coordinate``, ``npoints``, ``format_version``, ``columns``).
        ``({}, {})`` if the file is missing or unreadable.
    """
    if not os.path.isfile(path):
        return {}, {}

    meta = {}
    names = None
    try:
        with open(path, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    break
                body = line[1:].strip()
                if ':' not in body:
                    continue
                key, _, value = body.partition(':')
                key = key.strip().lower()
                value = value.strip()
                if key == 'columns':
                    names = value.split()
                    meta['columns'] = names
                else:
                    meta[key] = value
    except IOError as e:
        print(f"Error reading profile bundle {path}: {e}")
        return {}, {}

    if not names:
        print(f"Profile bundle {os.path.basename(path)} has no '# columns:' header")
        return {}, meta

    try:
        table = np.loadtxt(path, comments='#', ndmin=2)
    except (OSError, ValueError) as e:
        print(f"Error parsing profile bundle {path}: {e}")
        return {}, meta

    if table.shape[1] != len(names):
        print(f"Profile bundle {os.path.basename(path)}: header lists {len(names)} columns "
              f"but the table has {table.shape[1]} — truncating to the shorter of the two.")
    ncol = min(table.shape[1], len(names))

    if 'npoints' in meta:
        try:
            meta['npoints'] = int(meta['npoints'])
        except ValueError:
            pass

    columns = {names[i]: table[:, i] for i in range(ncol)}
    return columns, meta


def find_profile_bundles(path, timestep=None):
    """Find ASCII profile bundles written for a case.

    Args:
        path: Case directory (or any of its output subdirectories).
        timestep: Optional timestep to filter on.

    Returns:
        dict mapping ``(group, timestep)`` to the bundle path, e.g.
        ``{('flow', '60'): '.../domain1_tsp_avg_flow_yprofile_60.dat'}``.
    """
    dirs = resolve_case_dirs(path)
    pattern = re.compile(r'^domain\d+_tsp_avg_([a-z]+)_[xyz]profile_(\d+)\.dat$')

    found = {}
    # The bundled layout writes into 2_visu/data; the flat layout into 2_visu.
    for directory in dict.fromkeys((dirs['data'], dirs['visu'])):
        try:
            entries = os.listdir(directory)
        except OSError:
            continue
        for fname in entries:
            match = pattern.match(fname)
            if not match:
                continue
            group, ts = match.group(1), match.group(2)
            if timestep is not None and ts != str(timestep):
                continue
            found.setdefault((group, ts), os.path.join(directory, fname))
    return found


def load_tsp_avg_profiles(path, timestep, strip_prefix=True):
    """Load every time-and-space averaged profile available for a timestep.

    Reads the bundled ASCII tables first and falls back to the per-field
    ``1_data/domain1_tsp_avg_<quantity>_<iter>.dat`` files, so both output
    layouts give the same result.

    Args:
        path: Case directory (or any of its output subdirectories).
        timestep: Timestep to load.
        strip_prefix: Drop the ``tsp_avg_`` prefix from variable names.

    Returns:
        tuple ``(variables, coords)``.  ``variables`` maps variable name to a
        1-D array; ``coords`` holds the profile coordinate under its own name
        (``yc``, ``xc`` or ``zc``) plus ``direction``.
    """
    dirs = resolve_case_dirs(path)
    variables = {}
    coords = {}

    for (_group, _ts), bundle in sorted(find_profile_bundles(path, timestep).items()):
        columns, meta = read_profile_bundle(bundle)
        if not columns:
            continue
        coordinate = meta.get('coordinate', 'yc')
        direction = meta.get('direction', coordinate[:1])
        coords.setdefault('direction', direction)
        if coordinate in columns:
            coords.setdefault(coordinate, columns[coordinate])
        for name, values in columns.items():
            if name in ('index', coordinate):
                continue
            key = _strip_avg_prefix(name) if strip_prefix else name
            variables.setdefault(key, values)

    # Per-field layout: one two/three-column table per quantity.
    legacy_pattern = os.path.join(dirs['raw'], f'domain1_tsp_avg_*_{timestep}.dat')
    prefix, suffix = 'domain1_tsp_avg_', f'_{timestep}.dat'
    for file_path in sorted(glob.glob(legacy_pattern)):
        quantity = os.path.basename(file_path)[len(prefix):-len(suffix)]
        key = quantity if strip_prefix else f'tsp_avg_{quantity}'
        if key in variables:
            continue
        table = load_ts_avg_data(file_path)
        if table is None or table.ndim != 2 or table.shape[1] < 3:
            continue
        variables[key] = table[:, 2]
        coords.setdefault('yc', table[:, 1])
        coords.setdefault('direction', 'y')

    return variables, coords


# =====================================================================================================================================================
# BINARY BUNDLE METADATA
# =====================================================================================================================================================
#
# Alongside each bundled .bin, CHAPSim2 writes a plain-text manifest naming
# every field packed into it and where it starts:
#
#   CHAPSim_visu_bundle_v1 / CHAPSim_visu_slice_bundle_v1
#   group flow
#   domain 1
#   iter 60
#   precision_bytes 4
#   fields name original_file [source] [slice] center dimensions_kji ...
#   pr 2_visu/data/domain1_visu_pr_60.bin cell_centered Cell 64 80 64 4 0 1310720
#
# The XDMF carries the same information, so this reader is a fallback for
# cases where only the binaries were kept.

def read_visu_bundle_meta(path):
    """Read a ``*_meta_<iter>.dat`` visualisation bundle manifest.

    Returns:
        tuple ``(fields, meta)``.  ``fields`` is a list of dicts with keys
        ``name``, ``original_file``, ``slice`` (None for 3-D bundles),
        ``dims`` (k,j,i), ``precision_bytes``, ``offset_bytes`` and
        ``nbytes``.  ``meta`` holds ``format``, ``group``, ``domain`` and
        ``iter``.
    """
    if not os.path.isfile(path):
        return [], {}

    fields = []
    meta = {}
    columns = None
    try:
        with open(path, 'r') as f:
            for lineno, line in enumerate(f):
                parts = line.split()
                if not parts:
                    continue
                if lineno == 0:
                    meta['format'] = parts[0]
                    continue
                key = parts[0]
                if key in ('group', 'domain', 'iter', 'precision_bytes') and len(parts) >= 2:
                    meta[key] = parts[1]
                    continue
                if key == 'fields':
                    columns = parts[1:]
                    continue
                if columns is None:
                    continue
                # Trailing numeric columns are fixed: ... dims(3) precision offset nbytes
                if len(parts) < 7:
                    continue
                try:
                    dims = tuple(int(v) for v in parts[-6:-3])
                    precision, offset, nbytes = (int(parts[-3]), int(parts[-2]), int(parts[-1]))
                except ValueError:
                    continue
                record = {
                    'name': parts[0],
                    'original_file': parts[1] if len(parts) > 1 else None,
                    'slice': None,
                    'dims': dims,
                    'precision_bytes': precision,
                    'offset_bytes': offset,
                    'nbytes': nbytes,
                }
                if 'slice' in columns:
                    # Slice bundles carry the tag immediately after original_file.
                    tag = parts[2] if len(parts) > 2 else None
                    if tag and _SLICE_LABEL_RE.match(tag):
                        record['slice'] = tag
                fields.append(record)
    except IOError as e:
        print(f"Error reading bundle metadata {path}: {e}")
        return [], meta

    return fields, meta


# =====================================================================================================================================================
# XDMF FILE PATH UTILITIES
# =====================================================================================================================================================

def visu_file_paths(folder_path, case, timestep):
    """Return the XDMF files a case offers for a timestep.

    Covers instantaneous (``domain1_flow_60.xdmf``), 3-D time-averaged
    (``domain1_t_avg_flow_60.xdmf``) and time-and-space averaged plane
    (``domain1_tsp_avg_flow_zi1_60.xdmf``) output for every physics group.
    The tsp_avg plane tag names the periodic direction that was averaged out,
    so it is discovered rather than assumed to be ``zi1``.

    The returned list is ordered instantaneous → t_avg → tsp_avg within each
    group, and is filtered to files that exist.
    """
    xdmf_dir = resolve_case_dirs(case_path(folder_path, case))['xdmf']

    names = []
    for group in VISU_GROUPS:
        names.append(f'domain1_{group}_{timestep}.xdmf')
        names.append(f'domain1_t_avg_{group}_{timestep}.xdmf')
        names.append(f'domain1_tsp_avg_{group}_{timestep}.xdmf')
        # tsp_avg output of a flow with one periodic direction is a plane,
        # tagged with that direction (zi1 for a spanwise-periodic channel).
        for axis in 'xyz':
            names.append(f'domain1_tsp_avg_{group}_{axis}i1_{timestep}.xdmf')

    paths = [os.path.join(xdmf_dir, n) for n in names]
    existing = [p for p in paths if os.path.isfile(p)]
    # Returning only the files that exist keeps callers from probing a
    # directory that may be on a slow parallel filesystem.
    return existing if existing else paths


def group_xdmf_paths(folder_path, case, timestep, group='flow'):
    """Locate one physics group's XDMF output for a timestep, by averaging tier.

    Returns:
        dict with keys ``inst``, ``t_avg`` and ``tsp_avg``; each holds a path
        or None. Unlike visu_file_paths() this is keyed by meaning rather than
        position, so a caller after (say) the instantaneous field cannot be
        handed an averaged one just because the instantaneous file is absent.
    """
    xdmf_dir = resolve_case_dirs(case_path(folder_path, case))['xdmf']

    candidates = {
        'inst': [f'domain1_{group}_{timestep}.xdmf'],
        't_avg': [f'domain1_t_avg_{group}_{timestep}.xdmf'],
        # tsp_avg of a flow with one periodic direction is a plane tagged with
        # that direction; with two it is an ASCII profile and has no XDMF.
        'tsp_avg': ([f'domain1_tsp_avg_{group}_{timestep}.xdmf']
                    + [f'domain1_tsp_avg_{group}_{a}i1_{timestep}.xdmf' for a in 'xyz']),
    }

    found = {}
    for tier, names in candidates.items():
        found[tier] = next(
            (os.path.join(xdmf_dir, n) for n in names
             if os.path.isfile(os.path.join(xdmf_dir, n))), None)
    return found


def visu_catalogue(path, group='flow'):
    """Timesteps present for each data type of one physics group.

    A case does not carry every tier: statistics start partway through a run,
    thermo is written less often than flow, and a run with no averaging has no
    t_avg at all. Reading the directory is the only way to know which
    combinations will actually load.
    """
    try:
        entries = os.listdir(resolve_case_dirs(path)['xdmf'])
    except OSError:
        return {}

    prefixes = {'inst': f'domain1_{group}_',
                't_avg': f'domain1_t_avg_{group}_',
                'tsp_avg': f'domain1_tsp_avg_{group}_'}
    found = {}
    for name in entries:
        if not name.endswith('.xdmf') or '_grid' in name:
            continue
        stem = name[:-len('.xdmf')]
        tail = stem.rsplit('_', 1)[-1]
        if not tail.isdigit():
            continue
        if '_slices_visu_' in stem:
            if stem.startswith(prefixes['inst']):
                found.setdefault('2d_slice', set()).add(tail)
            continue
        # Longest prefix first: a t_avg name also starts with the inst prefix
        # only after the averaging tag, so test the specific ones first.
        for dtype, prefix in sorted(prefixes.items(), key=lambda kv: -len(kv[1])):
            if stem.startswith(prefix):
                found.setdefault(dtype, set()).add(tail)
                break
    if 'inst' in found:
        found.setdefault('2d_slice', set()).update(found['inst'])
    return {k: sorted(v, key=int) for k, v in found.items() if v}


def find_available_timesteps(path):
    """List the timesteps a case has XDMF field output for, sorted numerically.

    Mesh descriptors (``domain1_grid_yi8_0.xdmf``, ``domain1_grids_3d_0.xdmf``)
    are all stamped with iteration 0 and would otherwise advertise a timestep
    with no field data behind it.
    """
    xdmf_dir = resolve_case_dirs(path)['xdmf']
    steps = set()
    try:
        entries = os.listdir(xdmf_dir)
    except OSError:
        return []

    for fname in entries:
        if not fname.endswith('.xdmf') or '_grid_' in fname or '_grids_' in fname:
            continue
        tail = fname[:-len('.xdmf')].rsplit('_', 1)[-1]
        if tail.isdigit():
            steps.add(tail)
    return sorted(steps, key=int)


# =====================================================================================================================================================
# 2D SLICE FILE UTILITIES
# =====================================================================================================================================================

def parse_slice_label(label):
    """
    Parse a 2D slice label from a filename component.

    Examples:
        'yi8'  -> ('y', 8)   # xz slice at y index 8
        'xi5'  -> ('x', 5)   # yz slice at x index 5
        'zi3'  -> ('z', 3)   # xy slice at z index 3

    Returns:
        tuple: (direction, index) or None if not a valid slice label
    """
    match = _SLICE_LABEL_RE.match(str(label).strip())
    if match:
        return match.group(1), int(match.group(2))
    return None


def find_available_slices(visu_folder, timestep=None):
    """
    Find available 2D slice labels for a case.

    Picks up both the per-slice XDMF files (``domain1_flow_yi8_60.xdmf``) and
    the slice bundles (``domain1_flow_slices_visu_60.xdmf``), which pack every
    slice of a timestep into a single grid collection.

    Args:
        visu_folder: Case directory or any of its output subdirectories
        timestep: Optional timestep to filter for

    Returns:
        list of unique slice labels found, sorted (e.g., ['xi5', 'yi8', 'zi3'])
    """
    dirs = resolve_case_dirs(visu_folder)
    labels = set()
    bundle_re = re.compile(r'_slices_visu_(\d+)\.xdmf$')

    try:
        entries = os.listdir(dirs['xdmf'])
    except OSError:
        return []

    for fname in entries:
        if not fname.endswith('.xdmf'):
            continue
        # Grid files (domain1_grid_yi8_0.xdmf) describe the mesh, not a field.
        # tsp_avg files carry an xi1/yi1/zi1 tag naming the direction that was
        # averaged away, which is not a slice through the domain.
        if '_grid_' in fname or 'tsp_avg' in fname:
            continue

        match = _SLICE_FILE_RE.search(fname)
        if match:
            if timestep is None or match.group(2) == str(timestep):
                labels.add(match.group(1))
            continue

        bundle = bundle_re.search(fname)
        if bundle and (timestep is None or bundle.group(1) == str(timestep)):
            for grid in parse_xdmf_grids(os.path.join(dirs['xdmf'], fname)):
                if grid['tag']:
                    labels.add(grid['tag'])

    return sorted(labels)


#: Axis labels per computational direction. A pipe or annulus is rectilinear
#: in (x, r, theta), not in Cartesian space, so a slice of one is plotted in
#: those coordinates and has to be labelled as such — an unrolled (r, theta)
#: plane drawn on axes marked y and z would read as a square duct.
_AXIS_LABELS = {
    'cartesian': {'x': '$x$', 'y': '$y$', 'z': '$z$'},
    'cylindrical': {'x': '$x$', 'y': '$r$', 'z': r'$\theta$'},
}


def axis_labels(coordinate_system=None):
    """Plot labels for the three computational directions, keyed 'x'/'y'/'z'.

    Pass ``grid_info['coordinate_system']``; anything unrecognised (including
    None) falls back to Cartesian labels.
    """
    return _AXIS_LABELS.get(coordinate_system or 'cartesian', _AXIS_LABELS['cartesian'])


#: The azimuthal label, used to spot an axis that is an angle, not a length.
ANGLE_LABEL = _AXIS_LABELS['cylindrical']['z']


def plot_aspect(labels):
    """Matplotlib aspect for a plane spanned by ``labels``.

    ``'equal'`` keeps a slice geometrically faithful, but only when both axes
    are lengths. A pipe's cross-section is plotted in (r, theta): forcing
    equal there sets one radian equal to one length unit, which squashes the
    duct into a strip of whatever aspect 2*pi happens to give. Those planes
    get ``'auto'`` instead.
    """
    return 'auto' if ANGLE_LABEL in tuple(labels) else 'equal'


def slice_axis_info(slice_label, coordinate_system='cartesian'):
    """
    Get axis labels and grid coordinate keys for a 2D slice.

    Args:
        slice_label: Slice label string (e.g., 'yi8')
        coordinate_system: 'cartesian' or 'cylindrical'; usually
            ``grid_info['coordinate_system']`` from parse_xdmf_metadata.

    Returns:
        dict with keys:
            'plane': slice plane name ('xy', 'xz', or 'yz')
            'normal_dir': direction normal to slice ('x', 'y', or 'z')
            'normal_index': integer index along the normal direction
            'axis_labels': tuple of (xlabel, ylabel) for plotting
            'coord_keys': tuple of (coord1_key, coord2_key) grid_info keys
        or None if slice_label is invalid
    """
    parsed = parse_slice_label(slice_label)
    if parsed is None:
        return None

    direction, index = parsed
    # The plane is spanned by the two directions the slice does not cut.
    planes = {'y': ('x', 'z'), 'x': ('y', 'z'), 'z': ('x', 'y')}
    if direction not in planes:
        return None

    first, second = planes[direction]
    labels = axis_labels(coordinate_system)

    return {
        'plane': first + second,
        'normal_dir': direction,
        'normal_index': index,
        'axis_labels': (labels[first], labels[second]),
        'coord_keys': (f'grid_{first}', f'grid_{second}'),
    }


def parse_x_crop_input(text):
    """Parse crop input string to ``(x_min, x_max)``.

    Accepted format: ``"x_min,x_max"``.
    Returns ``None`` for blank input.
    """
    if text is None:
        return None

    value = str(text).strip()
    if not value:
        return None

    parts = [p.strip() for p in value.split(',') if p.strip()]
    if len(parts) != 2:
        raise ValueError("Expected format 'x_min,x_max'")

    x_min, x_max = float(parts[0]), float(parts[1])
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    return x_min, x_max


def apply_x_crop(data, x_coords, x_crop):
    """Crop array data along the last axis using an x-range.

    Args:
        data: ndarray to crop (1-D/2-D/3-D ...). Cropping is applied on axis ``-1``.
        x_coords: x-coordinate array (cell-centres ``nx`` or edges ``nx+1``).
        x_crop: tuple ``(x_min, x_max)`` or ``None``.

    Returns:
        tuple: ``(cropped_data, cropped_x_coords)``
    """
    if x_crop is None or x_coords is None:
        return data, x_coords

    x = np.asarray(x_coords)
    nx = data.shape[-1]
    x_min, x_max = x_crop

    if x.size == nx:
        centres = x
        use_edges = False
    elif x.size == nx + 1:
        centres = 0.5 * (x[:-1] + x[1:])
        use_edges = True
    else:
        return data, x_coords

    idx = np.where((centres >= x_min) & (centres <= x_max))[0]
    if idx.size == 0:
        print(f"Warning: No data points in x range [{x_min}, {x_max}]")
        return data, x_coords

    i0, i1 = int(idx[0]), int(idx[-1] + 1)
    cropped_data = np.take(data, indices=np.arange(i0, i1), axis=-1)
    if use_edges:
        cropped_x = x[i0:i1 + 1]
    else:
        cropped_x = x[i0:i1]

    return cropped_data, cropped_x

# =====================================================================================================================================================
# XDMF READING UTILITIES
# =====================================================================================================================================================

def _data_item_params(data_item, xdmf_dir):
    """
    Extract the binary read parameters from a DataItem element without
    touching the data itself.

    Returns:
        dict with bin_path, dims, dtype, seek keys, or None if the item
        cannot be read.
    """
    format_type = data_item.get('Format', 'Binary')
    if format_type != 'Binary':
        print(f"Unsupported XDMF data format: {format_type}")
        return None

    dims_str = data_item.get('Dimensions', '')
    dims = tuple(int(d) for d in dims_str.split()) if dims_str else None

    number_type = data_item.get('NumberType', 'Float')
    precision = int(data_item.get('Precision', '8'))
    seek = int(data_item.get('Seek', '0'))

    if number_type == 'Float':
        dtype = np.float32 if precision == 4 else np.float64
    elif number_type == 'Int':
        dtype = np.int32 if precision == 4 else np.int64
    else:
        dtype = np.float64

    bin_path = resolve_binary_path(
        data_item.text.strip() if data_item.text else None, xdmf_dir)
    if bin_path is None:
        return None

    return {
        'bin_path': bin_path,
        'dims': dims,
        'dtype': dtype,
        'seek': seek,
    }


def _read_binary(params, stride=1, handle=None):
    """
    Read one field from a binary file.

    Args:
        params: dict with bin_path, dims, dtype, seek (from _data_item_params)
        stride: subsample every `stride`-th element along each axis. When >1
            and `dims` has more than one axis, the file is memory-mapped so
            only the strided subset is ever paged into RAM — large domains
            no longer need to be fully resident just to build a decimated
            (e.g. PyVista) grid.
        handle: open binary file object for params['bin_path']. CHAPSim2 packs
            every field of a group into one .bin, so reusing one handle across
            a whole bundle avoids ~100 opens of the same file — a real cost on
            a parallel filesystem.

    Returns:
        numpy array or None on failure
    """
    bin_path = params['bin_path']
    dims = params['dims']
    dtype = params['dtype']
    seek = params['seek']

    try:
        if stride > 1 and dims and len(dims) > 1:
            mm = np.memmap(bin_path, dtype=dtype, mode='r', offset=seek, shape=tuple(dims))
            data = np.array(mm[tuple(slice(None, None, stride) for _ in dims)])
            del mm
            return data

        if dims:
            count = int(np.prod(dims))
        else:
            itemsize = np.dtype(dtype).itemsize
            count = (os.path.getsize(bin_path) - seek) // itemsize

        # Reading with an explicit count rather than to EOF: a bundled .bin
        # holds many fields, and an unbounded read on Lustre can stall.
        if handle is not None:
            handle.seek(seek)
            data = np.fromfile(handle, dtype=dtype, count=count)
        else:
            with open(bin_path, 'rb') as f:
                f.seek(seek)
                data = np.fromfile(f, dtype=dtype, count=count)

        if dims:
            expected_size = int(np.prod(dims))
            if data.size >= expected_size:
                data = data[:expected_size].reshape(dims)
            else:
                print(f"Warning: Data size mismatch for {bin_path} "
                      f"(got {data.size}, expected {expected_size})")

        return data

    except Exception as e:
        print(f"Error reading {bin_path}: {e}")
        return None


def read_binary_data_item(data_item, xdmf_dir):
    """Read the binary data behind an XDMF DataItem element."""
    params = _data_item_params(data_item, xdmf_dir)
    if params is None:
        return None
    return _read_binary(params)


def _strip_avg_prefix(name):
    """Return base variable name without known averaging prefixes."""
    for prefix in ('t_avg_', 'tsp_avg_'):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _is_selected_variable(name, required_vars):
    """Check whether an XDMF variable should be loaded."""
    if required_vars is None:
        return name in REQUIRED_VARS

    base_name = _strip_avg_prefix(name)
    return (name in required_vars) or (base_name in required_vars)

def _reduce_xdmf_array(data, average_z=False, average_x=False):
    """Reduce XDMF array dimensionality. Averages over requested axes then squeezes singletons."""
    if data is None or data.ndim != 3:
        return data

    if average_z and average_x:
        return data.mean(axis=(0, 2))
    if average_z:
        return data.mean(axis=0)
    if average_x:
        return data.mean(axis=2)

    return np.squeeze(data)


# =====================================================================================================================================================
# XDMF METADATA PARSING & SELECTIVE LOADING
# =====================================================================================================================================================

def parse_xdmf_grids(xdmf_path):
    """Describe every grid in an XDMF file without loading field data.

    CHAPSim2 writes slice bundles as a ``GridType="Collection"`` holding one
    uniform grid per slice, and each sub-grid reuses the same attribute names
    (``pr``, ``qx_ccc``, ...).  They therefore have to be kept apart rather
    than merged into one namespace.

    Args:
        xdmf_path: Path to the XDMF file

    Returns:
        list of dicts, one per uniform grid, each with keys ``name``, ``tag``
        (slice label such as ``yi8``, or None), ``node_dimensions``,
        ``cell_dimensions``, ``grid_x``/``grid_y``/``grid_z`` (node
        coordinates) and ``vars`` ({name: read params including 'shape'}).
        Treat the result as read-only: it is cached and shared between calls.
    """
    cache_key = _cache_key(xdmf_path)
    if cache_key is not None and cache_key in _XDMF_GRID_CACHE:
        return _XDMF_GRID_CACHE[cache_key]

    root = _parse_xdmf_xml(xdmf_path)
    if root is None:
        return []

    xdmf_dir = os.path.dirname(os.path.abspath(xdmf_path))
    grids = []

    for grid in root.iter('Grid'):
        # Direct children only: a Collection's iter() would otherwise absorb
        # the topology and attributes of every sub-grid it contains.
        topo = grid.find('Topology')
        attributes = grid.findall('Attribute')
        if topo is None and not attributes:
            continue

        entry = {
            'name': grid.get('Name', ''),
            'tag': None,
            'node_dimensions': None,
            'cell_dimensions': None,
            'vars': {},
        }

        tag_match = _SLICE_IN_NAME_RE.search(entry['name'])
        if tag_match:
            entry['tag'] = tag_match.group(1)

        if topo is not None:
            dims_str = topo.get('Dimensions')
            if dims_str:
                dims = tuple(int(d) for d in dims_str.split())
                entry['node_dimensions'] = dims
                entry['cell_dimensions'] = tuple(d - 1 for d in dims)

        geom = grid.find('Geometry')
        geom_type = geom.get('GeometryType') if geom is not None else None
        if geom_type == 'VXVYVZ':
            entry['coordinate_system'] = 'cartesian'
            for axis, data_item in zip('xyz', geom.findall('DataItem')):
                coords = read_binary_data_item(data_item, xdmf_dir)
                if coords is not None:
                    entry[f'grid_{axis}'] = coords
        elif geom_type == 'XYZ':
            entry.update(_cylindrical_axes(read_binary_data_item(geom.find('DataItem'), xdmf_dir),
                                           entry['node_dimensions']))

        cell_dims = entry['cell_dimensions']
        for attribute in attributes:
            name = attribute.get('Name')
            data_item = attribute.find('DataItem')
            if name is None or data_item is None:
                continue
            params = _data_item_params(data_item, xdmf_dir)
            if params is None:
                continue
            params['shape'] = _effective_shape(params, cell_dims)
            entry['vars'][name] = params

        grids.append(entry)

    if cache_key is not None:
        _cache_store(_XDMF_GRID_CACHE, cache_key, grids)
    return grids


def _cylindrical_axes(points, node_dims):
    """Recover the solver's (axial, radial, azimuthal) axes from a curvilinear mesh.

    Pipe and annulus cases write their mesh as one XYZ point list rather than
    three coordinate vectors, because the grid is only rectilinear in
    (x, r, theta), not in Cartesian space. CHAPSim2 builds those points as

        x = i * dx,   y = r(j) cos(theta_k),   z = r(j) sin(theta_k)

    so each computational axis varies along exactly one index and can be read
    straight back out. The toolkit's ``grid_y`` therefore holds the radius,
    matching the solver's index-2 direction and letting the wall-normal
    machinery work unchanged on a pipe.

    Args:
        points: (npoints, 3) array of node coordinates, k-major.
        node_dims: XDMF topology dimensions in (k, j, i) order.

    Returns:
        dict of ``grid_x``/``grid_y``/``grid_z`` plus ``coordinate_system``
        and the full ``grid_points`` block; empty if the data does not fit.
    """
    if points is None or node_dims is None or len(node_dims) != 3:
        return {}

    nk, nj, ni = node_dims
    if points.size != nk * nj * ni * 3:
        print(f"Curvilinear mesh has {points.size} values, expected "
              f"{nk * nj * ni * 3} for a {node_dims} grid — skipping coordinates.")
        return {}

    pts = points.reshape(nk, nj, ni, 3)
    radius = np.hypot(pts[0, :, 0, 1], pts[0, :, 0, 2])

    # theta is undefined on the axis, so read it off the outermost radius —
    # a pipe's first node sits at r = 0 and would otherwise give theta == 0.
    j_ref = int(np.argmax(radius))
    theta = np.arctan2(pts[:, j_ref, 0, 2], pts[:, j_ref, 0, 1])

    return {
        'coordinate_system': 'cylindrical',
        'grid_x': pts[0, 0, :, 0],
        'grid_y': radius,
        # atan2 wraps at +/-pi; the solver sweeps theta monotonically from 0.
        'grid_z': np.unwrap(theta) if theta.size > 1 else theta,
        'grid_points': pts,
    }


def _effective_shape(params, cell_dims):
    """Shape a field takes once a flat binary read is folded onto the grid."""
    raw_dims = params['dims']
    if raw_dims and len(raw_dims) > 1:
        return raw_dims
    if cell_dims:
        if raw_dims:
            size = int(np.prod(raw_dims))
        else:
            itemsize = np.dtype(params['dtype']).itemsize
            size = (os.path.getsize(params['bin_path']) - params['seek']) // itemsize
        if size == int(np.prod(cell_dims)):
            return cell_dims
        return raw_dims if raw_dims else (size,)
    return raw_dims


def _select_grid(grids, grid_select):
    """Pick one grid out of an XDMF file's grid list.

    ``grid_select`` may be a slice label ('yi8'), a grid name, or an integer
    index.  ``None`` takes the first grid.
    """
    if not grids:
        return None
    if grid_select is None or len(grids) == 1:
        # A single-grid file is unambiguous: honour it even when the caller
        # asked for a slice tag that this file does not spell out in its grid
        # name (per-slice files are named by their filename, not their grid).
        return grids[0]
    if isinstance(grid_select, int):
        return grids[grid_select] if -len(grids) <= grid_select < len(grids) else None
    key = str(grid_select)
    for grid in grids:
        if grid['tag'] == key or grid['name'] == key:
            return grid
    return None


def parse_xdmf_metadata(xdmf_path, grid_select=None):
    """
    Parse XDMF file structure and return variable names, shapes, and grid info
    without loading variable data.  Grid coordinates (small 1D arrays) are loaded.

    Args:
        xdmf_path: Path to the XDMF file
        grid_select: For files holding several grids (slice bundles), the slice
            label, grid name or index to read. Defaults to the first grid.

    Returns:
        tuple: (var_metadata, grid_info)
            var_metadata: dict of {name: {'shape': tuple, 'bin_path': str,
                          'dims': tuple, 'dtype': dtype, 'seek': int}}
            grid_info: dict with node_dimensions, cell_dimensions,
                       grid_x, grid_y, grid_z, plus 'available_grids'
                       (slice labels) and 'grid_tag' for bundles
    """
    grids = parse_xdmf_grids(xdmf_path)
    if not grids:
        return {}, {}

    chosen = _select_grid(grids, grid_select)
    if chosen is None:
        available = [g['tag'] or g['name'] for g in grids]
        print(f"Grid '{grid_select}' not found in {os.path.basename(xdmf_path)}. "
              f"Available: {', '.join(str(a) for a in available)}")
        return {}, {}

    grid_info = {k: v for k, v in chosen.items() if k != 'vars' and v is not None}
    grid_info.pop('name', None)
    grid_info['grid_tag'] = chosen['tag']
    if len(grids) > 1:
        grid_info['available_grids'] = [g['tag'] or g['name'] for g in grids]

    # Shallow copy: parse_xdmf_grids caches its result, and callers (the GUI
    # in particular) drop entries from var_metadata to filter the variable list.
    return dict(chosen['vars']), grid_info


def load_xdmf_variables(var_metadata, selected_vars, grid_info=None,
                        average_z=False, average_x=False, stride=1, quiet=False):
    """
    Load specific variables using pre-parsed XDMF metadata.

    Args:
        var_metadata: dict from parse_xdmf_metadata
        selected_vars: list of variable names to load
        grid_info: grid info dict (for reshaping flat arrays)
        average_z: If True, average over the z direction for 3D arrays.
        average_x: If True, average over the x direction for 3D arrays.
        stride: subsample every `stride`-th cell along each axis at read
            time (see _read_binary) so large domains don't need to be fully
            loaded just to be decimated afterwards.
        quiet: Suppress the per-variable progress log.

    Returns:
        dict: {variable_name: numpy_array}. Singleton dimensions are automatically
              squeezed (e.g. tsp_avg slice files become true 2D).
    """
    arrays = {}
    cell_dims = (grid_info or {}).get('cell_dimensions')

    missing = [n for n in selected_vars if n not in var_metadata]
    for name in missing:
        tqdm.write(f"  Variable '{name}' not found in metadata, skipping")
    present = [n for n in selected_vars if n in var_metadata]

    # Group by source file so each bundled .bin is opened once, not once per
    # field: a flow stats bundle holds ~90 fields in a single file.
    by_file = {}
    for name in present:
        by_file.setdefault(var_metadata[name]['bin_path'], []).append(name)

    with tqdm(total=len(present), desc="Loading selected variables", unit="var",
              disable=quiet or not present) as pbar:
        for bin_path, names in by_file.items():
            try:
                handle = open(bin_path, 'rb')
            except OSError as e:
                print(f"Error opening {bin_path}: {e}")
                pbar.update(len(names))
                continue
            try:
                for name in names:
                    data = _read_binary(var_metadata[name], stride=stride, handle=handle)
                    pbar.update(1)
                    if data is None:
                        continue

                    # Reshape flat arrays to 3D using cell dimensions
                    if data.ndim == 1 and cell_dims and data.size == int(np.prod(cell_dims)):
                        data = data.reshape(cell_dims)

                    arrays[name] = _reduce_xdmf_array(
                        data, average_z=average_z, average_x=average_x)
                    if not quiet:
                        tqdm.write(f"  Loaded {name}: shape {arrays[name].shape}")
            finally:
                handle.close()

    return arrays


def parse_xdmf_file(xdmf_path, load_all_vars=None, required_vars=None,
                    average_z=False, average_x=False, grid_select=None):
    """
    Parse XDMF file and extract data from associated binary files.

    Args:
        xdmf_path: Path to the XDMF file
        load_all_vars: If True, load all variables. If False, only load required ones.
                       If None, uses module-level LOAD_ALL_VARS setting.
        required_vars: Optional set/list of exact required variables (base or
                   prefixed names). If None, uses module REQUIRED_VARS.
        average_z: If True, average over the z direction for 3D arrays.
        average_x: If True, average over the x direction for 3D arrays.
        grid_select: Slice label, grid name or index for multi-grid files
                     (slice bundles). Defaults to the first grid.

    Returns:
        tuple: (arrays dict, grid_info dict). Singleton dimensions (e.g. from tsp_avg
               z-slice files) are automatically squeezed to give true 2D output.
    """
    if load_all_vars is None:
        load_all_vars = LOAD_ALL_VARS

    var_metadata, grid_info = parse_xdmf_metadata(xdmf_path, grid_select=grid_select)
    if not var_metadata:
        return {}, grid_info

    if load_all_vars:
        selected = list(var_metadata)
    else:
        selected = [n for n in var_metadata if _is_selected_variable(n, required_vars)]
        skipped = len(var_metadata) - len(selected)
        if skipped:
            names = [n for n in var_metadata if n not in set(selected)]
            tqdm.write(f"  Skipping {skipped} unneeded variables: "
                       f"{', '.join(names[:3])}{'...' if skipped > 3 else ''}")

    arrays = load_xdmf_variables(
        var_metadata, selected, grid_info=grid_info,
        average_z=average_z, average_x=average_x, quiet=True)

    return arrays, grid_info


def xdmf_reader_wrapper(file_names, case=None, timestep=None, load_all_vars=None, data_types=None,
                         required_vars=None,
                         average_z=False, average_x=False):
    """
    Reads XDMF files and extracts numpy arrays from binary data.

    Args:
        file_names (list): List of XDMF file paths to read
        case (str, optional): Case identifier for dictionary key
        timestep (str, optional): Timestep identifier for dictionary key
        load_all_vars (bool, optional): If True, load all variables. If False, only required ones.
        data_types (list, optional): List of data types to load. If None, loads all files.
            Valid types: 'inst', 't_avg', 'tsp_avg', or specific combinations
            like 't_avg_flow', 'tsp_avg_thermo', etc.
            Examples:
                - ['t_avg'] loads only time-averaged files (flow, thermo, mhd)
                - ['tsp_avg'] loads the time-and-space averaged planes
                - ['inst'] loads all instantaneous files (flow, thermo, mhd)
                - ['t_avg_flow'] loads only time-averaged flow files

        required_vars (set/list, optional): Minimal variable set to load.
            If None, falls back to module REQUIRED_VARS.

    Returns:
        tuple: (visu_arrays_dic dict, grid_info dict)

        If case and timestep are provided, returns nested dictionary:
        {
            "case_timestep": {
                "variable_name": numpy_array,
                ...
            }
        }

        Otherwise returns flat dictionary:
        {
            "variable_name": numpy_array,
            ...
        }
    """
    # Determine if we should use nested structure
    use_nested = case is not None and timestep is not None

    if use_nested:
        # Create the outer key for this case/timestep combination
        outer_key = f"{case}_{timestep}"
        visu_arrays_dic = {outer_key: {}}
        inner_dict = visu_arrays_dic[outer_key]
    else:
        visu_arrays_dic = {}
        inner_dict = visu_arrays_dic

    grid_info = {}

    # Filter to only existing files for accurate progress bar
    existing_files = [f for f in file_names if os.path.isfile(f)]

    # Select which files to load — purely from the caller's explicit
    # data_types; no guessing between tsp_avg/t_avg/inst tiers.
    if data_types is not None:
        filtered_files = []
        for f in existing_files:
            filename = os.path.basename(f)
            for dtype in data_types:
                if dtype == 'tsp_avg':
                    match = 'tsp_avg_' in filename
                elif dtype == 't_avg':
                    # 'tsp_avg_' also contains 't_avg'-like text; require the
                    # purely temporal prefix.
                    match = '_t_avg_' in f'_{filename}'
                elif dtype == 'inst':
                    match = 't_avg' not in filename
                else:
                    match = dtype in filename
                if match:
                    filtered_files.append(f)
                    break
        existing_files = filtered_files
        tqdm.write(f"Filtering for data types: {data_types}")

    for xdmf_file in tqdm(existing_files, desc="Processing XDMF files", unit="file"):
        try:
            tqdm.write(f"Opening file: {xdmf_file}")
            # tsp_avg output is already space-averaged over its periodic
            # direction, so the squeezed singleton must not be averaged again.
            is_tsp_avg = 'tsp_avg_' in os.path.basename(xdmf_file)

            arrays, file_grid_info = parse_xdmf_file(
                xdmf_file,
                load_all_vars=load_all_vars,
                required_vars=required_vars,
                average_z=False if is_tsp_avg else average_z,
                average_x=False if is_tsp_avg else average_x,
            )

            if arrays:
                if 'tsp_avg' in xdmf_file:
                    file_type = 'tsp_avg'
                elif 't_avg' in xdmf_file:
                    file_type = 't_avg'
                elif '_mhd_' in xdmf_file:
                    file_type = 'mhd'
                elif '_thermo_' in xdmf_file:
                    file_type = 'thermo'
                else:
                    file_type = 'flow'

                if not grid_info and file_grid_info:
                    grid_info = file_grid_info
                    if 'node_dimensions' in grid_info:
                        tqdm.write(f"Grid info: node_dimensions={grid_info['node_dimensions']}, cell_dimensions={grid_info.get('cell_dimensions', 'N/A')}")

                for var_name, var_data in arrays.items():
                    inner_dict[_strip_avg_prefix(var_name)] = var_data
                tqdm.write(f"Successfully extracted {len(arrays)} arrays from {file_type} file")
            else:
                tqdm.write(f"No requested variables in {os.path.basename(xdmf_file)} — skipped")

        except Exception as e:
            tqdm.write(f"Error processing {xdmf_file}: {str(e)}")
            continue

    # Warn about any variables that were requested but not found in any loaded file
    if not load_all_vars and inner_dict:
        check_vars = required_vars if required_vars is not None else _BASE_REQUIRED_VARS
        missing = sorted(v for v in check_vars if v not in inner_dict)
        if missing:
            tqdm.write(f"WARNING: {len(missing)} requested variable(s) not found in loaded files: {', '.join(missing)}")

    return visu_arrays_dic, grid_info




def extract_grid_info_from_arrays(grid_info):
    """
    Extract grid dimensions and bounds from grid coordinate arrays.

    Args:
        grid_info: Dictionary containing grid_x, grid_y, grid_z arrays

    Returns:
        dict: Enhanced grid information including bounds
    """
    enhanced_info = dict(grid_info)

    try:
        if 'grid_x' in grid_info and 'grid_y' in grid_info and 'grid_z' in grid_info:
            x = grid_info['grid_x']
            y = grid_info['grid_y']
            z = grid_info['grid_z']

            enhanced_info['bounds'] = (
                float(x.min()), float(x.max()),
                float(y.min()), float(y.max()),
                float(z.min()), float(z.max())
            )

            # Calculate average spacing
            if len(x) > 1:
                dx = (x.max() - x.min()) / (len(x) - 1)
            else:
                dx = 0
            if len(y) > 1:
                dy = (y.max() - y.min()) / (len(y) - 1)
            else:
                dy = 0
            if len(z) > 1:
                dz = (z.max() - z.min()) / (len(z) - 1)
            else:
                dz = 0

            enhanced_info['average_spacing'] = (dx, dy, dz)

    except Exception as e:
        print(f"Could not extract complete grid info: {str(e)}")

    return enhanced_info

# =====================================================================================================================================================
# OUTPUT UTILITIES
# =====================================================================================================================================================

def reader_output_summary(arrays_dict):
    """
    Provides a summary analysis of the extracted arrays.

    Args:
        arrays_dict (dict): Dictionary of numpy arrays (can be nested or flat)
    """
    print("\n" + "="*60)
    print("READER OUTPUT SUMMARY")
    print("="*60)

    # Check if this is a nested dictionary (case_timestep structure)
    first_key = next(iter(arrays_dict))
    is_nested = isinstance(arrays_dict[first_key], dict)

    if is_nested:
        # Handle nested structure: {case_timestep: {variable: array}}
        for case_timestep, variables in arrays_dict.items():
            print(f"\n{case_timestep}:")
            print("-" * 60)
            for var_name, array in variables.items():
                print(f"  {var_name}:")
                print(f"    Shape: {array.shape},  Min: {np.min(array):.6e},  Max: {np.max(array):.6e},  Mean: {np.mean(array):.6e}")
    else:
        # Handle flat structure: {variable: array}
        for key, array in arrays_dict.items():
            print(f"{key}:")
            print(f"  Shape: {array.shape},  Min value: {np.min(array):.6e},  Max value: {np.max(array):.6e}   Mean value: {np.mean(array):.6e}")
            print("-" * 40)

# =====================================================================================================================================================
# DATA CLEANING UTILITIES
# =====================================================================================================================================================

def clean_dat_file(input_file, output_file, expected_cols):
    clean_data = []
    bad_lines = []

    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num <= 3:
                continue
            try:
                values = [float(x) for x in line.split()]
                if len(values) == expected_cols:
                    clean_data.append(values)
                else:
                    bad_lines.append((line_num, len(values), line.strip()))

            except ValueError as e:
                bad_lines.append((line_num, 'ERROR', line.strip()))

    if bad_lines:
        print(f"Found {len(bad_lines)} problematic lines")

    np.savetxt(f'monitor_point_plots/{output_file}', clean_data, fmt='%.5E')
    print(f"\nSaved {len(clean_data)} clean lines to {output_file}")

    return np.array(clean_data)

# =====================================================================================================================================================
# PLOTTING UTILITIES
# =====================================================================================================================================================

def get_col(case, cases, colours):
    if len(cases) > 1:
        colour = colours[cases.index(case)]
    else:
        colour = colours[0]
    return colour

def print_flow_info(ux_data, Re_ref, Re_bulk, case, timestep, y_coords=None):

    Re_ref = int(Re_ref)
    if y_coords is not None:
        du = ux_data[0] - ux_data[1]
        dy = y_coords[0] - y_coords[1]
    else:
        du = ux_data[0, 2] - ux_data[1, 2]
        dy = ux_data[0, 1] - ux_data[1, 1]
    dudy = np.mean(du/dy)  # average over any spatial dims for scalar output
    tau_w = dudy/Re_ref # this should be ref Re not real bulk Re
    u_tau = np.sqrt(abs(dudy/Re_ref))
    Re_tau = u_tau * Re_ref
    print(f'Case: {case}, Timestep: {timestep}')
    print(f'Re_bulk = {Re_bulk}, u_tau = {u_tau:.6f}, tau_w = {tau_w:.6e}, Re_tau = {Re_tau:.2f}')
    print('-'*120)
    return

def get_plane_data(domain_array, plane, index):
    if plane == 'xy':
        return domain_array[index, :, :]
    elif plane == 'xz':
        return domain_array[:, index, :]
    elif plane == 'yz':
        return domain_array[:, :, index]
    else:
        print(f"Error: Invalid plane '{plane}' specified. Use 'xy', 'xz', or 'yz'.")
    return
