#!/usr/bin/env python3
"""
Pre-processing mesh resolution analysis for CHAPSim2.

Python port of the `apx_prerun_mod` routines `estimate_spacial_resolution` and
`estimate_temporal_resolution` (CHAPSim2/src/input_general.f90), so that a mesh
can be assessed against DNS resolution requirements from `input_chapsim.ini`
alone, before a job is submitted.

The wall-normal grid is rebuilt exactly as `Buildup_geometry_mesh_info` and the
`Buildup_grid_mapping_1D_*` routines (CHAPSim2/src/geometry.f90) build it, so
the reported spacings are those the solver will actually use.

Only wall-bounded cases (channel, pipe, annular) are assessed, matching the
solver, which returns early for every other icase.

Usage
-----
    python mesh_analysis.py [path/to/input_chapsim.ini]

With no argument the script prompts for a path, defaulting to
'input_chapsim.ini' in the current directory.
"""

import os
import sys

import numpy as np

# ====================================================================================================================================================
# CHAPSim2 parameter enumerations (src/modules.f90)
# ====================================================================================================================================================

ICASE_OTHERS = 0
ICASE_CHANNEL = 1
ICASE_PIPE = 2
ICASE_ANNULAR = 3

ICARTESIAN = 1
ICYLINDRICAL = 2

ISTRET_NO = 0
ISTRET_CENTRE = 1
ISTRET_2SIDES = 2
ISTRET_BOTTOM = 3
ISTRET_TOP = 4

MSTRET_NONE = 0
MSTRET_3FMD = 1
MSTRET_TANH = 2
MSTRET_POWL = 3

CASE_NAMES = {0: 'others', 1: 'channel', 2: 'pipe', 3: 'annular', 4: 'TGV3D',
              5: 'duct', 6: 'TGV2D', 7: 'Burgers', 8: 'algorithm test'}
ISTRET_NAMES = {0: 'uniform', 1: 'centre clustered', 2: 'two-side clustered',
                3: 'bottom clustered', 4: 'top clustered', 5: 'from input file'}
MSTRET_NAMES = {0: 'uniform', 1: '3-parameter function (Leizet2009JCP)',
                2: 'hyperbolic tangent', 3: 'power law'}

# ifluid index -> medium name understood by utils.get_fluid_properties.
# Supercritical water/CO2 (1, 2) have no property class in the toolkit.
FLUID_NAMES = {1: 'supercritical water', 2: 'supercritical CO2', 3: 'sodium',
               4: 'lead', 5: 'bismuth', 6: 'LBE', 7: 'water', 8: 'lithium',
               9: 'FLiBe', 10: 'PbLi'}
FLUID_TOOLKIT_KEYS = {3: 'sodium', 4: 'lead', 5: 'bismuth', 6: 'lbe',
                      8: 'lithium', 9: 'flibe', 10: 'pbli'}

# ====================================================================================================================================================
# Enumerated input values
# ====================================================================================================================================================
#
# CHAPSim2 input files name their enumerated settings:
#
#     icase=  channel
#     istret= twosides
#     rstret= 3fmd,0.10
#     ifluid= scp_water
#
# The solver's parse_icase/parse_istret/parse_rstret/parse_ifluid
# (src/input_general.f90) each accept the name or the equivalent integer, so
# the tables below carry both. Reading one as a plain int would quietly fall
# back to the default and assess a mesh the solver is not going to build.
#
# The canonical name is listed first; it is what write_input_file() emits.

ICASE_TOKENS = {
    'channel': ICASE_CHANNEL, '1': ICASE_CHANNEL,
    'pipe': ICASE_PIPE, '2': ICASE_PIPE,
    'annular': ICASE_ANNULAR, '3': ICASE_ANNULAR,
    'tgv3d': 4, '4': 4,
    'duct': 5, '5': 5,
    'tgv2d': 6, '6': 6,
    'burgers': 7, '7': 7,
    'algtest': 8, '8': 8,
    'others': ICASE_OTHERS, '0': ICASE_OTHERS,
}

ISTRET_TOKENS = {
    'no': ISTRET_NO, 'none': ISTRET_NO, '0': ISTRET_NO,
    'centre': ISTRET_CENTRE, 'center': ISTRET_CENTRE, '1': ISTRET_CENTRE,
    '2sides': ISTRET_2SIDES, 'twosides': ISTRET_2SIDES, '2': ISTRET_2SIDES,
    'bottom': ISTRET_BOTTOM, '3': ISTRET_BOTTOM,
    'top': ISTRET_TOP, '4': ISTRET_TOP,
}

MSTRET_TOKENS = {
    'uniform': MSTRET_NONE, 'none': MSTRET_NONE, '0': MSTRET_NONE,
    '3fmd': MSTRET_3FMD, '1': MSTRET_3FMD,
    'tanh': MSTRET_TANH, '2': MSTRET_TANH,
    'powl': MSTRET_POWL, 'powerlaw': MSTRET_POWL, '3': MSTRET_POWL,
}

IFLUID_TOKENS = {
    'scp_water': 1, '1': 1,
    'scp_co2': 2, '2': 2,
    'sodium': 3, '3': 3,
    'lead': 4, '4': 4,
    'bismuth': 5, '5': 5,
    'lbe': 6, '6': 6,
    'water': 7, '7': 7,
    'lithium': 8, '8': 8,
    'flibe': 9, '9': 9,
    'pbli': 10, '10': 10,
}


def _canonical_tokens(tokens):
    """Invert an enum table to value -> canonical (first-listed) name."""
    names = {}
    for name, value in tokens.items():
        if not name.isdigit():
            names.setdefault(value, name)
    return names


ICASE_INPUT_NAMES = _canonical_tokens(ICASE_TOKENS)
ISTRET_INPUT_NAMES = _canonical_tokens(ISTRET_TOKENS)
MSTRET_INPUT_NAMES = _canonical_tokens(MSTRET_TOKENS)
IFLUID_INPUT_NAMES = _canonical_tokens(IFLUID_TOKENS)


def _enum(tokens, what):
    """Build a cast that maps an input token to its integer enum value.

    Raises ValueError on an unrecognised token so get_entry() reports it
    rather than silently substituting a default.
    """
    def cast(token):
        key = str(token).strip().strip("'\"").lower()
        if key in tokens:
            return tokens[key]
        raise ValueError(
            f"Invalid {what} '{token}'. Supported values: "
            f"{', '.join(n for n in tokens if not n.isdigit())}.")

    cast.__name__ = f'_enum_{what}'
    return cast

# ====================================================================================================================================================
# Empirical DNS resolution limits (apx_prerun_mod parameters)
# ====================================================================================================================================================

DXPLUS_MAX = 10.0
DZPLUS_MAX = 5.0
DYPLUS_MAX = 1.0
CFL_MAX = 0.714
CTM_MAX = 0.1

MINP = 1.0e-20  # parameters_constant_mod

# Not from apx_prerun_mod: the solver reports the centreline spacing but sets no
# limit on it. This is the usual DNS rule of thumb for the coarsest wall-normal
# cell (Kim, Moin & Moser 1987; Moin & Mahesh 1998), used only to flag the value
# in the GUI. It is deliberately kept out of the ported report.
DYPLUS_CENTRE_MAX = 10.0

# ====================================================================================================================================================
# Input file parsing
# ====================================================================================================================================================


def parse_input_file(path):
    """
    Read an `input_chapsim.ini` file into {section: [(key, value), ...]}.

    Entries are kept in file order so that a key can also be recovered by its
    position within a section, mirroring the solver's sequential reads.

    Parameters
    ----------
    path : str
        Path to the CHAPSim2 input file.

    Returns
    -------
    dict
        Section name (lower case, without brackets) mapped to an ordered list
        of (key, raw value string) pairs.
    """
    sections = {}
    current = None

    with open(path, 'r') as f:
        for raw_line in f:
            line = raw_line.split('!')[0].split('#')[0].strip()
            if not line:
                continue

            if line.startswith('['):
                current = line.split(']')[0].lstrip('[').strip().lower()
                sections.setdefault(current, [])
                continue

            if current is None or '=' not in line:
                continue

            key, _, value = line.partition('=')
            sections[current].append((key.strip().lower(), value.strip()))

    return sections


def _tokens(value):
    """Split a Fortran list-directed value into its individual tokens."""
    return [tok for tok in value.replace(',', ' ').split() if tok]


def _to_float(token):
    """Convert a Fortran real literal (including 'd' exponents) to a float."""
    return float(token.lower().replace('d', 'e'))


def _to_bool(token):
    """Convert a Fortran logical literal ('.true.'/'.false.') to a bool."""
    return token.strip().strip('.').lower().startswith('t')


def _first_key(sections, section, *keys):
    """Return the first of `keys` present in `section`, else the first given.

    Lets a renamed input key be read under either spelling without the
    caller having to branch.
    """
    present = {entry_key for entry_key, _ in sections.get(section, [])}
    return next((k for k in keys if k in present), keys[0])


def get_entry(sections, section, key, cast=str, index=0, default=None, fuzzy=False):
    """
    Fetch a single value from a parsed input file.

    Parameters
    ----------
    sections : dict
        Output of `parse_input_file`.
    section : str
        Section name without brackets, e.g. 'mesh'.
    key : str
        Key name as written in the file, e.g. 'ncy'.
    cast : callable
        Conversion applied to the selected token.
    index : int
        Token index within the value, used for multi-domain or vector entries.
    default : object
        Returned when the section, key or token is absent.
    fuzzy : bool
        Match any key containing `key` as a substring rather than exactly.
        Used for the [mhd] section, whose key names vary between input files.

    Returns
    -------
    object
        The converted value, or `default`.
    """
    for entry_key, value in sections.get(section, []):
        matched = key in entry_key if fuzzy else entry_key == key
        if not matched:
            continue
        tokens = _tokens(value)
        if index >= len(tokens):
            return default
        try:
            return cast(tokens[index])
        except ValueError as exc:
            # A value that is present but unreadable means the analysis would
            # describe a different mesh from the one the solver builds, so say
            # so rather than quietly falling back.
            print(f"  Warning: [{section}] {entry_key}: {exc} Using {default!r}.")
            return default

    return default


# ====================================================================================================================================================
# Input file generation
# ====================================================================================================================================================

# Structure taken from CHAPSim2/tests/regression/channel_iso_periodic and
# channel_scp_inout_Tw. Generation rewrites values into this text in place
# rather than assembling a file from scratch, so everything the mesh analysis
# does not own (boundary conditions, schemes, io, probes) stays as the solver
# expects it.
DEFAULT_INPUT_TEMPLATE = """[process]
is_prerun= .false.
is_postprocess= .false.

[decomposition]
nxdomain= 1
p_row= 0
p_col= 0

[domain]
icase= channel
lxx= 8.0
lyt= 1.0
lyb= -1.0
lzz= 4.0

[flow]
initfl= poiseuille
irestartfrom= 0
veloinit= 0.0,0.0,0.0
noiselevel= 0.25
is_active_tripping= .false.
reni= 5000
nreni= 10000
ren= 5000

[mesh]
ncx= 64
ncy= 80
ncz= 64
istret= twosides
rstret= 3fmd,0.10

[bc]
ifbcx_u= 1,1,0.0,0.0
ifbcx_v= 1,1,0.0,0.0
ifbcx_w= 1,1,0.0,0.0
ifbcx_p= 1,1,0.0,0.0
ifbcx_t= 1,1,0.0,0.0
ifbcy_u= 4,4,0.0,0.0
ifbcy_v= 4,4,0.0,0.0
ifbcy_w= 4,4,0.0,0.0
ifbcy_p= 5,5,0.0,0.0
ifbcy_t= 4,4,0.0,0.0
ifbcz_u= 1,1,0.0,0.0
ifbcz_v= 1,1,0.0,0.0
ifbcz_w= 1,1,0.0,0.0
ifbcz_p= 1,1,0.0,0.0
ifbcz_t= 1,1,0.0,0.0
idriven= x_massflux
drivenfc= 0.0

[scheme]
dt= 1e-03
itimescheme= rk3
iaccuracy= cp4
iviscous= explicit
out_sponge_L_Re= 0.0, 100.0

[thermo]
ithermo= .false.
icht= .false.
igravity= 0.0,0.0,0.0
ifluid= water
ref_l0= 0.05
ref_t0= 570.0
inittm= const
irestartfrom= 0
tini= 570.0
inout_buffer= 0.0, 0.0
qw_ramp= .false., 0, 0

[mhd]
imhd_xdom= .false.
NStuart= .false., 0.0
NHartmn= .false., 0.0
B_static= 0.0, 1.0, 0.0

[simcontrol]
niterflowfirst= 1
niterflowlast= 60
niterthermofirst= 0
niterthermolast= 0

[io]
cpu_nfre= 10
ckpt_nfre= 50
visu_idim= 2
visu_nfre= 20
visu_nskip= 1,1,1
stat_istart= 30
stat_level= 3
stat_nskip= 1,1,1
is_record_xoutlet_read_xinlet= .false.,.false.
ndbfre_ndbstart_ndbend= 20,21,60
existing_output_policy= overwrite
restart_data_layout= bundled

[probe]
npp= 1
pt1= 3.141593,0.0,1.5707965
"""

MHD_SECTION_TEMPLATE = """
[mhd]
imhd_xdom= {imhd}
NStuart= .false., 0.0
NHartmn= {nhartmn}
B_static= 0.0, 1.0, 0.0
"""


def _fmt_real(value):
    """Format a real so Fortran list-directed input always reads it as a real."""
    text = f"{value:.10g}"
    if any(c in text for c in '.eE'):
        return text

    return text + '.0'


def _fmt_logical(flag):
    """Format a Fortran logical literal."""
    return '.true.' if flag else '.false.'


def write_input_file(cfg, path, template=None):
    """
    Write an `input_chapsim.ini` reflecting the current configuration.

    Values are substituted into an existing file's text rather than a new file
    being assembled, because the solver reads each section line by line: an
    omitted or reordered entry would be silently misparsed. Everything the mesh
    analysis does not own (boundary conditions, io, probes) is preserved
    verbatim from the template.

    Parameters
    ----------
    cfg : DomainConfig
        Configuration to write.
    path : str
        Destination file path.
    template : str, optional
        Template file contents. Defaults to a channel case built from
        CHAPSim2's own test input.

    Returns
    -------
    str
        The text written.
    """
    text = template if template is not None else DEFAULT_INPUT_TEMPLATE

    # (section, key) -> replacement value, matched case-insensitively.
    # Enumerated settings are written as names, which is how CHAPSim2 input
    # files read today; the solver still accepts the equivalent integers.
    managed = {
        ('domain', 'icase'): ICASE_INPUT_NAMES.get(cfg.icase, str(cfg.icase)),
        ('domain', 'lxx'): _fmt_real(cfg.lxx),
        ('domain', 'lyt'): _fmt_real(cfg.lyt),
        ('domain', 'lyb'): _fmt_real(cfg.lyb),
        ('domain', 'lzz'): _fmt_real(cfg.lzz),
        ('mesh', 'ncx'): str(cfg.nc[0]),
        ('mesh', 'ncy'): str(cfg.nc[1]),
        ('mesh', 'ncz'): str(cfg.nc[2]),
        ('mesh', 'istret'): ISTRET_INPUT_NAMES.get(cfg.istret, str(cfg.istret)),
        ('mesh', 'rstret'): (f"{MSTRET_INPUT_NAMES.get(cfg.mstret, cfg.mstret)},"
                             f"{_fmt_real(cfg.rstret)}"),
        ('flow', 'ren'): _fmt_real(cfg.ren),
        ('scheme', 'dt'): _fmt_real(cfg.dt),
        ('thermo', 'ithermo'): _fmt_logical(cfg.is_thermo),
        ('thermo', 'ifluid'): IFLUID_INPUT_NAMES.get(cfg.ifluid, str(cfg.ifluid)),
        # imhd was renamed imhd_xdom; a loaded template may use either.
        ('mhd', 'imhd'): _fmt_logical(cfg.is_mhd),
        ('mhd', 'imhd_xdom'): _fmt_logical(cfg.is_mhd),
        ('mhd', 'nhartmn'): (f"{_fmt_logical(cfg.is_mhd)}, "
                             f"{_fmt_real(cfg.hartmann or 0.0)}"),
    }
    if cfg.ref_t0 is not None:
        managed[('thermo', 'ref_t0')] = _fmt_real(cfg.ref_t0)

    out_lines = []
    section = None
    seen_sections = set()

    for raw_line in text.splitlines():
        stripped = raw_line.strip()

        if stripped.startswith('['):
            section = stripped.split(']')[0].lstrip('[').strip().lower()
            seen_sections.add(section)
            out_lines.append(raw_line)
            continue

        if section is None or '=' not in stripped:
            out_lines.append(raw_line)
            continue

        key = stripped.partition('=')[0].strip()
        replacement = managed.get((section, key.lower()))
        if replacement is None:
            out_lines.append(raw_line)
            continue

        out_lines.append(f"{key}= {replacement}")

    out_text = '\n'.join(out_lines).rstrip('\n') + '\n'

    # A loaded template may predate the MHD model; add the section if needed.
    if 'mhd' not in seen_sections and cfg.is_mhd:
        out_text += MHD_SECTION_TEMPLATE.format(
            imhd=_fmt_logical(True),
            nhartmn=f".true., {_fmt_real(cfg.hartmann or 0.0)}")

    with open(path, 'w') as f:
        f.write(out_text)

    return out_text


# ====================================================================================================================================================
# Domain configuration
# ====================================================================================================================================================


class DomainConfig:
    """
    Domain, mesh and flow settings needed for the resolution assessment.

    The defaults applied by `Read_input_parameters` (src/input_general.f90) are
    reproduced here: the case type overrides the y and z extents, fixes the
    coordinate system, and forces an even spanwise cell count in cylindrical
    coordinates.

    Only the first x-domain is analysed, matching the solver, which assesses
    domain(1) only.
    """

    def __init__(self, sections=None, path=None):
        sections = sections if sections is not None else {}
        self.path = path
        self.nxdomain = get_entry(sections, 'decomposition', 'nxdomain', int, default=1)

        # [domain]
        self.icase = get_entry(sections, 'domain', 'icase', _enum(ICASE_TOKENS, 'icase'),
                               default=ICASE_OTHERS)
        self.lxx = get_entry(sections, 'domain', 'lxx', _to_float, default=0.0)
        self.lyt = get_entry(sections, 'domain', 'lyt', _to_float, default=1.0)
        self.lyb = get_entry(sections, 'domain', 'lyb', _to_float, default=-1.0)
        self.lzz = get_entry(sections, 'domain', 'lzz', _to_float, default=0.0)

        # [mesh]
        self.nc = [get_entry(sections, 'mesh', 'ncx', int, default=0),
                   get_entry(sections, 'mesh', 'ncy', int, default=0),
                   get_entry(sections, 'mesh', 'ncz', int, default=0)]

        self.istret = get_entry(sections, 'mesh', 'istret', _enum(ISTRET_TOKENS, 'istret'),
                                default=ISTRET_NO)
        self.mstret = get_entry(sections, 'mesh', 'rstret', _enum(MSTRET_TOKENS, 'rstret method'),
                                index=0, default=MSTRET_3FMD)
        self.rstret = get_entry(sections, 'mesh', 'rstret', _to_float, index=1, default=0.0)

        # [flow] and [scheme]
        self.ren = get_entry(sections, 'flow', 'ren', _to_float, default=0.0)
        self.dt = get_entry(sections, 'scheme', 'dt', _to_float, default=0.0)

        # [thermo]
        self.is_thermo = get_entry(sections, 'thermo', 'ithermo', _to_bool, default=False)
        self.ifluid = get_entry(sections, 'thermo', 'ifluid', _enum(IFLUID_TOKENS, 'ifluid'),
                                default=0)
        self.ref_t0 = get_entry(sections, 'thermo', 'ref_t0', _to_float, default=None)

        # [mhd] - key names vary, so match on substrings
        self.is_mhd = get_entry(sections, 'mhd', 'mhd', _to_bool, default=False, fuzzy=True)
        self.hartmann = get_entry(sections, 'mhd', 'hartm', _to_float, index=1,
                                  default=None, fuzzy=True)

        # [io], reported by the temporal estimate. Renamed from is_wrt_read_bc
        # to is_record_xoutlet_read_xinlet; both hold the same
        # (record outlet, read inlet) pair.
        xbc_key = _first_key(sections, 'io',
                             'is_record_xoutlet_read_xinlet', 'is_wrt_read_bc')
        self.is_record_xoutlet = get_entry(sections, 'io', xbc_key, _to_bool,
                                           index=0, default=False)
        self.is_read_xinlet = get_entry(sections, 'io', xbc_key, _to_bool,
                                        index=1, default=False)

        self.apply_case_defaults()

    @classmethod
    def from_values(cls, **kwargs):
        """
        Build a configuration directly from parameter values rather than a file.

        Used by the GUI, where the mesh is varied interactively. Unknown keys
        raise, so a typo cannot silently leave a default in place.

        Parameters
        ----------
        **kwargs
            Any attribute set by `__init__`, e.g. icase, lxx, nc, istret.

        Returns
        -------
        DomainConfig
        """
        cfg = cls()
        for key, value in kwargs.items():
            if not hasattr(cfg, key):
                raise AttributeError(f"Unknown configuration field: {key}")
            setattr(cfg, key, value)
        cfg.apply_case_defaults()

        return cfg

    def apply_case_defaults(self):
        """
        Apply the case-dependent overrides and derived quantities.

        Mirrors `Read_input_parameters`: the case fixes the y and z extents and
        the coordinate system, cylindrical cases need an even spanwise cell
        count, and the homogeneous spacings follow from the cell counts.

        Idempotent, so it can be re-run after any field is changed.
        """
        if self.icase == ICASE_CHANNEL:
            self.lyb, self.lyt = -1.0, 1.0
        elif self.icase == ICASE_PIPE:
            self.lyb, self.lyt = 0.0, 1.0
            self.lzz = 2.0 * np.pi
        elif self.icase == ICASE_ANNULAR:
            self.lyt = 1.0
            self.lzz = 2.0 * np.pi

        self.icoordinate = (ICYLINDRICAL if self.icase in (ICASE_PIPE, ICASE_ANNULAR)
                            else ICARTESIAN)

        if self.icoordinate == ICYLINDRICAL and self.nc[2] % 2 != 0:
            self.nc[2] += 1

        self.is_stretching = self.istret != ISTRET_NO

        self.hx = self.lxx / self.nc[0] if self.nc[0] else 0.0
        self.hz = self.lzz / self.nc[2] if self.nc[2] else 0.0

    @property
    def is_wall_bounded(self):
        """True for the cases the solver assesses (channel, pipe, annular)."""
        return self.icase in (ICASE_CHANNEL, ICASE_PIPE, ICASE_ANNULAR)

    def summary(self):
        """Print the parsed configuration."""
        print(f"  Case                     : {CASE_NAMES.get(self.icase, self.icase)}")
        print("  Coordinate system        : "
              f"{'cylindrical' if self.icoordinate == ICYLINDRICAL else 'Cartesian'}")
        print(f"  Domain (Lx, Ly, Lz)      : {self.lxx:.6g}, "
              f"{self.lyt - self.lyb:.6g}, {self.lzz:.6g}")
        print(f"  y extent (lyb, lyt)      : {self.lyb:.6g}, {self.lyt:.6g}")
        print(f"  Cells (Ncx, Ncy, Ncz)    : {self.nc[0]}, {self.nc[1]}, {self.nc[2]}")
        print(f"  Stretching (istret)      : {ISTRET_NAMES.get(self.istret, self.istret)}")
        if self.is_stretching:
            print("  Stretch function (mstret): "
                  f"{MSTRET_NAMES.get(self.mstret, self.mstret)}")
            print(f"  Stretch factor (rstret)  : {self.rstret:.6g}")
        print(f"  Reynolds number (ren)    : {self.ren:.6g}")
        print(f"  Time step (dt)           : {self.dt:.6g}")
        print(f"  Thermal field solved     : {self.is_thermo}")
        if self.is_thermo:
            print(f"  Working fluid            : "
                  f"{FLUID_NAMES.get(self.ifluid, self.ifluid)}")
            if self.ref_t0 is not None:
                print(f"  Reference temperature    : {self.ref_t0:.6g} K")
        print(f"  MHD solved               : {self.is_mhd}")
        if self.is_mhd and self.hartmann is not None:
            print(f"  Hartmann number          : {self.hartmann:.6g}")
        if self.nxdomain > 1:
            print(f"\n  Note: {self.nxdomain} x-domains found; domain 1 is assessed, "
                  "as in the solver.")


# ====================================================================================================================================================
# Wall-normal grid construction (src/geometry.f90)
# ====================================================================================================================================================


def _heaviside(r):
    """Heaviside step as defined in math_mod: 0.5 at the origin."""
    return np.where(r > MINP, 1.0, np.where(r < -MINP, 0.0, 0.5))


def _eta(n, kind):
    """
    Uniform computational coordinate eta in [0, 1].

    Parameters
    ----------
    n : int
        Number of points to generate.
    kind : {'nd', 'cl'}
        'nd' for nodes (endpoints included), 'cl' for cell centres (staggered
        by half a cell).
    """
    if kind == 'nd':
        shift, delta = 0.0, 1.0 / (n - 1)
    elif kind == 'cl':
        shift, delta = 0.5 / n, 1.0 / n
    else:
        raise ValueError(f"Grid stretching location not defined: {kind}")

    return shift + np.arange(n) * delta


def _map_uniform(n, kind, cfg):
    """Unstretched mapping, used when istret == ISTRET_NO."""
    return _eta(n, kind) * (cfg.lyt - cfg.lyb) + cfg.lyb


def _map_3fmd(n, kind, cfg):
    """Three-parameter mapping of Leizet (2009) JCP, Eq. (53)."""
    eta = _eta(n, kind)

    if cfg.istret == ISTRET_2SIDES:
        gamma, delta = 1.0, 0.5
    elif cfg.istret == ISTRET_BOTTOM:
        gamma, delta = 0.5, 0.5
    elif cfg.istret == ISTRET_TOP:
        gamma, delta = 0.5, 0.0
    elif cfg.istret == ISTRET_CENTRE:
        # The solver warns here: the mapping is non-monotone for this option.
        gamma, delta = 1.0, 0.0
    else:
        raise ValueError(f"Grid stretching flag is not valid: {cfg.istret}")

    beta = cfg.rstret
    alpha = (-1.0 + np.sqrt(1.0 + 4.0 * np.pi**2 * beta**2)) / beta * 0.5

    cc = np.sqrt(alpha * beta + 1.0) / np.sqrt(beta)
    dd = cc / np.sqrt(alpha)
    ee = cc * np.sqrt(alpha)

    st1 = (1.0 - 2.0 * delta) / gamma * 0.5
    st2 = (3.0 - 2.0 * delta) / gamma * 0.5

    mm = np.pi * (gamma * eta + delta)
    y = (np.arctan(dd * np.tan(mm))
         - np.arctan(dd * np.tan(np.pi * delta))
         + np.pi * (_heaviside(eta - st1) + _heaviside(eta - st2)))
    y = y / (gamma * ee)

    if kind == 'nd':
        y[0] = 0.0
        y[-1] = 1.0

    return y * (cfg.lyt - cfg.lyb) + cfg.lyb


def _map_tanh(n, kind, cfg):
    """Hyperbolic tangent mapping."""
    eta = _eta(n, kind)

    if cfg.istret == ISTRET_2SIDES:
        ymin, ymax = -1.0, 1.0
    elif cfg.istret in (ISTRET_BOTTOM, ISTRET_TOP):
        ymin, ymax = 0.0, 1.0
    else:
        raise ValueError(f"Grid stretching flag is not valid: {cfg.istret}")

    beta = cfg.rstret * 20.0

    if cfg.istret == ISTRET_2SIDES:
        mm = np.tanh(beta * 0.5)
        y = np.tanh(beta * (eta - 0.5)) / mm
        y = (y + 1.0) * 0.5
    elif cfg.istret == ISTRET_BOTTOM:
        mm = np.tanh(beta)
        y = 1.0 - np.tanh(beta * (1.0 - eta)) / mm
    else:
        mm = np.tanh(beta)
        y = np.tanh(beta * eta) / mm

    ff = (cfg.lyt - cfg.lyb) / (ymax - ymin)

    return (y - ymin) * ff + cfg.lyb


def _map_powerlaw(n, kind, cfg):
    """Power-law mapping. Debug option in the solver, not intended for DNS."""
    eta = _eta(n, kind)

    if cfg.istret not in (ISTRET_2SIDES, ISTRET_BOTTOM, ISTRET_TOP):
        raise ValueError(f"Grid stretching flag is not valid: {cfg.istret}")

    beta = cfg.rstret
    if beta <= MINP:
        raise ValueError("Powerlaw stretching factor must be positive.")
    expo = 1.0 / beta

    if cfg.istret == ISTRET_2SIDES:
        y = np.where(eta <= 0.5,
                     0.5 * (2.0 * eta)**expo,
                     1.0 - 0.5 * (2.0 * (1.0 - eta))**expo)
    elif cfg.istret == ISTRET_BOTTOM:
        y = eta**expo
    else:
        y = 1.0 - (1.0 - eta)**expo

    ymin, ymax = 0.0, 1.0
    ff = (cfg.lyt - cfg.lyb) / (ymax - ymin)

    return (y - ymin) * ff + cfg.lyb


def build_y_grid(cfg):
    """
    Build the wall-normal node and cell-centre coordinates.

    Reproduces `Buildup_geometry_mesh_info`: yp holds np_geo(2) = Ncy + 1 nodes
    and yc holds Ncy cell centres, each generated from its own mapping rather
    than yc being the average of neighbouring yp.

    Parameters
    ----------
    cfg : DomainConfig

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Node coordinates yp and cell-centre coordinates yc.
    """
    n_node = cfg.nc[1] + 1
    n_cell = cfg.nc[1]

    if not cfg.is_stretching:
        h2 = (cfg.lyt - cfg.lyb) / n_cell
        yp = np.arange(n_node) * h2 + cfg.lyb
        yc = np.arange(n_cell) * h2 + h2 * 0.5 + cfg.lyb
        return yp, yc

    mappings = {MSTRET_NONE: _map_uniform,
                MSTRET_3FMD: _map_3fmd,
                MSTRET_TANH: _map_tanh,
                MSTRET_POWL: _map_powerlaw}
    if cfg.mstret not in mappings:
        raise ValueError(f"Unsupported stretching method: {cfg.mstret}")
    mapping = mappings[cfg.mstret]

    yp = mapping(n_node, 'nd', cfg)
    yc = mapping(n_cell, 'cl', cfg)

    return yp, yc


def check_y_grid(yp, cfg):
    """
    Warn if the generated grid does not span the requested y extent.

    The mapping routines renormalise the stretched coordinate onto [lyb, lyt];
    a mismatch here means the chosen istret/mstret combination does not cover
    the domain, which would otherwise show up only as odd wall spacings.
    """
    tol = 1.0e-8 * max(1.0, abs(cfg.lyt - cfg.lyb))

    if abs(yp[0] - cfg.lyb) > tol or abs(yp[-1] - cfg.lyt) > tol:
        print(f"\n  ! Warning: generated grid spans [{yp[0]:.6g}, {yp[-1]:.6g}] "
              f"but the domain is [{cfg.lyb:.6g}, {cfg.lyt:.6g}].")
        print("    The istret/mstret combination does not cover the full y extent; "
              "spacings below reflect the grid as built.")

        if cfg.mstret == MSTRET_TANH and cfg.istret == ISTRET_2SIDES:
            print("    Cause: in Buildup_grid_mapping_1D_tanh (geometry.f90) the "
                  "two-side branch normalises")
            print("    the mapping onto [0, 1] and then rescales it as if it spanned "
                  "[-1, 1], so only half")
            print("    the domain is covered. Use mstret = 1 (3fmd) for two-side "
                  "clustering until this is fixed.")

    if np.any(np.diff(yp) <= 0.0):
        print("\n  ! Warning: the wall-normal grid is not monotonically increasing.")


# ====================================================================================================================================================
# Skin friction correlations (apx_prerun_mod)
# ====================================================================================================================================================


def solve_prandtl_von_karman_cf(Re, icase):
    """
    Solve the Prandtl-von Karman relation for the friction factor by Newton
    iteration, as `solve_Prandtl_vonKarman_eq_for_cf` does.

    Parameters
    ----------
    Re : float
        Reynolds number.
    icase : int
        Case identifier, selecting the correlation constants.

    Returns
    -------
    float
        Friction factor cf.
    """
    if icase in (ICASE_ANNULAR, ICASE_PIPE):
        a, b = 2.0, -0.8
    else:
        a, b = 2.12, -0.65

    cf = 0.005
    tol = 1.0e-6

    for _ in range(50):
        f = 1.0 / np.sqrt(cf) - a * np.log10(Re * np.sqrt(cf)) + b
        df = (-0.5 / cf**1.5
              - a / (np.log(10.0) * (Re * np.sqrt(cf)) * 2.0 * np.sqrt(cf)))
        cf_new = cf - f / df
        # The solver exits before accepting the converged update; kept for parity.
        if abs(cf_new - cf) < tol:
            break
        cf = cf_new

    return cf


def estimate_skin_friction_factor(Re, icase):
    """
    Estimate the skin friction factor from empirical correlations.

    Pipe and annular flow use laminar, Blasius, McAdams or Prandtl-von Karman
    depending on Re; channel flow uses a laminar fit or Prandtl-von Karman.

    Parameters
    ----------
    Re : float
        Reynolds number.
    icase : int
        Case identifier.

    Returns
    -------
    (float, str)
        Friction factor cf and the name of the correlation used.
    """
    if icase in (ICASE_PIPE, ICASE_ANNULAR):
        if Re < 2300.0:
            return 64.0 / Re, 'laminar'
        if Re < 3.0e4:
            return 0.316 * Re**(-0.25), 'Blasius'
        if Re < 1.0e6:
            return 0.814 * Re**(-0.2), 'McAdams'
        return solve_prandtl_von_karman_cf(Re, icase), 'Prandtl-von Karman'

    if icase == ICASE_CHANNEL:
        if Re < 1.0e4:
            return 0.079 * Re**(-0.25), 'laminar'
        return solve_prandtl_von_karman_cf(Re, icase), 'Prandtl-von Karman'

    raise ValueError(f"No skin friction correlation for icase {icase}")


def reference_prandtl_number(cfg):
    """
    Prandtl number at the reference temperature, used to tighten the wall-normal
    limit when a thermal field is solved (the solver uses fluidparam%ftp0ref%Pr).

    Parameters
    ----------
    cfg : DomainConfig

    Returns
    -------
    (float or None, str)
        Prandtl number and a message describing its source, or (None, reason)
        when it cannot be evaluated.
    """
    if cfg.ref_t0 is None:
        return None, "no reference temperature (ref_t0) in the input file"

    if cfg.ifluid not in FLUID_TOOLKIT_KEYS:
        return None, (f"no property data for "
                      f"'{FLUID_NAMES.get(cfg.ifluid, cfg.ifluid)}' in the toolkit")

    try:
        from thermal_BC_calc import get_prandtl
        from utils import get_fluid_properties
    except ImportError as exc:
        return None, f"property modules unavailable ({exc})"

    fluid = get_fluid_properties(FLUID_TOOLKIT_KEYS[cfg.ifluid])
    Pr = get_prandtl(cfg.ref_t0, fluid)

    return Pr, (f"{FLUID_NAMES[cfg.ifluid]} at {cfg.ref_t0:.6g} K")


# ====================================================================================================================================================
# Spatial resolution assessment (estimate_spacial_resolution)
# ====================================================================================================================================================


def wall_dyplus(cfg, res):
    """
    Wall-adjacent Delta y+, accounting for which ends of the y range are walls.

    The solver reports yplus1 as the "near wall" value, but for a pipe j=1 sits
    on the axis (lyb = 0) and the only wall is at lyt, so yplus1 there is the
    centreline spacing. Channel and annular cases have walls at both ends, so
    the worse of the two is the meaningful figure.

    Parameters
    ----------
    cfg : DomainConfig
    res : dict
        Output of `analyse_spatial_resolution`.

    Returns
    -------
    float
        Delta y+ in the first cell off the wall.
    """
    if cfg.icase == ICASE_PIPE:
        return res['yplus3']

    return max(res['yplus1'], res['yplus3'])


def analyse_spatial_resolution(cfg, yp, yc):
    """
    Assess the mesh against DNS resolution requirements.

    Parameters
    ----------
    cfg : DomainConfig
    yp, yc : numpy.ndarray
        Wall-normal node and cell-centre coordinates.

    Returns
    -------
    dict
        Computed quantities, reused by the temporal estimate and the plot.
    """
    np2 = len(yp)  # np(2): y is never periodic for the wall-bounded cases

    # Inner and outer radii for cylindrical coordinates
    rmin, rmax = 1.0, 1.0
    if cfg.icoordinate == ICYLINDRICAL:
        rmin = yc[0]
        rmax = yp[np2 - 1]

    cf, correlation = estimate_skin_friction_factor(cfg.ren, cfg.icase)
    Re_tau = cfg.ren * np.sqrt(cf / 2.0)
    if cfg.icase == ICASE_PIPE:
        Re_tau = Re_tau / 2.0
    u_tau = Re_tau / cfg.ren

    # Sampled wall-normal spacings: near wall, centre, far wall
    j_mid = np2 // 2
    dy1 = yp[1] - yp[0]
    dy2 = yp[j_mid - 1] - yp[j_mid - 2]
    dy3 = yp[np2 - 1] - yp[np2 - 2]
    dy32 = yp[np2 - 2] - yp[np2 - 3]
    dy33 = yp[np2 - 3] - yp[np2 - 4]

    yplus1 = Re_tau * dy1
    yplus2 = Re_tau * dy2
    yplus3 = Re_tau * dy3
    dxplus = Re_tau * cfg.hx
    dzplus = Re_tau * cfg.hz * rmax
    dzplus2 = Re_tau * cfg.hz * rmin if cfg.icoordinate == ICYLINDRICAL else None

    # Cell-to-cell growth over the outermost near-wall layers
    growth_rate1 = (abs(dy33 - dy32) + min(dy33, dy32)) / min(dy33, dy32)
    growth_rate2 = (abs(dy32 - dy3) + min(dy32, dy3)) / min(dy32, dy3)

    dymax = max(dy1, dy2, dy3)
    dymin = min(dy1, dy2, dy3)

    # Recommended minimum resolution
    Pr, pr_source = (reference_prandtl_number(cfg) if cfg.is_thermo else (None, ''))
    dx_max = DXPLUS_MAX / Re_tau
    dz_max = DZPLUS_MAX / Re_tau / rmax
    dy_max = DYPLUS_MAX / Re_tau
    if cfg.is_thermo and Pr is not None:
        dy_max = min(DYPLUS_MAX, 1.0 / Pr) / Re_tau

    nx_min = int(np.ceil(cfg.lxx / dx_max))
    nz_min = int(np.ceil(cfg.lzz / dz_max))
    ny_min = int(np.ceil(cfg.nc[1] * dymin / dy_max))

    results = {
        'cf': cf, 'correlation': correlation, 'Re_tau': Re_tau, 'u_tau': u_tau,
        'rmin': rmin, 'rmax': rmax,
        'dy1': dy1, 'dy2': dy2, 'dy3': dy3, 'dymin': dymin, 'dymax': dymax,
        'yplus1': yplus1, 'yplus2': yplus2, 'yplus3': yplus3,
        'dxplus': dxplus, 'dzplus': dzplus, 'dzplus2': dzplus2,
        'growth_rate1': growth_rate1, 'growth_rate2': growth_rate2,
        'Pr': Pr, 'pr_source': pr_source,
        'nx_min': nx_min, 'ny_min': ny_min, 'nz_min': nz_min,
    }

    print_spatial_report(cfg, yp, results)

    return results


def print_spatial_report(cfg, yp, res):
    """Print the spatial resolution assessment."""
    Re_tau = res['Re_tau']

    print('\n' + '=' * 100)
    print('DNS MESH RESOLUTION ASSESSMENT')
    print('=' * 100)
    print('Recommended values are based on the empirical correlations in apx_prerun_mod.')

    # Domain lengths
    print('\n' + '-' * 100)
    print('Domain Length Check')
    print('-' * 100)
    print(f"  Streamwise (x): current | recom. min : {cfg.lxx:14.6f} | {2.0 * np.pi:14.6f}")
    if cfg.icase == ICASE_CHANNEL:
        print('    Note: Lx >= 2*pi for channel flow (~4*pi preferred for large-scale structures)')
    elif cfg.icase == ICASE_PIPE:
        print('    Note: Lx >= 2*pi for pipe flow (~8-10 pipe diameters)')
    if cfg.icoordinate == ICARTESIAN:
        print(f"  Spanwise (z):   current | recom. min : {cfg.lzz:14.6f} | {np.pi:14.6f}")
        print('    Note: Lz >= pi for adequate spanwise correlation')

    # Flow parameters
    print('\n' + '-' * 100)
    print('Flow Parameters')
    print('-' * 100)
    print(f"  Friction Reynolds number (Re_tau) : {Re_tau:14.6f}")
    print(f"  Friction velocity (u_tau)         : {res['u_tau']:14.6f}")
    print(f"  Skin friction coefficient (cf)    : {res['cf']:14.6f}  "
          f"({res['correlation']} correlation)")

    # Wall units
    print('\n' + '-' * 100)
    print('Current Mesh Resolution (wall units)')
    print('-' * 100)
    print('  Wall-normal direction (y):')
    print(f"    dy+ at near wall (j=1)          : {res['yplus1']:14.6f}")
    print(f"    dy+ at far wall  (j=np)         : {res['yplus3']:14.6f}")
    print(f"    dy+ at centre    (j=np/2)       : {res['yplus2']:14.6f}")

    if res['yplus1'] > 1.0:
        print('\n  ! Warning: wall spacing too large. dy+ should be <= 1.0 for DNS.')
        print(f"    Current dy+ = {res['yplus1']:.2f} -> increase Ncy or adjust stretching")
    elif res['yplus1'] < 0.5:
        print('\n  + Excellent wall resolution (dy+ < 0.5)')
    else:
        print('\n  + Acceptable wall resolution (dy+ <= 1.0)')

    print('\n  Streamwise direction (x):')
    print(f"    dx+                             : {res['dxplus']:14.6f}")
    print(f"    Recommended: dx+ <= {DXPLUS_MAX:.1f}")

    print('\n  Spanwise direction (z):')
    print(f"    dz+ (at outer wall)             : {res['dzplus']:14.6f}")
    if res['dzplus2'] is not None:
        print(f"    dz+ (at inner wall)             : {res['dzplus2']:14.6f}")
    print(f"    Recommended: dz+ <= {DZPLUS_MAX:.1f}")

    # Grid stretching
    print('\n' + '-' * 100)
    print('Grid Stretching Assessment')
    print('-' * 100)
    print(f"  Growth rate at wall (layers 2-3, 3-4) : "
          f"{res['growth_rate1']:10.6f} | {res['growth_rate2']:10.6f}")
    print('  Recommended: growth rate < 1.2-1.3 for DNS')
    if res['growth_rate1'] > 1.3 or res['growth_rate2'] > 1.3:
        print('  ! Warning: grid stretching is too aggressive. Reduce the stretching factor.')
        print('    High stretching can cause numerical errors and inaccurate statistics.')
    elif res['growth_rate1'] > 1.2 or res['growth_rate2'] > 1.2:
        print('  ~ Caution: growth rate approaching the upper limit')
    else:
        print('  + Grid stretching is acceptable')

    # Additional diagnostics beyond the solver's three-point sample
    dy = np.diff(yp)
    ratio = np.maximum(dy[1:] / dy[:-1], dy[:-1] / dy[1:])
    print('\n  Whole-grid diagnostics (not part of the solver report):')
    print(f"    dy+ range over all {len(dy)} cells    : "
          f"{Re_tau * dy.min():.6f} to {Re_tau * dy.max():.6f}")
    print(f"    Maximum cell-to-cell growth rate  : {ratio.max():.6f} "
          f"(cell {int(np.argmax(ratio)) + 1})")
    if cfg.icase == ICASE_PIPE:
        print(f"    dy+ at the wall (j=np)            : {wall_dyplus(cfg, res):.6f}")
        print('    Note: for a pipe, j=1 is the axis (y=0), not a wall, so the '
              '"near wall" and')
        print('    "far wall" labels above follow the solver but only j=np is a '
              'true wall.')

    # MHD
    if cfg.is_mhd:
        print('\n' + '-' * 100)
        print('MHD Boundary Layer')
        print('-' * 100)
        if cfg.hartmann is None:
            print('  ! Hartmann number not found in the input file; '
                  'the MHD check cannot be performed.')
            print('    Note: the solver accepts either a Stuart or a Hartmann number.')
        else:
            ha_bl = 1.0 / cfg.hartmann
            ha_bl_plus = ha_bl * Re_tau
            n_pnts_ha = int(np.count_nonzero(yp <= (ha_bl - 1.0)))
            print(f"  MHD boundary layer thickness (d_Ha)  : {ha_bl:14.6f}")
            print(f"  MHD boundary layer thickness (d_Ha+) : {ha_bl_plus:14.6f}")
            print(f"  Grid points in MHD boundary layer    : {n_pnts_ha:14d}")
            if n_pnts_ha < 10:
                print('  ! Warning: insufficient grid points in the MHD boundary layer. '
                      'At least 10 are recommended.')

    # Summary
    print('\n' + '-' * 100)
    print('Mesh Resolution Summary')
    print('-' * 100)
    if cfg.is_thermo:
        if res['Pr'] is not None:
            print(f"  Thermal field solved: Pr = {res['Pr']:.6f} ({res['pr_source']}); "
                  "the wall-normal limit is min(1, 1/Pr)/Re_tau.")
        else:
            print(f"  ! Thermal field solved but Pr is unknown ({res['pr_source']}).")
            print("    Falling back to the isothermal limit dy+ <= 1; the thermal "
                  "requirement may be stricter.")

    print('\n  Current mesh:')
    print(f"    Cells (Ncx, Ncy, Ncz)         : {cfg.nc[0]}, {cfg.nc[1]}, {cfg.nc[2]}")
    print(f"    Total cells                   : {cfg.nc[0] * cfg.nc[1] * cfg.nc[2]:,}")

    print('\n  Recommended minimum mesh for DNS:')
    print(f"    Cells (Ncx, Ncy, Ncz)         : "
          f"{res['nx_min']}, {res['ny_min']}, {res['nz_min']}")
    print(f"    Total cells                   : "
          f"{res['nx_min'] * res['ny_min'] * res['nz_min']:,}")

    if (cfg.nc[0] >= res['nx_min'] and cfg.nc[1] >= res['ny_min']
            and cfg.nc[2] >= res['nz_min']):
        print('\n  + Current mesh meets the minimum DNS resolution requirements')
    else:
        print('\n  ! Warning: current mesh is below the recommended DNS resolution.')
        if cfg.nc[0] < res['nx_min']:
            print(f"    Increase Ncx: {cfg.nc[0]} -> {res['nx_min']}")
        if cfg.nc[1] < res['ny_min']:
            print(f"    Increase Ncy: {cfg.nc[1]} -> {res['ny_min']}")
        if cfg.nc[2] < res['nz_min']:
            print(f"    Increase Ncz: {cfg.nc[2]} -> {res['nz_min']}")


# ====================================================================================================================================================
# Temporal resolution assessment (estimate_temporal_resolution)
# ====================================================================================================================================================


def analyse_temporal_resolution(cfg, res, diff=None):
    """
    Estimate time-step limits and iteration counts for the current mesh.

    The solver calls this immediately after the spatial estimate during prerun,
    reusing Re_tau and the sampled wall-normal spacings.

    Parameters
    ----------
    cfg : DomainConfig
    res : dict
        Output of `analyse_spatial_resolution`.
    diff : dict, optional
        Output of `analyse_diffusion_number`. When given, the diffusion numbers
        are reported alongside the other time-step limits, which is where they
        belong: they bound dt for the same reason.
    """
    Re_tau = res['Re_tau']

    dt_max_cfl1 = CFL_MAX * cfg.hx / 2.0
    dxyz_max = (1.0 / cfg.hx**2 + 1.0 / res['dymin']**2
                + 1.0 / (res['rmin'] * cfg.hz)**2)
    dt_max_cfl2 = cfg.ren / 2.0 / dxyz_max
    dt_max_phy = CTM_MAX * (cfg.ren / Re_tau / Re_tau)
    dt_min = min(dt_max_cfl1, dt_max_cfl2, dt_max_phy)

    print('\n' + '-' * 100)
    print('Temporal Resolution (based on isothermal/ constant property flow)')
    print('-' * 100)
    print(f"  Current dt                      : {cfg.dt:14.6e}")
    print(f"  dt_max (convection CFL)         : {dt_max_cfl1:14.6e}")
    print(f"  dt_max (diffusion CFL)          : {dt_max_cfl2:14.6e}")
    print(f"  dt_max (Kolmogorov limit)       : {dt_max_phy:14.6e}")
    print(f"  dt_max (dt+ = 0.1)              : "
          f"{0.1 * cfg.ren / Re_tau / Re_tau:14.6e}")
    if cfg.dt > dt_min:
        print(f"\n  ! Warning: current dt exceeds the smallest limit "
              f"({dt_min:.6e}).")
    else:
        print('\n  + Current dt is within all isothermal estimated limits')

    if diff is not None:
        print_diffusion_block(cfg, diff)

    if cfg.dt <= 0.0:
        return

    t_flth = cfg.lxx / 1.2
    nt_cur = int(np.ceil(t_flth / cfg.dt))
    nt_est = int(np.ceil(t_flth / dt_min))

    print('\n' + '-' * 100)
    print('Required Time Steps')
    print('-' * 100)
    print(f"  Flow-through time                        : {t_flth:14.6f}")
    print(f"  1 flow-through at estimated dt_max       : {nt_est:14,d}  "
          f"(dt = {dt_min:.6e})")
    print(f"  1 flow-through at current dt             : {nt_cur:14,d}  "
          f"(dt = {cfg.dt:.6e})")
    print(f"  Recommended 25 flow-throughs for stats   : {nt_cur * 25:14,d}")
    if cfg.is_record_xoutlet or cfg.is_read_xinlet:
        print(f"  Recommended 5 flow-throughs for db record : {nt_cur * 5:14,d}")
    print('\n  Note: statistics can start from any iteration when using running-average')
    print('  post-processing. Otherwise, allow the following before starting statistics:')
    print(f"  Recommended 6 flow-throughs before stats : {nt_cur * 6:14,d}")


# ====================================================================================================================================================
# Diffusion number (Check_cfl_diffusion, src/tools_solver.f90)
# ====================================================================================================================================================


def _mapping_derivative(n, kind, cfg):
    """
    d(eta)/dy at the requested points, i.e. yMappingpt/yMappingcc column 1.

    Reproduces the `mp(:, 1)` expressions in the `Buildup_grid_mapping_1D_*`
    routines (src/geometry.f90).

    Parameters
    ----------
    n : int
        Number of points.
    kind : {'nd', 'cl'}
        Nodes or cell centres.
    cfg : DomainConfig

    Returns
    -------
    numpy.ndarray
        d(eta)/dy at each point.
    """
    if not cfg.is_stretching:
        return np.ones(n)

    eta = _eta(n, kind)
    span = cfg.lyt - cfg.lyb

    if cfg.mstret == MSTRET_3FMD:
        if cfg.istret == ISTRET_2SIDES:
            gamma, delta = 1.0, 0.5
        elif cfg.istret == ISTRET_BOTTOM:
            gamma, delta = 0.5, 0.5
        elif cfg.istret == ISTRET_TOP:
            gamma, delta = 0.5, 0.0
        else:
            gamma, delta = 1.0, 0.0
        beta = cfg.rstret
        alpha = (-1.0 + np.sqrt(1.0 + 4.0 * np.pi**2 * beta**2)) / beta * 0.5
        mm = np.pi * (gamma * eta + delta)
        return (alpha / np.pi + np.sin(mm)**2 / np.pi / beta) / span

    if cfg.mstret == MSTRET_TANH:
        beta = cfg.rstret * 20.0
        if cfg.istret == ISTRET_2SIDES:
            mm = np.tanh(beta * 0.5)
            mp = 2.0 * mm / beta * np.cosh(beta * (eta - 0.5))**2
            ff = span / 2.0
        else:
            mm = np.tanh(beta)
            arg = beta * (1.0 - eta) if cfg.istret == ISTRET_BOTTOM else beta * eta
            mp = mm / beta * np.cosh(arg)**2
            ff = span
        return mp / ff

    if cfg.mstret == MSTRET_POWL:
        beta = cfg.rstret
        expo = 1.0 / beta
        with np.errstate(divide='ignore', invalid='ignore'):
            if cfg.istret == ISTRET_2SIDES:
                mp = np.where(eta <= 0.5,
                              expo * (2.0 * eta)**(expo - 1.0),
                              expo * (2.0 * (1.0 - eta))**(expo - 1.0))
            elif cfg.istret == ISTRET_BOTTOM:
                mp = expo * eta**(expo - 1.0)
            else:
                mp = expo * (1.0 - eta)**(expo - 1.0)
            # The solver replaces a vanishing derivative with 1 before inverting.
            mp = np.where(np.abs(mp) < MINP, 1.0, 1.0 / mp)
        return mp / span

    raise ValueError(f"Unsupported stretching method: {cfg.mstret}")


def cell_inv_dy(cfg):
    """
    1/dy at each cell centre, as the solver forms it.

    `Check_cfl_diffusion` uses `yMappingcc(j, 1) / h(2)` on a stretched grid,
    where h(2) is the uniform computational spacing, and the plain 1/h(2) of
    the physical spacing otherwise.

    Parameters
    ----------
    cfg : DomainConfig

    Returns
    -------
    numpy.ndarray
        1/dy for each of the Ncy cells.
    """
    n_cell = cfg.nc[1]

    if not cfg.is_stretching:
        h2 = (cfg.lyt - cfg.lyb) / n_cell
        return np.full(n_cell, 1.0 / h2)

    h2 = 1.0 / n_cell   # computational spacing
    return _mapping_derivative(n_cell, 'cl', cfg) / h2


def analyse_diffusion_number(cfg, yc, verbose=False):
    """
    Diffusion (von Neumann) numbers and the time steps that would bound them.

    Port of `Check_cfl_diffusion` (src/tools_solver.f90), which the solver runs
    each step on the live field. Before a run the transport properties are not
    known, so the reference state is assumed throughout: the dynamic viscosity
    and thermal conductivity are 1 in the solver's non-dimensionalisation. For
    an isothermal case that is exactly what the solver computes; for a heated
    case the true numbers scale with the local property ratios.

    Parameters
    ----------
    cfg : DomainConfig
    yc : numpy.ndarray
        Cell-centre wall-normal coordinates.
    verbose : bool
        Print the report.

    Returns
    -------
    dict
        Diffusion numbers, limiting time steps and where the maximum occurs.
    """
    rsp1 = 1.0 / cfg.hx**2
    rsp2 = cell_inv_dy(cfg)**2

    rsp3 = np.full(cfg.nc[1], 1.0 / cfg.hz**2)
    if cfg.icoordinate == ICYLINDRICAL:
        # rci = 1/yc: the azimuthal arc length collapses towards the axis
        rsp3 = rsp3 / yc**2

    rdxyz2 = rsp1 + rsp2 + rsp3
    j_max = int(np.argmax(rdxyz2))
    rmax = float(rdxyz2[j_max])

    rre = 1.0 / cfg.ren
    diff_mom = rmax * 2.0 * cfg.dt * rre
    dt_max_mom = 1.0 / (2.0 * rre * rmax)

    Pr, pr_source = (reference_prandtl_number(cfg) if cfg.is_thermo else (None, ''))
    diff_ene = dt_max_ene = None
    if cfg.is_thermo and Pr is not None:
        r_pr_ren = rre / Pr
        diff_ene = rmax * 2.0 * cfg.dt * r_pr_ren
        dt_max_ene = 1.0 / (2.0 * r_pr_ren * rmax)

    res = {
        'rdxyz2_max': rmax, 'j_max': j_max,
        'contrib_x': rsp1, 'contrib_y': float(rsp2[j_max]), 'contrib_z': float(rsp3[j_max]),
        'diff_mom': diff_mom, 'dt_max_mom': dt_max_mom,
        'diff_ene': diff_ene, 'dt_max_ene': dt_max_ene,
        'Pr': Pr, 'pr_source': pr_source,
    }

    if verbose:
        print_diffusion_block(cfg, res)

    return res


def print_diffusion_block(cfg, res):
    """
    Print the diffusion numbers as part of the temporal resolution section.

    Parameters
    ----------
    cfg : DomainConfig
    res : dict
        Output of `analyse_diffusion_number`.
    """
    print('\n  Diffusion numbers (constant properties):')
    print(f"  max(1/dx^2+1/dy^2+1/dz^2)     : {res['rdxyz2_max']:14.6e}  "
          f"  (cell j = {res['j_max'] + 1} of {cfg.nc[1]})")
    print(f"  contributions x | y | z       : {res['contrib_x']:.4e} | "
          f"{res['contrib_y']:.4e} | {res['contrib_z']:.4e}")
    print(f"  Momentum diffusion number     : {res['diff_mom']:14.6e}")
    print(f"  dt_max (momentum diffusion)   : {res['dt_max_mom']:14.6e}")
    if res['diff_mom'] > 1.0:
        print('    ! Warning: momentum diffusion number is larger than 1. '
              'Numerical instability could occur.')
        print(f"      Reduce dt below {res['dt_max_mom']:.6e}, or coarsen the mesh.")

    if cfg.is_thermo:
        if res['diff_ene'] is not None:
            print(f"  Energy diffusion number       : {res['diff_ene']:14.6e}")
            print(f"  dt_max (energy diffusion)     : {res['dt_max_ene']:14.6e}  "
                  f"(Pr = {res['Pr']:.6g}, {res['pr_source']})")
            if res['diff_ene'] > 1.0:
                print('    ! Warning: energy diffusion number is larger than 1. '
                      'Numerical instability could occur.')
                print(f"      Reduce dt below {res['dt_max_ene']:.6e}, or coarsen the mesh.")
        else:
            print(f"    Energy diffusion number       : unavailable "
                  f"({res['pr_source']})")

    print('\n  Note: evaluated at the reference state, where the non-dimensional')
    print('  viscosity and conductivity are 1; the solver uses the live fields.')
    if cfg.icoordinate == ICYLINDRICAL:
        print('    In cylindrical coordinates the azimuthal term 1/(r*dtheta)^2 grows')
        print('    sharply towards the axis, so the innermost cells usually set this limit.')


# ====================================================================================================================================================
# Mesh distribution plot
# ====================================================================================================================================================


def draw_mesh_distribution(fig, cfg, yp, res):
    """
    Draw the wall-normal spacing and cell growth rate onto a figure.

    Kept separate from `plot_mesh_distribution` so the GUI can render into an
    embedded figure without going through pyplot.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to draw into. Cleared first.
    cfg : DomainConfig
    yp : numpy.ndarray
        Wall-normal node coordinates.
    res : dict
        Output of `analyse_spatial_resolution`.
    """
    Re_tau = res['Re_tau']
    dy = np.diff(yp)
    yc_plot = 0.5 * (yp[:-1] + yp[1:])
    ratio = np.maximum(dy[1:] / dy[:-1], dy[:-1] / dy[1:])

    fig.clear()
    axes = fig.subplots(2, 1)

    axes[0].plot(yc_plot, Re_tau * dy, 'o-', color='C0', ms=3, lw=1.0,
                 label=r'$\Delta y^{+}$')
    axes[0].axhline(DYPLUS_MAX, color='C3', ls='--', lw=1.2,
                    label=rf'DNS limit $\Delta y^{{+}} = {DYPLUS_MAX:.0f}$')
    axes[0].axhline(res['dxplus'], color='C2', ls=':', lw=1.2,
                    label=rf"$\Delta x^{{+}} = {res['dxplus']:.2f}$")
    axes[0].axhline(res['dzplus'], color='C4', ls='-.', lw=1.2,
                    label=rf"$\Delta z^{{+}} = {res['dzplus']:.2f}$")
    axes[0].set_yscale('log')
    axes[0].set_xlabel(r'$y$')
    axes[0].set_ylabel(r'$\Delta y^{+}$')
    axes[0].set_title(f"Wall-normal spacing in wall units "
                      f"($Re_\\tau = {Re_tau:.1f}$, $N_{{cy}} = {cfg.nc[1]}$)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(np.arange(1, len(ratio) + 1), ratio, 'o-', color='C1', ms=3, lw=1.0,
                 label='cell-to-cell growth rate')
    axes[1].axhline(1.2, color='C8', ls='--', lw=1.2, label='caution (1.2)')
    axes[1].axhline(1.3, color='C3', ls='--', lw=1.2, label='limit (1.3)')
    axes[1].set_xlabel('cell index $j$')
    axes[1].set_ylabel(r'$\max(\Delta y_{j+1}/\Delta y_{j},\ \Delta y_{j}/\Delta y_{j+1})$')
    axes[1].set_title('Grid stretching')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()


def plot_mesh_distribution(cfg, yp, res, save_path):
    """
    Save the mesh distribution plot to an image file.

    Parameters
    ----------
    cfg : DomainConfig
    yp : numpy.ndarray
        Wall-normal node coordinates.
    res : dict
        Output of `analyse_spatial_resolution`.
    save_path : str
        Output image path.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })

    fig = plt.figure(figsize=(9, 8))
    draw_mesh_distribution(fig, cfg, yp, res)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\nSaved mesh distribution plot to: {save_path}")


# ====================================================================================================================================================
# Main
# ====================================================================================================================================================


def main():
    """Run the mesh analysis on a CHAPSim2 input file."""
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        prompt = input("Path to input_chapsim.ini (blank for current directory): ").strip()
        path = prompt if prompt else 'input_chapsim.ini'

    if os.path.isdir(path):
        path = os.path.join(path, 'input_chapsim.ini')

    if not os.path.isfile(path):
        print(f"Error: input file not found: {path}")
        return 1

    print('\n' + '=' * 100)
    print(f'CHAPSim2 MESH ANALYSIS: {os.path.abspath(path)}')
    print('=' * 100)

    sections = parse_input_file(path)
    cfg = DomainConfig(sections, path)
    cfg.summary()

    if not cfg.is_wall_bounded:
        print(f"\nCase '{CASE_NAMES.get(cfg.icase, cfg.icase)}' is not wall bounded. "
              "The solver assesses channel, pipe and annular cases only.")
        return 0

    if cfg.nc[1] < 5:
        print(f"\nError: Ncy = {cfg.nc[1]} is too small to assess "
              "(at least 5 wall-normal cells are needed).")
        return 1

    if cfg.ren <= 0.0:
        print(f"\nError: Reynolds number (ren) = {cfg.ren} is not usable.")
        return 1

    yp, yc = build_y_grid(cfg)
    check_y_grid(yp, cfg)

    res = analyse_spatial_resolution(cfg, yp, yc)
    diff = analyse_diffusion_number(cfg, yc)
    analyse_temporal_resolution(cfg, res, diff)

    print('\n' + '=' * 100)

    answer = input("\nSave mesh distribution plot? (y/n): ").strip().lower()
    if answer.startswith('y'):
        save_path = os.path.join(os.path.dirname(os.path.abspath(path)),
                                 'mesh_analysis.png')
        plot_mesh_distribution(cfg, yp, res, save_path)

    print('\nMesh analysis complete.')
    print('=' * 100)

    return 0


if __name__ == '__main__':
    sys.exit(main())
