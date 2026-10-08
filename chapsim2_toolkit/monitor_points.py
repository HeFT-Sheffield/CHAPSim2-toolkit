#!/usr/bin/env python3
import numpy as np

from chapsim2_toolkit import provenance as prov
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os
import re

mpl.rcParams.update({
"font.family": "serif",
"font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
"mathtext.fontset": "cm",
"axes.unicode_minus": False,
})

plt.rcParams['agg.path.chunksize'] = 10000 # Configure matplotlib for better performance with large datasets
plt.rcParams['path.simplify_threshold'] = 1.0

MAX_ABS_VALUE = 1e5

# ====================================================================================================================================================
# Robust y-limit calculation (IQR-based) for diverged simulations
# ====================================================================================================================================================

def compute_robust_ylim(data, padding=0.05, max_decades=3.0):
    """
    Compute robust y-axis limits by masking diverged data points.
    
    Uses the Median Absolute Deviation to identify the scale of the
    'physical' data.  Points whose distance from the median exceeds
    10^max_decades times the MAD are considered diverged and excluded.
    The y-limits are then set to the full range of the remaining data.
    
    Parameters
    ----------
    data : array-like
        The data array to compute limits for.
    padding : float
        Fractional padding added to the computed range for visual comfort.
    max_decades : float
        Number of orders of magnitude above the MAD to allow before
        treating a point as diverged (default 3.0, i.e. 1000 × MAD).
    
    Returns
    -------
    (ymin, ymax) or None if no clipping is needed.
    """
    finite_data = data[np.isfinite(data)]
    if len(finite_data) == 0:
        return None
    
    median = np.median(finite_data)
    mad = np.median(np.abs(finite_data - median))
    
    # Handle near-constant data where MAD ≈ 0
    if mad < 1e-15:
        # Use a fraction of the median as the scale, with a floor of 1.0
        # so that truly constant data at zero still works.
        mad = max(abs(median), 1.0) * 0.01
    
    # Threshold: points farther than 10^max_decades * MAD from the median
    # are considered diverged.  With max_decades=3 this means >1000× the
    # typical deviation — extremely permissive for physical data.
    threshold = mad * 10**max_decades
    mask = np.abs(finite_data - median) <= threshold
    clean_data = finite_data[mask]
    
    if len(clean_data) == 0:
        return None
    
    # If nothing was removed, data is well-behaved — let matplotlib auto-scale
    if len(clean_data) == len(finite_data):
        return None
    
    # Set limits to the full range of the non-diverged data
    ymin = np.min(clean_data)
    ymax = np.max(clean_data)
    
    span = ymax - ymin
    ymin -= padding * span
    ymax += padding * span
    
    return (ymin, ymax)


def apply_robust_ylim(ax, *series):
    """
    Apply diverged y-limits to a matplotlib axis if divergence is detected.
    Adds a text annotation when limits are clipped.

    Each series is judged on its own and the surviving ranges are unioned.
    Pooling them first would compare quantities of different magnitude:
    on a bulk-velocity panel, qx ~ 1 against qy = 0 and qz ~ 1e-6 makes qx
    look like the outlier, and the axis would be clipped to +/-1e-6 with the
    one curve anybody wanted to see left off the plot.
    """
    lo = hi = None
    clipped = False
    for data in series:
        data = np.asarray(data)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            continue
        limits = compute_robust_ylim(data)
        if limits is None:
            limits = (float(np.min(finite)), float(np.max(finite)))
        else:
            clipped = True
        lo = limits[0] if lo is None else min(lo, limits[0])
        hi = limits[1] if hi is None else max(hi, limits[1])

    if not clipped or lo is None or not np.isfinite([lo, hi]).all():
        return
    if hi <= lo:
        hi = lo + 1.0
    ax.set_ylim(lo, hi)
    ax.annotate('⚠ y-axis clipped (divergence detected)',
                 xy=(0.5, 1.0), xycoords='axes fraction',
                 ha='center', va='bottom', fontsize=8,
                 color='red', fontstyle='italic')


def add_stats_box(ax, data):
    """
    Add a small statistics text box to a subplot.
    Shows mean, std, min, max and median of the plotted data.
    """
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        return
    stats_text = (
        f"mean: {np.mean(finite):.4g}\n"
        f"std:  {np.std(finite):.4g}\n"
        f"min:  {np.min(finite):.4g}\n"
        f"max:  {np.max(finite):.4g}\n"
        f"med:  {np.median(finite):.4g}"
    )
    # Anchored outside the axes: inside, it lands on the legend or the data
    # on any panel whose curve runs low-left, which most monitor traces do.
    props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85)
    ax.text(1.01, 0.0, stats_text, transform=ax.transAxes,
            fontsize=7, verticalalignment='bottom', horizontalalignment='left',
            bbox=props, family='monospace')


def running_average(data, window):
    """
    Compute a centred running average using a cumulative-sum approach: O(n)
    instead of O(n × window) for convolution. Edges are padded with the
    boundary value so the output length equals the input length.
    """
    if window <= 1:
        return data.copy()
    n = len(data)
    pad_l = window // 2
    pad_r = window - 1 - pad_l
    padded = np.pad(data.astype(float), (pad_l, pad_r), mode='edge')
    cumsum = np.empty(len(padded) + 1, dtype=float)
    cumsum[0] = 0.0
    np.cumsum(padded, out=cumsum[1:])
    return (cumsum[window:window + n] - cumsum[:n]) / window


def plot_with_avg(ax, time, data, label, color, window):
    """
    Plot raw data and, if a running average window is set, overlay the
    running average.
    """
    ax.plot(time, data, label=label, linewidth=0.8, color=color, rasterized=True)
    if window > 1:
        avg = running_average(data, window)
        ax.plot(time, avg, label=f'{label} (avg, n={window})',
                linewidth=1.2, color='black', linestyle='--', alpha=0.7,
                rasterized=True)


# ====================================================================================================================================================
# Monitor file headers
# ====================================================================================================================================================
#
# CHAPSim2 monitor files are self-describing, each in its own way:
#
#   domain1_monitor_pt<N>_flow.dat    # iteration, t, u, v, w, p, phi, T
#   domain1_monitor_metrics_history.log
#                                     # column  1 : time
#                                     # column  2 : global mass balance ...
#   domain1_monitor_change_history.log
#                                     # columns: time; physical mass residual
#                                     #          at bulk, inlet, outlet; ...
#
# The first two name their columns machine-readably, so they are parsed.  The
# change history writes prose that does not map one-to-one onto columns, so it
# uses the layout below — guarded by a column count check, falling back to
# positional names if a future release changes it.

CHANGE_HISTORY_COLUMNS = [
    'time',
    'mass residual (bulk)', 'mass residual (inlet)', 'mass residual (outlet)',
    'projected mass residual (bulk)', 'projected mass residual (inlet)',
    'projected mass residual (outlet)',
    'global mass flux imbalance',
    'Poisson compatibility defect',
    'uniform Poisson-source correction',
    'Poisson projected-source amplitude',
    'Poisson zero-mode projection',
    'total mass', 'total mass drift', 'kinetic energy change rate',
]

COLUMN_RE = re.compile(r'^column\s+(\d+)\s*:\s*(.+)$')


def read_monitor_header(file_path):
    """Read the column names out of a monitor file's comment header.

    Returns:
        list of column names, or None when the header does not name them.
    """
    comments = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                if not stripped.startswith('#'):
                    break
                comments.append(stripped.lstrip('#').strip())
    except OSError:
        return None

    # 'column  N : name' records, one per line (metrics history).
    numbered = {}
    for entry in comments:
        match = COLUMN_RE.match(entry)
        if match:
            numbered[int(match.group(1))] = match.group(2).strip()
    if numbered:
        return [numbered[i] for i in sorted(numbered)]

    # A single comma-separated list on the last comment line (point monitors).
    if comments and ',' in comments[-1]:
        names = [n.strip() for n in comments[-1].split(',') if n.strip()]
        if len(names) > 1:
            return names

    return None


def load_monitor_data(file_path, max_abs_value=MAX_ABS_VALUE, sample=1):
    """
    Load a monitor file and drop diverged/invalid rows.

    All leading comment lines are skipped, and columns are named from the
    file's own header where it provides them, so a monitor file that gains a
    column (CHAPSim2 has done this more than once) still plots the right
    quantity under the right label.  Sample > 1 skips rows during parsing.

    Returns:
        tuple ``(data, columns)``: a 2-D array and the matching column names.
        ``(empty array, [])`` when nothing could be read.
    """
    columns = read_monitor_header(file_path)

    try:
        with open(file_path, 'r') as f:
            rows = (line for line in f if not line.lstrip().startswith('#'))
            lines = rows if sample <= 1 else (
                line for i, line in enumerate(rows) if i % sample == 0
            )
            data = np.loadtxt(lines, dtype=np.float64, ndmin=2)
    except Exception as e:
        print(f"Warning: Could not load {os.path.basename(file_path)}: {e}")
        return np.empty((0, 0)), []

    if data.size == 0:
        return np.empty((0, 0)), []

    ncol = data.shape[1]
    if columns is None:
        name = os.path.basename(file_path)
        if 'change_history' in name and ncol == len(CHANGE_HISTORY_COLUMNS):
            columns = list(CHANGE_HISTORY_COLUMNS)
        else:
            columns = ['time'] + [f'column {i + 1}' for i in range(1, ncol)]
    elif len(columns) != ncol:
        print(f"Note: {os.path.basename(file_path)} header names {len(columns)} columns "
              f"but the table has {ncol}; using the last {min(len(columns), ncol)}.")
        if len(columns) > ncol:
            # An older run may lack trailing columns (e.g. no thermo fields).
            columns = columns[:ncol]
        else:
            columns = ([f'column {i + 1}' for i in range(ncol - len(columns))] + columns)

    finite_mask = np.all(np.isfinite(data), axis=1)
    if ncol > 1:
        within_limit = np.all(np.abs(data[:, 1:]) <= max_abs_value, axis=1)
    else:
        within_limit = np.ones(data.shape[0], dtype=bool)

    keep_mask = finite_mask & within_limit
    skipped = np.count_nonzero(~keep_mask)
    if skipped > 0:
        print(
            f"Skipped {skipped} diverged/invalid rows in {os.path.basename(file_path)} "
            f"(non-time |value| > {max_abs_value:.0e} or non-finite)."
        )

    return data[keep_mask], columns


def column_lookup(columns):
    """Map column names to their index, under both exact and folded spellings.

    Case matters here: a point monitor writes both 't' (time) and 'T'
    (temperature), so folding case alone would collapse them onto one entry
    and hand back whichever came last. Exact spellings therefore win, with
    case-folded names kept as a fallback for the history files, whose
    headers are prose.
    """
    lookup = {}
    for i, name in enumerate(columns):
        lookup.setdefault(name.strip().lower(), i)   # first wins when folded
    for i, name in enumerate(columns):
        lookup[name.strip()] = i                     # exact spelling overrides
    return lookup


def pick(data, lookup, *candidates):
    """Return the first column matching any candidate name, else None.

    Matched exactly first, then case-insensitively, then as a substring, so
    'bulk velocity qx' is found by 'qx' while 't' still means time and not
    temperature.
    """
    for candidate in candidates:
        key = candidate.strip()
        if key in lookup:
            return data[:, lookup[key]]
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return data[:, lookup[key]]
    for candidate in candidates:
        key = candidate.strip().lower()
        # Only distinctive names are matched loosely: a bare 'u' or 'p' would
        # otherwise hit 'bulk velocity qx' or 'total mass drift'.
        if len(key) < 3:
            continue
        for name, index in lookup.items():
            if key in name:
                return data[:, index]
    return None


def plot_panels(panels, time, title, out_path, display, auto_ylim, avg_window):
    """Render one figure with a stacked panel per entry.

    ``panels`` is a list of ``(ylabel, [(label, series), ...])``; entries whose
    series are all missing are dropped, so a case without thermo or without an
    inlet simply gets a shorter figure.
    """
    panels = [(ylabel, [(lab, s) for lab, s in series if s is not None])
              for ylabel, series in panels]
    panels = [p for p in panels if p[1]]
    if not panels:
        print(f"  No plottable columns for {title}")
        return

    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 3 * len(panels)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    for ax, (ylabel, series) in zip(axes, panels):
        for i, (label, values) in enumerate(series):
            plot_with_avg(ax, time, values, label, f'C{i}', avg_window)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid()
        if auto_ylim:
            apply_robust_ylim(ax, *[v for _, v in series])
        add_stats_box(ax, series[0][1])

    axes[-1].set_xlabel('Time')
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight',
                metadata=prov.figure_metadata(out_path.rsplit('.', 1)[-1]))
    if display:
        plt.show()
    plt.close(fig)
    print(f'Saved {os.path.basename(out_path)}')


# ====================================================================================================================================================
# Input parameters
# ====================================================================================================================================================

def monitor_dir(path):
    """Resolve a user-supplied path to the directory holding monitor files.

    Accepts the monitor directory itself or the case directory above it.
    """
    path = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
    nested = os.path.join(path, '3_monitor')
    if os.path.basename(path) != '3_monitor' and os.path.isdir(nested):
        return nested
    return path


# Interactive configuration prompts
def get_yes_no(prompt, default='y'):
    """Get yes/no input from user."""
    response = input(f"{prompt} [{'Y/n' if default == 'y' else 'y/N'}]: ").strip().lower()
    if response == '':
        return default == 'y'
    return response in ['y', 'yes']

def get_int(prompt, default):
    """Get integer input from user."""
    response = input(f"{prompt} [{default}]: ").strip()
    if response == '':
        return default
    try:
        return int(response)
    except ValueError:
        print(f"Invalid input, using default: {default}")
        return default


def main():
    """Run the interactive monitor-plot session."""
    path = monitor_dir(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())

    print('='*100)
    print(f'Plotting monitor points from: {path}')
    print('='*100)

    # Get configuration from user
    print("\nConfiguration:")
    print("-" * 100)

    discovered_pts = sorted(
        int(m.group(1))
        for m in (re.match(r'domain\d+_monitor_pt(\d+)_flow\.dat$', f)
                  for f in (os.listdir(path) if os.path.isdir(path) else []))
        if m
    )
    if discovered_pts:
        print(f"Found monitor points: {', '.join(str(p) for p in discovered_pts)}")

    num_monitor_pts = get_int("Number of monitor points to plot",
                              max(discovered_pts) if discovered_pts else 5)
    sample_factor = get_int("Sample factor (plot every nth point)", 10)
    plt_pts = get_yes_no("Plot monitor points?", 'y')
    plt_bulk = get_yes_no("Plot bulk/change history?", 'y')
    display_plots = get_yes_no("Display plots interactively?", 'n')
    auto_ylim = get_yes_no("Auto-limit y-axis range for diverged data?", 'y')
    avg_window = get_int("Running average window size (1 = off)", 0)

    print("-" * 100)
    print()

    pt_files = [f'domain1_monitor_pt{i}_flow.dat' for i in range(1, num_monitor_pts + 1)]
    blk_files = ['domain1_monitor_metrics_history.log', 'domain1_monitor_change_history.log']

    # ====================================================================================================================================================

    if plt_pts:
        for file in pt_files:
            file_path = os.path.join(path, file)
            if not os.path.isfile(file_path):
                continue

            data, columns = load_monitor_data(file_path, sample=sample_factor)
            if data.size == 0:
                print(f"Skipping {file}: no valid data after filtering.")
                continue

            print(f'Plotting {len(data)} points for {file}...')
            lookup = column_lookup(columns)
            time = pick(data, lookup, 't', 'time')
            if time is None:
                time = data[:, 1] if data.shape[1] > 1 else data[:, 0]

            # The header names time 't' and temperature 'T', which differ only by
            # case, so temperature is matched case-sensitively.
            temp_index = next((i for i, n in enumerate(columns) if n.strip() == 'T'), None)
            panels = [
                ('u-velocity', [('u-velocity', pick(data, lookup, 'u'))]),
                ('v-velocity', [('v-velocity', pick(data, lookup, 'v'))]),
                ('w-velocity', [('w-velocity', pick(data, lookup, 'w'))]),
                ('Pressure', [('pressure', pick(data, lookup, 'p'))]),
                ('Pressure Correction', [('press. corr.', pick(data, lookup, 'phi'))]),
                ('Temperature', [('temperature',
                                  data[:, temp_index] if temp_index is not None else None)]),
            ]

            out = os.path.join(path, file.replace('domain1_monitor_', '').replace('.dat', '_plot.png'))
            plot_panels(panels, time, f'{file} - Monitor Point Data', out,
                        display_plots, auto_ylim, avg_window)

    if plt_bulk:
        for file in blk_files:
            file_path = os.path.join(path, file)
            if not os.path.isfile(file_path):
                continue

            data, columns = load_monitor_data(file_path, sample=sample_factor)
            if data.size == 0:
                print(f"Skipping {file}: no valid data after filtering.")
                continue

            lookup = column_lookup(columns)
            time = pick(data, lookup, 'time')
            if time is None:
                time = data[:, 0]

            if 'metrics_history' in file:
                panels = [
                    ('Mass conservation', [
                        ('global balance', pick(data, lookup, 'global mass balance')),
                        ('interior', pick(data, lookup, 'max. mass conservation (interior)')),
                        ('inlet', pick(data, lookup, 'max. mass conservation (inlet)')),
                        ('outlet', pick(data, lookup, 'max. mass conservation (outlet)')),
                    ]),
                    ('Kinetic energy', [
                        ('total kinetic energy', pick(data, lookup, 'total kinetic energy')),
                    ]),
                    ('Pressure', [
                        ('mean dpdx', pick(data, lookup, 'mean dpdx')),
                        ('global pressure drop', pick(data, lookup, 'global pressure drop')),
                    ]),
                    ('Bulk velocity', [
                        ('qx', pick(data, lookup, 'bulk velocity qx')),
                        ('qy', pick(data, lookup, 'bulk velocity qy')),
                        ('qz', pick(data, lookup, 'bulk velocity qz')),
                    ]),
                    ('Bulk mass flux', [
                        ('gx', pick(data, lookup, 'bulk mass flux gx')),
                        ('gy', pick(data, lookup, 'bulk mass flux gy')),
                        ('gz', pick(data, lookup, 'bulk mass flux gz')),
                    ]),
                    ('Bulk enthalpy', [
                        ('bulk enthalpy', pick(data, lookup, 'bulk enthalpy')),
                    ]),
                    ('Bulk temperature', [
                        ('bulk temperature', pick(data, lookup, 'bulk temperature')),
                    ]),
                ]
                title = 'Bulk Quantities'
            else:
                panels = [
                    ('Mass residual', [
                        ('bulk', pick(data, lookup, 'mass residual (bulk)')),
                        ('inlet', pick(data, lookup, 'mass residual (inlet)')),
                        ('outlet', pick(data, lookup, 'mass residual (outlet)')),
                    ]),
                    ('Mass flux imbalance', [
                        ('global', pick(data, lookup, 'global mass flux imbalance')),
                    ]),
                    ('Poisson diagnostics', [
                        ('compatibility defect', pick(data, lookup, 'Poisson compatibility defect')),
                        ('zero-mode projection', pick(data, lookup, 'Poisson zero-mode projection')),
                    ]),
                    ('Total mass', [
                        ('total mass', pick(data, lookup, 'total mass')),
                    ]),
                    ('Mass drift', [
                        ('drift from run start', pick(data, lookup, 'total mass drift')),
                    ]),
                    ('KE change rate', [
                        ('kinetic energy change rate', pick(data, lookup, 'kinetic energy change rate')),
                    ]),
                ]
                title = 'Change History'

            out = os.path.join(path, file.replace('domain1_monitor_', '').replace('.log', '_plot.png'))
            plot_panels(panels, time, title, out, display_plots, auto_ylim, avg_window)

    print('='*100)
    print(f'All plots saved to: {path}')
    print('='*100)



if __name__ == '__main__':
    main()
