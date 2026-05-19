#!/usr/bin/env python3
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os

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


def apply_robust_ylim(ax, data):
    """
    Apply diverged y-limits to a matplotlib axis if divergence is detected.
    Adds a text annotation when limits are clipped.
    """
    limits = compute_robust_ylim(data)
    if limits is not None:
        ax.set_ylim(limits)
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
    props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
    ax.text(0.02, 0.05, stats_text, transform=ax.transAxes,
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


def load_monitor_data(file_path, skiprows, max_abs_value=MAX_ABS_VALUE, sample=1):
    """
    Load monitor data and drop diverged/invalid rows.
    Sample > 1 skips rows during parsing.

    """
    try:
        with open(file_path, 'r') as f:
            for _ in range(skiprows):
                f.readline()
            lines = f if sample <= 1 else (
                line for i, line in enumerate(f) if i % sample == 0
            )
            data = np.loadtxt(lines, dtype=np.float64)
    except Exception as e:
        print(f"Warning: Could not load {os.path.basename(file_path)}: {e}")
        return np.empty((0, 0))

    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.size == 0:
        return np.empty((0, 0))

    finite_mask = np.all(np.isfinite(data), axis=1)
    if data.shape[1] > 1:
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

    return data[keep_mask]


# ====================================================================================================================================================
# Input parameters
# ====================================================================================================================================================

# Parse command line arguments for data path
if len(sys.argv) > 1:
    path = sys.argv[1]
    # Ensure path ends with trailing slash
    if not path.endswith('/'):
        path += '/'
else:
    # If no argument provided, use current working directory
    path = os.getcwd() + '/'

print('='*100)
print(f'Plotting monitor points from: {path}')
print('='*100)

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

# Get configuration from user
print("\nConfiguration:")
print("-" * 100)

num_monitor_pts = get_int("Number of monitor points to plot", 5)
thermo_on = get_yes_no("Include temperature data?", 'y')
sample_factor = get_int("Sample factor (plot every nth point)", 10)
plt_pts = get_yes_no("Plot monitor points?", 'y')
plt_bulk = get_yes_no("Plot bulk/change history?", 'y')
display_plots = get_yes_no("Display plots interactively?", 'n')
auto_ylim = get_yes_no("Auto-limit y-axis range for diverged data?", 'y')
avg_window = get_int("Running average window size (1 = off)", 5000)

print("-" * 100)
print()

# Generate file lists based on number of monitor points
pt_files = [f'domain1_monitor_pt{i}_flow.dat' for i in range(1, num_monitor_pts + 1)]
blk_files = ['domain1_monitor_metrics_history.log', 'domain1_monitor_change_history.log']

# ====================================================================================================================================================
 
if plt_pts:
    for file in pt_files:
        data = load_monitor_data(path + file, skiprows=3, sample=sample_factor)

        if data.size == 0:
            print(f"Skipping {file}: no valid data after filtering.")
            continue

        print(f'Plotting {len(data)} points for {file}...')

        time = data[:,0]
        u = data[:,1]
        v = data[:,2]
        w = data[:,3]
        p = data[:,4]
        phi = data[:,5]
        if thermo_on:
            T = data[:,6]

        # Create subplots for all variables
        num_subplots = 6 if thermo_on else 5
        fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 3*num_subplots), sharex=True)

        # Subplot 0: u-velocity
        plot_with_avg(axes[0], time, u, 'u-velocity', 'C0', avg_window)
        axes[0].set_ylabel('u-velocity')
        axes[0].legend()
        axes[0].grid()
        if auto_ylim: apply_robust_ylim(axes[0], u)
        add_stats_box(axes[0], u)

        # Subplot 1: v-velocity
        plot_with_avg(axes[1], time, v, 'v-velocity', 'C1', avg_window)
        axes[1].set_ylabel('v-velocity')
        axes[1].legend()
        axes[1].grid()
        if auto_ylim: apply_robust_ylim(axes[1], v)
        add_stats_box(axes[1], v)

        # Subplot 2: w-velocity
        plot_with_avg(axes[2], time, w, 'w-velocity', 'C2', avg_window)
        axes[2].set_ylabel('w-velocity')
        axes[2].legend()
        axes[2].grid()
        if auto_ylim: apply_robust_ylim(axes[2], w)
        add_stats_box(axes[2], w)

        # Subplot 3: Pressure
        plot_with_avg(axes[3], time, p, 'pressure', 'C3', avg_window)
        axes[3].set_ylabel('Pressure')
        axes[3].legend()
        axes[3].grid()
        if auto_ylim: apply_robust_ylim(axes[3], p)
        add_stats_box(axes[3], p)

        # Subplot 4: Pressure Correction
        plot_with_avg(axes[4], time, phi, 'press. corr.', 'C4', avg_window)
        axes[4].set_ylabel('Pressure Correction')
        axes[4].legend()
        axes[4].grid()
        if auto_ylim: apply_robust_ylim(axes[4], phi)
        add_stats_box(axes[4], phi)

        if thermo_on:
            # Subplot 5: Temperature
            plot_with_avg(axes[5], time, T, 'temperature', 'C5', avg_window)
            axes[5].set_ylabel('Temperature')
            axes[5].legend()
            axes[5].grid()
            if auto_ylim: apply_robust_ylim(axes[5], T)
            add_stats_box(axes[5], T)
            axes[5].set_xlabel('Time')
        else:
            axes[4].set_xlabel('Time')

        fig.suptitle(f'{file} - Monitor Point Data', fontsize=14)
        fig.tight_layout()
        fig.savefig(f'{path}{file.replace("domain1_monitor_","").replace(".dat","_plot")}.png', dpi=300, bbox_inches='tight')
        if display_plots:
            plt.show()
        plt.close(fig)

        print(f'Saved subplot figure for {file}')

if plt_bulk:
    for file in blk_files:
        blk_data = load_monitor_data(path + file, skiprows=2, sample=sample_factor)

        if blk_data.size == 0:
            print(f"Skipping {file}: no valid data after filtering.")
            continue

        if file == 'domain1_monitor_metrics_history.log':
            time = blk_data[:,0]
            MKE = blk_data[:,1]
            qx = blk_data[:,2]
            if thermo_on:
                gx = blk_data[:,3]
                T = blk_data[:,4]
                h = blk_data[:,5]

            # Create subplots for bulk quantities
            num_subplots = 4 if thermo_on else 2
            fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 3*num_subplots), sharex=True)

            # Subplot 0: Mean Kinetic Energy
            plot_with_avg(axes[0], time, MKE, 'Mean Kinetic Energy', 'C0', avg_window)
            axes[0].set_ylabel('Mean Kinetic Energy')
            axes[0].legend()
            axes[0].grid()
            if auto_ylim: apply_robust_ylim(axes[0], MKE)
            add_stats_box(axes[0], MKE)

            # Subplot 1: Bulk Velocity and Density * Bulk Velocity (on same plot)
            plot_with_avg(axes[1], time, qx, 'Bulk Velocity', 'C1', avg_window)
            if thermo_on:
                plot_with_avg(axes[1], time, gx, 'Density * Bulk Velocity', 'C2', avg_window)
            axes[1].set_ylabel('Velocity')
            axes[1].legend()
            axes[1].grid()
            if auto_ylim:
                combined = np.concatenate([qx, gx]) if thermo_on else qx
                apply_robust_ylim(axes[1], combined)
            add_stats_box(axes[1], qx)

            if thermo_on:
                # Subplot 2: Bulk Temperature
                plot_with_avg(axes[2], time, T, 'Bulk Temperature', 'C3', avg_window)
                axes[2].set_ylabel('Bulk Temperature')
                axes[2].legend()
                axes[2].grid()
                if auto_ylim: apply_robust_ylim(axes[2], T)
                add_stats_box(axes[2], T)

                # Subplot 3: Bulk Enthalpy
                plot_with_avg(axes[3], time, h, 'Bulk Enthalpy', 'C4', avg_window)
                axes[3].set_ylabel('Bulk Enthalpy')
                axes[3].legend()
                axes[3].grid()
                if auto_ylim: apply_robust_ylim(axes[3], h)
                add_stats_box(axes[3], h)
                axes[3].set_xlabel('Time')
            else:
                axes[1].set_xlabel('Time')

            fig.suptitle('Bulk Quantities', fontsize=14)
            fig.tight_layout()
            fig.savefig(f'{path}{file.replace("domain1_monitor_","").replace(".log","_plot")}.png', dpi=300, bbox_inches='tight')
            if display_plots:
                plt.show()
            plt.close(fig)
            print(f'Saved metrics history plot for {file}')
        
        if file == 'domain1_monitor_change_history.log':
            time = blk_data[:,0]
            mass_cons = blk_data[:,1]
            mass_chng_rt = blk_data[:,4]
            KE_chng_rt = blk_data[:,5]

            # Create subplots for change history
            fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

            # Subplot 0: Mass Conservation
            plot_with_avg(axes[0], time, mass_cons, 'Mass Conservation', 'C0', avg_window)
            axes[0].set_ylabel('Mass Conservation')
            axes[0].legend()
            axes[0].grid()
            if auto_ylim: apply_robust_ylim(axes[0], mass_cons)
            add_stats_box(axes[0], mass_cons)

            # Subplot 1: Mass Change Rate
            plot_with_avg(axes[1], time, mass_chng_rt, 'Mass Change Rate', 'C1', avg_window)
            axes[1].set_ylabel('Mass Change Rate')
            axes[1].legend()
            axes[1].grid()
            if auto_ylim: apply_robust_ylim(axes[1], mass_chng_rt)
            add_stats_box(axes[1], mass_chng_rt)

            # Subplot 2: Kinetic Energy Change Rate
            plot_with_avg(axes[2], time, KE_chng_rt, 'Kinetic Energy Change Rate', 'C2', avg_window)
            axes[2].set_ylabel('KE Change Rate')
            axes[2].legend()
            axes[2].grid()
            if auto_ylim: apply_robust_ylim(axes[2], KE_chng_rt)
            add_stats_box(axes[2], KE_chng_rt)
            axes[2].set_xlabel('Time')

            fig.suptitle('Change History', fontsize=14)
            fig.tight_layout()
            fig.savefig(f'{path}{file.replace("domain1_monitor_","").replace(".log","_plot")}.png', dpi=300, bbox_inches='tight')
            if display_plots:
                plt.show()
            plt.close(fig)
            print(f'Saved change history plot for {file}')

print('='*100)
print(f'All plots saved to: {path}')
print('='*100)
