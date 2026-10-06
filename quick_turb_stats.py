#!/usr/bin/env python3
"""
Quick Turbulence Statistics Plotter
Simplified script for plotting turbulence statistics from a single XDMF case.
Requires only numpy and matplotlib.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import math
import glob
from tqdm import tqdm

mpl.rcParams.update({
"font.family": "serif",
"font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
"mathtext.fontset": "cm",
"axes.unicode_minus": False,
})

# Import the shared XDMF reader from utils
import utils as ut

# Enable tab completion for path input
try:
    import readline
    def path_completer(text, state):
        # Expand ~ and environment variables
        expanded = os.path.expanduser(os.path.expandvars(text))
        # Add wildcard for glob matching
        if os.path.isdir(expanded):
            pattern = os.path.join(expanded, '*')
        else:
            pattern = expanded + '*'
        matches = glob.glob(pattern)
        # Add trailing slash for directories
        matches = [m + '/' if os.path.isdir(m) else m for m in matches]
        try:
            return matches[state]
        except IndexError:
            return None
    readline.set_completer(path_completer)
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind('tab: complete')
except ImportError:
    pass  # readline not available on all platforms


def load_xdmf_data(visu_folder, timestep, slice_label=None):
    """
    Load the averaged statistics a case offers for one timestep.

    Sources are tried most-reduced first, since each needs less work
    downstream than the last:

      1. ASCII profile bundles (``2_visu/data/domain1_tsp_avg_flow_yprofile_
         60.dat``) — already 1-D, written when two directions are periodic.
      2. tsp_avg plane XDMF (``domain1_tsp_avg_flow_zi1_60.xdmf``) — 2-D,
         averaged over the one periodic direction. The plane tag names that
         direction, so it is discovered rather than assumed.
      3. t_avg XDMF (``domain1_t_avg_flow_60.xdmf``) — full 3-D, averaged in
         time only, so it still needs spatial averaging here.

    Args:
        visu_folder: Path to the case's xdmf folder (or the case folder)
        timestep: Timestep string
        slice_label: Optional 2D slice label (e.g., 'yi8'). When set, loads
                     pre-sliced 2D data instead of full 3D fields.

    Returns:
        tuple ``(data, grid_info)``.
    """
    dirs = ut.resolve_case_dirs(visu_folder)
    xdmf_dir = dirs['xdmf']

    # 1. ASCII profile bundles — only meaningful for whole-domain statistics.
    if not slice_label:
        variables, coords = ut.load_tsp_avg_profiles(dirs['case'], timestep,
                                                     strip_prefix=False)
        if variables:
            print(f"Found time-and-space averaged profile table(s) — "
                  f"{len(variables)} variables, already 1D")
            direction = coords.get('direction', 'y')
            grid_info = {}
            coord = coords.get(f'{direction}c')
            if coord is not None:
                # Also expose it as grid_y: downstream this script treats the
                # profile axis as wall-normal whichever direction it runs along.
                grid_info[f'grid_{direction}'] = coord
                grid_info['grid_y'] = coord
            return variables, grid_info

    # 2/3. XDMF: build the candidate names for each tier.
    def names(prefix):
        if slice_label:
            return [f'domain1_{prefix}_{g}_{slice_label}_{timestep}.xdmf'
                    for g in ut.VISU_GROUPS]
        found = []
        for g in ut.VISU_GROUPS:
            found.append(f'domain1_{prefix}_{g}_{timestep}.xdmf')
            if prefix == 'tsp_avg':
                # Plane output, tagged with the averaged-out direction.
                found.extend(f'domain1_tsp_avg_{g}_{a}i1_{timestep}.xdmf' for a in 'xyz')
        return found

    tiers = [
        ('tsp_avg', "time-and-space averaged (tsp_avg) planes — no further spatial averaging needed"),
        ('t_avg', "time averaged (t_avg) fields — will spatially average"),
    ]

    existing_files = []
    for prefix, message in tiers:
        existing_files = [(n, os.path.join(xdmf_dir, n)) for n in names(prefix)
                          if os.path.isfile(os.path.join(xdmf_dir, n))]
        if existing_files:
            print(f"Using {message}")
            break
    else:
        print("Warning: No averaged data files found (tsp_avg or t_avg)")

    all_data = {}
    grid_info = {}

    with tqdm(total=len(existing_files), desc="Loading XDMF files", unit="file") as pbar:
        for pattern, filepath in existing_files:
            pbar.set_postfix_str(pattern[:30])

            # Use the shared XDMF reader from utils
            arrays, file_grid_info = ut.parse_xdmf_file(filepath, load_all_vars=False)

            # Store grid info from first file that has it
            if not grid_info and file_grid_info:
                grid_info = file_grid_info

            all_data.update(arrays)
            pbar.update(1)

    return all_data, grid_info


def extract_y_profile(data):
    """
    Extract 1D y-profile by averaging over x and z directions.
    Handles 3D (nz, ny, nx), 2D (ny, nx), and 1D (ny,) input.
    """
    if data.ndim == 3:
        return data.mean(axis=(0, 2))
    if data.ndim == 2:
        return data.mean(axis=1)
    return data


def compute_reynolds_stresses(data, grid_info, x_crop=None):
    """
    Compute Reynolds stresses from time-space averaged data.
    Handles prefixes (tsp_avg_, t_avg_)
    """
    results = {}

    # Map variable names - try different conventions and prefixes
    x_coords = grid_info.get('grid_x', None) if grid_info else None

    def crop_x_if_needed(values):
        if x_crop is None or values is None or values.ndim < 2 or x_coords is None:
            return values
        if values.shape[-1] not in (len(x_coords), len(x_coords) - 1):
            return values
        cropped, _ = ut.apply_x_crop(values, x_coords, x_crop)
        return cropped

    def find_var(base_names):
        # Try with prefixes first (tsp_avg_, t_avg_), then without
        prefixes = ['tsp_avg_', 't_avg_', '']
        for prefix in prefixes:
            for name in base_names:
                full_name = f'{prefix}{name}'
                if full_name in data:
                    return extract_y_profile(crop_x_if_needed(data[full_name]))
        return None

    # Velocity components
    u1 = find_var(['u1', 'ux', 'qx_ccc'])
    u2 = find_var(['u2', 'uy', 'qy_ccc'])
    u3 = find_var(['u3', 'uz', 'qz_ccc'])

    # Reynolds stress correlations
    uu11 = find_var(['uu11', 'uu', 'uxux'])
    uu12 = find_var(['uu12', 'uv', 'uxuy'])
    uu22 = find_var(['uu22', 'vv', 'uyuy'])
    uu33 = find_var(['uu33', 'ww', 'uzuz'])

    # Store velocity profile
    if u1 is not None:
        results['ux_velocity'] = u1

    # Compute Reynolds stresses: <u'u'> = <uu> - <u><u>
    if u1 is not None and uu11 is not None:
        results['u_prime_sq'] = uu11 - np.square(u1)

    if u1 is not None and u2 is not None and uu12 is not None:
        results['u_prime_v_prime'] = uu12 - u1 * u2

    if u2 is not None and uu22 is not None:
        results['v_prime_sq'] = uu22 - np.square(u2)

    if u3 is not None and uu33 is not None:
        results['w_prime_sq'] = uu33 - np.square(u3)

    # Compute TKE if all components available
    if all(k in results for k in ['u_prime_sq', 'v_prime_sq', 'w_prime_sq']):
        results['TKE'] = 0.5 * (results['u_prime_sq'] + results['v_prime_sq'] + results['w_prime_sq'])

    # Temperature if available
    T = find_var(['T', 'temperature', 'temp', 'TT'])
    if T is not None:
        results['temperature'] = T

    return results


def compute_normalisation(ux_profile, y_coords, Re_ref):
    """
    Compute wall shear stress and friction velocity for normalisation.
    """
    # Wall gradient (assuming wall at index 0)
    du = ux_profile[0] - ux_profile[1]

    if y_coords is not None and len(y_coords) > 1:
        dy = abs(y_coords[0] - y_coords[1])
    else:
        dy = 2.0 / len(ux_profile)  # Fallback assuming domain height of 2

    dudy = du / dy
    # Re_ref is the number the solver non-dimensionalised with (ren in the
    # input file), not the bulk Reynolds number - the viscous term carries
    # 1/ren, so that is what sets the wall stress. The two coincide for a
    # constant-mass-flux run and do not for a constant pressure gradient.
    tau_w = dudy / Re_ref
    u_tau = np.sqrt(abs(tau_w))
    u_tau_sq = abs(tau_w)

    Re_tau = u_tau * Re_ref

    return u_tau, u_tau_sq, Re_tau


def normalise_results(results, y_coords, Re_ref, ref_temp=None):
    """
    Normalise all results by friction velocity.
    """
    if 'ux_velocity' not in results:
        print("Warning: Cannot normalise without ux_velocity data")
        return results, None, None

    ux = results['ux_velocity']
    u_tau, u_tau_sq, Re_tau = compute_normalisation(ux, y_coords, Re_ref)

    print(f"u_tau = {u_tau:.6f}, u_tau^2 = {u_tau_sq:.6f}, Re_tau = {Re_tau:.2f}")

    normalised = {}

    with tqdm(total=len(results), desc="Normalising", unit="var") as pbar:
        for name, values in results.items():
            pbar.set_postfix_str(name)
            if name == 'ux_velocity':
                normalised[name] = values / u_tau
            elif name == 'temperature':
                if ref_temp is not None:
                    normalised[name] = values * ref_temp
                else:
                    normalised[name] = values
            else:
                normalised[name] = values / u_tau_sq
            pbar.update(1)

    return normalised, u_tau, Re_tau


def get_y_coordinates(grid_info, n_points):
    """
    Get y coordinates from grid info or generate default.
    """
    if 'grid_y' in grid_info:
        y = grid_info['grid_y']
        # Cell centers if node coordinates
        if len(y) == n_points + 1:
            y = 0.5 * (y[:-1] + y[1:])
        return y[:n_points] if len(y) >= n_points else y
    else:
        # Default: assume channel with y in [-1, 1]
        return np.linspace(-1, 1, n_points)


def plot_results(results, y, half_channel=False, save_path=None):
    """
    Create plots for profiles and Reynolds stresses.
    """
    # Separate into profiles and Reynolds stresses
    profiles = {}
    re_stresses = {}

    profile_keys = ['ux_velocity', 'temperature', 'TKE']
    re_stress_keys = ['u_prime_sq', 'u_prime_v_prime', 'v_prime_sq', 'w_prime_sq']

    for key in profile_keys:
        if key in results:
            profiles[key] = results[key]

    for key in re_stress_keys:
        if key in results:
            re_stresses[key] = results[key]

    # Plot profiles
    if profiles:
        n_profiles = len(profiles)
        ncols = min(n_profiles, 3)
        nrows = math.ceil(n_profiles / ncols)

        fig1, axs1 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                                   constrained_layout=True)
        fig1.canvas.manager.set_window_title('Profiles')

        if n_profiles == 1:
            axs1 = np.array([[axs1]])
        elif nrows == 1 or ncols == 1:
            axs1 = axs1.reshape(nrows, ncols)

        labels = {
            'ux_velocity': ('$U^+$', 'Streamwise Velocity'),
            'temperature': ('$T$', 'Temperature'),
            'TKE': ('$k^+$', 'Turbulent Kinetic Energy'),
        }

        for i, (key, values) in enumerate(profiles.items()):
            row, col = i // ncols, i % ncols
            ax = axs1[row, col]

            plot_y = y[:len(values)]
            if half_channel:
                plot_y = plot_y + 1  # Shift to [0, 2] range

            ax.plot(plot_y, values, 'b-', linewidth=1.5)
            ylabel, title = labels.get(key, (key, key))
            ax.set_xlabel('$y$')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_profiles, nrows * ncols):
            row, col = i // ncols, i % ncols
            axs1[row, col].set_visible(False)

        if save_path:
            fig1.savefig(os.path.join(save_path, 'profiles.png'), dpi=300, bbox_inches='tight')
            print(f"Saved profiles.png")

    # Plot Reynolds stresses
    if re_stresses:
        n_stresses = len(re_stresses)
        ncols = min(n_stresses, 2)
        nrows = math.ceil(n_stresses / ncols)

        fig2, axs2 = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                                   constrained_layout=True)
        fig2.canvas.manager.set_window_title('Reynolds Stresses')

        if n_stresses == 1:
            axs2 = np.array([[axs2]])
        elif nrows == 1 or ncols == 1:
            axs2 = axs2.reshape(nrows, ncols)

        labels = {
            'u_prime_sq': ("$\\langle u'u' \\rangle^+$", "$\\langle u'u' \\rangle$"),
            'u_prime_v_prime': ("$\\langle u'v' \\rangle^+$", "$\\langle u'v' \\rangle$"),
            'v_prime_sq': ("$\\langle v'v' \\rangle^+$", "$\\langle v'v' \\rangle$"),
            'w_prime_sq': ("$\\langle w'w' \\rangle^+$", "$\\langle w'w' \\rangle$"),
        }

        colors = {'u_prime_sq': 'r', 'u_prime_v_prime': 'g', 'v_prime_sq': 'm', 'w_prime_sq': 'c'}

        for i, (key, values) in enumerate(re_stresses.items()):
            row, col = i // ncols, i % ncols
            ax = axs2[row, col]

            plot_y = y[:len(values)]
            if half_channel:
                plot_y = plot_y + 1

            ax.plot(plot_y, values, color=colors.get(key, 'b'), linewidth=1.5)
            ylabel, title = labels.get(key, (key, key))
            ax.set_xlabel('$y$')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_stresses, nrows * ncols):
            row, col = i // ncols, i % ncols
            axs2[row, col].set_visible(False)

        if save_path:
            fig2.savefig(os.path.join(save_path, 'reynolds_stresses.png'), dpi=300, bbox_inches='tight')
            print(f"Saved reynolds_stresses.png")

    return fig1 if profiles else None, fig2 if re_stresses else None


def get_user_input():
    """
    Interactively get configuration from user.
    """
    print("=" * 60)
    print("Quick Turbulence Statistics Plotter")
    print("=" * 60)

    # Get visu folder path (supports tab completion, ~, and $ENV_VARS)
    print("\nTip: Use Tab for path completion, leave empty for current directory")
    case_folder = input("Path to case folder: ").strip()
    if not case_folder:
        case_folder = os.getcwd()
    case_folder = os.path.expanduser(os.path.expandvars(case_folder))

    # Accept the case folder, 2_visu, or 2_visu/xdmf — they all name one case.
    dirs = ut.resolve_case_dirs(case_folder)
    case_folder = dirs['case']
    visu_folder = dirs['xdmf']

    if not os.path.isdir(visu_folder):
        print(f"Error: No CHAPSim2 visualisation output under {dirs['visu']}")
        return None

    timesteps = ut.find_available_timesteps(visu_folder)
    if timesteps:
        print(f"\nAvailable timesteps: {timesteps}")

    timestep = input("Timestep: ").strip()

    # Detect available 2D slice files
    available_slices = ut.find_available_slices(visu_folder, timestep)

    slice_label = None
    if available_slices:
        print(f"\nAvailable 2D slices: {', '.join(available_slices)}")
        slice_input = input("Use a 2D slice? Enter label or leave blank for full 3D []: ").strip()
        if slice_input and slice_input in available_slices:
            slice_label = slice_input
            print(f"  -> Using 2D slice: {slice_label}")
        elif slice_input:
            # Try to match partial input
            matches = [s for s in available_slices if slice_input in s]
            if matches:
                slice_label = matches[0]
                print(f"  -> Using 2D slice: {slice_label}")
            else:
                print(f"  -> Slice '{slice_input}' not found, using full 3D data")

    # Thermo options
    thermo_input = input("Thermo enabled? (y/n) [n]: ").strip().lower()
    thermo_on = thermo_input == 'y'

    ref_temp = None
    if thermo_on:
        ref_temp = ut.ask_number('Reference temperature (K)', 300.0)

    # MHD options
    mhd_input = input("MHD enabled? (y/n) [n]: ").strip().lower()
    mhd_on = mhd_input == 'y'

    # Flow parameters
    forcing = input("Forcing type (CMF/CPG) [CMF]: ").strip().upper()
    if forcing not in ['CMF', 'CPG']:
        forcing = 'CMF'

    Re = ut.ask_number('Reynolds number (bulk)', 5000.0)

    # Half channel plot
    half_input = input("Plot half channel? (y/n) [n]: ").strip().lower()
    half_channel = half_input == 'y'

    x_crop = None
    x_crop_input = input("Optional x-crop range x_min,x_max [none]: ").strip()
    if x_crop_input:
        try:
            x_crop = ut.parse_x_crop_input(x_crop_input)
        except ValueError:
            print("Invalid x-crop format. Using full x-range.")

    # Save options
    save_input = input("Save figures? (y/n) [y]: ").strip().lower()
    save_fig = save_input != 'n'

    display_input = input("Display figures? (y/n) [n]: ").strip().lower()
    display_fig = display_input != 'n'

    return {
        'case_folder': case_folder,
        'visu_folder': visu_folder,
        'timestep': timestep,
        'thermo_on': thermo_on,
        'mhd_on': mhd_on,
        'forcing': forcing,
        'Re': Re,
        'ref_temp': ref_temp,
        'half_channel': half_channel,
        'x_crop': x_crop,
        'save_fig': save_fig,
        'display_fig': display_fig,
        'slice_label': slice_label,
    }


def main():
    """Main execution function."""
    # Get user input
    config = get_user_input()
    if config is None:
        return

    print("\n" + "=" * 60)
    print("Loading data...")
    print("(Loading only required variables for Reynolds stress computation)")
    print("=" * 60)

    # Check if tsp_avg files exist to determine if we need sampling
    visu_folder = config['visu_folder']
    timestep = config['timestep']

    # Load XDMF data
    data, grid_info = load_xdmf_data(visu_folder, timestep, slice_label=config.get('slice_label'))

    if not data:
        print("Error: No data loaded. Check file paths.")
        print("\nAvailable files in folder:")
        for f in os.listdir(config['visu_folder']):
            print(f"  {f}")
        return

    print(f"\nLoaded {len(data)} variables: {list(data.keys())}")

    if grid_info:
        print(f"Grid info: {grid_info.get('cell_dimensions', 'unknown')}")

    # Compute statistics
    print("\n" + "=" * 60)
    print("Computing statistics...")
    print("=" * 60)

    results = compute_reynolds_stresses(data, grid_info, x_crop=config.get('x_crop'))

    if not results:
        print("Error: Could not compute any statistics from loaded data.")
        print("Available variables:", list(data.keys()))
        return

    print(f"Computed: {list(results.keys())}")

    # Get y coordinates
    n_points = len(next(iter(results.values())))
    y = get_y_coordinates(grid_info, n_points)

    # Normalise
    print("\nNormalising...")
    results, u_tau, Re_tau = normalise_results(results, y, config['Re'], config['ref_temp'])

    # Handle half channel
    if config['half_channel']:
        half_n = n_points // 2
        y = y[:half_n]
        for key in results:
            vals = results[key]
            if key == 'u_prime_v_prime' or key == 'temperature':
                results[key] = vals[:half_n]
            else:
                # Symmetric average for other quantities
                results[key] = (vals[:half_n] + vals[::-1][:half_n]) / 2

    # Plot
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)

    save_path = config['case_folder'] if config['save_fig'] else None
    plot_results(results, y, config['half_channel'], save_path)

    if config['display_fig']:
        plt.show()

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
