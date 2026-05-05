#!/usr/bin/env python3
"""
2D Slice Visualiser for CFD Data
Simple script for generating 2D slices from XDMF datasets.
Requires only numpy, matplotlib, and tqdm.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
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
import operations as op

# Enable tab completion for path input
try:
    import readline
    def path_completer(text, state):
        expanded = os.path.expanduser(os.path.expandvars(text))
        if os.path.isdir(expanded):
            pattern = os.path.join(expanded, '*')
        else:
            pattern = expanded + '*'
        matches = glob.glob(pattern)
        matches = [m + '/' if os.path.isdir(m) else m for m in matches]
        try:
            return matches[state]
        except IndexError:
            return None
    readline.set_completer(path_completer)
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind('tab: complete')
except ImportError:
    pass


def get_available_timesteps(visu_folder):
    """Extract available timesteps from XDMF filenames."""
    xdmf_files = [f for f in os.listdir(visu_folder) if f.endswith('.xdmf')]
    timesteps = set()
    for f in xdmf_files:
        parts = f.replace('.xdmf', '').split('_')
        if parts:
            timesteps.add(parts[-1])
    return sorted(timesteps)


def get_available_variables(data):
    """List all available variables in the loaded data."""
    return sorted(data.keys())


def extract_slice(data_3d, plane, index, grid_info):
    """
    Extract a 2D slice from 3D data.

    Args:
        data_3d: 3D numpy array with shape (nz, ny, nx)
        plane: 'xy', 'xz', or 'yz'
        index: slice index in the third dimension
        grid_info: dictionary containing grid_x, grid_y, grid_z

    Returns:
        slice_data: 2D numpy array
        coord1: coordinates for first axis
        coord2: coordinates for second axis
        axis_labels: tuple of (xlabel, ylabel)
    """
    nz, ny, nx = data_3d.shape

    if plane == 'xy':
        # Slice at constant z
        if index >= nz:
            index = nz - 1
        slice_data = data_3d[index, :, :]
        coord1 = grid_info.get('grid_x', np.arange(nx))
        coord2 = grid_info.get('grid_y', np.arange(ny))
        axis_labels = ('$x$', '$y$')

    elif plane == 'xz':
        # Slice at constant y
        if index >= ny:
            index = ny - 1
        slice_data = data_3d[:, index, :]
        coord1 = grid_info.get('grid_x', np.arange(nx))
        coord2 = grid_info.get('grid_z', np.arange(nz))
        axis_labels = ('$x$', '$z$')

    elif plane == 'yz':
        # Slice at constant x
        if index >= nx:
            index = nx - 1
        slice_data = data_3d[:, :, index]
        coord1 = grid_info.get('grid_y', np.arange(ny))
        coord2 = grid_info.get('grid_z', np.arange(nz))
        axis_labels = ('$y$', '$z$')

    else:
        raise ValueError(f"Invalid plane '{plane}'. Use 'xy', 'xz', or 'yz'.")

    return slice_data, coord1, coord2, axis_labels


def infer_data_location(data_shape, grid_info):
    """Infer whether array is point-centred or cell-centred from shape."""
    if not isinstance(data_shape, tuple):
        return 'unknown'

    node_dims = grid_info.get('node_dimensions') if grid_info else None
    cell_dims = grid_info.get('cell_dimensions') if grid_info else None

    if node_dims and data_shape == tuple(node_dims):
        return 'point'
    if cell_dims and data_shape == tuple(cell_dims):
        return 'cell'

    # 2D slices can match one of the 3D plane projections.
    if len(data_shape) == 2:
        if node_dims:
            node_2d_shapes = {
                (node_dims[1], node_dims[2]),  # xy
                (node_dims[0], node_dims[2]),  # xz
                (node_dims[0], node_dims[1]),  # yz
            }
            if data_shape in node_2d_shapes:
                return 'point'
        if cell_dims:
            cell_2d_shapes = {
                (cell_dims[1], cell_dims[2]),  # xy
                (cell_dims[0], cell_dims[2]),  # xz
                (cell_dims[0], cell_dims[1]),  # yz
            }
            if data_shape in cell_2d_shapes:
                return 'cell'

    return 'unknown'


def process_data_arrays(data, selected_vars, grid_info, interpolate_cell_to_point=False):
    """Apply optional preprocessing steps to loaded arrays."""
    processed = {}
    interpolated_vars = set()

    for var in selected_vars:
        arr = data[var]
        if interpolate_cell_to_point and arr.ndim in (2, 3):
            location = infer_data_location(arr.shape, grid_info)
            if location == 'cell':
                arr = op.interpolate_cell_to_point_data(arr)
                interpolated_vars.add(var)
                print(f"  Interpolated {var}: cell -> point, new shape {arr.shape}")
            elif location == 'unknown':
                print(
                    f"  Warning: Could not determine data location for {var}. "
                    f"node_dims={grid_info.get('node_dimensions')}, "
                    f"cell_dims={grid_info.get('cell_dimensions')}"
                )
        processed[var] = arr

    return processed, interpolated_vars


COLORMAP_OPTIONS = [
    'viridis',
    'plasma',
    'inferno',
    'magma',
    'cividis',
    'RdBu_r',
    'coolwarm',
    'jet',
]


def choose_colormap(default='RdBu_r'):
    """Prompt for a colormap using a numbered menu."""
    print("\nColormaps:")
    for i, name in enumerate(COLORMAP_OPTIONS, start=1):
        print(f"  {i}. {name}")

    cmap_choice = input(f"Colormap [{default}]: ").strip()
    if not cmap_choice:
        return default

    if cmap_choice.isdigit():
        idx = int(cmap_choice) - 1
        if 0 <= idx < len(COLORMAP_OPTIONS):
            return COLORMAP_OPTIONS[idx]

    if cmap_choice in COLORMAP_OPTIONS:
        return cmap_choice

    print(f"  Warning: invalid colormap selection, using {default}")
    return default


def build_save_filename(variable_name, slice_token, timestep, interpolated=False, combined=False):
    """Build a timestep-aware output filename for a slice plot."""
    base_name = 'combined' if combined else variable_name
    if interpolated and not combined and base_name.endswith('_ccc'):
        base_name = base_name[:-4]

    parts = [base_name]
    if slice_token:
        parts.append(str(slice_token))
    if timestep:
        parts.append(str(timestep))
    parts.append('slice')
    return '_'.join(parts) + '.png'


def build_save_path(save_dir, variable_name, slice_token, timestep, interpolated=False, combined=False):
    """Build an absolute save path for a plot, or None if saving is disabled."""
    if not save_dir:
        return None
    filename = build_save_filename(
        variable_name,
        slice_token,
        timestep,
        interpolated=interpolated,
        combined=combined,
    )
    return os.path.join(save_dir, filename)


def get_slice_location(grid_info, plane, index):
    """Get the physical location of the slice."""
    if plane == 'xy':
        coord_key = 'grid_z'
    elif plane == 'xz':
        coord_key = 'grid_y'
    elif plane == 'yz':
        coord_key = 'grid_x'
    else:
        return None

    if coord_key in grid_info:
        coords = grid_info[coord_key]
        if index < len(coords):
            return coords[index]
    return index


def plot_slice(slice_data, coord1, coord2, axis_labels, variable_name,
               cmap='RdBu_r', vmin=None, vmax=None, symmetric=False,
               slice_info="", save_path=None, display=False,
               smooth_point_data=False, center_zero=False):
    """
    Plot a 2D slice with colorbar.

    Args:
        slice_data: 2D numpy array
        coord1, coord2: coordinate arrays for axes
        axis_labels: tuple of (xlabel, ylabel)
        variable_name: name of the variable being plotted
        cmap: colormap name
        vmin, vmax: color scale limits
        symmetric: if True, use symmetric color scale around zero
        slice_info: string describing slice location
        save_path: path to save figure (None to skip saving)
        display: whether to display the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    # Handle color scale
    data_min = np.nanmin(slice_data)
    data_max = np.nanmax(slice_data)
    plot_cmap = cmap
    norm = None

    if symmetric:
        max_abs = max(abs(data_min), abs(data_max))
        vmin = -max_abs
        vmax = max_abs
        if cmap == 'viridis':
            plot_cmap = 'RdBu_r'

    if vmin is None:
        vmin = data_min
    if vmax is None:
        vmax = data_max

    # Keep colorbar centered at zero while preserving asymmetric data limits.
    if center_zero and not symmetric and (vmin < 0 < vmax):
        norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        if cmap == 'viridis':
            plot_cmap = 'RdBu_r'

    color_kwargs = {'cmap': plot_cmap}
    if norm is not None:
        color_kwargs['norm'] = norm
    else:
        color_kwargs['vmin'] = vmin
        color_kwargs['vmax'] = vmax

    use_gouraud = (
        smooth_point_data
        and len(coord1) == slice_data.shape[1]
        and len(coord2) == slice_data.shape[0]
    )

    if use_gouraud:
        x_points, y_points = np.meshgrid(coord1, coord2)
        pcm = ax.pcolormesh(
            x_points,
            y_points,
            slice_data,
            shading='gouraud',
            **color_kwargs
        )
    else:
        if len(coord1) == slice_data.shape[1]:
            dx = np.diff(coord1)
            x_edges = np.concatenate([
                [coord1[0] - dx[0] / 2],
                coord1[:-1] + dx / 2,
                [coord1[-1] + dx[-1] / 2]
            ])
        else:
            x_edges = coord1

        if len(coord2) == slice_data.shape[0]:
            dy = np.diff(coord2)
            y_edges = np.concatenate([
                [coord2[0] - dy[0] / 2],
                coord2[:-1] + dy / 2,
                [coord2[-1] + dy[-1] / 2]
            ])
        else:
            y_edges = coord2

        pcm = ax.pcolormesh(
            x_edges,
            y_edges,
            slice_data,
            shading='flat',
            **color_kwargs
        )

    cbar = fig.colorbar(pcm, ax=ax, label=variable_name)

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(f'{variable_name} {slice_info}')
    ax.set_aspect('equal', adjustable='box')

    # Print statistics
    print(f"\nSlice statistics for {variable_name}:")
    print(f"  Min: {np.nanmin(slice_data):.6e}")
    print(f"  Max: {np.nanmax(slice_data):.6e}")
    print(f"  Mean: {np.nanmean(slice_data):.6e}")

    if save_path:
        fig.savefig(save_path, dpi=1000, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if display:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_combined_slices(slices_data, coord1, coord2, axis_labels, slice_info,
                         cmap='RdBu_r', symmetric=False, shared_scale=False,
                         save_path=None, display=False, point_data_vars=None,
                         center_zero=False):
    """
    Plot multiple 2D slices in a single figure with subplots.

    Args:
        slices_data: list of (variable_name, slice_data) tuples
        coord1, coord2: coordinate arrays for axes
        axis_labels: tuple of (xlabel, ylabel)
        slice_info: string describing slice location
        cmap: colormap name
        symmetric: if True, use symmetric color scale around zero
        shared_scale: if True, use same color scale across all subplots
        save_path: path to save figure (None to skip saving)
        display: whether to display the figure
    """
    import math

    n_vars = len(slices_data)
    nrows = min(n_vars, 3)
    ncols = math.ceil(n_vars / nrows)

    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                            constrained_layout=True)

    # Handle single subplot case
    if n_vars == 1:
        axs = np.array([[axs]])
    elif nrows == 1:
        axs = axs.reshape(1, -1)
    elif ncols == 1:
        axs = axs.reshape(-1, 1)

    # Compute shared scale if requested
    plot_cmap = cmap
    global_norm = None
    if shared_scale:
        all_data = np.concatenate([s[1].flatten() for s in slices_data])
        if symmetric:
            global_max = max(abs(np.nanmin(all_data)), abs(np.nanmax(all_data)))
            global_vmin, global_vmax = -global_max, global_max
            if cmap == 'viridis':
                plot_cmap = 'RdBu_r'
        else:
            global_vmin = np.nanmin(all_data)
            global_vmax = np.nanmax(all_data)
            if center_zero and (global_vmin < 0 < global_vmax):
                global_norm = mpl.colors.TwoSlopeNorm(vmin=global_vmin, vcenter=0.0, vmax=global_vmax)
                if cmap == 'viridis':
                    plot_cmap = 'RdBu_r'

    for i, (var_name, slice_data) in enumerate(slices_data):
        row, col = i // ncols, i % ncols
        ax = axs[row, col]
        local_cmap = plot_cmap
        norm = None

        # Handle color scale
        if shared_scale:
            vmin, vmax = global_vmin, global_vmax
            norm = global_norm
        elif symmetric:
            max_abs = max(abs(np.nanmin(slice_data)), abs(np.nanmax(slice_data)))
            vmin, vmax = -max_abs, max_abs
            if cmap == 'viridis':
                local_cmap = 'RdBu_r'
        else:
            vmin = np.nanmin(slice_data)
            vmax = np.nanmax(slice_data)
            if center_zero and (vmin < 0 < vmax):
                norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
                if cmap == 'viridis':
                    local_cmap = 'RdBu_r'

        color_kwargs = {'cmap': local_cmap}
        if norm is not None:
            color_kwargs['norm'] = norm
        else:
            color_kwargs['vmin'] = vmin
            color_kwargs['vmax'] = vmax

        use_gouraud = (
            point_data_vars is not None
            and var_name in point_data_vars
            and len(coord1) == slice_data.shape[1]
            and len(coord2) == slice_data.shape[0]
        )

        if use_gouraud:
            x_points, y_points = np.meshgrid(coord1, coord2)
            pcm = ax.pcolormesh(
                x_points,
                y_points,
                slice_data,
                shading='gouraud',
                **color_kwargs
            )
        else:
            if len(coord1) == slice_data.shape[1]:
                dx = np.diff(coord1)
                x_edges = np.concatenate([
                    [coord1[0] - dx[0] / 2],
                    coord1[:-1] + dx / 2,
                    [coord1[-1] + dx[-1] / 2]
                ])
            else:
                x_edges = coord1

            if len(coord2) == slice_data.shape[0]:
                dy = np.diff(coord2)
                y_edges = np.concatenate([
                    [coord2[0] - dy[0] / 2],
                    coord2[:-1] + dy / 2,
                    [coord2[-1] + dy[-1] / 2]
                ])
            else:
                y_edges = coord2

            pcm = ax.pcolormesh(
                x_edges,
                y_edges,
                slice_data,
                shading='flat',
                **color_kwargs
            )

        fig.colorbar(pcm, ax=ax, label=var_name)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_title(f'{var_name}')
        ax.set_aspect('equal', adjustable='box')

    # Hide unused subplots
    for i in range(n_vars, nrows * ncols):
        row, col = i // ncols, i % ncols
        axs[row, col].set_visible(False)

    fig.suptitle(f'2D Slices {slice_info}', fontsize=12)

    if save_path:
        fig.savefig(save_path, dpi=1000, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if display:
        plt.show()
    else:
        plt.close(fig)

    return fig


def get_user_input():
    """Interactively get configuration from user."""
    print("=" * 60)
    print("2D Slice Visualiser")
    print("=" * 60)

    # Get case folder path
    print("\nTip: Use Tab for path completion, leave empty for current directory")
    case_folder = input("Path to case folder: ").strip()
    if not case_folder:
        case_folder = os.getcwd()
    case_folder = os.path.expanduser(os.path.expandvars(case_folder))

    # Handle case where user navigated to 2_visu folder
    if os.path.basename(case_folder) == '2_visu':
        visu_folder = case_folder
        case_folder = os.path.dirname(case_folder)
    else:
        visu_folder = os.path.join(case_folder, '2_visu')

    if not os.path.isdir(visu_folder):
        print(f"Error: Directory not found: {visu_folder}")
        return None

    # List available timesteps
    timesteps = get_available_timesteps(visu_folder)
    if timesteps:
        print(f"\nAvailable timesteps: {timesteps}")

    timestep = input("Timestep: ").strip()

    # Detect available 2D slice files
    available_slices = ut.find_available_slices(visu_folder, timestep)

    # Select data type
    print("\nData types:")
    print("  1. inst     - Instantaneous 3D data")
    print("  2. t_avg    - Time-averaged 3D data")
    if available_slices:
        print(f"  3. 2d_slice - Pre-sliced 2D data (available: {', '.join(available_slices)})")
    data_type_choice = input("Data type [1]: ").strip()

    data_type_map = {'1': 'inst', '2': 't_avg',
                     'inst': 'inst', 't_avg': 't_avg',
                     '': 'inst'}

    slice_label = None
    is_2d_slice = False

    if data_type_choice == '3' or data_type_choice == '2d_slice':
        if not available_slices:
            print("No 2D slice files found for this timestep.")
            return None
        is_2d_slice = True

        # Let user pick a slice label
        print("\nAvailable 2D slices:")
        for i, label in enumerate(available_slices):
            info = ut.slice_axis_info(label)
            plane_desc = f"{info['plane']} plane" if info else label
            print(f"  {i+1}. {label:6s} - {plane_desc} (constant {info['normal_dir']} at index {info['normal_index']})")
        slice_choice = input(f"Select slice [1]: ").strip()
        try:
            idx = int(slice_choice) - 1 if slice_choice else 0
            slice_label = available_slices[max(0, min(idx, len(available_slices) - 1))]
        except ValueError:
            # Try matching label directly
            if slice_choice in available_slices:
                slice_label = slice_choice
            else:
                slice_label = available_slices[0]
        print(f"  -> Selected slice: {slice_label}")

        # Select sub-type for 2D slices
        print("\n2D slice data types:")
        print("  1. inst   - Instantaneous")
        print("  2. t_avg  - Time-averaged")
        sub_type_choice = input("Sub-type [1]: ").strip()
        sub_type_map = {'1': 'inst', '2': 't_avg', '': 'inst',
                        'inst': 'inst', 't_avg': 't_avg'}
        data_type = sub_type_map.get(sub_type_choice, 'inst')
    else:
        data_type = data_type_map.get(data_type_choice, 'inst')

    # Select physics type
    print("\nPhysics types:")
    print("  1. flow   - Flow variables (velocity, pressure, etc.)")
    print("  2. thermo - Thermal variables (temperature, etc.)")
    print("  3. mhd    - MHD variables (magnetic field, etc.)")
    physics_choice = input("Physics type [1]: ").strip()

    physics_map = {'1': 'flow', '2': 'thermo', '3': 'mhd',
                   'flow': 'flow', 'thermo': 'thermo', 'mhd': 'mhd',
                   '': 'flow'}
    physics_type = physics_map.get(physics_choice, 'flow')

    # Build filename based on data type
    if is_2d_slice:
        if data_type == 'inst':
            xdmf_filename = f"domain1_{physics_type}_{slice_label}_{timestep}.xdmf"
        else:
            xdmf_filename = f"domain1_{data_type}_{physics_type}_{slice_label}_{timestep}.xdmf"
    else:
        if data_type == 'inst':
            xdmf_filename = f"domain1_{physics_type}_{timestep}.xdmf"
        else:
            xdmf_filename = f"domain1_{data_type}_{physics_type}_{timestep}.xdmf"

    xdmf_path = os.path.join(visu_folder, xdmf_filename)
    if not os.path.isfile(xdmf_path):
        print(f"Error: File not found: {xdmf_filename}")
        print("\nAvailable files:")
        for f in sorted(os.listdir(visu_folder)):
            if f.endswith('.xdmf'):
                print(f"  {f}")
        return None

    return {
        'case_folder': case_folder,
        'visu_folder': visu_folder,
        'timestep': timestep,
        'data_type': data_type,
        'physics_type': physics_type,
        'xdmf_path': xdmf_path,
        'is_2d_slice': is_2d_slice,
        'slice_label': slice_label,
    }


def parse_variable_selection(var_choice, variables):
    """Parse variable selection string and return list of selected variables."""
    selected = []

    # Handle 'all' keyword
    if var_choice.lower() == 'all':
        return variables[:]

    # Split by comma or space
    parts = var_choice.replace(',', ' ').split()

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Handle range (e.g., 1-5)
        if '-' in part and part[0] != '-':
            try:
                start, end = part.split('-')
                start_idx = int(start) - 1
                end_idx = int(end) - 1
                for i in range(start_idx, end_idx + 1):
                    if 0 <= i < len(variables) and variables[i] not in selected:
                        selected.append(variables[i])
            except ValueError:
                # Not a valid range, try as variable name
                if part in variables and part not in selected:
                    selected.append(part)
        elif part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(variables) and variables[idx] not in selected:
                selected.append(variables[idx])
        elif part in variables and part not in selected:
            selected.append(part)

    return selected


def get_2d_plot_config(var_metadata, grid_info, slice_label):
    """Get plot configuration for already-2D slice data."""

    # List available variables (2D, or 3D with a singleton dimension)
    all_variables = sorted(var_metadata.keys())
    variables = [v for v in all_variables
                 if len(var_metadata[v].get('shape', ())) == 2
                 or (len(var_metadata[v].get('shape', ())) == 3
                     and min(var_metadata[v]['shape']) == 1)]

    if not variables:
        print("Error: No 2D variables found in dataset.")
        return None

    print(f"\nAvailable 2D variables ({len(variables)}):")
    for i, var in enumerate(variables):
        shape = var_metadata[var]['shape']
        print(f"  {i+1:2d}. {var:20s} shape: {shape}")

    print("\nSelection options:")
    print("  - Single e.g: 1 or variable_name")
    print("  - Multiple e.g: 1,3,5 or 1 3 5")
    print("  - Range e.g: 1-5")
    print("  - All: all")

    var_choice = input("\nVariables to plot: ").strip()
    selected_vars = parse_variable_selection(var_choice, variables)

    if not selected_vars:
        print("No valid variables selected, using first variable")
        selected_vars = [variables[0]]

    print(f"\nSelected {len(selected_vars)} variable(s): {selected_vars}")

    # Get slice axis info
    axis_info = ut.slice_axis_info(slice_label)

    # X-direction cropping (optional, for xz and xy planes)
    x_crop = None
    if axis_info and axis_info['plane'] in ['xz', 'xy']:
        grid_x = grid_info.get('grid_x', None)
        if grid_x is not None and len(grid_x) > 1:
            print(f"\nX-direction cropping (optional):")
            print(f"  Full x range: {grid_x.min():.4f} to {grid_x.max():.4f}")
            crop_input = input("Crop x range? (y/n) [n]: ").strip().lower()
            if crop_input == 'y':
                x_min_input = input(f"  x_min [{grid_x.min():.4f}]: ").strip()
                x_max_input = input(f"  x_max [{grid_x.max():.4f}]: ").strip()
                x_min = float(x_min_input) if x_min_input else grid_x.min()
                x_max = float(x_max_input) if x_max_input else grid_x.max()
                x_crop = (x_min, x_max)
                print(f"  -> Cropping x from {x_min:.4f} to {x_max:.4f}")

    interp_input = input("\nInterpolate cell data to point data? (y/n) [n]: ").strip().lower()
    interpolate_cell_to_point = interp_input == 'y'

    # Colormap selection
    cmap = choose_colormap('RdBu_r')

    # Color scale options
    print("\nColor scale options:")
    print("  1. Auto (min to max)")
    print("  2. Symmetric around zero")
    print("  3. Custom range")
    print("  4. Centre at zero (crop to data min/max)")
    scale_choice = input("Scale option [1]: ").strip()

    symmetric = False
    center_zero = False
    vmin, vmax = None, None

    if scale_choice == '2':
        symmetric = True
    elif scale_choice == '3':
        vmin_input = input("vmin: ").strip()
        vmax_input = input("vmax: ").strip()
        vmin = float(vmin_input) if vmin_input else None
        vmax = float(vmax_input) if vmax_input else None
        center_input = input("Centre custom range at zero? (y/n) [n]: ").strip().lower()
        center_zero = center_input == 'y'
        if center_zero and not (vmin is not None and vmax is not None and vmin < 0 < vmax):
            print("  Warning: Zero-centred scaling requires vmin < 0 < vmax. Using linear custom range.")
            center_zero = False
    elif scale_choice == '4':
        center_zero = True

    # Combined plot option
    combined_plot = False
    shared_scale = False
    if len(selected_vars) > 1:
        combined_input = input("\nCombine all variables in one figure? (y/n) [n]: ").strip().lower()
        combined_plot = combined_input == 'y'
        if combined_plot:
            shared_input = input("Use same colour scale for all plots? (y/n) [n]: ").strip().lower()
            shared_scale = shared_input == 'y'

    # Save options
    save_fig = True

    save_dir = None
    if save_fig:
        save_dir = input(f"Save directory [{os.getcwd()}]: ").strip()
        if not save_dir:
            save_dir = os.getcwd()
        save_dir = os.path.expanduser(os.path.expandvars(save_dir))

    display_input = input("Display figures? (y/n) [n]: ").strip().lower()
    display = display_input == 'y'

    return {
        'variables': selected_vars,
        'plane': axis_info['plane'] if axis_info else 'xz',
        'index': None,  # Not applicable for pre-sliced data
        'cmap': cmap,
        'symmetric': symmetric,
        'center_zero': center_zero,
        'vmin': vmin,
        'vmax': vmax,
        'save_dir': save_dir,
        'save_fig': save_fig,
        'display': display,
        'combined_plot': combined_plot,
        'shared_scale': shared_scale,
        'x_crop': x_crop,
        'interpolate_cell_to_point': interpolate_cell_to_point,
    }


def get_slice_config(var_metadata, grid_info):
    """Get slice configuration from user using pre-parsed variable metadata."""

    # List available variables (only 3D)
    all_variables = sorted(var_metadata.keys())
    variables = [v for v in all_variables if len(var_metadata[v].get('shape', ())) == 3]

    if not variables:
        print("Error: No 3D variables found in dataset.")
        return None

    print(f"\nAvailable 3D variables ({len(variables)}):")
    for i, var in enumerate(variables):
        shape = var_metadata[var]['shape']
        print(f"  {i+1:2d}. {var:20s} shape: {shape}")

    print("\nSelection options:")
    print("  - Single e.g: 1 or variable_name")
    print("  - Multiple e.g: 1,3,5 or 1 3 5")
    print("  - Range e.g: 1-5")
    print("  - All: all")

    var_choice = input("\nVariables to plot: ").strip()

    # Parse variable selection
    selected_vars = parse_variable_selection(var_choice, variables)

    if not selected_vars:
        print("No valid variables selected, using first variable")
        selected_vars = [variables[0]]

    print(f"\nSelected {len(selected_vars)} variable(s): {selected_vars}")

    # Get shape from first variable (all should be same for slicing)
    nz, ny, nx = var_metadata[selected_vars[0]]['shape']

    # Select slice plane
    print("\nSlice planes:")
    print(f"  xy - constant z (0 to {nz-1})")
    print(f"  xz - constant y (0 to {ny-1})")
    print(f"  yz - constant x (0 to {nx-1})")

    plane = input("Slice plane [xy]: ").strip().lower()
    if plane not in ['xy', 'xz', 'yz']:
        plane = 'xy'

    # Get index range for selected plane
    if plane == 'xy':
        max_idx = nz - 1
        coord_name = 'z'
        coords = grid_info.get('grid_z', np.arange(nz))
    elif plane == 'xz':
        max_idx = ny - 1
        coord_name = 'y'
        coords = grid_info.get('grid_y', np.arange(ny))
    else:  # yz
        max_idx = nx - 1
        coord_name = 'x'
        coords = grid_info.get('grid_x', np.arange(nx))

    print(f"\n{coord_name} range: {coords.min():.4f} to {coords.max():.4f} (indices 0 to {max_idx})")

    # Get slice index or location
    idx_input = input(f"Slice index or {coord_name} value [{max_idx//2}]: ").strip()

    if idx_input == '':
        index = max_idx // 2
    else:
        try:
            val = float(idx_input)
            if val <= max_idx and val == int(val):
                # Interpret as index
                index = int(val)
            else:
                # Interpret as coordinate value, find nearest index
                index = np.argmin(np.abs(coords - val))
                print(f"  -> Nearest index: {index} ({coord_name} = {coords[index]:.4f})")
        except ValueError:
            index = max_idx // 2

    index = max(0, min(index, max_idx))

    # X-direction cropping (optional, only for xy and xz planes)
    x_crop = None
    if plane in ['xy', 'xz']:
        grid_x = grid_info.get('grid_x', None)
        if grid_x is not None:
            print(f"\nX-direction cropping (optional):")
            print(f"  Full x range: {grid_x.min():.4f} to {grid_x.max():.4f}")
            crop_input = input("Crop x range? (y/n) [n]: ").strip().lower()
            if crop_input == 'y':
                x_min_input = input(f"  x_min [{grid_x.min():.4f}]: ").strip()
                x_max_input = input(f"  x_max [{grid_x.max():.4f}]: ").strip()
                x_min = float(x_min_input) if x_min_input else grid_x.min()
                x_max = float(x_max_input) if x_max_input else grid_x.max()
                x_crop = (x_min, x_max)
                print(f"  -> Cropping x from {x_min:.4f} to {x_max:.4f}")

    # Colormap selection
    cmap = choose_colormap('RdBu_r')

    # Color scale options
    print("\nColor scale options:")
    print("  1. Auto (min to max)")
    print("  2. Symmetric around zero")
    print("  3. Custom range")
    print("  4. Centre at zero (crop to data min/max)")
    scale_choice = input("Scale option [1]: ").strip()

    symmetric = False
    center_zero = False
    vmin, vmax = None, None

    if scale_choice == '2':
        symmetric = True
    elif scale_choice == '3':
        vmin_input = input("vmin: ").strip()
        vmax_input = input("vmax: ").strip()
        vmin = float(vmin_input) if vmin_input else None
        vmax = float(vmax_input) if vmax_input else None
        center_input = input("Centre custom range at zero? (y/n) [n]: ").strip().lower()
        center_zero = center_input == 'y'
        if center_zero and not (vmin is not None and vmax is not None and vmin < 0 < vmax):
            print("  Warning: Zero-centred scaling requires vmin < 0 < vmax. Using linear custom range.")
            center_zero = False
    elif scale_choice == '4':
        center_zero = True

    # Combined plot option (only ask if multiple variables selected)
    combined_plot = False
    shared_scale = False
    if len(selected_vars) > 1:
        combined_input = input("\nCombine all variables in one figure? (y/n) [n]: ").strip().lower()
        combined_plot = combined_input == 'y'

        if combined_plot:
            shared_input = input("Use same colour scale for all plots? (y/n) [n]: ").strip().lower()
            shared_scale = shared_input == 'y'

    # Save options
    save_fig = True

    save_dir = None
    if save_fig:
        save_dir = input(f"Save directory [{os.getcwd()}]: ").strip()
        if not save_dir:
            save_dir = os.getcwd()
        save_dir = os.path.expanduser(os.path.expandvars(save_dir))

    display_input = input("Display figures? (y/n) [n]: ").strip().lower()
    display = display_input == 'y'

    interp_input = input("\nInterpolate cell data to point data? (y/n) [y]: ").strip().lower()
    interpolate_cell_to_point = interp_input != 'n'

    return {
        'variables': selected_vars,
        'plane': plane,
        'index': index,
        'cmap': cmap,
        'symmetric': symmetric,
        'center_zero': center_zero,
        'vmin': vmin,
        'vmax': vmax,
        'save_dir': save_dir,
        'save_fig': save_fig,
        'display': display,
        'combined_plot': combined_plot,
        'shared_scale': shared_scale,
        'x_crop': x_crop,
        'interpolate_cell_to_point': interpolate_cell_to_point,
    }


def main():
    """Main execution function."""

    # Get initial configuration
    config = get_user_input()
    if config is None:
        return

    print("\n" + "=" * 60)
    print(f"Reading metadata from {config['xdmf_path']}...")
    print("=" * 60)

    # Step 1: Parse XDMF metadata (variable names, shapes, grid coords only)
    var_metadata, grid_info = ut.parse_xdmf_metadata(config['xdmf_path'])

    if not var_metadata:
        print("Error: No variables found in XDMF file. Check file path and format.")
        return

    print(f"\nFound {len(var_metadata)} variables (data not yet loaded)")

    if grid_info:
        dims = grid_info.get('cell_dimensions', grid_info.get('node_dimensions', 'unknown'))
        print(f"Grid dimensions: {dims}")

    is_2d_slice = config.get('is_2d_slice', False)
    slice_label = config.get('slice_label', None)

    # Step 2: Get slice/plot configuration
    if is_2d_slice:
        slice_config = get_2d_plot_config(var_metadata, grid_info, slice_label)
    else:
        slice_config = get_slice_config(var_metadata, grid_info)
    if slice_config is None:
        return

    # Step 3: Load only the selected variables
    print("\n" + "=" * 60)
    print(f"Loading {len(slice_config['variables'])} selected variable(s)...")
    print("=" * 60)

    data = ut.load_xdmf_variables(
        var_metadata,
        slice_config['variables'],
        grid_info=grid_info,
    )

    if not data:
        print("Error: Failed to load selected variables.")
        return

    data, interpolated_vars = process_data_arrays(
        data,
        slice_config['variables'],
        grid_info,
        interpolate_cell_to_point=slice_config.get('interpolate_cell_to_point', False)
    )

    print(f"\nLoaded {len(data)} variable(s)")

    print("\n" + "=" * 60)
    print(f"Generating {len(slice_config['variables'])} slice(s)...")
    print("=" * 60)

    if is_2d_slice:
        # Data is already 2D — determine axis info from slice label
        axis_info = ut.slice_axis_info(slice_label)
        axis_labels = axis_info['axis_labels'] if axis_info else ('$x$', '$y$')
        coord1_key, coord2_key = axis_info['coord_keys'] if axis_info else ('grid_x', 'grid_y')
        slice_info = f"({axis_info['normal_dir']} index = {axis_info['normal_index']})" if axis_info else ""

        # Get coordinate arrays from grid_info
        sample_var = data[slice_config['variables'][0]]
        coord1 = grid_info.get(coord1_key, np.arange(sample_var.shape[1]))
        coord2 = grid_info.get(coord2_key, np.arange(sample_var.shape[0]))

        if slice_config['combined_plot']:
            slices_data = []
            for variable in tqdm(slice_config['variables'], desc="Preparing", unit="var"):
                slice_data = data[variable]
                # Squeeze any singleton dimensions
                slice_data = np.squeeze(slice_data)
                if slice_config['x_crop'] is not None and axis_info and axis_info['plane'] in ['xz', 'xy']:
                    slice_data, coord1 = ut.apply_x_crop(slice_data, coord1, slice_config['x_crop'])
                slices_data.append((variable, slice_data))
                print(f"\n{variable}: min={np.nanmin(slice_data):.4e}, max={np.nanmax(slice_data):.4e}, mean={np.nanmean(slice_data):.4e}")

            save_path = build_save_path(
                slice_config['save_dir'],
                'combined',
                slice_label,
                config['timestep'],
                combined=True,
            )

            plot_combined_slices(
                slices_data, coord1, coord2, axis_labels, slice_info,
                cmap=slice_config['cmap'], symmetric=slice_config['symmetric'],
                shared_scale=slice_config['shared_scale'],
                save_path=save_path, display=slice_config['display'],
                point_data_vars=interpolated_vars,
                center_zero=slice_config.get('center_zero', False)
            )
        else:
            for variable in tqdm(slice_config['variables'], desc="Plotting", unit="var"):
                slice_data = np.squeeze(data[variable])
                c1, c2 = coord1, coord2
                if slice_config['x_crop'] is not None and axis_info and axis_info['plane'] in ['xz', 'xy']:
                    slice_data, c1 = ut.apply_x_crop(slice_data, c1, slice_config['x_crop'])

                save_path = build_save_path(
                    slice_config['save_dir'],
                    variable,
                    slice_label,
                    config['timestep'],
                    interpolated=variable in interpolated_vars,
                )

                plot_slice(
                    slice_data, c1, c2, axis_labels, variable,
                    cmap=slice_config['cmap'],
                    vmin=slice_config['vmin'], vmax=slice_config['vmax'],
                    symmetric=slice_config['symmetric'],
                    slice_info=slice_info, save_path=save_path,
                    display=slice_config['display'],
                    smooth_point_data=variable in interpolated_vars,
                    center_zero=slice_config.get('center_zero', False)
                )
    else:
        # 3D data — extract slice as before
        slice_loc = get_slice_location(grid_info, slice_config['plane'], slice_config['index'])
        if slice_config['plane'] == 'xy':
            slice_info = f"(z = {slice_loc:.4f})" if isinstance(slice_loc, float) else f"(z index = {slice_loc})"
        elif slice_config['plane'] == 'xz':
            slice_info = f"(y = {slice_loc:.4f})" if isinstance(slice_loc, float) else f"(y index = {slice_loc})"
        else:
            slice_info = f"(x = {slice_loc:.4f})" if isinstance(slice_loc, float) else f"(x index = {slice_loc})"

        if slice_config['combined_plot']:
            # Extract all slices first
            slices_data = []
            coord1, coord2, axis_labels = None, None, None

            for variable in tqdm(slice_config['variables'], desc="Extracting", unit="var"):
                var_data = data[variable]
                slice_data, coord1, coord2, axis_labels = extract_slice(
                    var_data,
                    slice_config['plane'],
                    slice_config['index'],
                    grid_info
                )

                # Apply x-direction cropping if specified (only for xy and xz planes)
                if slice_config['x_crop'] is not None and slice_config['plane'] in ['xy', 'xz']:
                    slice_data, coord1_cropped = ut.apply_x_crop(slice_data, coord1, slice_config['x_crop'])
                    coord1 = coord1_cropped

                slices_data.append((variable, slice_data))

                # Print statistics
                print(f"\n{variable}: min={np.nanmin(slice_data):.4e}, max={np.nanmax(slice_data):.4e}, mean={np.nanmean(slice_data):.4e}")

            # Build save path for combined figure
            save_path = build_save_path(
                slice_config['save_dir'],
                'combined',
                slice_config['plane'],
                config['timestep'],
                combined=True,
            )

            # Plot combined figure
            plot_combined_slices(
                slices_data, coord1, coord2, axis_labels, slice_info,
                cmap=slice_config['cmap'],
                symmetric=slice_config['symmetric'],
                shared_scale=slice_config['shared_scale'],
                save_path=save_path,
                display=slice_config['display'],
                point_data_vars=interpolated_vars,
                center_zero=slice_config.get('center_zero', False)
            )
        else:
            # Process each variable separately
            for variable in tqdm(slice_config['variables'], desc="Plotting", unit="var"):
                var_data = data[variable]
                slice_data, coord1, coord2, axis_labels = extract_slice(
                    var_data,
                    slice_config['plane'],
                    slice_config['index'],
                    grid_info
                )

                # Apply x-direction cropping if specified (only for xy and xz planes)
                if slice_config['x_crop'] is not None and slice_config['plane'] in ['xy', 'xz']:
                    slice_data, coord1 = ut.apply_x_crop(slice_data, coord1, slice_config['x_crop'])

                # Build save path for this variable
                save_path = build_save_path(
                    slice_config['save_dir'],
                    variable,
                    slice_config['plane'],
                    config['timestep'],
                    interpolated=variable in interpolated_vars,
                )

                # Plot
                plot_slice(
                    slice_data, coord1, coord2, axis_labels,
                    variable,
                    cmap=slice_config['cmap'],
                    vmin=slice_config['vmin'],
                    vmax=slice_config['vmax'],
                    symmetric=slice_config['symmetric'],
                    slice_info=slice_info,
                    save_path=save_path,
                    display=slice_config['display'],
                    smooth_point_data=variable in interpolated_vars,
                    center_zero=slice_config.get('center_zero', False)
                )

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)

    # Ask user whether to plot another or exit
    while True:
        choice = input("\nPlot another slice? (y/n): ").strip().lower()
        if choice in ('y', 'yes'):
            same_timestep = input("Same timestep? (y/n) [y]: ").strip().lower()
            if same_timestep not in ('', 'y', 'yes'):
                main()
                return
            if is_2d_slice:
                slice_config = get_2d_plot_config(var_metadata, grid_info, slice_label)
            else:
                slice_config = get_slice_config(var_metadata, grid_info)
            if slice_config is None:
                continue

            # Load selected variables
            data = ut.load_xdmf_variables(
                var_metadata,
                slice_config['variables'],
                grid_info=grid_info,
            )
            if not data:
                print("Error: Failed to load selected variables.")
                continue

            data, interpolated_vars = process_data_arrays(
                data,
                slice_config['variables'],
                grid_info,
                interpolate_cell_to_point=slice_config.get('interpolate_cell_to_point', False)
            )

            if is_2d_slice:
                axis_info = ut.slice_axis_info(slice_label)
                axis_labels = axis_info['axis_labels'] if axis_info else ('$x$', '$y$')
                coord1_key, coord2_key = axis_info['coord_keys'] if axis_info else ('grid_x', 'grid_y')
                slice_info = f"({axis_info['normal_dir']} index = {axis_info['normal_index']})" if axis_info else ""
                sample_var = data[slice_config['variables'][0]]
                coord1 = grid_info.get(coord1_key, np.arange(sample_var.shape[1]))
                coord2 = grid_info.get(coord2_key, np.arange(sample_var.shape[0]))

                if slice_config['combined_plot']:
                    slices_data = []
                    for variable in tqdm(slice_config['variables'], desc="Preparing", unit="var"):
                        s_data = np.squeeze(data[variable])
                        c1 = coord1
                        if slice_config['x_crop'] is not None and axis_info and axis_info['plane'] in ['xz', 'xy']:
                            s_data, c1 = ut.apply_x_crop(s_data, c1, slice_config['x_crop'])
                        slices_data.append((variable, s_data))
                        print(f"\n{variable}: min={np.nanmin(s_data):.4e}, max={np.nanmax(s_data):.4e}, mean={np.nanmean(s_data):.4e}")
                    save_path = build_save_path(
                        slice_config['save_dir'],
                        'combined',
                        slice_label,
                        config['timestep'],
                        combined=True,
                    )
                    plot_combined_slices(
                        slices_data, c1, coord2, axis_labels, slice_info,
                        cmap=slice_config['cmap'], symmetric=slice_config['symmetric'],
                        shared_scale=slice_config['shared_scale'], save_path=save_path,
                        display=slice_config['display'],
                        point_data_vars=interpolated_vars,
                        center_zero=slice_config.get('center_zero', False)
                    )
                else:
                    for variable in tqdm(slice_config['variables'], desc="Plotting", unit="var"):
                        s_data = np.squeeze(data[variable])
                        c1, c2 = coord1, coord2
                        if slice_config['x_crop'] is not None and axis_info and axis_info['plane'] in ['xz', 'xy']:
                            s_data, c1 = ut.apply_x_crop(s_data, c1, slice_config['x_crop'])
                        save_path = build_save_path(
                            slice_config['save_dir'],
                            variable,
                            slice_label,
                            config['timestep'],
                            interpolated=variable in interpolated_vars,
                        )
                        plot_slice(
                            s_data, c1, c2, axis_labels, variable,
                            cmap=slice_config['cmap'], vmin=slice_config['vmin'],
                            vmax=slice_config['vmax'], symmetric=slice_config['symmetric'],
                            slice_info=slice_info, save_path=save_path,
                            display=slice_config['display'],
                            smooth_point_data=variable in interpolated_vars,
                            center_zero=slice_config.get('center_zero', False)
                        )
            else:
                slice_loc = get_slice_location(grid_info, slice_config['plane'], slice_config['index'])
                if slice_config['plane'] == 'xy':
                    slice_info = f"(z = {slice_loc:.4f})" if isinstance(slice_loc, float) else f"(z index = {slice_loc})"
                elif slice_config['plane'] == 'xz':
                    slice_info = f"(y = {slice_loc:.4f})" if isinstance(slice_loc, float) else f"(y index = {slice_loc})"
                else:
                    slice_info = f"(x = {slice_loc:.4f})" if isinstance(slice_loc, float) else f"(x index = {slice_loc})"

                if slice_config['combined_plot']:
                    slices_data = []
                    coord1, coord2, axis_labels = None, None, None
                    for variable in tqdm(slice_config['variables'], desc="Extracting", unit="var"):
                        var_data = data[variable]
                        slice_data, coord1, coord2, axis_labels = extract_slice(
                            var_data, slice_config['plane'], slice_config['index'], grid_info
                        )
                        if slice_config['x_crop'] is not None and slice_config['plane'] in ['xy', 'xz']:
                            slice_data, coord1 = ut.apply_x_crop(slice_data, coord1, slice_config['x_crop'])
                        slices_data.append((variable, slice_data))
                        print(f"\n{variable}: min={np.nanmin(slice_data):.4e}, max={np.nanmax(slice_data):.4e}, mean={np.nanmean(slice_data):.4e}")

                    save_path = build_save_path(
                        slice_config['save_dir'],
                        'combined',
                        slice_config['plane'],
                        config['timestep'],
                        combined=True,
                    )
                    plot_combined_slices(
                        slices_data, coord1, coord2, axis_labels, slice_info,
                        cmap=slice_config['cmap'], symmetric=slice_config['symmetric'],
                        shared_scale=slice_config['shared_scale'], save_path=save_path,
                        display=slice_config['display'], point_data_vars=interpolated_vars,
                        center_zero=slice_config.get('center_zero', False)
                    )
                else:
                    for variable in tqdm(slice_config['variables'], desc="Plotting", unit="var"):
                        var_data = data[variable]
                        slice_data, coord1, coord2, axis_labels = extract_slice(
                            var_data, slice_config['plane'], slice_config['index'], grid_info
                        )
                        if slice_config['x_crop'] is not None and slice_config['plane'] in ['xy', 'xz']:
                            slice_data, coord1 = ut.apply_x_crop(slice_data, coord1, slice_config['x_crop'])
                        save_path = build_save_path(
                            slice_config['save_dir'],
                            variable,
                            slice_config['plane'],
                            config['timestep'],
                            interpolated=variable in interpolated_vars,
                        )
                        plot_slice(
                            slice_data, coord1, coord2, axis_labels, variable,
                            cmap=slice_config['cmap'], vmin=slice_config['vmin'],
                            vmax=slice_config['vmax'], symmetric=slice_config['symmetric'],
                            slice_info=slice_info, save_path=save_path,
                            display=slice_config['display'],
                            smooth_point_data=variable in interpolated_vars,
                            center_zero=slice_config.get('center_zero', False)
                        )

            print("\n" + "=" * 60)
            print("Complete!")
            print("=" * 60)
        elif choice in ('n', 'no', ''):
            print("Exiting.")
            break
        else:
            print("Please enter 'y' or 'n'.")


if __name__ == '__main__':
    main()
