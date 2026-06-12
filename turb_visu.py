#!/usr/bin/env python3

import os
import glob
import numpy as np
import pyvista as pv
from tqdm import tqdm

import utils as ut

try:
    import readline
    def _path_completer(text, state):
        expanded = os.path.expanduser(os.path.expandvars(text))
        pattern = os.path.join(expanded, '*') if os.path.isdir(expanded) else expanded + '*'
        matches = glob.glob(pattern)
        matches = [m + '/' if os.path.isdir(m) else m for m in matches]
        return matches[state] if state < len(matches) else None
    readline.set_completer(_path_completer)
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind('tab: complete')
except ImportError:
    pass


LARGE_GRID_THRESHOLD = 10_000_000

COLORMAPS = ['RdBu_r', 'viridis', 'plasma', 'inferno', 'coolwarm', 'jet']

OPACITY_PRESETS = ['linear', 'sigmoid', 'sigmoid_r', 'geom', 'geom_r']


# ---------------------------------------------------------------------------
# Data loading and grid construction
# ---------------------------------------------------------------------------

def get_available_timesteps(visu_folder):
    """Extract available timesteps from XDMF filenames."""
    timesteps = set()
    for f in os.listdir(visu_folder):
        if f.endswith('.xdmf'):
            parts = f.replace('.xdmf', '').split('_')
            if parts:
                timesteps.add(parts[-1])
    return sorted(timesteps)


def build_pyvista_grid(grid_info, data_dict, stride=1):
    """Build a PyVista RectilinearGrid from CHAPSim2 grid coordinates and data."""
    x = grid_info['grid_x'][::stride]
    y = grid_info['grid_y'][::stride]
    z = grid_info['grid_z'][::stride]

    # Cell counts after subsampling (one fewer than nodes)
    nx, ny, nz = len(x) - 1, len(y) - 1, len(z) - 1

    grid = pv.RectilinearGrid(x, y, z)

    for name, arr in data_dict.items():
        sampled = arr[::stride, ::stride, ::stride]
        # Clip to match grid cell count (off-by-one can arise at non-divisible strides)
        sampled = sampled[:nz, :ny, :nx]
        # CHAPSim2 arrays are (nz, ny, nx); C-order flatten matches VTK's x-fastest ordering
        grid.cell_data[name] = sampled.flatten()

    return grid


def compute_q_criterion(data_dict, grid_info):
    """
    Compute Q-criterion from velocity components qx_ccc, qy_ccc, qz_ccc.

    Q = -0.5 * (A_ij * A_ji) where A_ij = du_i/dx_j.
    Positive Q identifies vortex cores (rotation dominates strain).
    """
    for req in ('qx_ccc', 'qy_ccc', 'qz_ccc'):
        if req not in data_dict:
            print(f"  Cannot compute Q-criterion: '{req}' not loaded.")
            return None

    x_cell = 0.5 * (grid_info['grid_x'][:-1] + grid_info['grid_x'][1:])
    y_cell = 0.5 * (grid_info['grid_y'][:-1] + grid_info['grid_y'][1:])
    z_cell = 0.5 * (grid_info['grid_z'][:-1] + grid_info['grid_z'][1:])

    print("  Computing velocity gradients (this may take a moment)...")
    u, v, w = data_dict['qx_ccc'], data_dict['qy_ccc'], data_dict['qz_ccc']

    # np.gradient returns [d/dz, d/dy, d/dx] for shape (nz, ny, nx)
    du_dz, du_dy, du_dx = np.gradient(u, z_cell, y_cell, x_cell)
    dv_dz, dv_dy, dv_dx = np.gradient(v, z_cell, y_cell, x_cell)
    dw_dz, dw_dy, dw_dx = np.gradient(w, z_cell, y_cell, x_cell)

    Q = -0.5 * (
        du_dx**2 + dv_dy**2 + dw_dz**2
        + 2.0 * (du_dy * dv_dx + du_dz * dw_dx + dv_dz * dw_dy)
    )
    return Q


# ---------------------------------------------------------------------------
# User interaction helpers
# ---------------------------------------------------------------------------

def choose_colormap(default='RdBu_r'):
    """Prompt the user to pick a colormap from the list."""
    print("\nColormaps:")
    for i, name in enumerate(COLORMAPS, 1):
        print(f"  {i}. {name}")
    choice = input(f"Colormap [{default}]: ").strip()
    if not choice:
        return default
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(COLORMAPS):
            return COLORMAPS[idx]
    return choice if choice in COLORMAPS else default


def _ask_coord(prompt, coords):
    """Prompt for a coordinate position; returns the midpoint if left blank."""
    if coords is None:
        return None
    lo, hi = float(coords[0]), float(coords[-1])
    mid = 0.5 * (lo + hi)
    raw = input(f"{prompt} [{mid:.4f}, range {lo:.4f}–{hi:.4f}]: ").strip()
    if not raw:
        return mid
    try:
        return float(raw)
    except ValueError:
        return mid


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_user_input():
    """Collect case folder, timestep, and XDMF file path interactively."""
    print("=" * 60)
    print("3D Turbulence Visualiser")
    print("=" * 60)

    case_folder = input("\nPath to case folder: ").strip() or os.getcwd()
    case_folder = os.path.expanduser(os.path.expandvars(case_folder))

    if os.path.basename(case_folder) == '2_visu':
        visu_folder = case_folder
        case_folder = os.path.dirname(case_folder)
    else:
        visu_folder = os.path.join(case_folder, '2_visu')

    if not os.path.isdir(visu_folder):
        print(f"Error: {visu_folder} not found.")
        return None

    timesteps = get_available_timesteps(visu_folder)
    if timesteps:
        print(f"Available timesteps: {timesteps}")
    timestep = input("Timestep: ").strip()

    print("\nData types:  1. inst   2. t_avg")
    data_type = {'2': 't_avg', 't_avg': 't_avg'}.get(
        input("Data type [1]: ").strip(), 'inst')

    print("Physics types:  1. flow   2. thermo   3. mhd")
    physics_type = {'2': 'thermo', '3': 'mhd', 'thermo': 'thermo', 'mhd': 'mhd'}.get(
        input("Physics type [1]: ").strip(), 'flow')

    filename = (
        f"domain1_{physics_type}_{timestep}.xdmf" if data_type == 'inst'
        else f"domain1_{data_type}_{physics_type}_{timestep}.xdmf"
    )
    xdmf_path = os.path.join(visu_folder, filename)

    if not os.path.isfile(xdmf_path):
        print(f"Error: {filename} not found. Available files:")
        for f in sorted(os.listdir(visu_folder)):
            if f.endswith('.xdmf'):
                print(f"  {f}")
        return None

    return {'xdmf_path': xdmf_path, 'visu_folder': visu_folder, 'timestep': timestep}


def get_visualization_config(var_metadata, grid_info):
    """Collect visualization mode, variable choice, and rendering parameters."""
    variables = sorted(v for v, m in var_metadata.items()
                       if len(m.get('shape', ())) == 3)
    if not variables:
        print("Error: No 3D variables found.")
        return None

    print(f"\nAvailable variables ({len(variables)}):")
    for i, var in enumerate(variables, 1):
        print(f"  {i:2d}. {var:25s}  shape: {var_metadata[var]['shape']}")
    print(f"  {len(variables)+1:2d}. {'Q-criterion':25s}  (requires qx_ccc, qy_ccc, qz_ccc)")

    print("\nModes:  1. Slice   2. Iso-surface   3. Volume rendering   4. Streamlines   5. Glyphs")
    mode = {'2': 'iso', '3': 'volume', '4': 'streamlines', '5': 'glyphs',
            'iso': 'iso', 'volume': 'volume', 'streamlines': 'streamlines', 'glyphs': 'glyphs'}.get(
        input("Mode [1]: ").strip(), 'slice')

    var_choice = input(f"\nVariable [1]: ").strip()
    use_q = (var_choice == str(len(variables) + 1) or var_choice.lower() == 'q')

    if use_q:
        variable = 'Q-criterion'
        selected_vars = ['qx_ccc', 'qy_ccc', 'qz_ccc']
    else:
        try:
            idx = (int(var_choice) - 1) if var_choice else 0
        except ValueError:
            idx = variables.index(var_choice) if var_choice in variables else 0
        variable = variables[max(0, min(idx, len(variables) - 1))]
        selected_vars = [variable]

    cmap = choose_colormap('viridis' if mode in ('iso', 'streamlines') else 'RdBu_r')

    if mode in ('streamlines', 'glyphs'):
        selected_vars = list({'qx_ccc', 'qy_ccc', 'qz_ccc'} | set(selected_vars))

    cfg = {
        'mode': mode,
        'variable': variable,
        'selected_vars': selected_vars,
        'use_q_criterion': use_q,
        'cmap': cmap,
    }

    if mode == 'slice':
        print("\nSlice positions (leave blank to use midpoint, skip=none):")
        cfg['cut_x'] = _ask_coord("  YZ plane at x", grid_info.get('grid_x'))
        cfg['cut_y'] = _ask_coord("  XZ plane at y", grid_info.get('grid_y'))
        cfg['cut_z'] = _ask_coord("  XY plane at z", grid_info.get('grid_z'))

    elif mode == 'volume':
        print(f"\nOpacity presets: {', '.join(OPACITY_PRESETS)}")
        cfg['opacity'] = input("Opacity [sigmoid]: ").strip() or 'sigmoid'

    elif mode == 'streamlines':
        seed = input("\nSeed type (sphere/line) [line]: ").strip().lower() or 'line'
        cfg['stream_seed'] = seed if seed in ('sphere', 'line') else 'line'
        if cfg['stream_seed'] == 'sphere':
            print("Sphere seed (leave blank for domain centre / auto radius):")
            cx = _ask_coord("  Centre x", grid_info.get('grid_x'))
            cy = _ask_coord("  Centre y", grid_info.get('grid_y'))
            cz = _ask_coord("  Centre z", grid_info.get('grid_z'))
            cfg['stream_center'] = (cx, cy, cz)
            raw = input("  Radius [auto]: ").strip()
            cfg['stream_radius'] = float(raw) if raw else None
        else:
            gx, gy, gz = grid_info.get('grid_x'), grid_info.get('grid_y'), grid_info.get('grid_z')
            xmid = 0.5 * (float(gx[0]) + float(gx[-1])) if gx is not None else 0.0
            ymid = 0.5 * (float(gy[0]) + float(gy[-1])) if gy is not None else 0.0
            print("Line seed start point (leave blank for domain centre, full z-span):")
            x0 = _ask_coord("  Start x", gx) if gx is not None else xmid
            y0 = _ask_coord("  Start y", gy) if gy is not None else ymid
            raw = input(f"  Start z [{float(gz[0]):.4f}]: ").strip() if gz is not None else ''
            z0 = float(raw) if raw else float(gz[0])
            print("Line seed end point:")
            x1 = _ask_coord("  End x", gx) if gx is not None else xmid
            y1 = _ask_coord("  End y", gy) if gy is not None else ymid
            raw = input(f"  End z [{float(gz[-1]):.4f}]: ").strip() if gz is not None else ''
            z1 = float(raw) if raw else float(gz[-1])
            cfg['stream_pointa'] = (x0, y0, z0)
            cfg['stream_pointb'] = (x1, y1, z1)
        raw = input("  Number of seed points [50]: ").strip()
        cfg['stream_n_seeds'] = int(raw) if raw else 50
        raw = input("  Max integration steps [2000]: ").strip()
        cfg['stream_max_steps'] = int(raw) if raw else 2000
        cfg['stream_direction'] = input("  Direction (both/forward/backward) [both]: ").strip() or 'both'

    elif mode == 'glyphs':
        print("\nGlyph parameters (leave blank for auto):")
        raw = input("  Scale factor [auto]: ").strip()
        cfg['glyph_factor'] = float(raw) if raw else None
        raw = input("  Every N points [auto ~5000 glyphs]: ").strip()
        cfg['glyph_every_n'] = int(raw) if raw else None
        glyph = input("  Glyph type (arrow/cone) [arrow]: ").strip().lower() or 'arrow'
        cfg['glyph_type'] = glyph if glyph in ('arrow', 'cone') else 'arrow'

    screenshot = input("\nSave screenshot? (y/n) [n]: ").strip().lower() == 'y'
    cfg['screenshot_path'] = (
        (input("Screenshot path [visu_screenshot.png]: ").strip() or 'visu_screenshot.png')
        if screenshot else None
    )

    return cfg


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_scene(grid, cfg):
    """Build and display the PyVista interactive scene."""
    variable = cfg['variable']
    cmap = cfg['cmap']
    mode = cfg['mode']

    plotter = pv.Plotter(title=f"CHAPSim2 | {variable}")
    outline = grid.outline()
    plotter.add_mesh(outline, color='gray', line_width=1)

    if mode == 'slice':
        planes = [
            ('x', cfg.get('cut_x')),
            ('y', cfg.get('cut_y')),
            ('z', cfg.get('cut_z')),
        ]
        origins = {'x': lambda v: (v, 0, 0), 'y': lambda v: (0, v, 0), 'z': lambda v: (0, 0, v)}
        n_added = 0
        for normal, pos in planes:
            if pos is None:
                continue
            sl = grid.slice(normal=normal, origin=origins[normal](pos))
            plotter.add_mesh(sl, scalars=variable, cmap=cmap, show_scalar_bar=(n_added == 0))
            n_added += 1

        if n_added == 0:
            print("  No slice planes defined.")
            return

        plotter.add_axes()
        plotter.show_grid()

    elif mode == 'iso':
        arr = grid.cell_data[variable]
        vmin, vmax = float(arr.min()), float(arr.max())
        print(f"  {variable} range: {vmin:.4e} to {vmax:.4e}")
        raw_min = input(f"  Min iso-value [{0.5*(vmin+vmax):.4e}]: ").strip()
        iso_min = float(raw_min) if raw_min else 0.5 * (vmin + vmax)
        raw_steps = input(f"  Number of surfaces [1]: ").strip()
        iso_steps = int(raw_steps) if raw_steps else 1
        if iso_steps > 1:
            raw_max = input(f"  Max iso-value [{vmax:.4e}]: ").strip()
            iso_max = float(raw_max) if raw_max else vmax
            iso_vals = list(np.linspace(iso_min, iso_max, iso_steps))
        else:
            iso_vals = [iso_min]
        out_of_range = [v for v in iso_vals if not (vmin <= v <= vmax)]
        if out_of_range:
            print(f"  Warning: {len(out_of_range)} value(s) outside data range.")

        grid_pt = grid.cell_data_to_point_data()
        contours = grid_pt.contour(isosurfaces=iso_vals, scalars=variable)
        if contours.n_points == 0:
            print(f"  Warning: iso-surface(s) are empty.")
        else:
            plotter.add_mesh(contours, scalars=variable, cmap=cmap, show_scalar_bar=True)

        plotter.add_axes()

    elif mode == 'volume':
        plotter.add_volume(grid, scalars=variable, cmap=cmap,
                           opacity=cfg.get('opacity', 'sigmoid'),
                           show_scalar_bar=True)
        plotter.add_axes()

    elif mode == 'streamlines':
        for req in ('qx_ccc', 'qy_ccc', 'qz_ccc'):
            if req not in grid.cell_data:
                print(f"  Streamlines require velocity components ({req} missing).")
                return
        grid_pt = grid.cell_data_to_point_data()
        u = grid_pt.point_data['qx_ccc']
        v = grid_pt.point_data['qy_ccc']
        w = grid_pt.point_data['qz_ccc']
        vel = np.column_stack([u, v, w])
        grid_pt['velocity'] = vel
        grid_pt['velocity_magnitude'] = np.linalg.norm(vel, axis=1)

        common = dict(
            n_points=cfg.get('stream_n_seeds', 50),
            integration_direction=cfg.get('stream_direction', 'both'),
            max_steps=cfg.get('stream_max_steps', 2000),
        )
        if cfg.get('stream_seed') == 'sphere':
            bounds = grid.bounds
            auto_radius = 0.1 * min(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
            streamlines = grid_pt.streamlines(
                'velocity',
                source_center=cfg.get('stream_center', grid.center),
                source_radius=cfg.get('stream_radius') or auto_radius,
                **common,
            )
        else:
            streamlines = grid_pt.streamlines(
                'velocity',
                pointa=cfg['stream_pointa'],
                pointb=cfg['stream_pointb'],
                **common,
            )
        if streamlines.n_points == 0:
            print("  Warning: no streamlines generated — try adjusting the seed centre or radius.")
        else:
            plotter.add_mesh(streamlines, scalars='velocity_magnitude',
                             cmap=cmap, line_width=2, show_scalar_bar=True)
        plotter.add_axes()

    elif mode == 'glyphs':
        for req in ('qx_ccc', 'qy_ccc', 'qz_ccc'):
            if req not in grid.cell_data:
                print(f"  Glyphs require velocity components ({req} missing).")
                return
        grid_pt = grid.cell_data_to_point_data()
        u = grid_pt.point_data['qx_ccc']
        v = grid_pt.point_data['qy_ccc']
        w = grid_pt.point_data['qz_ccc']
        vel = np.column_stack([u, v, w])
        vel_mag = np.linalg.norm(vel, axis=1)
        grid_pt['velocity'] = vel
        grid_pt['velocity_magnitude'] = vel_mag

        every_n = cfg.get('glyph_every_n') or max(1, grid_pt.n_points // 5000)
        indices = np.arange(0, grid_pt.n_points, every_n)
        sub = pv.PolyData(grid_pt.points[indices])
        for key in grid_pt.point_data.keys():
            sub[key] = grid_pt.point_data[key][indices]

        factor = cfg.get('glyph_factor')
        if factor is None:
            bounds = grid.bounds
            min_dim = min(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
            vel_rms = float(np.sqrt(np.mean(vel_mag**2)))
            factor = 0.05 * min_dim / max(vel_rms, 1e-12)

        geom = pv.Arrow() if cfg.get('glyph_type', 'arrow') == 'arrow' else pv.Cone()
        glyphs = sub.glyph(orient='velocity', scale='velocity_magnitude', factor=factor, geom=geom)
        if glyphs.n_points == 0:
            print("  Warning: no glyphs generated.")
        else:
            plotter.add_mesh(glyphs, scalars='velocity_magnitude',
                             cmap=cmap, show_scalar_bar=True)
        plotter.add_axes()

    if cfg.get('screenshot_path'):
        plotter.show(screenshot=cfg['screenshot_path'])
        print(f"  Screenshot saved: {cfg['screenshot_path']}")
    else:
        plotter.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = get_user_input()
    if config is None:
        return

    print(f"\nReading metadata from {config['xdmf_path']}...")
    var_metadata, grid_info = ut.parse_xdmf_metadata(config['xdmf_path'])
    if not var_metadata:
        print("Error: No variables found in XDMF file.")
        return

    n_cells = int(np.prod(grid_info.get('cell_dimensions', (1,))))
    print(f"Grid: {grid_info.get('cell_dimensions')} = {n_cells:,} cells")

    vis_cfg = get_visualization_config(var_metadata, grid_info)
    if vis_cfg is None:
        return

    stride = 1
    if n_cells > LARGE_GRID_THRESHOLD:
        print(f"\nLarge grid detected ({n_cells:,} cells).")
        raw = input("Stride for subsampling (1 = full, 2 = half res, etc.) [2]: ").strip()
        try:
            stride = max(1, int(raw) if raw else 2)
        except ValueError:
            stride = 2
        if stride > 1:
            print(f"  Using stride {stride}.")

    print(f"\nLoading {len(vis_cfg['selected_vars'])} variable(s)...")
    data = ut.load_xdmf_variables(var_metadata, vis_cfg['selected_vars'], grid_info=grid_info)
    if not data:
        print("Error: Failed to load data.")
        return

    if vis_cfg['use_q_criterion']:
        q = compute_q_criterion(data, grid_info)
        if q is None:
            return
        data['Q-criterion'] = q

    print("Building PyVista grid...")
    grid = build_pyvista_grid(grid_info, data, stride=stride)
    print(f"  {grid.dimensions} node dimensions, {grid.n_cells:,} cells")

    print("\nRendering...")
    render_scene(grid, vis_cfg)

    print("Done.")


if __name__ == '__main__':
    main()
