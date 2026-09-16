#!/usr/bin/env python3

import os
import glob
import numpy as np
import pyvista as pv
from tqdm import tqdm

import utils as ut
import operations as op

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

# add_volume() must resample onto a dense 3D texture; above this cell count
# that can exhaust GPU/VRAM or hang the driver (which tends to freeze the
# whole desktop, not just this process), so volume mode is refused above it.
VOLUME_CELL_THRESHOLD = 30_000_000

COLORMAPS = ['RdBu_r', 'viridis', 'plasma', 'inferno', 'coolwarm', 'jet']

OPACITY_PRESETS = ['linear', 'sigmoid', 'sigmoid_r', 'geom', 'geom_r']

WALL_DISTANCE_FIELD = 'Wall-distance'


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
    """Build a PyVista RectilinearGrid from CHAPSim2 grid coordinates and data.

    `data_dict` arrays are expected to already be subsampled by `stride`
    (via load_xdmf_variables(..., stride=stride)) so large domains never
    need to be fully loaded into RAM just to build a decimated grid — only
    the (cheap) coordinate arrays are strided here to match.
    """
    x = grid_info['grid_x'][::stride]
    y = grid_info['grid_y'][::stride]
    z = grid_info['grid_z'][::stride]

    # Cell counts after subsampling (one fewer than nodes)
    nx, ny, nz = len(x) - 1, len(y) - 1, len(z) - 1

    grid = pv.RectilinearGrid(x, y, z)

    for name, arr in data_dict.items():
        # Clip to match grid cell count (off-by-one can arise between the
        # coordinate stride here and the data's own pre-strided shape).
        sampled = arr[:nz, :ny, :nx]
        # CHAPSim2 arrays are (nz, ny, nx); C-order flatten matches VTK's x-fastest ordering
        grid.cell_data[name] = sampled.flatten()

    return grid


def strided_grid_info(grid_info, stride):
    """Return grid_info with coordinate arrays subsampled by `stride`.

    """
    if stride <= 1:
        return grid_info
    return {
        **grid_info,
        'grid_x': grid_info['grid_x'][::stride],
        'grid_y': grid_info['grid_y'][::stride],
        'grid_z': grid_info['grid_z'][::stride],
    }


def compute_wall_distance(grid_info, stride=1):
    """Cell-centred distance from the nearest wall, shaped (nz, ny, nx). Channel Only.

    """
    gi = strided_grid_info(grid_info, stride)
    x, y, z = gi['grid_x'], gi['grid_y'], gi['grid_z']
    y_centres = 0.5 * (y[:-1] + y[1:])
    distance = np.minimum(y_centres - float(y[0]), float(y[-1]) - y_centres)
    nz, ny, nx = len(z) - 1, len(y_centres), len(x) - 1
    return np.broadcast_to(distance[None, :, None], (nz, ny, nx)).copy()


def strided_cell_count(grid_info, stride):
    """Predicted PyVista grid cell count after subsampling coordinates by `stride`."""
    counts = []
    for key in ('grid_x', 'grid_y', 'grid_z'):
        arr = grid_info.get(key)
        if arr is None:
            return None
        counts.append(max(0, len(arr[::stride]) - 1))
    return int(np.prod(counts))

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

    print("\nModes:  1. Slice   2. Iso-surface   3. Volume rendering   4. Streamlines   5. Glyphs")
    mode = {'2': 'iso', '3': 'volume', '4': 'streamlines', '5': 'glyphs',
            'iso': 'iso', 'volume': 'volume', 'streamlines': 'streamlines', 'glyphs': 'glyphs'}.get(
        input("Mode [1]: ").strip(), 'slice')

    var_choice = input(f"\nVariable [1]: ").strip()
    try:
        idx = (int(var_choice) - 1) if var_choice else 0
    except ValueError:
        idx = variables.index(var_choice) if var_choice in variables else 0
    variable = variables[max(0, min(idx, len(variables) - 1))]
    selected_vars = [variable]

    # ---- Statistics (ordered by complexity) ----
    print("\nStatistics:  1. None   2. Fluctuation (u' = u_inst − u_t_avg)   3. Q-criterion   4. Vorticity")
    stat_choice = input("Statistic [1]: ").strip()
    use_fluc = stat_choice in ('2', 'fluc', 'fluctuation')
    use_q    = stat_choice in ('3', 'q', 'q-criterion', 'qcriterion')
    use_vort = stat_choice in ('4', 'vort', 'vorticity')

    t_avg_xdmf = None
    if use_fluc:
        t_avg_xdmf = input("  Path to t_avg xdmf file: ").strip()
        if not os.path.isfile(t_avg_xdmf):
            print(f"  Warning: t_avg file not found: {t_avg_xdmf}")

    if use_q:
        selected_vars = list({'qx_ccc', 'qy_ccc', 'qz_ccc'} | set(selected_vars))

    vorticity_component = 'z'
    if use_vort:
        comp = input("  Vorticity component (x/y/z) [z]: ").strip().lower()
        vorticity_component = comp if comp in ('x', 'y', 'z') else 'z'
        selected_vars = list({'qx_ccc', 'qy_ccc', 'qz_ccc'} | set(selected_vars))

    # ---- Colour-by (iso-surfaces only): colour the surface by a different
    # field than the one that defines its geometry, e.g. fluctuation
    # isosurfaces coloured by vorticity, or Q-criterion coloured by velocity.
    color_by = None
    color_vorticity_component = 'z'
    if mode == 'iso':
        print("\nColour iso-surfaces by a different variable? (leave blank to use the same field)")
        print("  Options: a variable name, 'q' (Q-criterion), 'vort' (Vorticity), "
              "'wall' (distance from the wall)")
        color_choice = input("Colour by [same]: ").strip().lower()
        if color_choice in ('wall', 'wall-distance', 'wall_distance'):
            color_by = 'wall_distance'
        elif color_choice in ('q', 'q-criterion', 'qcriterion'):
            color_by = 'q_criterion'
            selected_vars = list({'qx_ccc', 'qy_ccc', 'qz_ccc'} | set(selected_vars))
        elif color_choice in ('vort', 'vorticity'):
            color_by = 'vorticity'
            comp = input("  Colour vorticity component (x/y/z) [z]: ").strip().lower()
            color_vorticity_component = comp if comp in ('x', 'y', 'z') else 'z'
            selected_vars = list({'qx_ccc', 'qy_ccc', 'qz_ccc'} | set(selected_vars))
        elif color_choice and color_choice in variables:
            color_by = color_choice
            selected_vars = list({color_choice} | set(selected_vars))
        elif color_choice:
            print(f"  Warning: '{color_choice}' not recognised; colouring by the iso-surface field instead.")

    cmap = choose_colormap('viridis' if mode in ('iso', 'streamlines') else 'RdBu_r')

    # Custom colour scale — leave either bound blank to auto-scale that side
    # from the rendered field's own min/max (same as before).
    vmin = vmax = None
    if input("Custom colour scale? (y/n) [n]: ").strip().lower() == 'y':
        raw_vmin = input("  vmin [auto]: ").strip()
        raw_vmax = input("  vmax [auto]: ").strip()
        vmin = float(raw_vmin) if raw_vmin else None
        vmax = float(raw_vmax) if raw_vmax else None

    if mode in ('streamlines', 'glyphs'):
        selected_vars = list({'qx_ccc', 'qy_ccc', 'qz_ccc'} | set(selected_vars))

    cfg = {
        'mode': mode,
        'variable': variable,
        'selected_vars': selected_vars,
        'use_q_criterion': use_q,
        'use_fluc': use_fluc,
        't_avg_xdmf': t_avg_xdmf,
        'use_vorticity': use_vort,
        'vorticity_component': vorticity_component,
        'color_by': color_by,
        'color_vorticity_component': color_vorticity_component,
        'cmap': cmap,
        'vmin': vmin,
        'vmax': vmax,
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

def _resolve_clim(cfg, arr):
    """Colour-scale (vmin, vmax) for add_mesh/add_volume's `clim`.

    Returns None (PyVista auto-scales per mesh, the previous behaviour)
    unless a custom scale was requested; a bound left blank falls back to
    `arr`'s own min/max.
    """
    vmin, vmax = cfg.get('vmin'), cfg.get('vmax')
    if vmin is None and vmax is None:
        return None
    lo = vmin if vmin is not None else float(arr.min())
    hi = vmax if vmax is not None else float(arr.max())
    return (lo, hi)


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
        clim = _resolve_clim(cfg, grid.cell_data[variable])
        n_added = 0
        for normal, pos in planes:
            if pos is None:
                continue
            sl = grid.slice(normal=normal, origin=origins[normal](pos))
            plotter.add_mesh(sl, scalars=variable, cmap=cmap, clim=clim, show_scalar_bar=(n_added == 0))
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
            # cell_data_to_point_data/contour carry every point-data array
            # along, not just the one used for the isovalue — so a different
            # colour_variable (already present on grid_pt) just works here.
            color_variable = cfg.get('color_variable') or variable
            if color_variable != variable:
                print(f"  Colouring iso-surface(s) by {color_variable}...")
            clim = _resolve_clim(cfg, grid.cell_data[color_variable])
            plotter.add_mesh(contours, scalars=color_variable, cmap=cmap, clim=clim, show_scalar_bar=True)

        plotter.add_axes()

    elif mode == 'volume':
        if grid.n_cells > VOLUME_CELL_THRESHOLD:
            print(f"  Error: grid too large for volume rendering ({grid.n_cells:,} "
                  f"cells > {VOLUME_CELL_THRESHOLD:,} limit). Increase the stride and try again.")
            return
        plotter.add_volume(grid, scalars=variable, cmap=cmap,
                           opacity=cfg.get('opacity', 'sigmoid'),
                           clim=_resolve_clim(cfg, grid.cell_data[variable]),
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
            clim = _resolve_clim(cfg, grid_pt.point_data['velocity_magnitude'])
            plotter.add_mesh(streamlines, scalars='velocity_magnitude',
                             cmap=cmap, clim=clim, line_width=2, show_scalar_bar=True)
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
            clim = _resolve_clim(cfg, vel_mag)
            plotter.add_mesh(glyphs, scalars='velocity_magnitude',
                             cmap=cmap, clim=clim, show_scalar_bar=True)
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

    if vis_cfg['mode'] == 'volume':
        predicted = strided_cell_count(grid_info, stride)
        if predicted is not None and predicted > VOLUME_CELL_THRESHOLD:
            print(f"\nError: volume rendering at stride={stride} would need "
                  f"~{predicted:,} cells (limit {VOLUME_CELL_THRESHOLD:,}).")
            print("Increase the stride and try again.")
            return

    print(f"\nLoading {len(vis_cfg['selected_vars'])} variable(s)...")
    data = ut.load_xdmf_variables(var_metadata, vis_cfg['selected_vars'], grid_info=grid_info, stride=stride)
    if not data:
        print("Error: Failed to load data.")
        return

    use_q    = vis_cfg['use_q_criterion']
    use_fluc = vis_cfg['use_fluc']
    use_vort = vis_cfg.get('use_vorticity', False)
    t_avg_xdmf = vis_cfg.get('t_avg_xdmf')
    variable = vis_cfg['variable']
    color_by = vis_cfg.get('color_by')
    color_vorticity_component = vis_cfg.get('color_vorticity_component', 'z')

    t_avg_data = {}
    if use_fluc and t_avg_xdmf:
        print(f"Loading t_avg data from {t_avg_xdmf}...")
        t_avg_meta, _ = ut.parse_xdmf_metadata(t_avg_xdmf)
        t_avg_var = op.INST_TO_TAVG_VAR.get(variable, variable)
        t_avg_data = ut.load_xdmf_variables(t_avg_meta, [t_avg_var], grid_info=grid_info, stride=stride)

    if use_q or use_vort or color_by in ('q_criterion', 'vorticity'):
        # Striding node arrays (len ncells+1) and cell arrays (len ncells) by
        # the same `stride` can land one cell apart (e.g. 1001 nodes -> 334
        # strided nodes -> 333 cells, vs 1000 cells -> 334 strided cells
        # directly) — clip to the node-derived count so coordinates and data
        # line up, same convention build_pyvista_grid uses.
        deriv_grid_info = strided_grid_info(grid_info, stride)
        nz = len(deriv_grid_info['grid_z']) - 1
        ny = len(deriv_grid_info['grid_y']) - 1
        nx = len(deriv_grid_info['grid_x']) - 1
        deriv_data = {k: v[:nz, :ny, :nx] for k, v in data.items()}

    if use_q:
        q = op.compute_q_criterion(deriv_data, deriv_grid_info)
        if q is None:
            return
        data['Q-criterion'] = q
        vis_cfg['variable'] = 'Q-criterion'
    elif use_vort:
        component = vis_cfg.get('vorticity_component', 'z')
        vorticity = op.compute_vorticity(deriv_data, deriv_grid_info, component)
        if vorticity is None:
            return
        vort_name = f'Vorticity_{component}'
        data[vort_name] = vorticity
        vis_cfg['variable'] = vort_name
    elif use_fluc and t_avg_data:
        t_avg_var = op.INST_TO_TAVG_VAR.get(variable, variable)
        fluc_name = f"{variable}'"
        data[fluc_name] = op.compute_inst_fluc(data[variable], t_avg_data[t_avg_var])
        vis_cfg['variable'] = fluc_name

    # Colour-by field (iso-surfaces): a second, independent scalar used only
    # for colouring the extracted surface, not for defining its geometry.
    color_field_name = None
    if color_by == 'q_criterion':
        color_field_name = 'Q-criterion'
        if color_field_name not in data:
            q = op.compute_q_criterion(deriv_data, deriv_grid_info)
            if q is None:
                return
            data[color_field_name] = q
    elif color_by == 'vorticity':
        color_field_name = f'Vorticity_{color_vorticity_component}'
        if color_field_name not in data:
            vorticity = op.compute_vorticity(deriv_data, deriv_grid_info, color_vorticity_component)
            if vorticity is None:
                return
            data[color_field_name] = vorticity
    elif color_by == 'wall_distance':
        color_field_name = WALL_DISTANCE_FIELD
        data[color_field_name] = compute_wall_distance(grid_info, stride)
    elif color_by:
        color_field_name = color_by

    if color_field_name and color_field_name != vis_cfg['variable']:
        vis_cfg['color_variable'] = color_field_name

    print("Building PyVista grid...")
    grid = build_pyvista_grid(grid_info, data, stride=stride)
    print(f"  {grid.dimensions} node dimensions, {grid.n_cells:,} cells")

    print("\nRendering...")
    render_scene(grid, vis_cfg)

    print("Done.")


if __name__ == '__main__':
    main()
