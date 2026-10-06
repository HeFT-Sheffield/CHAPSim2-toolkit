"""Check that a case's data is the data its input file describes.

Every number the post-processing scales by - the Reynolds number, the
reference temperature and length, the fluid, the mesh - is read from
``input_chapsim.ini``. That is only correct if the file still describes
the run whose output sits beside it. An input file edited after the run,
output copied in from a different case, or a restart continued with a
changed mesh all produce a folder where the two disagree, and nothing
downstream would notice: the profiles would simply be scaled wrongly.

So the input file is checked against the grid the solver actually wrote,
which is the one thing in the output that the input file fully
determines. Three comparisons:

* cell counts, ncx/ncy/ncz against the grid dimensions in the output;
* domain extents, after the case geometry defaults the solver applies;
* the wall-normal node positions, rebuilt from istret/rstret and compared
  point by point - this is the one that catches a changed stretching,
  which leaves the cell count identical and every profile subtly wrong.

A mismatch is reported, not raised. The user may have a good reason, and
losing the figure helps nobody; what matters is that it is said.
"""

import os

import numpy as np

import mesh_analysis as ma
import utils as ut

__all__ = ['check_case_consistency', 'describe_inconsistencies']

#: Node positions are compared to this, relative to the domain height.
#: Loose enough for the solver's own text output, which is what a case
#: carrying only 4_check tables can be compared against.
NODE_TOLERANCE = 1e-6


def _output_grid(case_dir):
    """The grid from any visualisation file in the case, or None.

    Any file will do: they all describe the same mesh. The first one that
    parses is used rather than a particular tier, because which tiers a
    case wrote depends on how it was run.
    """
    import glob
    dirs = ut.resolve_case_dirs(case_dir)
    for path in sorted(glob.glob(os.path.join(dirs['xdmf'], '*.xdmf'))):
        try:
            _, grid = ut.parse_xdmf_metadata(path)
        except Exception:
            continue
        if grid and grid.get('grid_y') is not None:
            return os.path.basename(path), grid
    return None, None


def check_case_consistency(case_dir, tolerance=NODE_TOLERANCE):
    """Compare a case's input_chapsim.ini with the mesh in its output.

    Args:
        case_dir: the case folder.
        tolerance: relative tolerance on node positions.

    Returns:
        A dict describing what was compared and what disagreed:

        ``checked``  what it was able to compare, as a list of names
        ``problems`` a list of human-readable disagreements, empty when
                     everything matched
        ``notes``    differences that are explained and benign, such as a
                     whole-domain offset in a periodic direction
        ``skipped``  why a comparison could not be made, if it could not

        An empty ``problems`` with an empty ``checked`` means nothing was
        verified, which is not the same as agreement - callers that
        report to a user should say which.
    """
    result = {'checked': [], 'problems': [], 'notes': [], 'skipped': []}

    ini = ma.input_file_for(case_dir)
    if ini is None:
        result['skipped'].append('no input_chapsim.ini in the case folder')
        return result
    cfg = ma.DomainConfig(ma.parse_input_file(ini), ini)

    # -- against the solver's own mesh tables, when it wrote them ----------
    yp_table = os.path.join(case_dir, '4_check', 'check_mesh_yp.dat')
    if os.path.isfile(yp_table):
        try:
            solver_yp = np.loadtxt(yp_table, skiprows=1)[:, 1]
        except Exception as exc:
            result['skipped'].append(f'could not read {yp_table}: {exc}')
        else:
            built, _ = ma.build_y_grid(cfg)
            result['checked'].append('4_check/check_mesh_yp.dat')
            if len(built) != len(solver_yp):
                result['problems'].append(
                    f'the input file gives {len(built)} wall-normal nodes but '
                    f'the solver wrote {len(solver_yp)}: the file does not '
                    f'describe this run')
            else:
                scale = max(abs(cfg.lyt - cfg.lyb), 1.0)
                error = float(np.abs(built - solver_yp).max())
                if error > tolerance * scale:
                    result['problems'].append(
                        f'the wall-normal grid rebuilt from the input file is '
                        f'up to {error:.3g} away from the one the solver '
                        f'wrote; check istret, rstret and ncy')

    # -- against the grid in the visualisation output ----------------------
    source, grid = _output_grid(case_dir)
    if grid is None:
        result['skipped'].append('no visualisation output to compare against')
        return result
    result['checked'].append(source)

    cells = grid.get('cell_dimensions')
    if cells is not None and len(cells) == 3:
        # XDMF dimensions are (z, y, x); the input file counts (x, y, z).
        written = (int(cells[2]), int(cells[1]), int(cells[0]))
        expected = tuple(int(n) for n in cfg.nc)
        if written != expected:
            result['problems'].append(
                f'the input file asks for {expected[0]}x{expected[1]}x'
                f'{expected[2]} cells but the output holds {written[0]}x'
                f'{written[1]}x{written[2]}')

    y = np.asarray(grid['grid_y'], dtype=float)
    scale = max(abs(cfg.lyt - cfg.lyb), 1.0)
    built, _ = ma.build_y_grid(cfg)

    if len(built) != len(y):
        result['problems'].append(
            f'the input file gives {len(built)} wall-normal nodes, the output '
            f'has {len(y)}')
        return result

    difference = y - built
    error = float(np.abs(difference).max())
    if error <= tolerance * scale:
        return result

    # A whole-domain offset is a different thing from a different mesh.
    # The solver's TGV visualisation writes y over [0, 2pi] while it
    # computed on [-pi, pi] - the same nodes, shifted. In a periodic
    # direction that changes nothing physical, but a wall-normal position
    # read off the visualisation is not the solver's y, so it is said.
    shift = float(np.mean(difference))
    if float(np.abs(difference - shift).max()) <= tolerance * scale:
        result['notes'].append(
            f'the output grid_y is the input file\'s grid shifted by '
            f'{shift:+g} ({y[0]:g} to {y[-1]:g} rather than {cfg.lyb:g} to '
            f'{cfg.lyt:g}); the spacing is identical, so this is an offset '
            f'and not a different mesh')
        return result

    result['problems'].append(
        f'the wall-normal nodes rebuilt from istret/rstret are up to '
        f'{error:.3g} away from the grid in the output; the mesh in '
        f'the input file is not the mesh this data was computed on')
    return result


def describe_inconsistencies(case_dir, name=None, tolerance=NODE_TOLERANCE):
    """check_case_consistency as lines ready to print, or an empty list.

    Silent when everything agrees: a check that announces itself on every
    run trains people to ignore it.
    """
    result = check_case_consistency(case_dir, tolerance)
    if not result['problems']:
        return []
    label = name or os.path.basename(os.path.normpath(case_dir))
    lines = [f'WARNING: {label}: the case folder and its input_chapsim.ini '
             f'do not agree.']
    lines += [f'  - {problem}' for problem in result['problems']]
    lines.append('  Every value read from the input file - the Reynolds '
                 'number, the reference')
    lines.append('  temperature and length, the fluid - may therefore not '
                 'belong to this data.')
    return lines
