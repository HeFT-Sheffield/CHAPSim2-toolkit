"""3D visualisation: the grid it builds and the scene it renders.

turb_visu was the last module with no coverage, and the easiest place for
a silent error: an off-by-one in the stride, or a transposed array, gives
a picture rather than an exception, and a picture that is wrong in a way
nobody checks.

The geometry and indexing need no display. The render does, and is run
off-screen; it is skipped where pyvista or a GL context is missing, which
on a cluster is normal.
"""

import os
import tempfile

import numpy as np

from _harness import skip


def _pv():
    """pyvista, or a skip."""
    try:
        import pyvista
    except ImportError:
        skip('pyvista is not installed')
    return pyvista


def _visu():
    _pv()
    from chapsim2_toolkit import turb_visu
    return turb_visu


def _grid_info(nx=9, ny=7, nz=5):
    """Node coordinates, as parse_xdmf_metadata returns them."""
    return {
        'grid_x': np.linspace(0.0, 4.0, nx),
        'grid_y': np.linspace(-1.0, 1.0, ny),
        'grid_z': np.linspace(0.0, 2.0, nz),
        'node_dimensions': (nz, ny, nx),
        'cell_dimensions': (nz - 1, ny - 1, nx - 1),
        'coordinate_system': 'cartesian',
    }


def _field(grid_info, stride=1):
    """A (nz, ny, nx) cell array whose value encodes its own index."""
    gi = _visu().strided_grid_info(grid_info, stride)
    nz = len(gi['grid_z']) - 1
    ny = len(gi['grid_y']) - 1
    nx = len(gi['grid_x']) - 1
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx),
                          indexing='ij')
    return (k * 10000 + j * 100 + i).astype(float)


# ---------------------------------------------------------------------------
# Building the grid
# ---------------------------------------------------------------------------

def test_the_grid_has_one_cell_per_data_point():
    visu = _visu()
    info = _grid_info()
    data = {'u': _field(info)}
    grid = visu.build_pyvista_grid(info, data)
    assert grid.n_cells == data['u'].size


def test_the_array_is_not_transposed_on_the_way_in():
    """CHAPSim2 writes (nz, ny, nx) and VTK orders x fastest. Getting this
    wrong still renders - it renders the wrong thing."""
    visu = _visu()
    info = _grid_info(nx=9, ny=7, nz=5)
    data = {'u': _field(info)}
    grid = visu.build_pyvista_grid(info, data)
    flat = grid.cell_data['u']
    nz, ny, nx = 4, 6, 8                      # cells, not nodes
    # VTK cell (i, j, k) is at i + nx*(j + ny*k); the value encodes k,j,i.
    for k, j, i in ((0, 0, 0), (1, 2, 3), (nz - 1, ny - 1, nx - 1)):
        assert flat[i + nx * (j + ny * k)] == k * 10000 + j * 100 + i


def test_the_predicted_cell_count_matches_the_grid_built():
    """strided_cell_count is what warns a user before a huge render; it
    has to agree with what actually gets built."""
    visu = _visu()
    info = _grid_info(nx=17, ny=13, nz=9)
    for stride in (1, 2, 3, 4):
        data = {'u': _field(info, stride)}
        grid = visu.build_pyvista_grid(info, data, stride=stride)
        assert visu.strided_cell_count(info, stride) == grid.n_cells, \
            f'stride {stride}'


def test_striding_keeps_the_domain_bounds():
    """Subsampling decimates the mesh; it must not shrink the domain."""
    visu = _visu()
    info = _grid_info(nx=17, ny=13, nz=9)
    strided = visu.strided_grid_info(info, 4)
    for key in ('grid_x', 'grid_y', 'grid_z'):
        assert strided[key][0] == info[key][0]
    # the far end may be clipped by the stride, but never extended
    for key in ('grid_x', 'grid_y', 'grid_z'):
        assert strided[key][-1] <= info[key][-1]


def test_a_stride_of_one_returns_the_grid_unchanged():
    visu = _visu()
    info = _grid_info()
    assert visu.strided_grid_info(info, 1) is info


# ---------------------------------------------------------------------------
# Wall distance
# ---------------------------------------------------------------------------

def test_the_wall_distance_is_zero_at_both_walls_and_largest_in_the_middle():
    visu = _visu()
    info = _grid_info(ny=21)
    distance = visu.compute_wall_distance(info)
    nz, ny, nx = distance.shape
    column = distance[0, :, 0]
    assert abs(column[0] - column[-1]) < 1e-12        # symmetric
    assert column.argmax() in (ny // 2, ny // 2 - 1)  # peak at the centre
    assert column[0] > 0                              # cell centres, not nodes
    half_height = 0.5 * (info['grid_y'][-1] - info['grid_y'][0])
    assert column.max() < half_height


def test_the_wall_distance_has_the_same_shape_as_the_field():
    visu = _visu()
    info = _grid_info(nx=11, ny=9, nz=7)
    for stride in (1, 2):
        assert visu.compute_wall_distance(info, stride).shape \
            == _field(info, stride).shape


# ---------------------------------------------------------------------------
# Colour limits
# ---------------------------------------------------------------------------

def test_no_custom_limits_lets_pyvista_autoscale():
    visu = _visu()
    assert visu._resolve_clim({}, np.array([0.0, 1.0])) is None


def test_one_blank_limit_falls_back_to_the_data():
    visu = _visu()
    arr = np.array([-2.0, 5.0])
    assert visu._resolve_clim({'vmin': -1.0}, arr) == (-1.0, 5.0)
    assert visu._resolve_clim({'vmax': 1.0}, arr) == (-2.0, 1.0)


def test_both_limits_are_used_as_given():
    visu = _visu()
    arr = np.array([-2.0, 5.0])
    assert visu._resolve_clim({'vmin': 0.0, 'vmax': 1.0}, arr) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# Rendering, off-screen
# ---------------------------------------------------------------------------

def _offscreen():
    pv = _pv()
    try:
        plotter = pv.Plotter(off_screen=True)
        plotter.close()
    except Exception as exc:                  # no GL context on this machine
        skip(f'no off-screen rendering available ({type(exc).__name__})')
    return pv


def test_a_slice_scene_renders_to_an_image():
    """The whole path: grid, slice planes, screenshot."""
    pv = _offscreen()
    visu = _visu()
    info = _grid_info(nx=9, ny=9, nz=9)
    grid = visu.build_pyvista_grid(info, {'u': _field(info)})

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(grid.slice(normal='y'), scalars='u', cmap='RdBu_r')
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'scene.png')
        plotter.screenshot(path)
        plotter.close()
        assert os.path.getsize(path) > 0


def test_a_contour_of_a_linear_field_is_a_plane():
    """A value that varies only in y must contour to a surface spanning
    x and z - a cheap check that the axes are not swapped."""
    _offscreen()
    visu = _visu()
    info = _grid_info(nx=11, ny=11, nz=11)
    y_centres = 0.5 * (info['grid_y'][:-1] + info['grid_y'][1:])
    field = np.broadcast_to(y_centres[None, :, None], (10, 10, 10)).copy()
    grid = visu.build_pyvista_grid(info, {'u': field})

    surface = grid.cell_data_to_point_data().contour([0.0], scalars='u')
    assert surface.n_points > 0
    bounds = surface.bounds
    assert abs(bounds[3] - bounds[2]) < 0.1            # thin in y
    assert bounds[1] - bounds[0] > 1.0                 # spans x
    assert bounds[5] - bounds[4] > 1.0                 # spans z
