"""Reading XDMF descriptors and the binaries behind them."""

import os
import tempfile

import numpy as np

from _harness import build_cartesian_case, build_cylindrical_case

import utils as ut


def test_fields_read_back_exactly():
    """Values and axis order both: a transposed read would still have the
    right shape here, so the content is checked element by element."""
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cartesian_case(tmp)
        path = os.path.join(built['dirs']['xdmf'], 'domain1_flow_60.xdmf')
        arrays, grid = ut.parse_xdmf_file(path, load_all_vars=True)
        assert np.array_equal(arrays['qx_ccc'], built['qx'])
        assert np.array_equal(arrays['qy_ccc'], built['qy'])
        assert grid['cell_dimensions'] == built['qx'].shape


def test_second_field_is_read_from_its_own_offset():
    """Fields share one .bin; a missed Seek silently returns the first."""
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cartesian_case(tmp)
        path = os.path.join(built['dirs']['xdmf'], 'domain1_flow_60.xdmf')
        arrays, _ = ut.parse_xdmf_file(path, load_all_vars=True)
        assert not np.array_equal(arrays['qx_ccc'], arrays['qy_ccc'])


def test_grid_coordinates_skip_the_record_header():
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cartesian_case(tmp)
        _, grid = ut.parse_xdmf_metadata(
            os.path.join(built['dirs']['xdmf'], 'domain1_flow_60.xdmf'))
        assert np.allclose(grid['grid_x'], built['x_nodes'])
        assert np.allclose(grid['grid_y'], built['y_nodes'])
        assert np.allclose(grid['grid_z'], built['z_nodes'])


def test_binaries_resolve_after_the_case_is_moved():
    """The XDMF holds relative paths; a copied case must still load."""
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cartesian_case(tmp)
        moved = os.path.join(tmp, 'moved')
        os.rename(built['dirs']['case'], moved)
        arrays, _ = ut.parse_xdmf_file(
            os.path.join(moved, '2_visu', 'xdmf', 'domain1_flow_60.xdmf'),
            load_all_vars=True)
        assert np.array_equal(arrays['qx_ccc'], built['qx'])


def test_slice_bundle_keeps_its_grids_apart():
    """Sub-grids reuse attribute names; merging them loses every slice but
    one, which is what the reader used to do."""
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cartesian_case(tmp)
        bundle = os.path.join(built['dirs']['xdmf'],
                              'domain1_flow_slices_visu_60.xdmf')
        grids = ut.parse_xdmf_grids(bundle)
        assert [g['tag'] for g in grids] == ['yi3', 'zi1']

        for tag, expected, shape in (('yi3', built['slice_yi3'], (4, 6)),
                                     ('zi1', built['slice_zi1'], (8, 6))):
            meta, grid = ut.parse_xdmf_metadata(bundle, grid_select=tag)
            assert grid['grid_tag'] == tag
            data = ut.load_xdmf_variables(meta, ['qx_ccc'], grid, quiet=True)
            assert data['qx_ccc'].shape == shape, tag
            assert np.allclose(data['qx_ccc'], expected), tag


def test_unknown_grid_is_reported_not_guessed():
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cartesian_case(tmp)
        bundle = os.path.join(built['dirs']['xdmf'],
                              'domain1_flow_slices_visu_60.xdmf')
        meta, grid = ut.parse_xdmf_metadata(bundle, grid_select='yi999')
        assert meta == {} and grid == {}


def test_single_grid_file_ignores_a_grid_selection():
    """A per-slice file names its slice in the filename, not the grid, so a
    tag that does not appear inside it must not be treated as a miss."""
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cartesian_case(tmp)
        meta, _ = ut.parse_xdmf_metadata(
            os.path.join(built['dirs']['xdmf'], 'domain1_flow_60.xdmf'),
            grid_select='yi3')
        assert 'qx_ccc' in meta


def test_tsp_avg_plane_squeezes_to_two_dimensions():
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cartesian_case(tmp)
        arrays, _ = ut.parse_xdmf_file(
            os.path.join(built['dirs']['xdmf'],
                         'domain1_tsp_avg_flow_zi1_60.xdmf'), load_all_vars=True)
        assert np.allclose(arrays['tsp_avg_u1'], built['u1_plane'])


def test_averaging_reduces_the_requested_axes():
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cartesian_case(tmp)
        path = os.path.join(built['dirs']['xdmf'], 'domain1_flow_60.xdmf')
        az, _ = ut.parse_xdmf_file(path, load_all_vars=True, average_z=True)
        assert np.allclose(az['qx_ccc'], built['qx'].mean(axis=0))
        both, _ = ut.parse_xdmf_file(path, load_all_vars=True,
                                     average_z=True, average_x=True)
        assert np.allclose(both['qx_ccc'], built['qx'].mean(axis=(0, 2)))


def test_required_vars_filter_loads_only_what_was_asked_for():
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cartesian_case(tmp)
        arrays, _ = ut.parse_xdmf_file(
            os.path.join(built['dirs']['xdmf'], 'domain1_flow_60.xdmf'),
            load_all_vars=False, required_vars={'qx_ccc'})
        assert set(arrays) == {'qx_ccc'}


def test_stride_subsamples_at_read_time():
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cartesian_case(tmp)
        meta, grid = ut.parse_xdmf_metadata(
            os.path.join(built['dirs']['xdmf'], 'domain1_flow_60.xdmf'))
        data = ut.load_xdmf_variables(meta, ['qx_ccc'], grid, stride=2, quiet=True)
        assert np.array_equal(data['qx_ccc'], built['qx'][::2, ::2, ::2])


def test_cylindrical_mesh_yields_axial_radial_azimuthal_axes():
    """Pipe and annulus cases write one XYZ point list; the three
    computational axes have to be recovered from it."""
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cylindrical_case(tmp)
        _, grid = ut.parse_xdmf_metadata(
            os.path.join(built['dirs']['xdmf'], 'domain1_flow_60.xdmf'))
        assert grid['coordinate_system'] == 'cylindrical'
        assert np.allclose(grid['grid_x'], built['x_nodes'])
        assert np.allclose(grid['grid_y'], built['r_nodes'])   # radius
        assert np.allclose(grid['grid_z'], built['theta_nodes'])


def test_azimuth_is_unwrapped_and_read_off_a_finite_radius():
    """theta is undefined on a pipe's axis, so it must not be sampled there,
    and atan2 wrapping must not fold the sweep back on itself."""
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cylindrical_case(tmp, inner=0.0)
        _, grid = ut.parse_xdmf_metadata(
            os.path.join(built['dirs']['xdmf'], 'domain1_flow_60.xdmf'))
        theta = grid['grid_z']
        assert np.all(np.diff(theta) > 0), theta
        assert np.isclose(theta[-1], 2.0 * np.pi)


def test_annulus_radius_starts_at_the_inner_wall():
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cylindrical_case(tmp, name='annulus', inner=0.4)
        _, grid = ut.parse_xdmf_metadata(
            os.path.join(built['dirs']['xdmf'], 'domain1_flow_60.xdmf'))
        assert np.isclose(grid['grid_y'][0], 0.4)
        assert np.isclose(grid['grid_y'][-1], 1.0)


def test_cached_parse_notices_a_rewritten_file():
    """Documents are cached on (path, mtime, size); a regenerated case must
    not serve the old contents."""
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cartesian_case(tmp)
        path = os.path.join(built['dirs']['xdmf'], 'domain1_flow_60.xdmf')
        first, _ = ut.parse_xdmf_file(path, load_all_vars=True)
        assert set(first) == {'qx_ccc', 'qy_ccc'}

        text = open(path).read().replace('Name="qy_ccc"', 'Name="renamed"')
        os.utime(path, (0, 0))                 # force a different mtime
        with open(path, 'w') as fh:
            fh.write(text)
        again, _ = ut.parse_xdmf_file(path, load_all_vars=True)
        assert 'renamed' in again, sorted(again)


def test_missing_file_returns_empty_rather_than_raising():
    arrays, grid = ut.parse_xdmf_file('/nonexistent/domain1_flow_0.xdmf')
    assert arrays == {} and grid == {}
