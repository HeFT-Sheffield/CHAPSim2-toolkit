"""Joining two domains into one case.

Both node arrays carry the node at the interface, so joining them whole
described one cell more than there was data for; the XDMF then declared a
grid the reader rejected. Together with a stale argument in its only
caller, this meant the script had never run.
"""

import io
import contextlib
import os
import tempfile

import numpy as np

from _harness import build_cartesian_case

from chapsim2_toolkit import utils as ut


def _stitch(tmp, x_offset=0.0):
    from chapsim2_toolkit.stitch_domains import stitch_domains

    a = build_cartesian_case(tmp, name='dom1')
    b = build_cartesian_case(tmp, name='dom2')
    out = os.path.join(tmp, 'stitched')
    with contextlib.redirect_stdout(io.StringIO()):
        stitch_domains(os.path.join(a['dirs']['xdmf'], 'domain1_flow_60.xdmf'),
                       os.path.join(b['dirs']['xdmf'], 'domain1_flow_60.xdmf'),
                       out, output_prefix='stitched', x_offset=x_offset)
    return a, b, out


def test_the_joined_grid_describes_exactly_the_joined_data():
    with tempfile.TemporaryDirectory() as tmp:
        a, b, out = _stitch(tmp)
        arrays, grid = ut.parse_xdmf_file(
            os.path.join(ut.resolve_case_dirs(out)['xdmf'], 'stitched_flow_60.xdmf'),
            load_all_vars=True)
        qx = arrays['qx_ccc']
        nx = a['qx'].shape[2] + b['qx'].shape[2]
        assert qx.shape == (a['qx'].shape[0], a['qx'].shape[1], nx), qx.shape
        assert len(grid['grid_x']) == nx + 1          # nodes bound cells exactly


def test_each_half_keeps_its_own_values():
    with tempfile.TemporaryDirectory() as tmp:
        a, b, out = _stitch(tmp)
        arrays, _ = ut.parse_xdmf_file(
            os.path.join(ut.resolve_case_dirs(out)['xdmf'], 'stitched_flow_60.xdmf'),
            load_all_vars=True)
        split = a['qx'].shape[2]
        assert np.array_equal(arrays['qx_ccc'][:, :, :split], a['qx'])
        assert np.array_equal(arrays['qx_ccc'][:, :, split:], b['qx'])
        assert np.array_equal(arrays['qy_ccc'][:, :, :split], a['qy'])


def test_the_joined_axis_increases_and_the_others_are_untouched():
    with tempfile.TemporaryDirectory() as tmp:
        a, _, out = _stitch(tmp)
        _, grid = ut.parse_xdmf_file(
            os.path.join(ut.resolve_case_dirs(out)['xdmf'], 'stitched_flow_60.xdmf'),
            load_all_vars=True)
        assert np.all(np.diff(grid['grid_x']) > 0)
        assert np.allclose(grid['grid_y'], a['y_nodes'])
        assert np.allclose(grid['grid_z'], a['z_nodes'])


def test_the_result_is_a_case_the_toolkit_can_open():
    with tempfile.TemporaryDirectory() as tmp:
        _, _, out = _stitch(tmp)
        dirs = ut.resolve_case_dirs(out)
        assert os.path.isdir(dirs['xdmf']) and os.path.isdir(dirs['mesh'])
        assert ut.find_available_timesteps(out) == ['60']


def test_overlapping_domains_are_refused():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _stitch(tmp, x_offset=-10.0)
        except ValueError:
            return
        raise AssertionError('expected ValueError for domains that overlap')
