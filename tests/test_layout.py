"""Locating a case on disk.

The toolkit stopped reading CHAPSim2 output entirely when the solver moved
its visualisation files into 2_visu/{xdmf,data,mesh}. These pin the
resolution down so a further move is caught here rather than by a user.
"""

import os
import tempfile

import numpy as np

from _harness import build_cartesian_case

import utils as ut


def test_every_case_subdirectory_resolves_to_the_same_case():
    """A user may browse to any of these; all name one case."""
    with tempfile.TemporaryDirectory() as tmp:
        case = build_cartesian_case(tmp)['dirs']['case']
        for entry in ('', '2_visu', '2_visu/xdmf', '2_visu/data', '2_visu/mesh',
                      '1_data', '3_monitor'):
            got = ut.resolve_case_dirs(os.path.join(case, entry))
            assert got['case'] == case, entry
            assert got['xdmf'] == os.path.join(case, '2_visu', 'xdmf'), entry


def test_trailing_separator_is_tolerated():
    with tempfile.TemporaryDirectory() as tmp:
        case = build_cartesian_case(tmp)['dirs']['case']
        assert ut.resolve_case_dirs(case + '/2_visu/data/')['case'] == case


def test_flat_layout_still_resolves():
    """Older runs kept everything directly in 2_visu; those still load."""
    with tempfile.TemporaryDirectory() as tmp:
        case = os.path.join(tmp, 'legacy')
        os.makedirs(os.path.join(case, '2_visu'))
        dirs = ut.resolve_case_dirs(case)
        assert dirs['xdmf'] == dirs['data'] == dirs['mesh'] == dirs['visu']


def test_timesteps_exclude_mesh_descriptors():
    """Grid files are all stamped iteration 0 and carry no field data."""
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cartesian_case(tmp)
        xdmf = built['dirs']['xdmf']
        open(os.path.join(xdmf, 'domain1_grids_3d_0.xdmf'), 'w').close()
        open(os.path.join(xdmf, 'domain1_grid_yi3_0.xdmf'), 'w').close()
        assert ut.find_available_timesteps(built['dirs']['case']) == ['60']


def test_timesteps_sort_numerically():
    with tempfile.TemporaryDirectory() as tmp:
        xdmf = build_cartesian_case(tmp)['dirs']['xdmf']
        for step in (9, 100, 20):
            open(os.path.join(xdmf, f'domain1_flow_{step}.xdmf'), 'w').close()
        steps = ut.find_available_timesteps(os.path.dirname(os.path.dirname(xdmf)))
        assert steps == ['9', '20', '60', '100'], steps


def test_slice_labels_come_from_per_slice_files_and_bundles():
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cartesian_case(tmp)
        # yi3 and zi1 are inside the bundle; this one is a standalone file.
        open(os.path.join(built['dirs']['xdmf'], 'domain1_flow_xi2_60.xdmf'), 'w').close()
        labels = ut.find_available_slices(built['dirs']['case'], '60')
        assert labels == ['xi2', 'yi3', 'zi1'], labels


def test_tsp_avg_planes_are_not_offered_as_slices():
    """zi1 on a tsp_avg file names the averaged-out direction, not a cut."""
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cartesian_case(tmp)
        os.remove(os.path.join(built['dirs']['xdmf'],
                               'domain1_flow_slices_visu_60.xdmf'))
        assert ut.find_available_slices(built['dirs']['case'], '60') == []


def test_group_paths_are_keyed_by_meaning_not_position():
    """A caller after the instantaneous field must never get an averaged one."""
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cartesian_case(tmp)
        parent = os.path.dirname(built['dirs']['case'])
        name = os.path.basename(built['dirs']['case'])
        tiers = ut.group_xdmf_paths(parent, name, '60')
        assert os.path.basename(tiers['inst']) == 'domain1_flow_60.xdmf'
        assert os.path.basename(tiers['tsp_avg']) == 'domain1_tsp_avg_flow_zi1_60.xdmf'
        assert tiers['t_avg'] is None       # this case writes none


def test_slice_axis_info_matches_the_coordinate_system():
    cart = ut.slice_axis_info('xi4')
    assert cart['plane'] == 'yz'
    assert cart['axis_labels'] == ('$y$', '$z$')
    cyl = ut.slice_axis_info('xi4', 'cylindrical')
    assert cyl['plane'] == 'yz'
    assert cyl['axis_labels'] == ('$r$', r'$\theta$')
    assert ut.slice_axis_info('not-a-slice') is None


def test_plot_aspect_is_equal_only_between_two_lengths():
    """Equal aspect on (r, theta) would set one radian to one length unit."""
    assert ut.plot_aspect(('$x$', '$z$')) == 'equal'
    assert ut.plot_aspect(('$x$', '$r$')) == 'equal'
    assert ut.plot_aspect(('$r$', r'$\theta$')) == 'auto'
