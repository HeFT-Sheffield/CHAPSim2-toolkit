"""Time-and-space averaged statistics in their two ASCII shapes.

A doubly periodic case reduces these all the way to a 1-D table; the
per-field layout writes one small table per quantity instead. Both have to
land in the same place for the rest of the toolkit.
"""

import os
import tempfile

import numpy as np

from _harness import build_cartesian_case

from chapsim2_toolkit import utils as ut


def test_profile_bundle_columns_are_read_from_the_header():
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cartesian_case(tmp)
        path = os.path.join(built['dirs']['data'],
                            'domain1_tsp_avg_flow_yprofile_60.dat')
        columns, meta = ut.read_profile_bundle(path)
        assert meta['direction'] == 'y'
        assert meta['coordinate'] == 'yc'
        assert meta['npoints'] == len(built['yc'])
        assert np.allclose(columns['yc'], built['yc'])
        assert np.allclose(columns['tsp_avg_u1'], built['prof_u1'])


def test_bundles_are_discovered_by_group_and_timestep():
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cartesian_case(tmp)
        found = ut.find_profile_bundles(built['dirs']['case'])
        assert list(found) == [('flow', '60')]
        assert ut.find_profile_bundles(built['dirs']['case'], '60')
        assert ut.find_profile_bundles(built['dirs']['case'], '20') == {}


def test_loading_strips_the_averaging_prefix():
    """The rest of the toolkit works in base names: u1, not tsp_avg_u1."""
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cartesian_case(tmp)
        variables, coords = ut.load_tsp_avg_profiles(built['dirs']['case'], '60')
        assert set(variables) == {'u1', 'uu11'}
        assert coords['direction'] == 'y'
        assert np.allclose(coords['yc'], built['yc'])
        assert np.allclose(variables['u1'], built['prof_u1'])


def test_prefix_can_be_kept_when_asked():
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cartesian_case(tmp)
        variables, _ = ut.load_tsp_avg_profiles(built['dirs']['case'], '60',
                                                strip_prefix=False)
        assert set(variables) == {'tsp_avg_u1', 'tsp_avg_uu11'}


def test_per_field_tables_are_read_when_there_is_no_bundle():
    """The older layout: one three-column table per quantity in 1_data."""
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cartesian_case(tmp)
        os.remove(os.path.join(built['dirs']['data'],
                               'domain1_tsp_avg_flow_yprofile_60.dat'))
        yc, u1 = built['yc'], built['prof_u1']
        with open(os.path.join(built['dirs']['raw'],
                               'domain1_tsp_avg_u1_60.dat'), 'w') as fh:
            for n, (y, v) in enumerate(zip(yc, u1), start=1):
                fh.write(f'{n:8d} {y:24.16E} {v:24.16E}\n')
        variables, coords = ut.load_tsp_avg_profiles(built['dirs']['case'], '60')
        assert np.allclose(variables['u1'], u1)
        assert np.allclose(coords['yc'], yc)


def test_bundle_wins_over_a_per_field_table_for_the_same_quantity():
    with tempfile.TemporaryDirectory() as tmp:
        built = build_cartesian_case(tmp)
        with open(os.path.join(built['dirs']['raw'],
                               'domain1_tsp_avg_u1_60.dat'), 'w') as fh:
            for n, y in enumerate(built['yc'], start=1):
                fh.write(f'{n:8d} {y:24.16E} {-999.0:24.16E}\n')
        variables, _ = ut.load_tsp_avg_profiles(built['dirs']['case'], '60')
        assert np.allclose(variables['u1'], built['prof_u1'])


def test_a_bundle_without_a_column_header_is_refused_not_guessed():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'domain1_tsp_avg_flow_yprofile_60.dat')
        with open(path, 'w') as fh:
            fh.write('# CHAPSim2 time-and-space averaged profile\n')
            fh.write('1 0.0 1.0\n2 0.5 2.0\n')
        columns, meta = ut.read_profile_bundle(path)
        assert columns == {}


def test_missing_bundle_returns_empty():
    assert ut.read_profile_bundle('/nonexistent/profile.dat') == ({}, {})


def test_visu_bundle_manifest_is_parsed():
    """The .bin manifests describe every field packed into one file."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'domain1_flow_visu_meta_60.dat')
        with open(path, 'w') as fh:
            fh.write('CHAPSim_visu_bundle_v1\ngroup flow\ndomain 1\niter 60\n'
                     'precision_bytes 4\n'
                     'fields name original_file source center dimensions_kji '
                     'precision_bytes offset_bytes nbytes\n'
                     'pr 2_visu/data/domain1_visu_pr_60.bin cell_centered Cell '
                     '64 80 64 4 0 1310720\n'
                     'qx_ccc 2_visu/data/domain1_visu_qx_60.bin x_staggered_to_cell '
                     'Cell 64 80 64 4 1310720 1310720\n')
        fields, meta = ut.read_visu_bundle_meta(path)
        assert meta['group'] == 'flow' and meta['iter'] == '60'
        assert [f['name'] for f in fields] == ['pr', 'qx_ccc']
        assert fields[1]['offset_bytes'] == 1310720
        assert fields[0]['dims'] == (64, 80, 64)
        assert fields[0]['slice'] is None


def test_slice_bundle_manifest_carries_its_tags():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'domain1_flow_slices_visu_meta_60.dat')
        with open(path, 'w') as fh:
            fh.write('CHAPSim_visu_slice_bundle_v1\ngroup flow\niter 60\n'
                     'fields name original_file slice dimensions_kji '
                     'precision_bytes offset_bytes nbytes\n'
                     'pr 2_visu/data/domain1_visu_pr_xi16_60.bin xi16 '
                     '64 80 1 4 0 20480\n')
        fields, _ = ut.read_visu_bundle_meta(path)
        assert fields[0]['slice'] == 'xi16'
        assert fields[0]['dims'] == (64, 80, 1)
