"""Read CHAPSim2's own regression output.

The synthetic cases pin the formats as this toolkit understands them.
These read what the solver actually wrote, which is what catches the
formats changing underneath us - the failure that started all of this.

Skipped when CHAPSim2 is not checked out alongside; point CHAPSIM2_TESTS
at its tests directory to run them elsewhere.
"""

import glob
import os

import numpy as np

from _harness import skip, solver_tests_dir

import mesh_analysis as ma
import operations as op
import utils as ut


def _cases():
    root = solver_tests_dir()
    if root is None:
        return []
    return sorted(os.path.dirname(p)
                  for p in glob.glob(os.path.join(root, '*', '*', '2_visu')))


def _require_cases():
    cases = _cases()
    if not cases:
        skip('CHAPSim2 test output not available (set CHAPSIM2_TESTS)')
    return cases


def test_every_case_is_recognised():
    """Layout, timesteps and coordinate system, for every case the solver
    ships - Cartesian and cylindrical, isothermal and heated, periodic and
    with inlets."""
    failures = []
    for case in _require_cases():
        dirs = ut.resolve_case_dirs(case)
        if not os.path.isdir(dirs['xdmf']):
            failures.append(f'{os.path.basename(case)}: no xdmf directory')
            continue
        if not ut.find_available_timesteps(case):
            failures.append(f'{os.path.basename(case)}: no timesteps found')
    assert not failures, failures


def test_every_field_file_loads():
    """Open one field from every tier each case offers. A format change
    shows up here as an empty read."""
    failures = []
    for case in _require_cases():
        name = os.path.basename(case)
        parent = os.path.dirname(case)
        steps = ut.find_available_timesteps(case)
        if not steps:
            continue
        for step in (steps[-1],):
            for group in ut.VISU_GROUPS:
                tiers = ut.group_xdmf_paths(parent, name, step, group=group)
                for tier, path in tiers.items():
                    if path is None:
                        continue
                    arrays, grid = ut.parse_xdmf_file(path, load_all_vars=True)
                    if not arrays:
                        failures.append(f'{name} {group}/{tier}: no arrays')
                        continue
                    sample = next(iter(arrays.values()))
                    if not np.isfinite(np.asarray(sample)).all():
                        failures.append(f'{name} {group}/{tier}: non-finite values')
                    if not grid.get('coordinate_system'):
                        failures.append(f'{name} {group}/{tier}: no coordinate system')
    assert not failures, failures


def test_slice_bundles_expose_every_slice():
    """Each sub-grid must survive as its own slice, with its own data."""
    failures = []
    for case in _require_cases():
        name = os.path.basename(case)
        xdmf = ut.resolve_case_dirs(case)['xdmf']
        for bundle in sorted(glob.glob(os.path.join(xdmf, '*_slices_visu_*.xdmf'))):
            grids = ut.parse_xdmf_grids(bundle)
            tags = [g['tag'] for g in grids]
            if len(tags) < 2:
                failures.append(f'{name}/{os.path.basename(bundle)}: {len(tags)} grid(s)')
                continue
            if len(set(tags)) != len(tags):
                failures.append(f'{name}/{os.path.basename(bundle)}: duplicate tags {tags}')
            # Every slice must read from its own place in the bundle.
            # Comparing values would not do: at iteration 0 the pressure is
            # uniformly zero, so two slices of it are legitimately identical.
            offsets = {}
            for tag in tags:
                meta, _ = ut.parse_xdmf_metadata(bundle, grid_select=tag)
                var = next(iter(meta), None)
                if var is not None:
                    offsets[tag] = (meta[var]['bin_path'], meta[var]['seek'])
            if len(set(offsets.values())) != len(offsets):
                failures.append(f'{name}/{os.path.basename(bundle)}: '
                                f'slices share a read offset - grids merged?')
    assert not failures, failures


def test_profile_tables_load_where_they_exist():
    failures = []
    found_any = False
    for case in _require_cases():
        bundles = ut.find_profile_bundles(case)
        if not bundles:
            continue
        found_any = True
        for (group, step), path in bundles.items():
            variables, coords = ut.load_tsp_avg_profiles(case, step)
            if not variables:
                failures.append(f'{os.path.basename(case)} {group}@{step}: empty')
                continue
            direction = coords.get('direction', 'y')
            if coords.get(f'{direction}c') is None:
                failures.append(f'{os.path.basename(case)} {group}@{step}: no coordinate')
    assert found_any, 'no profile tables anywhere - has the format changed?'
    assert not failures, failures


def test_wall_normal_grid_matches_what_the_solver_built():
    """mesh_analysis rebuilds the grid; 4_check/check_mesh_yp.dat is the
    grid the solver actually used. They must agree to round-off."""
    root = solver_tests_dir()
    if root is None:
        skip('CHAPSim2 test output not available (set CHAPSIM2_TESTS)')
    checked, failures = 0, []
    for ini in sorted(glob.glob(os.path.join(root, '*', '*', 'input_chapsim.ini'))):
        cfg = ma.DomainConfig(ma.parse_input_file(ini), ini)
        if not cfg.is_wall_bounded:
            continue
        reference = os.path.join(os.path.dirname(ini), '4_check', 'check_mesh_yp.dat')
        if not os.path.isfile(reference):
            continue
        yp, _ = ma.build_y_grid(cfg)
        ref = np.loadtxt(reference, skiprows=1, usecols=(1,))
        if len(ref) != len(yp):
            failures.append(f'{os.path.basename(os.path.dirname(ini))}: '
                            f'{len(yp)} nodes vs {len(ref)}')
            continue
        error = float(np.abs(yp - ref).max())
        if error > 1e-12:
            failures.append(f'{os.path.basename(os.path.dirname(ini))}: '
                            f'max |dy| = {error:.2e}')
        checked += 1
    assert checked, 'no wall-bounded cases with a reference grid were found'
    assert not failures, failures


def test_wall_side_follows_the_geometry_on_real_profiles():
    """A pipe's first cell is the axis; a channel's is a wall."""
    root = solver_tests_dir()
    if root is None:
        skip('CHAPSim2 test output not available (set CHAPSIM2_TESTS)')
    seen, failures = set(), []
    for case in _require_cases():
        name = os.path.basename(case)
        shape = ('pipe' if name.startswith('pipe')
                 else 'annular' if name.startswith('annular')
                 else 'channel' if name.startswith('channel') else None)
        if shape is None:
            continue
        profile = _mean_streamwise_profile(case)
        if profile is None:
            continue
        side = op.wall_side(profile)
        expected = 'upper' if shape == 'pipe' else 'lower'
        if side != expected:
            failures.append(f'{name}: wall_side={side}, expected {expected}')
        seen.add(shape)
    assert 'pipe' in seen and 'channel' in seen, f'coverage too thin: {seen}'
    assert not failures, failures


def _mean_streamwise_profile(case):
    """u1 averaged onto the wall-normal axis, from whatever this case has."""
    parent, name = os.path.dirname(case), os.path.basename(case)
    for step in reversed(ut.find_available_timesteps(case) or []):
        path = ut.group_xdmf_paths(parent, name, step)['tsp_avg']
        if path is None:
            continue
        meta, grid = ut.parse_xdmf_metadata(path)
        if 'tsp_avg_u1' not in meta:
            continue
        data = ut.load_xdmf_variables(meta, ['tsp_avg_u1'], grid, quiet=True)
        profile = np.asarray(data['tsp_avg_u1'])
        while profile.ndim > 1:
            profile = profile.mean(axis=-1)
        return profile
    variables, _ = ut.load_tsp_avg_profiles(
        case, (ut.find_available_timesteps(case) or ['0'])[-1])
    return variables.get('u1')
