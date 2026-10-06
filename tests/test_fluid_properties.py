"""Fluid properties, against the solver rather than against themselves.

Post-processing that uses a different conductivity than the run used is
not measuring that run. So the checks here are all external: the
coefficients are compared against CHAPSim2/src/modules.f90 by parsing it,
and the supercritical water table against the solver's own
4_check/check_ftplist_dim.dat, which is the property list it actually
built. Tests needing the solver skip themselves without it.
"""

import os
import re
import warnings

import numpy as np

from _harness import skip, solver_tests_dir

import fluid_properties as fp


# ---------------------------------------------------------------------------
# Against the solver's source
# ---------------------------------------------------------------------------

def _modules_f90():
    """CHAPSim2/src/modules.f90, or a skip."""
    tests = solver_tests_dir()
    if tests is None:
        skip('CHAPSim2 not available (set CHAPSIM2_TESTS)')
    path = os.path.join(os.path.dirname(tests), 'src', 'modules.f90')
    if not os.path.isfile(path):
        skip(f'{path} not found')
    with open(path) as fh:
        return fh.read()


def _fortran_array(source, name):
    """The numbers from `real(WP), parameter :: name(...) = (/ ... /)`."""
    match = re.search(name + r'\s*\([^)]*\)\s*=\s*\(/(.*?)/\)', source, re.S)
    if match is None:
        match = re.search(name + r'\s*=\s*([0-9eEdD_.+-]+)_WP', source)
        if match is None:
            raise AssertionError(f'{name} not found in modules.f90')
        return [float(match.group(1).replace('_WP', ''))]
    body = match.group(1).replace('&', ' ').replace('\n', ' ')
    return [float(tok.replace('_WP', '').replace('E', 'e'))
            for tok in re.findall(r'-?\s*[0-9][0-9.eEdD+-]*_WP', body)
            for tok in [tok.replace(' ', '')]]


#: toolkit name -> the suffix modules.f90 uses.
_SUFFIX = {'sodium': 'Na', 'lead': 'Pb', 'bismuth': 'Bi', 'lbe': 'LBE',
           'lithium': 'Li', 'flibe': 'FLiBe', 'pbli': 'PbLi'}


def test_the_correlation_coefficients_match_the_solver():
    """Every coefficient, read out of modules.f90 and compared.

    This is the test that catches the solver changing a correlation: the
    toolkit would otherwise go on using the old one and quietly disagree
    with every run made after the change.
    """
    source = _modules_f90()
    mismatches = []
    for name, suffix in _SUFFIX.items():
        ours = fp._COEFFICIENTS[name]
        for key in ('CoD', 'CoK', 'CoCp', 'CoH', 'CoM'):
            # modules.f90 spells bismuth's constants with a capital I.
            for candidate in (f'{key}_{suffix}', f'{key}_{suffix.upper()}'):
                try:
                    theirs = _fortran_array(source, candidate)
                except AssertionError:
                    continue
                break
            else:
                mismatches.append(f'{name}.{key}: not found in modules.f90')
                continue
            if len(theirs) != len(ours[key]) or not np.allclose(
                    theirs, ours[key], rtol=0, atol=0):
                mismatches.append(
                    f'{name}.{key}: solver {theirs} != toolkit {ours[key]}')
    assert not mismatches, 'coefficients have drifted:\n  ' + \
        '\n  '.join(mismatches)


def test_the_melting_and_boiling_points_match_the_solver():
    source = _modules_f90()
    for name, suffix in _SUFFIX.items():
        for key, attr in (('TM0', 'T_melt'), ('TB0', 'T_boil')):
            for candidate in (f'{key}_{suffix}', f'{key}_{suffix.upper()}'):
                match = re.search(candidate + r'\s*=\s*([0-9.eEdD+-]+)_WP',
                                  source)
                if match:
                    break
            assert match, f'{key}_{suffix} not found'
            fluid = fp.get_fluid_properties(name)
            assert getattr(fluid, attr) == float(match.group(1)), \
                f'{name} {attr}'


# ---------------------------------------------------------------------------
# Against the solver's own property list
# ---------------------------------------------------------------------------

def _case_with_ftplist():
    root = solver_tests_dir()
    if root is None:
        skip('CHAPSim2 test output not available (set CHAPSIM2_TESTS)')
    case = os.path.join(root, 'functional', 'MHD_channel_scp_inout_Tw')
    if not os.path.isfile(os.path.join(case, '4_check',
                                       'check_ftplist_dim.dat')):
        skip('no check_ftplist_dim.dat in this checkout')
    return case


def test_supercritical_water_matches_the_table_the_solver_built():
    """check_ftplist_dim.dat is the solver's own dimensional property list.

    It is written with six significant figures, so agreement to 1e-5
    relative is exact agreement.
    """
    case = _case_with_ftplist()
    ref = np.loadtxt(os.path.join(case, '4_check', 'check_ftplist_dim.dat'),
                     skiprows=1)
    water = fp.get_fluid_properties('scp_water', case_dir=case)
    T = ref[:, 1]
    for column, got in ((0, water.enthalpy(T)),
                        (2, water.density_mass(T)),
                        (3, water.viscosity(T)),
                        (4, water.thermal_conductivity(T)),
                        (5, water.electrical_conductivity(T)),
                        (6, water.heat_capacity_p(T)),
                        (7, water.coeff_vol_exp(T))):
        assert np.allclose(got, ref[:, column], rtol=1e-5), \
            f'column {column} disagrees with the solver'


def test_the_table_is_read_from_the_case_not_the_bundled_copy():
    """A case run against a different table must be read against that one."""
    case = _case_with_ftplist()
    water = fp.get_fluid_properties('scp_water', case_dir=case)
    assert os.path.dirname(water.path) == case


def test_the_bundled_table_is_used_when_the_case_has_none(tmpdir=None):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        water = fp.get_fluid_properties('scp_water', case_dir=tmp)
        assert os.path.dirname(water.path) == fp.REFERENCE_DIR


def test_the_bundled_tables_are_the_solver_s_own():
    """A divergent copy would be worse than no copy."""
    root = solver_tests_dir()
    if root is None:
        skip('CHAPSim2 not available (set CHAPSIM2_TESTS)')
    for filename in fp.TABLE_FLUIDS.values():
        theirs = os.path.join(os.path.dirname(root), 'validation',
                              'thermal_properties', filename)
        if not os.path.isfile(theirs):
            continue
        with open(theirs, 'rb') as fh:
            want = fh.read()
        with open(os.path.join(fp.REFERENCE_DIR, filename), 'rb') as fh:
            assert fh.read() == want, f'{filename} differs from the solver'


# ---------------------------------------------------------------------------
# The table model
# ---------------------------------------------------------------------------

def test_interpolation_between_two_rows_is_linear():
    """The solver interpolates linearly; near the pseudo-critical peak the
    choice of interpolation visibly changes Cp, so it has to match."""
    water = fp.get_fluid_properties('scp_water')
    table = water._table
    lo, hi = table[100, 2], table[101, 2]
    mid = 0.5 * (lo + hi)
    expected = 0.5 * (table[100, 6] + table[101, 6])
    assert abs(water.heat_capacity_p(mid) - expected) < 1e-9 * abs(expected)


def test_a_table_value_at_a_tabulated_point_is_that_value():
    water = fp.get_fluid_properties('scp_water')
    row = water._table[2000]
    assert abs(water.density_mass(row[2]) - row[3]) < 1e-12 * row[3]


def test_both_supercritical_fluids_load():
    for name, pressure in (('scp_water', 23.5), ('scp_co2', 8.0)):
        fluid = fp.get_fluid_properties(name)
        assert fluid.pressure == pressure
        assert fluid.T_max > fluid.T_min
        assert len(fluid._table) > 1000


# ---------------------------------------------------------------------------
# Range handling
# ---------------------------------------------------------------------------

def test_outside_the_valid_range_is_nan_not_an_extrapolation():
    """The solver stops the run. Post-processing cannot, so it leaves a gap."""
    sodium = fp.get_fluid_properties('sodium')
    assert np.isnan(sodium.density_mass(300.0))      # below melting
    assert np.isnan(sodium.density_mass(2000.0))     # above boiling
    assert not np.isnan(sodium.density_mass(800.0))


def test_a_range_gap_appears_per_point_not_per_array():
    sodium = fp.get_fluid_properties('sodium')
    values = sodium.viscosity(np.array([300.0, 800.0, 2000.0]))
    assert np.isnan(values[0]) and np.isnan(values[2])
    assert not np.isnan(values[1])


def test_water_outside_its_table_is_nan():
    water = fp.get_fluid_properties('scp_water')
    assert np.isnan(water.density_mass(300.0))
    assert np.isnan(water.density_mass(1500.0))


def test_a_non_physical_correlation_value_becomes_nan_with_a_warning():
    """PbLi's viscosity cubic goes negative at about 859 K, well inside the
    range the solver declares valid. A negative viscosity must not reach a
    Prandtl number."""
    fp._WARNED.clear()
    pbli = fp.get_fluid_properties('pbli')
    assert pbli.viscosity(600.0) > 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert np.isnan(pbli.viscosity(1200.0))
    assert any(issubclass(w.category, fp.PropertyRangeWarning)
               for w in caught), 'the non-physical value was silent'


def test_the_warning_is_raised_once_not_once_per_point():
    fp._WARNED.clear()
    pbli = fp.get_fluid_properties('pbli')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        pbli.viscosity(np.linspace(900.0, 1900.0, 5000))
        pbli.viscosity(np.linspace(900.0, 1900.0, 5000))
    assert len(caught) == 1


# ---------------------------------------------------------------------------
# Every fluid, sanity
# ---------------------------------------------------------------------------

def test_every_fluid_is_usable_just_above_its_lower_bound():
    """Melting point for a metal, the table's first row for a supercritical
    fluid: every fluid has to give real numbers somewhere."""
    for name in fp.FLUIDS:
        fluid = fp.get_fluid_properties(name)
        lo, hi = fluid.T_min, fluid.T_max
        T = np.linspace(lo + 0.01 * (hi - lo), lo + 0.1 * (hi - lo), 25)
        for quantity in ('density_mass', 'thermal_conductivity',
                         'heat_capacity_p', 'viscosity'):
            values = np.asarray(getattr(fluid, quantity)(T))
            assert np.all(values > 0), f'{name}.{quantity} is not positive'
            assert np.all(np.isfinite(values)), f'{name}.{quantity} is not finite'


def test_no_fluid_ever_returns_a_negative_property():
    """Across the whole declared range. Where a correlation breaks down the
    answer must be NaN - a gap - and never a negative density or viscosity
    that would travel into a Reynolds or Prandtl number unnoticed."""
    fp._WARNED.clear()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fp.PropertyRangeWarning)
        for name in fp.FLUIDS:
            fluid = fp.get_fluid_properties(name)
            T = np.linspace(fluid.T_min, fluid.T_max, 500)
            for quantity in ('density_mass', 'thermal_conductivity',
                             'heat_capacity_p', 'viscosity'):
                values = np.asarray(getattr(fluid, quantity)(T))
                finite = values[np.isfinite(values)]
                assert np.all(finite > 0), \
                    f'{name}.{quantity} goes negative instead of NaN'


def test_pbli_is_only_usable_over_part_of_the_range_the_solver_claims():
    """Documents a real limit: the viscosity cubic in modules.f90 crosses
    zero near 859 K, but the solver treats PbLi as valid to TB0 = 1943 K."""
    fp._WARNED.clear()
    pbli = fp.get_fluid_properties('pbli')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fp.PropertyRangeWarning)
        T = np.linspace(pbli.T_min, pbli.T_max, 20000)
        usable = T[np.isfinite(pbli.viscosity(T))]
    assert pbli.T_boil == 1943.0
    assert 855.0 < usable.max() < 865.0


def test_the_prandtl_number_is_mu_cp_over_k():
    for name in fp.FLUIDS:
        fluid = fp.get_fluid_properties(name)
        T = 0.5 * (fluid.T_min + fluid.T_max)
        expected = (fluid.viscosity(T) * fluid.heat_capacity_p(T)
                    / fluid.thermal_conductivity(T))
        got = fluid.prandtl(T)
        if np.isnan(expected):
            assert np.isnan(got)
        else:
            assert abs(got - expected) < 1e-12 * abs(expected)


def test_the_thermal_diffusivity_is_k_over_rho_cp():
    fluid = fp.get_fluid_properties('sodium')
    T = 800.0
    expected = (fluid.thermal_conductivity(T)
                / fluid.density_mass(T) / fluid.heat_capacity_p(T))
    assert abs(fluid.thermal_diffusivity(T) - expected) < 1e-15


def test_the_electrical_conductivity_is_one_as_the_solver_sets_it():
    """sigma_e is unity by construction: MHD cases carry the conductivity
    in the Stuart or Hartmann number instead."""
    for name in ('sodium', 'lithium', 'scp_water'):
        assert fp.get_fluid_properties(name).electrical_conductivity(700.0) == 1.0


def test_lithium_viscosity_is_in_pascal_seconds_like_every_other_fluid():
    """It used to be the one fluid returning micro Pa s from viscosity(),
    a factor of a million out for anything that did not special-case it."""
    lithium = fp.get_fluid_properties('lithium')
    assert 1e-5 < lithium.viscosity(800.0) < 1e-2
    assert abs(lithium.viscosity_uPa_s(800.0)
               - 1e6 * lithium.viscosity(800.0)) < 1e-9


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def test_the_solver_s_own_ifluid_tokens_are_accepted():
    """read_case_parameters hands these straight through."""
    import mesh_analysis as ma
    for index, key in ma.IFLUID_TO_TOOLKIT.items():
        assert fp.get_fluid_properties(key) is not None


def test_the_short_names_the_scripts_have_always_taken_still_work():
    for short, full in (('li', 'lithium'), ('na', 'sodium'), ('pb', 'lead'),
                        ('bi', 'bismuth'), ('lbe', 'lbe'), ('flibe', 'flibe'),
                        ('pbli', 'pbli')):
        assert fp.get_fluid_properties(short).name == full


def test_an_unknown_fluid_says_what_is_available():
    try:
        fp.get_fluid_properties('unobtainium')
    except ValueError as exc:
        assert 'lithium' in str(exc) and 'scp_water' in str(exc)
    else:
        raise AssertionError('an unknown fluid was accepted')


def test_no_fluid_at_all_explains_where_it_should_come_from():
    try:
        fp.get_fluid_properties(None)
    except ValueError as exc:
        assert 'input_chapsim.ini' in str(exc)
    else:
        raise AssertionError('None was accepted as a fluid')


def test_the_deprecated_class_names_still_build_the_right_fluid():
    """thermal_BC_calc and older scripts import these by name."""
    import utils as ut
    assert ut.LiquidSodiumProperties().name == 'sodium'
    assert ut.LiquidLithiumProperties().name == 'lithium'
    assert ut.LiquidPbLiProperties().name == 'pbli'
