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


def _fortran_scalar(source, name):
    """The value of `real(WP), parameter :: name = <literal>_WP`."""
    match = re.search(name + r'\s*=\s*([0-9][0-9.eEdD+-]*)_WP', source)
    if match is None:
        raise AssertionError(f'{name} not found in modules.f90')
    return float(match.group(1).replace('D', 'e').replace('d', 'e'))


def _fortran_array(source, name):
    """The numbers from `real(WP), parameter :: name(...) = (/ ... /)`.

    Entries may be named constants or a quotient of two, as CoM_PbLi's
    `EA_PbLi / RU_GAS` is, so a term is resolved before it is read.
    """
    match = re.search(name + r'\s*\([^)]*\)\s*=\s*\(/(.*?)/\)', source, re.S)
    if match is None:
        return [_fortran_scalar(source, name)]
    body = match.group(1).replace('&', ' ').replace('\n', ' ')
    # Strip trailing comments, then split on the commas between entries.
    body = body.split('!')[0]
    values = []
    for term in body.split(','):
        term = term.strip()
        if not term:
            continue
        literal = re.fullmatch(r'(-?)\s*([0-9][0-9.eEdD+-]*)_WP', term)
        if literal:
            sign = -1.0 if literal.group(1) == '-' else 1.0
            values.append(sign * float(literal.group(2).replace('D', 'e')))
            continue
        quotient = re.fullmatch(r'([A-Za-z_]\w*)\s*/\s*([A-Za-z_]\w*)', term)
        if quotient:
            values.append(_fortran_scalar(source, quotient.group(1))
                          / _fortran_scalar(source, quotient.group(2)))
            continue
        named = re.fullmatch(r'[A-Za-z_]\w*', term)
        if named:
            values.append(_fortran_scalar(source, term))
            continue
        raise AssertionError(f'{name}: cannot read the term {term!r}')
    return values


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
        for key in ('CoD', 'CoK', 'CoCp', 'CoM'):
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


def test_the_enthalpy_coefficients_are_the_integral_of_the_heat_capacity():
    """Both the solver and this derive CoH from CoCp rather than
    transcribing it, so that dH/dT = Cp holds exactly. Transcribing both
    is how they came to disagree: the sodium CoH in modules.f90 was LBE's
    until the solver started deriving it."""
    for name in _SUFFIX:
        fluid = fp.get_fluid_properties(name)
        cp = fp._COEFFICIENTS[name]['CoCp']
        assert fluid.CoH == [-cp[0], 0.0, cp[2], cp[3] / 2.0, cp[4] / 3.0]


def test_dh_dt_is_cp_for_every_fluid():
    """The identity the derivation exists to guarantee, checked numerically
    rather than algebraically - the point is the code, not the algebra."""
    for name in _SUFFIX:
        fluid = fp.get_fluid_properties(name)
        T = np.linspace(fluid.T_min + 10, fluid.T_max - 10, 200)
        step = 1e-4
        slope = (np.asarray(fluid.enthalpy(T + step))
                 - np.asarray(fluid.enthalpy(T - step))) / (2 * step)
        cp = np.asarray(fluid.heat_capacity_p(T))
        assert np.nanmax(np.abs(slope - cp) / np.abs(cp)) < 1e-7, name


def test_the_property_range_is_the_phase_range_unless_a_fit_narrows_it():
    """Phase limits and correlation limits are different questions: a fit
    does not hold over the whole liquid range merely because the material
    is liquid there."""
    for name in _SUFFIX:
        fluid = fp.get_fluid_properties(name)
        assert fluid.T_min >= fluid.T_melt
        assert fluid.T_max <= fluid.T_boil
        if name in fp._COEFFICIENTS and not fp._COEFFICIENTS[name].get('limits'):
            assert fluid.T_min == fluid.T_melt, name
            assert fluid.T_max == fluid.T_boil, name
            assert fluid.T_min_source == 'melting point'
            assert fluid.T_max_source == 'boiling point'


def test_pbli_is_narrowed_by_its_viscosity_correlation():
    """Both ends of PbLi's range are set by the Schulz viscosity fit, not
    by melting and boiling, and the toolkit can say so."""
    source = _modules_f90()
    bounds = {}
    for key in ('TMUmin_PbLi', 'TMUmax_PbLi'):
        match = re.search(key + r'\s*=\s*([0-9.eEdD+-]+)_WP', source)
        assert match, f'{key} not found in modules.f90'
        bounds[key] = float(match.group(1))
    pbli = fp.get_fluid_properties('pbli')
    assert pbli.T_min == bounds['TMUmin_PbLi']
    assert pbli.T_max == bounds['TMUmax_PbLi']
    assert pbli.T_melt < pbli.T_min and pbli.T_max < pbli.T_boil
    assert 'viscosity' in pbli.T_min_source
    assert 'viscosity' in pbli.T_max_source


def test_pbli_viscosity_is_the_arrhenius_form_not_the_cubic():
    """The cubic that stood here crossed zero at 858.996 K, inside the
    range its own source stated - a fit not to use, rather than one to
    range-limit. Schulz 1991 replaces it."""
    source = _modules_f90()
    match = re.search(r'EA_PbLi\s*=\s*([0-9.eEdD+-]+)_WP', source)
    assert match, 'EA_PbLi not found in modules.f90'
    ru = re.search(r'RU_GAS\s*=\s*([0-9.eEdD+-]+)_WP', source)
    assert ru, 'RU_GAS not found in modules.f90'
    assert abs(fp.RU_GAS - float(ru.group(1))) < 1e-12

    pbli = fp.get_fluid_properties('pbli')
    assert fp._COEFFICIENTS['pbli']['m_form'] == 'arrhenius'
    T = np.linspace(pbli.T_min, pbli.T_max, 100)
    expected = 1.87e-4 * np.exp(float(match.group(1)) / fp.RU_GAS / T)
    assert np.allclose(pbli.viscosity(T), expected)
    # Liquid metal viscosities sit around 1e-3 Pa s; the cubic gave
    # 6e-5 and falling at the top of this interval.
    assert np.all(np.asarray(pbli.viscosity(T)) > 1e-3)


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
    for name, pressure in (('scp_water', 23.5e6), ('scp_co2', 8.0e6)):
        fluid = fp.get_fluid_properties(name)
        assert fluid.pressure == pressure       # Pa, not the file's MPa
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


import contextlib


@contextlib.contextmanager
def _fluid_with_a_bad_fit():
    """A fluid whose density correlation goes negative inside its range.

    No shipped correlation does this any more - PbLi's cubic was the last
    and it has been replaced - so the guard that catches it would go
    untested if it were only exercised against real fluids. It is worth
    keeping: it is the thing that stops a negative viscosity or density
    reaching a Reynolds or Prandtl number if a future fit misbehaves.
    """
    name = '_test_bad_fit'
    fp._COEFFICIENTS[name] = dict(
        TM0=400.0, TB0=1000.0, HM0=0.0,
        CoD=[1000.0, -2.0],          # crosses zero at 500 K, inside the range
        CoK=[10.0, 0.0, 0.0],
        CoB=5000.0,
        CoCp=[0.0, 0.0, 1000.0, 0.0, 0.0],
        CoM=[500.0, 1e-4, 0.0], m_form='arrhenius',
    )
    fp._WARNED.clear()
    try:
        yield fp.FunctionProperties(name)
    finally:
        del fp._COEFFICIENTS[name]
        fp._WARNED.clear()


def test_a_non_physical_correlation_value_becomes_nan_with_a_warning():
    """A negative density must not reach a Reynolds number."""
    with _fluid_with_a_bad_fit() as fluid:
        assert fluid.density_mass(450.0) > 0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            assert np.isnan(fluid.density_mass(600.0))
        assert any(issubclass(w.category, fp.PropertyRangeWarning)
                   for w in caught), 'the non-physical value was silent'


def test_the_warning_is_raised_once_not_once_per_point():
    """A 5000-point profile should say it once, not 5000 times."""
    with _fluid_with_a_bad_fit() as fluid:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fluid.density_mass(np.linspace(600.0, 900.0, 5000))
            fluid.density_mass(np.linspace(600.0, 900.0, 5000))
        assert len(caught) == 1


def test_no_shipped_correlation_trips_the_guard():
    """The guard exists for a future bad fit, not a present one."""
    fp._WARNED.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for name in fp.FLUIDS:
            fluid = fp.get_fluid_properties(name)
            T = np.linspace(fluid.T_min, fluid.T_max, 1000)
            for quantity in ('density_mass', 'viscosity',
                             'thermal_conductivity', 'heat_capacity_p'):
                getattr(fluid, quantity)(T)
    assert not [w for w in caught
                if issubclass(w.category, fp.PropertyRangeWarning)]



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


def test_the_whole_declared_range_of_every_fluid_is_now_usable():
    """Once PbLi is capped at its fit's root, no fluid has a region inside
    its own range where a property comes back NaN. The guard stays, but it
    should no longer fire on any shipped correlation."""
    fp._WARNED.clear()
    for name in fp.FLUIDS:
        fluid = fp.get_fluid_properties(name)
        T = np.linspace(fluid.T_min, fluid.T_max, 2000)
        for quantity in ('density_mass', 'viscosity', 'thermal_conductivity',
                         'heat_capacity_p'):
            values = np.asarray(getattr(fluid, quantity)(T))
            assert np.all(np.isfinite(values)), \
                f'{name}.{quantity} has gaps inside its own range'


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


def test_every_property_is_si():
    """Viscosity in Pa s, not micro Pa s; pressure in Pa, not MPa. Lithium
    used to be the one fluid returning micro Pa s from viscosity(), a
    factor of a million out for anything that did not special-case it, and
    the table pressure was left in the MPa the file states it in."""
    for name in fp.FLUIDS:
        fluid = fp.get_fluid_properties(name)
        # Near the lower bound, where every fluid is valid - PbLi's
        # viscosity cubic has already gone NaN by the midpoint.
        T = fluid.T_min + 0.05 * (fluid.T_max - fluid.T_min)
        # liquid metals ~1e-4, supercritical water ~1e-5: all Pa s
        assert 1e-6 < fluid.viscosity(T) < 1e-1, f'{name} viscosity'
        assert 1e2 < fluid.density_mass(T) < 2e4, f'{name} density'
        assert 0.1 < fluid.thermal_conductivity(T) < 1e3, f'{name} k'
        assert 1e2 < fluid.heat_capacity_p(T) < 1e6, f'{name} cp'
        assert not hasattr(fluid, 'viscosity_uPa_s'), \
            f'{name} still offers a non-SI accessor'
    assert fp.get_fluid_properties('scp_water').pressure == 23.5e6
    assert fp.get_fluid_properties('scp_co2').pressure == 8.0e6


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


def test_enthalpy_inverts_back_to_the_temperature_it_came_from():
    """compute_heat_transfer_coeff goes the other way, from the bulk
    enthalpy in the data to a bulk temperature, so the two directions have
    to agree."""
    for name in fp.FLUIDS:
        fluid = fp.get_fluid_properties(name)
        T = np.linspace(fluid.T_min + 1, fluid.T_max - 1, 50)
        back = fluid.temperature_from_enthalpy(fluid.enthalpy(T))
        # The correlation fluids go through the solver's own 1024-point
        # table, so they carry its discretisation error and no more.
        assert np.nanmax(np.abs(back - T)) < 1e-2, name


def test_an_enthalpy_outside_the_range_gives_nan_not_an_edge_value():
    water = fp.get_fluid_properties('scp_water')
    assert np.isnan(water.temperature_from_enthalpy(1.0))
    assert np.isnan(water.temperature_from_enthalpy(1e12))


def test_a_non_dimensional_enthalpy_is_dimensionalised_first():
    """The solver writes h as (h - h0)/(T0 cp0); that is what arrives from
    the data, and ref_temp is how the caller says so."""
    water = fp.get_fluid_properties('scp_water')
    T0 = 645.15
    T = 700.0
    h0 = water.enthalpy(T0)
    cp0 = water.heat_capacity_p(T0)
    undim = (water.enthalpy(T) - h0) / (T0 * cp0)
    assert abs(water.temperature_from_enthalpy(undim, ref_temp=T0) - T) < 1e-6


def test_the_deprecated_class_names_still_build_the_right_fluid():
    """thermal_BC_calc and older scripts import these by name."""
    import utils as ut
    assert ut.LiquidSodiumProperties().name == 'sodium'
    assert ut.LiquidLithiumProperties().name == 'lithium'
    assert ut.LiquidPbLiProperties().name == 'pbli'
