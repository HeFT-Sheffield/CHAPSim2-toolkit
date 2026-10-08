"""Physics helpers whose answers depend on the geometry.

The wall shear was taken at the first wall-normal cell regardless. For a
pipe that cell is the axis, where the gradient vanishes, so u_tau came out
several times too small and dragged every normalised profile with it.
"""

import numpy as np

from chapsim2_toolkit import operations as op


def _channel(ny=48):
    """u = 1 - y^2 on y in [-1, 1]: no slip at both ends."""
    y = np.linspace(-1.0, 1.0, ny + 1)
    yc = 0.5 * (y[:-1] + y[1:])
    return yc, 1.0 - yc ** 2


def _pipe(ny=48):
    """u = 1 - r^2 on r in [0, 1]: maximum on the axis, no slip at the wall."""
    r = np.linspace(0.0, 1.0, ny + 1)
    rc = 0.5 * (r[:-1] + r[1:])
    return rc, 1.0 - rc ** 2


def _annulus(ny=48, inner=0.4):
    """No slip at both radii."""
    r = np.linspace(inner, 1.0, ny + 1)
    rc = 0.5 * (r[:-1] + r[1:])
    mid = 0.5 * (inner + 1.0)
    return rc, 1.0 - ((rc - mid) / (0.5 * (1.0 - inner))) ** 2


def test_wall_side_picks_the_wall_not_the_axis():
    assert op.wall_side(_pipe()[1]) == 'upper'


def test_wall_side_keeps_lower_where_both_ends_are_walls():
    """Either end is valid for these, and 'lower' is what earlier results
    were computed with, so they must not move."""
    assert op.wall_side(_channel()[1]) == 'lower'
    assert op.wall_side(_annulus()[1]) == 'lower'


def test_wall_side_tolerates_degenerate_input():
    assert op.wall_side(np.zeros(8)) == 'lower'
    assert op.wall_side(np.array([1.0])) == 'lower'


def test_pipe_wall_shear_is_taken_at_the_wall():
    """Analytic: for u = 1 - r^2, du/dr at r = 1 is -2, so tau_w = 2/Re."""
    rc, u = _pipe(ny=400)
    tau = abs(float(op.compute_wall_shear_stress_from_velocity(u, 5000.0, y_coords=rc)))
    assert np.isclose(tau, 2.0 / 5000.0, rtol=2e-2), tau


def test_taking_the_axis_instead_would_be_wrong_by_a_wide_margin():
    """Guards the sign of the bug: the axis gradient is near zero."""
    rc, u = _pipe(ny=400)
    at_wall = abs(float(op.compute_wall_shear_stress_from_velocity(
        u, 5000.0, y_coords=rc, wall='upper')))
    at_axis = abs(float(op.compute_wall_shear_stress_from_velocity(
        u, 5000.0, y_coords=rc, wall='lower')))
    assert at_wall > 20 * at_axis, (at_wall, at_axis)


def test_channel_wall_shear_is_unchanged_by_the_detection():
    """u = 1 - y^2 gives du/dy = 2 at y = -1."""
    yc, u = _channel(ny=400)
    auto = float(op.compute_wall_shear_stress_from_velocity(u, 5000.0, y_coords=yc))
    forced = float(op.compute_wall_shear_stress_from_velocity(
        u, 5000.0, y_coords=yc, wall='lower'))
    assert auto == forced
    assert np.isclose(abs(auto), 2.0 / 5000.0, rtol=2e-2)


def test_wall_shear_scales_inversely_with_reynolds_number():
    yc, u = _channel()
    a = abs(float(op.compute_wall_shear_stress_from_velocity(u, 1000.0, y_coords=yc)))
    b = abs(float(op.compute_wall_shear_stress_from_velocity(u, 2000.0, y_coords=yc)))
    assert np.isclose(a, 2 * b, rtol=1e-9)


def test_two_cells_are_required():
    try:
        op.compute_wall_shear_stress_from_velocity(np.array([1.0]), 100.0)
    except ValueError:
        return
    raise AssertionError('expected ValueError for a single wall-normal cell')


def test_integral_length_scale_runs_on_this_numpy():
    """np.trapz became np.trapezoid in NumPy 2.0; the toolkit must work on
    both, and this raised AttributeError on NumPy 1.x."""
    sep = np.linspace(0.0, 1.0, 64)
    rho = np.cos(2 * np.pi * sep)[:, None]
    out = op.compute_integral_length_scale(rho, sep)
    assert np.isfinite(out).all()


def test_integral_length_scale_of_a_single_mode():
    """A single spanwise mode has an integral scale of 1/k."""
    k = 2.0 * np.pi
    sep = np.linspace(0.0, 1.0 / 4.0, 2048)       # to the first zero crossing
    rho = np.cos(k * sep)[:, None]
    scale = float(op.compute_integral_length_scale(rho, sep)[0])
    assert np.isclose(scale, 1.0 / k, rtol=1e-3), (scale, 1.0 / k)


def test_pressure_terms_accept_isothermal_input():
    """These indexed tke_comp_dict['f'] directly and raised KeyError for a
    case with no density field, although they already handle f is None."""
    n = 5
    d = {'press_velocity_fluc_grad_tensor': np.zeros((3, 3, n)),
         'pressure_strain_tensor': np.zeros((3, 3, n))}
    assert 'pressure_transport' in op.compute_pressure_transport(d, 'uu11')
    assert 'pressure_strain' in op.compute_pressure_strain(d, 'uu11')


def test_interpolate_wall_point_extrapolates_to_each_end():
    yc, u = _channel(ny=200)
    lower = float(op.interpolate_wall_point(u, y_coords=yc, wall='lower'))
    upper = float(op.interpolate_wall_point(u, y_coords=yc, wall='upper'))
    assert abs(lower) < 1e-3 and abs(upper) < 1e-3     # no slip at both walls


# ---------------------------------------------------------------------------
# Half-domain plots measure from a wall. They used y + 1 and 1 - y, which
# holds only for a channel's walls at y = +-1: a pipe came out from 1 at its
# axis to 2 at its wall, and an annulus started at 1 + r_inner.
# ---------------------------------------------------------------------------

def test_the_walls_are_where_the_grid_ends():
    for make, expected in ((_channel, (-1.0, 1.0)), (_annulus, (0.4, 1.0))):
        y = make()[0]
        assert np.allclose(op.wall_positions(y, 'channel'), expected)


def test_a_pipe_has_no_lower_wall():
    lower, upper = op.wall_positions(_pipe()[0], 'pipe')
    assert lower is None and np.isclose(upper, 1.0)


def test_grid_nodes_place_the_walls_exactly_on_a_stretched_grid():
    nodes = np.tanh(2.0 * np.linspace(-1.0, 1.0, 33)) / np.tanh(2.0)
    yc = 0.5 * (nodes[:-1] + nodes[1:])
    assert op.wall_positions(yc, 'channel', nodes) == (-1.0, 1.0)


def test_half_a_channel_starts_at_its_wall_on_either_side():
    yc = _channel()[0]
    walls = op.wall_positions(yc, 'channel')
    lower = op.wall_distance(yc, 'lower', walls)
    upper = op.wall_distance(yc, 'upper', walls)
    assert np.allclose(lower, upper)               # symmetric grid
    assert np.isclose(lower[0], 0.5 * (yc[1] - yc[0]))
    assert np.isclose(lower[-1], 1.0 + yc[len(yc) - len(yc) // 2 - 1])   # last cell kept


def test_half_a_pipe_is_its_whole_radius_from_the_wall():
    rc = _pipe()[0]
    side = op.half_domain_side('lower', 'pipe')
    assert side == 'wall'
    distance = op.wall_distance(rc, side, op.wall_positions(rc, 'pipe'))
    assert np.allclose(distance, 1.0 - rc[::-1])
    assert distance[0] > 0 and np.all(np.diff(distance) > 0)
    profile = op.apply_half_channel(_pipe()[1], side)
    assert len(profile) == len(distance) and abs(profile[0]) < 0.05   # wall first


def test_half_an_annulus_is_measured_from_the_inner_or_outer_wall():
    rc = _annulus()[0]
    walls = op.wall_positions(rc, 'annulus')
    assert np.isclose(op.wall_distance(rc, 'lower', walls)[0], rc[0] - 0.4)
    assert np.isclose(op.wall_distance(rc, 'upper', walls)[0], 1.0 - rc[-1])


def test_the_full_domain_measures_to_the_nearer_wall():
    yc = _channel()[0]
    distance = op.wall_distance(yc, None, op.wall_positions(yc, 'channel'))
    assert np.allclose(distance, 1.0 - np.abs(yc))
    rc = _pipe()[0]
    assert np.allclose(op.wall_distance(rc, None, op.wall_positions(rc, 'pipe')), 1.0 - rc)


def test_a_pipe_peak_is_searched_over_the_whole_radius():
    field = np.outer(np.linspace(0.0, 1.0, 10), np.ones(3))   # largest at the wall
    assert np.allclose(op.compute_peak_over_y(field, half='wall'), 1.0)


# ---------------------------------------------------------------------------
# Plot labelling and robustness. These live here rather than in a GUI test
# because they are properties of the statistics, not of any widget.
# ---------------------------------------------------------------------------

def _plotter(**overrides):
    from chapsim2_toolkit import turb_stats as ts

    class _Module:
        pass

    module = _Module()
    module.norm_by_u_tau_sq = True
    module.norm_ux_by_u_tau = True
    for key, value in overrides.items():
        setattr(module, key, value)
    config = ts.Config.from_module(module)
    return ts.TurbulencePlotter(config, ts.PlotConfig(), None)


def test_dimensionless_statistics_are_not_labelled_as_normalised():
    """The value and label exclusion lists were separate and drifted: the
    turbulent Prandtl number was computed unnormalised but captioned
    Pr_t/u_tau^2."""
    from chapsim2_toolkit import turb_stats as ts
    plotter = _plotter()
    for name in ts.NOT_NORMALISED_BY_U_TAU_SQ:
        label = plotter._get_stat_ylabel(name, name)
        assert 'u_\\tau^2' not in label, (name, label)


def test_dimensional_statistics_still_say_they_are_normalised():
    plotter = _plotter()
    assert 'u_\\tau^2' in plotter._get_stat_ylabel('u_prime_sq', 'u_prime_sq')
    assert 'u_\\tau^2' in plotter._get_stat_ylabel('TKE', 'TKE')


def test_a_wall_distance_axis_reads_the_same_for_every_geometry():
    for geometry in ('channel', 'pipe', 'annulus'):
        half = _plotter(geometry=geometry, half_channel_plot=True)
        assert half._get_y_profile_xlabel() == '$y_w$'
        assert half._corr_y_axis_label() == '$y_w$'
    assert _plotter(geometry='channel')._get_y_profile_xlabel() == '$y$'
    assert _plotter(geometry='pipe')._get_y_profile_xlabel() == '$r$'
    assert _plotter(geometry='pipe', norm_y_to_y_plus=True,
                    half_channel_plot=True)._get_y_profile_xlabel() == '$y^+$'


def test_the_lumley_colour_bar_is_a_wall_distance_even_over_the_full_domain():
    assert _plotter()._get_y_profile_xlabel(wall_distance=True) == '$y_w$'


def test_a_pipe_half_domain_ignores_the_side():
    for side in ('lower', 'upper', 'average'):
        plotter = _plotter(geometry='pipe', half_channel_plot=True, half_channel_side=side)
        assert plotter.config.half_side == 'wall'
    assert _plotter(geometry='pipe').config.half_side is None
    assert _plotter(half_channel_plot=True, half_channel_side='upper').config.half_side == 'upper'


def test_plotted_wall_distance_runs_from_the_wall_for_a_pipe():
    rc, u = _pipe()

    class _Loader:
        y_coords = rc

        def get(self, case, quantity, timestep):
            return u

    plotter = _plotter(geometry='pipe', half_channel_plot=True)
    plotter.data_loader = _Loader()
    assert np.allclose(plotter._get_y_plus('c', 't'), 1.0 - rc[::-1])


def test_an_all_nan_series_is_skipped_not_plotted():
    """Pr_t is 0/0 until the temperature fluctuations develop. Drawing it
    raised IndexError inside matplotlib's marker spacing and took out every
    other figure with it."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure

    plotter = _plotter()
    fig = Figure()
    ax = fig.add_subplot(111)
    x = np.linspace(-1.0, 1.0, 80)
    plotter._plot_line(ax, x, np.full_like(x, np.nan), 'all nan', 'C0', marker='o')
    assert len(ax.lines) == 0
    fig.canvas.draw()                       # the crash was at draw time


def test_a_partly_finite_series_is_still_drawn():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure

    plotter = _plotter()
    fig = Figure()
    ax = fig.add_subplot(111)
    x = np.linspace(-1.0, 1.0, 80)
    y = np.full_like(x, np.nan)
    y[30:40] = 1.0
    plotter._plot_line(ax, x, y, 'partly finite', 'C0', marker='o')
    assert len(ax.lines) == 1
    fig.canvas.draw()


# ---------------------------------------------------------------------------
# Which numpy axis holds which direction. The gradient helpers used to decide
# this from the averaging flags, which say what the loader was asked to do,
# not what shape the data arrived in - tsp_avg output is already 2-D with both
# flags off, so the body-force and budget gradients indexed axis 2 of a 2-D
# array and raised AxisError.
# ---------------------------------------------------------------------------

def test_axis_map_for_each_data_shape():
    cases = [
        ((4, 8, 6), False, False, {'x': 2, 'y': 1, 'z': 0}),   # full 3-D
        ((8, 6),    False, False, {'x': 1, 'y': 0, 'z': None}),  # tsp_avg plane
        ((8, 6),    False, True,  {'x': 1, 'y': 0, 'z': None}),  # z averaged here
        ((4, 8),    True,  False, {'x': None, 'y': 1, 'z': 0}),  # x averaged here
        ((8,),      True,  True,  {'x': None, 'y': 0, 'z': None}),
    ]
    for shape, average_x, average_z, expected in cases:
        got = op.field_axes(np.zeros(shape), average_x, average_z)
        assert got == expected, (shape, average_x, average_z, got)


def test_body_forces_accept_a_two_dimensional_plane():
    """tsp_avg is the default input and is 2-D; this raised AxisError."""
    ny, nx = 8, 6
    y = np.linspace(-1.0, 1.0, ny)
    data = {'pr': np.random.rand(ny, nx), 'f': np.ones((ny, nx)),
            'j1': np.zeros((ny, nx)), 'j2': np.zeros((ny, nx)),
            'j3': np.zeros((ny, nx))}
    comp = op.compute_force_components(data, y, average_z=False, average_x=False)
    assert comp['pr_grad'][0].shape == (ny, nx)      # d/dx
    assert comp['pr_grad'][1].shape == (ny, nx)      # d/dy
    assert np.all(comp['pr_grad'][2] == 0.0)         # z averaged away


def test_body_forces_still_accept_a_three_dimensional_field():
    nz, ny, nx = 4, 8, 6
    y = np.linspace(-1.0, 1.0, ny)
    data = {'pr': np.random.rand(nz, ny, nx), 'f': np.ones((nz, ny, nx))}
    comp = op.compute_force_components(data, y, average_z=False, average_x=False)
    assert all(g.shape == (nz, ny, nx) for g in comp['pr_grad'])


def test_the_budget_reports_which_inputs_were_absent():
    """A term built only from missing variables is not zero, it is
    uncomputed, and the caller needs to be able to tell."""
    comp = op.compute_budget_components({'u1': np.zeros((8, 6))},
                                        np.linspace(-1.0, 1.0, 8))
    assert 'dudx11' in comp['_missing']
    assert 'u1' not in comp['_missing']


# ---------------------------------------------------------------------------
# Heat transfer coefficient and Nusselt number
# ---------------------------------------------------------------------------
#
# The Nusselt number took the conductivity at the reference temperature,
# with a comment in the source saying it should be the bulk. For a heated
# supercritical case that is not a small error: k moves by a factor of five
# through the pseudo-critical region, so Nu came out up to 77% wrong, and
# wrongest exactly where the physics is interesting.

from chapsim2_toolkit import utils as ut


def _heated_duct(nx=16, ny=32, T_wall=1.10, h_bulk=0.0, h_slope=4.0e-3):
    """A z-averaged (ny, nx) field with the bulk enthalpy rising along x.

    Everything is non-dimensional, as the solver writes it: T by the
    reference temperature, h by (T0 cp0).
    """
    y = np.linspace(-1.0, 1.0, ny + 1)
    yc = 0.5 * (y[:-1] + y[1:])
    x_ramp = h_bulk + h_slope * np.arange(nx)

    # Uniform mass flux, so the enthalpy-weighted bulk is just the ramp.
    fu = np.ones((ny, nx))
    fuh = np.broadcast_to(x_ramp, (ny, nx)).copy()
    # Temperature: wall value at the ends, parabolic towards the middle.
    profile = T_wall - 0.05 * (1.0 - yc ** 2)
    temp = np.broadcast_to(profile[:, None], (ny, nx)).copy()
    # Cell centres: the fields live at cell centres, so the integration
    # coordinates have to as well.
    return temp, fuh, fu, yc


def test_the_bulk_temperature_comes_back_with_the_coefficient():
    """So that the Nusselt number uses the same one, not its own."""
    temp, fuh, fu, y = _heated_duct()
    water = ut.get_fluid_properties('scp_water')
    T0 = 645.15
    coeff, bulk = op.compute_wall_heat_transfer_coeff(
        1.0, temp, T0, fuh, fu, y_coords=y, fluid=water,
        return_bulk_temperature=True)
    assert np.shape(bulk) == np.shape(coeff)
    # The bulk enthalpy ramp inverted back to a temperature, rising with x.
    assert np.all(np.diff(bulk) > 0)
    assert abs(bulk[0] - T0) < 1e-6          # h = 0 is the reference state


def test_without_asking_the_coefficient_is_returned_alone():
    temp, fuh, fu, y = _heated_duct()
    water = ut.get_fluid_properties('scp_water')
    coeff = op.compute_wall_heat_transfer_coeff(
        1.0, temp, 645.15, fuh, fu, y_coords=y, fluid=water)
    assert np.ndim(coeff) == 1


def test_no_fluid_is_an_error_not_a_silent_fallback_to_lithium():
    """It used to default to lithium, so a water case got a lithium
    conductivity with nothing said."""
    temp, fuh, fu, y = _heated_duct()
    try:
        op.compute_wall_heat_transfer_coeff(
            1.0, temp, 645.15, fuh, fu, y_coords=y, fluid=None)
    except ValueError as exc:
        assert 'fluid' in str(exc)
    else:
        raise AssertionError('a missing fluid was filled in silently')


def test_the_nusselt_number_takes_a_conductivity_profile():
    h = np.array([2.0, 4.0, 6.0])
    k = np.array([1.0, 2.0, 3.0])
    nu = op.compute_wall_Nusselt_number(h, 0.5, {'k': k})
    assert np.allclose(nu, h * 0.5 / k)


def test_a_scalar_conductivity_still_works():
    assert abs(op.compute_wall_Nusselt_number(4.0, 0.5, {'k': 2.0}) - 1.0) < 1e-12


def test_using_the_bulk_conductivity_changes_the_answer_substantially():
    """The reason the fix matters. Across the pseudo-critical region of
    supercritical water, k falls by a factor of five, so a Nusselt number
    formed with k at the reference temperature is out by tens of percent."""
    water = ut.get_fluid_properties('scp_water')
    T0 = 645.15
    k_ref = water.thermal_conductivity(T0)
    # A bulk temperature on the far side of the peak.
    k_bulk = water.thermal_conductivity(700.0)
    assert abs(k_bulk / k_ref - 1.0) > 0.5, \
        'the two conductivities should differ sharply here'


# ---------------------------------------------------------------------------
# The reference Reynolds number
# ---------------------------------------------------------------------------

def test_a_non_integer_reynolds_number_is_not_truncated():
    """compute_u_tau_quantities opened with int(Re), so a Reynolds number
    read from a case as 180.9 became 180 before every u_tau normalisation
    in the toolkit. Small, but wrong and invisible."""
    yc, u = _channel()
    u_tau_exact, _, _ = op.compute_u_tau_quantities(u, 180.9, yc)
    u_tau_floor, _, _ = op.compute_u_tau_quantities(u, 180.0, yc)
    assert u_tau_exact != u_tau_floor
    # 1/sqrt(Re) scaling, so the ratio is the square root of the ratio of Re
    assert abs(u_tau_exact / u_tau_floor - np.sqrt(180.0 / 180.9)) < 1e-12


def test_y_plus_is_not_truncated_either():
    yc, u = _channel()
    exact = op.norm_y_to_y_plus(yc, u, 180.9, yc)
    floor = op.norm_y_to_y_plus(yc, u, 180.0, yc)
    assert not np.allclose(exact, floor)


def test_the_wall_shear_scales_as_one_over_the_reference_reynolds_number():
    """tau_w = (1/Re) du/dy: the viscous term carries 1/ren, which is why
    it is the reference Reynolds number and not the bulk one."""
    yc, u = _channel()
    a = op.compute_wall_shear_stress_from_velocity(u, 1000.0, y_coords=yc)
    b = op.compute_wall_shear_stress_from_velocity(u, 2000.0, y_coords=yc)
    assert abs(a / b - 2.0) < 1e-12


# ---------------------------------------------------------------------------
# Axis labels
# ---------------------------------------------------------------------------
#
# Found on a production case: a Reynolds-stress plot was labelled
# <u'u'>/U_bulk^2/u_tau^2. The data was normalised by u_tau^2 alone - the
# peak was the textbook 7.3, not 6.1 - so the label claimed a
# normalisation that had not been applied, on a figure headed for a paper.

def test_a_stress_normalised_by_u_tau_squared_says_only_that():
    plotter = _plotter()
    label = plotter._get_stat_ylabel('u_prime_sq', "<u'u'>")
    assert 'u_\\tau^2' in label
    assert 'U_{ref}' not in label and 'U_{bulk}' not in label, \
        f'label claims two normalisations: {label}'


def test_every_stress_label_carries_one_denominator():
    plotter = _plotter()
    for name in ('u_prime_sq', 'v_prime_sq', 'w_prime_sq',
                 'u_prime_v_prime', 'TKE'):
        label = plotter._get_stat_ylabel(name, name)
        assert label.count('/') == 1, f'{name}: {label}'


def test_without_u_tau_normalisation_the_reference_velocity_is_named():
    """The solver non-dimensionalises by the reference velocity, not the
    bulk; those differ for anything but a constant-mass-flux run."""
    plotter = _plotter(norm_by_u_tau_sq=False)
    label = plotter._get_stat_ylabel('u_prime_sq', "<u'u'>")
    assert 'U_{ref}' in label
    assert 'u_\\tau' not in label


def test_the_mean_velocity_label_switches_rather_than_accumulates():
    assert '/u_\\tau$' in _plotter(norm_ux_by_u_tau=True)._get_stat_ylabel(
        'ux_velocity', 'Ux')
    assert 'U_{ref}' in _plotter(norm_ux_by_u_tau=False)._get_stat_ylabel(
        'ux_velocity', 'Ux')


def test_quantities_that_are_not_scaled_keep_their_own_label():
    """Pr_t and Nu are dimensionless already; dividing their label by
    u_tau^2 was a bug fixed earlier and must not come back."""
    plotter = _plotter()
    for name in ('turb_prandtl', 'nusselt_number'):
        assert 'u_\\tau' not in plotter._get_stat_ylabel(name, name)
