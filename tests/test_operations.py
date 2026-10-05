"""Physics helpers whose answers depend on the geometry.

The wall shear was taken at the first wall-normal cell regardless. For a
pipe that cell is the axis, where the gradient vanishes, so u_tau came out
several times too small and dragged every normalised profile with it.
"""

import numpy as np

import operations as op


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
