"""Monitor files.

These have gained columns more than once. Reading them by position gave
plots of the wrong quantity under confident labels - bulk temperature that
was really a mass residual - so the headers are parsed instead, and these
tests hold that.
"""

import os
import tempfile

import numpy as np

from _harness import build_cartesian_case, write_monitor_files

import monitor_points as mp


def _monitor(tmp, thermo=True):
    case = build_cartesian_case(tmp)['dirs']['case']
    return write_monitor_files(case, thermo=thermo), case


def test_point_header_names_its_columns():
    with tempfile.TemporaryDirectory() as tmp:
        monitor, _ = _monitor(tmp)
        cols = mp.read_monitor_header(
            os.path.join(monitor, 'domain1_monitor_pt1_flow.dat'))
        assert cols == ['iteration', 't', 'u', 'v', 'w', 'p', 'phi', 'T']


def test_numbered_history_header_is_parsed_in_order():
    with tempfile.TemporaryDirectory() as tmp:
        monitor, _ = _monitor(tmp)
        cols = mp.read_monitor_header(
            os.path.join(monitor, 'domain1_monitor_metrics_history.log'))
        assert cols[0] == 'time'
        assert cols[5] == 'total kinetic energy'
        assert cols[-1] == 'bulk temperature'


def test_all_comment_lines_are_skipped_whatever_the_header_length():
    """The metrics header is 18 lines now and was 2; a fixed skiprows broke."""
    with tempfile.TemporaryDirectory() as tmp:
        monitor, _ = _monitor(tmp, thermo=True)
        data, cols = mp.load_monitor_data(
            os.path.join(monitor, 'domain1_monitor_metrics_history.log'))
        assert data.shape == (6, 16)
        assert len(cols) == 16


def test_time_is_time_and_not_temperature():
    """'t' and 'T' differ only by case. Folding them picked temperature and
    every point plot was drawn against it."""
    with tempfile.TemporaryDirectory() as tmp:
        monitor, _ = _monitor(tmp, thermo=True)
        data, cols = mp.load_monitor_data(
            os.path.join(monitor, 'domain1_monitor_pt1_flow.dat'))
        lookup = mp.column_lookup(cols)
        time = mp.pick(data, lookup, 't', 'time')
        temp = mp.pick(data, lookup, 'T')
        assert time.max() < 1e-3, time          # the synthetic run is 6e-5 long
        assert np.allclose(temp, 570.0)


def test_columns_are_selected_by_name_not_position():
    with tempfile.TemporaryDirectory() as tmp:
        monitor, _ = _monitor(tmp, thermo=True)
        data, cols = mp.load_monitor_data(
            os.path.join(monitor, 'domain1_monitor_metrics_history.log'))
        lookup = mp.column_lookup(cols)
        assert np.allclose(mp.pick(data, lookup, 'total kinetic energy'), 0.61)
        assert np.allclose(mp.pick(data, lookup, 'bulk velocity qx'), 1.002)
        assert np.allclose(mp.pick(data, lookup, 'bulk temperature'), 1.0)


def test_an_isothermal_run_simply_has_fewer_columns():
    with tempfile.TemporaryDirectory() as tmp:
        monitor, _ = _monitor(tmp, thermo=False)
        data, cols = mp.load_monitor_data(
            os.path.join(monitor, 'domain1_monitor_metrics_history.log'))
        assert data.shape[1] == 11
        lookup = mp.column_lookup(cols)
        assert mp.pick(data, lookup, 'bulk temperature') is None
        assert np.allclose(mp.pick(data, lookup, 'total kinetic energy'), 0.61)


def test_prose_header_falls_back_to_the_declared_layout():
    with tempfile.TemporaryDirectory() as tmp:
        monitor, _ = _monitor(tmp)
        data, cols = mp.load_monitor_data(
            os.path.join(monitor, 'domain1_monitor_change_history.log'))
        assert len(cols) == data.shape[1] == len(mp.CHANGE_HISTORY_COLUMNS)
        lookup = mp.column_lookup(cols)
        assert np.allclose(mp.pick(data, lookup, 'kinetic energy change rate'), -1.05e-2)
        assert np.allclose(mp.pick(data, lookup, 'total mass'), 64.0)


def test_short_candidates_are_not_matched_loosely():
    """A bare 'u' must not drift onto 'bulk velocity qx' by substring.

    Loose matching is limited to candidates of three characters or more,
    so a one- or two-letter name only ever matches a column exactly.
    """
    lookup = mp.column_lookup(['time', 'bulk velocity qx'])
    data = np.zeros((2, 2))
    assert mp.pick(data, lookup, 'u') is None
    assert mp.pick(data, lookup, 'qx') is None
    assert mp.pick(data, lookup, 'velocity') is not None     # long enough to be safe
    assert mp.pick(data, lookup, 'bulk velocity qx') is not None   # exact


def test_diverged_rows_are_dropped():
    with tempfile.TemporaryDirectory() as tmp:
        monitor, _ = _monitor(tmp)
        path = os.path.join(monitor, 'domain1_monitor_pt1_flow.dat')
        with open(path, 'a') as fh:
            fh.write(' '.join(['7', '7e-5'] + ['1e30'] * 6) + '\n')
            fh.write(' '.join(['8', '8e-5'] + ['NaN'] * 6) + '\n')
        data, _ = mp.load_monitor_data(path)
        assert data.shape[0] == 6, data.shape


def test_sampling_thins_the_rows():
    with tempfile.TemporaryDirectory() as tmp:
        monitor, _ = _monitor(tmp)
        path = os.path.join(monitor, 'domain1_monitor_pt1_flow.dat')
        full, _ = mp.load_monitor_data(path)
        every_other, _ = mp.load_monitor_data(path, sample=2)
        assert every_other.shape[0] == (full.shape[0] + 1) // 2


def test_monitor_dir_accepts_a_case_or_the_monitor_folder():
    with tempfile.TemporaryDirectory() as tmp:
        monitor, case = _monitor(tmp)
        assert mp.monitor_dir(case) == monitor
        assert mp.monitor_dir(monitor) == monitor


def test_robust_ylim_judges_each_series_separately():
    """Pooled, a steady qx ~ 1 beside qy = 0 and qz ~ 1e-6 reads as the
    outlier, and the axis was clipped with the useful curve left off."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure

    qx = np.full(40, 1.042)
    qy = np.zeros(40)
    qz = np.full(40, -7.3e-6)
    ax = Figure().add_subplot(111)
    ax.plot(qx); ax.plot(qy); ax.plot(qz)
    mp.apply_robust_ylim(ax, qx, qy, qz)
    lo, hi = ax.get_ylim()
    assert lo <= 1.042 <= hi, (lo, hi)


def test_robust_ylim_still_clips_a_real_divergence():
    data = np.concatenate([np.full(60, 1.0), np.array([1e12])])
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    ax = Figure().add_subplot(111)
    mp.apply_robust_ylim(ax, data)
    assert ax.get_ylim()[1] < 1e6
