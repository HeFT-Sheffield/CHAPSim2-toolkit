"""The GUI, driven headlessly.

gui.py is the largest module in the toolkit and the one with the most
behaviour that only shows up when a real widget tree exists: tabs built
lazily, a case broadcast to every tab, controls filled from the case's own
input file, defaults switched off when the data cannot support them.

None of that was covered. These tests stand a real Tk up against whatever
display is available - xvfb on CI - and drive the parts that do not need a
human: adopting a case, scanning what it offers, building a Config from
the widgets. Plotting is left alone; it is slow, and what it produces is
already checked where the figures are made.

Skipped when ttkbootstrap is missing or there is no display.
"""

import io
import contextlib
import os
import tempfile

from _harness import (skip, require_gui, gui_root, gui_app,
                      solver_tests_dir, build_cartesian_case)


def _gui():
    """The gui module, or a skip.

    Importing it is itself the thing that fails when ttkbootstrap is
    missing, so the check has to come first - otherwise a Python without
    the dependency reports an error where it should report a skip.
    """
    require_gui()
    import gui
    return gui


# ---------------------------------------------------------------------------
# Cases to drive the GUI with
# ---------------------------------------------------------------------------

#: A real solver case with heat transfer and MHD, so the parameters read
#: from its input file are distinctive enough to tell apart from defaults.
MHD_CASE = 'functional/MHD_channel_scp_inout_Tw'


def _solver_case(relative):
    root = solver_tests_dir()
    if root is None:
        skip('CHAPSim2 test output not available (set CHAPSIM2_TESTS)')
    case = os.path.join(root, relative)
    if not os.path.isdir(case):
        skip(f'{relative} is not in this checkout of CHAPSim2')
    return case


def _text(widget):
    return widget.get('1.0', 'end').strip()


@contextlib.contextmanager
def _tab(factory):
    """One tab, parented to a hidden root."""
    with gui_root() as root:
        import tkinter as tk
        yield factory(tk.Frame(root))


# ---------------------------------------------------------------------------
# The window itself
# ---------------------------------------------------------------------------

def test_every_tab_builds():
    """Lazy tabs defer the work; nothing should break when it finally runs."""
    with gui_app() as app:
        for holder in app._holders:
            holder.realise()
            assert holder.inner is not None
        assert len(app._holders) == len(app.App.TABS if hasattr(app, 'App')
                                        else type(app).TABS)


def test_realising_a_tab_twice_does_not_rebuild_it():
    with gui_app() as app:
        holder = app._holders[0]
        first = holder.realise()
        assert holder.realise() is first


def test_closing_restores_the_real_streams():
    """App redirects stdout into the console widget for the window's life."""
    import sys
    before = sys.stdout
    with gui_app():
        assert sys.stdout is not before          # redirected while open
    assert sys.stdout is before                  # and put back on close


def test_the_case_bar_rejects_a_path_that_is_not_a_directory():
    with gui_app() as app:
        app.case_path.set('/definitely/not/here')
        app._apply_case()
        assert 'Not a directory' in app._case_summary.cget('text')


def test_a_case_is_described_by_what_it_actually_holds():
    case = _solver_case(MHD_CASE)
    with gui_app() as app:
        summary = app._describe(case)
        assert os.path.basename(case) in summary
        assert 'input_chapsim.ini' in summary
        assert 'timesteps:' in summary


def test_applying_a_case_reaches_every_tab_without_complaint():
    """_apply_case swallows a tab's exception into a print, so a broken
    adopt_case is invisible unless something reads the console."""
    case = _solver_case(MHD_CASE)
    with gui_app() as app:
        app.case_path.set(case)
        app._apply_case()
        logged = app.console.text.get('1.0', 'end')
        assert 'could not adopt case' not in logged
        adopted = [h.inner for h in app._holders if h.inner is not None]
        assert len(adopted) == len(type(app).TABS)


# ---------------------------------------------------------------------------
# Turbulence Statistics: the run's parameters come from the case
# ---------------------------------------------------------------------------

def test_the_stats_tab_fills_itself_from_the_case_input_file():
    gui = _gui()
    case = _solver_case(MHD_CASE)
    with _tab(gui.TurbStatsTab) as tab:
        tab.adopt_case(case)
        assert float(_text(tab._t_re)) == 5000.0
        assert float(_text(tab._t_ref_temp)) == 645.15
        assert float(_text(tab._t_ref_len)) == 0.0015
        assert tab.vars['geometry'].get() == 'channel'
        assert tab.vars['thermo_on'].get() is True
        assert tab.vars['mhd_on'].get() is True


def test_a_hartmann_number_reaches_the_tab_as_the_stuart_number():
    """Ha = 10 at Re = 5000 is N = 0.02. Nobody should work that out by hand."""
    gui = _gui()
    case = _solver_case(MHD_CASE)
    with _tab(gui.TurbStatsTab) as tab:
        tab.adopt_case(case)
        assert abs(float(_text(tab._t_stuart_number)) - 0.02) < 1e-12
        field = [float(v) for v in _text(tab._t_mag_field_dir).split(',')]
        assert field == [0.0, 10.0, 0.0]


def test_averaging_is_never_switched_on_for_a_direction_with_an_inlet():
    """Averaging x on an inlet/outlet case folds the two ends together."""
    gui = _gui()
    case = _solver_case(MHD_CASE)          # inout: x is not periodic
    with _tab(gui.TurbStatsTab) as tab:
        tab.adopt_case(case)
        assert tab.vars['average_x_direction'].get() is False


def test_a_case_with_no_input_file_leaves_the_controls_alone():
    gui = _gui()
    with tempfile.TemporaryDirectory() as tmp:
        case = os.path.join(tmp, 'bare')
        build_cartesian_case(case)
        with _tab(gui.TurbStatsTab) as tab:
            before = _text(tab._t_re)
            tab.adopt_case(case)
            assert _text(tab._t_re) == before


def test_the_widgets_build_a_config_the_post_processing_accepts():
    gui = _gui()
    import turb_stats as ts
    case = _solver_case(MHD_CASE)
    with _tab(gui.TurbStatsTab) as tab:
        tab.adopt_case(case)
        config = tab._build_config_obj()
        assert isinstance(config, ts.Config)
        assert config.cases == [os.path.basename(case)]
        assert config.folder_path == os.path.dirname(case)
        assert config.Re == [5000.0]
        assert config.geometry == 'channel'


@contextlib.contextmanager
def _dialogs_answer(path):
    """Answer the file dialogs with one path, and make message boxes mute.

    Both kinds are modal: asksaveasfilename waits for a user who is not
    there, and the showinfo at the end of _save_cfg blocks the run outright.
    The message boxes are collected rather than discarded, because an error
    box is how this code reports a failure - a test that ignored them would
    pass while the GUI was telling the user it had not worked.
    """
    gui = _gui()
    files = {name: getattr(gui.filedialog, name)
             for name in ('asksaveasfilename', 'askopenfilename')}
    boxen = {name: getattr(gui.messagebox, name)
             for name in ('showinfo', 'showerror', 'showwarning')}
    seen = []
    for name in files:
        setattr(gui.filedialog, name, lambda **kw: path)
    for name in boxen:
        def record(title, message='', _name=name, **kw):
            seen.append((_name, title, message))
        setattr(gui.messagebox, name, record)
    try:
        yield seen
    finally:
        for name, original in files.items():
            setattr(gui.filedialog, name, original)
        for name, original in boxen.items():
            setattr(gui.messagebox, name, original)


def _no_complaint(boxes):
    bad = [b for b in boxes if b[0] in ('showerror', 'showwarning')]
    assert not bad, f'the GUI reported a problem: {bad}'


def test_settings_survive_a_round_trip_through_a_config_file():
    """Save then load is how a user moves between the GUI and turb_stats.py.

    The file it writes has to be one config.py can be, and reading it back
    has to put the same values on screen - otherwise the two halves of the
    toolkit disagree about a case and neither says so.
    """
    gui = _gui()
    case = _solver_case(MHD_CASE)
    with _tab(gui.TurbStatsTab) as tab:
        tab.adopt_case(case)
        tab._t_re.delete('1.0', 'end')
        tab._t_re.insert('1.0', '1234')
        tab.vars['geometry'].set('pipe')
        tab.vars['tke_on'].set(True)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'config.py')
            with _dialogs_answer(path) as boxes:
                tab._save_cfg()
                _no_complaint(boxes)
                assert os.path.isfile(path), 'nothing was written'

                tab._t_re.delete('1.0', 'end')
                tab._t_re.insert('1.0', '1')
                tab.vars['geometry'].set('channel')
                tab.vars['tke_on'].set(False)

                tab._load_cfg()
                _no_complaint(boxes)

            assert _text(tab._t_re) == '1234'
            assert tab.vars['geometry'].get() == 'pipe'
            assert tab.vars['tke_on'].get() is True


def test_the_config_it_writes_is_one_turb_stats_can_read():
    gui = _gui()
    import turb_stats as ts
    import importlib.util
    case = _solver_case(MHD_CASE)
    with _tab(gui.TurbStatsTab) as tab:
        tab.adopt_case(case)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'config.py')
            with _dialogs_answer(path) as boxes:
                tab._save_cfg()
                _no_complaint(boxes)
            spec = importlib.util.spec_from_file_location('_saved_cfg', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            config = ts.Config.from_module(module)
            assert config.cases == [os.path.basename(case)]
            assert config.Re == [5000.0]


# ---------------------------------------------------------------------------
# Scanning: the tabs report what a case holds, not what they hope it holds
# ---------------------------------------------------------------------------

def test_the_slice_tab_finds_the_timesteps_in_a_case():
    gui = _gui()
    import utils as ut
    case = _solver_case(MHD_CASE)
    with _tab(gui.SliceTab) as tab:
        tab.adopt_case(case)
        offered = list(tab._ts_combo['values'])
        assert offered
        assert set(offered) <= set(ut.find_available_timesteps(case))


def test_the_slice_tab_on_an_empty_folder_offers_nothing_rather_than_stale():
    gui = _gui()
    case = _solver_case(MHD_CASE)
    with _tab(gui.SliceTab) as tab:
        tab.adopt_case(case)
        assert list(tab._ts_combo['values'])
        with tempfile.TemporaryDirectory() as tmp:
            tab.adopt_case(tmp)
            assert not list(tab._ts_combo['values'])


def test_the_3d_tab_finds_the_timesteps_in_a_case():
    gui = _gui()
    case = _solver_case(MHD_CASE)
    with _tab(gui.TurbVisuTab) as tab:
        tab.adopt_case(case)
        assert list(tab._ts_combo['values'])


def test_the_monitor_tab_takes_the_case_folder_not_the_monitor_folder():
    """_run resolves 3_monitor underneath, so the case itself is what it wants."""
    gui = _gui()
    case = _solver_case(MHD_CASE)
    with _tab(gui.MonitorPointsTab) as tab:
        tab.adopt_case(case)
        assert tab._path.get() == case


# ---------------------------------------------------------------------------
# Mesh Analysis
# ---------------------------------------------------------------------------

def test_the_mesh_tab_loads_the_cases_input_file():
    gui = _gui()
    case = _solver_case(MHD_CASE)
    with _tab(gui.MeshAnalysisTab) as tab:
        tab.adopt_case(case)
        assert tab._path.get() == os.path.join(case, 'input_chapsim.ini')
        assert float(tab._ren.get()) == 5000.0


def test_the_mesh_tab_ignores_a_case_with_no_input_file():
    gui = _gui()
    with tempfile.TemporaryDirectory() as tmp:
        with _tab(gui.MeshAnalysisTab) as tab:
            before = tab._path.get()
            tab.adopt_case(tmp)
            assert tab._path.get() == before


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

def test_every_help_topic_renders():
    gui = _gui()
    import help_content
    with _tab(gui.HelpTab) as tab:
        for index, (title, _) in enumerate(help_content.TOPICS):
            tab._show(index)
            shown = tab._text.get('1.0', 'end')
            assert shown.strip(), f'{title} renders empty'


def test_a_help_topic_can_be_opened_by_name():
    """The list title and the body heading differ, so match on the body."""
    gui = _gui()
    import help_content
    wanted = dict(help_content.TOPICS)['Troubleshooting']
    with _tab(gui.HelpTab) as tab:
        tab.show_topic('Troubleshooting')
        assert tab._text.get('1.0', 'end').strip() == wanted.strip()
        index = [t for t, _ in help_content.TOPICS].index('Troubleshooting')
        assert tab._list.curselection() == (index,)


def test_asking_for_a_topic_that_is_not_there_does_not_raise():
    gui = _gui()
    with _tab(gui.HelpTab) as tab:
        tab.show_topic('No such topic')


# ---------------------------------------------------------------------------
# Small widgets
# ---------------------------------------------------------------------------

def test_the_metric_strip_shows_and_clears_values():
    gui = _gui()
    with gui_root() as root:
        import tkinter as tk
        strip = gui.MetricStrip(tk.Frame(root), [('re_tau', 'Re_tau'),
                                                 ('u_tau', 'u_tau')])
        strip.set('re_tau', '180.4')
        assert '180.4' in strip._labels['re_tau'].cget('text')
        strip.clear()
        assert '180.4' not in strip._labels['re_tau'].cget('text')
