#!/usr/bin/env python3
"""CHAPSim2 Toolkit GUI"""

import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
import threading
import sys
import os
import glob
import traceback

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imported at module level rather than lazily like the other tools: it pulls in
# nothing beyond numpy, and the mesh tab needs its constants while building its
# widgets.
import mesh_analysis as ma


# =====================================================================================
# Shared utilities
# =====================================================================================

class ScrollableFrame(ttk.Frame):
    """Vertically scrollable frame driven by the mousewheel only (no bar)."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._canvas = ttk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.inner = ttk.Frame(self._canvas)
        self.inner.bind('<Configure>',
                        lambda e: self._canvas.configure(scrollregion=self._canvas.bbox('all')))
        self._canvas.create_window((0, 0), window=self.inner, anchor='nw')
        self._canvas.pack(fill='both', expand=True)
        self._canvas.bind('<Enter>', lambda e: self._bind_wheel())
        self._canvas.bind('<Leave>', lambda e: self._unbind_wheel())
        self._scroll_accum = 0
        self._scroll_job = None
        self._scroll_debounce_ms = 40

    def _bind_wheel(self):
        self._canvas.bind_all('<MouseWheel>', self._scroll)
        self._canvas.bind_all('<Button-4>', self._scroll)
        self._canvas.bind_all('<Button-5>', self._scroll)

    def _unbind_wheel(self):
        self._canvas.unbind_all('<MouseWheel>')
        self._canvas.unbind_all('<Button-4>')
        self._canvas.unbind_all('<Button-5>')

    def _scroll(self, event):
        if event.num == 4:
            self._scroll_accum += -1
        elif event.num == 5:
            self._scroll_accum += 1
        else:
            self._scroll_accum += int(-1 * (event.delta / 120))
        if self._scroll_job is None:
            self._scroll_job = self._canvas.after(self._scroll_debounce_ms, self._apply_scroll)

    def _apply_scroll(self):
        self._canvas.yview_scroll(self._scroll_accum, 'units')
        self._scroll_accum = 0
        self._scroll_job = None


class FigurePanel(ttk.Frame):
    """Embeds a matplotlib Figure with a NavigationToolbar."""

    def __init__(self, parent, placeholder='No plot yet.', **kwargs):
        super().__init__(parent, **kwargs)
        self._placeholder_text = placeholder
        self._placeholder = ttk.Label(self, text=placeholder, anchor='center')
        self._placeholder.pack(expand=True)
        self._canvas = None
        self._toolbar = None

    def show(self, fig):
        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        if self._toolbar is not None:
            self._toolbar.destroy()
            self._toolbar = None
        for w in self.winfo_children():
            w.destroy()
        self._canvas = FigureCanvasTkAgg(fig, master=self)
        self._toolbar = NavigationToolbar2Tk(self._canvas, self)
        self._toolbar.update()
        self._canvas.get_tk_widget().pack(fill='both', expand=True)
        self._canvas.draw()

    def reset(self):
        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        if self._toolbar is not None:
            self._toolbar.destroy()
            self._toolbar = None
        for w in self.winfo_children():
            w.destroy()
        self._placeholder = ttk.Label(self, text=self._placeholder_text, anchor='center')
        self._placeholder.pack(expand=True)


class TextRedirect:
    """Redirect stdout/stderr to a console widget, thread-safely.

    ``widget`` may be a ConsolePanel (preferred: it can also update its
    collapsed-state beacon) or a plain Text widget.
    """

    def __init__(self, widget):
        self._target = widget

    def write(self, msg):
        # Schedule all Tk operations on the main thread — never call Tk from a
        # worker thread. Both ConsolePanel.write and _log_to satisfy this.
        try:
            if isinstance(self._target, ConsolePanel):
                self._target.after(0, self._target.write, msg)
            else:
                self._target.after(0, _log_to, self._target, msg, False)
        except Exception:
            pass

    def flush(self):
        pass


def _make_console(parent, height=7):
    # ttk.ScrolledText wraps an autostyled ttk.Text internally, so this
    # follows the active theme automatically — no manual colours needed.
    w = ttk.ScrolledText(
        parent, height=height, state='disabled',
        font=('Monospace', 8), wrap='word',
    )
    return w


def _log_to(widget, msg, newline=True):
    """Append ``msg`` to a console text widget.

    ``newline=False`` is used by TextRedirect so that partial writes from
    ``print()`` are forwarded verbatim. A missing console (standalone use)
    falls back to the real stdout rather than raising.
    """
    if widget is None:
        print(msg)
        return
    widget.configure(state='normal')
    widget.insert(tk.END, msg + ('\n' if newline else ''))
    widget.see(tk.END)
    widget.configure(state='disabled')


class ConsolePanel(ttk.Frame):
    """Collapsible console docked to the right edge of the main window.

    Collapsed (the default) it is just a slim vertical strip, so it costs
    almost no space. Clicking the strip expands a scrollable log over the
    right-hand side, which is typically empty anyway.

    The Text widget is created once and never destroyed, so a long-lived
    ``sys.stdout`` redirect stays valid across open/close cycles.
    """

    OPEN_WIDTH = 420
    STRIP_WIDTH = 26

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._open = False

        # --- slim strip, always visible (the only collapsed footprint) ---
        self._strip = ttk.Frame(self, width=self.STRIP_WIDTH)
        self._strip.pack_propagate(False)
        self._strip.pack(side='left', fill='y')
        self._toggle = ttk.Button(self._strip, text='◀', width=2,
                                  command=self.toggle)
        self._toggle.pack(side='top', pady=(6, 2))
        # Shows a dot when output arrives while the panel is collapsed, so
        # background progress isn't silently missed.
        self._beacon = ttk.Label(self._strip, text='', anchor='n')
        self._beacon.pack(side='top')

        # --- body, packed only while expanded ---
        self._body = ttk.Frame(self, width=self.OPEN_WIDTH)
        self._body.pack_propagate(False)

        head = ttk.Frame(self._body)
        head.pack(fill='x')
        ttk.Label(head, text='Console').pack(side='left', padx=6, pady=3)
        ttk.Button(head, text='▶', width=2,
                   command=self.hide).pack(side='right', pady=3)
        ttk.Button(head, text='Clear', width=7,
                   command=self.clear).pack(side='right', padx=2, pady=3)

        self.text = _make_console(self._body, height=10)
        self.text.pack(fill='both', expand=True, padx=3, pady=(0, 4))

    # -- visibility ---------------------------------------------------------- #

    @property
    def is_open(self):
        return self._open

    def toggle(self):
        self.hide() if self._open else self.show()

    def show(self):
        if self._open:
            return
        self._body.pack(side='left', fill='both', expand=True)
        self._toggle.configure(text='▶')
        self._beacon.configure(text='')
        self._open = True

    def hide(self):
        if not self._open:
            return
        self._body.pack_forget()
        self._toggle.configure(text='◀')
        self._open = False

    # -- output -------------------------------------------------------------- #

    def write(self, msg, newline=False):
        """Append raw output (e.g. from print()) and flag if collapsed."""
        _log_to(self.text, msg, newline)
        if not self._open:
            self._beacon.configure(text='●')

    def clear(self):
        self.text.configure(state='normal')
        self.text.delete('1.0', tk.END)
        self.text.configure(state='disabled')
        self._beacon.configure(text='')


def find_console(widget):
    """Return the app-level ConsolePanel hosting ``widget``, or None.

    Walks up the widget's master chain, so tabs work whether or not they are
    embedded in an App that provides a console.
    """
    w = widget
    while w is not None:
        panel = getattr(w, 'console', None)
        if isinstance(panel, ConsolePanel):
            return panel
        w = getattr(w, 'master', None)
    return None


class ConsoleConsumer:
    """Mixin giving a tab access to the app-wide ConsolePanel.

    Use as ``class SomeTab(ConsoleConsumer, ttk.Frame)`` and drop any local
    console widget. When no ConsolePanel ancestor exists (standalone use),
    output falls back to stdout.
    """

    @property
    def _console(self):
        """The shared console's Text widget, or None when standalone."""
        panel = find_console(self)
        return panel.text if panel is not None else None

    def _log(self, msg):
        panel = find_console(self)
        if panel is None:
            print(msg)
            return
        # Marshalled to the main thread: _log is called from worker threads.
        self.after(0, panel.write, msg, True)

    def _clear_console(self):
        panel = find_console(self)
        if panel is not None:
            panel.clear()


# =====================================================================================
# Monitor-points helper functions (copied to avoid importing the module which runs
# global-level code at import time)
# =====================================================================================

def _mp_load(file_path, skiprows, max_val=1e5, sample=1):
    try:
        with open(file_path, 'r') as f:
            for _ in range(skiprows):
                f.readline()
            lines = f if sample <= 1 else (
                line for i, line in enumerate(f) if i % sample == 0
            )
            data = np.loadtxt(lines, dtype=np.float64)
    except Exception:
        return np.empty((0, 0))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.size == 0:
        return np.empty((0, 0))
    finite = np.all(np.isfinite(data), axis=1)
    within = (np.all(np.abs(data[:, 1:]) <= max_val, axis=1)
              if data.shape[1] > 1 else np.ones(data.shape[0], dtype=bool))
    return data[finite & within]


def _mp_running_avg(data, window):
    if window <= 1:
        return data.copy()
    n = len(data)
    pad_l = window // 2
    pad_r = window - 1 - pad_l
    padded = np.pad(data.astype(float), (pad_l, pad_r), mode='edge')
    cumsum = np.empty(len(padded) + 1, dtype=float)
    cumsum[0] = 0.0
    np.cumsum(padded, out=cumsum[1:])
    return (cumsum[window:window + n] - cumsum[:n]) / window


def _mp_robust_ylim(data, padding=0.05, max_decades=3.0):
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        return None
    median = np.median(finite)
    mad = np.median(np.abs(finite - median))
    if mad < 1e-15:
        mad = max(abs(median), 1.0) * 0.01
    mask = np.abs(finite - median) <= mad * 10 ** max_decades
    clean = finite[mask]
    if len(clean) == 0 or len(clean) == len(finite):
        return None
    ymin, ymax = np.min(clean), np.max(clean)
    span = ymax - ymin
    return ymin - padding * span, ymax + padding * span


def _mp_apply_ylim(ax, data):
    lim = _mp_robust_ylim(data)
    if lim is not None:
        ax.set_ylim(lim)
        ax.annotate('y-axis clipped', xy=(0.5, 1.0), xycoords='axes fraction',
                    ha='center', va='bottom', fontsize=7, color='red', fontstyle='italic')


def _mp_stats_box(ax, data):
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        return
    txt = (f"mean: {np.mean(finite):.4g}\nstd:  {np.std(finite):.4g}\n"
           f"min:  {np.min(finite):.4g}\nmax:  {np.max(finite):.4g}\n"
           f"med:  {np.median(finite):.4g}")
    ax.text(0.02, 0.05, txt, transform=ax.transAxes, fontsize=7,
            va='bottom', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))


def _mp_plot_avg(ax, t, d, label, color, window):
    ax.plot(t, d, label=label, linewidth=0.8, color=color, rasterized=True)
    if window > 1:
        ax.plot(t, _mp_running_avg(d, window), label=f'{label} (avg)',
                linewidth=1.2, color='black', linestyle='--', alpha=0.6,
                rasterized=True)


# =====================================================================================
# TURB STATS TAB
# =====================================================================================

class TurbStatsTab(ConsoleConsumer, ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent)
        self._figures = {}
        self._build_ui()

    # ------ Layout -------------------------------------------------------------------

    def _build_ui(self):
        pw = ttk.Panedwindow(self, orient='horizontal')
        pw.pack(fill='both', expand=True)

        left = ttk.Frame(pw, width=440)
        left.pack_propagate(False)
        pw.add(left, weight=0)

        right = ttk.Frame(pw)
        pw.add(right, weight=1)

        self._build_config(left)
        self._build_plot(right)

    # ------ Config panel (left) ------------------------------------------------------

    def _build_config(self, parent):
        # Button bar
        bar = ttk.Frame(parent)
        bar.pack(fill='x', padx=5, pady=4)
        ttk.Button(bar, text='Run', command=self._run).pack(side='left', padx=2)
        ttk.Button(bar, text='Load config.py', command=self._load_cfg).pack(side='left', padx=2)
        ttk.Button(bar, text='Save config.py', command=self._save_cfg).pack(side='left', padx=2)

        scroll = ScrollableFrame(parent)
        scroll.pack(fill='both', expand=True, padx=4, pady=2)
        f = scroll.inner

        # ---- widget helpers (closures over f) ----
        self.vars = {}

        def sv(name, default):
            self.vars[name] = tk.StringVar(value=str(default))
            return self.vars[name]

        def bv(name, default):
            self.vars[name] = tk.BooleanVar(value=default)
            return self.vars[name]

        def sec(title):
            lf = ttk.Labelframe(f, text=title, padding=(8, 6))
            lf.pack(fill='x', padx=4, pady=3)
            return lf

        def erow(frame, label, var):
            r = ttk.Frame(frame)
            r.pack(fill='x', pady=1)
            ttk.Label(r, text=label, width=24, anchor='w').pack(side='left')
            ttk.Entry(r, textvariable=var).pack(side='left', fill='x', expand=True)
            return r

        def brow(frame, label, var, label_above=False):
            r = ttk.Frame(frame)
            r.pack(fill='x', pady=1)
            if label_above:
                ttk.Label(r, text=label, anchor='w').pack(side='top', fill='x')
                entry_row = ttk.Frame(r)
                entry_row.pack(side='top', fill='x')
            else:
                ttk.Label(r, text=label, width=24, anchor='w').pack(side='left')
                entry_row = r
            ttk.Entry(entry_row, textvariable=var).pack(side='left', fill='x', expand=True)
            ttk.Button(entry_row, text='…', width=3,
                       command=lambda v=var: v.set(filedialog.askdirectory() or v.get())
                       ).pack(side='left')

        def crow(frame, label, var, values):
            r = ttk.Frame(frame)
            r.pack(fill='x', pady=1)
            ttk.Label(r, text=label, width=24, anchor='w').pack(side='left')
            ttk.Combobox(r, textvariable=var, values=values,
                         state='readonly', width=14).pack(side='left')
            return r

        def chk(frame, label, var, bootstyle=None):
            kwargs = {'bootstyle': bootstyle} if bootstyle else {}
            w = ttk.Checkbutton(frame, text=label, variable=var, **kwargs)
            w.pack(anchor='w', pady=1)
            return w

        def trow(frame, label, height=2, label_above=False):
            """Text widget row; returns the Text widget.

            label_above: put the label on its own line above the text box
            instead of beside it — for labels too long for the fixed
            width=24 label column, which would otherwise be clipped behind
            the box.
            """
            r = ttk.Frame(frame)
            r.pack(fill='x', pady=1)
            if label_above:
                ttk.Label(r, text=label, anchor='w').pack(side='top', fill='x')
                inner = ttk.Frame(r)
                inner.pack(side='top', fill='x', expand=True)
            else:
                ttk.Label(r, text=label, width=24, anchor='nw').pack(side='left', anchor='n')
                inner = ttk.Frame(r)
                inner.pack(side='left', fill='x', expand=True)
            t = ttk.Text(inner, height=height, width=26, font=('TkDefaultFont', 9),
                         relief='flat', borderwidth=0)
            sb = ttk.Scrollbar(inner, orient='vertical', command=t.yview)
            t.configure(yscrollcommand=sb.set)
            t.pack(side='left', fill='x', expand=True)
            sb.pack(side='left', fill='y')
            t.row = r  # outer row frame, for callers that need to show/hide the whole row
            return t

        # ---- Input Data ----
        s = sec('Case Loading')
        brow(s, ' Directory containing case folders:', sv('folder_path', ''), label_above=True)
        self._t_cases = trow(s, ' Case folder names (one per line)', height=3, label_above=True)
        self._t_cases.insert('1.0', 'Tests')
        self._t_timesteps = trow(s, ' Timesteps (one per line)', height=3, label_above=True)
        self._t_timesteps.insert('1.0', '680000')
        crow(s, ' Input format', sv('input_format', 'xdmf'), ['xdmf', 'text'])
        crow(s, ' Data type', sv('xdmf_data_type', 'tsp_avg'), ['tsp_avg', 't_avg', 'inst'])

        s = sec('Isothermal Input Data')
        self._t_re = trow(s, ' Bulk Reynolds no. (one per case if different)', height=2, label_above=True)
        self._t_re.insert('1.0', '5000')
        crow(s, ' Flow forcing', sv('forcing', 'CMF'), ['CMF', 'CPG'])

        # ---- Thermal / MHD ----
        s = sec('Thermal / MHD Input Data')
        thermal_toggle = chk(s, ' Thermal statistics', bv('thermo_on', False), bootstyle='round-toggle')
        self._t_ref_temp = trow(s, ' Ref. temperature (K)', height=2)
        self._t_ref_temp.insert('1.0', '570')
        self._t_ref_len = trow(s, ' Ref. length (m)', height=2)
        self._t_ref_len.insert('1.0', '0.05')
        self._t_ref_ubulk = trow(s, ' Ref. U_bulk (m/s)', height=2)
        self._t_ref_ubulk.insert('1.0', '0.0900625')
        self._t_wall_hf = trow(s, ' Wall heat flux (W/m²)', height=2)
        self._t_wall_hf.insert('1.0', '0.0')
        working_fluid_row = crow(s, ' Working fluid', sv('working_fluid', 'lithium'),
             ['lithium', 'sodium', 'lead', 'bismuth', 'lbe', 'flibe', 'pbli'])
        self._t_gravity_dir = trow(s, ' Gravity direction (x,y,z)', height=2)
        self._t_gravity_dir.insert('1.0', '0, -1, 0')
        mhd_toggle = chk(s, ' MHD statistics', bv('mhd_on', False), bootstyle='round-toggle')
        self._t_mag_field_dir = trow(s, ' Magnetic field dir. (x,y,z)', height=2, label_above=True)
        self._t_mag_field_dir.insert('1.0', '0, 1, 0')
        self._t_stuart_number = trow(s, ' Stuart number (N)', height=2)
        self._t_stuart_number.insert('1.0', '0.0')

        input_thermal_rows = [self._t_ref_temp.row, self._t_ref_len.row, self._t_ref_ubulk.row,
                               self._t_wall_hf.row, working_fluid_row, self._t_gravity_dir.row]
        input_mhd_rows = [self._t_mag_field_dir.row, self._t_stuart_number.row]

        # ---- Averaging ----
        s = sec('Averaging')
        chk(s, ' Average x direction', bv('average_x_direction', False))
        chk(s, ' Average z direction', bv('average_z_direction', True))
        chk(s, ' Average over timesteps', bv('average_over_timesteps', False))

        # ---- Statistics to Compute ----
        s = sec('Profiles')
        ux_row = chk(s, 'u_x velocity', bv('ux_velocity_on', True))
        uy_row = chk(s, 'u_y velocity', bv('uy_velocity_on', False))
        uz_row = chk(s, 'u_z velocity', bv('uz_velocity_on', False))
        temp_row = chk(s, 'Temperature', bv('temp_on', False))
        friction_row = chk(s, 'Friction coefficient', bv('coeff_friction_on', False))
        vort_row = chk(s, 'Vorticity (requires full 3D field)', bv('mean_vorticity_on', False))
        profile_rows = [ux_row, uy_row, uz_row, temp_row, friction_row, vort_row]

        s = sec('Basic Statistics')
        chk(s, 'TKE', bv('tke_on', False))
        chk(s, "u'u' Reynolds stress", bv('u_prime_sq_on', False))
        chk(s, "u'v' Reynolds stress", bv('u_prime_v_prime_on', False))
        chk(s, "v'v' Reynolds stress", bv('v_prime_sq_on', False))
        chk(s, "v'w' Reynolds stress", bv('v_prime_w_prime_on', False))
        chk(s, "w'w' Reynolds stress", bv('w_prime_sq_on', False))
        chk(s, "Vorticity Fluctuation RMS (requires full 3D field)", bv('vorticity_on', False))
        crow(s, 'Vorticity component', sv('vorticity_component', 'z'), ['x', 'y', 'z'])

        s = sec('Advanced Statistics')
        chk(s, "Peak TKE growth", bv('tke_peak_growth_on', False))
        chk(s, 'Reynolds-stress anisotropy (Lumley triangle)', bv('reynolds_anisotropy_on', False))
        chk(s, 'Vorticity anisotropy (Lumley triangle, requires full 3D field)', bv('vorticity_anisotropy_on', False))
        chk(s, 'Reynolds Stress Budget terms', bv('re_stress_budget_on', False))
        crow(s, 'Budget component', sv('re_stress_component', 'uu11'),
             ['total', 'uu11', 'uu12', 'uu22', 'uu33'])
        thermal_stats_section = sec('Thermal Statistics')
        s = thermal_stats_section
        chk(s, 'Wall Heat transfer coeff.', bv('heat_transf_coeff_on', False))
        chk(s, 'Wall Nusselt number', bv('Nusselt_number_on', False))
        chk(s, 'Wall Turbulent Prandtl number', bv('turb_prandtl_on', False))

        mhd_stats_section = sec('MHD Statistics')
        s = mhd_stats_section
        chk(s, 'jx current density (mean)', bv('j1_mean_on', False))
        chk(s, 'jy current density (mean)', bv('j2_mean_on', False))
        chk(s, 'jz current density (mean)', bv('j3_mean_on', False))
        chk(s, "jx' RMS (fluc)", bv('j1_rms_on', False))
        chk(s, "jy' RMS (fluc)", bv('j2_rms_on', False))
        chk(s, "jz' RMS (fluc)", bv('j3_rms_on', False))
        chk(s, 'Lorentz force x (mean)', bv('lorentz_force_x_on', False))
        chk(s, 'Lorentz force y (mean)', bv('lorentz_force_y_on', False))
        chk(s, 'Lorentz force z (mean)', bv('lorentz_force_z_on', False))

        # ---- Profile Options ----
        s = sec('Profile Options')
        crow(s, 'Profile direction', sv('profile_direction', 'y'), ['y', 'x', 'both'])
        erow(s, 'Slice coordinates (x)', sv('slice_coords', ''))
        erow(s, 'Slice coordinates (y)', sv('x_profile_y_coords', ''))
        erow(s, 'Domain crop (x)', sv('x_crop', ''))

        # ---- Normalisation ----
        s = sec('Normalisation')
        norm_utau_row = chk(s, 'Normalise by u_τ²', bv('norm_by_u_tau_sq', True))
        norm_ux_row = chk(s, 'Normalise U_x by u_τ', bv('norm_ux_by_u_tau', True))
        norm_yplus_row = chk(s, 'Normalise y to y⁺', bv('norm_y_to_y_plus', False))
        norm_temp_row = chk(s, 'Normalise T by T_ref', bv('norm_temp_by_ref_temp', False))
        norm_rows = [norm_utau_row, norm_ux_row, norm_yplus_row, norm_temp_row]

        # ---- Plotting ----
        s = sec('Plotting')
        crow(s, 'Domain', sv('channel_plot_mode', 'full channel'),
             ['full channel', 'half channel', 'surface plot'])
        crow(s, 'Half channel side', sv('half_channel_side', 'lower'), ['lower', 'upper', 'average'])
        crow(s, 'Axis scale', sv('axis_scale', 'linear'), ['linear', 'log'])
        chk(s, 'Large text', bv('large_text_on', False))

        # ---- Reference Data ----
        s = sec('Reference Data')
        loglaw_row = chk(s, 'Log-law reference', bv('ux_velocity_log_ref_on', True))
        mhd_nk_row = chk(s, 'MHD NK reference', bv('mhd_NK_ref_on', False))
        nk_hartmann_row = crow(s, 'NK reference Hartmann no.', sv('mhd_NK_ref_case', 'Ha_6'), ['Ha_4', 'Ha_6'])
        mkm180_row = chk(s, 'MKM180 reference', bv('mkm180_ch_ref_on', False))
        ref_rows = [loglaw_row, mhd_nk_row, nk_hartmann_row, mkm180_row]

        # ---- Thermal/MHD-driven visibility --------------------------------------
        # Toggles/rows within each affected section are always fully re-packed in
        # their canonical order (not left partially managed) — pack() otherwise
        # just appends at the end of whichever siblings are currently managed,
        # not back into their original slot. Same reasoning applies one level up:
        # the Thermal/MHD Statistics section boxes themselves are hidden entirely
        # (not just emptied), which needs every top-level section re-packed in
        # order too.
        all_sections = [c for c in f.winfo_children() if isinstance(c, ttk.Labelframe)]

        def _update_thermal_mhd_visibility(*_args):
            thermo_on = self.vars['thermo_on'].get()
            mhd_on = self.vars['mhd_on'].get()

            for sec_frame in all_sections:
                sec_frame.pack_forget()
            for sec_frame in all_sections:
                if sec_frame is thermal_stats_section and not thermo_on:
                    continue
                if sec_frame is mhd_stats_section and not mhd_on:
                    continue
                sec_frame.pack(fill='x', padx=4, pady=3)

            for w in [thermal_toggle] + input_thermal_rows + [mhd_toggle] + input_mhd_rows:
                w.pack_forget()
            thermal_toggle.pack(anchor='w', pady=1)
            if thermo_on:
                for w in input_thermal_rows:
                    w.pack(fill='x', pady=1)
            mhd_toggle.pack(anchor='w', pady=1)
            if mhd_on:
                for w in input_mhd_rows:
                    w.pack(fill='x', pady=1)

            for w in profile_rows:
                w.pack_forget()
            for w in profile_rows:
                if w is temp_row and not thermo_on:
                    continue
                w.pack(anchor='w', pady=1)

            for w in norm_rows:
                w.pack_forget()
            for w in norm_rows:
                if w is norm_temp_row and not thermo_on:
                    continue
                w.pack(anchor='w', pady=1)

            for w in ref_rows:
                w.pack_forget()
            for w in ref_rows:
                if w in (mhd_nk_row, nk_hartmann_row) and not mhd_on:
                    continue
                if w is nk_hartmann_row:
                    w.pack(fill='x', pady=1)
                else:
                    w.pack(anchor='w', pady=1)

        self.vars['thermo_on'].trace_add('write', _update_thermal_mhd_visibility)
        self.vars['mhd_on'].trace_add('write', _update_thermal_mhd_visibility)
        _update_thermal_mhd_visibility()

    # ------ Plot panel (right) -------------------------------------------------------

    def _build_plot(self, parent):
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill='x', padx=5, pady=3)
        ttk.Label(ctrl, text='Figure:').pack(side='left')
        self._fig_var = tk.StringVar()
        self._fig_combo = ttk.Combobox(ctrl, textvariable=self._fig_var,
                                       state='readonly', width=30)
        self._fig_combo.pack(side='left', padx=4)
        self._fig_combo.bind('<<ComboboxSelected>>', self._on_fig_select)

        self._panel = FigurePanel(parent, placeholder='Run the pipeline to generate plots.')
        self._panel.pack(fill='both', expand=True)

    def _on_fig_select(self, _event=None):
        key = self._fig_var.get()
        if key in self._figures:
            self._panel.show(self._figures[key])

    # ------ Helpers ------------------------------------------------------------------

    def _get_text(self, widget):
        return widget.get('1.0', tk.END).strip()

    def _parse_strs(self, text):
        return [s.strip() for s in text.replace(',', '\n').split('\n') if s.strip()]

    def _parse_floats(self, text):
        result = []
        for tok in text.replace(',', '\n').split('\n'):
            tok = tok.strip()
            if tok:
                try:
                    result.append(float(tok))
                except ValueError:
                    pass
        return result

    def _build_config_obj(self):
        from turb_stats import Config
        v = self.vars

        cases = self._parse_strs(self._get_text(self._t_cases)) or ['']
        timesteps = self._parse_strs(self._get_text(self._t_timesteps)) or ['']
        Re = self._parse_floats(self._get_text(self._t_re)) or [1.0]
        ref_temp = self._parse_floats(self._get_text(self._t_ref_temp)) or [300.0]
        ref_length = self._parse_floats(self._get_text(self._t_ref_len)) or [1.0]
        ref_bulk_velocity = self._parse_floats(self._get_text(self._t_ref_ubulk)) or [1.0]
        wall_heat_flux = self._parse_floats(self._get_text(self._t_wall_hf)) or [0.0]
        gravity_direction = (self._parse_floats(self._get_text(self._t_gravity_dir)) + [0.0, 0.0, 0.0])[:3]
        mag_field_direction = (self._parse_floats(self._get_text(self._t_mag_field_dir)) + [0.0, 0.0, 0.0])[:3]
        stuart_number = (self._parse_floats(self._get_text(self._t_stuart_number)) or [0.0])[0]

        return Config(
            folder_path=v['folder_path'].get(),
            input_format=v['input_format'].get(),
            cases=cases,
            timesteps=timesteps,
            thermo_on=v['thermo_on'].get(),
            mhd_on=v['mhd_on'].get(),
            forcing=v['forcing'].get(),
            Re=Re,
            ref_temp=ref_temp,
            ref_length=ref_length,
            ref_bulk_velocity=ref_bulk_velocity,
            wall_heat_flux=wall_heat_flux,
            working_fluid=v['working_fluid'].get(),
            gravity_direction=gravity_direction,
            mag_field_direction=mag_field_direction,
            stuart_number=stuart_number,
            ux_velocity_on=v['ux_velocity_on'].get(),
            uy_velocity_on=v['uy_velocity_on'].get(),
            uz_velocity_on=v['uz_velocity_on'].get(),
            temp_on=v['temp_on'].get(),
            heat_transf_coeff_on=v['heat_transf_coeff_on'].get(),
            Nusselt_number_on=v['Nusselt_number_on'].get(),
            turb_prandtl_on=v['turb_prandtl_on'].get(),
            coeff_friction_on=v['coeff_friction_on'].get(),
            tke_on=v['tke_on'].get(),
            tke_peak_growth_on=v['tke_peak_growth_on'].get(),
            profile_direction=v['profile_direction'].get(),
            slice_coords=v['slice_coords'].get(),
            x_crop=v['x_crop'].get(),
            x_profile_y_coords=v['x_profile_y_coords'].get(),
            surface_plot_on=v['channel_plot_mode'].get() == 'surface plot',
            u_prime_sq_on=v['u_prime_sq_on'].get(),
            u_prime_v_prime_on=v['u_prime_v_prime_on'].get(),
            v_prime_sq_on=v['v_prime_sq_on'].get(),
            v_prime_w_prime_on=v['v_prime_w_prime_on'].get(),
            w_prime_sq_on=v['w_prime_sq_on'].get(),
            j1_mean_on=v['j1_mean_on'].get(),
            j2_mean_on=v['j2_mean_on'].get(),
            j3_mean_on=v['j3_mean_on'].get(),
            j1_rms_on=v['j1_rms_on'].get(),
            j2_rms_on=v['j2_rms_on'].get(),
            j3_rms_on=v['j3_rms_on'].get(),
            lorentz_force_x_on=v['lorentz_force_x_on'].get(),
            lorentz_force_y_on=v['lorentz_force_y_on'].get(),
            lorentz_force_z_on=v['lorentz_force_z_on'].get(),
            re_stress_budget_on=v['re_stress_budget_on'].get(),
            re_stress_component=v['re_stress_component'].get(),
            average_z_direction=v['average_z_direction'].get(),
            average_x_direction=v['average_x_direction'].get(),
            average_over_timesteps=v['average_over_timesteps'].get(),
            norm_by_u_tau_sq=v['norm_by_u_tau_sq'].get(),
            norm_ux_by_u_tau=v['norm_ux_by_u_tau'].get(),
            norm_y_to_y_plus=v['norm_y_to_y_plus'].get(),
            norm_temp_by_ref_temp=v['norm_temp_by_ref_temp'].get(),
            half_channel_plot=v['channel_plot_mode'].get() == 'half channel',
            half_channel_side=v['half_channel_side'].get(),
            linear_y_scale=v['axis_scale'].get() == 'linear',
            log_y_scale=v['axis_scale'].get() == 'log',
            xdmf_data_type=v['xdmf_data_type'].get(),
            display_fig=False,          # always embedded; never plt.show()
            save_fig=True,
            save_to_path=True,
            large_text_on=v['large_text_on'].get(),
            plot_name='',
            ux_velocity_log_ref_on=v['ux_velocity_log_ref_on'].get(),
            mhd_NK_ref_on=v['mhd_NK_ref_on'].get(),
            mhd_NK_ref_case=v['mhd_NK_ref_case'].get(),
            mkm180_ch_ref_on=v['mkm180_ch_ref_on'].get(),
            mean_vorticity_on=v['mean_vorticity_on'].get(),
            vorticity_on=v['vorticity_on'].get(),
            vorticity_component=v['vorticity_component'].get(),
            reynolds_anisotropy_on=v['reynolds_anisotropy_on'].get(),
            vorticity_anisotropy_on=v['vorticity_anisotropy_on'].get(),
        )

    # ------ Run pipeline -------------------------------------------------------------

    def _run(self):
        self._clear_console()

        try:
            config = self._build_config_obj()
        except Exception as exc:
            messagebox.showerror('Config error', str(exc))
            return

        def worker():
            try:
                from turb_stats import (
                    create_data_loader, ReferenceData,
                    TurbulenceStatsPipeline, PlotConfig, TurbulencePlotter,
                )
                print('Loading data…')
                loader = create_data_loader(config)
                loader.load_all()

                print('Loading reference data…')
                ref = ReferenceData(config)
                ref.load_all()

                print('Computing statistics…')
                pipeline = TurbulenceStatsPipeline(config, loader)
                pipeline.compute_all()

                print('Processing…')
                pipeline.process_all()

                print('Generating plots…')
                plot_cfg = PlotConfig()
                plotter = TurbulencePlotter(config, plot_cfg, loader)
                grouped = pipeline.get_statistics_by_class()
                figs = plotter.plot_by_class(grouped, ref)

                spectrum_fig = plotter.plot_spectrum(pipeline.spectrum_computer)
                if spectrum_fig is not None:
                    figs['Spectrum'] = spectrum_fig

                if config.save_fig and figs:
                    plotter.save_figures_by_class(figs)

                self.after(0, lambda: self._update_figures(figs))
                print('Done.')
            except Exception:
                traceback.print_exc()

        threading.Thread(target=worker, daemon=True).start()

    def _update_figures(self, figs):
        self._figures = figs
        keys = list(figs.keys())
        self._fig_combo['values'] = keys
        if keys:
            self._fig_var.set(keys[0])
            self._panel.show(figs[keys[0]])

    # ------ Load / Save config.py ----------------------------------------------------

    def _load_cfg(self):
        path = filedialog.askopenfilename(
            title='Open config.py',
            filetypes=[('Python files', '*.py'), ('All files', '*.*')],
        )
        if not path:
            return
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location('_tmp_cfg', path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            v = self.vars
            str_fields = {
                'folder_path': '', 'input_format': 'xdmf', 'forcing': 'CMF',
                'working_fluid': 'lithium', 'profile_direction': 'y',
                'slice_coords': '', 'x_crop': '', 'x_profile_y_coords': '',
                're_stress_component': 'uu11', 'vorticity_component': 'z',
                'half_channel_side': 'lower', 'mhd_NK_ref_case': 'Ha_6',
            }
            bool_fields = {
                'thermo_on': False, 'mhd_on': False,
                'average_x_direction': False, 'average_z_direction': True,
                'average_over_timesteps': False,
                'ux_velocity_on': True, 'uy_velocity_on': False, 'uz_velocity_on': False,
                'temp_on': False, 'tke_on': False, 'tke_peak_growth_on': False, 'coeff_friction_on': False,
                'mean_vorticity_on': False, 'vorticity_on': False,
                'reynolds_anisotropy_on': False, 'vorticity_anisotropy_on': False,
                'u_prime_sq_on': False, 'u_prime_v_prime_on': False,
                'v_prime_sq_on': False, 'v_prime_w_prime_on': False, 'w_prime_sq_on': False,
                'j1_mean_on': False, 'j2_mean_on': False, 'j3_mean_on': False,
                'j1_rms_on': False, 'j2_rms_on': False, 'j3_rms_on': False,
                'lorentz_force_x_on': False, 'lorentz_force_y_on': False, 'lorentz_force_z_on': False,
                're_stress_budget_on': False, 'heat_transf_coeff_on': False,
                'Nusselt_number_on': False, 'turb_prandtl_on': False,
                'norm_by_u_tau_sq': True, 'norm_ux_by_u_tau': True,
                'norm_y_to_y_plus': False, 'norm_temp_by_ref_temp': False,
                'large_text_on': False, 'ux_velocity_log_ref_on': True,
                'mhd_NK_ref_on': False, 'mkm180_ch_ref_on': False,
            }
            for name, default in str_fields.items():
                if name in v:
                    v[name].set(getattr(mod, name, default))
            for name, default in bool_fields.items():
                if name in v:
                    v[name].set(getattr(mod, name, default))
            v['axis_scale'].set('log' if getattr(mod, 'log_y_scale', False) else 'linear')
            if getattr(mod, 'surface_plot_on', False):
                v['channel_plot_mode'].set('surface plot')
            elif getattr(mod, 'half_channel_plot', False):
                v['channel_plot_mode'].set('half channel')
            else:
                v['channel_plot_mode'].set('full channel')

            def set_t(widget, items):
                widget.delete('1.0', tk.END)
                widget.insert('1.0', '\n'.join(str(x) for x in (items or [])))

            set_t(self._t_cases, getattr(mod, 'cases', []))
            set_t(self._t_timesteps, getattr(mod, 'timesteps', []))
            set_t(self._t_re, getattr(mod, 'Re', []))
            set_t(self._t_ref_temp, getattr(mod, 'ref_temp', []))
            set_t(self._t_ref_len, getattr(mod, 'ref_length', []))
            set_t(self._t_ref_ubulk, getattr(mod, 'ref_bulk_velocity', []))
            set_t(self._t_wall_hf, getattr(mod, 'wall_heat_flux', []))
            set_t(self._t_gravity_dir, getattr(mod, 'gravity_direction', [0.0, -1.0, 0.0]))
            set_t(self._t_mag_field_dir, getattr(mod, 'mag_field_direction', [0.0, 1.0, 0.0]))
            set_t(self._t_stuart_number, [getattr(mod, 'stuart_number', 0.0)])
        except Exception as exc:
            messagebox.showerror('Load error', str(exc))

    def _save_cfg(self):
        path = filedialog.asksaveasfilename(
            title='Save config.py',
            defaultextension='.py',
            initialfile='config.py',
            filetypes=[('Python files', '*.py'), ('All files', '*.*')],
        )
        if not path:
            return
        v = self.vars

        def gl(widget):
            return [x.strip() for x in widget.get('1.0', tk.END).strip().split('\n') if x.strip()]

        def fmts(items):
            return '[' + ', '.join(f"'{x}'" for x in items) + ']'

        def fmtn(items):
            return '[' + ', '.join(items) + ']'

        cases = gl(self._t_cases)
        tss = gl(self._t_timesteps)
        Re = gl(self._t_re)
        ref_temp = gl(self._t_ref_temp)
        ref_len = gl(self._t_ref_len)
        ref_ubulk = gl(self._t_ref_ubulk)
        wall_hf = gl(self._t_wall_hf)
        gravity_direction = (self._parse_floats(self._get_text(self._t_gravity_dir)) + [0.0, 0.0, 0.0])[:3]
        mag_field_direction = (self._parse_floats(self._get_text(self._t_mag_field_dir)) + [0.0, 0.0, 0.0])[:3]
        stuart_number = (self._parse_floats(self._get_text(self._t_stuart_number)) or [0.0])[0]

        lines = [
            '# Configuration file for turb_stats (generated by CHAPSim2 GUI)',
            '',
            f"folder_path = '{v['folder_path'].get()}'",
            f"input_format = '{v['input_format'].get()}'",
            f"cases = {fmts(cases)}",
            f"timesteps = {fmts(tss)}",
            f"forcing = '{v['forcing'].get()}'",
            f"Re = {fmtn(Re)}",
            '',
            f"thermo_on = {v['thermo_on'].get()}",
            f"ref_temp = {fmtn(ref_temp)}",
            f"ref_length = {fmtn(ref_len)}",
            f"ref_bulk_velocity = {fmtn(ref_ubulk)}",
            f"wall_heat_flux = {fmtn(wall_hf)}",
            f"working_fluid = '{v['working_fluid'].get()}'",
            '',
            f"mhd_on = {v['mhd_on'].get()}",
            f"gravity_direction = {gravity_direction}",
            f"mag_field_direction = {mag_field_direction}",
            f"stuart_number = {stuart_number}",
            '',
            f"average_x_direction = {v['average_x_direction'].get()}",
            f"average_z_direction = {v['average_z_direction'].get()}",
            f"average_over_timesteps = {v['average_over_timesteps'].get()}",
            '',
            f"ux_velocity_on = {v['ux_velocity_on'].get()}",
            f"uy_velocity_on = {v['uy_velocity_on'].get()}",
            f"uz_velocity_on = {v['uz_velocity_on'].get()}",
            f"temp_on = {v['temp_on'].get()}",
            f"tke_on = {v['tke_on'].get()}",
            f"tke_peak_growth_on = {v['tke_peak_growth_on'].get()}",
            f"coeff_friction_on = {v['coeff_friction_on'].get()}",
            f"mean_vorticity_on = {v['mean_vorticity_on'].get()}",
            f"vorticity_on = {v['vorticity_on'].get()}",
            f"vorticity_component = '{v['vorticity_component'].get()}'",
            f"reynolds_anisotropy_on = {v['reynolds_anisotropy_on'].get()}",
            f"vorticity_anisotropy_on = {v['vorticity_anisotropy_on'].get()}",
            '',
            f"profile_direction = '{v['profile_direction'].get()}'",
            f"slice_coords = '{v['slice_coords'].get()}'",
            f"x_crop = '{v['x_crop'].get()}'",
            f"x_profile_y_coords = '{v['x_profile_y_coords'].get()}'",
            f"surface_plot_on = {v['channel_plot_mode'].get() == 'surface plot'}",
            '',
            f"u_prime_sq_on = {v['u_prime_sq_on'].get()}",
            f"u_prime_v_prime_on = {v['u_prime_v_prime_on'].get()}",
            f"v_prime_sq_on = {v['v_prime_sq_on'].get()}",
            f"v_prime_w_prime_on = {v['v_prime_w_prime_on'].get()}",
            f"w_prime_sq_on = {v['w_prime_sq_on'].get()}",
            '',
            f"j1_mean_on = {v['j1_mean_on'].get()}",
            f"j2_mean_on = {v['j2_mean_on'].get()}",
            f"j3_mean_on = {v['j3_mean_on'].get()}",
            f"j1_rms_on = {v['j1_rms_on'].get()}",
            f"j2_rms_on = {v['j2_rms_on'].get()}",
            f"j3_rms_on = {v['j3_rms_on'].get()}",
            f"lorentz_force_x_on = {v['lorentz_force_x_on'].get()}",
            f"lorentz_force_y_on = {v['lorentz_force_y_on'].get()}",
            f"lorentz_force_z_on = {v['lorentz_force_z_on'].get()}",
            '',
            f"re_stress_budget_on = {v['re_stress_budget_on'].get()}",
            f"re_stress_component = '{v['re_stress_component'].get()}'",
            '',
            f"heat_transf_coeff_on = {v['heat_transf_coeff_on'].get()}",
            f"Nusselt_number_on = {v['Nusselt_number_on'].get()}",
            f"turb_prandtl_on = {v['turb_prandtl_on'].get()}",
            '',
            f"norm_by_u_tau_sq = {v['norm_by_u_tau_sq'].get()}",
            f"norm_ux_by_u_tau = {v['norm_ux_by_u_tau'].get()}",
            f"norm_y_to_y_plus = {v['norm_y_to_y_plus'].get()}",
            f"norm_temp_by_ref_temp = {v['norm_temp_by_ref_temp'].get()}",
            '',
            f"half_channel_plot = {v['channel_plot_mode'].get() == 'half channel'}",
            f"half_channel_side = '{v['half_channel_side'].get()}'",
            f"linear_y_scale = {v['axis_scale'].get() == 'linear'}",
            f"log_y_scale = {v['axis_scale'].get() == 'log'}",
            'display_fig = False',
            'save_fig = True',
            'save_to_path = True',
            f"large_text_on = {v['large_text_on'].get()}",
            '',
            f"ux_velocity_log_ref_on = {v['ux_velocity_log_ref_on'].get()}",
            f"mhd_NK_ref_on = {v['mhd_NK_ref_on'].get()}",
            f"mhd_NK_ref_case = '{v['mhd_NK_ref_case'].get()}'",
            f"mkm180_ch_ref_on = {v['mkm180_ch_ref_on'].get()}",
        ]
        try:
            with open(path, 'w') as fh:
                fh.write('\n'.join(lines) + '\n')
            messagebox.showinfo('Saved', f'Config saved to:\n{path}')
        except Exception as exc:
            messagebox.showerror('Save error', str(exc))


# =====================================================================================
# SLICE TAB
# =====================================================================================

COLORMAPS = ['RdBu_r', 'viridis', 'plasma', 'inferno', 'magma',
             'coolwarm', 'bwr', 'seismic', 'jet', 'turbo', 'gray']


class SliceTab(ConsoleConsumer, ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent)
        self._var_meta = {}
        self._grid_info = {}
        self._current_fig = None
        # Loaded arrays are kept alive between plots: they are already resident
        # after a load, so retaining them costs no extra memory, and plotting
        # options (cmap, vmin/vmax, crop, combined, ...) don't affect them.
        # The key is the dataset identity — only these fields force a reload.
        self._data = None
        self._data_key = None
        self._build_ui()

    # ------ Layout -------------------------------------------------------------------

    def _build_ui(self):
        pw = ttk.Panedwindow(self, orient='horizontal')
        pw.pack(fill='both', expand=True)

        left = ttk.Frame(pw, width=380)
        left.pack_propagate(False)
        pw.add(left, weight=0)

        right = ttk.Frame(pw)
        pw.add(right, weight=1)

        self._build_controls(left)
        self._panel = FigurePanel(right, placeholder='Load variables and click Plot.')
        self._panel.pack(fill='both', expand=True)

    # ------ Controls (left) ----------------------------------------------------------

    def _build_controls(self, parent):
        # --- Path section always visible above the scroll area ---
        path_frame = ttk.Labelframe(parent, text='Data Path')
        path_frame.pack(fill='x', padx=4, pady=(4, 0))

        self._case_path = tk.StringVar()
        r_path = ttk.Frame(path_frame)
        r_path.pack(fill='x', pady=1)
        ttk.Label(r_path, text='Case folder:', width=12, anchor='w').pack(side='left')
        ttk.Entry(r_path, textvariable=self._case_path).pack(side='left', fill='x', expand=True)
        ttk.Button(r_path, text='…', width=3, command=self._browse).pack(side='left')

        r_scan = ttk.Frame(path_frame)
        r_scan.pack(fill='x', pady=(2, 1))
        ttk.Button(r_scan, text='Scan for timesteps', command=self._scan).pack(side='left')
        ttk.Label(r_scan, text='  (auto-runs after Browse)', foreground='grey',
                  font=('TkDefaultFont', 8)).pack(side='left')

        # --- Scrollable rest of controls ---
        scroll = ScrollableFrame(parent)
        scroll.pack(fill='both', expand=True, padx=4, pady=4)
        f = scroll.inner

        def sec(title):
            lf = ttk.Labelframe(f, text=title, padding=(8, 6))
            lf.pack(fill='x', padx=4, pady=3)
            return lf

        def row(frame, label, widget, width=14):
            r = ttk.Frame(frame)
            r.pack(fill='x', pady=1)
            ttk.Label(r, text=label, width=width, anchor='w').pack(side='left')
            widget(r)

        # Timestep & type (directly first section in scroll area)
        s = sec('Timestep & Type')
        self._ts = tk.StringVar()
        r_ts = ttk.Frame(s); r_ts.pack(fill='x', pady=1)
        ttk.Label(r_ts, text='Timestep:', width=14, anchor='w').pack(side='left')
        self._ts_combo = ttk.Combobox(r_ts, textvariable=self._ts, state='readonly', width=16)
        self._ts_combo.pack(side='left')

        self._dtype = tk.StringVar(value='t_avg')
        row(s, 'Data type:', lambda r: ttk.Combobox(r, textvariable=self._dtype,
            values=['t_avg', 'tsp_avg', 'inst', '2d_slice'],
            state='readonly', width=16).pack(side='left'))

        self._phys = tk.StringVar(value='flow')
        row(s, 'Physics:', lambda r: ttk.Combobox(r, textvariable=self._phys,
            values=['flow', 'thermo', 'mhd'],
            state='readonly', width=16).pack(side='left'))

        self._slice_lbl = tk.StringVar()
        row(s, 'Slice label:', lambda r: ttk.Entry(r, textvariable=self._slice_lbl,
            width=12).pack(side='left'))

        ttk.Button(s, text='Load variables', command=self._load_vars).pack(anchor='w', pady=2)

        # Variables
        s = sec('Variables')
        self._var_lb = ttk.Listbox(s, selectmode='extended', height=8, exportselection=False,
                                    activestyle='none', relief='flat', borderwidth=0)
        sb2 = ttk.Scrollbar(s, orient='vertical', command=self._var_lb.yview)
        self._var_lb.configure(yscrollcommand=sb2.set)
        self._var_lb.pack(side='left', fill='both', expand=True)
        sb2.pack(side='right', fill='y')

        # Slice config
        s = sec('Slice Configuration')
        self._plane = tk.StringVar(value='xy')
        row(s, 'Plane:', lambda r: ttk.Combobox(r, textvariable=self._plane,
            values=['xy', 'xz', 'yz'], state='readonly', width=8).pack(side='left'))

        self._idx = tk.IntVar(value=0)
        r2 = ttk.Frame(s)
        r2.pack(fill='x', pady=1)
        ttk.Label(r2, text='Slice index:', width=14, anchor='w').pack(side='left')
        self._idx_spin = ttk.Spinbox(r2, textvariable=self._idx, from_=0, to=9999, width=8)
        self._idx_spin.pack(side='left')
        self._idx_coord_lbl = ttk.Label(r2, text='')
        self._idx_coord_lbl.pack(side='left', padx=4)
        self._idx.trace_add('write', self._update_coord_label)

        self._xcrop = tk.StringVar()
        row(s, 'x crop:', lambda r: ttk.Entry(r, textvariable=self._xcrop,
            width=14).pack(side='left'))
        ttk.Label(s, text='(x_min,x_max — xy/xz planes only)', foreground='grey',
                  font=('TkDefaultFont', 8)).pack(anchor='w')

        # Statistics (fluctuation / vorticity)
        s = sec('Statistics')
        self._use_fluc = tk.BooleanVar(value=False)
        ttk.Checkbutton(s, text="Fluctuation (u' = inst − t_avg)", variable=self._use_fluc,
                         command=self._on_fluc_toggle).pack(anchor='w')

        r_ta = ttk.Frame(s)
        r_ta.pack(fill='x', pady=1)
        ttk.Label(r_ta, text='t_avg file:', width=14, anchor='w').pack(side='left')
        self._t_avg_path = tk.StringVar()
        ttk.Entry(r_ta, textvariable=self._t_avg_path, width=16).pack(side='left', fill='x', expand=True)
        ttk.Button(r_ta, text='...', width=3, command=self._browse_t_avg).pack(side='left')

        self._use_vort = tk.BooleanVar(value=False)
        r_vort = ttk.Frame(s)
        r_vort.pack(fill='x', pady=1)
        ttk.Checkbutton(r_vort, text='Vorticity  (requires qx_ccc, qy_ccc, qz_ccc)',
                         variable=self._use_vort).pack(side='left')
        self._vort_component = tk.StringVar(value='z')
        ttk.Combobox(r_vort, textvariable=self._vort_component, values=['x', 'y', 'z'],
                     state='readonly', width=4).pack(side='left', padx=4)

        # Plot options
        s = sec('Plot Options')
        self._cmap = tk.StringVar(value='RdBu_r')
        row(s, 'Colormap:', lambda r: ttk.Combobox(r, textvariable=self._cmap,
            values=COLORMAPS, width=14).pack(side='left'))

        r3 = ttk.Frame(s)
        r3.pack(fill='x', pady=1)
        ttk.Label(r3, text='Colour scale:', width=14, anchor='w').pack(side='left')
        self._cscale = tk.StringVar(value='auto')
        for val, lbl in [('auto', 'Auto'), ('sym', 'Symmetric'), ('custom', 'Custom')]:
            ttk.Radiobutton(r3, text=lbl, variable=self._cscale, value=val).pack(side='left')

        r4 = ttk.Frame(s)
        r4.pack(fill='x', pady=1)
        ttk.Label(r4, text='vmin / vmax:', width=14, anchor='w').pack(side='left')
        self._vmin = tk.StringVar()
        self._vmax = tk.StringVar()
        ttk.Entry(r4, textvariable=self._vmin, width=8).pack(side='left')
        ttk.Label(r4, text=' / ').pack(side='left')
        ttk.Entry(r4, textvariable=self._vmax, width=8).pack(side='left')

        self._interp = tk.BooleanVar(value=False)
        ttk.Checkbutton(s, text='Interpolate cell → point', variable=self._interp).pack(anchor='w')

        self._combined = tk.BooleanVar(value=True)
        ttk.Checkbutton(s, text='Combined plot (all selected vars)', variable=self._combined).pack(anchor='w')

        r5 = ttk.Frame(s)
        r5.pack(fill='x', pady=3)
        ttk.Button(r5, text='Plot', command=self._plot).pack(side='left', padx=2)
        ttk.Button(r5, text='Save…', command=self._save_plot).pack(side='left', padx=2)

    # ------ Helpers ------------------------------------------------------------------

    def _browse(self):
        d = filedialog.askdirectory(title='Select case folder (containing 2_visu/)')
        if d:
            self._case_path.set(d)
            self._scan()

    def _visu_folder(self):
        base = self._case_path.get().rstrip('/')
        candidate = os.path.join(base, '2_visu')
        if os.path.isdir(candidate):
            return candidate
        return base

    def _xdmf_path(self):
        visu = self._visu_folder()
        ts = self._ts.get()
        dtype = self._dtype.get()
        phys = self._phys.get()
        sl = self._slice_lbl.get().strip()
        if dtype == 'inst':
            # Instantaneous files have no 'inst' prefix: domain1_{phys}_{ts}.xdmf
            if sl:
                name = f'domain1_{phys}_{sl}_{ts}.xdmf'
            else:
                name = f'domain1_{phys}_{ts}.xdmf'
        elif dtype == '2d_slice':
            # 2D slice: domain1_{phys}_{slice_label}_{ts}.xdmf
            name = f'domain1_{phys}_{sl}_{ts}.xdmf' if sl else f'domain1_{phys}_{ts}.xdmf'
        else:
            # t_avg / tsp_avg: domain1_{dtype}_{phys}_{ts}.xdmf
            name = f'domain1_{dtype}_{phys}_{ts}.xdmf'
        path = os.path.join(visu, name)
        if not os.path.exists(path):
            # Fallback: glob for any matching file
            pattern = (f'domain1_{phys}_{ts}*.xdmf' if dtype == 'inst'
                       else f'domain1_{dtype}_{phys}_{ts}*.xdmf')
            matches = glob.glob(os.path.join(visu, pattern))
            if matches:
                return matches[0]
        return path

    def _default_t_avg_path(self):
        visu = self._visu_folder()
        ts = self._ts.get()
        phys = self._phys.get()
        sl = self._slice_lbl.get().strip()
        # Mirrors _xdmf_path(): a slice label produces a slice-tagged filename
        # for both 'inst' and '2d_slice' dtypes.
        if sl:
            name = f'domain1_t_avg_{phys}_{sl}_{ts}.xdmf'
        else:
            name = f'domain1_t_avg_{phys}_{ts}.xdmf'
        return os.path.join(visu, name)

    def _on_fluc_toggle(self):
        if self._use_fluc.get() and not self._t_avg_path.get().strip():
            self._t_avg_path.set(self._default_t_avg_path())

    def _browse_t_avg(self):
        path = filedialog.askopenfilename(title='Select t_avg xdmf file',
                                           filetypes=[('XDMF', '*.xdmf'), ('All files', '*.*')])
        if path:
            self._t_avg_path.set(path)

    def _scan(self):
        visu = self._visu_folder()
        try:
            from slice import get_available_timesteps
            tss = get_available_timesteps(visu)
            self._ts_combo['values'] = tss
            if tss:
                self._ts.set(tss[0])
            self._log(f'Found {len(tss)} timestep(s): {", ".join(tss)}')
        except Exception as exc:
            self._log(f'Scan error: {exc}')

    def _load_vars(self):
        xdmf = self._xdmf_path()
        self._log(f'Reading metadata: {xdmf}')
        try:
            from utils import parse_xdmf_metadata
            self._var_meta, self._grid_info = parse_xdmf_metadata(xdmf)
            names = sorted(self._var_meta.keys())
            self._var_lb.delete(0, tk.END)
            for n in names:
                self._var_lb.insert(tk.END, n)
            self._log(f'Loaded {len(names)} variable(s).')
            # Update index spin max
            gy = self._grid_info.get('grid_y')
            if gy is not None:
                self._idx_spin.configure(to=len(gy) - 1)
        except Exception as exc:
            self._log(f'Error: {exc}\n{traceback.format_exc()}')

    def _update_coord_label(self, *_):
        if not self._grid_info:
            return
        try:
            from slice import get_slice_location
            loc = get_slice_location(self._grid_info, self._plane.get(), self._idx.get())
            if loc is not None:
                self._idx_coord_lbl.configure(text=f'coord = {loc:.4f}')
        except Exception:
            pass

    def _selected_vars(self):
        return [self._var_lb.get(i) for i in self._var_lb.curselection()]

    def _plot(self):
        sel = self._selected_vars()
        if not sel:
            messagebox.showwarning('No variables', 'Select at least one variable.')
            return
        if not self._var_meta:
            messagebox.showwarning('No metadata', 'Load variables first.')
            return

        def worker():
            try:
                from utils import (parse_xdmf_metadata, load_xdmf_variables, slice_axis_info,
                                    parse_x_crop_input, apply_x_crop)
                from slice import (extract_slice, plot_slice, plot_combined_slices,
                                   process_data_arrays, get_slice_location, apply_fluctuation,
                                   apply_vorticity)

                try:
                    x_crop = parse_x_crop_input(self._xcrop.get())
                except ValueError as exc:
                    self._log(f'Invalid x crop ({exc}); ignoring.')
                    x_crop = None

                use_vort = self._use_vort.get()
                load_vars = list({'qx_ccc', 'qy_ccc', 'qz_ccc'} | set(sel)) if use_vort else sel

                xdmf = self._xdmf_path()
                var_meta, grid = parse_xdmf_metadata(xdmf)

                # Re-load only when the dataset identity changes. `data` is a
                # shallow copy so the derived fields that apply_vorticity /
                # apply_fluctuation write back cannot leak into the retained
                # base across runs.
                key = (xdmf, tuple(sorted(load_vars)))
                if self._data is not None and key == self._data_key:
                    self._log('Reusing loaded data (dataset unchanged).')
                    data = dict(self._data)
                else:
                    self._log('Loading data…')
                    data = load_xdmf_variables(var_meta, load_vars, grid)
                    self._data = dict(data)
                    self._data_key = key

                plot_vars = sel
                if use_vort:
                    component = self._vort_component.get()
                    self._log(f'Computing vorticity (ω_{component})…')
                    plot_vars = apply_vorticity(data, grid, component)
                elif self._use_fluc.get():
                    t_avg_path = self._t_avg_path.get().strip()
                    if not t_avg_path or not os.path.isfile(t_avg_path):
                        self._log(f'Warning: t_avg file not found: {t_avg_path}. Skipping fluctuation.')
                    else:
                        self._log(f'Computing fluctuation against {t_avg_path}…')
                        plot_vars = apply_fluctuation(data, sel, grid, t_avg_path)

                interp = self._interp.get()
                processed, interp_vars = process_data_arrays(data, plot_vars, grid, interp)

                cmap = self._cmap.get()
                cscale = self._cscale.get()
                symmetric = (cscale == 'sym')
                vmin = float(self._vmin.get()) if (cscale == 'custom' and self._vmin.get().strip()) else None
                vmax = float(self._vmax.get()) if (cscale == 'custom' and self._vmax.get().strip()) else None

                plane = self._plane.get()
                idx = self._idx.get()
                ts = self._ts.get()

                sample = next(iter(processed.values()))
                is_2d = sample.ndim <= 2

                if is_2d:
                    axis_info = slice_axis_info(self._slice_lbl.get().strip())
                    if axis_info:
                        c1_key, c2_key = axis_info['coord_keys']
                        axis_labels = axis_info['axis_labels']
                        crop_plane = axis_info['plane']
                    else:
                        c1_key, c2_key = 'grid_x', 'grid_y'
                        axis_labels = ('x', 'y')
                        crop_plane = plane
                    coord1 = grid.get(c1_key, np.arange(sample.shape[-1] if sample.ndim > 1 else 1))
                    coord2 = grid.get(c2_key, np.arange(sample.shape[0]))
                    slice_info = f'2D data, t={ts}'
                    slices = [(vn, processed[vn]) for vn in plot_vars if vn in processed]

                    if x_crop is not None and crop_plane in ('xy', 'xz'):
                        cropped_slices = []
                        for vn, sd in slices:
                            sd, coord1 = apply_x_crop(sd, coord1, x_crop)
                            cropped_slices.append((vn, sd))
                        slices = cropped_slices
                else:
                    slices = []
                    coord1 = coord2 = axis_labels = None
                    for vn in plot_vars:
                        if vn not in processed:
                            continue
                        sd, c1, c2, al = extract_slice(processed[vn], plane, idx, grid)
                        if x_crop is not None and plane in ('xy', 'xz'):
                            sd, c1 = apply_x_crop(sd, c1, x_crop)
                        slices.append((vn, sd))
                        coord1, coord2, axis_labels = c1, c2, al
                    loc = get_slice_location(grid, plane, idx)
                    slice_info = (f'{plane}-plane idx={idx} ({loc:.4f}), t={ts}'
                                  if loc is not None else f'{plane}-plane idx={idx}, t={ts}')

                if self._combined.get() and len(slices) > 1:
                    fig = plot_combined_slices(
                        slices, coord1, coord2, axis_labels,
                        slice_info=slice_info, cmap=cmap, symmetric=symmetric,
                        display=False, point_data_vars=interp_vars,
                    )
                else:
                    figs = []
                    for vn, arr in slices:
                        fig = plot_slice(
                            arr, coord1, coord2, axis_labels, vn,
                            cmap=cmap, vmin=vmin, vmax=vmax, symmetric=symmetric,
                            slice_info=slice_info, display=False,
                        )
                        figs.append(fig)
                    fig = figs[-1] if figs else None

                if fig:
                    self._current_fig = fig
                    self.after(0, lambda: self._panel.show(fig))
                    self._log('Plot complete.')
            except Exception as exc:
                self._log(f'Error: {exc}\n{traceback.format_exc()}')

        threading.Thread(target=worker, daemon=True).start()

    def _save_plot(self):
        if not self._current_fig:
            messagebox.showwarning('No plot', 'Generate a plot first.')
            return
        path = filedialog.asksaveasfilename(
            title='Save plot',
            defaultextension='.png',
            filetypes=[('PNG', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg')],
        )
        if path:
            self._current_fig.savefig(path, dpi=300, bbox_inches='tight')
            self._log(f'Saved to {path}')


# =====================================================================================
# MONITOR POINTS TAB
# =====================================================================================

class MonitorPointsTab(ConsoleConsumer, ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent)
        self._figures = []   # list of (label, Figure)
        self._build_ui()

    # ------ Layout -------------------------------------------------------------------

    def _build_ui(self):
        pw = ttk.Panedwindow(self, orient='horizontal')
        pw.pack(fill='both', expand=True)

        left = ttk.Frame(pw, width=320)
        left.pack_propagate(False)
        pw.add(left, weight=0)

        right = ttk.Frame(pw)
        pw.add(right, weight=1)

        self._build_controls(left)
        self._panel = FigurePanel(right, placeholder='Configure and click Run.')
        self._panel.pack(fill='both', expand=True)

    # ------ Controls (left) ----------------------------------------------------------

    def _build_controls(self, parent):
        f = ttk.Frame(parent)
        f.pack(fill='both', expand=True, padx=4, pady=4)

        # Path
        s = ttk.Labelframe(f, text='Data Path')
        s.pack(fill='x', padx=4, pady=3)
        self._path = tk.StringVar()
        r = ttk.Frame(s)
        r.pack(fill='x')
        ttk.Entry(r, textvariable=self._path).pack(side='left', fill='x', expand=True)
        ttk.Button(r, text='…', width=3,
                   command=lambda: self._path.set(filedialog.askdirectory() or self._path.get())
                   ).pack(side='left')

        # Options
        s = ttk.Labelframe(f, text='Options')
        s.pack(fill='x', padx=4, pady=3)

        def spin_row(frame, label, var, lo, hi):
            r = ttk.Frame(frame)
            r.pack(fill='x', pady=1)
            ttk.Label(r, text=label, width=22, anchor='w').pack(side='left')
            ttk.Spinbox(r, textvariable=var, from_=lo, to=hi, width=9).pack(side='left')

        self._npts = tk.IntVar(value=5)
        spin_row(s, 'Monitor points:', self._npts, 1, 99)

        self._thermo = tk.BooleanVar(value=True)
        ttk.Checkbutton(s, text='Include temperature', variable=self._thermo).pack(anchor='w')

        self._sample = tk.IntVar(value=10)
        spin_row(s, 'Sample factor:', self._sample, 1, 9999)

        self._window = tk.IntVar(value=0)
        spin_row(s, 'Running avg. window:', self._window, 1, 999999)

        self._auto_ylim = tk.BooleanVar(value=True)
        ttk.Checkbutton(s, text='Auto y-lim (divergence detect)',
                        variable=self._auto_ylim).pack(anchor='w')

        self._plt_pts = tk.BooleanVar(value=True)
        ttk.Checkbutton(s, text='Plot monitor points', variable=self._plt_pts).pack(anchor='w')

        self._plt_bulk = tk.BooleanVar(value=True)
        ttk.Checkbutton(s, text='Plot bulk/change history', variable=self._plt_bulk).pack(anchor='w')

        self._save = tk.BooleanVar(value=False)
        ttk.Checkbutton(s, text='Save plots to data folder', variable=self._save).pack(anchor='w')

        ttk.Button(f, text='Run', command=self._run).pack(fill='x', padx=4, pady=6)

        # Figure selector
        s2 = ttk.Labelframe(f, text='Figures')
        s2.pack(fill='both', expand=True, padx=4, pady=3)
        self._fig_lb = ttk.Listbox(s2, height=10, exportselection=False,
                                    activestyle='none', relief='flat', borderwidth=0)
        sb = ttk.Scrollbar(s2, orient='vertical', command=self._fig_lb.yview)
        self._fig_lb.configure(yscrollcommand=sb.set)
        self._fig_lb.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        self._fig_lb.bind('<<ListboxSelect>>', self._on_select)

    # ------ Helpers ------------------------------------------------------------------

    def _on_select(self, _event=None):
        sel = self._fig_lb.curselection()
        if sel and sel[0] < len(self._figures):
            _, fig = self._figures[sel[0]]
            self._panel.show(fig)

    # ------ Run ----------------------------------------------------------------------

    def _run(self):
        path = self._path.get().strip()
        if not path:
            messagebox.showwarning('No path', 'Select the data directory.')
            return
        if not path.endswith('/'):
            path += '/'

        self._clear_console()

        n_pts = self._npts.get()
        thermo = self._thermo.get()
        sample = self._sample.get()
        window = self._window.get()
        auto_ylim = self._auto_ylim.get()
        plt_pts = self._plt_pts.get()
        plt_bulk = self._plt_bulk.get()
        save = self._save.get()

        def worker():
            figures = []
            try:
                if plt_pts:
                    for i in range(1, n_pts + 1):
                        fname = f'domain1_monitor_pt{i}_flow.dat'
                        fpath = path + fname
                        if not os.path.exists(fpath):
                            self._log(f'Not found: {fname}')
                            continue
                        data = _mp_load(fpath, skiprows=3, sample=sample)
                        if data.size == 0:
                            self._log(f'No valid data in {fname}')
                            continue
                        self._log(f'Plotting {len(data)} points for {fname}…')

                        t = data[:, 1]
                        u, v, w = data[:, 2], data[:, 3], data[:, 4]
                        p, phi = data[:, 5], data[:, 6]
                        T = data[:, 7] if (thermo and data.shape[1] > 7) else None

                        scalar_fields = [('pressure', p, 'C3'),
                                         ('press. corr.', phi, 'C4')]
                        if T is not None:
                            scalar_fields.append(('temperature', T, 'C5'))

                        n_sub = 1 + len(scalar_fields)
                        fig = Figure(figsize=(10, 3 * n_sub))
                        axes = fig.subplots(n_sub, 1, sharex=True)

                        # Combined velocity subplot
                        for lbl, arr, col in [('u', u, 'C0'), ('v', v, 'C1'), ('w', w, 'C2')]:
                            _mp_plot_avg(axes[0], t, arr, lbl, col, window)
                        axes[0].set_ylabel('Velocity')
                        axes[0].legend(fontsize=7)
                        axes[0].grid(True, alpha=0.4)
                        if auto_ylim:
                            _mp_apply_ylim(axes[0], np.concatenate([u, v, w]))

                        for ax, (lbl, arr, col) in zip(axes[1:], scalar_fields):
                            _mp_plot_avg(ax, t, arr, lbl, col, window)
                            ax.set_ylabel(lbl)
                            ax.legend(fontsize=7)
                            ax.grid(True, alpha=0.4)
                            if auto_ylim:
                                _mp_apply_ylim(ax, arr)
                            _mp_stats_box(ax, arr)
                        axes[-1].set_xlabel('Time')
                        fig.suptitle(f'{fname} — Monitor Point Data', fontsize=12)
                        fig.tight_layout()
                        if save:
                            out = f'{path}{fname.replace("domain1_monitor_","").replace(".dat","_plot")}.png'
                            fig.savefig(out, dpi=150, bbox_inches='tight')
                        figures.append((f'Pt {i}', fig))

                if plt_bulk:
                    for fname in ['domain1_monitor_metrics_history.log',
                                  'domain1_monitor_change_history.log']:
                        fpath = path + fname
                        if not os.path.exists(fpath):
                            self._log(f'Not found: {fname}')
                            continue
                        data = _mp_load(fpath, skiprows=2, sample=sample)
                        if data.size == 0:
                            continue

                        if 'metrics' in fname:
                            t = data[:, 0]
                            MKE, qx = data[:, 1], data[:, 2]
                            has_th = thermo and data.shape[1] > 5
                            if has_th:
                                gx, T, h = data[:, 3], data[:, 4], data[:, 5]
                            n_sub = 4 if has_th else 2
                            fig = Figure(figsize=(10, 3 * n_sub))
                            axes = fig.subplots(n_sub, 1, sharex=True)
                            _mp_plot_avg(axes[0], t, MKE, 'Mean Kinetic Energy', 'C0', window)
                            axes[0].set_ylabel('MKE')
                            axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.4)
                            if auto_ylim: _mp_apply_ylim(axes[0], MKE)
                            _mp_stats_box(axes[0], MKE)

                            _mp_plot_avg(axes[1], t, qx, 'Bulk Velocity', 'C1', window)
                            if has_th:
                                _mp_plot_avg(axes[1], t, gx, 'ρ·U_bulk', 'C2', window)
                            axes[1].set_ylabel('Velocity')
                            axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.4)
                            if auto_ylim:
                                _mp_apply_ylim(axes[1], np.concatenate([qx, gx]) if has_th else qx)
                            _mp_stats_box(axes[1], qx)

                            if has_th:
                                _mp_plot_avg(axes[2], t, T, 'Bulk Temperature', 'C3', window)
                                axes[2].set_ylabel('Bulk T')
                                axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.4)
                                if auto_ylim: _mp_apply_ylim(axes[2], T)
                                _mp_stats_box(axes[2], T)

                                _mp_plot_avg(axes[3], t, h, 'Bulk Enthalpy', 'C4', window)
                                axes[3].set_ylabel('Bulk h')
                                axes[3].legend(fontsize=7); axes[3].grid(True, alpha=0.4)
                                if auto_ylim: _mp_apply_ylim(axes[3], h)
                                _mp_stats_box(axes[3], h)
                                axes[3].set_xlabel('Time')
                            else:
                                axes[1].set_xlabel('Time')

                            fig.suptitle('Bulk Quantities', fontsize=12)
                            fig.tight_layout()
                            if save:
                                out = f'{path}{fname.replace("domain1_monitor_","").replace(".log","_plot")}.png'
                                fig.savefig(out, dpi=150, bbox_inches='tight')
                            figures.append(('Bulk Quantities', fig))

                        elif 'change' in fname:
                            t = data[:, 0]
                            mass_cons = data[:, 1]
                            mass_rt = data[:, 4]
                            ke_rt = data[:, 5]
                            fig = Figure(figsize=(10, 9))
                            axes = fig.subplots(3, 1, sharex=True)
                            for ax, arr, lbl, col in zip(
                                axes,
                                [mass_cons, mass_rt, ke_rt],
                                ['Mass Conservation', 'Mass Change Rate', 'KE Change Rate'],
                                ['C0', 'C1', 'C2'],
                            ):
                                _mp_plot_avg(ax, t, arr, lbl, col, window)
                                ax.set_ylabel(lbl)
                                ax.legend(fontsize=7); ax.grid(True, alpha=0.4)
                                if auto_ylim: _mp_apply_ylim(ax, arr)
                                _mp_stats_box(ax, arr)
                            axes[2].set_xlabel('Time')
                            fig.suptitle('Change History', fontsize=12)
                            fig.tight_layout()
                            if save:
                                out = f'{path}{fname.replace("domain1_monitor_","").replace(".log","_plot")}.png'
                                fig.savefig(out, dpi=150, bbox_inches='tight')
                            figures.append(('Change History', fig))

                self.after(0, lambda: self._update_figs(figures))
                self._log('Done.')
            except Exception as exc:
                self._log(f'Error: {exc}\n{traceback.format_exc()}')

        threading.Thread(target=worker, daemon=True).start()

    def _update_figs(self, figures):
        self._figures = figures
        self._fig_lb.delete(0, tk.END)
        for label, _ in figures:
            self._fig_lb.insert(tk.END, label)
        if figures:
            self._fig_lb.selection_set(0)
            self._panel.show(figures[0][1])


# =====================================================================================
# 3D VISUALISATION TAB
# =====================================================================================

_VIS_OPACITY = ['linear', 'sigmoid', 'sigmoid_r', 'geom', 'geom_r']


class Visu3DPanel(ttk.Frame):
    """
    Embeds a PyVista off-screen renderer in a tkinter Canvas.

    Renders via pv.Plotter(off_screen=True) — no special VTK Tk library required.
    The resulting image is painted onto a Canvas widget:
      • Left-drag   → orbit  (azimuth / elevation)
      • Right-drag  → pan
      • Scroll      → zoom
    Requires Pillow for image display.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._plotter     = None
        self._photo       = None   # keep reference so GC doesn't collect it
        self._drag_start  = None
        self._pan_start   = None
        self._resize_job  = None

        self._canvas = tk.Canvas(self, bg='#1a1a1a', cursor='crosshair',
                                 highlightthickness=0)
        self._canvas.pack(fill='both', expand=True)

        self._hint_id = self._canvas.create_text(
            300, 200, text='Configure the left panel and click Render.',
            fill='#888888', font=('TkDefaultFont', 10))

        # Mouse bindings
        self._canvas.bind('<ButtonPress-1>',   self._on_press)
        self._canvas.bind('<B1-Motion>',       self._on_orbit)
        self._canvas.bind('<ButtonPress-3>',   self._on_pan_press)
        self._canvas.bind('<B3-Motion>',       self._on_pan)
        self._canvas.bind('<Button-4>',        lambda e: self._zoom(1.1))
        self._canvas.bind('<Button-5>',        lambda e: self._zoom(0.9))
        self._canvas.bind('<MouseWheel>',
                          lambda e: self._zoom(1.1 if e.delta > 0 else 0.9))
        self._canvas.bind('<Configure>',       self._on_configure)

    # ------ Scene building -----------------------------------------------------------

    def render(self, grid, cfg):
        """Replace the current scene.  Must be called from the main (Tk) thread."""
        import pyvista as pv
        import turb_visu as tv

        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:
                pass

        w = max(self._canvas.winfo_width(),  600)
        h = max(self._canvas.winfo_height(), 400)
        self._plotter = pv.Plotter(off_screen=True, window_size=[w, h])

        variable = cfg['variable']
        cmap     = cfg['cmap']
        mode     = cfg['mode']

        self._plotter.add_mesh(grid.outline(), color='gray', line_width=1)

        if mode == 'slice':
            origins = {
                'x': lambda v: (v, 0, 0),
                'y': lambda v: (0, v, 0),
                'z': lambda v: (0, 0, v),
            }
            clim = tv._resolve_clim(cfg, grid.cell_data[variable])
            n_added = 0
            for normal in ('x', 'y', 'z'):
                pos = cfg.get(f'cut_{normal}')
                if pos is None:
                    continue
                sl = grid.slice(normal=normal, origin=origins[normal](pos))
                self._plotter.add_mesh(sl, scalars=variable, cmap=cmap, clim=clim,
                                       show_scalar_bar=(n_added == 0))
                n_added += 1
            self._plotter.add_axes()
            if n_added:
                self._plotter.show_grid()

        elif mode == 'iso':
            iso_vals = cfg.get('iso_vals', [0.0])
            grid_pt = grid.cell_data_to_point_data()
            contours = grid_pt.contour(isosurfaces=iso_vals, scalars=variable)
            if contours.n_points > 0:
                # cell_data_to_point_data/contour carry every point-data
                # array along, not just the one used for the isovalue — so a
                # different color_variable (already on grid_pt) just works.
                color_variable = cfg.get('color_variable') or variable
                clim = tv._resolve_clim(cfg, grid.cell_data[color_variable])
                self._plotter.add_mesh(contours, scalars=color_variable, cmap=cmap, clim=clim,
                                       show_scalar_bar=True)
            self._plotter.add_axes()

        elif mode == 'volume':
            if grid.n_cells > tv.VOLUME_CELL_THRESHOLD:
                print(f"Refusing volume render: {grid.n_cells:,} cells "
                      f"> {tv.VOLUME_CELL_THRESHOLD:,} limit. Increase the stride.")
                return
            self._plotter.add_volume(grid, scalars=variable, cmap=cmap,
                                     opacity=cfg.get('opacity', 'sigmoid'),
                                     clim=tv._resolve_clim(cfg, grid.cell_data[variable]),
                                     show_scalar_bar=True)
            self._plotter.add_axes()

        self._plotter.reset_camera()
        if self._hint_id is not None:
            self._canvas.delete(self._hint_id)
            self._hint_id = None
        self._refresh()

    # ------ Rendering ----------------------------------------------------------------

    def _refresh(self):
        """Render a frame off-screen and paint it onto the canvas."""
        if self._plotter is None:
            return
        try:
            from PIL import Image, ImageTk
        except ImportError:
            self._canvas.delete('all')
            self._canvas.create_text(
                300, 200,
                text='Install Pillow (pip install Pillow) to display renders.',
                fill='#ff8888', font=('TkDefaultFont', 10))
            return

        self._plotter.render()  # force re-render; screenshot()'s internal render() is a no-op when VTK dirty flag isn't set
        img_arr = self._plotter.screenshot(return_img=True)
        pil_img = Image.fromarray(img_arr)

        w = self._canvas.winfo_width()
        h = self._canvas.winfo_height()
        if w > 1 and h > 1 and pil_img.size != (w, h):
            pil_img = pil_img.resize((w, h), Image.BILINEAR)

        self._photo = ImageTk.PhotoImage(pil_img)
        self._canvas.delete('all')
        self._canvas.create_image(0, 0, anchor='nw', image=self._photo)
        self._canvas.update_idletasks()

    # ------ Camera controls ----------------------------------------------------------

    def _on_press(self, event):
        self._drag_start = (event.x, event.y)

    def _on_orbit(self, event):
        if self._drag_start is None or self._plotter is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._drag_start = (event.x, event.y)
        self._plotter.camera.Azimuth(-dx * 0.4)
        self._plotter.camera.Elevation(dy * 0.4)
        self._refresh()

    def _on_pan_press(self, event):
        self._pan_start = (event.x, event.y)

    def _on_pan(self, event):
        if self._pan_start is None or self._plotter is None:
            return
        import numpy as np
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self._pan_start = (event.x, event.y)
        cam = self._plotter.camera
        pos = np.array(cam.GetPosition())
        fp  = np.array(cam.GetFocalPoint())
        vu  = np.array(cam.GetViewUp())
        fwd   = fp - pos
        right = np.cross(fwd, vu);  right /= np.linalg.norm(right)
        up    = np.cross(right, fwd); up   /= np.linalg.norm(up)
        scale = np.linalg.norm(fwd) * 0.001
        delta = (-dx * right + dy * up) * scale
        cam.SetPosition(*(pos + delta))
        cam.SetFocalPoint(*(fp  + delta))
        self._plotter.renderer.ResetCameraClippingRange()
        self._refresh()

    def _zoom(self, factor):
        if self._plotter is None:
            return
        self._plotter.camera.Zoom(factor)
        self._plotter.renderer.ResetCameraClippingRange()
        self._refresh()

    def _on_configure(self, event):
        if self._plotter is None:
            return
        if self._resize_job is not None:
            self._canvas.after_cancel(self._resize_job)
        self._resize_job = self._canvas.after(150, self._handle_resize)

    def _handle_resize(self):
        self._resize_job = None
        if self._plotter is None:
            return
        w = self._canvas.winfo_width()
        h = self._canvas.winfo_height()
        if w > 1 and h > 1:
            self._plotter.window_size = [w, h]
            self._refresh()

    def save_screenshot(self, path):
        """Save the current view to a file."""
        if self._plotter is not None:
            self._plotter.screenshot(filename=path)


class TurbVisuTab(ConsoleConsumer, ttk.Frame):
    """3D visualisation tab — renders inside the GUI panel."""

    def __init__(self, parent):
        super().__init__(parent)
        self._var_meta = {}
        self._grid_info = {}
        # Loaded arrays are kept alive between renders (see SliceTab.__init__).
        # stride is part of the key because it is applied at read time.
        self._data = None
        self._data_key = None
        self._build_ui()

    # ------ Layout -------------------------------------------------------------------

    def _build_ui(self):
        pw = ttk.Panedwindow(self, orient='horizontal')
        pw.pack(fill='both', expand=True)

        left = ttk.Frame(pw, width=360)
        left.pack_propagate(False)
        pw.add(left, weight=0)

        right = ttk.Frame(pw)
        pw.add(right, weight=1)

        self._build_controls(left)
        self._build_right(right)

    def _build_right(self, parent):
        """Right panel: embedded 3D render widget."""
        self._visu_panel = Visu3DPanel(parent)
        self._visu_panel.pack(fill='both', expand=True)

    # ------ Controls (left) ----------------------------------------------------------

    def _build_controls(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(fill='x', padx=5, pady=4)
        ttk.Button(bar, text='Render', command=self._render).pack(side='left', padx=2)

        scroll = ScrollableFrame(parent)
        scroll.pack(fill='both', expand=True, padx=4, pady=2)
        f = scroll.inner

        def sec(title):
            lf = ttk.Labelframe(f, text=title, padding=(8, 6))
            lf.pack(fill='x', padx=4, pady=3)
            return lf

        def row(frame, label, widget_fn, width=14):
            r = ttk.Frame(frame)
            r.pack(fill='x', pady=1)
            ttk.Label(r, text=label, width=width, anchor='w').pack(side='left')
            widget_fn(r)
            return r

        # ---- Path ----
        s = sec('Data Path')
        self._case_path = tk.StringVar()
        r = ttk.Frame(s); r.pack(fill='x', pady=1)
        ttk.Label(r, text='Case folder:', width=14, anchor='w').pack(side='left')
        ttk.Entry(r, textvariable=self._case_path).pack(side='left', fill='x', expand=True)
        ttk.Button(r, text='…', width=3, command=self._browse).pack(side='left')

        r2 = ttk.Frame(s); r2.pack(fill='x', pady=1)
        ttk.Button(r2, text='Scan for timesteps', command=self._scan).pack(side='left')

        self._ts = tk.StringVar()
        r3 = ttk.Frame(s); r3.pack(fill='x', pady=1)
        ttk.Label(r3, text='Timestep:', width=14, anchor='w').pack(side='left')
        self._ts_combo = ttk.Combobox(r3, textvariable=self._ts, state='readonly', width=16)
        self._ts_combo.pack(side='left')

        self._dtype = tk.StringVar(value='inst')
        row(s, 'Data type:', lambda r: ttk.Combobox(
            r, textvariable=self._dtype,
            values=['inst', 't_avg'], state='readonly', width=16).pack(side='left'))

        self._phys = tk.StringVar(value='flow')
        row(s, 'Physics:', lambda r: ttk.Combobox(
            r, textvariable=self._phys,
            values=['flow', 'thermo', 'mhd'], state='readonly', width=16).pack(side='left'))

        ttk.Button(s, text='Load variables', command=self._load_vars).pack(anchor='w', pady=2)

        # ---- Variables ----
        s = sec('Variable')
        self._var_lb = ttk.Listbox(
            s, selectmode='single', height=7, exportselection=False,
            activestyle='none', relief='flat', borderwidth=0,
        )
        sb2 = ttk.Scrollbar(s, orient='vertical', command=self._var_lb.yview)
        self._var_lb.configure(yscrollcommand=sb2.set)
        self._var_lb.pack(side='left', fill='both', expand=True)
        sb2.pack(side='right', fill='y')

        # ---- Statistics ----
        s = sec('Statistics')
        self._stat_mode = tk.StringVar(value='none')
        for val, label in [
            ('none',        'None'),
            ('fluctuation', "Fluctuation  (u' = u_inst - u_t_avg)"),
            ('q_criterion', 'Q-criterion  (requires qx_ccc, qy_ccc, qz_ccc)'),
            ('vorticity',   'Vorticity  (requires qx_ccc, qy_ccc, qz_ccc)'),
        ]:
            ttk.Radiobutton(s, text=label, variable=self._stat_mode, value=val).pack(anchor='w', padx=4, pady=1)
        r_vc = ttk.Frame(s)
        r_vc.pack(fill='x', padx=4, pady=1)
        ttk.Label(r_vc, text='Vorticity component:').pack(side='left')
        self._vort_component = tk.StringVar(value='z')
        ttk.Combobox(r_vc, textvariable=self._vort_component, values=['x', 'y', 'z'],
                     state='readonly', width=4).pack(side='left', padx=4)

        # ---- Visualisation mode ----
        s = sec('Visualisation')
        self._mode = tk.StringVar(value='slice')
        row(s, 'Mode:', lambda r: ttk.Combobox(
            r, textvariable=self._mode,
            values=['slice', 'iso', 'volume'], state='readonly', width=16).pack(side='left'))

        self._cmap = tk.StringVar(value='RdBu_r')
        row(s, 'Colormap:', lambda r: ttk.Combobox(
            r, textvariable=self._cmap, values=COLORMAPS, width=14).pack(side='left'))

        self._vmin = tk.StringVar()
        self._vmax = tk.StringVar()
        r_scale = ttk.Frame(s)
        r_scale.pack(fill='x', pady=1)
        ttk.Label(r_scale, text='Colour scale:', width=14, anchor='w').pack(side='left')
        ttk.Entry(r_scale, textvariable=self._vmin, width=8).pack(side='left')
        ttk.Label(r_scale, text=' / ').pack(side='left')
        ttk.Entry(r_scale, textvariable=self._vmax, width=8).pack(side='left')
        ttk.Label(s, text='(vmin / vmax — blank = auto)', foreground='grey',
                  font=('TkDefaultFont', 8)).pack(anchor='w')

        self._stride = tk.IntVar(value=1)
        row(s, 'Stride:', lambda r: ttk.Spinbox(
            r, textvariable=self._stride, from_=1, to=16, width=6).pack(side='left'))

        # ---- Slice planes ----
        s = sec('Slice Planes')
        self._cut_x = tk.StringVar()
        self._cut_y = tk.StringVar()
        self._cut_z = tk.StringVar()
        row(s, 'x  (YZ plane):', lambda r: ttk.Entry(
            r, textvariable=self._cut_x, width=12).pack(side='left'))
        row(s, 'y  (XZ plane):', lambda r: ttk.Entry(
            r, textvariable=self._cut_y, width=12).pack(side='left'))
        row(s, 'z  (XY plane):', lambda r: ttk.Entry(
            r, textvariable=self._cut_z, width=12).pack(side='left'))

        # ---- Iso-surface ----
        s = sec('Iso-surface')
        self._iso_min = tk.StringVar()
        self._iso_max = tk.StringVar()
        self._iso_steps = tk.StringVar(value='1')
        row(s, 'Min:', lambda r: ttk.Entry(r, textvariable=self._iso_min, width=12).pack(side='left'))
        row(s, 'Max:', lambda r: ttk.Entry(r, textvariable=self._iso_max, width=12).pack(side='left'))
        row(s, 'Steps:', lambda r: ttk.Entry(r, textvariable=self._iso_steps, width=6).pack(side='left'))

        self._color_by = tk.StringVar(value='same')

        def _mk_color_by(r):
            self._color_by_combo = ttk.Combobox(
                r, textvariable=self._color_by,
                values=['same', 'Q criterion', 'vorticity', 'distance from wall'],
                state='readonly', width=16)
            self._color_by_combo.pack(side='left')
        row(s, 'Colour by:', _mk_color_by)
        self._color_vort_component = tk.StringVar(value='z')
        row(s, 'Colour vort. comp.:', lambda r: ttk.Combobox(
            r, textvariable=self._color_vort_component, values=['x', 'y', 'z'],
            state='readonly', width=4).pack(side='left'))

        # ---- Volume rendering ----
        s = sec('Volume Rendering')
        self._opacity = tk.StringVar(value='sigmoid')
        row(s, 'Opacity:', lambda r: ttk.Combobox(
            r, textvariable=self._opacity,
            values=_VIS_OPACITY, state='readonly', width=14).pack(side='left'))

        # ---- Screenshot ----
        s = sec('Screenshot')
        self._screenshot_path = tk.StringVar(value='visu_screenshot.png')
        r4 = ttk.Frame(s); r4.pack(fill='x', pady=1)
        ttk.Label(r4, text='Path:', width=14, anchor='w').pack(side='left')
        ttk.Entry(r4, textvariable=self._screenshot_path).pack(side='left', fill='x', expand=True)
        ttk.Button(r4, text='…', width=3, command=self._browse_screenshot).pack(side='left')
        ttk.Button(s, text='Save screenshot', command=self._save_screenshot).pack(anchor='w', pady=2)

    # ------ Helpers ------------------------------------------------------------------

    def _browse(self):
        d = filedialog.askdirectory(title='Select case folder (containing 2_visu/)')
        if d:
            self._case_path.set(d)
            self._scan()

    def _browse_screenshot(self):
        p = filedialog.asksaveasfilename(
            title='Screenshot path',
            defaultextension='.png',
            filetypes=[('PNG', '*.png'), ('JPEG', '*.jpg')],
        )
        if p:
            self._screenshot_path.set(p)

    def _visu_folder(self):
        base = self._case_path.get().rstrip('/')
        candidate = os.path.join(base, '2_visu')
        return candidate if os.path.isdir(candidate) else base

    def _xdmf_path(self):
        visu = self._visu_folder()
        ts = self._ts.get()
        dtype = self._dtype.get()
        phys = self._phys.get()
        name = (f'domain1_{phys}_{ts}.xdmf' if dtype == 'inst'
                else f'domain1_{dtype}_{phys}_{ts}.xdmf')
        return os.path.join(visu, name)

    def _scan(self):
        visu = self._visu_folder()
        try:
            from slice import get_available_timesteps
            tss = get_available_timesteps(visu)
            self._ts_combo['values'] = tss
            if tss:
                self._ts.set(tss[0])
            self._log(f'Found {len(tss)} timestep(s): {", ".join(tss)}')
        except Exception as exc:
            self._log(f'Scan error: {exc}')

    def _load_vars(self):
        xdmf = self._xdmf_path()
        self._log(f'Reading metadata: {xdmf}')
        try:
            from utils import parse_xdmf_metadata
            self._var_meta, self._grid_info = parse_xdmf_metadata(xdmf)
            names = sorted(v for v, m in self._var_meta.items()
                           if len(m.get('shape', ())) == 3)
            self._var_lb.delete(0, tk.END)
            for n in names:
                self._var_lb.insert(tk.END, n)
            self._color_by_combo['values'] = [
                'same', 'q_criterion', 'vorticity', 'wall_distance'] + names
            self._log(f'Loaded {len(names)} 3D variable(s).')

            # Show domain range as hints for slice plane entries
            gi = self._grid_info
            for axis, key in [('x', 'grid_x'), ('y', 'grid_y'), ('z', 'grid_z')]:
                arr = gi.get(key)
                if arr is not None:
                    mid = 0.5 * (float(arr[0]) + float(arr[-1]))
                    self._log(
                        f'  {axis} range: {arr[0]:.4f} – {arr[-1]:.4f}  (mid = {mid:.4f})')
                    getattr(self, f'_cut_{axis}').set(f'{mid:.4f}')
        except Exception as exc:
            self._log(f'Error: {exc}\n{traceback.format_exc()}')

    def _selected_var(self):
        sel = self._var_lb.curselection()
        return self._var_lb.get(sel[0]) if sel else None

    # ------ Render -------------------------------------------------------------------

    def _render(self):
        stat_mode = self._stat_mode.get()
        use_q    = (stat_mode == 'q_criterion')
        use_fluc = (stat_mode == 'fluctuation')
        use_vort = (stat_mode == 'vorticity')
        variable = self._selected_var()
        if not use_q and not use_vort and variable is None:
            messagebox.showwarning('No variable', 'Select a variable.')
            return
        if not self._var_meta:
            messagebox.showwarning('No metadata', 'Load variables first.')
            return

        mode   = self._mode.get()
        cmap   = self._cmap.get()
        stride = max(1, self._stride.get())

        if mode == 'volume':
            import turb_visu as tv
            predicted = tv.strided_cell_count(self._grid_info, stride)
            if predicted is not None and predicted > tv.VOLUME_CELL_THRESHOLD:
                messagebox.showwarning(
                    'Grid too large',
                    f'Volume rendering at stride={stride} would need ~{predicted:,} cells '
                    f'(limit {tv.VOLUME_CELL_THRESHOLD:,}). Increase the stride and try again.',
                )
                return

        if use_q or use_vort:
            selected_vars = list({'qx_ccc', 'qy_ccc', 'qz_ccc'})
        else:
            selected_vars = [variable]

        # Colour-by (iso-surfaces only): colour the surface by a different
        # field than the one that defines its geometry.
        color_by_choice = self._color_by.get()
        color_by = None if color_by_choice == 'same' else color_by_choice
        if color_by in ('q_criterion', 'vorticity'):
            selected_vars = list({'qx_ccc', 'qy_ccc', 'qz_ccc'} | set(selected_vars))
        elif color_by and color_by != 'wall_distance':
            # wall_distance is purely geometric — nothing extra to load.
            selected_vars = list({color_by} | set(selected_vars))

        def _parse_float(s, fallback):
            try:
                return float(s.strip())
            except (ValueError, AttributeError):
                return fallback

        gi = self._grid_info

        # t_avg xdmf path — used when fluctuation is requested
        t_avg_xdmf = os.path.join(
            self._visu_folder(),
            f'domain1_t_avg_{self._phys.get()}_{self._ts.get()}.xdmf',
        )

        # Common statistics keys added to every cfg
        stats = {
            'use_q_criterion': use_q,
            'use_fluc': use_fluc,
            't_avg_xdmf': t_avg_xdmf,
            'use_vorticity': use_vort,
            'vorticity_component': self._vort_component.get(),
            'color_by': color_by,
            'color_vorticity_component': self._color_vort_component.get(),
            'vmin': _parse_float(self._vmin.get(), None),
            'vmax': _parse_float(self._vmax.get(), None),
        }

        if mode == 'slice':
            def _mid(key):
                arr = gi.get(key)
                return 0.5 * (float(arr[0]) + float(arr[-1])) if arr is not None else None
            cfg = {
                'mode': 'slice',
                'variable': variable,
                'cmap': cmap,
                **stats,
                'cut_x': _parse_float(self._cut_x.get(), _mid('grid_x')),
                'cut_y': _parse_float(self._cut_y.get(), _mid('grid_y')),
                'cut_z': _parse_float(self._cut_z.get(), _mid('grid_z')),
            }
        elif mode == 'iso':
            try:
                iso_steps = max(1, int(self._iso_steps.get() or '1'))
            except ValueError:
                iso_steps = 1
            cfg = {
                'mode': 'iso',
                'variable': variable,
                'cmap': cmap,
                **stats,
                'iso_min': _parse_float(self._iso_min.get(), None),
                'iso_max': _parse_float(self._iso_max.get(), None),
                'iso_steps': iso_steps,
            }
        elif mode == 'volume':
            cfg = {
                'mode': 'volume',
                'variable': variable,
                'cmap': cmap,
                **stats,
                'opacity': self._opacity.get(),
            }
        var_meta = dict(self._var_meta)

        def worker():
            try:
                import turb_visu as tv
                from utils import load_xdmf_variables, parse_xdmf_metadata

                self._log(f'Loading {selected_vars}…')

                # Re-load only when the dataset identity changes. `data` is a
                # shallow copy so derived fields (Q-criterion, vorticity, wall
                # distance, fluctuation) cannot leak into the retained base.
                key = (self._xdmf_path(), tuple(sorted(selected_vars)), stride)
                if self._data is not None and key == self._data_key:
                    self._log('Reusing loaded data (dataset unchanged).')
                    data = dict(self._data)
                else:
                    data = load_xdmf_variables(var_meta, selected_vars, grid_info=gi, stride=stride)
                    if data:
                        self._data = dict(data)
                        self._data_key = key
                if not data:
                    self._log('Error: failed to load data.')
                    return

                import operations as op
                if use_q:
                    self._log('Computing Q-criterion…')
                    # Striding node arrays (len ncells+1) and cell arrays (len
                    # ncells) by the same `stride` can land one cell apart, so
                    # clip data to the node-derived count — same convention
                    # build_pyvista_grid uses — before differentiating.
                    q_grid_info = tv.strided_grid_info(gi, stride)
                    nz = len(q_grid_info['grid_z']) - 1
                    ny = len(q_grid_info['grid_y']) - 1
                    nx = len(q_grid_info['grid_x']) - 1
                    q_data = {k: v[:nz, :ny, :nx] for k, v in data.items()}
                    q = op.compute_q_criterion(q_data, q_grid_info)
                    if q is None:
                        return
                    data['Q-criterion'] = q
                    cfg['variable'] = 'Q-criterion'
                elif use_vort:
                    component = cfg.get('vorticity_component', 'z')
                    self._log(f'Computing vorticity (ω_{component})…')
                    # Same stride-alignment clipping as Q-criterion above.
                    v_grid_info = tv.strided_grid_info(gi, stride)
                    nz = len(v_grid_info['grid_z']) - 1
                    ny = len(v_grid_info['grid_y']) - 1
                    nx = len(v_grid_info['grid_x']) - 1
                    v_data = {k: v[:nz, :ny, :nx] for k, v in data.items()}
                    vorticity = op.compute_vorticity(v_data, v_grid_info, component)
                    if vorticity is None:
                        return
                    vort_name = f'Vorticity_{component}'
                    data[vort_name] = vorticity
                    cfg['variable'] = vort_name
                elif use_fluc:
                    self._log(f"Computing fluctuation {variable}'…")
                    t_avg_meta, _ = parse_xdmf_metadata(cfg['t_avg_xdmf'])
                    t_avg_var = op.INST_TO_TAVG_VAR.get(variable, variable)
                    t_avg_data = load_xdmf_variables(t_avg_meta, [t_avg_var], grid_info=gi, stride=stride)
                    fluc_name = f"{variable}'"
                    data[fluc_name] = op.compute_inst_fluc(data[variable], t_avg_data[t_avg_var])
                    cfg['variable'] = fluc_name

                # Colour-by field (iso-surfaces): a second, independent
                # scalar used only for colouring the extracted surface, not
                # for defining its geometry.
                color_by = cfg.get('color_by')
                color_field_name = None
                if color_by == 'q_criterion':
                    color_field_name = 'Q-criterion'
                    if color_field_name not in data:
                        self._log('Computing Q-criterion (colour)…')
                        c_grid_info = tv.strided_grid_info(gi, stride)
                        nz = len(c_grid_info['grid_z']) - 1
                        ny = len(c_grid_info['grid_y']) - 1
                        nx = len(c_grid_info['grid_x']) - 1
                        c_data = {k: v[:nz, :ny, :nx] for k, v in data.items()}
                        q = op.compute_q_criterion(c_data, c_grid_info)
                        if q is None:
                            return
                        data[color_field_name] = q
                elif color_by == 'vorticity':
                    color_component = cfg.get('color_vorticity_component', 'z')
                    color_field_name = f'Vorticity_{color_component}'
                    if color_field_name not in data:
                        self._log(f'Computing vorticity (colour, ω_{color_component})…')
                        c_grid_info = tv.strided_grid_info(gi, stride)
                        nz = len(c_grid_info['grid_z']) - 1
                        ny = len(c_grid_info['grid_y']) - 1
                        nx = len(c_grid_info['grid_x']) - 1
                        c_data = {k: v[:nz, :ny, :nx] for k, v in data.items()}
                        vorticity = op.compute_vorticity(c_data, c_grid_info, color_component)
                        if vorticity is None:
                            return
                        data[color_field_name] = vorticity
                elif color_by == 'wall_distance':
                    self._log('Computing distance from the wall (colour)…')
                    color_field_name = tv.WALL_DISTANCE_FIELD
                    data[color_field_name] = tv.compute_wall_distance(gi, stride)
                elif color_by:
                    color_field_name = color_by

                if color_field_name and color_field_name != cfg['variable']:
                    cfg['color_variable'] = color_field_name

                self._log('Building grid…')
                grid = tv.build_pyvista_grid(gi, data, stride=stride)
                self._log(f'Grid: {grid.dimensions}, {grid.n_cells:,} cells')

                if mode == 'iso':
                    import numpy as _np
                    iso_variable = cfg['variable']
                    arr = grid.cell_data[iso_variable]
                    vmin, vmax = float(arr.min()), float(arr.max())
                    self._log(f'  {iso_variable} range: {vmin:.4e} – {vmax:.4e}')
                    iso_min = cfg['iso_min'] if cfg['iso_min'] is not None else 0.5 * (vmin + vmax)
                    iso_steps = cfg['iso_steps']
                    if iso_steps > 1:
                        iso_max = cfg['iso_max'] if cfg['iso_max'] is not None else vmax
                        cfg['iso_vals'] = list(_np.linspace(iso_min, iso_max, iso_steps))
                    else:
                        cfg['iso_vals'] = [iso_min]
                    self._log(f'  Iso-values: {[f"{v:.4e}" for v in cfg["iso_vals"]]}')

                # Rendering must happen on the main (Tk) thread.
                self._log('Rendering…')
                self.after(0, lambda g=grid, c=dict(cfg): self._visu_panel.render(g, c))
            except Exception:
                self._log(f'Error:\n{traceback.format_exc()}')

        threading.Thread(target=worker, daemon=True).start()

    def _save_screenshot(self):
        path = self._screenshot_path.get().strip() or 'visu_screenshot.png'
        try:
            self._visu_panel.save_screenshot(path)
            self._log(f'Screenshot saved: {path}')
        except Exception as exc:
            self._log(f'Screenshot error: {exc}')


class LinkedSlider(ttk.Frame):
    """A slider and an entry bound to one variable, kept in sync.

    Typing a value outside the slider's range is allowed and leaves the slider
    pinned at its end stop, so the range is a convenience rather than a limit.

    `step` snaps values dragged on the slider to a multiple of that increment
    (ttk.Scale, unlike tk.Scale, has no resolution option). Values typed into
    the entry are left exactly as entered, so the step constrains the slider
    without limiting what can be set.
    """

    def __init__(self, parent, label, var, lo, hi, integer=True, step=None,
                 label_width=15, entry_width=8, on_change=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._var = var
        self._lo, self._hi = lo, hi
        self._integer = integer
        self._step = step
        self._on_change = on_change
        self._syncing = False

        ttk.Label(self, text=label, width=label_width, anchor='w').pack(side='left')
        ttk.Entry(self, textvariable=var, width=entry_width).pack(side='right')
        self._scale = ttk.Scale(self, from_=lo, to=hi, orient='horizontal',
                                command=self._from_scale)
        self._scale.pack(side='left', fill='x', expand=True, padx=(0, 6))

        self._push_to_scale()
        var.trace_add('write', lambda *_: self._from_var())

    def _from_scale(self, value):
        if self._syncing:
            return
        value = float(value)
        if self._step:
            # Snap to the step, but never past the ends of the range.
            value = min(max(round(value / self._step) * self._step, self._lo), self._hi)
        value = int(round(value)) if self._integer else round(value, 4)
        if self._current() == value:
            return
        self._syncing = True
        try:
            self._var.set(value)
        finally:
            self._syncing = False
        if self._on_change:
            self._on_change()

    def _from_var(self):
        if self._syncing:
            return
        self._push_to_scale()
        if self._on_change:
            self._on_change()

    def _current(self):
        try:
            return self._var.get()
        except (tk.TclError, ValueError):
            return None

    def _push_to_scale(self):
        value = self._current()
        if value is None:
            return
        self._syncing = True
        try:
            self._scale.set(min(max(float(value), self._lo), self._hi))
        finally:
            self._syncing = False


class MeshAnalysisTab(ttk.Frame):
    """Interactive pre-run mesh resolution assessment.

    Wraps mesh_analysis.py: the mesh parameters are driven from sliders, the
    wall-normal grid is rebuilt exactly as the solver builds it, and the
    resolution report and spacing plot refresh as the values change.
    """

    # Label -> value for the readonly combos
    CASES = {'channel': ma.ICASE_CHANNEL, 'pipe': ma.ICASE_PIPE,
             'annular': ma.ICASE_ANNULAR}
    ISTRETS = {'uniform': ma.ISTRET_NO, 'two-side clustered': ma.ISTRET_2SIDES,
               'bottom clustered': ma.ISTRET_BOTTOM, 'top clustered': ma.ISTRET_TOP}
    MSTRETS = {'3fmd': ma.MSTRET_3FMD, 'tanh': ma.MSTRET_TANH, 'power law': ma.MSTRET_POWL}
    FLUIDS = {name: idx for idx, name in ma.FLUID_NAMES.items()}

    def __init__(self, parent):
        super().__init__(parent)
        self._template = None      # text of a loaded input file, for regeneration
        self._last = None          # (cfg, yp, res) of the most recent successful run
        self._update_job = None
        self._building = True
        self._build_ui()
        self._building = False
        self._schedule_update()

    # ------ Layout -------------------------------------------------------------------

    def _build_ui(self):
        pw = ttk.Panedwindow(self, orient='horizontal')
        pw.pack(fill='both', expand=True)

        left = ttk.Frame(pw, width=380)
        left.pack_propagate(False)
        pw.add(left, weight=0)

        right = ttk.Frame(pw)
        pw.add(right, weight=1)

        self._build_controls(left)

        # The metrics bar spans the full width above both panes.
        self._build_metrics(right)

        content = ttk.Panedwindow(right, orient='horizontal')
        content.pack(fill='both', expand=True)

        plot_pane = ttk.Frame(content)
        content.add(plot_pane, weight=3)
        self._build_figure(plot_pane)

        report_pane = ttk.Frame(content)
        content.add(report_pane, weight=2)
        ttk.Label(report_pane, text='Resolution report:').pack(anchor='w', padx=4, pady=(4, 0))
        # hbar=True also sets wrap='none': the report is fixed-width text with
        # 100-character rules, which would look broken re-wrapped in this pane.
        self._report = ttk.ScrolledText(
            report_pane, height=16, state='disabled',
            hbar=True, font=('Monospace', 8),
        )
        self._report.pack(fill='both', expand=True, padx=4, pady=2)

    def _build_figure(self, parent):
        # A persistent canvas, unlike FigurePanel which rebuilds on every show:
        # the sliders redraw continuously, so the canvas is created once and
        # only its contents are replaced.
        self._fig = Figure(figsize=(6, 7))
        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        toolbar = NavigationToolbar2Tk(self._canvas, parent)
        toolbar.update()
        self._canvas.get_tk_widget().pack(fill='both', expand=True)

    def _build_metrics(self, parent):
        s = ttk.Frame(parent)
        s.pack(fill='x', padx=4, pady=(4, 0))
        self._metrics = {}
        for i, (key, label) in enumerate([('re_tau', 'Re_tau'), ('dyplus', 'dy+ wall'),
                                          ('dycentre', 'dy+ centre'),
                                          ('dxplus', 'dx+'), ('dzplus', 'dz+'),
                                          ('growth', 'max growth'), ('diffnum', 'diff. number'),
                                          ('cells', 'total cells')]):
            cell = ttk.Frame(s)
            cell.grid(row=0, column=i, sticky='ew', padx=3)
            s.columnconfigure(i, weight=1)
            ttk.Label(cell, text=label, anchor='center',
                      font=('TkDefaultFont', 8)).pack(fill='x')
            value = ttk.Label(cell, text='-', anchor='center',
                              font=('TkDefaultFont', 12, 'bold'))
            value.pack(fill='x')
            self._metrics[key] = value

    # ------ Controls (left) ----------------------------------------------------------

    def _build_controls(self, parent):
        scroller = ScrollableFrame(parent)
        scroller.pack(fill='both', expand=True)
        f = scroller.inner

        def combo_row(frame, label, var, values):
            r = ttk.Frame(frame)
            r.pack(fill='x', pady=1)
            ttk.Label(r, text=label, width=15, anchor='w').pack(side='left')
            c = ttk.Combobox(r, textvariable=var, values=values, state='readonly', width=18)
            c.pack(side='left', fill='x', expand=True)
            c.bind('<<ComboboxSelected>>', lambda _e: self._schedule_update())
            return c

        def entry_row(frame, label, var, width=12):
            r = ttk.Frame(frame)
            r.pack(fill='x', pady=1)
            ttk.Label(r, text=label, width=15, anchor='w').pack(side='left')
            e = ttk.Entry(r, textvariable=var, width=width)
            e.pack(side='left', fill='x', expand=True)
            return e

        def slider_row(frame, label, var, lo, hi, integer=True, step=None):
            w = LinkedSlider(frame, label, var, lo, hi, integer=integer, step=step,
                             on_change=self._schedule_update)
            w.pack(fill='x', pady=2)
            return w

        # -- Input file --
        s = ttk.Labelframe(f, text='Input File', padding=(8, 6))
        s.pack(fill='x', padx=4, pady=3)
        self._path = tk.StringVar()
        r = ttk.Frame(s)
        r.pack(fill='x', pady=1)
        ttk.Entry(r, textvariable=self._path).pack(side='left', fill='x', expand=True)
        ttk.Button(r, text='…', width=3, command=self._browse).pack(side='left')
        r2 = ttk.Frame(s)
        r2.pack(fill='x', pady=2)
        ttk.Button(r2, text='Load', command=self._load).pack(side='left', fill='x', expand=True)
        ttk.Button(r2, text='Generate…', command=self._generate).pack(side='left', fill='x',
                                                                     expand=True)

        # -- Case and domain --
        s = ttk.Labelframe(f, text='Case & Domain', padding=(8, 6))
        s.pack(fill='x', padx=4, pady=3)
        self._case = tk.StringVar(value='channel')
        combo_row(s, 'Case:', self._case, list(self.CASES))
        self._case.trace_add('write', lambda *_: self._on_case_change())

        self._lxx = tk.DoubleVar(value=8.0)
        slider_row(s, 'Lx:', self._lxx, 1.0, 80.0, integer=False)
        self._lzz = tk.DoubleVar(value=4.0)
        self._lzz_slider = slider_row(s, 'Lz:', self._lzz, 1.0, 25.0, integer=False)
        self._lyb = tk.DoubleVar(value=-1.0)
        self._lyb_entry = entry_row(s, 'y bottom:', self._lyb)
        self._lyt = tk.DoubleVar(value=1.0)
        self._lyt_entry = entry_row(s, 'y top:', self._lyt)
        self._lyb.trace_add('write', lambda *_: self._schedule_update())
        self._lyt.trace_add('write', lambda *_: self._schedule_update())
        self._extent_note = ttk.Label(s, text='', anchor='w', font=('TkDefaultFont', 8))
        self._extent_note.pack(fill='x', pady=(2, 0))

        # -- Mesh --
        s = ttk.Labelframe(f, text='Mesh', padding=(8, 6))
        s.pack(fill='x', padx=4, pady=3)
        self._ncx = tk.IntVar(value=64)
        slider_row(s, 'Ncx:', self._ncx, 8, 2048, step=10)
        self._ncy = tk.IntVar(value=80)
        slider_row(s, 'Ncy:', self._ncy, 8, 2048, step=10)
        self._ncz = tk.IntVar(value=64)
        slider_row(s, 'Ncz:', self._ncz, 8, 2048, step=10)
        ttk.Label(s, text='Sliders step in 10s; type an exact count in the box.',
                  anchor='w', font=('TkDefaultFont', 8)).pack(fill='x', pady=(2, 0))

        self._istret = tk.StringVar(value='two-side clustered')
        combo_row(s, 'Clustering:', self._istret, list(self.ISTRETS))
        self._mstret = tk.StringVar(value='3fmd')
        combo_row(s, 'Stretch func.:', self._mstret, list(self.MSTRETS))
        self._rstret = tk.DoubleVar(value=0.1)
        slider_row(s, 'Stretch factor:', self._rstret, 0.001, 1.0, integer=False)
        ttk.Label(s, text='Smaller stretch factor = stronger near-wall clustering.',
                  anchor='w', font=('TkDefaultFont', 8)).pack(fill='x', pady=(2, 0))

        # -- Flow --
        s = ttk.Labelframe(f, text='Flow', padding=(8, 6))
        s.pack(fill='x', padx=4, pady=3)
        self._ren = tk.DoubleVar(value=5000.0)
        entry_row(s, 'Reynolds no.:', self._ren)
        self._dt = tk.StringVar(value='1e-03')
        entry_row(s, 'Time step dt:', self._dt)
        self._ren.trace_add('write', lambda *_: self._schedule_update())
        self._dt.trace_add('write', lambda *_: self._schedule_update())

        # -- Thermo --
        s = ttk.Labelframe(f, text='Thermal', padding=(8, 6))
        s.pack(fill='x', padx=4, pady=3)
        self._thermo = tk.BooleanVar(value=False)
        ttk.Checkbutton(s, text=' Solve thermal field', variable=self._thermo,
                        bootstyle='round-toggle',
                        command=self._schedule_update).pack(anchor='w', pady=1)
        self._fluid = tk.StringVar(value='lithium')
        combo_row(s, 'Fluid:', self._fluid, list(self.FLUIDS))
        self._ref_t0 = tk.DoubleVar(value=570.0)
        entry_row(s, 'Ref. temp. (K):', self._ref_t0)
        self._ref_t0.trace_add('write', lambda *_: self._schedule_update())

        # -- MHD --
        s = ttk.Labelframe(f, text='MHD', padding=(8, 6))
        s.pack(fill='x', padx=4, pady=3)
        self._mhd = tk.BooleanVar(value=False)
        ttk.Checkbutton(s, text=' Solve MHD', variable=self._mhd,
                        bootstyle='round-toggle',
                        command=self._schedule_update).pack(anchor='w', pady=1)
        self._hartmann = tk.DoubleVar(value=30.0)
        slider_row(s, 'Hartmann no.:', self._hartmann, 1.0, 500.0, integer=False)

        # -- Actions --
        s = ttk.Frame(f)
        s.pack(fill='x', padx=4, pady=6)
        self._auto = tk.BooleanVar(value=True)
        ttk.Checkbutton(s, text=' Update automatically', variable=self._auto,
                        bootstyle='round-toggle').pack(anchor='w', pady=2)
        ttk.Button(s, text='Update', command=self._update).pack(fill='x', pady=2)
        ttk.Button(s, text='Save Plot…', command=self._save_plot).pack(fill='x', pady=2)

    # ------ Reading the controls -----------------------------------------------------

    def _on_case_change(self):
        """Grey out the extents the selected case overrides."""
        case = self.CASES.get(self._case.get(), ma.ICASE_CHANNEL)

        if case == ma.ICASE_CHANNEL:
            note = 'Channel: y fixed to [-1, 1] by the solver.'
            lyb_state, lyt_state, lzz_state = 'disabled', 'disabled', 'normal'
        elif case == ma.ICASE_PIPE:
            note = 'Pipe: y fixed to [0, 1] and Lz to 2*pi by the solver.'
            lyb_state, lyt_state, lzz_state = 'disabled', 'disabled', 'disabled'
        else:
            note = 'Annular: y top fixed to 1 and Lz to 2*pi; set y bottom (inner radius).'
            lyb_state, lyt_state, lzz_state = 'normal', 'disabled', 'disabled'

        self._extent_note.configure(text=note)
        self._lyb_entry.configure(state=lyb_state)
        self._lyt_entry.configure(state=lyt_state)
        for child in self._lzz_slider.winfo_children():
            try:
                child.configure(state=lzz_state)
            except tk.TclError:
                pass

        self._schedule_update()

    def _build_config(self):
        """Assemble a DomainConfig from the current control values."""
        fluid = self.FLUIDS.get(self._fluid.get(), 8)

        return ma.DomainConfig.from_values(
            icase=self.CASES.get(self._case.get(), ma.ICASE_CHANNEL),
            lxx=float(self._lxx.get()),
            lyb=float(self._lyb.get()),
            lyt=float(self._lyt.get()),
            lzz=float(self._lzz.get()),
            nc=[int(self._ncx.get()), int(self._ncy.get()), int(self._ncz.get())],
            istret=self.ISTRETS.get(self._istret.get(), ma.ISTRET_NO),
            mstret=self.MSTRETS.get(self._mstret.get(), ma.MSTRET_3FMD),
            rstret=float(self._rstret.get()),
            ren=float(self._ren.get()),
            dt=float(str(self._dt.get()).replace('d', 'e')),
            is_thermo=bool(self._thermo.get()),
            ifluid=fluid,
            ref_t0=float(self._ref_t0.get()),
            is_mhd=bool(self._mhd.get()),
            hartmann=float(self._hartmann.get()),
        )

    def _apply_config(self, cfg):
        """Push a loaded configuration back into the controls."""
        inverse = {v: k for k, v in self.CASES.items()}
        self._case.set(inverse.get(cfg.icase, 'channel'))
        self._lxx.set(round(cfg.lxx, 4))
        self._lzz.set(round(cfg.lzz, 4))
        self._lyb.set(round(cfg.lyb, 4))
        self._lyt.set(round(cfg.lyt, 4))
        self._ncx.set(cfg.nc[0])
        self._ncy.set(cfg.nc[1])
        self._ncz.set(cfg.nc[2])
        self._istret.set({v: k for k, v in self.ISTRETS.items()}.get(cfg.istret, 'uniform'))
        self._mstret.set({v: k for k, v in self.MSTRETS.items()}.get(cfg.mstret, '3fmd'))
        self._rstret.set(round(cfg.rstret, 4))
        self._ren.set(cfg.ren)
        self._dt.set(f'{cfg.dt:g}')
        self._thermo.set(cfg.is_thermo)
        if cfg.ifluid in ma.FLUID_NAMES:
            self._fluid.set(ma.FLUID_NAMES[cfg.ifluid])
        if cfg.ref_t0 is not None:
            self._ref_t0.set(cfg.ref_t0)
        self._mhd.set(cfg.is_mhd)
        if cfg.hartmann:
            self._hartmann.set(cfg.hartmann)

    # ------ File actions -------------------------------------------------------------

    def _browse(self):
        p = filedialog.askopenfilename(
            title='Select input_chapsim.ini',
            filetypes=[('CHAPSim2 input', '*.ini'), ('All files', '*')])
        if p:
            self._path.set(p)
            self._load()

    def _load(self):
        path = self._path.get().strip()
        if not path:
            messagebox.showwarning('Mesh Analysis', 'Choose an input file first.')
            return
        if not os.path.isfile(path):
            messagebox.showerror('Mesh Analysis', f'File not found:\n{path}')
            return

        try:
            with open(path, 'r') as fh:
                self._template = fh.read()
            cfg = ma.DomainConfig(ma.parse_input_file(path), path)
        except Exception as exc:
            messagebox.showerror('Mesh Analysis', f'Could not read input file:\n{exc}')
            return

        if not cfg.is_wall_bounded:
            messagebox.showwarning(
                'Mesh Analysis',
                f"Case '{ma.CASE_NAMES.get(cfg.icase, cfg.icase)}' is not wall bounded.\n"
                'Only channel, pipe and annular cases can be assessed.')
            return

        self._building = True
        try:
            self._apply_config(cfg)
        finally:
            self._building = False
        self._on_case_change()

    def _generate(self):
        """Write the current settings to an input file."""
        try:
            cfg = self._build_config()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror('Mesh Analysis', f'Invalid parameter value:\n{exc}')
            return

        path = filedialog.asksaveasfilename(
            title='Write input_chapsim.ini',
            initialfile='input_chapsim.ini',
            defaultextension='.ini',
            filetypes=[('CHAPSim2 input', '*.ini'), ('All files', '*')])
        if not path:
            return

        try:
            ma.write_input_file(cfg, path, template=self._template)
        except Exception as exc:
            messagebox.showerror('Mesh Analysis', f'Could not write input file:\n{exc}')
            return

        self._path.set(path)
        source = 'loaded file' if self._template else 'built-in channel template'
        messagebox.showinfo(
            'Mesh Analysis',
            f'Input file written to:\n{path}\n\n'
            f'Based on the {source}; only the mesh, domain, flow, thermal and MHD '
            'values shown here were changed. Check the remaining sections '
            '(boundary conditions, io, probes) before running.')

    def _save_plot(self):
        if self._last is None:
            messagebox.showwarning('Mesh Analysis', 'Nothing to save yet.')
            return
        path = filedialog.asksaveasfilename(
            title='Save plot', initialfile='mesh_analysis.png', defaultextension='.png',
            filetypes=[('PNG', '*.png'), ('PDF', '*.pdf'), ('All files', '*')])
        if not path:
            return
        try:
            self._fig.savefig(path, dpi=300, bbox_inches='tight')
        except Exception as exc:
            messagebox.showerror('Mesh Analysis', f'Could not save plot:\n{exc}')
            return
        messagebox.showinfo('Mesh Analysis', f'Plot saved to:\n{path}')

    # ------ Update -------------------------------------------------------------------

    def _schedule_update(self):
        """Debounce updates so dragging a slider does not redraw continuously."""
        if self._building or not self._auto.get():
            return
        if self._update_job is not None:
            self.after_cancel(self._update_job)
        self._update_job = self.after(180, self._update)

    def _update(self):
        self._update_job = None
        try:
            cfg = self._build_config()
        except (ValueError, tk.TclError) as exc:
            self._set_report(f'Invalid parameter value: {exc}')
            return

        if cfg.nc[1] < 5:
            self._set_report('Ncy must be at least 5 to assess the wall-normal grid.')
            return
        if cfg.ren <= 0.0:
            self._set_report('The Reynolds number must be positive.')
            return

        import contextlib
        import io as _io

        buffer = _io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                yp, _yc = ma.build_y_grid(cfg)
                ma.check_y_grid(yp, cfg)
                res = ma.analyse_spatial_resolution(cfg, yp, _yc)
                diff = ma.analyse_diffusion_number(cfg, _yc)
                ma.analyse_temporal_resolution(cfg, res, diff)
        except Exception:
            self._set_report(f'{buffer.getvalue()}\n\n{traceback.format_exc()}')
            return

        self._last = (cfg, yp, res)
        self._sync_derived(cfg)
        self._set_report(buffer.getvalue())
        self._update_metrics(cfg, yp, res, diff)

        try:
            ma.draw_mesh_distribution(self._fig, cfg, yp, res)
            self._canvas.draw_idle()
        except Exception:
            self._set_report(f'{buffer.getvalue()}\n\nPlot error:\n{traceback.format_exc()}')

    def _sync_derived(self, cfg):
        """Show the values the solver actually derives.

        The case overrides the y extents and, for cylindrical cases, Lz; the
        even-cell rule can also bump Ncz. Writing them back keeps the controls
        honest about what will be run.
        """
        self._building = True
        try:
            for var, value in ((self._lyb, cfg.lyb), (self._lyt, cfg.lyt),
                               (self._lzz, cfg.lzz)):
                if abs(var.get() - value) > 1e-12:
                    var.set(round(value, 6))
            if self._ncz.get() != cfg.nc[2]:
                self._ncz.set(cfg.nc[2])
        except (tk.TclError, ValueError):
            pass
        finally:
            self._building = False

    def _update_metrics(self, cfg, yp, res, diff=None):
        """Refresh the headline numbers, colour-coded against the DNS limits."""
        dy = np.diff(yp)
        growth = np.maximum(dy[1:] / dy[:-1], dy[:-1] / dy[1:]).max()

        def style(value, limit):
            """Green within the limit, amber up to 50% over it, red beyond."""
            if value > 1.5 * limit:
                return 'danger'
            return 'warning' if value > limit else 'success'

        def growth_style(value):
            # Growth is a ratio about 1, so "50% over the limit" does not carry
            # over — 1.5 x 1.3 would call a ruinous 1.9 acceptable. The solver's
            # own caution/limit pair is used instead.
            if value > 1.3:
                return 'danger'
            return 'warning' if value > 1.2 else 'success'

        cells = cfg.nc[0] * cfg.nc[1] * cfg.nc[2]
        # Not res['yplus1'] — for a pipe that is the axis spacing, not the wall.
        dyplus = ma.wall_dyplus(cfg, res)
        entries = {
            're_tau': (f"{res['Re_tau']:.1f}", 'secondary'),
            'dyplus': (f"{dyplus:.2f}", style(dyplus, ma.DYPLUS_MAX)),
            # Graded against the DNS rule of thumb, not an apx_prerun_mod limit.
            'dycentre': (f"{res['yplus2']:.2f}", style(res['yplus2'],
                                                      ma.DYPLUS_CENTRE_MAX)),
            'dxplus': (f"{res['dxplus']:.1f}", style(res['dxplus'], ma.DXPLUS_MAX)),
            'dzplus': (f"{res['dzplus']:.1f}", style(res['dzplus'], ma.DZPLUS_MAX)),
            'growth': (f"{growth:.3f}", growth_style(growth)),
            # Above 1 the solver warns of possible instability, so 1 is the limit.
            'diffnum': ((f"{diff['diff_mom']:.3g}", style(diff['diff_mom'], 1.0))
                        if diff else ('-', 'secondary')),
            'cells': (f"{cells / 1e6:.2f}M" if cells >= 1e6 else f"{cells:,}", 'secondary'),
        }
        for key, (text, bootstyle) in entries.items():
            self._metrics[key].configure(text=text, bootstyle=bootstyle)

    def _set_report(self, text):
        self._report.configure(state='normal')
        self._report.delete('1.0', tk.END)
        self._report.insert(tk.END, text)
        self._report.see('1.0')
        self._report.configure(state='disabled')


# =====================================================================================
# APPLICATION
# =====================================================================================

class LazyTab(ttk.Frame):
    """Notebook page whose contents are built the first time it is shown.

    Building all five tabs up front dominated startup, and most of that work
    was for pages the user may never open. The page is registered with the
    notebook immediately, so tab order and labels are unaffected, but the tab
    class is not instantiated until the page is first selected.
    """

    def __init__(self, parent, factory, **kwargs):
        super().__init__(parent, **kwargs)
        self._factory = factory
        self.inner = None

    def realise(self):
        """Build the page if it has not been built yet."""
        if self.inner is None:
            self.inner = self._factory(self)
            self.inner.pack(fill='both', expand=True)

        return self.inner


class App(ttk.Window):

    TABS = [
        ('  Mesh Analysis  ',         lambda: MeshAnalysisTab),
        ('  Monitoring Points  ',     lambda: MonitorPointsTab),
        ('  Slice Visualisation  ',   lambda: SliceTab),
        ('  3D Visualisation  ',      lambda: TurbVisuTab),
        ('  Turbulence Statistics  ', lambda: TurbStatsTab),
    ]

    def __init__(self):
        super().__init__(theme='pydata-dark')
        self.title('CHAPSim2 Toolkit')
        self.geometry('1400x820')
        self.minsize(920, 600)
        self.protocol('WM_DELETE_WINDOW', self._on_close)
        self._build()

    def _build(self):
        # Console drawer on the right: laid out before the notebook so the
        # notebook's fill/expand yields the strip only the width it needs.
        self.console = ConsolePanel(self)
        self.console.pack(side='right', fill='y')

        nb = ttk.Notebook(self)
        nb.pack(side='left', fill='both', expand=True)
        self._nb = nb
        self._holders = []

        for text, factory in self.TABS:
            holder = LazyTab(nb, lambda parent, f=factory: f()(parent))
            nb.add(holder, text=text)
            self._holders.append(holder)

        nb.bind('<<NotebookTabChanged>>', self._on_tab_changed)
        # The event above may already have fired for the initial selection
        # while the binding was not yet in place, so realise it explicitly.
        self._realise_current()

        # One app-wide redirect: TextRedirect marshals writes onto the Tk
        # thread, so print() from worker threads (and from any tab's tools)
        # lands in the console. Restored in _on_close.
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = TextRedirect(self.console)
        sys.stderr = TextRedirect(self.console)

    def _on_tab_changed(self, _event=None):
        self._realise_current()

    def _realise_current(self):
        """Build the selected page, showing a busy cursor while it happens."""
        try:
            holder = self._nb.nametowidget(self._nb.select())
        except (tk.TclError, KeyError):
            return
        if not isinstance(holder, LazyTab) or holder.inner is not None:
            return

        self.configure(cursor='watch')
        try:
            self.update_idletasks()
        except tk.TclError:
            pass
        try:
            holder.realise()
        finally:
            self.configure(cursor='')

    def _on_close(self):
        # Restore the real streams so late writes (and interpreter shutdown)
        # don't target a destroyed widget.
        try:
            sys.stdout, sys.stderr = self._stdout, self._stderr
        except AttributeError:
            pass
        # The PyVista/VTK off-screen plotter holds a GL context that must be
        # closed explicitly while the display connection is still alive.
        # Without this, it's only finalized during uncontrolled interpreter
        # shutdown (after Tk has already destroyed its windows), which can
        # hang for several seconds tearing down the context against a
        # display that's already gone — worse still through XWayland.
        # With lazy tabs the 3D page may never have been built, in which case
        # there is no context to close.
        for holder in self._holders:
            if isinstance(holder.inner, TurbVisuTab):
                plotter = getattr(holder.inner._visu_panel, '_plotter', None)
                if plotter is not None:
                    plotter.close()
        self.destroy()


if __name__ == '__main__':
    app = App()
    app.mainloop()
