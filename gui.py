#!/usr/bin/env python3
"""CHAPSim2 Toolkit GUI"""

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
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

_BG       = '#222222'
_INPUT_BG = '#303030'
_FG       = '#ffffff'
_SEL_BG   = '#375a7f'
_SEL_FG   = '#ffffff'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =====================================================================================
# Shared utilities
# =====================================================================================

class ScrollableFrame(ttk.Frame):
    """Vertically scrollable frame with mousewheel support."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, background=_BG)
        sb = ttk.Scrollbar(self, orient='vertical', command=self._canvas.yview)
        self.inner = ttk.Frame(self._canvas)
        self.inner.bind('<Configure>',
                        lambda e: self._canvas.configure(scrollregion=self._canvas.bbox('all')))
        self._canvas.create_window((0, 0), window=self.inner, anchor='nw')
        self._canvas.configure(yscrollcommand=sb.set)
        self._canvas.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        self.inner.bind('<Enter>', lambda e: self._bind_wheel())
        self.inner.bind('<Leave>', lambda e: self._unbind_wheel())

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
            self._canvas.yview_scroll(-1, 'units')
        elif event.num == 5:
            self._canvas.yview_scroll(1, 'units')
        else:
            self._canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')


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
    """Redirect stdout/stderr to a ScrolledText widget, thread-safely."""

    def __init__(self, widget):
        self._w = widget

    def write(self, msg):
        # Schedule all Tk operations on the main thread — never call Tk from a worker thread.
        try:
            self._w.after(0, self._append, msg)
        except Exception:
            pass

    def _append(self, msg):
        try:
            self._w.configure(state='normal')
            self._w.insert(tk.END, msg)
            self._w.see(tk.END)
            self._w.configure(state='disabled')
        except tk.TclError:
            pass

    def flush(self):
        pass


def _make_console(parent, height=7):
    w = scrolledtext.ScrolledText(
        parent, height=height, state='disabled',
        font=('Monospace', 8), wrap='word',
        background=_INPUT_BG, foreground=_FG,
        insertbackground=_FG, selectbackground=_SEL_BG,
        selectforeground=_SEL_FG, relief='flat', borderwidth=0,
    )
    return w


def _log_to(widget, msg):
    widget.configure(state='normal')
    widget.insert(tk.END, msg + '\n')
    widget.see(tk.END)
    widget.configure(state='disabled')


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

class TurbStatsTab(ttk.Frame):

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
            lf = ttk.LabelFrame(f, text=title)
            lf.pack(fill='x', padx=4, pady=3)
            return lf

        def erow(frame, label, var):
            r = ttk.Frame(frame)
            r.pack(fill='x', pady=1)
            ttk.Label(r, text=label, width=24, anchor='w').pack(side='left')
            ttk.Entry(r, textvariable=var).pack(side='left', fill='x', expand=True)

        def brow(frame, label, var):
            r = ttk.Frame(frame)
            r.pack(fill='x', pady=1)
            ttk.Label(r, text=label, width=24, anchor='w').pack(side='left')
            ttk.Entry(r, textvariable=var).pack(side='left', fill='x', expand=True)
            ttk.Button(r, text='…', width=3,
                       command=lambda v=var: v.set(filedialog.askdirectory() or v.get())
                       ).pack(side='left')

        def crow(frame, label, var, values):
            r = ttk.Frame(frame)
            r.pack(fill='x', pady=1)
            ttk.Label(r, text=label, width=24, anchor='w').pack(side='left')
            ttk.Combobox(r, textvariable=var, values=values,
                         state='readonly', width=14).pack(side='left')

        def chk(frame, label, var):
            ttk.Checkbutton(frame, text=label, variable=var).pack(anchor='w', pady=1)

        def trow(frame, label, height=2):
            """Text widget row; returns the Text widget."""
            r = ttk.Frame(frame)
            r.pack(fill='x', pady=1)
            ttk.Label(r, text=label, width=24, anchor='nw').pack(side='left', anchor='n')
            inner = ttk.Frame(r)
            inner.pack(side='left', fill='x', expand=True)
            t = tk.Text(inner, height=height, width=26, font=('TkDefaultFont', 9),
                        background=_INPUT_BG, foreground=_FG, insertbackground=_FG,
                        selectbackground=_SEL_BG, selectforeground=_SEL_FG,
                        relief='flat', borderwidth=0)
            sb = ttk.Scrollbar(inner, orient='vertical', command=t.yview)
            t.configure(yscrollcommand=sb.set)
            t.pack(side='left', fill='x', expand=True)
            sb.pack(side='left', fill='y')
            return t

        # ---- Input Data ----
        s = sec('Input Data')
        brow(s, 'Folder path', sv('folder_path', ''))
        crow(s, 'Input format', sv('input_format', 'visu'), ['visu', 'text'])
        crow(s, 'Data type', sv('xdmf_data_type', 'tsp_avg'),
             ['tsp_avg', 't_avg', 'inst'])
        self._t_cases = trow(s, 'Cases (one per line)', height=3)
        self._t_cases.insert('1.0', 'Tests')
        self._t_timesteps = trow(s, 'Timesteps (one per line)', height=3)
        self._t_timesteps.insert('1.0', '680000')
        self._t_re = trow(s, 'Re (one per case)', height=2)
        self._t_re.insert('1.0', '5000')
        crow(s, 'Forcing', sv('forcing', 'CMF'), ['CMF', 'CPG'])
        erow(s, 'Slice label', sv('slice_label', ''))

        # ---- Thermal / MHD ----
        s = sec('Thermal / MHD')
        chk(s, 'Thermal statistics on', bv('thermo_on', True))
        chk(s, 'MHD statistics on', bv('mhd_on', True))
        self._t_ref_temp = trow(s, 'Ref. temperature (K)', height=2)
        self._t_ref_temp.insert('1.0', '570')
        self._t_ref_len = trow(s, 'Ref. length (m)', height=2)
        self._t_ref_len.insert('1.0', '0.05')
        self._t_ref_ubulk = trow(s, 'Ref. U_bulk (m/s)', height=2)
        self._t_ref_ubulk.insert('1.0', '0.0900625')
        self._t_wall_hf = trow(s, 'Wall heat flux (W/m²)', height=2)
        self._t_wall_hf.insert('1.0', '0.0')
        crow(s, 'Working fluid', sv('working_fluid', 'lithium'),
             ['lithium', 'sodium', 'lead', 'bismuth', 'lbe', 'flibe', 'pbli'])
        self._t_gravity_dir = trow(s, 'Gravity dir. (x,y,z)', height=2)
        self._t_gravity_dir.insert('1.0', '0, -1, 0')
        self._t_mag_field_dir = trow(s, 'Mag. field dir. (x,y,z)', height=2)
        self._t_mag_field_dir.insert('1.0', '0, 1, 0')
        self._t_stuart_number = trow(s, 'Stuart number (N)', height=2)
        self._t_stuart_number.insert('1.0', '0.0')

        # ---- Averaging ----
        s = sec('Averaging')
        chk(s, 'Average x direction', bv('average_x_direction', False))
        chk(s, 'Average z direction', bv('average_z_direction', True))

        # ---- Statistics to Compute ----
        s = sec('Statistics to Compute')
        chk(s, 'u_x velocity', bv('ux_velocity_on', True))
        chk(s, 'u_y velocity', bv('uy_velocity_on', False))
        chk(s, 'u_z velocity', bv('uz_velocity_on', False))
        chk(s, 'Temperature', bv('temp_on', False))
        chk(s, 'TKE', bv('tke_on', False))
        chk(s, 'Friction coefficient', bv('coeff_friction_on', False))
        chk(s, "u'u' Reynolds stress", bv('u_prime_sq_on', False))
        chk(s, "u'v' Reynolds stress", bv('u_prime_v_prime_on', False))
        chk(s, "v'v' Reynolds stress", bv('v_prime_sq_on', False))
        chk(s, "v'w' Reynolds stress", bv('v_prime_w_prime_on', False))
        chk(s, "w'w' Reynolds stress", bv('w_prime_sq_on', False))
        chk(s, 'Reynolds Stress Budget terms', bv('re_stress_budget_on', False))
        crow(s, 'Budget component', sv('re_stress_component', 'uu11'),
             ['total', 'uu11', 'uu12', 'uu22', 'uu33'])
        chk(s, 'Heat transfer coeff.', bv('heat_transf_coeff_on', False))
        chk(s, 'Nusselt number', bv('Nusselt_number_on', False))
        chk(s, 'Turbulent Prandtl number', bv('turb_prandtl_on', False))
        chk(s, '2D surface plots', bv('surface_plot_on', False))

        # ---- Profile Options ----
        s = sec('Profile Options')
        crow(s, 'Profile direction', sv('profile_direction', 'y'), ['y', 'x', 'both'])
        erow(s, 'Slice coords (x)', sv('slice_coords', ''))
        erow(s, 'x crop', sv('x_crop', ''))
        erow(s, 'x-prof. y coords', sv('x_profile_y_coords', ''))

        # ---- Normalisation ----
        s = sec('Normalisation')
        chk(s, 'Normalise by u_τ²', bv('norm_by_u_tau_sq', True))
        chk(s, 'Normalise U_x by u_τ', bv('norm_ux_by_u_tau', True))
        chk(s, 'Normalise y to y⁺', bv('norm_y_to_y_plus', False))
        chk(s, 'Normalise T by T_ref', bv('norm_temp_by_ref_temp', False))

        # ---- Plotting ----
        s = sec('Plotting')
        chk(s, 'Half channel', bv('half_channel_plot', False))
        chk(s, 'Linear y scale', bv('linear_y_scale', True))
        chk(s, 'Log y scale', bv('log_y_scale', False))
        chk(s, 'Multi-plot', bv('multi_plot', True))
        chk(s, 'Save figures', bv('save_fig', True))
        chk(s, 'Save to folder path', bv('save_to_path', True))
        chk(s, 'Large text', bv('large_text_on', False))
        erow(s, 'Plot name', sv('plot_name', ''))

        # ---- Reference Data ----
        s = sec('Reference Data')
        chk(s, 'Log-law reference', bv('ux_velocity_log_ref_on', True))
        chk(s, 'MHD NK reference', bv('mhd_NK_ref_on', False))
        chk(s, 'MKM180 reference', bv('mkm180_ch_ref_on', False))

        # ---- Console ----
        ttk.Label(parent, text='Console output:').pack(anchor='w', padx=5)
        self._console = _make_console(parent, height=7)
        self._console.pack(fill='x', padx=5, pady=2)

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
            profile_direction=v['profile_direction'].get(),
            slice_coords=v['slice_coords'].get(),
            x_crop=v['x_crop'].get(),
            x_profile_y_coords=v['x_profile_y_coords'].get(),
            surface_plot_on=v['surface_plot_on'].get(),
            u_prime_sq_on=v['u_prime_sq_on'].get(),
            u_prime_v_prime_on=v['u_prime_v_prime_on'].get(),
            v_prime_sq_on=v['v_prime_sq_on'].get(),
            v_prime_w_prime_on=v['v_prime_w_prime_on'].get(),
            w_prime_sq_on=v['w_prime_sq_on'].get(),
            re_stress_budget_on=v['re_stress_budget_on'].get(),
            re_stress_component=v['re_stress_component'].get(),
            average_z_direction=v['average_z_direction'].get(),
            average_x_direction=v['average_x_direction'].get(),
            norm_by_u_tau_sq=v['norm_by_u_tau_sq'].get(),
            norm_ux_by_u_tau=v['norm_ux_by_u_tau'].get(),
            norm_y_to_y_plus=v['norm_y_to_y_plus'].get(),
            norm_temp_by_ref_temp=v['norm_temp_by_ref_temp'].get(),
            slice_label=v['slice_label'].get(),
            half_channel_plot=v['half_channel_plot'].get(),
            linear_y_scale=v['linear_y_scale'].get(),
            log_y_scale=v['log_y_scale'].get(),
            multi_plot=v['multi_plot'].get(),
            xdmf_data_type=v['xdmf_data_type'].get(),
            display_fig=False,          # always embedded; never plt.show()
            save_fig=v['save_fig'].get(),
            save_to_path=v['save_to_path'].get(),
            large_text_on=v['large_text_on'].get(),
            plot_name=v['plot_name'].get(),
            ux_velocity_log_ref_on=v['ux_velocity_log_ref_on'].get(),
            mhd_NK_ref_on=v['mhd_NK_ref_on'].get(),
            mkm180_ch_ref_on=v['mkm180_ch_ref_on'].get(),
        )

    # ------ Run pipeline -------------------------------------------------------------

    def _run(self):
        self._console.configure(state='normal')
        self._console.delete('1.0', tk.END)
        self._console.configure(state='disabled')

        try:
            config = self._build_config_obj()
        except Exception as exc:
            messagebox.showerror('Config error', str(exc))
            return

        def worker():
            old_out, old_err = sys.stdout, sys.stderr
            redir = TextRedirect(self._console)
            sys.stdout = redir
            sys.stderr = redir
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

                if config.save_fig and figs:
                    plotter.save_figures_by_class(figs)

                self.after(0, lambda: self._update_figures(figs))
                print('Done.')
            except Exception:
                traceback.print_exc()
            finally:
                sys.stdout = old_out
                sys.stderr = old_err

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
                'folder_path': '', 'input_format': 'visu', 'forcing': 'CMF',
                'slice_label': '', 'working_fluid': 'lithium', 'profile_direction': 'y',
                'slice_coords': '', 'x_crop': '', 'x_profile_y_coords': '',
                're_stress_component': 'uu11', 'plot_name': '',
            }
            bool_fields = {
                'thermo_on': False, 'mhd_on': False,
                'average_x_direction': False, 'average_z_direction': True,
                'ux_velocity_on': True, 'uy_velocity_on': False, 'uz_velocity_on': False,
                'temp_on': False, 'tke_on': False, 'coeff_friction_on': False,
                'u_prime_sq_on': False, 'u_prime_v_prime_on': False,
                'v_prime_sq_on': False, 'v_prime_w_prime_on': False, 'w_prime_sq_on': False,
                're_stress_budget_on': False, 'heat_transf_coeff_on': False,
                'Nusselt_number_on': False, 'turb_prandtl_on': False, 'surface_plot_on': False,
                'norm_by_u_tau_sq': True, 'norm_ux_by_u_tau': True,
                'norm_y_to_y_plus': False, 'norm_temp_by_ref_temp': False,
                'half_channel_plot': False, 'linear_y_scale': True, 'log_y_scale': False,
                'multi_plot': True, 'save_fig': True, 'save_to_path': True,
                'large_text_on': False, 'ux_velocity_log_ref_on': True,
                'mhd_NK_ref_on': False, 'mkm180_ch_ref_on': False,
            }
            for name, default in str_fields.items():
                if name in v:
                    v[name].set(getattr(mod, name, default))
            for name, default in bool_fields.items():
                if name in v:
                    v[name].set(getattr(mod, name, default))

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
            f"slice_label = '{v['slice_label'].get()}'",
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
            '',
            f"ux_velocity_on = {v['ux_velocity_on'].get()}",
            f"uy_velocity_on = {v['uy_velocity_on'].get()}",
            f"uz_velocity_on = {v['uz_velocity_on'].get()}",
            f"temp_on = {v['temp_on'].get()}",
            f"tke_on = {v['tke_on'].get()}",
            f"coeff_friction_on = {v['coeff_friction_on'].get()}",
            '',
            f"profile_direction = '{v['profile_direction'].get()}'",
            f"slice_coords = '{v['slice_coords'].get()}'",
            f"x_crop = '{v['x_crop'].get()}'",
            f"x_profile_y_coords = '{v['x_profile_y_coords'].get()}'",
            f"surface_plot_on = {v['surface_plot_on'].get()}",
            '',
            f"u_prime_sq_on = {v['u_prime_sq_on'].get()}",
            f"u_prime_v_prime_on = {v['u_prime_v_prime_on'].get()}",
            f"v_prime_sq_on = {v['v_prime_sq_on'].get()}",
            f"v_prime_w_prime_on = {v['v_prime_w_prime_on'].get()}",
            f"w_prime_sq_on = {v['w_prime_sq_on'].get()}",
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
            f"half_channel_plot = {v['half_channel_plot'].get()}",
            f"linear_y_scale = {v['linear_y_scale'].get()}",
            f"log_y_scale = {v['log_y_scale'].get()}",
            f"multi_plot = {v['multi_plot'].get()}",
            'display_fig = False',
            f"save_fig = {v['save_fig'].get()}",
            f"save_to_path = {v['save_to_path'].get()}",
            f"large_text_on = {v['large_text_on'].get()}",
            f"plot_name = '{v['plot_name'].get()}'",
            '',
            f"ux_velocity_log_ref_on = {v['ux_velocity_log_ref_on'].get()}",
            f"mhd_NK_ref_on = {v['mhd_NK_ref_on'].get()}",
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


class SliceTab(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent)
        self._var_meta = {}
        self._grid_info = {}
        self._current_fig = None
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
        path_frame = ttk.LabelFrame(parent, text='Data Path')
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
            lf = ttk.LabelFrame(f, text=title)
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
        self._var_lb = tk.Listbox(s, selectmode='extended', height=8, exportselection=False,
                                   background=_INPUT_BG, foreground=_FG,
                                   selectbackground=_SEL_BG, selectforeground=_SEL_FG,
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

        # Console
        ttk.Label(f, text='Console:').pack(anchor='w', padx=4, pady=(6, 0))
        self._console = _make_console(f, height=6)
        self._console.pack(fill='x', padx=4, pady=2)

    # ------ Helpers ------------------------------------------------------------------

    def _log(self, msg):
        self.after(0, lambda: _log_to(self._console, msg))

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

    def _scan(self):
        visu = self._visu_folder()
        try:
            from slice import get_available_timesteps
            tss = get_available_timesteps(visu)
            self._ts_combo['values'] = tss
            if tss:
                self._ts.set(tss[0])
            _log_to(self._console, f'Found {len(tss)} timestep(s): {", ".join(tss)}')
        except Exception as exc:
            _log_to(self._console, f'Scan error: {exc}')

    def _load_vars(self):
        xdmf = self._xdmf_path()
        _log_to(self._console, f'Reading metadata: {xdmf}')
        try:
            from utils import parse_xdmf_metadata
            self._var_meta, self._grid_info = parse_xdmf_metadata(xdmf)
            names = sorted(self._var_meta.keys())
            self._var_lb.delete(0, tk.END)
            for n in names:
                self._var_lb.insert(tk.END, n)
            _log_to(self._console, f'Loaded {len(names)} variable(s).')
            # Update index spin max
            gy = self._grid_info.get('grid_y')
            if gy is not None:
                self._idx_spin.configure(to=len(gy) - 1)
        except Exception as exc:
            _log_to(self._console, f'Error: {exc}\n{traceback.format_exc()}')

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
                from utils import parse_xdmf_metadata, load_xdmf_variables
                from slice import (extract_slice, plot_slice, plot_combined_slices,
                                   process_data_arrays, get_slice_location)

                xdmf = self._xdmf_path()
                self._log('Loading data…')
                var_meta, grid = parse_xdmf_metadata(xdmf)
                data = load_xdmf_variables(var_meta, sel, grid)

                interp = self._interp.get()
                processed, interp_vars = process_data_arrays(data, sel, grid, interp)

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
                    coord1 = grid.get('grid_x', np.arange(sample.shape[-1] if sample.ndim > 1 else 1))
                    coord2 = grid.get('grid_y', np.arange(sample.shape[0]))
                    axis_labels = ('x', 'y')
                    slice_info = f'2D data, t={ts}'
                    slices = [(vn, processed[vn]) for vn in sel if vn in processed]
                else:
                    slices = []
                    coord1 = coord2 = axis_labels = None
                    for vn in sel:
                        if vn not in processed:
                            continue
                        sd, c1, c2, al = extract_slice(processed[vn], plane, idx, grid)
                        slices.append((vn, sd))
                        if coord1 is None:
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
            _log_to(self._console, f'Saved to {path}')


# =====================================================================================
# MONITOR POINTS TAB
# =====================================================================================

class MonitorPointsTab(ttk.Frame):

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
        s = ttk.LabelFrame(f, text='Data Path')
        s.pack(fill='x', padx=4, pady=3)
        self._path = tk.StringVar()
        r = ttk.Frame(s)
        r.pack(fill='x')
        ttk.Entry(r, textvariable=self._path).pack(side='left', fill='x', expand=True)
        ttk.Button(r, text='…', width=3,
                   command=lambda: self._path.set(filedialog.askdirectory() or self._path.get())
                   ).pack(side='left')

        # Options
        s = ttk.LabelFrame(f, text='Options')
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
        s2 = ttk.LabelFrame(f, text='Figures')
        s2.pack(fill='both', expand=True, padx=4, pady=3)
        self._fig_lb = tk.Listbox(s2, height=10, exportselection=False,
                                   background=_INPUT_BG, foreground=_FG,
                                   selectbackground=_SEL_BG, selectforeground=_SEL_FG,
                                   activestyle='none', relief='flat', borderwidth=0)
        sb = ttk.Scrollbar(s2, orient='vertical', command=self._fig_lb.yview)
        self._fig_lb.configure(yscrollcommand=sb.set)
        self._fig_lb.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        self._fig_lb.bind('<<ListboxSelect>>', self._on_select)

        # Console
        ttk.Label(f, text='Console:').pack(anchor='w', padx=4, pady=(4, 0))
        self._console = _make_console(f, height=6)
        self._console.pack(fill='x', padx=4, pady=2)

    # ------ Helpers ------------------------------------------------------------------

    def _log(self, msg):
        self.after(0, lambda: _log_to(self._console, msg))

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

        self._console.configure(state='normal')
        self._console.delete('1.0', tk.END)
        self._console.configure(state='disabled')

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
# APPLICATION
# =====================================================================================

class App(ttk.Window):

    def __init__(self):
        super().__init__(themename='darkly')
        self.title('CHAPSim2 Toolkit')
        self.geometry('1400x820')
        self.minsize(920, 600)
        self._build()

    def _build(self):
        nb = ttk.Notebook(self)
        nb.pack(fill='both', expand=True)
        nb.add(TurbStatsTab(nb), text='  Turbulence Statistics  ')
        nb.add(SliceTab(nb),     text='  Slice Visualisation  ')
        nb.add(MonitorPointsTab(nb), text='  Monitoring Points  ')


if __name__ == '__main__':
    app = App()
    app.mainloop()
