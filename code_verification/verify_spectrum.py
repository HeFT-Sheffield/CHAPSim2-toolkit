#!/usr/bin/env python3
"""
Verify the 1D energy spectrum against analytic cases.

The whole SpectrumComputer path -- XDMF discovery, y and x-station selection,
line averaging, dropping the k = 0 and Nyquist modes, averaging over
snapshots -- is checked end to end on synthetic cases written in the layout
CHAPSim2 produces (case/2_visu/*.xdmf + case/1_data/*.bin), using the writers
from verify_two_point_correlation.

The planted fields are single Fourier modes, e.g.

    u(z, y, x) = U(y, x) + A(y, x) cos(kz z)

whose energy per mode is known exactly: A^2/2 at kz, nothing in any other
mode -- so every number below has an analytic target.
"""

import contextlib
import io
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

# Add parent dir so we can import operations / turb_stats
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chapsim2_toolkit import operations as op
from chapsim2_toolkit.turb_stats import (Config, SpectrumComputer, TurbulenceStatsPipeline, PlotConfig,
                        TurbulencePlotter, create_data_loader)
from verify_two_point_correlation import _attribute_xml, _write_bin, _write_xdmf

# ── synthetic case definition ──────────────────────────────────────────
NZ, NY, NX = 48, 17, 40
LZ, LX = 2.0 * np.pi, 4.0 * np.pi
MZ, MX = 3, 5                        # mode numbers of the planted waves
KZ = 2.0 * np.pi * MZ / LZ
KX = 2.0 * np.pi * MX / LX
PHI = np.pi / 3.0
CASE = 'synthetic'

results = []


def check(name, condition, extra=''):
    results.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f'   [{extra}]' if extra else ''))


# ── writing a synthetic CHAPSim2 visu case ─────────────────────────────
def write_grid(root):
    """Create the case folders and grid files; returns the cell-centre coordinates."""
    data = root / CASE / '1_data'
    data.mkdir(parents=True)
    (root / CASE / '2_visu').mkdir(parents=True)
    nodes = {'x': np.linspace(0.0, LX, NX + 1),
             'y': np.linspace(-1.0, 1.0, NY + 1),
             'z': np.linspace(0.0, LZ, NZ + 1)}
    for axis, values in nodes.items():
        _write_bin(data / f'domain1_grid_{axis}.bin', values, header_bytes=4)
    return {axis: 0.5 * (values[:-1] + values[1:]) for axis, values in nodes.items()}


def write_snapshot(root, timestep, u_inst):
    """Write one instantaneous qx_ccc field, laid out (nz, ny, nx), as a flow file."""
    bin_name = f'domain1_qx_ccc_{timestep}.bin'
    _write_bin(root / CASE / '1_data' / bin_name, u_inst)
    _write_xdmf(root / CASE / '2_visu' / f'domain1_flow_{timestep}.xdmf', 'flow',
                (NZ + 1, NY + 1, NX + 1), [_attribute_xml('qx_ccc', bin_name, u_inst.shape)])


def run(root, timestep, **kwargs):
    """Run the computer over the synthetic case and return its results dict."""
    options = dict(direction='z', y_coords_str='-0.5, 0.5')
    options.update(kwargs)
    computer = SpectrumComputer(str(root), **options)
    if not computer.compute_for_case(CASE, timestep):
        return None
    return computer.processed_results[(CASE, timestep)]


def run_quietly(root, timestep, **kwargs):
    """run(), also returning everything it printed."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(io.StringIO()):
        out = run(root, timestep, **kwargs)
    return out, printed.getvalue()


def main():
    """Run every check; returns a process exit status."""
    # ── 1. the kernel ─────────────────────────────────────────────────────
    print('\n--- kernel ---')
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((5, 64))
    signal -= signal.mean(axis=-1, keepdims=True)
    k_axis, E = op.compute_1d_spectrum(signal, dx=0.1)
    check('Parseval: sum(E) == mean(f^2)', np.allclose(E.sum(axis=-1), (signal ** 2).mean(axis=-1)))
    check('k is in rad/length', np.isclose(k_axis[1], 2.0 * np.pi / (64 * 0.1)))

    root = Path(tempfile.mkdtemp(prefix='chapsim2_spectrum_'))
    try:
        cells = write_grid(root)
        y, x, z = cells['y'], cells['x'], cells['z']

        # Snapshot 100: a spanwise wave whose amplitude varies with y and x,
        # on a mean that varies with both, so a wrongly-picked station or a
        # mean leaking into the spectrum shows up as a wrong answer.
        amp = 1.0 + 0.5 * np.cos(np.pi * y)[:, None] + 0.1 * x[None, :]
        mean = 10.0 * (1.0 - y ** 2)[:, None] + x[None, :]
        u_z = mean[None] + amp[None] * np.cos(KZ * z)[:, None, None]
        write_snapshot(root, '000100', u_z)

        # ── 2. z spectrum ─────────────────────────────────────────────────
        print('\n--- z spectrum at x stations ---')
        out, printed = run_quietly(root, '000100', x_coords_str='1.0, 9.0')
        check('computer produced a result', out is not None)
        y_idx = [int(np.argmin(np.abs(y - v))) for v in (-0.5, 0.5)]
        x_idx = [int(np.argmin(np.abs(x - v))) for v in (1.0, 9.0)]
        check('y locations are the nearest cells', np.allclose(out['y'], y[y_idx]))
        check('x stations are the nearest cells', np.allclose(out['x'], x[x_idx]))
        check('k = 0 and Nyquist dropped: modes 1 .. nz/2 - 1',
              len(out['k']) == NZ // 2 - 1 and np.isclose(out['k'][0], 2.0 * np.pi / LZ),
              f"{len(out['k'])} modes from k = {out['k'][0]:.4f}")
        check('E has shape (n_y, n_x, n_k)', out['E'].shape == (2, 2, NZ // 2 - 1), f"{out['E'].shape}")
        target = 0.5 * amp[np.ix_(y_idx, x_idx)] ** 2
        check('E(kz) == A^2/2 at every y and station', np.allclose(out['E'][:, :, MZ - 1], target),
              f"max err {np.abs(out['E'][:, :, MZ - 1] - target).max():.2e}")
        others = np.delete(out['E'], MZ - 1, axis=2)
        check('no energy in any other mode (the mean is not leaked)', np.abs(others).max() < 1e-20,
              f'max {np.abs(others).max():.1e}')

        print('\n--- x-station handling ---')
        mid, printed = run_quietly(root, '000100')
        check('no x stations -> one mid-domain station', np.allclose(mid['x'], [x[NX // 2]]))
        check('and that is reported', 'mid-domain station' in printed)
        avg, _ = run_quietly(root, '000100', average_x=True)
        check('average_x -> one x-averaged spectrum, no station', avg['E'].shape[1] == 1 and len(avg['x']) == 0)
        check('x-averaged E(kz) == mean over x of A^2/2',
              np.allclose(avg['E'][:, 0, MZ - 1], 0.5 * (amp[y_idx] ** 2).mean(axis=1)))
        both, _ = run_quietly(root, '000100', x_coords_str='1.0', average_x=True)
        check('given x stations take precedence over average_x', np.allclose(both['x'], [x[x_idx[0]]]))

        # ── 3. x spectrum ─────────────────────────────────────────────────
        # Snapshot 200: a streamwise wave plus a spanwise one. Along x the
        # spanwise wave is constant, so only B shows up in an x spectrum.
        amp_x = 0.3 + 0.2 * y ** 2
        amp_z = 0.7 - 0.1 * y
        u_x = ((10.0 * (1.0 - y ** 2))[None, :, None]
               + amp_x[None, :, None] * np.cos(KX * x + PHI)[None, None, :]
               + amp_z[None, :, None] * np.cos(KZ * z)[:, None, None])
        write_snapshot(root, '000200', u_x)

        print('\n--- x spectrum ---')
        out_x, printed = run_quietly(root, '000200', direction='x', average_x=True)
        check('modes 1 .. nx/2 - 1 along x',
              len(out_x['k']) == NX // 2 - 1 and np.isclose(out_x['k'][0], 2.0 * np.pi / LX))
        check('averaged over z: one spectrum per y, no station',
              out_x['E'].shape == (2, 1, NX // 2 - 1) and len(out_x['x']) == 0)
        check('E(kx) == B^2/2', np.allclose(out_x['E'][:, 0, MX - 1], 0.5 * amp_x[y_idx] ** 2))
        others = np.delete(out_x['E'], MX - 1, axis=2)
        check('no energy in any other kx', np.abs(others).max() < 1e-20, f'max {np.abs(others).max():.1e}')
        check('no periodicity warning when x is declared homogeneous', 'WARNING' not in printed)
        _, printed = run_quietly(root, '000200', direction='x')
        check("periodicity warning with 'Average x direction' off", 'WARNING' in printed)
        out_z, _ = run_quietly(root, '000200', x_coords_str='3.0')
        check('the same snapshot along z holds only the spanwise wave',
              np.allclose(out_z['E'][:, 0, MZ - 1], 0.5 * amp_z[y_idx] ** 2))

        # ── 4. Nyquist mode of a cell-centred field ───────────────────────
        print('\n--- Nyquist mode ---')
        # qx_ccc is the average of the two x-face values, which cancels the
        # x-direction Nyquist mode exactly.
        q = rng.standard_normal((NZ, NY, NX))
        u_ccc = 0.5 * (q + np.roll(q, -1, axis=2))
        write_snapshot(root, '000500', u_ccc)
        line = u_ccc[:, 4, :] - u_ccc[:, 4, :].mean(axis=-1, keepdims=True)
        full = op.compute_1d_spectrum(line, dx=LX / NX)[1].mean(axis=0)
        check('cell-centring zeroes the full spectrum\'s Nyquist mode', full[-1] < 1e-25 * full.max(),
              f'{full[-1]:.1e} vs max {full.max():.1e}')
        out_n, _ = run_quietly(root, '000500', direction='x', average_x=True)
        check('the computer leaves it out', np.isclose(out_n['k'][-1], (NX // 2 - 1) * 2.0 * np.pi / LX))
        check('so the spectrum stays within a few decades on a log axis',
              out_n['E'].min() > 1e-6 * out_n['E'].max(),
              f"min/max {out_n['E'].min() / out_n['E'].max():.1e}")

        # ── 5. averaging over snapshots ───────────────────────────────────
        print('\n--- averaging over snapshots ---')
        amp_1, amp_2 = 1.0 + 0.2 * y, 2.0 - 0.3 * y
        write_snapshot(root, '000300', 5.0 + amp_1[None, :, None] * np.cos(KZ * z)[:, None, None]
                       * np.ones((1, 1, NX)))
        write_snapshot(root, '000400', 5.0 + amp_2[None, :, None] * np.cos(KZ * z)[:, None, None]
                       * np.ones((1, 1, NX)))
        target_avg = 0.25 * (amp_1[y_idx] ** 2 + amp_2[y_idx] ** 2)
        out_avg, printed = run_quietly(root, 'avg', timesteps=['000300', '000400'], x_coords_str='2.0')
        check("timestep 'avg' averages the configured snapshots",
              out_avg is not None and np.allclose(out_avg['E'][:, 0, MZ - 1], target_avg))
        check('and says how many it used', 'averaged 2 snapshot(s)' in printed)
        out_gap, printed = run_quietly(root, 'avg', timesteps=['000300', '999999', '000400'],
                                       x_coords_str='2.0')
        check('a missing snapshot is reported and left out',
              'Missing instantaneous flow file' in printed and out_gap is not None
              and np.allclose(out_gap['E'][:, 0, MZ - 1], target_avg))
        none, printed = run_quietly(root, 'avg', timesteps=[])
        check("'avg' with no snapshots fails cleanly", none is None and 'no usable snapshots' in printed)

        # ── 6. degraded inputs ────────────────────────────────────────────
        print('\n--- degraded inputs ---')
        slice_2d = np.ones((1, NY, NX))
        write_snapshot(root, '000600', slice_2d)
        flat, printed = run_quietly(root, '000600')
        check('a 2D slice is reported, not raised', flat is None and 'full 3D' in printed)
        no_y, printed = run_quietly(root, '000100', y_coords_str='')
        check('no y locations -> skipped', no_y is None and 'no y-coordinates' in printed)
        missing, _ = run_quietly(root / 'nonexistent', '000100')
        check('a missing case is reported, not raised', missing is None)

        # ── 7. through the pipeline, as the GUI runs it ───────────────────
        print('\n--- pipeline with average_over_timesteps ---')
        module = types.SimpleNamespace(
            folder_path=str(root), input_format='visu', cases=[CASE],
            timesteps=['000300', '000400'], average_over_timesteps=True,
            ux_velocity_on=False, save_fig=False, display_fig=False,
            spectrum_on=True, spectrum_direction='z',
            spectrum_y_coords='-0.5, 0.5', spectrum_x_coords='2.0, 8.0',
        )
        config = Config.from_module(module)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            loader = create_data_loader(config)
            loader.load_all()
            pipeline = TurbulenceStatsPipeline(config, loader)
            pipeline.compute_all()
            pipeline.process_all()
            fig = TurbulencePlotter(config, PlotConfig(), loader).plot_spectrum(pipeline.spectrum_computer)
        spectra = pipeline.spectrum_computer.processed_results
        check("the loader's 'avg' timestep yields a spectrum", (CASE, 'avg') in spectra)
        check('of the averaged snapshots',
              (CASE, 'avg') in spectra and np.allclose(spectra[(CASE, 'avg')]['E'][:, 0, MZ - 1], target_avg))
        lines = fig.axes[0].get_lines() if fig is not None else []
        curves = [line for line in lines if line.get_label() != '$k^{-5/3}$']
        reference = [line for line in lines if line.get_label() == '$k^{-5/3}$']
        check('one plotted curve per y location and x station', len(curves) == 4, f'{len(curves)} curves')
        check('plus one k^-5/3 reference line, on by default', len(reference) == 1)

        print('\n--- k^-5/3 reference line ---')
        k_ref, E_ref = reference[0].get_xdata(), reference[0].get_ydata()
        slope = np.polyfit(np.log(k_ref), np.log(E_ref), 1)[0]
        check('its slope is -5/3', np.isclose(slope, -5.0 / 3.0), f'{slope:.6f}')
        k_all = np.concatenate([line.get_xdata() for line in curves])
        check('it spans the middle half of the log k range',
              np.allclose(np.log(k_ref[[0, -1]]), np.log(k_all.min()) + np.array([0.25, 0.75])
                          * np.log(k_all.max() / k_all.min())))
        clear = True
        for line in curves:
            k, E = line.get_xdata(), line.get_ydata()
            inside = (k >= k_ref[0]) & (k <= k_ref[-1])
            clear &= bool(np.all(E[inside] <= E_ref[0] * (k[inside] / k_ref[0]) ** (-5.0 / 3.0)))
        check('and lies above every curve there', clear)

        config_off = Config.from_module(types.SimpleNamespace(**{**vars(module), 'spectrum_kolmogorov_ref_on': False}))
        fig_off = TurbulencePlotter(config_off, PlotConfig(), loader).plot_spectrum(pipeline.spectrum_computer)
        labels_off = [line.get_label() for line in fig_off.axes[0].get_lines()]
        check('spectrum_kolmogorov_ref_on = False leaves only the curves',
              '$k^{-5/3}$' not in labels_off and len(labels_off) == 4, f'{len(labels_off)} lines')

    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f'\n{sum(results)}/{len(results)} checks passed')
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
