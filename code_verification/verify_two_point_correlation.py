#!/usr/bin/env python3
"""
Verify the spanwise two-point correlation against analytic cases.

The kernels in operations.py are checked directly, and the whole
TwoPointCorrelationComputer path -- XDMF discovery, fluctuation construction,
x-station selection, slab chunking, centreline folding, half-channel reduction
-- is checked end to end on a synthetic case written to a temporary directory
in the same layout CHAPSim2 produces (case/2_visu/*.xdmf + case/1_data/*.bin).

The synthetic field is a single spanwise Fourier mode,

    u(z, y, x) = U(y, x) + A(y, x) cos(k z)
    v(z, y, x) = V(y, x) + B(y, x) cos(k z + phi)

whose correlations are known exactly:

    rho_uu(dz) = cos(k dz),  rho_uv(dz) = cos(k dz + phi),  L_z = 1/k

so every number below has an analytic target rather than a reference dataset.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

# Add parent dir so we can import operations / turb_stats
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import operations as op
from turb_stats import TwoPointCorrelationComputer

# ── synthetic case definition ──────────────────────────────────────────
NZ, NY, NX = 48, 17, 12
LZ = 2.0 * np.pi
MODE = 1
K = 2.0 * np.pi * MODE / LZ          # spanwise wavenumber of the planted mode
PHI = np.pi / 3.0                    # u-v phase lag -> rho_uv(0) = cos(PHI) = 0.5
CASE, TIMESTEP = 'synthetic', '000100'

results = []


def check(name, condition, extra=''):
    results.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f'   [{extra}]' if extra else ''))


# ── writing a synthetic CHAPSim2 visu case ─────────────────────────────
def _write_bin(path, array, header_bytes=0):
    with open(path, 'wb') as f:
        if header_bytes:
            f.write(b'\x00' * header_bytes)
        np.asarray(array, dtype=np.float64).tofile(f)


def _attribute_xml(name, bin_name, dims):
    return f"""      <Attribute Name="{name}" AttributeType="Scalar" Center="Cell">
        <DataItem ItemType="Uniform"
                 NumberType="Float"
                 Precision="8"
                 Format="Binary"
                 Dimensions="{dims[0]} {dims[1]} {dims[2]}">
          ../1_data/{bin_name}
        </DataItem>
      </Attribute>"""


def _write_xdmf(path, grid_name, nodes, attributes):
    nz1, ny1, nx1 = nodes
    geometry = ''.join(f"""        <DataItem ItemType="Uniform"
                 Dimensions="{n}"
                 NumberType="Float"
                 Precision="8"
                 Format="Binary"
                 Seek="4">
          ../1_data/domain1_grid_{axis}.bin
        </DataItem>
""" for axis, n in (('x', nx1), ('y', ny1), ('z', nz1)))

    path.write_text(f"""<?xml version="1.0" ?>
<Xdmf Version="3.0">
 <Domain>
   <Grid Name="{grid_name}" GridType="Uniform">
     <Topology TopologyType="3DRectMesh" Dimensions=" {nz1} {ny1} {nx1}"/>
     <Geometry GeometryType="VXVYVZ">
{geometry}      </Geometry>
{chr(10).join(attributes)}
   </Grid>
 </Domain>
</Xdmf>
""")


def build_case(root, with_t_avg=True):
    """Write the synthetic case and return its exact fields for comparison."""
    visu = root / CASE / '2_visu'
    data = root / CASE / '1_data'
    visu.mkdir(parents=True)
    data.mkdir(parents=True)

    x_nodes = np.linspace(0.0, 4.0, NX + 1)
    y_nodes = np.linspace(-1.0, 1.0, NY + 1)
    z_nodes = np.linspace(0.0, LZ, NZ + 1)
    for axis, nodes in (('x', x_nodes), ('y', y_nodes), ('z', z_nodes)):
        _write_bin(data / f'domain1_grid_{axis}.bin', nodes, header_bytes=4)

    y_cell = 0.5 * (y_nodes[:-1] + y_nodes[1:])
    x_cell = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    z_cell = 0.5 * (z_nodes[:-1] + z_nodes[1:])

    # Amplitudes vary with both y and x, so a wrongly-sliced station or a
    # mis-broadcast mean shows up as a wrong answer rather than passing by luck.
    amp_u = 1.0 + 0.5 * np.cos(np.pi * y_cell)[:, None] + 0.1 * x_cell[None, :]
    amp_v = 0.4 + 0.2 * y_cell[:, None] ** 2 + 0.05 * x_cell[None, :]
    mean_u = 10.0 * (1.0 - y_cell ** 2)[:, None] + x_cell[None, :]
    mean_v = -3.0 + 0.5 * y_cell[:, None] - 0.2 * x_cell[None, :]

    wave = np.cos(K * z_cell)[:, None, None]
    wave_shift = np.cos(K * z_cell + PHI)[:, None, None]
    u_inst = mean_u[None, :, :] + amp_u[None, :, :] * wave
    v_inst = mean_v[None, :, :] + amp_v[None, :, :] * wave_shift

    _write_bin(data / f'domain1_qx_ccc_{TIMESTEP}.bin', u_inst)
    _write_bin(data / f'domain1_qy_ccc_{TIMESTEP}.bin', v_inst)
    _write_xdmf(visu / f'domain1_flow_{TIMESTEP}.xdmf', 'flow', (NZ + 1, NY + 1, NX + 1),
                [_attribute_xml('qx_ccc', f'domain1_qx_ccc_{TIMESTEP}.bin', (NZ, NY, NX)),
                 _attribute_xml('qy_ccc', f'domain1_qy_ccc_{TIMESTEP}.bin', (NZ, NY, NX))])

    if with_t_avg:
        _write_bin(data / f'domain1_t_avg_u1_{TIMESTEP}.bin',
                   np.broadcast_to(mean_u[None, :, :], (NZ, NY, NX)))
        _write_bin(data / f'domain1_t_avg_u2_{TIMESTEP}.bin',
                   np.broadcast_to(mean_v[None, :, :], (NZ, NY, NX)))
        _write_xdmf(visu / f'domain1_t_avg_flow_{TIMESTEP}.xdmf', 't_avg_flow',
                    (NZ + 1, NY + 1, NX + 1),
                    [_attribute_xml('t_avg_u1', f'domain1_t_avg_u1_{TIMESTEP}.bin', (NZ, NY, NX)),
                     _attribute_xml('t_avg_u2', f'domain1_t_avg_u2_{TIMESTEP}.bin', (NZ, NY, NX))])

    return dict(y_cell=y_cell, x_cell=x_cell, z_cell=z_cell,
                amp_u=amp_u, amp_v=amp_v, u_inst=u_inst, v_inst=v_inst)


def run(root, **kwargs):
    """Run the computer over the synthetic case and return its results dict."""
    options = dict(components='uu', y_coords_str='', x_coords_str='',
                   timesteps=[TIMESTEP], mean_mode='t_avg', symmetry_avg=False)
    options.update(kwargs)
    computer = TwoPointCorrelationComputer(str(root), **options)
    if not computer.compute_for_case(CASE, TIMESTEP):
        return None
    return computer.processed_results[(CASE, TIMESTEP)]


def main():
    """Run every check; returns a process exit status."""
    # ── 1. kernels against analytic answers ───────────────────────────────
    print('\n--- kernels ---')
    z = np.arange(NZ) * LZ / NZ
    signal = np.broadcast_to(np.cos(K * z)[:, None], (NZ, 3)).copy()
    R = op.compute_two_point_correlation_z(signal, signal)
    rho = op.normalise_correlation(R)
    sep = z[:R.shape[0]]
    check('rho of a single mode is cos(k dz)',
          np.allclose(rho[:, 0], np.cos(K * sep), atol=1e-12),
          f'max err {np.abs(rho[:, 0] - np.cos(K * sep)).max():.2e}')
    L = op.compute_integral_length_scale(rho, sep)
    check('integral scale of that mode is 1/k', abs(L[0] - 1.0 / K) < 0.01 / K,
          f'{L[0]:.5f} vs {1.0 / K:.5f}')
    check('periodic FFT estimator == direct wrap-around sum',
          np.allclose(op.compute_two_point_correlation_z(signal, signal, max_sep=7),
                      np.stack([(signal * np.roll(signal, -d, axis=0)).mean(axis=0) for d in range(7)])))

    # ── 2. end to end on the synthetic case ───────────────────────────────
    root = Path(tempfile.mkdtemp(prefix='chapsim2_corr_'))
    try:
        exact = build_case(root)
        sep_exact = exact['z_cell'][:NZ // 2 + 1] - exact['z_cell'][0]

        print('\n--- end to end: correlation values ---')
        out = run(root, components='uu,uv', x_coords_str='0.5, 3.5')
        check('computer produced both components', out is not None and set(out) == {'uu', 'uv'})

        uu = out['uu']
        check('separation axis matches the z grid', np.allclose(uu['sep'], sep_exact))
        check('two x stations selected', uu['R'].shape == (NZ // 2 + 1, NY, 2), f"shape {uu['R'].shape}")
        check('rho_uu(0) == 1 everywhere', np.allclose(uu['rho'][0], 1.0))
        target = np.cos(K * sep_exact)[:, None, None]
        check('rho_uu(dz) == cos(k dz) at every y and x',
              np.allclose(uu['rho'], target, atol=1e-10),
              f"max err {np.abs(uu['rho'] - target).max():.2e}")
        # The cell nearest each requested station, so the check does not depend on
        # where the request happens to land in the grid.
        station_idx = [int(np.argmin(np.abs(exact['x_cell'] - v))) for v in (0.5, 3.5)]
        check('R_uu(0) == <u\' u\'> of the planted field',
              np.allclose(uu['R'][0], 0.5 * exact['amp_u'][:, station_idx] ** 2),
              f'stations at cells {station_idx}')

        uv = out['uv']
        target_uv = np.cos(K * sep_exact + PHI)[:, None, None]
        check('rho_uv(dz) == cos(k dz + phi)', np.allclose(uv['rho'], target_uv, atol=1e-10),
              f"max err {np.abs(uv['rho'] - target_uv).max():.2e}")
        check('rho_uv(0) == cos(phi) = 0.5', np.allclose(uv['rho'][0], np.cos(PHI)))
        check('cross-correlation stays bounded by 1', np.abs(uv['rho']).max() <= 1.0 + 1e-12)

        print('\n--- end to end: integral length scale ---')
        check('L_z == 1/k at every y and x', np.allclose(uu['L'], 1.0 / K, rtol=0.01),
              f"spread {uu['L'].min():.5f} to {uu['L'].max():.5f} vs {1.0 / K:.5f}")

        print('\n--- fluctuation construction ---')
        snap = run(root, components='uu', x_coords_str='0.5, 3.5', mean_mode='snapshot')
        check("snapshot mean gives the same rho as the t_avg mean (the planted mean is exact)",
              np.allclose(snap['uu']['rho'], uu['rho'], atol=1e-10))
        check('but the raw R differs from an unsubtracted mean',
              not np.allclose(uu['R'][0], (exact['u_inst'] ** 2).mean(axis=0)[:, station_idx]))

        print('\n--- x-station handling ---')
        mid = run(root, components='uu')
        check('no x stations given -> one mid-domain station', mid['uu']['R'].shape[2] == 1)
        check('mid station is nx//2', np.isclose(mid['uu']['x'][0], exact['x_cell'][NX // 2]))
        avg = run(root, components='uu', average_x=True)
        check('average_x -> a single x-averaged profile', avg['uu']['R'].shape[2] == 1)
        check('x-averaged R == mean of all stations',
              np.allclose(avg['uu']['R'][0, :, 0],
                          (0.5 * exact['amp_u'] ** 2).mean(axis=1), atol=1e-10))

        print('\n--- slab chunking ---')
        original_chunk = TwoPointCorrelationComputer._X_CHUNK
        try:
            TwoPointCorrelationComputer._X_CHUNK = 5     # forces 3 slabs over 12 columns
            chunked = run(root, components='uu,uv', average_x=True)
            TwoPointCorrelationComputer._X_CHUNK = 10_000
            single = run(root, components='uu,uv', average_x=True)
        finally:
            TwoPointCorrelationComputer._X_CHUNK = original_chunk
        check('chunked x-average == single-slab x-average (uu)',
              np.allclose(chunked['uu']['R'], single['uu']['R'], atol=1e-12))
        check('chunked x-average == single-slab x-average (uv)',
              np.allclose(chunked['uv']['R'], single['uv']['R'], atol=1e-12))

        print('\n--- centreline folding ---')
        folded = run(root, components='uu,uv', x_coords_str='0.5', symmetry_avg=True)
        # amp_u is even in y by construction, so folding must leave R_uu alone.
        check('folding leaves the even R_uu unchanged',
              np.allclose(folded['uu']['R'], uu['R'][:, :, [0]], atol=1e-12))
        # R_uv is even here too, so folding it with the odd (-1)^1 sign must cancel it.
        check('folding an even R_uv with the odd sign cancels it (parity is enforced)',
              np.allclose(folded['uv']['R'], 0.0, atol=1e-12))

        print('\n--- half-channel selection ---')
        half_len = NY - NY // 2
        lower = run(root, components='uu', x_coords_str='0.5', half_channel_side='lower')
        upper = run(root, components='uu', x_coords_str='0.5', half_channel_side='upper')
        both = run(root, components='uu', x_coords_str='0.5', half_channel_side='average')
        check('lower half keeps n - n//2 rows', lower['uu']['R'].shape[1] == half_len,
              f"{lower['uu']['R'].shape[1]} of {NY}")
        check('lower half is the y=-1 side', np.allclose(lower['uu']['R'][0, :, 0],
                                                        uu['R'][0, :half_len, 0]))
        check('upper half is the flipped y=+1 side',
              np.allclose(upper['uu']['R'][0, :, 0], np.flip(uu['R'][0, :, 0])[:half_len]))
        check('both halves start at the wall', np.isclose(lower['uu']['y'][0], upper['uu']['y'][0]),
              f"lower {lower['uu']['y'][0]:.4f}, upper {upper['uu']['y'][0]:.4f}")
        check("y is wall distance, increasing inward", lower['uu']['y'][0] < lower['uu']['y'][-1])
        check("'average' side averages the two halves",
              np.allclose(both['uu']['R'][0, :, 0],
                          op.symmetric_average(uu['R'][0, :, 0])))
        # u'v' is antisymmetric about the centreline, so 'average' must fall back
        # to the lower half rather than cancelling the component.
        uv_avg = run(root, components='uv', x_coords_str='0.5', half_channel_side='average')
        check("'average' falls back to 'lower' for the antisymmetric uv",
              np.allclose(uv_avg['uv']['R'][0, :, 0], uv['R'][0, :half_len, 0]))

        print('\n--- degraded inputs ---')
        no_t_avg_root = Path(tempfile.mkdtemp(prefix='chapsim2_corr_nomean_'))
        try:
            build_case(no_t_avg_root, with_t_avg=False)
            fallback = run(no_t_avg_root, components='uu', x_coords_str='0.5')
            check('missing t_avg file falls back to the snapshot mean',
                  fallback is not None and np.allclose(fallback['uu']['rho'][0], 1.0))
        finally:
            shutil.rmtree(no_t_avg_root, ignore_errors=True)

        missing = run(root / 'nonexistent', components='uu')
        check('a missing case is reported, not raised', missing is None)

        print('\n--- spectrum consistency (Wiener-Khinchin) ---')
        line = exact['u_inst'][:, 5, 3]
        line = line - line.mean()
        k_axis, E = op.compute_1d_spectrum(line, dx=float(LZ / NZ))
        R_line = op.compute_two_point_correlation_z(line, line)
        check('sum(E(k)) == R(0)', np.isclose(E.sum(), R_line[0]),
              f'{E.sum():.12f} vs {R_line[0]:.12f}')
        peak_k = k_axis[int(np.argmax(E))]
        check('spectral peak sits at the planted wavenumber', np.isclose(peak_k, K),
              f'{peak_k:.6f} vs {K:.6f}')

    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f'\n{sum(results)}/{len(results)} checks passed')
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
