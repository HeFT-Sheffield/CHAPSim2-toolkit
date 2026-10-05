"""Shared helpers for the toolkit's tests.

Two things live here:

* ``skip()``, which defers to pytest when it is installed and otherwise
  raises a sentinel the bundled runner understands, so the suite works
  either way;
* builders that write synthetic CHAPSim2 cases on disk.

The cases are generated rather than committed. The formats are small and
exactly specified, the field values are analytic so a test can assert the
number it should get, and nothing binary ends up in the repository.
"""

import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

__all__ = ['skip', 'Skip', 'SKIP_EXCEPTIONS', 'solver_tests_dir',
           'build_cartesian_case', 'build_cylindrical_case',
           'write_monitor_files']


class Skip(Exception):
    """Raised by skip() when pytest is not installed."""


try:                                             # pragma: no cover - env dependent
    import pytest as _pytest
except ImportError:                              # pragma: no cover - env dependent
    _pytest = None


def skip(reason):
    """Skip the current test, under pytest or the bundled runner."""
    if _pytest is not None:
        _pytest.skip(reason)
    raise Skip(reason)


#: What a skip looks like to a caller. pytest's own exception is included
#: because the bundled runner may still be used on a machine that has
#: pytest installed - run_tests.py --no-pytest - and skip() prefers it.
SKIP_EXCEPTIONS = (Skip,) if _pytest is None else (Skip, _pytest.skip.Exception)


#: Where CHAPSim2's own test cases live, for the tests that read real output.
#: Override with CHAPSIM2_TESTS when the solver is checked out elsewhere.
_DEFAULT_SOLVER_TESTS = os.path.expanduser(
    '~/Work_RSDevelopment/1_CHAPSim/CHAPSim2/tests')


def solver_tests_dir():
    """CHAPSim2's tests directory, or None when it is not available."""
    path = os.environ.get('CHAPSIM2_TESTS', _DEFAULT_SOLVER_TESTS)
    return path if os.path.isdir(path) else None


# =====================================================================================
# Writing CHAPSim2 output formats
# =====================================================================================

def _case_dirs(root, name):
    case = os.path.join(root, name)
    dirs = {'case': case,
            'xdmf': os.path.join(case, '2_visu', 'xdmf'),
            'data': os.path.join(case, '2_visu', 'data'),
            'mesh': os.path.join(case, '2_visu', 'mesh'),
            'raw': os.path.join(case, '1_data'),
            'monitor': os.path.join(case, '3_monitor')}
    for key, path in dirs.items():
        if key != 'case':
            os.makedirs(path, exist_ok=True)
    return dirs


def _write_grid_1d(path, values):
    """Cartesian coordinate file: int32 count, then float64 values."""
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<i', len(values)))
        np.asarray(values, dtype='<f8').tofile(fh)


def _write_grid_xyz(path, x, y, z, nnode):
    """Curvilinear mesh: three int32 dims, then (x,y,z) per node, k-major."""
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<3i', *nnode))
        ni, nj, nk = nnode
        buf = np.empty((nk, nj, ni, 3), dtype='<f8')
        buf[..., 0], buf[..., 1], buf[..., 2] = x, y, z
        buf.tofile(fh)


def _attribute(name, dims_kji, bin_rel, seek, precision=4):
    return f"""      <Attribute Name="{name}" AttributeType="Scalar" Center="Cell">
        <DataItem ItemType="Uniform"
                  NumberType="Float"
                  Precision="{precision}"
                  Format="Binary"
                  Seek="{seek}"
                  Dimensions="{dims_kji[0]} {dims_kji[1]} {dims_kji[2]}">
          {bin_rel}
        </DataItem>
      </Attribute>"""


def _xdmf_rect(path, grid_name, nnode_kji, grid_files, attributes):
    """A VXVYVZ rectilinear grid, as CHAPSim2 writes for Cartesian cases."""
    nk, nj, ni = nnode_kji
    geom = ''.join(f"""        <DataItem ItemType="Uniform"
                  Dimensions="{n}"
                  NumberType="Float"
                  Precision="8"
                  Format="Binary"
                  Seek="4">
          {f}
        </DataItem>
""" for f, n in zip(grid_files, (ni, nj, nk)))
    with open(path, 'w') as fh:
        fh.write(f"""<?xml version="1.0" ?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="{grid_name}" GridType="Uniform">
      <Topology TopologyType="3DRectMesh" Dimensions="{nk} {nj} {ni}"/>
      <Geometry GeometryType="VXVYVZ">
{geom}      </Geometry>
{chr(10).join(attributes)}
    </Grid>
  </Domain>
</Xdmf>
""")


def _xdmf_curvilinear(path, grid_name, nnode_kji, mesh_file, attributes):
    """A 3DSMesh/XYZ grid, as CHAPSim2 writes for pipe and annulus cases."""
    nk, nj, ni = nnode_kji
    with open(path, 'w') as fh:
        fh.write(f"""<?xml version="1.0" ?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="{grid_name}" GridType="Uniform">
      <Topology TopologyType="3DSMesh" Dimensions="{nk} {nj} {ni}"/>
      <Geometry GeometryType="XYZ">
        <DataItem ItemType="Uniform"
                  Dimensions="{nk * nj * ni} 3"
                  NumberType="Float"
                  Precision="8"
                  Format="Binary"
                  Endian="Little"
                  Seek="12">
          {mesh_file}
        </DataItem>
      </Geometry>
{chr(10).join(attributes)}
    </Grid>
  </Domain>
</Xdmf>
""")


def _xdmf_collection(path, outer_name, members):
    """A slice bundle: one Collection holding a uniform grid per slice."""
    blocks = []
    for grid_name, nnode_kji, grid_files, attributes in members:
        nk, nj, ni = nnode_kji
        geom = ''.join(f"""        <DataItem ItemType="Uniform"
                  Dimensions="{n}"
                  NumberType="Float"
                  Precision="8"
                  Format="Binary"
                  Seek="4">
          {f}
        </DataItem>
""" for f, n in zip(grid_files, (ni, nj, nk)))
        blocks.append(f"""    <Grid Name="{grid_name}" GridType="Uniform">
      <Topology TopologyType="3DRectMesh" Dimensions="{nk} {nj} {ni}"/>
      <Geometry GeometryType="VXVYVZ">
{geom}      </Geometry>
{chr(10).join(attributes)}
    </Grid>""")
    with open(path, 'w') as fh:
        fh.write(f"""<?xml version="1.0" ?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="{outer_name}" GridType="Collection" CollectionType="Spatial">
{chr(10).join(blocks)}
    </Grid>
  </Domain>
</Xdmf>
""")


# =====================================================================================
# Synthetic cases
# =====================================================================================

#: Shape of the synthetic cases. Small enough to build in milliseconds, big
#: enough that an axis swap or an off-by-one shows up.
NX, NY, NZ = 6, 8, 4
TIMESTEP = '60'


def build_cartesian_case(root, name='chan_iso_periodic'):
    """Write a channel-like case: y in [-1, 1], x and z periodic.

    Carries an instantaneous 3-D field, a tsp_avg plane, a slice bundle and
    a tsp_avg profile table, so one case exercises every tier the readers
    have to cope with.

    Returns a dict describing the exact values written, for assertions.
    """
    d = _case_dirs(root, name)
    x_nodes = np.linspace(0.0, 3.0, NX + 1)
    y_nodes = np.linspace(-1.0, 1.0, NY + 1)
    z_nodes = np.linspace(0.0, 2.0, NZ + 1)
    for axis, nodes in (('x', x_nodes), ('y', y_nodes), ('z', z_nodes)):
        _write_grid_1d(os.path.join(d['mesh'], f'domain1_grid_{axis}.bin'), nodes)
    # The averaged plane keeps one cell in z, so its grid file holds 2 nodes.
    _write_grid_1d(os.path.join(d['mesh'], 'domain1_grid_zi1.bin'), z_nodes[:2])
    _write_grid_1d(os.path.join(d['mesh'], 'domain1_grid_yi3.bin'), y_nodes[3:5])

    # Instantaneous field, (nz, ny, nx), distinct per axis so a transposed
    # read cannot pass.
    k, j, i = np.meshgrid(np.arange(NZ), np.arange(NY), np.arange(NX), indexing='ij')
    qx = (100.0 * k + 10.0 * j + i).astype(np.float32)
    qy = (-qx).astype(np.float32)
    inst_bin = os.path.join(d['data'], f'domain1_flow_visu_{TIMESTEP}.bin')
    with open(inst_bin, 'wb') as fh:
        qx.tofile(fh)
        qy.tofile(fh)
    nbytes = qx.nbytes
    _xdmf_rect(
        os.path.join(d['xdmf'], f'domain1_flow_{TIMESTEP}.xdmf'), 'flow',
        (NZ + 1, NY + 1, NX + 1),
        ['../mesh/domain1_grid_x.bin', '../mesh/domain1_grid_y.bin',
         '../mesh/domain1_grid_z.bin'],
        [_attribute('qx_ccc', (NZ, NY, NX), f'../data/domain1_flow_visu_{TIMESTEP}.bin', 0),
         _attribute('qy_ccc', (NZ, NY, NX), f'../data/domain1_flow_visu_{TIMESTEP}.bin', nbytes)])

    # tsp_avg plane: averaged over z, so one cell in k.
    u1 = np.arange(NY * NX, dtype=np.float32).reshape(1, NY, NX)
    plane_bin = os.path.join(d['data'], f'domain1_tsp_avg_flow_zi1_{TIMESTEP}.bin')
    u1.tofile(plane_bin)
    _xdmf_rect(
        os.path.join(d['xdmf'], f'domain1_tsp_avg_flow_zi1_{TIMESTEP}.xdmf'),
        'tsp_avg_flow_zi1', (2, NY + 1, NX + 1),
        ['../mesh/domain1_grid_x.bin', '../mesh/domain1_grid_y.bin',
         '../mesh/domain1_grid_zi1.bin'],
        [_attribute('tsp_avg_u1', (1, NY, NX),
                    f'../data/domain1_tsp_avg_flow_zi1_{TIMESTEP}.bin', 0)])

    # Slice bundle: two slices sharing attribute names, which is what makes
    # a grid collection worth testing at all.
    yi3 = np.full((NZ, 1, NX), 3.0, dtype=np.float32)
    zi1 = np.full((1, NY, NX), 7.0, dtype=np.float32)
    bundle = os.path.join(d['data'], f'domain1_flow_slices_visu_{TIMESTEP}.bin')
    with open(bundle, 'wb') as fh:
        yi3.tofile(fh)
        zi1.tofile(fh)
    rel = f'../data/domain1_flow_slices_visu_{TIMESTEP}.bin'
    _xdmf_collection(
        os.path.join(d['xdmf'], f'domain1_flow_slices_visu_{TIMESTEP}.xdmf'),
        f'domain1_flow_slices_visu_{TIMESTEP}',
        [(f'domain1_flow_y_slice1_yi3_iter{TIMESTEP}', (NZ + 1, 2, NX + 1),
          ['../mesh/domain1_grid_x.bin', '../mesh/domain1_grid_yi3.bin',
           '../mesh/domain1_grid_z.bin'],
          [_attribute('qx_ccc', (NZ, 1, NX), rel, 0)]),
         (f'domain1_flow_z_slice1_zi1_iter{TIMESTEP}', (2, NY + 1, NX + 1),
          ['../mesh/domain1_grid_x.bin', '../mesh/domain1_grid_y.bin',
           '../mesh/domain1_grid_zi1.bin'],
          [_attribute('qx_ccc', (1, NY, NX), rel, yi3.nbytes)])])

    # tsp_avg profile table, the shape a doubly periodic case produces.
    yc = 0.5 * (y_nodes[:-1] + y_nodes[1:])
    prof_u1 = np.linspace(0.0, 1.0, NY)
    prof_uu = np.linspace(2.0, 3.0, NY)
    with open(os.path.join(d['data'],
                           f'domain1_tsp_avg_flow_yprofile_{TIMESTEP}.dat'), 'w') as fh:
        fh.write('# CHAPSim2 time-and-space averaged profile\n'
                 '# format_version: CHAPSim_profile_ascii_v1\n'
                 '# direction: y\n# coordinate: yc\n'
                 f'# npoints: {NY}\n'
                 '# columns: index yc tsp_avg_u1 tsp_avg_uu11\n')
        for n in range(NY):
            fh.write(f'{n + 1:8d} {yc[n]:24.16E} {prof_u1[n]:24.16E} {prof_uu[n]:24.16E}\n')

    return {'dirs': d, 'x_nodes': x_nodes, 'y_nodes': y_nodes, 'z_nodes': z_nodes,
            'yc': yc, 'qx': qx, 'qy': qy, 'u1_plane': u1[0],
            'prof_u1': prof_u1, 'prof_uu11': prof_uu,
            'slice_yi3': 3.0, 'slice_zi1': 7.0, 'timestep': TIMESTEP}


def build_cylindrical_case(root, name='pipe_iso_periodic', inner=0.0):
    """Write a pipe (inner=0) or annulus (0<inner<1) case.

    The mesh is one XYZ point list built the way CHAPSim2 builds it:
    x axial, y = r cos(theta), z = r sin(theta).
    """
    d = _case_dirs(root, name)
    x_nodes = np.linspace(0.0, 4.0, NX + 1)
    r_nodes = np.linspace(inner, 1.0, NY + 1)
    th_nodes = np.linspace(0.0, 2.0 * np.pi, NZ + 1)

    ii, jj, kk = np.meshgrid(np.arange(NX + 1), np.arange(NY + 1),
                             np.arange(NZ + 1), indexing='ij')
    X = x_nodes[ii].transpose(2, 1, 0)
    R = r_nodes[jj].transpose(2, 1, 0)
    TH = th_nodes[kk].transpose(2, 1, 0)
    _write_grid_xyz(os.path.join(d['mesh'], 'domain1_grids_3d.bin'),
                    X, R * np.cos(TH), R * np.sin(TH),
                    (NX + 1, NY + 1, NZ + 1))

    # A laminar-like profile: maximum on the axis, no slip at the wall, so
    # wall-side detection has something physical to find.
    rc = 0.5 * (r_nodes[:-1] + r_nodes[1:])
    prof = (1.0 - rc ** 2).astype(np.float32)
    field = np.broadcast_to(prof[None, :, None], (NZ, NY, NX)).astype(np.float32)
    bin_path = os.path.join(d['data'], f'domain1_flow_visu_{TIMESTEP}.bin')
    np.ascontiguousarray(field).tofile(bin_path)
    _xdmf_curvilinear(
        os.path.join(d['xdmf'], f'domain1_flow_{TIMESTEP}.xdmf'), 'flow',
        (NZ + 1, NY + 1, NX + 1), '../mesh/domain1_grids_3d.bin',
        [_attribute('qx_ccc', (NZ, NY, NX),
                    f'../data/domain1_flow_visu_{TIMESTEP}.bin', 0)])

    return {'dirs': d, 'x_nodes': x_nodes, 'r_nodes': r_nodes,
            'theta_nodes': th_nodes, 'rc': rc, 'profile': prof,
            'timestep': TIMESTEP, 'inner': inner}


def write_monitor_files(case_dir, thermo=True, nrows=6):
    """Write the three monitor files, in the current column layouts."""
    monitor = os.path.join(case_dir, '3_monitor')
    os.makedirs(monitor, exist_ok=True)
    t = np.linspace(1e-5, 6e-5, nrows)

    # Point monitor. 't' is time and 'T' is temperature: the two differ only
    # by case, which is exactly what a column lookup has to get right.
    cols = ['iteration', 't', 'u', 'v', 'w', 'p', 'phi'] + (['T'] if thermo else [])
    with open(os.path.join(monitor, 'domain1_monitor_pt1_flow.dat'), 'w') as fh:
        fh.write(' # domain-id :            1 pt-id :            1\n')
        fh.write(' # probe pts location    1.0  0.0  0.5\n')
        fh.write(' # ' + ', '.join(cols) + '\n')
        for n in range(nrows):
            row = [n + 1, t[n], 1.5, 0.01, -0.02, 0.3, -0.004] + ([570.0] if thermo else [])
            fh.write(' '.join(f'{v:14.5E}' for v in row) + '\n')

    # Bulk history: 16 columns with thermo, 11 without.
    names = ['time', 'global mass balance', 'max. mass conservation (interior)',
             'max. mass conservation (inlet)', 'max. mass conservation (outlet)',
             'total kinetic energy', 'mean dpdx', 'global pressure drop',
             'bulk velocity qx', 'bulk velocity qy', 'bulk velocity qz']
    values = [0.0, 1e-15, 1e-15, 1e-15, 1e-15, 0.61, -7.5e-4, 0.048, 1.002, 0.0, 2.5e-3]
    if thermo:
        names += ['bulk mass flux gx', 'bulk mass flux gy', 'bulk mass flux gz',
                  'bulk enthalpy', 'bulk temperature']
        values += [1.002, 0.0, 2.5e-3, 6.7e-12, 1.0]
    with open(os.path.join(monitor, 'domain1_monitor_metrics_history.log'), 'w') as fh:
        fh.write(' # domain-id :            1 pt-id :            0\n')
        fh.write(' # columns description:\n')
        for n, name in enumerate(names, start=1):
            fh.write(f' # column {n:2d} : {name}\n')
        for n in range(nrows):
            row = list(values)
            row[0] = t[n]
            fh.write(' '.join(f'{v:17.8E}' for v in row) + '\n')

    # Change history: prose header, 15 columns.
    with open(os.path.join(monitor, 'domain1_monitor_change_history.log'), 'w') as fh:
        fh.write(' # domain-id :            1 pt-id :            0\n')
        fh.write(' # columns: time; physical mass residual at bulk, inlet, outlet;\n')
        fh.write(' #          total mass; total mass drift; kinetic energy change rate\n')
        for n in range(nrows):
            row = [t[n]] + [1e-15] * 11 + [64.0, -3.4e-7, -1.05e-2]
            fh.write(' '.join(f'{v:17.8E}' for v in row) + '\n')

    return monitor
