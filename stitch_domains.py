"""
Stitch two CHAPSim2 domains together along the streamwise (x) direction. This is not recommended.

Reads XDMF + binary data from two separate domain directories, concatenates
grids and flow fields along x, and writes a new set of XDMF + binary files
for the combined domain.

Usage:
    python stitch_domains.py

Configure the paths and settings in the __main__ block at the bottom.
"""

import os
import struct
import numpy as np
import xml.etree.ElementTree as ET
from utils import parse_xdmf_file


def read_domain(xdmf_path):
    """
    Read all 3D data from a single XDMF file.

    Returns:
        arrays: dict of {var_name: np.ndarray} with shape (nz, ny, nx)
        grid_info: dict with 'grid_x', 'grid_y', 'grid_z', 'node_dimensions', etc.
    """
    return parse_xdmf_file(xdmf_path, load_all_vars=True)


def write_grid_binary(filepath, data):
    """
    Write a 1D grid coordinate array to a CHAPSim2-format binary file.

    Grid files use stream I/O: a 4-byte int32 element count header, followed
    by the raw float64 coordinate data. The XDMF references these with Seek="4"
    to skip the header.

    See CHAPSim2/src/io_visulisation.f90 write_visu_ini():
        write(iogrid) int(size(xp1), kind=int32)
        write(iogrid) xp1
    """
    flat = data.astype(np.float64).ravel()
    with open(filepath, 'wb') as f:
        f.write(struct.pack('<i', len(flat)))  # 4-byte int32: element count
        flat.tofile(f)


def write_field_binary(filepath, data):
    """
    Write a flow field array to a raw binary file (no header).

    Flow field data is written by decomp_2d_write_one (MPI-IO) as raw float64
    with no record markers or headers. The XDMF references these without Seek.

    See CHAPSim2/src/io_tools.f90 write_one_3d_array() -> decomp_2d_write_one().
    """
    flat = data.astype(np.float64).ravel()
    with open(filepath, 'wb') as f:
        flat.tofile(f)


def generate_xdmf(grid_name, node_dims, cell_dims, grid_files, attributes, output_path):
    """
    Generate an XDMF file referencing the binary data files.

    Args:
        grid_name: Name for the XDMF grid (e.g. 'flow', 'tsp_avg_flow')
        node_dims: (nz, ny, nx) node dimensions
        cell_dims: (nz, ny, nx) cell dimensions
        grid_files: dict with 'x', 'y', 'z' paths relative to the XDMF file
        attributes: list of (var_name, bin_path_relative) tuples
        output_path: where to write the .xdmf file
    """
    nz, ny, nx = node_dims

    root = ET.Element('Xdmf', Version='3.0')
    domain = ET.SubElement(root, 'Domain')
    grid = ET.SubElement(domain, 'Grid', Name=grid_name, GridType='Uniform')

    # Topology
    ET.SubElement(grid, 'Topology',
                  TopologyType='3DRectMesh',
                  Dimensions=f' {nz} {ny} {nx}')

    # Geometry
    geom = ET.SubElement(grid, 'Geometry', GeometryType='VXVYVZ')
    for coord, dim in [('x', nx), ('y', ny), ('z', nz)]:
        di = ET.SubElement(geom, 'DataItem',
                           ItemType='Uniform',
                           Dimensions=str(dim),
                           NumberType='Float',
                           Precision='8',
                           Format='Binary',
                           Seek='4')
        di.text = f'\n          {grid_files[coord]}\n        '

    # Attributes (flow variables) — no Seek, raw binary from decomp_2d
    cz, cy, cx = cell_dims
    for var_name, bin_rel_path in attributes:
        attr = ET.SubElement(grid, 'Attribute',
                             Name=var_name,
                             AttributeType='Scalar',
                             Center='Cell')
        di = ET.SubElement(attr, 'DataItem',
                           ItemType='Uniform',
                           NumberType='Float',
                           Precision='8',
                           Format='Binary',
                           Dimensions=f'{cz} {cy} {cx}')
        di.text = f'\n          {bin_rel_path}\n        '

    # Write with XML declaration
    tree = ET.ElementTree(root)
    ET.indent(tree, space='  ')
    tree.write(output_path, xml_declaration=True, encoding='unicode')
    print(f"Wrote XDMF: {output_path}")


def stitch_domains(xdmf_path_1, xdmf_path_2, output_dir, output_prefix='stitched',
                   x_offset=None):
    """
    Stitch two domains along the streamwise (x) direction.

    Args:
        xdmf_path_1: Path to the first (upstream) domain XDMF file
        xdmf_path_2: Path to the second (downstream) domain XDMF file
        output_dir: Directory to write combined output files
        output_prefix: Filename prefix for output files (default: 'stitched')
        x_offset: If set, shift domain 2's x-coordinates by this amount relative
                  to domain 1's x_max. If None, domain 2's x-grid is appended directly
                  (use this if the absolute coordinates are already correct).
    """
    print(f"Reading domain 1: {xdmf_path_1}")
    arrays_1, grid_1 = read_domain(xdmf_path_1)
    print(f"Reading domain 2: {xdmf_path_2}")
    arrays_2, grid_2 = read_domain(xdmf_path_2)

    if not arrays_1 or not arrays_2:
        raise RuntimeError("Failed to read one or both XDMF files.")

    # --- Validate y and z grids match ---
    y1, y2 = grid_1['grid_y'], grid_2['grid_y']
    z1, z2 = grid_1['grid_z'], grid_2['grid_z']

    if y1.shape != y2.shape or not np.allclose(y1, y2, atol=1e-12):
        raise ValueError(f"Y grids do not match: shapes {y1.shape} vs {y2.shape}")
    if z1.shape != z2.shape or not np.allclose(z1, z2, atol=1e-12):
        raise ValueError(f"Z grids do not match: shapes {z1.shape} vs {z2.shape}")

    print(f"Y grids match ({y1.shape[0]} nodes), Z grids match ({z1.shape[0]} nodes)")

    # --- Stitch x grid ---
    x1 = grid_1['grid_x']
    x2 = grid_2['grid_x']

    if x_offset is not None:
        # Shift domain 2 so it starts at domain 1's x_max + offset
        x2_shifted = x2 - x2[0] + x1[-1] + x_offset
    else:
        x2_shifted = x2

    x_combined = np.concatenate([x1, x2_shifted])

    nx_combined = len(x_combined)
    ny = len(y1)
    nz = len(z1)
    node_dims = (nz, ny, nx_combined)
    cell_dims = (nz - 1, ny - 1, nx_combined - 1)

    print(f"Domain 1: x = [{x1[0]:.4f}, {x1[-1]:.4f}], nx = {len(x1)}")
    print(f"Domain 2: x = [{x2_shifted[0]:.4f}, {x2_shifted[-1]:.4f}], nx = {len(x2)}")
    print(f"Combined: x = [{x_combined[0]:.4f}, {x_combined[-1]:.4f}], nx = {nx_combined}")
    print(f"Combined node dimensions: {node_dims}")
    print(f"Combined cell dimensions: {cell_dims}")

    # --- Create output directories (CHAPSim2's 2_visu/{xdmf,data,mesh}) ---
    visu_dir = os.path.join(output_dir, '2_visu')
    xdmf_dir = os.path.join(visu_dir, 'xdmf')
    data_dir = os.path.join(visu_dir, 'data')
    mesh_dir = os.path.join(visu_dir, 'mesh')
    for directory in (visu_dir, xdmf_dir, data_dir, mesh_dir):
        os.makedirs(directory, exist_ok=True)

    # --- Write grid binary files (with 4-byte element count header) ---
    write_grid_binary(os.path.join(mesh_dir, f'{output_prefix}_grid_x.bin'), x_combined)
    write_grid_binary(os.path.join(mesh_dir, f'{output_prefix}_grid_y.bin'), y1)
    write_grid_binary(os.path.join(mesh_dir, f'{output_prefix}_grid_z.bin'), z1)
    print("Wrote grid files")

    # --- Stitch and write flow field arrays ---
    # Find common variables between the two domains
    common_vars = sorted(set(arrays_1.keys()) & set(arrays_2.keys()))
    missing_in_2 = set(arrays_1.keys()) - set(arrays_2.keys())
    missing_in_1 = set(arrays_2.keys()) - set(arrays_1.keys())

    if missing_in_2:
        print(f"Warning: variables only in domain 1 (skipped): {missing_in_2}")
    if missing_in_1:
        print(f"Warning: variables only in domain 2 (skipped): {missing_in_1}")

    print(f"Stitching {len(common_vars)} common variables...")

    attributes = []
    for var_name in common_vars:
        a1 = arrays_1[var_name]
        a2 = arrays_2[var_name]

        # Flow fields are (nz, ny, nx) — concatenate along axis=2 (x)
        if a1.ndim == 3 and a2.ndim == 3:
            combined = np.concatenate([a1, a2], axis=2)
        elif a1.ndim == 1 and a2.ndim == 1:
            # 1D profile data (e.g. from tsp_avg which is space-averaged)
            combined = np.concatenate([a1, a2])
        else:
            print(f"  Skipping {var_name}: incompatible shapes {a1.shape} vs {a2.shape}")
            continue

        # Write raw binary (no header), matching decomp_2d_write_one format
        bin_filename = f'{output_prefix}_{var_name}.bin'
        write_field_binary(os.path.join(data_dir, bin_filename), combined)
        attributes.append((var_name, f'../data/{bin_filename}'))
        print(f"  {var_name}: {a1.shape} + {a2.shape} -> {combined.shape}")

    # --- Write XDMF file ---
    # Determine grid name from the original XDMF
    original_name = os.path.basename(xdmf_path_1)
    # Extract grid name from original (e.g. 'flow', 'tsp_avg_flow')
    # Pattern: domain1_<grid_name>_<timestep>.xdmf
    parts = original_name.replace('.xdmf', '').split('_', 1)
    if len(parts) > 1:
        # Remove the trailing timestep
        name_with_ts = parts[1]
        # Split off the last numeric segment (timestep)
        name_parts = name_with_ts.rsplit('_', 1)
        if len(name_parts) > 1 and name_parts[1].isdigit():
            grid_name = name_parts[0]
            timestep = name_parts[1]
        else:
            grid_name = name_with_ts
            timestep = ''
    else:
        grid_name = 'flow'
        timestep = ''

    grid_files = {
        'x': f'../mesh/{output_prefix}_grid_x.bin',
        'y': f'../mesh/{output_prefix}_grid_y.bin',
        'z': f'../mesh/{output_prefix}_grid_z.bin',
    }

    if timestep:
        xdmf_filename = f'{output_prefix}_{grid_name}_{timestep}.xdmf'
    else:
        xdmf_filename = f'{output_prefix}_{grid_name}.xdmf'

    xdmf_path = os.path.join(xdmf_dir, xdmf_filename)
    generate_xdmf(grid_name, node_dims, cell_dims, grid_files, attributes, xdmf_path)

    print(f"\nDone! Stitched domain written to: {output_dir}")
    print(f"  XDMF: {xdmf_path}")
    print(f"  Data: {data_dir}/")

    return xdmf_path


if __name__ == '__main__':

    # ==================================================================================
    # CONFIGURATION — edit these paths to match your setup
    # ==================================================================================

    # Paths to the two domain XDMF files to stitch
    # These should be the same type (e.g. both 'flow', both 'tsp_avg_flow', etc.)
    XDMF_DOMAIN_1 = '/path/to/case1/2_visu/xdmf/domain1_flow_680000.xdmf'
    XDMF_DOMAIN_2 = '/path/to/case2/2_visu/xdmf/domain1_flow_680000.xdmf'

    # Output directory for the stitched domain
    OUTPUT_DIR = '/path/to/output/stitched_case'

    # Prefix for output filenames (replaces 'domain1' in the originals)
    OUTPUT_PREFIX = 'stitched'

    # X offset: spacing between domain 1's last node and domain 2's first node.
    # Set to None if domain 2's x-coordinates are already in the correct global frame.
    # Set to a float (e.g. the grid spacing dx) if domain 2 starts from x=0 and needs shifting.
    X_OFFSET = None  # e.g. 0.03141593 for a typical dx

    # ==================================================================================

    stitch_domains(XDMF_DOMAIN_1, XDMF_DOMAIN_2, OUTPUT_DIR,
                   output_prefix=OUTPUT_PREFIX, x_offset=X_OFFSET)
