# CHAPSim2 Toolkit

[![tests](https://github.com/weiwangstfc/CHAPSim2-toolkit/actions/workflows/tests.yml/badge.svg)](https://github.com/weiwangstfc/CHAPSim2-toolkit/actions/workflows/tests.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

Python pre- and post-processing for the
[CHAPSim2](https://github.com/CHAPSim/CHAPSim2) direct numerical simulation
solver, built on NumPy and Matplotlib. It reads CHAPSim2's XDMF, binary and
ASCII output directly and produces:

- **turbulence statistics**: mean velocity and temperature, Reynolds
  stresses and their budgets, TKE, vorticity, anisotropy (Lumley triangles),
  1D energy spectra, two-point correlations, wall shear, Nusselt and
  turbulent Prandtl numbers, and MHD current density and Lorentz force;
- **visualisation**: 2D slices of any field, and 3D slices, isosurfaces and
  volume rendering;
- **run monitoring**: bulk and probe-point histories;
- **pre-run mesh analysis**: checks a mesh against DNS resolution limits
  before the job is submitted.

Channels and ducts (Cartesian) and pipes and annuli (cylindrical) are all
supported, with and without heat transfer and MHD.

The toolkit is copyright the Science and Technology Facilities Council
(STFC), UKRI, and The University of Sheffield, and is released under the
BSD 3-Clause licence. It is developed under the
[CCP-NTH](https://ccpnth.ac.uk/) project.

## Contents

- [Quick start](#quick-start)
- [Installation](#installation)
- [Tools](#tools)
- [CHAPSim2 output it reads](#chapsim2-output-it-reads)
- [Run parameters come from the case](#run-parameters-come-from-the-case)
- [Fluid properties](#fluid-properties)
- [Figure provenance](#figure-provenance)
- [Testing](#testing)
- [Changes](#changes)
- [Contributors](#contributors)
- [Citing](#citing)
- [Licence and copyright](#licence-and-copyright)
- [Acknowledgements](#acknowledgements)
- [Reference data](#reference-data)

## Quick start

```bash
git clone https://github.com/weiwangstfc/CHAPSim2-toolkit.git
cd CHAPSim2-toolkit
pip install -e ".[gui]"

chapsim2-gui                    # the graphical interface, on a workstation
```

On a cluster, use the scripts instead of the GUI. Copy the shipped
`config.py` (in `chapsim2_toolkit/`) beside your data, edit it, and run from
there:

```bash
cd /scratch/me/run12
cp /path/to/CHAPSim2-toolkit/chapsim2_toolkit/config.py .
python -m chapsim2_toolkit.turb_stats          # or: chapsim2-turbstats
```

## Installation

The code lives in the `chapsim2_toolkit/` package. It works installed or
straight from a clone. Both are supported, and a shared group checkout on a
cluster is expected to use the clone.

```bash
pip install -e .                  # from a clone: editable, so git pull keeps working
pip install ".[gui,3d]"           # a normal install, with the GUI and 3D extras
pip install -r requirements.txt   # just the dependencies, no install
conda env create -f environment.yml
```

**Requirements.** The scripts need only NumPy, Matplotlib and tqdm, and run
on Python 3.8 or newer. Each optional dependency is used by one part only:

| Package | Needed by |
|---|---|
| `ttkbootstrap` ≥ 2 | the GUI, which therefore needs **Python 3.10+** (on ttkbootstrap 1 the main window cannot be built at all) |
| `pyvista` | 3D visualisation |
| `pandas` | `thermal_BC_calc.py` |

**Entry points.** An install provides console scripts and module entry
points:

```bash
chapsim2-gui
chapsim2-turbstats --config config.py
python -m chapsim2_toolkit.turb_stats --config config.py
```

From a clone with nothing installed, the top-level scripts are thin wrappers
onto the package, so the old commands still work:

```bash
python gui.py
python /path/to/CHAPSim2-toolkit/turb_stats.py     # from your data directory
```

Use `python -m chapsim2_toolkit.<script>` where you can. It puts your working
directory first on `sys.path`, which is how it finds the `config.py` sitting
beside your data.

## Tools

| Tool | Command | What it does |
|---|---|---|
| GUI | `chapsim2-gui` | All of the below in one window |
| Turbulence statistics | `chapsim2-turbstats` | Profiles, stresses, budgets, spectra and correlations for one or more cases, set up in `config.py` |
| Quick statistics | `chapsim2-quickstats` | One case, two figures, minimal dependencies |
| Slices | `chapsim2-slice` | 2D maps of any output field |
| Monitors | `chapsim2-monitor` | Bulk and probe-point histories |
| Mesh analysis | `chapsim2-mesh` | Resolution check of an `input_chapsim.ini` |
| 3D visualisation | `python turb_visu.py` | Slices, isosurfaces and volume rendering (PyVista) |
| Thermal boundary conditions | `python thermal_BC_calc.py` | Grashof number to wall ΔT or heat flux; fluid property tables |
| Domain stitching | `python stitch_domains.py` | Joins two domains along x (not recommended) |

**gui.py** covers turbulence statistics, slice visualisation, monitoring
points, 3D visualisation and mesh analysis. Choose a case once in the Case
bar at the top and every tab is set up from it. Each tab works out which data
types and timesteps exist, and reads everything the run recorded in its
`input_chapsim.ini` (see [below](#run-parameters-come-from-the-case)). A Help
tab covers each tab and the output formats. The Mesh Analysis tab is
interactive: load an `input_chapsim.ini` (or start from the built-in
template), adjust cell counts, stretching and flow parameters with sliders,
and the resolution report, headline metrics and spacing plot update live.
The adjusted settings can be written back out as an input file. The GUI is
unlikely to work on an HPC system; use the individual scripts there.

**turb_stats.py** is the main post-processing script. It produces velocity,
temperature and Reynolds stress profiles, Reynolds stress budget terms, and
heat transfer statistics such as the turbulent Prandtl and Nusselt numbers,
from CHAPSim2 ASCII profile tables or XDMF output. The cases to compare, the
statistics to produce and the plotting options are set in a `config.py`:
**the one in your working directory**, or one named with
`--config path/to/config.py`, falling back to the toolkit's own. The file
actually used is printed at the start of every run. Copy the shipped
`config.py` beside your data and edit that copy, not the one in the
checkout, especially if the checkout is shared. The run's own parameters are
not set there; they are read from each case's `input_chapsim.ini`. Plots are
saved to `output_dir` when it is set, otherwise to `turb_stats_plots/` beside
the toolkit.

**quick_turb_stats.py** post-processes a single case into one figure of
velocity, TKE and temperature and one of Reynolds stresses. It needs only
NumPy, Matplotlib and tqdm, which suits HPC systems. Run it on a serial,
data-analysis or interactive node, since bandwidth is usually throttled on
login nodes. Interactive input.

**slice.py** draws 2D maps of any output field, with Matplotlib plotting
options. As above, run it on a serial or interactive node. Interactive input.

**monitor_points.py** plots bulk and point monitors, and can crop diverged
data. Columns are read from each file's own header, so a monitor file that
gains or loses a column still plots the right quantity under the right
label. Pass a case folder or its `3_monitor` directory, or run the script
inside one. Interactive input.

**mesh_analysis.py** is a pre-processing resolution check. It reads a case's
`input_chapsim.ini`, rebuilds the wall-normal grid exactly as the solver
does, and assesses it against DNS requirements: dy+, dx+, dz+, grid
stretching, the MHD boundary layer, and the recommended minimum mesh and
time step. It is a Python port of the `estimate_spacial_resolution` and
`estimate_temporal_resolution` routines in CHAPSim2's `apx_prerun_mod`, so a
mesh can be checked before a job is submitted. Run
`python mesh_analysis.py path/to/input_chapsim.ini`, or with no argument to
be prompted. It can also save a plot of the mesh distribution.

**thermal_BC_calc.py** converts a Grashof number to a wall temperature
difference or heat flux. It also tabulates any fluid's properties in the
format CHAPSim2 reads: the same eight SI columns as a `NIST_*.DAT`, so the
file it writes can be given back to the solver as a table-based fluid.

## CHAPSim2 output it reads

The toolkit targets the current CHAPSim2 output layout:

```
<case>/
├── 1_data/                        restart and checkpoint binaries, spectra, stats bundles
├── 2_visu/
│   ├── xdmf/                      .xdmf descriptors
│   ├── data/                      field binaries (.bin) + bundle manifests (*_meta_*.dat)
│   └── mesh/                      grid coordinate binaries
├── 3_monitor/                     monitor point files and history logs
└── 4_check/                       mesh and property check tables
```

Point any script at the case folder (`2_visu` or `2_visu/xdmf` are also
accepted) and it finds the rest. Older runs that kept everything flat in
`2_visu/`, with the binaries in `1_data/`, still load, but they are not the
target.

Time-and-space averaged (`tsp_avg`) statistics come in one of two shapes,
depending on how many directions are periodic. Both are read transparently:

- **one periodic direction** (e.g. a spatially developing channel): a 2D
  plane, `2_visu/xdmf/domain1_tsp_avg_flow_zi1_<iter>.xdmf`, tagged with the
  direction averaged out;
- **two periodic directions** (the canonical channel): a 1D ASCII profile
  table, `2_visu/data/domain1_tsp_avg_flow_yprofile_<iter>.dat`, with its
  columns named in the header.

Instantaneous 2D slices likewise come either as per-slice files
(`domain1_flow_yi8_<iter>.xdmf`) or packed into a per-timestep bundle
(`domain1_flow_slices_visu_<iter>.xdmf`), a grid collection holding every
slice. The scripts offer the same slice list either way.

Pipe and annulus cases write a curvilinear mesh (one XYZ point list), because
their grid is rectilinear only in (x, r, θ). Those three axes are recovered
from it, so `grid_y` holds the radius and the wall-normal statistics work as
they do for a channel. Slices of a cylindrical case are plotted in
computational coordinates: a cross-section appears unrolled as (r, θ), with
the axes labelled accordingly, not as a disc.

## Run parameters come from the case

A case's `input_chapsim.ini` is the solver's own record of how the run was
set up, so the toolkit reads it instead of asking for the same numbers
again. On loading a case, in the GUI or through `config.py`, these are
filled in:

| Parameter | From |
|---|---|
| Reynolds number | `[flow] ren` |
| reference temperature, reference length | `[thermo] ref_t0`, `ref_l0` |
| geometry | `[domain] icase` |
| heat transfer on, MHD on | `[thermo] ithermo`, `[mhd] imhd_xdom` |
| working fluid | `[thermo] ifluid` |
| gravity direction, magnetic field direction | `[thermo] igravity`, `[mhd] B_static` |
| Stuart number | `[mhd] NStuart`, or Ha²/Re from `NHartmn` |
| which directions may safely be averaged | `[bc] ifbc*` |

In `config.py` these are the settings left as `None`. Setting one overrides
the case and prints a warning saying the two disagree, because that is
usually a slip rather than an intention. A Reynolds number transcribed
wrongly rescales every u_τ-normalised profile, with nothing on the plot to
show for it.

The bulk velocity and the wall heat flux are outcomes of a run rather than
inputs to it, so they are not in the input file and are still given by hand.

The Nusselt number evaluates the conductivity at the bulk temperature, which
varies along a heated duct, not at the reference temperature. For
supercritical water that is the difference between a right answer and one up
to 77% out, since `k` falls by a factor of five through the pseudo-critical
region.

Loading a case also checks that the data is the data the input file
describes. It rebuilds the mesh from `ncx/ncy/ncz`, `istret` and `rstret`
and compares it with the grid the solver wrote. An input file edited after
the run, or output copied in from a different case, would otherwise go
unnoticed and silently rescale every profile. A changed clustering is the
case that matters: the cell count still matches, so only the node positions
reveal it.

## Fluid properties

Properties are a port of the solver's own model in `src/input_thermo.f90`,
with the coefficients from `src/modules.f90`, because post-processing that
uses a different conductivity from the run's is not measuring that run. As in
the solver there are two kinds of fluid:

- **supercritical water and CO₂**, interpolated from the NIST tables. These
  are read from the case folder when it has them (the solver opens them by
  bare filename from its working directory, so that copy is the
  authoritative one), and otherwise from `Reference_Data/thermal_properties/`;
- **liquid metals**: sodium, lead, bismuth, LBE, lithium, FLiBe and PbLi-17,
  from polynomial correlations in T.

These are verified against the solver. Every coefficient is compared with
`modules.f90` by parsing it, and supercritical water is checked against the
`4_check/check_ftplist_dim.dat` property list the solver itself writes.

Everything is SI, with no exceptions: K, kg/m³, Pa·s, W/(m·K), J/(kg·K),
J/kg, 1/K and Pa. The NIST files give pressure in MPa and the solver keeps
it that way, so it is converted once on reading and nothing downstream has to
remember which. Enthalpy can be inverted too: `temperature_from_enthalpy` is
a port of the solver's `ftp_refresh_thermal_properties_from_H`, which is how
a bulk temperature is recovered from the bulk enthalpy in the data.

Enthalpy coefficients are derived from the heat-capacity ones rather than
transcribed, as the solver now does, so that `dH/dT = Cp` holds exactly. This
is checked numerically for every fluid.

The valid temperature range is the melting-to-boiling range intersected with
the validity range of each correlation that has one, and the toolkit records
which property sets each end. A fit does not hold over the whole liquid range
merely because the material is liquid there: PbLi-17 is valid from 521 to
625 K, both ends set by its viscosity correlation, against a phase range of
508 to 1943 K.

Where the solver would stop the run, at a temperature outside that range,
the toolkit returns NaN instead, so one bad cell leaves a gap rather than
costing the whole figure.

## Figure provenance

Every figure the toolkit saves records what made it, in the file's own
metadata: the toolkit version, the git revision and whether the working tree
was dirty, the cases and timesteps, the `config.py` used, and the time. PNG
files carry it in tEXt chunks and PDF files in the document info dictionary,
so nothing is drawn on the plot and nothing can be separated from it.

```bash
python -m chapsim2_toolkit.provenance figure.png
```
```
figure.png:
  cases            pipe_iso_periodic
  config           /scratch/me/run12/config.py
  created          2026-10-06T14:03:54+01:00
  git_dirty        yes
  git_revision     85592c6
  toolkit          CHAPSim2-toolkit 0.2.0 (85592c6-dirty)
```

`git_dirty` is the field that matters. A commit does not describe the code
that ran if the tree was modified, so a figure marked dirty cannot be
regenerated from that revision alone.

## Testing

```bash
python run_tests.py                 # everything
python run_tests.py test_xdmf       # one module
python run_tests.py -k cylindrical  # tests matching a name
```

The runner uses pytest when it is installed and falls back to a built-in
runner when it is not, so the suite also runs on a cluster where installing
packages is awkward. GitHub Actions runs the same commands on every push and
pull request, on the oldest and newest Python versions the package supports.

Most tests build small synthetic cases on disk in the formats CHAPSim2
writes, so nothing binary lives in the repository and the expected values
are analytic. `tests/test_solver_cases.py` additionally reads CHAPSim2's own
regression output, covering every case and every averaging tier, which is
what catches the solver's formats changing. Those tests skip themselves when
the solver is not checked out alongside; set `CHAPSIM2_TESTS` to point at its
`tests/` directory.

`tests/test_gui.py` builds a real widget tree and drives it: applying a case
to every tab, the values each tab takes from the case's `input_chapsim.ini`,
the timestep scans, the `config.py` round trip and the Help topics. Plotting
is not tested. These tests need `ttkbootstrap` and a display, and skip
without either. Run them with `xvfb-run -a python run_tests.py test_gui` on a
headless machine, which is what CI does.

`code_verification/` holds standalone checks of the budget terms, spectra
and two-point correlations against analytic cases.

## Changes

`CHANGELOG.md` records what changed between versions. Entries marked
**behaviour change** alter a number you may already have published: the
Nusselt number, the wall shear on a pipe, the fluid properties and the
Reynolds number truncation all changed in 0.2.0, some of them substantially.
Read those before regenerating a figure made with an earlier version.

## Contributors

- **Alex Old** (PhD student, The University of Sheffield): original and
  principal author. He designed and wrote the toolkit as part of his PhD.
- **Wei Wang** (Principal Computational Scientist, STFC, UKRI): co-author and
  maintainer. She co-supervised Alex's PhD project, added support for pipe
  and annulus cases, adapted the toolkit to the current CHAPSim2 output
  layout and the solver's fluid property model, and added the test suite and
  CI.

See [`CONTRIBUTORS.md`](CONTRIBUTORS.md) for roles and for how to contribute.
Contributions are welcome through issues and pull requests.

This repository is a fork of
[HeFT-Sheffield/CHAPSim2-toolkit](https://github.com/HeFT-Sheffield/CHAPSim2-toolkit),
where the toolkit was first developed.

The October 2026 work (compatibility with current CHAPSim2 output, the test
suite, CI and the GUI changes) was carried out with AI assistance. The
individual commits record this in their `Co-Authored-By` trailers.

## Citing

If you use the toolkit in published work, please cite it using
[`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button).

## Licence and copyright

Copyright (c) 2025-2026, Science and Technology Facilities Council (STFC),
UKRI, and The University of Sheffield.

Released under the [BSD 3-Clause licence](LICENSE), the same licence as the
CHAPSim2 solver.

Version 0.2.0 and earlier were released under the MIT licence.

## Acknowledgements

CHAPSim2 and this toolkit are developed under
[CCP-NTH](https://ccpnth.ac.uk/), the Collaborative Computational Project in
Nuclear Thermal Hydraulics, with computational support from CoSeC, the
Computational Science Centre for Research Communities.

## Reference data

Reference data is provided for an isothermal channel (MKM180), a square duct
(KTH), and isothermal and heated MHD channels (NK). All of it is openly
available from published sources. Copyright for the reference datasets
remains with their original authors and publishers; see the individual data
files for citations.
