# CHAPSim2-toolkit

[![tests](https://github.com/weiwangstfc/CHAPSim2-toolkit/actions/workflows/tests.yml/badge.svg)](https://github.com/weiwangstfc/CHAPSim2-toolkit/actions/workflows/tests.yml)

A python post-processing and toolkit program based on NumPy and Matplotlib for DNS solver CHAPSim2.

## Install:

The toolkit is run from a checkout — the modules sit at the top level, so an
install places only metadata and does not put them on the import path. There is
no `chapsim2-*` command and nothing becomes importable; `setup.py` exists to
pull in the dependencies and for nothing else. What you need is those:

```bash
pip install -r requirements.txt          # everything, including the GUI and 3D
pip install numpy matplotlib tqdm        # enough for the scripts on a cluster
conda env create -f environment.yml      # or a conda environment
conda activate chapsim2-toolkit
```

Then run the scripts in place, e.g. `python gui.py` or `python turb_stats.py`.

`pandas` is used only by thermal_BC_calc.py, `ttkbootstrap` only by gui.py and
`pyvista` only by the 3D visualisation; the post-processing scripts need just
numpy, matplotlib and tqdm.

The GUI needs **Python 3.10 or newer**, because it needs ttkbootstrap 2 — on
ttkbootstrap 1 the main window cannot be constructed at all. The scripts have
no such floor and run on 3.8.

## CHAPSim2 output it reads:

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

Point any script at the case folder (`2_visu` or `2_visu/xdmf` are also accepted) and it
finds the rest. Older runs that kept everything flat in `2_visu/` with the binaries in
`1_data/` still load, but they are not the target.

Time-and-space averaged (`tsp_avg`) statistics arrive in one of two shapes, depending on
how many directions are periodic, and both are read transparently:

- **one periodic direction** (e.g. a spatially developing channel): a 2-D plane,
  `2_visu/xdmf/domain1_tsp_avg_flow_zi1_<iter>.xdmf`, tagged with the direction averaged out;
- **two periodic directions** (the canonical channel): a 1-D ASCII profile table,
  `2_visu/data/domain1_tsp_avg_flow_yprofile_<iter>.dat`, with its columns named in the header.

Instantaneous 2-D slices likewise come either as per-slice files
(`domain1_flow_yi8_<iter>.xdmf`) or packed into a per-timestep bundle
(`domain1_flow_slices_visu_<iter>.xdmf`), a grid collection holding every slice; the
scripts offer the same slice list either way.

Pipe and annulus cases write a curvilinear mesh (one XYZ point list) because their grid
is only rectilinear in (x, r, θ). Those three axes are recovered from it, so `grid_y`
holds the radius and the wall-normal statistics work as they do for a channel. Slices of
a cylindrical case are plotted in computational coordinates — a cross-section appears
unrolled as (r, θ), with the axes labelled accordingly, not as a disc.

## Run parameters come from the case:

A case's `input_chapsim.ini` is the solver's own record of how the run was set
up, so the toolkit reads it instead of asking for the same numbers again. On
loading a case — in the GUI or through config.py — these are filled in:

| | from |
|---|---|
| Reynolds number | `[flow] ren` |
| reference temperature, reference length | `[thermo] ref_t0`, `ref_l0` |
| geometry | `[domain] icase` |
| heat transfer on, MHD on | `[thermo] ithermo`, `[mhd] imhd_xdom` |
| working fluid | `[thermo] ifluid` |
| gravity direction, magnetic field direction | `[thermo] igravity`, `[mhd] B_static` |
| Stuart number | `[mhd] NStuart`, or Ha²/Re from `NHartmn` |
| which directions may safely be averaged | `[bc] ifbc*` |

In config.py these are the settings left as `None`. Setting one overrides the
case and prints a warning saying the two disagree, because that is usually a
slip rather than an intention — a Reynolds number transcribed wrongly rescales
every u_τ-normalised profile with nothing on the plot to show for it.

The bulk velocity and the wall heat flux are outcomes of a run rather than
inputs to it, so they are not in the input file and are still given by hand.

The Nusselt number evaluates the conductivity at the bulk temperature, which
varies along a heated duct, not at the reference temperature. For
supercritical water that is the difference between a right answer and one up
to 77% out, since `k` falls by a factor of five through the pseudo-critical
region.

Loading a case also checks that the data is the data the input file describes,
by rebuilding the mesh from `ncx/ncy/ncz`, `istret` and `rstret` and comparing
it with the grid the solver wrote. An input file edited after the run, or
output copied in from a different case, would otherwise go unnoticed and
silently rescale every profile. A changed clustering is the case that matters:
the cell count still matches, so only the node positions reveal it.

## Fluid properties:

Properties are a port of the solver's own model in `src/input_thermo.f90`,
with the coefficients from `src/modules.f90`, because post-processing that
uses a different conductivity than the run used is not measuring that run.
As in the solver there are two states:

- **supercritical water and CO₂** — interpolated from the NIST tables, read
  from the case folder when it has one (the solver opens them by bare
  filename from its working directory, so that copy is the authoritative
  one) and otherwise from `Reference_Data/thermal_properties/`;
- **liquid metals** — sodium, lead, bismuth, LBE, lithium, FLiBe and
  PbLi-17, from the polynomial correlations in T.

Verified against the solver: every coefficient is compared with
`modules.f90` by parsing it, and supercritical water against the
`4_check/check_ftplist_dim.dat` property list the solver itself writes.

Everything is SI, with no exceptions: K, kg/m³, Pa·s, W/(m·K), J/(kg·K),
J/kg, 1/K and Pa. The NIST files state pressure in MPa and the solver keeps
it that way, so it is converted once on read and nothing downstream has to
remember which. Enthalpy inverts too — `temperature_from_enthalpy` is a port
of the solver's `ftp_refresh_thermal_properties_from_H`, which is how a bulk
temperature is recovered from the bulk enthalpy in the data.

Enthalpy coefficients are derived from the heat-capacity ones rather than
transcribed, as the solver now does, so that `dH/dT = Cp` holds exactly —
checked numerically for every fluid.

The temperature range is the melting-to-boiling phase range intersected with
the validity range of each correlation that has one, and the toolkit records
which property set each end. A fit does not hold over the whole liquid range
merely because the material is liquid there: PbLi-17 is 521–625 K, both ends
set by its viscosity correlation, against a phase range of 508–1943 K.

Where the solver stops the run — a temperature outside that range — the
toolkit returns NaN instead, so one bad cell leaves a gap rather than
costing the whole figure.

## Scripts:

**gui.py**: This launches a user interface for turbulence statistics, slice visualisation, monitoring points, 3D visualisation and mesh analysis, run 'python gui.py'. Choose a case once in the Case bar at the top and every tab is set up from it — each needs the case in a different form and works that out for itself, including which data types and timesteps exist and everything the run recorded in its `input_chapsim.ini` (see above). A Help tab covers each tab and the output formats. The Mesh Analysis tab is interactive: load an input_chapsim.ini (or start from the built-in template), adjust cell counts, stretching and flow parameters with sliders, and the resolution report, headline metrics and spacing plot update live. The adjusted settings can be written back out as an input file. This will likely not work on HPCs, use interactive script input instead (run each script individually).

**quick_turb_stats.py**: Single case post-processing script, outputs a figure for velocity/TKE/Temperature and a figure for Reynolds stresses. Ideal for use on HPCs with only numpy, matplotlib and tqdm dependencies. Recommended to be run on a serial/ data analysis or interactive node as bandwidth is typically throttled on login nodes. Interactive input.

**slice.py**: 2D visualisation of any output parameter with matplotlib plotting options. Also recommended to be run on a serial/ data analysis or interactive node as bandwidth is typically throttled on login nodes. Interactive input.

**turb_stats.py**: Main post-processing script to provide velocity, temperature, Reynolds stress profiles, Reynolds stress budget terms and heat transfer statistics such as turbulent Prantl number and Nusselt number from CHAPSim2 ASCII profile tables or xdmf output. Cases for comparison, which statistics to produce and the plotting options are specified in config.py. The run's own parameters are not: they are read from each case's `input_chapsim.ini` (see below). Plots are saved to `output_dir` when one is set, otherwise to turb_stats_plots/ beside the toolkit.

**monitor_points.py**: Plotting for bulk and point monitors, includes functionality to crop diverged data. Columns are read from each file's own header, so a monitor file that gains or loses a column still plots the right quantity under the right label. Pass a case folder or its 3_monitor directory, or run the script inside one. Interactive Input.

**mesh_analysis.py**: Pre-processing mesh resolution analysis. Reads a case's input_chapsim.ini, rebuilds the wall-normal grid exactly as the solver does, and assesses it against DNS resolution requirements (dy+, dx+, dz+, grid stretching, MHD boundary layer, recommended minimum mesh and time step). Python port of the estimate_spacial_resolution/estimate_temporal_resolution routines in CHAPSim2's apx_prerun_mod, so a mesh can be checked before a job is submitted. Run 'python mesh_analysis.py path/to/input_chapsim.ini', or with no argument to be prompted. Optionally saves a mesh distribution plot.

**thermal_BC_calc.py**: Convert a Grashof number to a wall temperature difference or heat flux, and tabulate any fluid's properties in the format CHAPSim2 reads — the same eight SI columns as a `NIST_*.DAT`, so the file it writes can be handed back to the solver as a table-based fluid.

## Tests:

```bash
python run_tests.py                 # everything
python run_tests.py test_xdmf       # one module
python run_tests.py -k cylindrical  # tests matching a name
```

Uses pytest when it is installed and falls back to a built-in runner when it is
not, so the suite also runs on a cluster where installing packages is awkward.
GitHub Actions runs the same commands on every push and pull request, on the
oldest and newest Python the package claims to support.

Most tests build small synthetic cases on disk in the formats CHAPSim2 writes,
so nothing binary lives in the repository and the expected values are analytic.
`tests/test_solver_cases.py` additionally reads CHAPSim2's own regression output
— every case, every averaging tier — which is what catches the solver's formats
changing. Those tests skip themselves when the solver is not checked out
alongside; set `CHAPSIM2_TESTS` to point at its `tests/` directory.

`tests/test_gui.py` builds a real widget tree and drives it: applying a case to
every tab, the values each tab takes from the case's `input_chapsim.ini`, the
timestep scans, the config.py round trip and the Help topics. Plotting is left
alone. These need `ttkbootstrap` and a display, and skip without either — run
them under `xvfb-run -a python run_tests.py test_gui` on a headless machine,
which is what CI does.

## Authors and provenance:

The toolkit was written by **Alex Old** (University of Sheffield), who remains
its principal author — most of the code here is his. It is currently maintained
by **Wei Wang** (UKRI-STFC); this repository is a fork of
[AlexOld1/CHAPSim2_python_toolkit](https://github.com/AlexOld1/CHAPSim2_python_toolkit),
and the intention is to maintain it under [CCP-NTH](https://ccpnth.ac.uk/)
alongside the [CHAPSim2 solver](https://github.com/CHAPSim/CHAPSim2).

Please cite it using `CITATION.cff` (GitHub's "Cite this repository" button).

Licensed under the MIT Licence; see `LICENSE`. Note that the CHAPSim2 solver
itself is BSD-3-Clause and copyright UKRI-STFC — the two are separate works
under separate licences.

The October 2026 work — compatibility with current CHAPSim2 output, the test
suite, CI and the GUI changes — was carried out with AI assistance; the
individual commits record this in their `Co-Authored-By` trailers.

## Acknowledgments:

CHAPSim2 and this toolkit are developed under the project of
[CCP-NTH](https://ccpnth.ac.uk/), with computational support from CoSeC, the
Computational Science Centre for Research Communities.

## Reference Data:

Isothermal channel (MKM180), square duct (KTH) reference data is provided as well as isothermal and heated MHD reference data (NK). All reference data is openly accessible from published sources. Copyright for reference datasets remains with the original authors/publishers. See individual data files for citations.