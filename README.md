# CHAPSim2-toolkit

[![tests](https://github.com/weiwangstfc/CHAPSim2-toolkit/actions/workflows/tests.yml/badge.svg)](https://github.com/weiwangstfc/CHAPSim2-toolkit/actions/workflows/tests.yml)

A python post-processing and toolkit program based on NumPy and Matplotlib for DNS solver CHAPSim2.

## Install:

The toolkit is run from a checkout — the modules sit at the top level, so an
install places only metadata and does not put them on the import path. What you
need is the dependencies:

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

## Scripts:

**gui.py**: This launches a user interface for turbulence statistics, slice visualisation, monitoring points, 3D visualisation and mesh analysis, run 'python gui.py'. Choose a case once in the Case bar at the top and every tab is set up from it — each needs the case in a different form and works that out for itself, including which data types and timesteps exist, the geometry, and which directions are periodic and so safe to average. A Help tab covers each tab and the output formats. The Mesh Analysis tab is interactive: load an input_chapsim.ini (or start from the built-in template), adjust cell counts, stretching and flow parameters with sliders, and the resolution report, headline metrics and spacing plot update live. The adjusted settings can be written back out as an input file. This will likely not work on HPCs, use interactive script input instead (run each script individually).

**quick_turb_stats.py**: Single case post-processing script, outputs a figure for velocity/TKE/Temperature and a figure for Reynolds stresses. Ideal for use on HPCs with only numpy, matplotlib and tqdm dependencies. Recommended to be run on a serial/ data analysis or interactive node as bandwidth is typically throttled on login nodes. Interactive input.

**slice.py**: 2D visualisation of any output parameter with matplotlib plotting options. Also recommended to be run on a serial/ data analysis or interactive node as bandwidth is typically throttled on login nodes. Interactive input.

**turb_stats.py**: Main post-processing script to provide velocity, temperature, Reynolds stress profiles, Reynolds stress budget terms and heat transfer statistics such as turbulent Prantl number and Nusselt number from CHAPSim2 ASCII profile tables or xdmf output. Input parameters, cases for comparison, plotting options etc. are specified on config.py file for interactive input. Plots are saved to `output_dir` when one is set, otherwise to turb_stats_plots/ beside the toolkit. Averaging directions are checked against the case's own input_chapsim.ini, and a warning is printed if a direction with an inlet and an outlet is being averaged.

**monitor_points.py**: Plotting for bulk and point monitors, includes functionality to crop diverged data. Columns are read from each file's own header, so a monitor file that gains or loses a column still plots the right quantity under the right label. Pass a case folder or its 3_monitor directory, or run the script inside one. Interactive Input.

**mesh_analysis.py**: Pre-processing mesh resolution analysis. Reads a case's input_chapsim.ini, rebuilds the wall-normal grid exactly as the solver does, and assesses it against DNS resolution requirements (dy+, dx+, dz+, grid stretching, MHD boundary layer, recommended minimum mesh and time step). Python port of the estimate_spacial_resolution/estimate_temporal_resolution routines in CHAPSim2's apx_prerun_mod, so a mesh can be checked before a job is submitted. Run 'python mesh_analysis.py path/to/input_chapsim.ini', or with no argument to be prompted. Optionally saves a mesh distribution plot.

**thermal_BC_calc.py**: Property functions for liquid metals in CHAPSim2, functionality to output NIST format data file, convert a given Grashof number to constant wall temperature difference or heat flux (channel flow), calculate Prandtl number. Interactive input.

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