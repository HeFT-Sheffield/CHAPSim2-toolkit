# CHAPSim2-toolkit
A python post-processing and toolkit program based on NumPy and Matplotlib for DNS solver CHAPSim2.

## Install:

Dependencies are given in requirements.txt.
pip: Navigate to base directory and run 'pip install .'
conda: Navigate to base directory and run 'conda env create -f environment.yml' to create a conda environment for the program then 'conda activate chapsim2-toolkit' to use the environment.

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

**gui.py**: This launches a user interface for turbulence statistics, slice visualisation, monitoring points and mesh analysis, run 'python gui.py'. The Mesh Analysis tab is interactive: load an input_chapsim.ini (or start from the built-in template), adjust cell counts, stretching and flow parameters with sliders, and the resolution report, headline metrics and spacing plot update live. The adjusted settings can be written back out as an input file. This will likely not work on HPCs, use interactive script input instead (run each script individually).

**quick_turb_stats.py**: Single case post-processing script, outputs a figure for velocity/TKE/Temperature and a figure for Reynolds stresses. Ideal for use on HPCs with only numpy, matplotlib and tqdm dependencies. Recommended to be run on a serial/ data analysis or interactive node as bandwidth is typically throttled on login nodes. Interactive input.

**slice.py**: 2D visualisation of any output parameter with matplotlib plotting options. Also recommended to be run on a serial/ data analysis or interactive node as bandwidth is typically throttled on login nodes. Interactive input.

**turb_stats.py**: Main post-processing script to provide velocity, temperature, Reynolds stress profiles, Reynolds stress budget terms and heat transfer statistics such as turbulent Prantl number and Nusselt number from CHAPSim2 ASCII profile tables or xdmf output. Input parameters, cases for comparison, plotting options etc. are specified on config.py file for interactive input. Plots saved in turb_stats_plots/ and to file path.

**monitor_points.py**: Plotting for bulk and point monitors, includes functionality to crop diverged data. Columns are read from each file's own header, so a monitor file that gains or loses a column still plots the right quantity under the right label. Pass a case folder or its 3_monitor directory, or run the script inside one. Interactive Input.

**mesh_analysis.py**: Pre-processing mesh resolution analysis. Reads a case's input_chapsim.ini, rebuilds the wall-normal grid exactly as the solver does, and assesses it against DNS resolution requirements (dy+, dx+, dz+, grid stretching, MHD boundary layer, recommended minimum mesh and time step). Python port of the estimate_spacial_resolution/estimate_temporal_resolution routines in CHAPSim2's apx_prerun_mod, so a mesh can be checked before a job is submitted. Run 'python mesh_analysis.py path/to/input_chapsim.ini', or with no argument to be prompted. Optionally saves a mesh distribution plot.

**thermal_BC_calc.py**: Property functions for liquid metals in CHAPSim2, functionality to output NIST format data file, convert a given Grashof number to constant wall temperature difference or heat flux (channel flow), calculate Prandtl number. Interactive input.

## Reference Data:

Isothermal channel (MKM180), square duct (KTH) reference data is provided as well as isothermal and heated MHD reference data (NK). All reference data is openly accessible from published sources. Copyright for reference datasets remains with the original authors/publishers. See individual data files for citations.