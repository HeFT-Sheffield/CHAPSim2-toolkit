"""Text shown by the toolkit's Help tab.

Kept apart from gui.py so the wording can be edited without going near the
widget code. Each entry is (topic title, body); the Help tab lists the titles
and shows one body at a time.

Keep the bodies plain text at <= 78 columns: they are rendered in a fixed
pitch font, unwrapped, so the reader can rely on the alignment.
"""

__all__ = ['TOPICS', 'VERSION']

VERSION = '0.1.0'


_ABOUT = """\
CHAPSim2 Toolkit
================

A post-processing and pre-processing front end for the CHAPSim2 DNS solver,
built on NumPy, Matplotlib and PyVista.

  Version   {version}
  Licence   MIT
  Source    https://github.com/AlexOld1/CHAPSim2_python_toolkit

What it reads
-------------
The output written by current CHAPSim2: XDMF descriptors with binary field
data, the ASCII statistics tables, and the monitor logs. Both Cartesian
(channel, duct) and cylindrical (pipe, annulus) cases are supported, with
and without heat transfer and MHD.

Older runs that kept every visualisation file flat in 2_visu/ still load,
but current output is what the toolkit is built around.

The five working tabs
---------------------
  Mesh Analysis           check a mesh against DNS resolution limits
                          before submitting the job
  Monitoring Points       probe and bulk histories over a run
  Slice Visualisation     2-D contour maps of any field
  3D Visualisation        slices, isosurfaces and volume rendering
  Turbulence Statistics   profiles, Reynolds stresses, budgets,
                          spectra, two-point correlations

Everything here is also available as standalone scripts for machines with
no display - see README.md. The scripts and the GUI share the same reader
and the same plotting code, so they produce the same numbers.
""".format(version=VERSION)


_GETTING_STARTED = """\
How to use
==========

1. Choose the case, once
------------------------
Use the Case bar at the top of the window. Point it at a CHAPSim2 case
folder - the directory holding 1_data/, 2_visu/, 3_monitor/. Any
subdirectory of the case will do; it is normalised to the case root.

The line underneath summarises what was found:

    pipe_scp_inout_qw | timesteps: 0, 20, 40, 60 | input_chapsim.ini

Read it before going further. If it says "no visualisation output", or the
timestep you want is missing, the path is wrong or the run has not written
that data yet - better to find out here than after a long load.

Choosing a case sets up all the tabs at once: each one needs the case in a
slightly different form and works that out for itself. Every tab still has
its own path field, so one can be pointed at a different case if you want
to compare.

2. Pick a tab and work
----------------------
Each tab has its controls on the left and its output on the right, with a
strip of headline numbers above the figure. The tab-by-tab topics in this
list cover the details.

3. Watch the console
--------------------
The narrow strip on the right edge is the console; the dot appears when
there is new output. Click the arrow to open it. Everything the tools
print goes there - what was loaded, what was skipped and why, and the full
traceback if something fails. It is the first place to look when a result
is not what you expected.

Where figures go
----------------
Slice and 3D save on request, through their own Save buttons.

Turbulence Statistics writes every figure it produces. By default they go
to turb_stats_plots/ beside the toolkit; set an Output directory in the
Output section to send them elsewhere. It does not write into your case
folders.
"""


_LAYOUT = """\
Case folder layout
==================

A CHAPSim2 case directory:

    <case>/
      input_chapsim.ini     the solver input - read by Mesh Analysis
      1_data/               restart and checkpoint binaries, spectra
      2_visu/
        xdmf/               .xdmf descriptors - the index into everything
        data/               field binaries (.bin) and ASCII tables
        mesh/               grid coordinates
      3_monitor/            probe and history logs
      4_check/              mesh and property tables written by the solver

Point the Case bar at <case>. Any of the subdirectories also work.

What the data types mean
------------------------
  inst       one instantaneous snapshot
  t_avg      averaged in time, still 3-D
  tsp_avg    averaged in time and over the periodic directions
  2d_slice   a pre-written plane, far cheaper to read than a full field

A case does not carry all of them. Statistics only start once the solver
reaches stat_istart, thermo is often written less often than flow, and a
run with no averaging has no t_avg at all. The Slice and 3D tabs report
what each group holds when they scan, and start you on a type that exists.

Time-and-space averages come in two shapes, depending on the case:

  one periodic direction   a 2-D plane, tagged with the averaged-out
                           direction (zi1 for a spanwise-periodic channel)
  two periodic directions  a 1-D ASCII profile table in 2_visu/data/

Both are read automatically; you do not need to know which you have.

Pipes and annuli
----------------
These are solved in (x, r, theta), not (x, y, z). The toolkit recovers
those axes and labels them accordingly: a profile axis reads r, and a
cross-section is drawn unrolled as (r, theta) rather than as a disc. The
wall-normal statistics work exactly as they do for a channel.
"""


_MESH = """\
Mesh Analysis
=============

Checks a mesh against DNS resolution requirements before you submit the
job. It rebuilds the wall-normal grid exactly as the solver does, so the
spacings reported are the ones the solver will use.

Using it
--------
Choosing a case loads its input_chapsim.ini automatically. Otherwise use
Browse and Load.

Then change anything - cell counts, clustering, stretch factor, Reynolds
number, time step - and the report, the headline numbers and both plots
update as you drag. Generate... writes the adjusted settings back out as
an input file the solver will accept.

Reading the headline strip
--------------------------
  Re_tau         friction Reynolds number from an empirical correlation
  dy+ wall       first cell in wall units; DNS wants <= 1
  dy+ centre     coarsest wall-normal spacing
  dx+, dz+       streamwise and spanwise spacing; <= 10 and <= 5
  max growth     largest cell-to-cell ratio; keep under about 1.2
  diff. number   diffusion number for the chosen time step
  total cells    the size of the job you are about to submit

Values are coloured against those limits. The report below spells out
each one and says what to change.

Notes
-----
The case fixes some extents: a channel is y in [-1, 1], a pipe is r in
[0, 1] with Lz = 2*pi. Those boxes are greyed out because the solver
overrides them. For an annulus, "y bottom" is the inner radius, so it
stays editable and must lie between 0 and 1.

The predicted Re_tau comes from a correlation for a developed flow. A
short or still-developing run will measure something lower; that is the
run, not the mesh.
"""


_MONITOR = """\
Monitoring Points
=================

Plots what the solver recorded as it ran: probe points and the bulk and
mass-conservation histories. This is how you check a run is healthy and
has settled before spending time on statistics.

Using it
--------
Choosing a case points this tab at it; 3_monitor is found underneath.
Set how many probe points to read, then Run.

  Sample factor         plot every nth row - a long run holds 10^5+ rows
                        per trace and drawing them all is slow
  Running avg. window   overlay a centred running mean; useful for
                        spotting a drift under the noise
  Auto y-lim            clip the axis when a trace genuinely diverges,
                        so one blow-up does not flatten the rest

Figures
-------
  Pt N              u, v, w, pressure, pressure correction, temperature
                    at one probe
  Bulk Quantities   mass conservation, kinetic energy, pressure, bulk
                    velocity, bulk mass flux, bulk enthalpy and
                    temperature
  Change History    mass residuals, Poisson diagnostics, total mass and
                    its drift, kinetic energy change rate

Columns are read from each file's own header rather than by position, so
a monitor file that gains or loses a column still plots the right
quantity. Panels whose columns a run did not write are dropped - an
isothermal case simply gets a shorter figure.

What to look for
----------------
Mass conservation should sit at round-off. Bulk velocity should be steady
under constant mass flux. Kinetic energy should plateau before you trust
any statistics gathered over that period.
"""


_SLICE = """\
Slice Visualisation
===================

2-D contour maps of any field, either cut from a 3-D snapshot or read
from a plane the solver wrote.

Steps
-----
1. Choose the case. The tab scans and reports what each data type holds,
   then selects a type and timestep that exist.
2. Check Data type, Physics and Timestep.
3. Click "Load variables". The list fills.
4. Select a variable in the list. Nothing plots until you do.
5. Set the Plane and Slice index - the coordinate for that index is shown
   beside it - and click Plot.

Reading a pre-written plane is much cheaper than loading a full 3-D
field. Set Data type to 2d_slice and put a label such as yi24 in the
Slice label box; the available labels are listed when the case is
scanned. They are read straight out of the timestep's slice bundle.

Options
-------
  Fluctuation    u' = u_inst - u_t_avg, needs a t_avg file for the same
                 timestep; the path is filled in if one exists
  Vorticity      one component, computed from qx_ccc, qy_ccc, qz_ccc
  x crop         restrict the streamwise range, for xy and xz planes
  Colour scale   auto, symmetric about zero, or a range you give.
                 Symmetric is the honest choice for a signed quantity:
                 it keeps zero at the middle of the colour map

The headline strip reports the grid, the coordinate system, which plane
was cut, where, and the min, max and mean of what is drawn.

Variable names
--------------
  qx_ccc, qy_ccc, qz_ccc   instantaneous velocity at cell centres
  gx_ccc, ...              mass flux, for variable-density cases
  pr, phi                  pressure and pressure correction
  u1, u2, u3, uu11, ...    averaged statistics, in tsp_avg and t_avg data
"""


_VISU3D = """\
3D Visualisation
================

Interactive 3-D views through PyVista: slice planes, isosurfaces and
volume rendering. Needs the pyvista package; the other tabs do not.

Steps
-----
1. Choose the case, then Load variables and select one.
2. Pick a Mode and click Render.
3. Drag to orbit, right-drag to pan, scroll to zoom. "Screenshot" writes
   the current view to a file.

  Stride   read every nth cell. Start at 2 or 4 on a large grid: a full
           DNS field can be hundreds of millions of cells and the
           decimation happens at read time, not after loading it all.

Statistics
----------
  Fluctuation    u' against the matching t_avg field
  Q-criterion    vortex identification; isosurface mode is the useful one
  Vorticity      one component

Volume rendering refuses to start above a cell-count limit - raise the
stride and try again. It is also the slowest mode by a wide margin;
isosurfaces are usually the better way to look at structure.
"""


_STATS = """\
Turbulence Statistics
=====================

The main post-processing tab: mean profiles, Reynolds stresses, budget
terms, heat transfer statistics, spectra and two-point correlations.

Steps
-----
1. Choose the case. The case folder names, the latest timestep and
   everything the run itself recorded are filled in - see "From the
   case file" below.
2. Check the values that came in, and set the few the input file does
   not record: the bulk velocity and the wall heat flux.
3. Tick the statistics you want.
4. Run. Figures appear in the selector above the plot.

From the case file
------------------
A case's input_chapsim.ini is the one authoritative record of how it was
run, so these are read from it rather than typed in again:

  Reynolds number, reference temperature and reference length
  Geometry (from icase), and whether heat transfer and MHD were on
  Working fluid, gravity direction, magnetic field direction
  Stuart number - taken from NStuart, or derived as Ha^2/Re from NHartmn
  Which directions are periodic, which sets the safe averaging defaults

Anything you change by hand is kept. The point of reading the case is
that a Reynolds number transcribed wrongly rescales every u_tau
normalised profile with nothing on the plot to show for it.

The bulk velocity and the wall heat flux are outcomes of the run rather
than inputs to it, so they are not in the input file and still have to be
given here. A case with no input_chapsim.ini leaves every box as it is.

Loading a case also checks that the output is the output this input file
describes, by rebuilding the mesh from the cell counts and the stretching
and comparing it with the grid the solver wrote. If they disagree the
console says so - the usual cause is an input file edited after the run.

Fluid properties
----------------
These follow the solver's own model, so the numbers here are the numbers
the run used:

  supercritical water, CO2   interpolated from the NIST table, taken from
                             the case folder when it has one
  liquid metals              sodium, lead, bismuth, LBE, lithium, FLiBe
                             and PbLi-17, from correlations in T

Outside a fluid's valid range the properties come back as NaN and plot as
a gap, rather than as a confident extrapolation.

Input
-----
  Input format   xdmf for binary output, text for the ASCII profile
                 tables
  Data type      tsp_avg for time-and-space averages, t_avg for 3-D
                 time averages that still need averaging here
  Geometry       labels the wall-normal axis and hides controls that do
                 not apply - a pipe has one wall and an axis, so there
                 is no second side to choose

Averaging
---------
Average over a direction only where the flow is periodic. Averaging the
streamwise direction of a developing flow mixes the inlet with the
outlet. For a case with an inlet and an outlet, leave x unaveraged and
use the profile and slice coordinate boxes to pick stations.

The headline strip
------------------
  Re_bulk    what you entered
  Re_tau     measured from the wall gradient of the loaded profile
  u_tau      friction velocity
  tau_w      wall shear stress

If Re_tau is far from what Mesh Analysis predicted, the run has probably
not developed yet.

Normalisation
-------------
The u_tau used for normalisation is measured from the profile, not from
a correlation, so it reflects the state of the run. For a pipe the wall
gradient is taken at the wall, not at the axis.

Output
------
Save figures writes to the Output directory, or to turb_stats_plots/
beside the toolkit if none is set.

Load config.py / Save config.py exchange settings with the standalone
turb_stats.py script, so a case set up here can be re-run without a
display, on a cluster.
"""


_TROUBLE = """\
When something does not work
============================

"Load variables" finds nothing
------------------------------
The data type does not exist for that timestep. The console names the
missing file and lists what the group does have, for example:

    No such file: domain1_t_avg_flow_60.xdmf
    (this case has no t_avg flow; available: inst: 0, 20, 40, 60; tsp_avg: 60)

Choosing the case again re-scans and selects something that exists.

Plot does nothing
-----------------
No variable is selected. Click one in the Variables list; loading them
does not select one.

A physics group is missing at one timestep
------------------------------------------
Normal. Flow, thermo and MHD are written at their own frequencies, and
statistics only start at stat_istart. Use a timestep the console lists
for that group.

Re_tau looks far too small
--------------------------
Usually the run has not developed. Compare against the Mesh Analysis
prediction, which assumes a developed flow, and check in Monitoring
Points that the kinetic energy has plateaued.

Nothing happens after Run
-------------------------
Open the console. Loading a large 3-D field takes time, and any error
appears there with its traceback rather than in a dialog.

A pipe cross-section is drawn as a rectangle
--------------------------------------------
Intended. The plane is (r, theta) - a radius against an angle - so it is
drawn unrolled and labelled r and theta. Forcing it into a disc would
require interpolating onto a Cartesian grid, which is not what the solver
computed.

Figures appear in turb_stats_plots/
-----------------------------------
That is the default when no Output directory is set. The toolkit does not
write into your case folders.
"""


#: (title, body) in the order the Help tab lists them.
TOPICS = [
    ('About',                 _ABOUT),
    ('How to use',            _GETTING_STARTED),
    ('Case folder layout',    _LAYOUT),
    ('Mesh Analysis',         _MESH),
    ('Monitoring Points',     _MONITOR),
    ('Slice Visualisation',   _SLICE),
    ('3D Visualisation',      _VISU3D),
    ('Turbulence Statistics', _STATS),
    ('Troubleshooting',       _TROUBLE),
]
