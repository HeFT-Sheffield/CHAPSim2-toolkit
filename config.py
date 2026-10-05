# Configuration file for turb_stats script ============================================================================================================
#
# Settings left as None are read from each case's own input_chapsim.ini when
# the case is loaded: the Reynolds number, the reference temperature and
# length, the geometry, the working fluid, whether heat transfer and MHD were
# on, the gravity and magnetic field directions and the Stuart number. That
# file is the solver's own record of how the run was set up, so it is the one
# place these belong. Set a value here only to override the case, and the
# toolkit will say so if the two disagree.
#
# The bulk velocity and the wall heat flux are outcomes of a run rather than
# inputs to it, so they are not in the input file and are given below.

# Define input cases ==================================================================================================================================

folder_path = '' # parent of the case folders; each case holds 1_data/, 2_visu/{xdmf,data,mesh}/, 3_monitor/
input_format = 'visu' # 'visu' (.xdmf + .bin) or 'text' (tsp_avg ASCII profile tables)
xdmf_data_type = 'tsp_avg' # 'tsp_avg' (space-averaged: a plane, or a 1D profile table) or 't_avg' (3D)
slice_label = '' # 2D slice label (e.g. 'yi8' for xz slice at y index 8), leave blank for full 3D data
cases = ['Tests'] # case names must match folder names exactly. Add multiple in a list.
timesteps = ['680000'] # Add multiple in a list
average_over_timesteps = False # calculates mean over multiple timestep files for each case
# Average only over periodic directions: averaging a direction with an inlet and an
# outlet folds the two ends of the domain together. The case's input_chapsim.ini is
# checked against these and a warning is printed if they disagree.
average_x_direction = False # set False for spatially developing (inlet/outlet) flows
average_z_direction = True # set False for duct flows, and for tsp_avg input (already averaged)

forcing = 'CMF' # 'CMF' or 'CPG', constant mass flux or pressure gradient.
# None means 'read it from the case's input_chapsim.ini', which is where the solver
# recorded it. Set a value only to override, and the toolkit will say if it disagrees.
Re = None # e.g. [5000]; one per case if they differ. Use the bulk value for CPG.

# Thermo/ Variable Properties
thermo_on = None # from the case's ithermo. The reference values below are only used when it is on.
ref_temp = None # [K]; from the case's ref_t0
ref_length = None # [m]; from the case's ref_l0
ref_bulk_velocity = [0.0900625] # m/s
wall_heat_flux = [0.0] # W/m^2, positive for heating, negative for cooling
working_fluid = None # from the case's ifluid. Only liquid metals have property data.
gravity_direction = None # [x, y, z]; from the case's igravity

# Magnetohydrodynamics
mhd_on = None # from the case's imhd
mag_field_direction = None # [x, y, z]; from the case's B_static
stuart_number = None # from the case's NStuart, or derived as Ha^2/Re from NHartmn

# Output ==============================================================================================================================================

# Profiles
ux_velocity_on = True
uy_velocity_on = False
uz_velocity_on = False
temp_on = False
tke_on = False
coeff_friction_on = False

profile_direction = 'y' # 'y' (wall-normal), 'x' (streamwise), or 'both'
slice_coords = '' # y-profiles: x coords for slices, e.g. '0.5,1.0' (blank = streamwise avg)
x_crop = '' # x-range crop for 2D visu data, e.g. '0.0,1.0' (blank = full x-range)
x_profile_y_coords = '' # x-profiles: y coords for slices, e.g. '0.0,0.5' (blank = channel centreline)
surface_plot_on = False # Plot 2D (y,x) surface contour maps of each statistic (requires 2D data)

# Reynolds stresses
u_prime_sq_on = False
u_prime_v_prime_on = False
w_prime_sq_on = False
v_prime_sq_on = False

# Spanwise two-point velocity correlation (needs instantaneous 3D visu data)
two_point_corr_on = False
two_point_corr_components = 'uu' # pairs of u/v/w, e.g. 'uu' or 'uu,vv,ww,uv'
two_point_corr_y_coords = '' # y coords for the R(dz) line plot, e.g. '-0.9,-0.5' (blank = no line plot)
two_point_corr_x_coords = '' # x stations to correlate at (blank = mid-domain, or all x if average_x_direction)
two_point_corr_max_sep = 0 # max separation in cells (0 = half the spanwise domain)
two_point_corr_mean_mode = 't_avg' # 't_avg' (u' = u_inst - u_t_avg) or 'snapshot' (remove the snapshot's own z-mean)
# The correlation is folded about the centreline only when half_channel_side = 'average' (symmetric channels only;
# a warning is printed if the lower and upper walls differ)

# Reynolds Stress Budget terms
re_stress_budget_on = False
re_stress_component = 'uu11' # 'total' or 'uu11', 'uu12' etc. for individual components

# thermo statistics
heat_transf_coeff_on = False # x profile only
Nusselt_number_on = False # x profile only
turb_prandtl_on = False

# Processing options ----------------------------------------------------------------------------------------------------------------------------------

# normalisation
norm_by_u_tau_sq = True
norm_ux_by_u_tau = True
norm_y_to_y_plus = False
norm_temp_by_ref_temp = False

geometry = None # 'channel', 'pipe', 'annulus' or 'duct'; from the case's icase

# Plotting options ------------------------------------------------------------------------------------------------------------------------------------

half_channel_plot = False
linear_y_scale = True
log_y_scale = False
display_fig = False
save_fig = True
save_to_path = True # legacy: also write into folder_path. Prefer output_dir below.
output_dir = '' # where figures are written; blank = turb_stats_plots/ beside the toolkit
large_text_on = False # Increase axes, label, title and legend font sizes for readability
plot_name = '' # name for saved plot files, leave blank for default naming

# reference data options
ux_velocity_log_ref_on = True
mhd_NK_ref_on = False # MHD turbulent channel at Re_tau=150, Ha=(4,6), Noguchi & Kasagi 1994 (thtlabs.jp)
mkm180_ch_ref_on = False # Turbulent channel at Re_tau=180, Moser, Kim & Mansour 1999 (DOI: 10.1017/S002211209900708X)

#====================================================================================================================================================
