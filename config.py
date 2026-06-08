# Configuration file for turb_stats script ============================================================================================================

# Define input cases ==================================================================================================================================

folder_path = '' # format: folder_path/case/1_data/quantity_timestep.dat
input_format = 'visu' # 'text' (.dat) or 'visu' (.xdmf)
xdmf_data_type = 'tsp_avg' # 't_avg' (3D) or 'tsp_avg' (2D).
slice_label = '' # 2D slice label (e.g. 'yi8' for xz slice at y index 8), leave blank for full 3D data
cases = ['Tests'] # case names must match folder names exactly. Add multiple in a list.
timesteps = ['680000'] # Add multiple in a list
average_over_timesteps = False # calculates mean over multiple timestep files for each case
average_x_direction = False # Averaging valid for periodic directions, set to False for spatially developing flows
average_z_direction = True # Averaging valid for periodic directions, set to False for duct flows

forcing = 'CMF' # 'CMF' or 'CPG', constant mass flux or pressure gradient.
Re = [5000] # indexing matches 'cases' if different Re used for different cases. Use bulk reference value for CPG.

# Thermo/ Variable Properties
thermo_on = True # Below reference values are used for thermo statistics, not necessary for isothermal flows
ref_temp = [570] # Kelvin
ref_length = [0.05] # m
ref_bulk_velocity = [0.0900625] # m/s
wall_heat_flux = [0.0] # W/m^2, positive for heating, negative for cooling
working_fluid = 'lithium' # Only liquid metals available.
gravity_direction = [0, 0, 0] # [x, y, z]

# Magnetohydrodynamics
mhd_on = True
mag_field_direction = [0, 0, 0] # [x, y, z]
stuart_number = 0.0

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

# Plotting options ------------------------------------------------------------------------------------------------------------------------------------

half_channel_plot = False
linear_y_scale = True
log_y_scale = False
multi_plot = True
display_fig = False
save_fig = True
save_to_path = True
large_text_on = False # Increase axes, label, title and legend font sizes for readability
plot_name = '' # name for saved plot files, leave blank for default naming

# reference data options
ux_velocity_log_ref_on = True
mhd_NK_ref_on = False # MHD turbulent channel at Re_tau=150, Ha=(4,6), Noguchi & Kasagi 1994 (thtlabs.jp)
mkm180_ch_ref_on = False # Turbulent channel at Re_tau=180, Moser, Kim & Mansour 1999 (DOI: 10.1017/S002211209900708X)

#====================================================================================================================================================
