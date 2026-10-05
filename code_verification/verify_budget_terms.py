#!/usr/bin/env python3
"""
Verify TKE budget extraction functions from operations.py for the Ruu (uu11)
component using Vreman & Kuerten Chan180_FD2 reference data.

Code assumes homogeneity in x and z directions.
"""

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent dir so we can import operations
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import operations as op

# ── paths & constants ──────────────────────────────────────────────────
DATA   = Path(__file__).parent / "Chan180_FD2_all"
Re_tau = 180
Re     = Re_tau                   # Re = 1/nu used by operations.py

# ── load data ──────────────────────────────────────────────────────────
basic_u = np.loadtxt(DATA / "Chan180_FD2_basic_u.txt", comments="%")
basic_v = np.loadtxt(DATA / "Chan180_FD2_basic_v.txt", comments="%")
basic_w = np.loadtxt(DATA / "Chan180_FD2_basic_w.txt", comments="%")
basic_p = np.loadtxt(DATA / "Chan180_FD2_basic_p.txt", comments="%")
grad_yc = np.loadtxt(DATA / "Chan180_FD2_grad_yc.txt", comments="%")
grad_ys = np.loadtxt(DATA / "Chan180_FD2_grad_ys.txt", comments="%")
budget  = np.loadtxt(DATA / "Chan180_FD2_budget_Ruu.txt", comments="%")

# ── grids ──────────────────────────────────────────────────────────────
yp_c = basic_u[:, 0]             # y+ cell-centres (128 pts)
yp_s = grad_ys[:, 0]             # y+ staggered    (129 pts)
y_c  = yp_c / Re_tau             # physical y
N    = len(yp_c)
zeros = np.zeros(N)

# interpolate staggered → cell-centre
def s2c(f_s, yp_src=None):
    if yp_src is None:
        yp_src = yp_s
    return np.interp(yp_c, yp_src, f_s)

# ═══════════════════════════════════════════════════════════════════════
# EXTRACT RAW QUANTITIES
# ═══════════════════════════════════════════════════════════════════════
# Mean velocities (on yc)
yp_v = basic_v[:, 0]  # y+ for basic_v (staggered, 128 pts w/o wall)
U = basic_u[:, 1];  V = s2c(basic_v[:, 1], yp_v);  W = basic_w[:, 1]

# Reynolds stresses <u'_i u'_j>
uu = basic_u[:, 2]**2;  vv = s2c(basic_v[:, 2]**2, yp_v);  ww = basic_w[:, 2]**2
uv = s2c(basic_v[:, 5], yp_v);  uw = basic_u[:, 6];  vw = s2c(basic_v[:, 6], yp_v)

# Mean velocity gradient: only d<u>/dy nonzero in channel flow
dudy = s2c(grad_ys[:, 1])

# Dissipation-tensor diagonal for (0,0): sum_k <(du'/dx_k)^2>
diss_11 = grad_yc[:, 3]**2 + s2c(grad_ys[:, 4])**2 + grad_yc[:, 5]**2

# Triple correlation <u'u'v'> (for turbulent diffusion)
uuv = basic_u[:, 5]

# Pressure-velocity <p'u'_i>
pu = basic_p[:, 3];  pv = basic_p[:, 4];  pw = basic_p[:, 5]

# Pressure-strain correlations <p' du'_i/dx_j> (only diagonals available)
pdudx_11 = basic_p[:, 6]   # <p' du'/dx>
pdudx_22 = basic_p[:, 7]   # <p' dv'/dy>
pdudx_33 = basic_p[:, 8]   # <p' dw'/dz>

# ═══════════════════════════════════════════════════════════════════════
# BUILD tke_comp_dict  (same format as op.compute_TKE_components output)
# ═══════════════════════════════════════════════════════════════════════

# Reynolds stress tensor (3,3,N)
reynolds_stress_tensor = np.array([
    [uu, uv, uw],
    [uv, vv, vw],
    [uw, vw, ww],
])

# Mean velocity gradient tensor (3,3,N) — only dU/dy nonzero
mean_velocity_grad_tensor = np.zeros((3, 3, N))
mean_velocity_grad_tensor[0, 1] = dudy

# Dissipation tensor (3,3,N) — only (0,0) needed for uu11
dissipation_tensor = np.zeros((3, 3, N))
dissipation_tensor[0, 0] = diss_11

# Turbulent convection tensors: d<u'_i u'_j u'_k>/dx_k per direction
# Only y-direction (x2) nonzero in channel flow
turb_conv_tensor_x1 = np.zeros((3, 3, N))
turb_conv_tensor_x2 = np.zeros((3, 3, N))
turb_conv_tensor_x2[0, 0] = np.gradient(uuv, y_c)   # d<u'u'v'>/dy
turb_conv_tensor_x3 = np.zeros((3, 3, N))

# Laplacian of Reynolds stresses: d²<u'_i u'_j>/dy² (only y matters)
lap_re_stress_x2 = np.zeros((3, 3, N))
lap_re_stress_x2[0, 0] = op.second_derivative(uu, y_c, axis=0)

# Pressure-velocity fluctuation gradient: d<p'u'_i>/dx_j (only y nonzero)
press_vel_grad = np.zeros((3, 3, N))
press_vel_grad[0, 1] = np.gradient(pu, y_c)

# Pressure-strain correlation tensor <p' du'_i/dx_j> (3,3,N)
# Chan180 data provides diagonal terms directly as fluctuation correlations
pressure_strain_tensor = np.zeros((3, 3, N))
pressure_strain_tensor[0, 0] = pdudx_11
pressure_strain_tensor[1, 1] = pdudx_22
pressure_strain_tensor[2, 2] = pdudx_33

# Assemble the dictionary
tke_comp_dict = {
    'U1': U, 'U2': V, 'U3': W,
    'pr': basic_p[:, 1],
    'TKE': 0.5 * (uu + vv + ww),
    'mean_velocity_grad_tensor':        mean_velocity_grad_tensor,
    'fluc_velocity_grad_tensor':        np.zeros((3, 3, N)),
    'reynolds_stress_tensor':           reynolds_stress_tensor,
    'dissipation_tensor':               dissipation_tensor,
    'turb_conv_tensor_x1':              turb_conv_tensor_x1,
    'turb_conv_tensor_x2':              turb_conv_tensor_x2,
    'turb_conv_tensor_x3':              turb_conv_tensor_x3,
    'lap_re_stress_tensor_x1':          np.zeros((3, 3, N)),
    'lap_re_stress_tensor_x2':          lap_re_stress_x2,
    'lap_re_stress_tensor_x3':          np.zeros((3, 3, N)),
    'pr_prime':                         zeros,
    'press_velocity_fluc_grad_tensor':  press_vel_grad,
    'pressure_strain_tensor':           pressure_strain_tensor,
    'mean_conv_tensor_x1':              np.zeros((3, 3, N)),
    'mean_conv_tensor_x2':              np.zeros((3, 3, N)),
    'mean_conv_tensor_x3':              np.zeros((3, 3, N)),
}

# ═══════════════════════════════════════════════════════════════════════
# CALL operations.py BUDGET FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════
production   = next(iter(op.compute_production(tke_comp_dict, 'uu11').values()))
dissipation  = next(iter(op.compute_dissipation(Re, tke_comp_dict, 'uu11').values()))
turb_diff    = next(iter(op.compute_turbulent_convection(tke_comp_dict, 'uu11').values()))
visc_diff    = next(iter(op.compute_viscous_diffusion(Re, tke_comp_dict, 'uu11').values()))
press_trans  = next(iter(op.compute_pressure_transport(tke_comp_dict, 'uu11').values()))
press_strain = next(iter(op.compute_pressure_strain(tke_comp_dict, 'uu11').values()))

# ═══════════════════════════════════════════════════════════════════════
# REFERENCE VALUES
# ═══════════════════════════════════════════════════════════════════════
ref_prod = budget[:, 1];  ref_pstrain = budget[:, 2];  ref_diss = budget[:, 3]
ref_turb = budget[:, 4];  ref_pdiff   = budget[:, 5];  ref_visc = budget[:, 6]

# ═══════════════════════════════════════════════════════════════════════
# COMPARE (skip boundary points where FD stencils are unreliable)
# ═══════════════════════════════════════════════════════════════════════
s = slice(2, -2)

def report(name, computed, reference):
    err  = np.abs(computed[s] - reference[s])
    rmax = np.max(np.abs(reference[s]))
    print(f"  {name:25s}  max|err| = {err.max():.4e}   "
          f"max|rel| = {(err.max() / rmax * 100):.2f}%")

print("Budget-term verification for Ruu (uu11) using operations.py:")
print("-" * 65)
report("Production",          production,   ref_prod)
report("Dissipation",         dissipation,  ref_diss)
report("Turbulent diffusion", turb_diff,    ref_turb)
report("Viscous diffusion",   visc_diff,    ref_visc)
report("Pressure transport",  press_trans,  ref_pdiff)
report("Pressure strain",     press_strain, ref_pstrain)

# ═══════════════════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
terms = [
    ("Production",          production,   ref_prod),
    ("Dissipation",         dissipation,  ref_diss),
    ("Turbulent diffusion", turb_diff,    ref_turb),
    ("Viscous diffusion",   visc_diff,    ref_visc),
    ("Pressure strain",     press_strain, ref_pstrain),
    ("Pressure transport",  press_trans,  ref_pdiff),
]
for ax, (name, comp, ref) in zip(axes.flat, terms):
    ax.plot(yp_c, ref,  "k-",  label="Reference (budget file)")
    ax.plot(yp_c, comp, "r--", label="op.compute_* (operations.py)")
    ax.set_title(name);  ax.set_ylabel("Budget term")
    ax.legend(fontsize=8);  ax.set_xlim(0, 180)
for ax in axes[-1]:
    ax.set_xlabel("y⁺")
fig.suptitle("Ruu Budget Verification — operations.py vs Chan180 FD2", fontsize=14)
fig.tight_layout()
plt.savefig(Path(__file__).parent / "budget_Ruu_verification.png", dpi=150)
# Only pop a window when there is one to pop: under a non-interactive
# backend (CI, a cluster, a headless run) show() just warns.
if matplotlib.get_backend().lower() not in ('agg', 'pdf', 'ps', 'svg', 'template'):
    plt.show()
print("\nPlot saved to budget_Ruu_verification.png")
