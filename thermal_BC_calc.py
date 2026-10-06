#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====================================================================================================================================================

"""
Thermophysical properties of mediums in CHAPSim2 are provided in classes.
Functions for generating property tables or temp. difference from Grahsof number are provided.
"""

from utils import (
    get_fluid_properties,
)

def generate_property_table(T_min, T_max, fluid='lithium', pressure=1.0e5,
                            n_points=1024, save_dat=False, filename=None):
    """Tabulate a fluid's properties in the format CHAPSim2 reads.

    The columns and their units are the ones
    buildup_property_relations_from_table expects in a NIST_*.DAT, so the
    file this writes can be handed straight back to the solver as a
    table-based fluid - which is the only use a property table has here.

    It used to emit molar quantities for lithium alone, in a mix of kJ/mol,
    J/mol/K and micro Pa s, in a layout the solver cannot read. Everything
    is SI now and every fluid works.

    Args:
        T_min, T_max: temperature range in K. Clipped to the range the
            fluid's properties are valid over.
        fluid: any name get_fluid_properties takes.
        pressure: in Pa. Written to the file in MPa, which is the
            convention of the solver's own tables. The correlations are at
            atmospheric pressure; the supercritical fluids carry their own
            and this argument is ignored for them.
        n_points: rows. The solver samples a correlation at 1024 points.
        save_dat: write the table to disk as well as returning it.
        filename: where to write; defaults to NIST_<FLUID>.DAT.

    Returns:
        A pandas DataFrame with one row per temperature.
    """
    props = get_fluid_properties(fluid)

    lo, hi = max(T_min, props.T_min), min(T_max, props.T_max)
    if lo >= hi:
        raise ValueError(
            f'{fluid} has no valid properties between {T_min} and {T_max} K; '
            f'its range is {props.T_min:g} to {props.T_max:g} K')
    if lo > T_min or hi < T_max:
        print(f'Note: clipped to {lo:g}-{hi:g} K, the range {fluid} is '
              f'valid over.')

    T = np.linspace(lo, hi, n_points)
    if getattr(props, 'pressure', None) is not None:
        pressure = props.pressure

    # Column 1 is MPa in the solver's own NIST files - its header says so,
    # and it reads the value into a scratch variable and discards it. The
    # file keeps that convention so the solver can read this back; the
    # Python side is SI throughout and converts on read.
    data = {
        'P(Mpa)': np.full(n_points, float(pressure) / 1e6),
        'H(J/KG)': props.enthalpy(T),
        'T(K)': T,
        'D(KG/M3)': props.density_mass(T),
        'M(PA-S)': props.viscosity(T),
        'K(W/M-K)': props.thermal_conductivity(T),
        'CP(J/KG-K)': props.heat_capacity_p(T),
        'Beta(1/K)': props.coeff_vol_exp(T),
    }
    df = pd.DataFrame(data)

    if save_dat:
        filename = filename or f'NIST_{fluid.upper()}.DAT'
        with open(filename, 'w') as fh:
            # The solver skips one header line and then reads eight reals.
            fh.write('#' + '\t'.join(df.columns) + '\n')
            for row in df.itertuples(index=False):
                fh.write('\t'.join(f'{v:.11E}' for v in row) + '\n')
        print(f'Property table saved to {filename}')

    return df


def Grahsof_to_temp_diff(grahsof, beta, L, mu, rho):
    """
    Convert Grahsof number to temperature difference in K.
    
    Grahsof = g * beta * Delta_T * L^3 / nu^2
    Delta_T = Grahsof * nu^2 / (g * beta * L^3)
    
    Assumptions:
    - g = 9.81 m/s²
    
    Parameters:
    -----------
    Grahsof : float
        Grahsof number
    Returns:
    """
    g = 9.81
    nu = mu / rho
    delta_T = (grahsof * nu**2) / (g * beta * L**3)
    
    return delta_T

def get_heat_flux(delta_T, k, L_ref):
    """
    Calculate heat flux from temperature difference.

    Parameters:
    -----------
    delta_T : float
        Temperature difference in K
    k : float
        Thermal conductivity in W/(m·K)
    L_ref : float
        Reference length in m

    Returns:
    --------
    float
        Heat flux in W/m²
    """
    heat_flux = k * delta_T / L_ref
    return heat_flux

def calc_fourier_number(k, rho, cp, L, timestep, nondim_time=True):
    """
    Calculate Fourier number.

    For dimensional time:    Fo = alpha * t / L^2
    For non-dimensional time (U_ref=1): Fo = alpha * t* / L
        where t_actual = t* * L (since t* = t / L with U_ref = 1 m/s)

    Parameters:
    -----------
    k : float
        Thermal conductivity in W/(m·K)
    rho : float
        Density in kg/m³
    cp : float
        Specific heat capacity at constant pressure in J/(kg·K)
    L : float
        Characteristic length in m
    timestep : float
        Time (dimensional in s, or non-dimensional if nondim_time=True)
    nondim_time : bool
        If True, timestep is non-dimensional (t* = t/L with U_ref=1 m/s)
        If False, timestep is dimensional in seconds

    Returns:
    --------
    float
        Fourier number (dimensionless)
    """
    alpha = k / (rho * cp)
    if nondim_time:
        # t_actual = t* * L, so Fo = alpha * t* * L / L^2 = alpha * t* / L
        Fo = alpha * timestep / L
    else:
        # Dimensional time: Fo = alpha * t / L^2
        Fo = alpha * timestep / (L**2)
    return Fo

def get_prandtl(temp, fluid):
    """
    Calculate Prandtl number.

    Pr = (c_p * mu) / k

    Parameters:
    -----------
    temp : float
        Temperature in K
    fluid : object
        Fluid properties object from get_fluid_properties().

    Returns:
    --------
    float
        Prandtl number (dimensionless)
    """
    # Every fluid now reports viscosity in Pa s. It did not used to: the
    # lithium class alone returned micro Pa s, and this function corrected
    # for it by type, so anything else that called viscosity() on lithium
    # was out by a factor of a million.
    return fluid.prandtl(temp)

def generate_normalized_property_table(fluid, T_ref, delta_T, n_points=20):
    """
    Map the change in thermophysical properties across a temperature range,
    each non-dimensionalised by its own value at the reference temperature.

    Temperature ranges from T_ref to T_ref + delta_T, so every property
    column starts at 1.0 (its value at T_ref) — useful for seeing which
    properties vary most strongly (and whether a constant-property
    assumption is reasonable) over the temperature difference applied at
    a thermal boundary condition.

    Parameters:
    -----------
    fluid : object
        Fluid properties object (e.g. from get_fluid_properties()).
    T_ref : float
        Reference temperature in K — the T=0 point for the normalisation.
    delta_T : float
        Temperature difference in K to map over.
    n_points : int
        Number of temperature points.

    Returns:
    --------
    pandas.DataFrame
        Temperature and non-dimensional temperature columns, plus each
        property normalised by its value at T_ref.
    """
    T_array = np.linspace(T_ref, T_ref + delta_T, n_points)

    properties = {
        'Density': fluid.density_mass,
        'Viscosity': fluid.viscosity,
        'Thermal Conductivity': fluid.thermal_conductivity,
        'Specific Heat (Cp)': fluid.heat_capacity_p,
        'Volumetric Expansion': fluid.coeff_vol_exp,
        'Prandtl Number': lambda T: get_prandtl(T, fluid),
    }
    if hasattr(fluid, 'electrical_conductivity'):
        properties['Electrical Conductivity'] = fluid.electrical_conductivity

    data = {
        'Temperature (K)': T_array,
        'Non-dim Temperature': T_array / T_ref,
    }
    for name, func in properties.items():
        ref_value = func(T_ref)
        data[f'{name} (normalised)'] = [func(T) / ref_value for T in T_array]

    return pd.DataFrame(data)

def plot_normalized_properties(df, fluid_name='Fluid', save_fig=False, filename='normalized_properties.png'):
    """
    Plot non-dimensionalised property variation across a mapped temperature range.

    Parameters:
    -----------
    df : pandas.DataFrame
        Output of generate_normalized_property_table().
    fluid_name : str
        Fluid name for the plot title.
    save_fig : bool
        Whether to save the figure to disk.
    filename : str
        Output filename if save_fig=True.

    Returns:
    --------
    matplotlib.figure.Figure
    """
    prop_cols = [c for c in df.columns if c.endswith('(normalised)')]

    fig, ax = plt.subplots(figsize=(8, 6))
    for col in prop_cols:
        ax.plot(df['Non-dim Temperature'], df[col], marker='o', markersize=3,
                label=col.replace(' (normalised)', ''))

    ax.axhline(1.0, color='grey', linestyle='--', linewidth=1)
    ax.set_xlabel(r'Non-dimensional temperature, $T / T_{ref}$')
    ax.set_ylabel(r'Property / Property at $T_{ref}$')
    ax.set_title(f'Normalised thermophysical property variation — {fluid_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_fig:
        fig.savefig(filename, dpi=150)
        print(f"Plot saved to {filename}")

    return fig

#: Keyed on the fluid's own name, so a property object identifies itself
#: without the caller having to know which class it is.
FLUID_NAMES = {
    'lithium': "Liquid Lithium (Li)",
    'sodium': "Liquid Sodium (Na)",
    'lead': "Liquid Lead (Pb)",
    'bismuth': "Liquid Bismuth (Bi)",
    'lbe': "Liquid Lead-Bismuth Eutectic (LBE)",
    'flibe': "Liquid FLiBe (2LiF-BeF2)",
    'pbli': "Liquid Lead-Lithium Eutectic (Pb-83Li-17)",
    'scp_water': "Supercritical Water (23.5 MPa)",
    'scp_co2': "Supercritical CO2 (8 MPa)",
}


def interactive_calculation():
    """
    Interactive input for calculating temperature difference and heat flux
    from Grashof number, with material property summary.
    """
    print("\n" + "=" * 70)
    print("  Thermophysical Properties Calculator")
    print("=" * 70)

    # Get medium selection
    print("\nAvailable media:")
    print("  Li    - Liquid Lithium")
    print("  Na    - Liquid Sodium")
    print("  Pb    - Liquid Lead")
    print("  Bi    - Liquid Bismuth")
    print("  LBE   - Lead-Bismuth Eutectic")
    print("  FLiBe - Molten Salt (2LiF-BeF2)")
    print("  PbLi  - Lead-Lithium (Pb-17Li)")

    while True:
        medium_input = input("\nSelect medium (Li/Na/Pb/Bi/LBE/FLiBe/PbLi): ").strip()
        try:
            fluid = get_fluid_properties(medium_input)
            break
        except ValueError as e:
            print(f"Error: {e}")

    # Get Grashof number
    while True:
        try:
            grashof_input = input("Enter Grashof number (e.g., 5e7): ").strip()
            grashof = float(grashof_input)
            if grashof <= 0:
                print("Grashof number must be positive.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # Get reference temperature
    while True:
        try:
            T_ref = float(input(f"Enter reference temperature in K (valid range: {fluid.T_melt:.1f} - {fluid.T_boil:.1f}): ").strip())
            if T_ref < fluid.T_melt:
                print(f"Warning: Temperature below melting point ({fluid.T_melt} K)")
            if T_ref > fluid.T_boil:
                print(f"Warning: Temperature above boiling point ({fluid.T_boil} K)")
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # Get reference length
    while True:
        try:
            L_ref = float(input("Enter reference length in m: ").strip())
            if L_ref <= 0:
                print("Reference length must be positive.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # Get timestep for Fourier number calculation
    while True:
        try:
            timestep = float(input("Enter timestep (non-dimensional, t* = t/L with U_ref=1): ").strip())
            if timestep <= 0:
                print("Timestep must be positive.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # Calculate properties
    mu = fluid.viscosity(T_ref)
    rho = fluid.density_mass(T_ref)
    k = fluid.thermal_conductivity(T_ref)
    cp = fluid.heat_capacity_p(T_ref)
    beta = fluid.coeff_vol_exp(T_ref)
    Pr = get_prandtl(T_ref, fluid)

    # Calculate temperature difference and heat flux
    delta_T = Grahsof_to_temp_diff(grashof, beta, L_ref, mu, rho)
    heat_flux = k * delta_T / L_ref
    T_hot = T_ref + delta_T

    # Calculate Fourier number (using non-dimensional time from CHAPSim2)
    Fo = calc_fourier_number(k, rho, cp, L_ref, timestep, nondim_time=True)
    alpha = k / (rho * cp)  # thermal diffusivity for display

    # Print results
    medium_name = FLUID_NAMES.get(fluid.name, "Unknown Fluid")

    print("\n" + "=" * 70)
    print(f"  Results for {medium_name}")
    print("=" * 70)

    print("\n--- Input Parameters ---")
    print(f"  Grashof number:            {grashof:.3e}")
    print(f"  Reference temperature:     {T_ref:.2f} K")
    print(f"  Reference length:          {L_ref:.4f} m")
    print(f"  Timestep (non-dim):        {timestep:.6e}")

    print("\n--- Material Properties at T_ref ---")
    print(f"  Density (rho):             {rho:.2f} kg/m³")
    print(f"  Dynamic viscosity (mu):    {mu:.6e} Pa·s")
    print(f"  Thermal conductivity (k):  {k:.4f} W/(m·K)")
    print(f"  Specific heat (Cp):        {cp:.2f} J/(kg·K)")
    print(f"  Thermal diffusivity (α):   {alpha:.6e} m²/s")
    print(f"  Vol. expansion (beta):     {beta:.6e} 1/K")
    print(f"  Prandtl number (Pr):       {Pr:.6f}")

    print("\n--- Calculated Results ---")
    print(f"  Temperature difference:    {delta_T:.6f} K")
    print(f"  Hot wall temperature:      {T_hot:.2f} K")
    print(f"  Heat flux (q''):           {heat_flux:.4f} W/m²")
    #print(f"  Fourier number (Fo):       {Fo:.6e}")

    print("\n" + "=" * 70 + "\n")

    map_input = input("Map normalised property variation over this temperature "
                       "difference? (y/n): ").strip().lower()
    if map_input == 'y':
        medium_name = FLUID_NAMES.get(fluid.name, "Unknown Fluid")
        prop_df = generate_normalized_property_table(fluid, T_ref, delta_T)
        print(f"\n{prop_df.to_string(index=False)}\n")
        plot_normalized_properties(prop_df, fluid_name=medium_name)
        plt.show()

    return {
        'grashof': grashof,
        'T_ref': T_ref,
        'L_ref': L_ref,
        'timestep': timestep,
        'delta_T': delta_T,
        'heat_flux': heat_flux,
        'Fo': Fo,
        'alpha': alpha,
        'Pr': Pr,
        'rho': rho,
        'mu': mu,
        'k': k,
        'cp': cp,
        'beta': beta
    }

if __name__ == '__main__':
    interactive_calculation()
