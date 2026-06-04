import os
import numpy as np
import xml.etree.ElementTree as ET
import mmap
from tqdm import tqdm

# =====================================================================================================================================================
# XDMF READING CONFIGURATION
# =====================================================================================================================================================

# Set to True to load all variables, False to load only required ones (faster)
LOAD_ALL_VARS = False

# Variables required for computation (base names without prefixes)
_BASE_REQUIRED_VARS = [
    # Velocities
    'u1', 'u2', 'u3',
    # Reynolds stresses
    'uu11', 'uu12', 'uu13', 'uu22', 'uu23', 'uu33',
    # Triple correlations
    'uuu111', 'uuu112', 'uuu113', 'uuu122', 'uuu123', 'uuu133',
    'uuu222', 'uuu223', 'uuu233', 'uuu333',
    # Velocity gradients
    'dudx11', 'dudx12', 'dudx13', 'dudx21', 'dudx22', 'dudx23', 'dudx31', 'dudx32', 'dudx33',
    # Dissipation terms
    'dudu11', 'dudu12', 'dudu13', 'dudu22', 'dudu23', 'dudu33',
    # Pressure terms
    'pr', 'pru1', 'pru2', 'pru3',
    'prdu11', 'prdu12', 'prdu13', 'prdu21', 'prdu22', 'prdu23', 'prdu31', 'prdu32', 'prdu33',
    # Density / volume fraction (variable properties)
    'f',
    'fu1', 'fu2', 'fu3',
    'fuu11', 'fuu12', 'fuu13', 'fuu22', 'fuu23', 'fuu33',
    'fuuu111', 'fuuu112', 'fuuu113', 'fuuu122', 'fuuu123', 'fuuu133',
    'fuuu222', 'fuuu223', 'fuuu233', 'fuuu333',
    'fuh1', 'fuh2', 'fuh3',
    'fuuh11', 'fuuh12', 'fuuh13', 'fuuh22', 'fuuh23', 'fuuh33',
    'fh',
    # Temperature
    'T', 'TT', 'Tu1', 'Tu2', 'Tu3',
    # MHD
    'e',
    'j1', 'j2', 'j3',
    'ej1', 'ej2', 'ej3',
    'jj11', 'jj12', 'jj13', 'jj22', 'jj23', 'jj33',
    'eu1', 'eu2', 'eu3',
    'ju11', 'ju12', 'ju13', 'ju21', 'ju22', 'ju23', 'ju31', 'ju32', 'ju33',
]

# Build full set including prefixed versions (tsp_avg_, t_avg_)
REQUIRED_VARS = set(_BASE_REQUIRED_VARS)
for prefix in ['tsp_avg_', 't_avg_']:
    REQUIRED_VARS.update(f'{prefix}{var}' for var in _BASE_REQUIRED_VARS)

# =====================================================================================================================================================
# =====================================================================================================================================================
# THERMAL PROPERTIES CLASSES
# =====================================================================================================================================================

class LiquidLithiumProperties:
    """
    Thermophysical properties of liquid lithium.
    Valid range: Tm (453.65 K) to approximately 1500 K
    
    Note: Pressure effects are neglected for most properties as liquids
    are assumed incompressible. Properties are primarily temperature-dependent.
    No functions are given for properties irrelevant to CHAPSim2.

    References:
    - R.W. Ohse (Ed.) Handbook of Thermodynamic and Transport Properties of Alkali Metals, 
    Intern. Union of Pure and Applied Chemistry Chemical Data Series No. 30. Oxford: Blackwell Scientific Publ., 1985, pp. 987. 
    (All properties except Cp)
    - C.B. Alcock, M.W. Chase, V.P. Itkin, J. Phys. Chem. Ref. Data 23 (1994) 385. (Cp)
    """
    
    def __init__(self):
        self.T_melt = 453.65  # K, melting point
        self.T_boil = 1615.0  # K, boiling point at 1 atm
        self.M = 6.9410  # g/mol, molar mass of Li
        
    def phase(self, T, P):
        """Determine phase based on temperature"""
        if T < self.T_melt:
            return "Solid"
        elif T < self.T_boil:
            return "Liquid"
        else:
            return "Vapor"
    
    def density_mass(self, T):
        """Mass density in kg/m³"""
        return 278.5 - 0.04657 * T + 274.6 * (1 - T / 3500)**0.467
    
    def density_molar(self, T):
        """Molar density in mol/L"""
        rho_mass = self.density_mass(T)  # kg/m³
        rho_mol = rho_mass / self.M  # mol/m³
        return rho_mol / 1000  # mol/L
    
    def molar_volume(self, T):
        """Molar volume in L/mol"""
        return 1.0 / self.density_molar(T)
    
    def internal_energy(self, T, T_ref):
        """
        Molar internal energy in kJ/mol relative to reference temperature.
        For liquids: dU ≈ Cv*dT
        """
        Cv = self.heat_capacity_v(T)  # J/(mol·K)
        U = Cv * (T - T_ref) / 1000  # kJ/mol
        return U
    
    def enthalpy(self, T, T_ref=None):
        """
        Molar enthalpy in kJ/mol.

        If T_ref is None, returns absolute enthalpy from the fitted Cp(T) integral.
        If T_ref is provided, returns delta enthalpy relative to T_ref.
        """
        H_abs = (4754 * T - (0.925 * T**2) / 2 + (0.000291 * T**3) / 3) / 1000
        if T_ref is None:
            return H_abs

        H_ref = (4754 * T_ref - (0.925 * T_ref**2) / 2 + (0.000291 * T_ref**3) / 3) / 1000
        return H_abs - H_ref
    
    def temperature_from_enthalpy(self, H, ref_temp=None):
        """
        Invert enthalpy relation to find absolute temperature from enthalpy.
        
        Handles both scalar and array inputs of enthalpy values.
        """
        
        h_ref = None
        if ref_temp is not None:
            h_ref = self.enthalpy(ref_temp)          # kJ/kg
            ref_cp = self.heat_capacity_p(ref_temp) / 1000  # kJ/(kg·K), matching enthalpy units
        if np.ndim(H) == 0:
            # Scalar case
            h_val = float(H)
            return self._temperature_from_enthalpy_scalar(h_val, h_ref, ref_temp, ref_cp)
        else:
            # Array case
            H_arr = np.asarray(H).ravel()
            result = np.zeros_like(H_arr, dtype=float)
            for i, h in enumerate(H_arr):
                result[i] = self._temperature_from_enthalpy_scalar(float(h), h_ref, ref_temp, ref_cp)
            return result.reshape(np.asarray(H).shape)
    
    def _temperature_from_enthalpy_scalar(self, H, h_ref=None, ref_temp=None, ref_cp=None):
        """
        Invert enthalpy relation for a single scalar enthalpy value.
        """
        # Coefficients of a*T^3 + b*T^2 + c*T + d = 0 from enthalpy(T) - H = 0
        a = 0.000291 / 3.0
        b = -0.925 / 2.0
        c = 4754.0
        H_dim = H * ref_temp * ref_cp + h_ref if h_ref is not None else H
        d = -1000 * H_dim

        roots = np.roots([a, b, c, d])
        real_roots = roots[np.abs(roots.imag) < 1e-8].real

        valid_roots = real_roots[(real_roots >= self.T_melt) & (real_roots <= self.T_boil)]
        if valid_roots.size == 0:
            raise ValueError(f"No physical temperature root found for enthalpy H={H}.")

        residuals = np.abs([self.enthalpy(T) - H_dim for T in valid_roots])
        return float(valid_roots[np.argmin(residuals)])
    
    def entropy(self, T, T_ref):
        """
        Molar entropy in J/(mol·K) relative to reference temperature.
        dS = Cp/T * dT for constant pressure
        """
        Cp = self.heat_capacity_p(T)  # J/(mol·K)
        if T > T_ref:
            S = Cp * np.log(T / T_ref)
        else:
            S = 0
        return S
    
    def heat_capacity_p(self, T):
        """Molar heat capacity at constant pressure Cp in J/(mol·K)"""
        Cp_mass = 4754 - 0.925 * T + 0.000291 * T**2  # J/(kg·K)
        return Cp_mass
    
    def heat_capacity_p_molar(self, T):
        """Molar heat capacity at constant pressure Cp in J/(mol·K)"""
        Cp_mass = 4754 - 0.925 * T + 0.000291 * T**2  # J/(kg·K)
        Cp_molar = Cp_mass * self.M / 1000  # J/(mol·K)
        return Cp_molar
    
    def heat_capacity_v(self, T):
        """
        Molar heat capacity at constant volume Cv in J/(mol·K)
        For liquids: Cv ≈ Cp (difference is small)
        """
        return self.heat_capacity_p(T) * 0.98  # Approximate
    
    def speed_of_sound(self, T):
        """
        Speed of sound in m/s
        Estimated from bulk modulus and density
        """
        # Bulk modulus for Li ~ 11-12 GPa
        K = 11.5e9  # Pa
        rho = self.density_mass(T)  # kg/m³
        c = np.sqrt(K / rho)
        return c
    
    def joule_thomson(self, T):
        """
        Joule-Thomson coefficient in K/MPa
        For liquids, typically very small and can be negative or positive
        Approximated as nearly zero
        """
        return 0.0  # Negligible for liquids
    
    def viscosity(self, T):
        """Dynamic viscosity in µPa·s"""
        mu_Pa_s = np.exp(-4.164 - 0.6374 * np.log(T) + (292.1 / T))  # Pa·s
        return mu_Pa_s * 1e6  # Convert to µPa·s
    
    def thermal_conductivity(self, T):
        """Thermal conductivity in W/(m·K)"""
        return 22.28 + 0.05 * T - 0.00001243 * T**2  # W/(m·K)
    
    def coeff_vol_exp(self, T):
        """Coefficient of volume expansion in 1/K"""
        return 1 / ( 5620 - T )

class LiquidPbLiProperties:
    """
    Thermophysical properties of liquid PbLi (Pb-17Li eutectic alloy).
    Valid range: Tm (508.0 K) to approximately 1943 K

    Note: Pressure effects are neglected for most properties as liquids
    are assumed incompressible. Properties are primarily temperature-dependent.
    No functions are given for properties irrelevant to CHAPSim2.

    References:
    - Correlations from CHAPSim2/src/modules.f90
    """

    def __init__(self):
        self.T_melt = 508.0   # K, melting point
        self.T_boil = 1943.0  # K, boiling point at 1 atm
        self.H_melt = 33.9e3  # J/kg, latent heat of melting

    def phase(self, T, P=None):
        """Determine phase based on temperature"""
        if T < self.T_melt:
            return "Solid"
        elif T < self.T_boil:
            return "Liquid"
        else:
            return "Vapor"

    def density_mass(self, T):
        """
        Mass density in kg/m³
        D = CoD(0) + CoD(1) * T
        CoD_PbLi = (10520.4, -1.1905)
        """
        return 10520.4 - 1.1905 * T

    def thermal_conductivity(self, T):
        """
        Thermal conductivity in W/(m·K)
        K = CoK(0) + CoK(1) * T + CoK(2) * T^2
        CoK_PbLi = (9.148, 1.963E-2, 0.0)
        """
        return 9.148 + 1.963e-2 * T

    def coeff_vol_exp(self, T):
        """
        Coefficient of volume expansion in 1/K
        B = 1 / (CoB - T)
        CoB_PbLi = 8836.8
        """
        return 1.0 / (8836.8 - T)

    def heat_capacity_p(self, T):
        """
        Specific heat capacity at constant pressure Cp in J/(kg·K)
        Cp = CoCp(-2) * T^(-2) + CoCp(-1) * T^(-1) + CoCp(0) + CoCp(1) * T + CoCp(2) * T^2
        CoCp_PbLi = (0.0, 0.0, 195.0, -9.116E-3, 0.0)
        """
        return 195.0 - 9.116e-3 * T

    def heat_capacity_v(self, T):
        """
        Specific heat capacity at constant volume Cv in J/(kg·K)
        For liquids: Cv ≈ Cp (difference is small)
        """
        return self.heat_capacity_p(T) * 0.98  # Approximate

    def enthalpy(self, T, T_ref):
        """
        Specific enthalpy in J/kg relative to reference temperature.
        H = HM0 + CoH(0) + CoH(1) * (T - TM0) + CoH(2) * (T^2 - TM0^2)
        CoH_PbLi = (0.0, 0.0, 195.0, -4.558E-3, 0.0)

        Integrated from Cp: H = int(Cp dT) = 195*T - 4.558E-3 * T^2 / 2
        """
        return 195.0 * (T - T_ref) - 4.558e-3 * (T**2 - T_ref**2) / 2

    def viscosity(self, T):
        """
        Dynamic viscosity in Pa·s
        M = CoM(0) + CoM(1) * T + CoM(2) * T^2 + CoM(3) * T^3
        CoM_PbLi = (0.0061091, -2.2574E-5, 3.766E-8, -2.2887E-11)
        """
        return 0.0061091 - 2.2574e-5 * T + 3.766e-8 * T**2 - 2.2887e-11 * T**3

    def viscosity_uPa_s(self, T):
        """Dynamic viscosity in µPa·s"""
        return self.viscosity(T) * 1e6

class LiquidSodiumProperties:
    """
    Thermophysical properties of liquid sodium (Na).
    Valid range: Tm (371.0 K) to approximately 1155 K

    References:
    - Correlations from CHAPSim2/src/modules.f90
    """

    def __init__(self):
        self.T_melt = 371.0   # K, melting point
        self.T_boil = 1155.0  # K, boiling point at 1 atm
        self.H_melt = 113.0e3 # J/kg, latent heat of melting

    def phase(self, T, P=None):
        """Determine phase based on temperature"""
        if T < self.T_melt:
            return "Solid"
        elif T < self.T_boil:
            return "Liquid"
        else:
            return "Vapor"

    def density_mass(self, T):
        """
        Mass density in kg/m³
        D = CoD(0) + CoD(1) * T
        CoD_Na = (1014.0, -0.235)
        """
        return 1014.0 - 0.235 * T

    def thermal_conductivity(self, T):
        """
        Thermal conductivity in W/(m·K)
        K = CoK(0) + CoK(1) * T + CoK(2) * T^2
        CoK_Na = (104.0, -0.047, 0.0)
        """
        return 104.0 - 0.047 * T

    def coeff_vol_exp(self, T):
        """
        Coefficient of volume expansion in 1/K
        B = 1 / (CoB - T)
        CoB_Na = 4316.0
        """
        return 1.0 / (4316.0 - T)

    def heat_capacity_p(self, T):
        """
        Specific heat capacity at constant pressure Cp in J/(kg·K)
        Cp = CoCp(-2) * T^(-2) + CoCp(-1) * T^(-1) + CoCp(0) + CoCp(1) * T + CoCp(2) * T^2
        CoCp_Na = (-3.001e6, 0.0, 1658.0, -0.8479, 4.454E-4)
        """
        return -3.001e6 * T**(-2) + 1658.0 - 0.8479 * T + 4.454e-4 * T**2

    def heat_capacity_v(self, T):
        """
        Specific heat capacity at constant volume Cv in J/(kg·K)
        For liquids: Cv ≈ Cp (difference is small)
        """
        return self.heat_capacity_p(T) * 0.98

    def enthalpy(self, T, T_ref):
        """
        Specific enthalpy in J/kg relative to reference temperature.
        Integrated from Cp
        """
        def H(t):
            return 3.001e6 / t + 1658.0 * t - 0.8479 * t**2 / 2 + 4.454e-4 * t**3 / 3
        return H(T) - H(T_ref)

    def viscosity(self, T):
        """
        Dynamic viscosity in Pa·s
        M = exp(CoM(-1) / T + CoM(0) + CoM(1) * ln(T))
        CoM_Na = (556.835, -6.4406, -0.3958)
        """
        return np.exp(556.835 / T - 6.4406 - 0.3958 * np.log(T))

    def viscosity_uPa_s(self, T):
        """Dynamic viscosity in µPa·s"""
        return self.viscosity(T) * 1e6

class LiquidLeadProperties:
    """
    Thermophysical properties of liquid lead (Pb).
    Valid range: Tm (600.6 K) to approximately 2021 K

    References:
    - Correlations from CHAPSim2/src/modules.f90
    """

    def __init__(self):
        self.T_melt = 600.6   # K, melting point
        self.T_boil = 2021.0  # K, boiling point at 1 atm
        self.H_melt = 23.07e3 # J/kg, latent heat of melting

    def phase(self, T, P=None):
        """Determine phase based on temperature"""
        if T < self.T_melt:
            return "Solid"
        elif T < self.T_boil:
            return "Liquid"
        else:
            return "Vapor"

    def density_mass(self, T):
        """
        Mass density in kg/m³
        D = CoD(0) + CoD(1) * T
        CoD_Pb = (11441.0, -1.2795)
        """
        return 11441.0 - 1.2795 * T

    def thermal_conductivity(self, T):
        """
        Thermal conductivity in W/(m·K)
        K = CoK(0) + CoK(1) * T + CoK(2) * T^2
        CoK_Pb = (9.2, 0.011, 0.0)
        """
        return 9.2 + 0.011 * T

    def coeff_vol_exp(self, T):
        """
        Coefficient of volume expansion in 1/K
        B = 1 / (CoB - T)
        CoB_Pb = 8942.0
        """
        return 1.0 / (8942.0 - T)

    def heat_capacity_p(self, T):
        """
        Specific heat capacity at constant pressure Cp in J/(kg·K)
        Cp = CoCp(-2) * T^(-2) + CoCp(-1) * T^(-1) + CoCp(0) + CoCp(1) * T + CoCp(2) * T^2
        CoCp_Pb = (-1.524e6, 0.0, 176.2, -4.923E-2, 1.544E-5)
        """
        return -1.524e6 * T**(-2) + 176.2 - 4.923e-2 * T + 1.544e-5 * T**2

    def heat_capacity_v(self, T):
        """
        Specific heat capacity at constant volume Cv in J/(kg·K)
        For liquids: Cv ≈ Cp (difference is small)
        """
        return self.heat_capacity_p(T) * 0.98

    def enthalpy(self, T, T_ref):
        """
        Specific enthalpy in J/kg relative to reference temperature.
        Integrated from Cp
        """
        def H(t):
            return 1.524e6 / t + 176.2 * t - 4.923e-2 * t**2 / 2 + 1.544e-5 * t**3 / 3
        return H(T) - H(T_ref)

    def viscosity(self, T):
        """
        Dynamic viscosity in Pa·s
        M = CoM(0) * exp(CoM(-1) / T)
        CoM_Pb = (1069.0, 4.55E-4, 0.0)
        """
        return 4.55e-4 * np.exp(1069.0 / T)

    def viscosity_uPa_s(self, T):
        """Dynamic viscosity in µPa·s"""
        return self.viscosity(T) * 1e6

class LiquidBismuthProperties:
    """
    Thermophysical properties of liquid bismuth (Bi).
    Valid range: Tm (544.6 K) to approximately 1831 K

    References:
    - Correlations from CHAPSim2/src/modules.f90
    """

    def __init__(self):
        self.T_melt = 544.6   # K, melting point
        self.T_boil = 1831.0  # K, boiling point at 1 atm
        self.H_melt = 53.3e3  # J/kg, latent heat of melting

    def phase(self, T, P=None):
        """Determine phase based on temperature"""
        if T < self.T_melt:
            return "Solid"
        elif T < self.T_boil:
            return "Liquid"
        else:
            return "Vapor"

    def density_mass(self, T):
        """
        Mass density in kg/m³
        D = CoD(0) + CoD(1) * T
        CoD_Bi = (10725.0, -1.22)
        """
        return 10725.0 - 1.22 * T

    def thermal_conductivity(self, T):
        """
        Thermal conductivity in W/(m·K)
        K = CoK(0) + CoK(1) * T + CoK(2) * T^2
        CoK_Bi = (7.34, 9.5E-3, 0.0)
        """
        return 7.34 + 9.5e-3 * T

    def coeff_vol_exp(self, T):
        """
        Coefficient of volume expansion in 1/K
        B = 1 / (CoB - T)
        CoB_BI = 8791.0
        """
        return 1.0 / (8791.0 - T)

    def heat_capacity_p(self, T):
        """
        Specific heat capacity at constant pressure Cp in J/(kg·K)
        Cp = CoCp(-2) * T^(-2) + CoCp(-1) * T^(-1) + CoCp(0) + CoCp(1) * T + CoCp(2) * T^2
        CoCp_Bi = (7.183e6, 0.0, 118.2, 5.934E-3, 0.0)
        """
        return 7.183e6 * T**(-2) + 118.2 + 5.934e-3 * T

    def heat_capacity_v(self, T):
        """
        Specific heat capacity at constant volume Cv in J/(kg·K)
        For liquids: Cv ≈ Cp (difference is small)
        """
        return self.heat_capacity_p(T) * 0.98

    def enthalpy(self, T, T_ref):
        """
        Specific enthalpy in J/kg relative to reference temperature.
        Integrated from Cp
        """
        def H(t):
            return -7.183e6 / t + 118.2 * t + 5.934e-3 * t**2 / 2
        return H(T) - H(T_ref)

    def viscosity(self, T):
        """
        Dynamic viscosity in Pa·s
        M = CoM(0) * exp(CoM(-1) / T)
        CoM_Bi = (780.0, 4.456E-4, 0.0)
        """
        return 4.456e-4 * np.exp(780.0 / T)

    def viscosity_uPa_s(self, T):
        """Dynamic viscosity in µPa·s"""
        return self.viscosity(T) * 1e6

class LiquidLBEProperties:
    """
    Thermophysical properties of liquid Lead-Bismuth Eutectic (LBE).
    Valid range: Tm (398.0 K) to approximately 1927 K

    References:
    - Correlations from CHAPSim2/src/modules.f90
    """

    def __init__(self):
        self.T_melt = 398.0   # K, melting point
        self.T_boil = 1927.0  # K, boiling point at 1 atm
        self.H_melt = 38.6e3  # J/kg, latent heat of melting

    def phase(self, T, P=None):
        """Determine phase based on temperature"""
        if T < self.T_melt:
            return "Solid"
        elif T < self.T_boil:
            return "Liquid"
        else:
            return "Vapor"

    def density_mass(self, T):
        """
        Mass density in kg/m³
        D = CoD(0) + CoD(1) * T
        CoD_LBE = (11065.0, 1.293)
        Note: Positive coefficient - check source data
        """
        return 11065.0 + 1.293 * T

    def thermal_conductivity(self, T):
        """
        Thermal conductivity in W/(m·K)
        K = CoK(0) + CoK(1) * T + CoK(2) * T^2
        CoK_LBE = (3.284, 1.617E-2, -2.305E-6)
        """
        return 3.284 + 1.617e-2 * T - 2.305e-6 * T**2

    def coeff_vol_exp(self, T):
        """
        Coefficient of volume expansion in 1/K
        B = 1 / (CoB - T)
        CoB_LBE = 8558.0
        """
        return 1.0 / (8558.0 - T)

    def heat_capacity_p(self, T):
        """
        Specific heat capacity at constant pressure Cp in J/(kg·K)
        Cp = CoCp(-2) * T^(-2) + CoCp(-1) * T^(-1) + CoCp(0) + CoCp(1) * T + CoCp(2) * T^2
        CoCp_LBE = (-4.56e5, 0.0, 164.8, -3.94E-2, 1.25E-5)
        """
        return -4.56e5 * T**(-2) + 164.8 - 3.94e-2 * T + 1.25e-5 * T**2

    def heat_capacity_v(self, T):
        """
        Specific heat capacity at constant volume Cv in J/(kg·K)
        For liquids: Cv ≈ Cp (difference is small)
        """
        return self.heat_capacity_p(T) * 0.98

    def enthalpy(self, T, T_ref):
        """
        Specific enthalpy in J/kg relative to reference temperature.
        Integrated from Cp
        """
        def H(t):
            return 4.56e5 / t + 164.8 * t - 3.94e-2 * t**2 / 2 + 1.25e-5 * t**3 / 3
        return H(T) - H(T_ref)

    def viscosity(self, T):
        """
        Dynamic viscosity in Pa·s
        M = CoM(0) * exp(CoM(-1) / T)
        CoM_LBE = (754.1, 4.94E-4, 0.0)
        """
        return 4.94e-4 * np.exp(754.1 / T)

    def viscosity_uPa_s(self, T):
        """Dynamic viscosity in µPa·s"""
        return self.viscosity(T) * 1e6

class LiquidFLiBeProperties:
    """
    Thermophysical properties of liquid FLiBe (2LiF-BeF2 molten salt).
    Valid range: Tm (732.1 K) to approximately 1703 K

    References:
    - Correlations from CHAPSim2/src/modules.f90
    """

    def __init__(self):
        self.T_melt = 732.1   # K, melting point
        self.T_boil = 1703.0  # K, boiling point at 1 atm
        self.H_melt = 17.47e5 # J/kg, integral(Cp(TM0))

    def phase(self, T, P=None):
        """Determine phase based on temperature"""
        if T < self.T_melt:
            return "Solid"
        elif T < self.T_boil:
            return "Liquid"
        else:
            return "Vapor"

    def density_mass(self, T):
        """
        Mass density in kg/m³
        D = CoD(0) + CoD(1) * T
        CoD_FLiBe = (2413.03, -0.4884)
        """
        return 2413.03 - 0.4884 * T

    def thermal_conductivity(self, T):
        """
        Thermal conductivity in W/(m·K)
        K = CoK(0) + CoK(1) * T + CoK(2) * T^2
        CoK_FLiBe = (1.1, 0.0, 0.0)
        """
        return 1.1

    def coeff_vol_exp(self, T):
        """
        Coefficient of volume expansion in 1/K
        B = 1 / (CoB - T)
        CoB_FLiBe = 4940.7
        """
        return 1.0 / (4940.7 - T)

    def heat_capacity_p(self, T):
        """
        Specific heat capacity at constant pressure Cp in J/(kg·K)
        Cp = CoCp(-2) * T^(-2) + CoCp(-1) * T^(-1) + CoCp(0) + CoCp(1) * T + CoCp(2) * T^2
        CoCp_FLiBe = (0.0, 0.0, 2386.0, 0.0, 0.0)
        """
        return 2386.0

    def heat_capacity_v(self, T):
        """
        Specific heat capacity at constant volume Cv in J/(kg·K)
        For liquids: Cv ≈ Cp (difference is small)
        """
        return self.heat_capacity_p(T) * 0.98

    def enthalpy(self, T, T_ref):
        """
        Specific enthalpy in J/kg relative to reference temperature.
        Integrated from Cp (constant Cp)
        """
        return 2386.0 * (T - T_ref)

    def viscosity(self, T):
        """
        Dynamic viscosity in Pa·s
        M = CoM(0) * exp(CoM(-1) / T)
        CoM_FLiBe = (4022.0, 7.803E-5, 0.0)
        """
        return 7.803e-5 * np.exp(4022.0 / T)

    def viscosity_uPa_s(self, T):
        """Dynamic viscosity in µPa·s"""
        return self.viscosity(T) * 1e6


def get_fluid_properties(medium):
    """Return a thermal properties object for the requested medium name."""
    medium_lower = medium.lower()
    if medium_lower in ['li', 'lithium']:
        return LiquidLithiumProperties()
    elif medium_lower in ['na', 'sodium']:
        return LiquidSodiumProperties()
    elif medium_lower in ['pb', 'lead']:
        return LiquidLeadProperties()
    elif medium_lower in ['bi', 'bismuth']:
        return LiquidBismuthProperties()
    elif medium_lower in ['lbe', 'pb-bi', 'pbbi']:
        return LiquidLBEProperties()
    elif medium_lower in ['flibe', 'fli-be', '2lif-bef2']:
        return LiquidFLiBeProperties()
    elif medium_lower in ['pbli', 'pb-li', 'pbli17', 'pb17li']:
        return LiquidPbLiProperties()
    else:
        raise ValueError(f"Unknown medium: {medium}. Available options: Li, Na, Pb, Bi, LBE, FLiBe, PbLi")


# XML PARSING HELPER
# =====================================================================================================================================================

def _parse_xdmf_xml(xdmf_path):
    """
    Parse an XDMF file, handling files that have been appended
    (multiple root elements).

    When a simulation appends to an existing XDMF file, the result is
    multiple consecutive <Xdmf>...</Xdmf> blocks, which is not valid XML.
    This function wraps them in a synthetic root and returns the *last*
    <Xdmf> element (the most recent write).

    Returns:
        ET.Element: The root element to iterate over, or None on failure.
    """
    try:
        return ET.parse(xdmf_path).getroot()
    except ET.ParseError:
        pass

    # Likely multiple root elements — read to string, wrap in synthetic root, take last entry
    import re
    try:
        with open(xdmf_path, 'r') as f:
            xml_content = f.read()
    except IOError as e:
        print(f"Error reading {xdmf_path}: {e}")
        return None

    cleaned = re.sub(r'<\?xml[^?]*\?>', '', xml_content)
    try:
        wrapper = ET.fromstring(f"<_wrapper>{cleaned}</_wrapper>")
        xdmf_elements = list(wrapper)
        if xdmf_elements:
            print(f"Note: {os.path.basename(xdmf_path)} contains {len(xdmf_elements)} "
                  f"appended entries — using the last one.")
            return xdmf_elements[-1]
    except ET.ParseError as e:
        print(f"Error parsing {xdmf_path} (even after handling appended entries): {e}")

    return None

# =====================================================================================================================================================
# TEXT DATA UTILITIES
# =====================================================================================================================================================

def case_path(folder_path, case):
    """Build a normalised case path from folder and case inputs."""
    base = os.path.expanduser(os.path.expandvars(str(folder_path).strip()))
    case_name = str(case).strip().strip('/\\')
    if not base:
        return case_name
    return os.path.normpath(os.path.join(base, case_name))

def data_filepath(folder_path, case, quantity, timestep):
    return os.path.join(
        case_path(folder_path, case),
        '1_data',
        f'domain1_tsp_avg_{quantity}_{timestep}.dat'
    )

def load_ts_avg_data(data_filepath):
    try:
        return np.loadtxt(data_filepath)
    except OSError:
        print(f'Error loading data for {data_filepath}')
        return None

def get_quantities(thermo_on):
    quantities = ['u1', 'u2', 'u3', 'uu11', 'uu12', 'uu22','uu33','pr']
    if thermo_on:
        quantities.extend(['T', 'Tu2'])
    return quantities

# =====================================================================================================================================================
# XDMF FILE PATH UTILITIES
# =====================================================================================================================================================

def visu_file_paths(folder_path, case, timestep):
    """Return XDMF file paths for instantaneous, time-averaged, and tsp_avg (zi1) data."""
    case_dir = case_path(folder_path, case)
    visu = os.path.join(case_dir, '2_visu')
    file_names = [
        os.path.join(visu, f'domain1_flow_{timestep}.xdmf'),
        os.path.join(visu, f'domain1_t_avg_flow_{timestep}.xdmf'),
        os.path.join(visu, f'domain1_tsp_avg_flow_zi1_{timestep}.xdmf'),
        os.path.join(visu, f'domain1_thermo_{timestep}.xdmf'),
        os.path.join(visu, f'domain1_t_avg_thermo_{timestep}.xdmf'),
        os.path.join(visu, f'domain1_tsp_avg_thermo_zi1_{timestep}.xdmf'),
        os.path.join(visu, f'domain1_mhd_{timestep}.xdmf'),
        os.path.join(visu, f'domain1_t_avg_mhd_{timestep}.xdmf'),
        os.path.join(visu, f'domain1_tsp_avg_mhd_zi1_{timestep}.xdmf'),
    ]
    return file_names


# =====================================================================================================================================================
# 2D SLICE FILE UTILITIES
# =====================================================================================================================================================

def parse_slice_label(label):
    """
    Parse a 2D slice label from a filename component.

    Examples:
        'yi8'  -> ('y', 8)   # xz slice at y index 8
        'xi5'  -> ('x', 5)   # yz slice at x index 5
        'zi3'  -> ('z', 3)   # xy slice at z index 3

    Returns:
        tuple: (direction, index) or None if not a valid slice label
    """
    import re
    match = re.match(r'^([xyz])i(\d+)$', label)
    if match:
        return match.group(1), int(match.group(2))
    return None


def find_available_slices(visu_folder, timestep=None):
    """
    Find available 2D slice labels in a visu folder.

    Args:
        visu_folder: Path to the 2_visu folder
        timestep: Optional timestep to filter for

    Returns:
        list of unique slice labels found, sorted (e.g., ['xi5', 'yi8', 'zi3'])
    """
    import re
    labels = set()
    try:
        for f in os.listdir(visu_folder):
            if not f.endswith('.xdmf'):
                continue
            match = re.search(r'_([xyz]i\d+)_(\d+)\.xdmf$', f)
            if match:
                label = match.group(1)
                ts = match.group(2)
                if timestep is None or ts == str(timestep):
                    labels.add(label)
    except OSError:
        pass
    return sorted(labels)


def visu_slice_file_paths(folder_path, case, timestep, slice_label):
    """
    Generate XDMF file paths for 2D slice data.

    Args:
        folder_path: Base folder path
        case: Case name
        timestep: Timestep string
        slice_label: Slice label (e.g., 'yi8')

    Returns:
        list of file paths
    """
    case_dir = case_path(folder_path, case)
    file_names = [
        os.path.join(case_dir, '2_visu', f'domain1_flow_{slice_label}_{timestep}.xdmf'),
        os.path.join(case_dir, '2_visu', f'domain1_t_avg_flow_{slice_label}_{timestep}.xdmf'),
        os.path.join(case_dir, '2_visu', f'domain1_tsp_avg_flow_{slice_label}_{timestep}.xdmf'),
        os.path.join(case_dir, '2_visu', f'domain1_thermo_{slice_label}_{timestep}.xdmf'),
        os.path.join(case_dir, '2_visu', f'domain1_t_avg_thermo_{slice_label}_{timestep}.xdmf'),
        os.path.join(case_dir, '2_visu', f'domain1_tsp_avg_thermo_{slice_label}_{timestep}.xdmf'),
        os.path.join(case_dir, '2_visu', f'domain1_mhd_{slice_label}_{timestep}.xdmf'),
        os.path.join(case_dir, '2_visu', f'domain1_t_avg_mhd_{slice_label}_{timestep}.xdmf'),
        os.path.join(case_dir, '2_visu', f'domain1_tsp_avg_mhd_{slice_label}_{timestep}.xdmf'),
    ]
    return file_names


def slice_axis_info(slice_label):
    """
    Get axis labels and grid coordinate keys for a 2D slice.

    Args:
        slice_label: Slice label string (e.g., 'yi8')

    Returns:
        dict with keys:
            'plane': slice plane name ('xy', 'xz', or 'yz')
            'normal_dir': direction normal to slice ('x', 'y', or 'z')
            'normal_index': integer index along the normal direction
            'axis_labels': tuple of (xlabel, ylabel) for plotting
            'coord_keys': tuple of (coord1_key, coord2_key) grid_info keys
        or None if slice_label is invalid
    """
    parsed = parse_slice_label(slice_label)
    if parsed is None:
        return None

    direction, index = parsed

    if direction == 'y':
        # xz slice at constant y
        return {
            'plane': 'xz',
            'normal_dir': 'y',
            'normal_index': index,
            'axis_labels': ('$x$', '$z$'),
            'coord_keys': ('grid_x', 'grid_z'),
        }
    elif direction == 'x':
        # yz slice at constant x
        return {
            'plane': 'yz',
            'normal_dir': 'x',
            'normal_index': index,
            'axis_labels': ('$y$', '$z$'),
            'coord_keys': ('grid_y', 'grid_z'),
        }
    elif direction == 'z':
        # xy slice at constant z
        return {
            'plane': 'xy',
            'normal_dir': 'z',
            'normal_index': index,
            'axis_labels': ('$x$', '$y$'),
            'coord_keys': ('grid_x', 'grid_y'),
        }
    return None


def parse_x_crop_input(text):
    """Parse crop input string to ``(x_min, x_max)``.

    Accepted format: ``"x_min,x_max"``.
    Returns ``None`` for blank input.
    """
    if text is None:
        return None

    value = str(text).strip()
    if not value:
        return None

    parts = [p.strip() for p in value.split(',') if p.strip()]
    if len(parts) != 2:
        raise ValueError("Expected format 'x_min,x_max'")

    x_min, x_max = float(parts[0]), float(parts[1])
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    return x_min, x_max


def apply_x_crop(data, x_coords, x_crop):
    """Crop array data along the last axis using an x-range.

    Args:
        data: ndarray to crop (1-D/2-D/3-D ...). Cropping is applied on axis ``-1``.
        x_coords: x-coordinate array (cell-centres ``nx`` or edges ``nx+1``).
        x_crop: tuple ``(x_min, x_max)`` or ``None``.

    Returns:
        tuple: ``(cropped_data, cropped_x_coords)``
    """
    if x_crop is None or x_coords is None:
        return data, x_coords

    x = np.asarray(x_coords)
    nx = data.shape[-1]
    x_min, x_max = x_crop

    if x.size == nx:
        centres = x
        use_edges = False
    elif x.size == nx + 1:
        centres = 0.5 * (x[:-1] + x[1:])
        use_edges = True
    else:
        return data, x_coords

    idx = np.where((centres >= x_min) & (centres <= x_max))[0]
    if idx.size == 0:
        print(f"Warning: No data points in x range [{x_min}, {x_max}]")
        return data, x_coords

    i0, i1 = int(idx[0]), int(idx[-1] + 1)
    cropped_data = np.take(data, indices=np.arange(i0, i1), axis=-1)
    if use_edges:
        cropped_x = x[i0:i1 + 1]
    else:
        cropped_x = x[i0:i1]

    return cropped_data, cropped_x

# =====================================================================================================================================================
# XDMF READING UTILITIES
# =====================================================================================================================================================

def read_binary_data_item(data_item, xdmf_dir):
    """
    Read binary data from a DataItem element.

    Args:
        data_item: XML DataItem element
        xdmf_dir: Directory containing the XDMF file

    Returns:
        numpy array or None if reading fails
    """
    format_type = data_item.get('Format', 'Binary')
    if format_type != 'Binary':
        print(f"Unsupported format: {format_type}")
        return None

    # Get data properties
    dims_str = data_item.get('Dimensions', '')
    dims = tuple(int(d) for d in dims_str.split()) if dims_str else None

    number_type = data_item.get('NumberType', 'Float')
    precision = int(data_item.get('Precision', '8'))
    seek = int(data_item.get('Seek', '0'))

    # Determine numpy dtype
    if number_type == 'Float':
        dtype = np.float32 if precision == 4 else np.float64
    elif number_type == 'Int':
        dtype = np.int32 if precision == 4 else np.int64
    else:
        dtype = np.float64

    # Get binary file path
    bin_path = data_item.text.strip() if data_item.text else None
    if bin_path is None:
        return None

    # Resolve relative path - look in ../1_data relative to xdmf_dir
    data_dir = os.path.normpath(os.path.join(xdmf_dir, '..', '1_data'))
    bin_filename = os.path.basename(bin_path)
    bin_path = os.path.join(data_dir, bin_filename)

    if not os.path.isfile(bin_path):
        print(f"Binary file not found: {bin_path}")
        return None

    try:
        itemsize = np.dtype(dtype).itemsize
        if dims:
            count = int(np.prod(dims))
        else:
            # Fallback: calculate from file size
            file_size = os.path.getsize(bin_path)
            count = (file_size - seek) // itemsize

            # Standard reading with explicit count (fixes Lustre EOF hanging)
        with open(bin_path, 'rb') as f:
            f.seek(seek)
            data = np.fromfile(f, dtype=dtype, count=count)

        if dims:
            expected_size = int(np.prod(dims))
            if data.size >= expected_size:
                data = data[:expected_size].reshape(dims)
            elif data.size < expected_size:
                print(f"Warning: Data size mismatch for {bin_path} (got {data.size}, expected {expected_size})")

        return data

    except Exception as e:
        print(f"Error reading {bin_path}: {e}")
        return None


def _strip_avg_prefix(name):
    """Return base variable name without known averaging prefixes."""
    for prefix in ('t_avg_', 'tsp_avg_'):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _is_selected_variable(name, required_vars):
    """Check whether an XDMF variable should be loaded."""
    if required_vars is None:
        return name in REQUIRED_VARS

    base_name = _strip_avg_prefix(name)
    return (name in required_vars) or (base_name in required_vars)

def _reduce_xdmf_array(data, average_z=False, average_x=False):
    """Reduce XDMF array dimensionality. Averages over requested axes then squeezes singletons."""
    if data is None or data.ndim != 3:
        return data

    if average_z and average_x:
        return data.mean(axis=(0, 2))
    if average_z:
        return data.mean(axis=0)
    if average_x:
        return data.mean(axis=2)

    return np.squeeze(data)


def parse_xdmf_file(xdmf_path, load_all_vars=None, required_vars=None,
                    average_z=False, average_x=False):
    """
    Parse XDMF file and extract data from associated binary files.

    Args:
        xdmf_path: Path to the XDMF file
        load_all_vars: If True, load all variables. If False, only load required ones.
                       If None, uses module-level LOAD_ALL_VARS setting.
        required_vars: Optional set/list of exact required variables (base or
                   prefixed names). If None, uses module REQUIRED_VARS.
        average_z: If True, average over the z direction for 3D arrays.
        average_x: If True, average over the x direction for 3D arrays.

    Returns:
        tuple: (arrays dict, grid_info dict). Singleton dimensions (e.g. from tsp_avg
               z-slice files) are automatically squeezed to give true 2D output.
    """
    if load_all_vars is None:
        load_all_vars = LOAD_ALL_VARS

    if not os.path.isfile(xdmf_path):
        return {}, {}

    root = _parse_xdmf_xml(xdmf_path)
    if root is None:
        return {}, {}

    arrays = {}
    grid_info = {}
    xdmf_dir = os.path.dirname(xdmf_path)

    # Find grid information
    for grid in root.iter('Grid'):
        # Get topology dimensions
        for topo in grid.iter('Topology'):
            dims_str = topo.get('Dimensions')
            if dims_str:
                dims = tuple(int(d) for d in dims_str.split())
                grid_info['node_dimensions'] = dims
                grid_info['cell_dimensions'] = tuple(d - 1 for d in dims)

        # Collect all data items to read
        read_tasks = []

        # Get geometry (grid coordinates)
        for geom in grid.iter('Geometry'):
            geom_type = geom.get('GeometryType')
            if geom_type == 'VXVYVZ':
                data_items = list(geom.iter('DataItem'))
                coord_names = ['x', 'y', 'z']
                for i, data_item in enumerate(data_items[:3]):
                    read_tasks.append(('grid', f'grid_{coord_names[i]}', data_item))

        # Get attributes (flow variables) - filter to only required ones if enabled
        skipped_vars = []
        for attribute in grid.iter('Attribute'):
            name = attribute.get('Name')
            data_item = attribute.find('DataItem')
            if data_item is not None:
                if load_all_vars or _is_selected_variable(name, required_vars):
                    read_tasks.append(('array', name, data_item))
                else:
                    skipped_vars.append(name)

        # Read all data items
        if skipped_vars:
            tqdm.write(f"  Skipping {len(skipped_vars)} unneeded variables: {', '.join(skipped_vars[:3])}{'...' if len(skipped_vars) > 3 else ''}")

        xdmf_name = os.path.basename(xdmf_path)
        for task_type, name, data_item in tqdm(read_tasks, desc=f"  Reading {xdmf_name}", unit="var", leave=False):
            data_3d = read_binary_data_item(data_item, xdmf_dir)
            if data_3d is not None:
                # Reshape flat arrays to 3D using topology dimensions if needed
                if len(data_3d.shape) == 1 and 'cell_dimensions' in grid_info:
                    cell_dims = grid_info['cell_dimensions']
                    if data_3d.size == int(np.prod(cell_dims)):
                        data_3d = data_3d.reshape(cell_dims)
                data = _reduce_xdmf_array(
                    data_3d,
                    average_z=average_z,
                    average_x=average_x,
                )
                if data is not data_3d:
                    del data_3d
                if task_type == 'grid':
                    grid_info[name] = data
                else:
                    arrays[name] = data

    return arrays, grid_info


# =====================================================================================================================================================
# XDMF METADATA PARSING & SELECTIVE LOADING
# =====================================================================================================================================================

def _extract_data_item_params(data_item, xdmf_dir):
    """
    Extract binary read parameters from a DataItem XML element
    without actually reading the binary file.

    Returns:
        dict with bin_path, dims, dtype, seek keys, or None if invalid
    """
    format_type = data_item.get('Format', 'Binary')
    if format_type != 'Binary':
        return None

    dims_str = data_item.get('Dimensions', '')
    dims = tuple(int(d) for d in dims_str.split()) if dims_str else None

    number_type = data_item.get('NumberType', 'Float')
    precision = int(data_item.get('Precision', '8'))
    seek = int(data_item.get('Seek', '0'))

    if number_type == 'Float':
        dtype = np.float32 if precision == 4 else np.float64
    elif number_type == 'Int':
        dtype = np.int32 if precision == 4 else np.int64
    else:
        dtype = np.float64

    bin_path_text = data_item.text.strip() if data_item.text else None
    if bin_path_text is None:
        return None

    data_dir = os.path.normpath(os.path.join(xdmf_dir, '..', '1_data'))
    bin_filename = os.path.basename(bin_path_text)
    bin_path = os.path.join(data_dir, bin_filename)

    if not os.path.isfile(bin_path):
        return None

    return {
        'bin_path': bin_path,
        'dims': dims,
        'dtype': dtype,
        'seek': seek,
    }


def _read_binary_from_params(params):
    """
    Read binary data using pre-extracted parameters (from _extract_data_item_params).

    Args:
        params: dict with bin_path, dims, dtype, seek

    Returns:
        numpy array or None on failure
    """
    bin_path = params['bin_path']
    dims = params['dims']
    dtype = params['dtype']
    seek = params['seek']

    try:
        itemsize = np.dtype(dtype).itemsize
        if dims:
            count = int(np.prod(dims))
        else:
            file_size = os.path.getsize(bin_path)
            count = (file_size - seek) // itemsize


        with open(bin_path, 'rb') as f:
            f.seek(seek)
            data = np.fromfile(f, dtype=dtype, count=count)

        if dims:
            expected_size = int(np.prod(dims))
            if data.size >= expected_size:
                data = data[:expected_size].reshape(dims)
            elif data.size < expected_size:
                print(f"Warning: Data size mismatch for {bin_path} "
                      f"(got {data.size}, expected {expected_size})")

        return data

    except Exception as e:
        print(f"Error reading {bin_path}: {e}")
        return None


def parse_xdmf_metadata(xdmf_path):
    """
    Parse XDMF file structure and return variable names, shapes, and grid info
    without loading variable data.  Grid coordinates (small 1D arrays) are loaded.

    Args:
        xdmf_path: Path to the XDMF file

    Returns:
        tuple: (var_metadata, grid_info)
            var_metadata: dict of {name: {'shape': tuple, 'bin_path': str,
                          'dims': tuple, 'dtype': dtype, 'seek': int}}
            grid_info: dict with node_dimensions, cell_dimensions,
                       grid_x, grid_y, grid_z
    """
    if not os.path.isfile(xdmf_path):
        return {}, {}

    root = _parse_xdmf_xml(xdmf_path)
    if root is None:
        return {}, {}

    var_metadata = {}
    grid_info = {}
    xdmf_dir = os.path.dirname(xdmf_path)

    for grid in root.iter('Grid'):
        # Get topology dimensions
        for topo in grid.iter('Topology'):
            dims_str = topo.get('Dimensions')
            if dims_str:
                dims = tuple(int(d) for d in dims_str.split())
                grid_info['node_dimensions'] = dims
                grid_info['cell_dimensions'] = tuple(d - 1 for d in dims)

        # Load grid coordinates (small 1D arrays — always needed for slicing)
        for geom in grid.iter('Geometry'):
            geom_type = geom.get('GeometryType')
            if geom_type == 'VXVYVZ':
                data_items = list(geom.iter('DataItem'))
                coord_names = ['x', 'y', 'z']
                for i, data_item in enumerate(data_items[:3]):
                    data = read_binary_data_item(data_item, xdmf_dir)
                    if data is not None:
                        grid_info[f'grid_{coord_names[i]}'] = data

        # Extract variable metadata without loading binary data
        for attribute in grid.iter('Attribute'):
            name = attribute.get('Name')
            data_item = attribute.find('DataItem')
            if data_item is not None:
                params = _extract_data_item_params(data_item, xdmf_dir)
                if params is not None:
                    # Compute effective shape (after potential reshape)
                    raw_dims = params['dims']
                    if raw_dims and len(raw_dims) > 1:
                        params['shape'] = raw_dims
                    elif 'cell_dimensions' in grid_info:
                        cell_dims = grid_info['cell_dimensions']
                        if raw_dims:
                            size = int(np.prod(raw_dims))
                        else:
                            itemsize = np.dtype(params['dtype']).itemsize
                            file_size = os.path.getsize(params['bin_path'])
                            size = (file_size - params['seek']) // itemsize
                        if size == int(np.prod(cell_dims)):
                            params['shape'] = cell_dims
                        else:
                            params['shape'] = raw_dims if raw_dims else (size,)
                    else:
                        params['shape'] = raw_dims
                    var_metadata[name] = params

    return var_metadata, grid_info


def load_xdmf_variables(var_metadata, selected_vars, grid_info=None,
                        average_z=False, average_x=False):
    """
    Load specific variables using pre-parsed XDMF metadata.

    Args:
        var_metadata: dict from parse_xdmf_metadata
        selected_vars: list of variable names to load
        grid_info: grid info dict (for reshaping flat arrays)
        average_z: If True, average over the z direction for 3D arrays.
        average_x: If True, average over the x direction for 3D arrays.

    Returns:
        dict: {variable_name: numpy_array}. Singleton dimensions are automatically
              squeezed (e.g. tsp_avg slice files become true 2D).
    """
    arrays = {}

    for name in tqdm(selected_vars, desc="Loading selected variables", unit="var"):
        if name not in var_metadata:
            tqdm.write(f"  Variable '{name}' not found in metadata, skipping")
            continue

        params = var_metadata[name]
        data = _read_binary_from_params(params)
        if data is None:
            continue

        # Reshape flat arrays to 3D using cell dimensions
        if len(data.shape) == 1 and grid_info and 'cell_dimensions' in grid_info:
            cell_dims = grid_info['cell_dimensions']
            if data.size == int(np.prod(cell_dims)):
                data = data.reshape(cell_dims)

        data = _reduce_xdmf_array(data, average_z=average_z, average_x=average_x)

        arrays[name] = data
        tqdm.write(f"  Loaded {name}: shape {data.shape}")

    return arrays


def xdmf_reader_wrapper(file_names, case=None, timestep=None, load_all_vars=None, data_types=None,
                         required_vars=None,
                         average_z=False, average_x=False):
    """
    Reads XDMF files and extracts numpy arrays from binary data.

    Args:
        file_names (list): List of XDMF file paths to read
        case (str, optional): Case identifier for dictionary key
        timestep (str, optional): Timestep identifier for dictionary key
        load_all_vars (bool, optional): If True, load all variables. If False, only required ones.
        data_types (list, optional): List of data types to load. If None, loads all files.
            Valid types: 'inst', 't_avg',
            or specific combinations like 't_avg_flow', 't_avg_thermo', etc.
            Note: tsp_avg data is stored as .txt files in 1_data/ and is
            NOT available as XDMF.  Use the text data loader for tsp_avg data.
            Examples:
                - ['t_avg'] loads only time-averaged files (flow, thermo, mhd)
                - ['inst'] loads all instantaneous files (flow, thermo, mhd)
                - ['t_avg_flow'] loads only time-averaged flow files

        required_vars (set/list, optional): Minimal variable set to load.
            If None, falls back to module REQUIRED_VARS.

    Returns:
        tuple: (visu_arrays_dic dict, grid_info dict)

        If case and timestep are provided, returns nested dictionary:
        {
            "case_timestep": {
                "variable_name": numpy_array,
                ...
            }
        }

        Otherwise returns flat dictionary:
        {
            "variable_name": numpy_array,
            ...
        }
    """
    # Determine if we should use nested structure
    use_nested = case is not None and timestep is not None

    if use_nested:
        # Create the outer key for this case/timestep combination
        outer_key = f"{case}_{timestep}"
        visu_arrays_dic = {outer_key: {}}
        inner_dict = visu_arrays_dic[outer_key]
    else:
        visu_arrays_dic = {}
        inner_dict = visu_arrays_dic

    grid_info = {}

    # Filter to only existing files for accurate progress bar
    existing_files = [f for f in file_names if os.path.isfile(f)]

    # Select which files to load, by priority or explicit data_types
    if data_types is not None:
        filtered_files = []
        for f in existing_files:
            filename = os.path.basename(f)
            for dtype in data_types:
                if dtype == 't_avg':
                    if 't_avg_' in filename:
                        filtered_files.append(f)
                        break
                elif dtype == 'inst':
                    if 't_avg' not in filename:
                        filtered_files.append(f)
                        break
                else:
                    if dtype in filename:
                        filtered_files.append(f)
                        break
        existing_files = filtered_files
        tqdm.write(f"Filtering for data types: {data_types}")
    else:
        # Auto-select tier: tsp_avg_zi1 (if average_z) > t_avg > inst
        tsp_files  = [f for f in existing_files if 'tsp_avg_' in os.path.basename(f)]
        t_avg_files = [f for f in existing_files if '_t_avg_' in os.path.basename(f)]
        inst_files  = [f for f in existing_files if 't_avg' not in os.path.basename(f)]
        if average_z and tsp_files:
            existing_files = tsp_files
            tqdm.write("Using tsp_avg_zi1 (pre-z-averaged) files")
        elif t_avg_files:
            existing_files = t_avg_files
            tqdm.write("Using 3D time-averaged files")
        else:
            existing_files = inst_files
            tqdm.write("No time-averaged files found, loading instantaneous files")

    for xdmf_file in tqdm(existing_files, desc="Processing XDMF files", unit="file"):
        try:
            tqdm.write(f"Opening file: {xdmf_file}")
            is_tsp_avg = 'tsp_avg_' in os.path.basename(xdmf_file)

            arrays, file_grid_info = parse_xdmf_file(
                xdmf_file,
                load_all_vars=load_all_vars,
                required_vars=required_vars,
                average_z=False if is_tsp_avg else average_z,
                average_x=average_x
            )

            if arrays:
                if 't_avg' in xdmf_file:
                    file_type = 't_avg'
                elif '_mhd_' in xdmf_file:
                    file_type = 'mhd'
                elif '_thermo_' in xdmf_file:
                    file_type = 'thermo'
                else:
                    file_type = 'flow'

                if not grid_info and file_grid_info:
                    grid_info = file_grid_info
                    if 'node_dimensions' in grid_info:
                        tqdm.write(f"Grid info: node_dimensions={grid_info['node_dimensions']}, cell_dimensions={grid_info.get('cell_dimensions', 'N/A')}")

                for var_name, var_data in arrays.items():
                    if var_name.startswith('tsp_avg_'):
                        base_name = var_name[8:]
                    elif var_name.startswith('t_avg_'):
                        base_name = var_name[6:]
                    else:
                        base_name = var_name
                    inner_dict[base_name] = var_data
                tqdm.write(f"Successfully extracted {len(arrays)} arrays from {file_type} file")
            else:
                tqdm.write(f"Warning: No valid output from {xdmf_file}, file missing or empty")

        except Exception as e:
            tqdm.write(f"Error processing {xdmf_file}: {str(e)}")
            continue

    # Warn about any variables that were requested but not found in any loaded file
    if not load_all_vars and inner_dict:
        check_vars = required_vars if required_vars is not None else _BASE_REQUIRED_VARS
        missing = sorted(v for v in check_vars if v not in inner_dict)
        if missing:
            tqdm.write(f"WARNING: {len(missing)} requested variable(s) not found in loaded files: {', '.join(missing)}")

    return visu_arrays_dic, grid_info


def extract_grid_info_from_arrays(grid_info):
    """
    Extract grid dimensions and bounds from grid coordinate arrays.

    Args:
        grid_info: Dictionary containing grid_x, grid_y, grid_z arrays

    Returns:
        dict: Enhanced grid information including bounds
    """
    enhanced_info = dict(grid_info)

    try:
        if 'grid_x' in grid_info and 'grid_y' in grid_info and 'grid_z' in grid_info:
            x = grid_info['grid_x']
            y = grid_info['grid_y']
            z = grid_info['grid_z']

            enhanced_info['bounds'] = (
                float(x.min()), float(x.max()),
                float(y.min()), float(y.max()),
                float(z.min()), float(z.max())
            )

            # Calculate average spacing
            if len(x) > 1:
                dx = (x.max() - x.min()) / (len(x) - 1)
            else:
                dx = 0
            if len(y) > 1:
                dy = (y.max() - y.min()) / (len(y) - 1)
            else:
                dy = 0
            if len(z) > 1:
                dz = (z.max() - z.min()) / (len(z) - 1)
            else:
                dz = 0

            enhanced_info['average_spacing'] = (dx, dy, dz)

    except Exception as e:
        print(f"Could not extract complete grid info: {str(e)}")

    return enhanced_info

# =====================================================================================================================================================
# OUTPUT UTILITIES
# =====================================================================================================================================================

def reader_output_summary(arrays_dict):
    """
    Provides a summary analysis of the extracted arrays.

    Args:
        arrays_dict (dict): Dictionary of numpy arrays (can be nested or flat)
    """
    print("\n" + "="*60)
    print("READER OUTPUT SUMMARY")
    print("="*60)

    # Check if this is a nested dictionary (case_timestep structure)
    first_key = next(iter(arrays_dict))
    is_nested = isinstance(arrays_dict[first_key], dict)

    if is_nested:
        # Handle nested structure: {case_timestep: {variable: array}}
        for case_timestep, variables in arrays_dict.items():
            print(f"\n{case_timestep}:")
            print("-" * 60)
            for var_name, array in variables.items():
                print(f"  {var_name}:")
                print(f"    Shape: {array.shape},  Min: {np.min(array):.6e},  Max: {np.max(array):.6e},  Mean: {np.mean(array):.6e}")
    else:
        # Handle flat structure: {variable: array}
        for key, array in arrays_dict.items():
            print(f"{key}:")
            print(f"  Shape: {array.shape},  Min value: {np.min(array):.6e},  Max value: {np.max(array):.6e}   Mean value: {np.mean(array):.6e}")
            print("-" * 40)

# =====================================================================================================================================================
# DATA CLEANING UTILITIES
# =====================================================================================================================================================

def clean_dat_file(input_file, output_file, expected_cols):
    clean_data = []
    bad_lines = []

    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num <= 3:
                continue
            try:
                values = [float(x) for x in line.split()]
                if len(values) == expected_cols:
                    clean_data.append(values)
                else:
                    bad_lines.append((line_num, len(values), line.strip()))

            except ValueError as e:
                bad_lines.append((line_num, 'ERROR', line.strip()))

    if bad_lines:
        print(f"Found {len(bad_lines)} problematic lines")

    np.savetxt(f'monitor_point_plots/{output_file}', clean_data, fmt='%.5E')
    print(f"\nSaved {len(clean_data)} clean lines to {output_file}")

    return np.array(clean_data)

# =====================================================================================================================================================
# PLOTTING UTILITIES
# =====================================================================================================================================================

def get_col(case, cases, colours):
    if len(cases) > 1:
        colour = colours[cases.index(case)]
    else:
        colour = colours[0]
    return colour

def print_flow_info(ux_data, Re_ref, Re_bulk, case, timestep, y_coords=None):

    Re_ref = int(Re_ref)
    if y_coords is not None:
        du = ux_data[0] - ux_data[1]
        dy = y_coords[0] - y_coords[1]
    else:
        du = ux_data[0, 2] - ux_data[1, 2]
        dy = ux_data[0, 1] - ux_data[1, 1]
    dudy = np.mean(du/dy)  # average over any spatial dims for scalar output
    tau_w = dudy/Re_ref # this should be ref Re not real bulk Re
    u_tau = np.sqrt(abs(dudy/Re_ref))
    Re_tau = u_tau * Re_ref
    print(f'Case: {case}, Timestep: {timestep}')
    print(f'Re_bulk = {Re_bulk}, u_tau = {u_tau:.6f}, tau_w = {tau_w:.6e}, Re_tau = {Re_tau:.2f}')
    print('-'*120)
    return

def get_plane_data(domain_array, plane, index):
    if plane == 'xy':
        return domain_array[index, :, :]
    elif plane == 'xz':
        return domain_array[:, index, :]
    elif plane == 'yz':
        return domain_array[:, :, index]
    else:
        print(f"Error: Invalid plane '{plane}' specified. Use 'xy', 'xz', or 'yz'.")
    return
