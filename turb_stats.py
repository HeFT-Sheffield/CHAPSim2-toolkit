#!/usr/bin/env python3
# turb_stats.py
# This script processes computes first order turbulence statistics from time and space averaged data for channel flow simulations.

# import libraries ------------------------------------------------------------------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
import math
import os
from tqdm import tqdm

mpl.rcParams.update({
"font.family": "serif",
"font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
"mathtext.fontset": "cm",
"axes.unicode_minus": False,
})

# import modules --------------------------------------------------------------------------------------------------------------------------------------
import operations as op
import utils as ut

# =====================================================================================================================================================
# CONFIGURATION CLASSES
# =====================================================================================================================================================

@dataclass
class Config:
    """Configuration wrapper for all settings"""
    folder_path: str
    input_format: str
    cases: List[str]
    timesteps: List[str]
    average_over_timesteps: bool
    thermo_on: bool
    forcing: str
    Re: List[float]
    ref_temp: List[float]
    ref_length: List[float]
    ref_bulk_velocity: List[float]
    wall_heat_flux: List[float]
    working_fluid: str
    gravity_direction: list[float]
    mhd_on: bool
    mag_field_direction: list[float]
    stuart_number: float

    ux_velocity_on: bool
    uy_velocity_on: bool
    uz_velocity_on: bool
    temp_on: bool
    heat_transf_coeff_on: bool
    Nusselt_number_on: bool
    turb_prandtl_on: bool
    coeff_friction_on: bool
    tke_on: bool
    profile_direction: str
    slice_coords: str
    x_crop: str
    x_profile_y_coords: str
    surface_plot_on: bool
    u_prime_sq_on: bool
    u_prime_v_prime_on: bool
    v_prime_sq_on: bool
    v_prime_w_prime_on: bool
    w_prime_sq_on: bool

    re_stress_budget_on: bool
    re_stress_component: str
    average_z_direction: bool
    average_x_direction: bool

    norm_by_u_tau_sq: bool
    norm_ux_by_u_tau: bool
    norm_y_to_y_plus: bool
    norm_temp_by_ref_temp: bool

    half_channel_plot: bool
    linear_y_scale: bool
    log_y_scale: bool
    multi_plot: bool
    display_fig: bool
    save_fig: bool
    save_to_path: bool
    large_text_on: bool
    plot_name: str

    ux_velocity_log_ref_on: bool
    mhd_NK_ref_on: bool
    mkm180_ch_ref_on: bool
    xdmf_data_type: str = 'tsp_avg'

    body_force_on: bool = False
    body_force_component: str = '1'

    # Peak (wall-normal max) TKE and components vs x — see PeakTKE / PeakReynoldsStressuu*.
    tke_peak_growth_on: bool = False

    # Vorticity profiles — see MeanVorticity / VorticityFluctuationRMS.
    mean_vorticity_on: bool = False
    vorticity_on: bool = False
    vorticity_component: str = 'z'

    # Anisotropy-tensor invariants, plotted as Lumley triangles — see
    # AnisotropyComputer.
    reynolds_anisotropy_on: bool = False
    vorticity_anisotropy_on: bool = False

    # Which wall half_channel_plot shows: 'lower' -> the y=-1 wall, 'upper'
    # -> the y=+1 wall, 'average' -> the mean for symmetric flows.
    half_channel_side: str = 'lower'

    # Mean electric current density profiles — see MeanCurrentDensityj1/j2/j3.
    j1_mean_on: bool = False
    j2_mean_on: bool = False
    j3_mean_on: bool = False

    # RMS electric current density fluctuation profiles — see
    # CurrentDensityRMSj1/j2/j3.
    j1_rms_on: bool = False
    j2_rms_on: bool = False
    j3_rms_on: bool = False

    # Mean Lorentz body-force profiles — see MeanLorentzForcex/y/z.
    lorentz_force_x_on: bool = False
    lorentz_force_y_on: bool = False
    lorentz_force_z_on: bool = False

    # Which Noguchi & Kasagi reference curve mhd_NK_ref_on overlays — applied
    # to every plotted case, independent of case name (see _plot_reference_data).
    mhd_NK_ref_case: str = 'Ha_6'

    # 1D streamwise-velocity-fluctuation energy spectrum — see SpectrumComputer.
    spectrum_on: bool = False
    spectrum_direction: str = 'z'
    spectrum_y_coords: str = ''

    @classmethod
    def from_module(cls, config_module):
        """Create Config from imported config module"""
        return cls(
            folder_path=getattr(config_module, 'folder_path', ''),
            input_format=getattr(config_module, 'input_format', 'dat'),
            cases=getattr(config_module, 'cases', []),
            timesteps=getattr(config_module, 'timesteps', []),
            average_over_timesteps=getattr(config_module, 'average_over_timesteps', False),
            thermo_on=getattr(config_module, 'thermo_on', False),
            forcing=getattr(config_module, 'forcing', 'CMF'),
            Re=getattr(config_module, 'Re', [1.0]),
            ref_temp=getattr(config_module, 'ref_temp', [300.0]),
            ref_length=getattr(config_module, 'ref_length', [1.0]),
            ref_bulk_velocity=getattr(config_module, 'ref_bulk_velocity', [1.0]),
            wall_heat_flux=getattr(config_module, 'wall_heat_flux', [0.0]),
            working_fluid=getattr(config_module, 'working_fluid', 'lithium'),
            gravity_direction=getattr(config_module, 'gravity_direction', [0.0, -1.0, 0.0]),
            mhd_on=getattr(config_module, 'mhd_on', False),
            mag_field_direction=getattr(config_module, 'mag_field_direction', [0.0, 1.0, 0.0]),
            stuart_number=getattr(config_module, 'stuart_number', 0.0),
            ux_velocity_on=getattr(config_module, 'ux_velocity_on', False),
            uy_velocity_on=getattr(config_module, 'uy_velocity_on', False),
            uz_velocity_on=getattr(config_module, 'uz_velocity_on', False),
            temp_on=getattr(config_module, 'temp_on', False),
            heat_transf_coeff_on=getattr(config_module, 'heat_transf_coeff_on', False),
            Nusselt_number_on=getattr(config_module, 'Nusselt_number_on', False),
            turb_prandtl_on=getattr(config_module, 'turb_prandtl_on', False),
            coeff_friction_on=getattr(config_module, 'coeff_friction_on', False),
            tke_on=getattr(config_module, 'tke_on', False),
            u_prime_sq_on=getattr(config_module, 'u_prime_sq_on', False),
            u_prime_v_prime_on=getattr(config_module, 'u_prime_v_prime_on', False),
            v_prime_sq_on=getattr(config_module, 'v_prime_sq_on', False),
            v_prime_w_prime_on=getattr(config_module, 'v_prime_w_prime_on', False),
            w_prime_sq_on=getattr(config_module, 'w_prime_sq_on', False),
            profile_direction=getattr(config_module, 'profile_direction', 'y'),
            slice_coords=getattr(config_module, 'slice_coords', ''),
            x_crop=getattr(config_module, 'x_crop', ''),
            x_profile_y_coords=getattr(config_module, 'x_profile_y_coords', ''),
            surface_plot_on=getattr(config_module, 'surface_plot_on', False),
            re_stress_budget_on=getattr(config_module, 're_stress_budget_on', False),
            re_stress_component=getattr(config_module, 're_stress_component', 'total'),
            average_z_direction=getattr(config_module, 'average_z_direction', True),
            average_x_direction=getattr(config_module, 'average_x_direction', False),
            norm_by_u_tau_sq=getattr(config_module, 'norm_by_u_tau_sq', False),
            norm_ux_by_u_tau=getattr(config_module, 'norm_ux_by_u_tau', False),
            norm_y_to_y_plus=getattr(config_module, 'norm_y_to_y_plus', False),
            norm_temp_by_ref_temp=getattr(config_module, 'norm_temp_by_ref_temp', False),
            half_channel_plot=getattr(config_module, 'half_channel_plot', False),
            linear_y_scale=getattr(config_module, 'linear_y_scale', True),
            log_y_scale=getattr(config_module, 'log_y_scale', False),
            multi_plot=getattr(config_module, 'multi_plot', True),
            display_fig=getattr(config_module, 'display_fig', False),
            save_fig=getattr(config_module, 'save_fig', True),
            save_to_path=getattr(config_module, 'save_to_path', False),
            large_text_on=getattr(config_module, 'large_text_on', False),
            plot_name=getattr(config_module, 'plot_name', ''),
            ux_velocity_log_ref_on=getattr(config_module, 'ux_velocity_log_ref_on', False),
            mhd_NK_ref_on=getattr(config_module, 'mhd_NK_ref_on', False),
            mkm180_ch_ref_on=getattr(config_module, 'mkm180_ch_ref_on', False),
            xdmf_data_type=getattr(config_module, 'xdmf_data_type', 'tsp_avg'),
            body_force_on=getattr(config_module, 'body_force_on', False),
            body_force_component=getattr(config_module, 'body_force_component', '1'),
            tke_peak_growth_on=getattr(config_module, 'tke_peak_growth_on', False),
            mean_vorticity_on=getattr(config_module, 'mean_vorticity_on', False),
            vorticity_on=getattr(config_module, 'vorticity_on', False),
            vorticity_component=getattr(config_module, 'vorticity_component', 'z'),
            reynolds_anisotropy_on=getattr(config_module, 'reynolds_anisotropy_on', False),
            vorticity_anisotropy_on=getattr(config_module, 'vorticity_anisotropy_on', False),
            half_channel_side=getattr(config_module, 'half_channel_side', 'lower'),
            j1_mean_on=getattr(config_module, 'j1_mean_on', False),
            j2_mean_on=getattr(config_module, 'j2_mean_on', False),
            j3_mean_on=getattr(config_module, 'j3_mean_on', False),
            j1_rms_on=getattr(config_module, 'j1_rms_on', False),
            j2_rms_on=getattr(config_module, 'j2_rms_on', False),
            j3_rms_on=getattr(config_module, 'j3_rms_on', False),
            lorentz_force_x_on=getattr(config_module, 'lorentz_force_x_on', False),
            lorentz_force_y_on=getattr(config_module, 'lorentz_force_y_on', False),
            lorentz_force_z_on=getattr(config_module, 'lorentz_force_z_on', False),
            mhd_NK_ref_case=getattr(config_module, 'mhd_NK_ref_case', 'Ha_6'),
            spectrum_on=getattr(config_module, 'spectrum_on', False),
            spectrum_direction=getattr(config_module, 'spectrum_direction', 'z'),
            spectrum_y_coords=getattr(config_module, 'spectrum_y_coords', ''),
        )

@dataclass
class PlotConfig:
    """Configuration for plotting aesthetics"""

    colors_1: Dict[str, str] = None
    colors_2: Dict[str, str] = None
    stat_labels: Dict[str, str] = None
    budget_colors: Dict[str, str] = None
    visible_palette: List[str] = None

    def __post_init__(self):
        if self.colors_1 is None:
            self.colors_1 = {
                'ux_velocity': '#1f77b4',
                'uy_velocity': '#17becf',
                'uz_velocity': '#ff7f0e',
                'u_prime_sq': '#d62728',
                'u_prime_v_prime': '#2ca02c',
                'w_prime_sq': '#9467bd',
                'v_prime_sq': '#8c564b',
                'v_prime_w_prime': "#bf19c5",
            }

        if self.colors_2 is None:
            self.colors_2 = {
                'ux_velocity': '#e41a1c',
                'uy_velocity': '#377eb8',
                'uz_velocity': '#4daf4a',
                'u_prime_sq': '#ff7f00',
                'u_prime_v_prime': "#63ff6b",
                'w_prime_sq': '#377eb8',
                'v_prime_sq': '#984ea3',
                'v_prime_w_prime': "#bf19c5",
            }

        if self.budget_colors is None:
            self.budget_colors = {
                'production': '#d62728',
                'dissipation': '#1f77b4',
                'mean_convection': '#2ca02c',
                'viscous_diffusion': '#ff7f0e',
                'pressure_transport': '#9467bd',
                'turbulent_diffusion': '#17becf',
                'buoyancy': '#e377c2',
                'mhd': '#8c564b',
                'pressure_strain': '#bcbd22',
                'turbulent_convection': '#7f7f7f',
                'pressure_force': '#9467bd',
                'buoyancy_force': '#e377c2',
                'lorentz_force': '#8c564b',
            }

        if self.visible_palette is None:
            # 40 mutually-distinct, saturated colours (not pastels) — large
            # enough that realistic figures (many cases/x-coordinates/stats)
            # don't wrap around and repeat a colour on two different series.
            self.visible_palette = [
                '#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e',
                '#8c564b', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                '#332288', '#117733', '#882255', '#44aa99', '#aa4499',
                '#88ccee', '#ddcc77', '#cc6677', '#661100', '#0c7c59',
                '#e6194b', '#f58231', '#ffe119', '#bcf60c', '#3cb44b',
                '#46f0f0', '#4363d8', '#911eb4', '#f032e6', '#808000',
                '#000075', '#800000', '#9a6324', '#008080', '#ff1493',
                '#20b2aa', '#556b2f', '#ff4500', '#6a5acd', '#2f4f4f',
            ]

        if self.stat_labels is None:
            self.stat_labels = {
                "ux_velocity": "Streamwise Velocity",
                "uy_velocity": "Wall-Normal Velocity",
                "uz_velocity": "Spanwise Velocity",
                "u_prime_sq": "<u'u'>",
                "u_prime_v_prime": "<u'v'>",
                "v_prime_sq": "<v'v'>",
                "v_prime_w_prime": "<v'w'>",
                "w_prime_sq": "<w'w'>",
                "TKE": "Turbulent Kinetic Energy",
                "heat_transfer_coeff": "Heat Transfer Coefficient",
                "nusselt_number": "Nusselt Number",
                # TKE Budget terms
                "production": "Production",
                "dissipation": "Dissipation",
                "convection": "Convection",
                "viscous_diffusion": "Viscous Diffusion",
                "pressure_transport": "Pressure Transport",
                "turbulent_diffusion": "Turbulent Diffusion",
                "buoyancy": "Buoyancy",
                "mhd": "MHD (Lorentz)",
                # Body force terms
                "pressure_force": "Pressure Force",
                "buoyancy_force": "Buoyancy Force",
                "lorentz_force": "Lorentz Force",
            }

# =====================================================================================================================================================
# DATA LOADING & MANAGEMENT
# =====================================================================================================================================================

def create_data_loader(config: Config, data_types: List[str] = None):
    """
    Factory function to create the appropriate data loader based on input_format.

    Args:
        config: Configuration object
        data_types: List of XDMF data types to load. Only used for XDMF format.

    Returns:
        Data loader instance (TurbulenceTextData or TurbulenceXDMFData)
    """
    fmt = config.input_format.lower()

    if fmt in ['xdmf', 'visu']:
        required_vars = _build_required_xdmf_vars(config)
        re_stress_enabled = (config.u_prime_sq_on or config.u_prime_v_prime_on or
                             config.v_prime_sq_on or config.v_prime_w_prime_on or
                             config.w_prime_sq_on or config.tke_on or config.vorticity_on or
                             config.reynolds_anisotropy_on or config.vorticity_anisotropy_on or
                             config.j1_rms_on or config.j2_rms_on or config.j3_rms_on or
                             config.tke_peak_growth_on)
        re_stress_budget_enabled = config.re_stress_budget_on

        # Build data_types from what is actually enabled so we don't try
        # to read files that may not exist.

        if data_types is None:
            data_types = [config.xdmf_data_type]
        else:
            data_types = list(data_types)

        if re_stress_enabled or re_stress_budget_enabled or config.body_force_on:
            if not any(dtype == 't_avg' or dtype == ('tsp_avg') for dtype in data_types):
                data_types.append('t_avg')
                print("Added 't_avg' to data_types because enabled stats require time-averaged fields.")
        print(f"Using XDMF data loader with data_types={data_types}...")
        
        return TurbulenceXDMFData(
            config.folder_path,
            config.cases,
            config.timesteps,
            data_types=data_types,
            required_vars=required_vars,
            average_z=config.average_z_direction,
            average_x=config.average_x_direction,
            x_crop=config.x_crop,
            average_over_timesteps=config.average_over_timesteps,
        )
    elif fmt in ['dat', 'text']:
        print("Using text (.dat) data loader...")
        return TurbulenceTextData(
            config.folder_path,
            config.cases,
            config.timesteps,
            config.thermo_on
        )
    else:
        raise ValueError(f"Unknown input_format: '{config.input_format}'. Must be 'dat', 'text', 'xdmf', or 'visu'")


def _build_required_xdmf_vars(config: Config) -> Optional[set]:
    """Build the minimal variable set required by enabled computations.

    Returns ``None`` when broad loading is required (e.g., budget terms).
    """
    # TKE budget computations need many coupled terms; keep broad loading.
    if config.re_stress_budget_on:
        return None

    required = set()

    if config.ux_velocity_on:
        required.add('u1')
    if config.uy_velocity_on:
        required.add('u2')
    if config.uz_velocity_on:
        required.add('u3')

    if config.temp_on or config.heat_transf_coeff_on or config.Nusselt_number_on:
        required.update({'T', 'Temperature', 'temp', 
                         'fuh1', 'fu1'})

    if config.u_prime_sq_on:
        required.update({'u1', 'uu11'})
    if config.u_prime_v_prime_on:
        required.update({'u1', 'u2', 'uu12'})
    if config.v_prime_sq_on:
        required.update({'u2', 'uu22'})
    if config.v_prime_w_prime_on:
        required.update({'u2', 'u3', 'uu23'})
    if config.w_prime_sq_on:
        required.update({'u3', 'uu33'})
    if config.tke_on or config.tke_peak_growth_on:
        required.update({'u1', 'u2', 'u3', 'uu11', 'uu22', 'uu33'})
    if config.coeff_friction_on:
        required.add('u1')
    if config.heat_transf_coeff_on or config.Nusselt_number_on:
        required.update({'T', 'fuh1', 'fu1'})
    if config.turb_prandtl_on:
        required.update({'u1', 'u2', 'uu12', 'T', 'Tu2'})

    if config.body_force_on:
        required.add('pr')
        if config.thermo_on:
            required.add('f')
        if config.mhd_on:
            required.update({'j1', 'j2', 'j3'})

    if config.mean_vorticity_on:
        required.update({'u1', 'u2', 'u3'})
    if config.vorticity_on:
        required.update({'u1', 'u2', 'u3', 'uu11', 'uu22', 'uu33'})
    if config.reynolds_anisotropy_on:
        required.update({'u1', 'u2', 'u3', 'uu11', 'uu12', 'uu13', 'uu22', 'uu23', 'uu33'})
    if config.vorticity_anisotropy_on:
        required.update({'u1', 'u2', 'u3', 'uu11', 'uu22', 'uu33'})

    if config.j1_mean_on:
        required.add('j1')
    if config.j2_mean_on:
        required.add('j2')
    if config.j3_mean_on:
        required.add('j3')
    if config.j1_rms_on:
        required.update({'j1', 'jj11'})
    if config.j2_rms_on:
        required.update({'j2', 'jj22'})
    if config.j3_rms_on:
        required.update({'j3', 'jj33'})

    if config.lorentz_force_x_on or config.lorentz_force_y_on or config.lorentz_force_z_on:
        required.update({'j1', 'j2', 'j3'})

    # Ensure downstream normalization/flow-info/coordinate logic can run.
    if required:
        required.add('u1')

    return required
    

class TurbulenceTextData:
    """Manages time-space averaged data loading and access from .dat files"""

    def __init__(self, folder_path: str, cases: List[str], timesteps: List[str], thermo_on: bool):
        self.folder_path = folder_path
        self.cases = cases
        self.timesteps = timesteps
        # Nested structure: {case_timestep: {quantity: array}}
        self.data: Dict[str, Dict[str, np.ndarray]] = {}
        self.y_coords: Optional[np.ndarray] = None

    def load_all(self) -> None:
        """Load all time-space averaged .dat files found in each case directory."""
        for case in self.cases:
            for timestep in self.timesteps:
                self._load_all_for_case_timestep(case, timestep)

    def _load_all_for_case_timestep(self, case: str, timestep: str) -> None:
        """Discover and load every domain1_tsp_avg_*_{timestep}.dat file for a case."""
        import glob
        data_dir = os.path.join(ut.case_path(self.folder_path, case), '1_data')
        pattern = os.path.join(data_dir, f'domain1_tsp_avg_*_{timestep}.dat')
        files = sorted(glob.glob(pattern))
        if not files:
            print(f'No .dat files found for {case}, {timestep}')
            return
        key = f"{case}_{timestep}"
        if key not in self.data:
            self.data[key] = {}
        for file_path in files:
            fname = os.path.basename(file_path)
            # extract quantity from domain1_tsp_avg_{quantity}_{timestep}.dat
            suffix = f'_{timestep}.dat'
            prefix = 'domain1_tsp_avg_'
            quantity = fname[len(prefix):-len(suffix)]
            data = ut.load_ts_avg_data(file_path)
            if data is not None:
                self.data[key][quantity] = data
                if self.y_coords is None and data.ndim == 2 and data.shape[1] >= 2:
                    self.y_coords = data[:, 1]
            else:
                print(f'.dat file is empty for {case}, {timestep}, {quantity}')

    def get(self, case: str, quantity: str, timestep: str) -> Optional[np.ndarray]:
        """Get specific data array (value column only)."""
        key = f"{case}_{timestep}"
        arr = self.data.get(key, {}).get(quantity)
        if arr is None:
            return None
        return arr[:, 2]

    def get_raw_dict(self, case: str, timestep: str) -> Optional[Dict[str, np.ndarray]]:
        """Get the full data dictionary for a case/timestep (for TKE budget computation)."""
        key = f"{case}_{timestep}"
        raw = self.data.get(key, None)
        if raw is None:
            return None
        return {name: arr[:, 2] for name, arr in raw.items()}

    def has(self, case: str, quantity: str, timestep: str) -> bool:
        """Check if data exists"""
        key = f"{case}_{timestep}"
        return key in self.data and quantity in self.data[key]

    def keys(self):
        """Return all data keys (case_timestep format)"""
        return self.data.keys()

class TurbulenceXDMFData:
    """Manages all data from .xdmf files -- stores native numpy arrays."""

    def __init__(self, folder_path: str, cases: List[str], timesteps: List[str],
                 data_types: List[str] = None, average_z: bool = True, average_x: bool = False,
                 x_crop: str = '', required_vars: Optional[set] = None,
                 average_over_timesteps: bool = False):
        self.folder_path = folder_path
        self.cases = cases
        self.timesteps = list(timesteps)
        self.data_types = data_types
        self.required_vars = required_vars
        self.average_z = average_z
        self.average_x = average_x
        self.average_over_timesteps = average_over_timesteps
        # Nested structure: {case_timestep: {variable: array}}
        self.data: Dict[str, Dict[str, np.ndarray]] = {}
        self.grid_info: Dict = {}
        self.y_coords: Optional[np.ndarray] = None
        self.x_coords: Optional[np.ndarray] = None
        self._full_x_coords: Optional[np.ndarray] = None
        self.z_coords: Optional[np.ndarray] = None
        try:
            self.x_crop: Optional[Tuple[float, float]] = ut.parse_x_crop_input(x_crop)
        except ValueError:
            self.x_crop = None
            print("Invalid x_crop in config. Using full x-range.")

    def _apply_x_crop_to_arrays(self, arrays_for_key: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply optional x-crop once to all compatible arrays for a case/timestep."""
        x_ref = self._full_x_coords if self._full_x_coords is not None else self.x_coords
        if self.x_crop is None or x_ref is None:
            return arrays_for_key

        cropped_arrays: Dict[str, np.ndarray] = {}
        cropped_x_coords = None

        for name, values in arrays_for_key.items():
            if values.ndim >= 2 and values.shape[-1] in (len(x_ref), len(x_ref) - 1):
                cropped_values, candidate_x = ut.apply_x_crop(values, x_ref, self.x_crop)
                cropped_arrays[name] = cropped_values
                if cropped_x_coords is None and candidate_x is not None:
                    cropped_x_coords = candidate_x
            else:
                cropped_arrays[name] = values

        if cropped_x_coords is not None:
            self.x_coords = cropped_x_coords

        return cropped_arrays

    def load_all(self) -> None:
        """Load all XDMF files for all cases and timesteps"""
        for case in self.cases:
            for timestep in self.timesteps:
                self._load_single(case, timestep)
            if self.average_over_timesteps:
                self._average_over_timesteps(case, self.timesteps)
        if self.average_over_timesteps:
            self.timesteps = ['avg']

    def _load_single(self, case: str, timestep: str) -> None:
        """Load XDMF files for a single case and timestep"""
        key = f"{case}_{timestep}"

        # Get file paths for this case/timestep — always the fixed 'zi1'
        # spanwise-average tsp_avg naming (see utils.visu_file_paths).
        file_names = ut.visu_file_paths(self.folder_path, case, timestep)

        # Check which files exist
        existing_files = [f for f in file_names if os.path.isfile(f)]

        if not existing_files:
            print(f'No .xdmf files found for {case}, {timestep}')
            return

        # Load the XDMF files with averaging flags
        arrays, grid_info = ut.xdmf_reader_wrapper(
            file_names, case=case, timestep=timestep,
            data_types=self.data_types,
            required_vars=self.required_vars,
            average_z=self.average_z, average_x=self.average_x
        )

        # xdmf_reader_wrapper always returns the {case_timestep: {...}} key, even
        # when nothing was read — so test the inner dict, not just the key.
        if arrays and arrays.get(key):
            # Store grid info if not already stored
            if not self.grid_info and grid_info:
                self.grid_info = grid_info

            # Compute cell-centre coordinates from grid node coordinates
            sample = next(iter(arrays[key].values()))

            if self.y_coords is None and grid_info:
                y_nodes = grid_info.get('grid_y', None)
                if y_nodes is not None:
                    if sample.ndim == 3:
                        ny = sample.shape[1]  # (nz, ny, nx)
                    else:
                        ny = sample.shape[0]  # (ny, nx) or (ny,)
                    if len(y_nodes) == ny + 1:
                        self.y_coords = 0.5 * (y_nodes[:-1] + y_nodes[1:])
                    elif len(y_nodes) == ny:
                        self.y_coords = y_nodes.copy()
                    else:
                        self.y_coords = np.linspace(-1, 1, ny)

            if self.x_coords is None and grid_info:
                x_nodes = grid_info.get('grid_x', None)
                if x_nodes is not None and sample.ndim >= 2:
                    if sample.ndim == 3:
                        nx = sample.shape[2]  # (nz, ny, nx)
                    else:
                        nx = sample.shape[1]  # (ny, nx)
                    if len(x_nodes) == nx + 1:
                        self.x_coords = 0.5 * (x_nodes[:-1] + x_nodes[1:])
                    elif len(x_nodes) == nx:
                        self.x_coords = x_nodes.copy()
                    else:
                        self.x_coords = np.linspace(x_nodes.min(), x_nodes.max(), nx)
                    self._full_x_coords = self.x_coords

            if self.z_coords is None and grid_info:
                z_nodes = grid_info.get('grid_z', None)
                if z_nodes is not None and sample.ndim == 3:
                    nz = sample.shape[0]  # (nz, ny, nx)
                    if len(z_nodes) == nz + 1:
                        self.z_coords = 0.5 * (z_nodes[:-1] + z_nodes[1:])
                    elif len(z_nodes) == nz:
                        self.z_coords = z_nodes.copy()
                    else:
                        self.z_coords = np.linspace(z_nodes.min(), z_nodes.max(), nz)

            # Apply optional x-crop once at load stage
            self.data[key] = self._apply_x_crop_to_arrays(dict(arrays[key]))
            print(f"Loaded {len(self.data[key])} variables from XDMF files for {case}, {timestep}")
        else:
            print(f'No {self.data_types} arrays extracted from XDMF files for '
                  f'{case}, {timestep} — skipping this timestep')

    def _average_over_timesteps(self, case: str, timesteps: List[str]) -> None:
        """Average loaded variable arrays across timesteps for a given case and store under '{case}_avg'."""
        keys = [f"{case}_{ts}" for ts in timesteps if f"{case}_{ts}" in self.data]
        if not keys:
            print(f"No loaded data to average for case '{case}'")
            return

        all_vars: set = set()
        for key in keys:
            all_vars.update(self.data[key].keys())

        avg_data: Dict[str, np.ndarray] = {}
        for var in all_vars:
            arrays = [self.data[key][var] for key in keys if var in self.data[key]]
            if arrays:
                avg_data[var] = np.mean(np.stack(arrays, axis=0), axis=0)

        self.data[f"{case}_avg"] = avg_data
        print(f"Averaged {len(keys)} timestep(s) for case '{case}' -> '{case}_avg' ({len(avg_data)} variables)")

    def get(self, case: str, variable: str, timestep: str) -> Optional[np.ndarray]:
        """Get specific data array"""
        key = f"{case}_{timestep}"
        if key in self.data and variable in self.data[key]:
            return self.data[key][variable]
        return None

    def has(self, case: str, variable: str, timestep: str) -> bool:
        """Check if data exists"""
        key = f"{case}_{timestep}"
        return key in self.data and variable in self.data[key]

    def keys(self):
        """Return all data keys (case_timestep format)"""
        return self.data.keys()

    def get_variables(self, case: str, timestep: str) -> List[str]:
        """Get list of all variables available for a case/timestep"""
        key = f"{case}_{timestep}"
        if key in self.data:
            return list(self.data[key].keys())
        return []

    def get_raw_dict(self, case: str, timestep: str) -> Optional[Dict[str, np.ndarray]]:
        """Get the full data dictionary for a case/timestep (for TKE budget computation)."""
        key = f"{case}_{timestep}"
        return self.data.get(key, None)

class ReferenceData:
    """Manages all reference datasets"""

    # Reference data file paths — resolved relative to this file's own
    # location (not the process's current working directory), since the
    # GUI can be launched from a different cwd than the repo root and
    # np.loadtxt would otherwise silently fail to find them.
    _REF_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Reference_Data')

    REF_MKM180_MEANS_PATH = os.path.join(_REF_DATA_DIR, 'MKM180_profiles', 'chan180.means')
    REF_MKM180_REYSTRESS_PATH = os.path.join(_REF_DATA_DIR, 'MKM180_profiles', 'chan180.reystress')

    NK_REF_PATHS = {
        'ref_NK_Ha_6': os.path.join(_REF_DATA_DIR, 'Noguchi&Kasagi_mhd_ref_data', 'thtlabs_Ha_6_turb.txt'),
        'ref_NK_Ha_4': os.path.join(_REF_DATA_DIR, 'Noguchi&Kasagi_mhd_ref_data', 'thtlabs_Ha_4_turb.txt'),
        'ref_NK_uu12_Ha_6': os.path.join(_REF_DATA_DIR, 'Noguchi&Kasagi_mhd_ref_data', 'thtlabs_Ha_6_uv_rms.txt'),
        'ref_NK_uu12_Ha_4': os.path.join(_REF_DATA_DIR, 'Noguchi&Kasagi_mhd_ref_data', 'thtlabs_Ha_4_uv_rms.txt'),
    }

    #REF_XCOMP_HA_6_PATH = os.path.join(_REF_DATA_DIR, 'XCompact3D_mhd_validation', 'u_prime_sq.txt')

    def __init__(self, config: Config):
        self.config = config
        self.mkm180_stats: Optional[Dict[str, np.ndarray]] = None
        self.mkm180_y: Optional[np.ndarray] = None
        self.mkm180_y_plus: Optional[np.ndarray] = None

        self.NK_H4_stats: Optional[Dict[str, np.ndarray]] = None
        self.NK_H6_stats: Optional[Dict[str, np.ndarray]] = None
        self.NK_ref_y_H4: Optional[np.ndarray] = None
        self.NK_ref_y_H6: Optional[np.ndarray] = None
        self.NK_ref_y_uu12_H4: Optional[np.ndarray] = None
        self.NK_ref_y_uu12_H6: Optional[np.ndarray] = None

        self.xcomp_H6_stats: Optional[Dict[str, np.ndarray]] = None
        self.xcomp_yplus_uu11_H6: Optional[np.ndarray] = None

    def load_all(self) -> None:
        """Load all enabled reference datasets"""
        if self.config.mkm180_ch_ref_on:
            self._load_mkm180()

        if self.config.mhd_NK_ref_on:
            self._load_noguchi_kasagi()

        #if self.config.mhd_XCompact_ref_on:
        #    self._load_xcompact()

    def _load_mkm180(self) -> None:
        """Load MKM180 reference data"""
        try:
            ref_means = np.loadtxt(self.REF_MKM180_MEANS_PATH)
            ref_reystress = np.loadtxt(self.REF_MKM180_REYSTRESS_PATH)

            self.mkm180_y = ref_means[:, 0]
            self.mkm180_y_plus = ref_means[:, 1]

            self.mkm180_stats = {
                'ux_velocity': ref_means[:, 2],
                'u_prime_sq': ref_reystress[:, 2],
                'v_prime_sq': ref_reystress[:, 3],
                'w_prime_sq': ref_reystress[:, 4],
                'u_prime_v_prime': ref_reystress[:, 5]
            }
            print("mkm180 reference data loaded successfully.")
        except Exception as e:
            print(f"mkm180 reference is disabled or required data is missing: {e}")

    def _load_noguchi_kasagi(self) -> None:
        """Load Noguchi & Kasagi MHD reference data"""
        try:
            NK_data = {}
            for key, path in self.NK_REF_PATHS.items():
                NK_data[key] = np.loadtxt(path)

            self.NK_ref_y_H4 = NK_data['ref_NK_Ha_4'][:, 1]
            self.NK_ref_y_H6 = NK_data['ref_NK_Ha_6'][:, 1]
            self.NK_ref_y_uu12_H4 = NK_data['ref_NK_uu12_Ha_4'][:, 1]
            self.NK_ref_y_uu12_H6 = NK_data['ref_NK_uu12_Ha_6'][:, 1]

            self.NK_H4_stats = {
                'ux_velocity': NK_data['ref_NK_Ha_4'][:, 2] * 1.02169,
                'u_prime_sq': np.square(NK_data['ref_NK_Ha_4'][:, 3]),
                'u_prime_v_prime': -1 * NK_data['ref_NK_uu12_Ha_4'][:, 2],
                'v_prime_sq': np.square(NK_data['ref_NK_Ha_4'][:, 4]),
                'w_prime_sq': np.square(NK_data['ref_NK_Ha_4'][:, 5])
            }

            self.NK_H6_stats = {
                'ux_velocity': NK_data['ref_NK_Ha_6'][:, 2],
                'u_prime_sq': np.square(NK_data['ref_NK_Ha_6'][:, 3]),
                'u_prime_v_prime': -1 * NK_data['ref_NK_uu12_Ha_6'][:, 2],
                'v_prime_sq': np.square(NK_data['ref_NK_Ha_6'][:, 4]),
                'w_prime_sq': np.square(NK_data['ref_NK_Ha_6'][:, 5])
            }
            print("Noguchi & Kasagi MHD channel reference data loaded successfully.")
        except Exception as e:
            print(f"NK mhd reference is disabled or required data is missing: {e}")

# =====================================================================================================================================================
# FLOW PROFILE CLASSES
# =====================================================================================================================================================

class Profiles(ABC):
    """Abstract base class for velocity and temperature profiles"""

    def __init__(self, name: str, label: str, required_quantities: List[str]):
        self.name = name
        self.label = label
        self.required_quantities = required_quantities
        self.raw_results: Dict[Tuple[str, str], np.ndarray] = {}
        self.processed_results: Dict[Tuple[str, str], np.ndarray] = {}

    @abstractmethod
    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute the profile from required data"""
        pass

    def compute_for_case(self, case: str, timestep: str, data_loader: TurbulenceTextData) -> bool:
        """Compute profile for a specific case and timestep"""
        # Gather required data
        data_dict = {}
        for quantity in self.required_quantities:
            if not data_loader.has(case, quantity, timestep):
                print(f"Missing {quantity} data for {self.name} calculation: {case}, {timestep}")
                return False
            data_dict[quantity] = data_loader.get(case, quantity, timestep)

        # Compute profile
        result = self.compute(data_dict)
        self.raw_results[(case, timestep)] = result
        return True

    def get_half_domain(self, values: np.ndarray) -> np.ndarray:
        """Get first half of domain"""
        return values[:(len(values)//2)]


class StreamwiseVelocity(Profiles):
    """Streamwise velocity profile (u1)"""

    def __init__(self):
        super().__init__('ux_velocity', 'Streamwise Velocity', ['u1'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.read_profile(data_dict['u1'])


class WallNormalVelocity(Profiles):
    """Wall-normal velocity profile (u2)"""

    def __init__(self):
        super().__init__('uy_velocity', 'Wall-Normal Velocity', ['u2'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.read_profile(data_dict['u2'])


class SpanwiseVelocity(Profiles):
    """Spanwise velocity profile (u3)"""

    def __init__(self):
        super().__init__('uz_velocity', 'Spanwise Velocity', ['u3'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.read_profile(data_dict['u3'])


class FrictionCoefficient(Profiles):
    """Wall friction coefficient profile (x-direction only)"""

    def __init__(self, cases: List[str], Re: List[float]):
        super().__init__('coeff_friction', 'Wall Friction Coefficient', ['u1'])
        self.cases = cases
        self.Re = Re
        self.x_profile_only = True

    def _get_ref_re(self, case: str) -> float:
        return float(self.Re[self.cases.index(case)]) if len(self.Re) > 1 else float(self.Re[0])

    def compute_for_case(self, case: str, timestep: str, data_loader) -> bool:
        """Compute friction coefficient from the wall gradient of ``u1``."""
        if not data_loader.has(case, 'u1', timestep):
            print(f"Missing u1 data for {self.name} calculation: {case}, {timestep}")
            return False

        data_dict = {'u1': data_loader.get(case, 'u1', timestep)}
        ref_Re = self._get_ref_re(case)
        y_coords = getattr(data_loader, 'y_coords', None)
        result = self.compute(data_dict, ref_Re, y_coords)
        self.raw_results[(case, timestep)] = result
        return True

    def compute(self, data_dict: Dict[str, np.ndarray], ref_Re: float = 1.0, y_coords: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute friction coefficient from near-wall interpolated velocity points."""
        u1_data = data_dict['u1']
        if u1_data.ndim == 3:
            # Keep x-profile output by averaging z before wall operation.
            u1_data = u1_data.mean(axis=2)
        tau_w = op.compute_wall_shear_stress_from_velocity(u1_data, ref_Re, y_coords=y_coords)
        return op.compute_wall_friction_coeff(tau_w, ref_rho=1.0, ref_bulk_velocity=1.0)


class TurbulentKineticEnergy(Profiles):
    """Turbulent Kinetic Energy (TKE)"""

    def __init__(self):
        super().__init__('TKE', 'Turbulent Kinetic Energy', ['u1', 'u2', 'u3', 'uu11', 'uu22', 'uu33'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        u_prime_sq = op.compute_normal_stress(data_dict['u1'], data_dict['uu11'])
        v_prime_sq = op.compute_normal_stress(data_dict['u2'], data_dict['uu22'])
        w_prime_sq = op.compute_normal_stress(data_dict['u3'], data_dict['uu33'])
        return op.compute_tke(u_prime_sq, v_prime_sq, w_prime_sq)


class PeakGrowth(Profiles):
    """Base for 'peak value at each x' statistics: the wall-normal maximum
    of a quantity, plotted against x to show how it progresses down the
    channel.
    
    """

    def __init__(self, name: str, label: str, required_quantities: List[str],
                 half_channel_side: Optional[str] = None):
        super().__init__(name, label, required_quantities)
        self.x_profile_only = True
        self.half_channel_side = half_channel_side


class PeakTKE(PeakGrowth):
    """Peak (wall-normal maximum) TKE at each streamwise location."""

    def __init__(self, half_channel_side: Optional[str] = None):
        super().__init__('tke_peak', 'Max. k', ['u1', 'u2', 'u3', 'uu11', 'uu22', 'uu33'],
                         half_channel_side)

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        u_prime_sq = op.compute_normal_stress(data_dict['u1'], data_dict['uu11'])
        v_prime_sq = op.compute_normal_stress(data_dict['u2'], data_dict['uu22'])
        w_prime_sq = op.compute_normal_stress(data_dict['u3'], data_dict['uu33'])
        tke = op.compute_tke(u_prime_sq, v_prime_sq, w_prime_sq)
        return op.compute_peak_over_y(tke, half=self.half_channel_side)


class PeakReynoldsStress(PeakGrowth):
    """Peak (wall-normal maximum) Reynolds normal stress component at each
    streamwise location — one class parameterized by which velocity/stress
    fields to use, instantiated once per component (u'u', v'v', w'w').
    """

    def __init__(self, name: str, label: str, velocity_key: str, stress_key: str,
                 half_channel_side: Optional[str] = None):
        super().__init__(f'{name}_max', f'Max. {label}', [velocity_key, stress_key],
                         half_channel_side)
        self._velocity_key = velocity_key
        self._stress_key = stress_key

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.compute_peak_over_y(
            op.compute_normal_stress(data_dict[self._velocity_key], data_dict[self._stress_key]),
            half=self.half_channel_side,
        )


class MeanVorticity(Profiles):
    """Mean vorticity profile: curl of the time-averaged velocity field
    (u1, u2, u3), via op.compute_vorticity.

    compute_vorticity needs the full, unaveraged (nz, ny, nx) field to
    differentiate — so unlike the other Profiles stats, this requires
    average_z_direction and average_x_direction both off.
    """

    def __init__(self, component: str, average_z: bool, average_x: bool):
        super().__init__('vorticity_mean', f"Vorticity ($\\omega_{component}$)", ['u1', 'u2', 'u3'])
        self.component = component
        self.average_z = average_z
        self.average_x = average_x

    def compute_for_case(self, case: str, timestep: str, data_loader) -> bool:
        if self.average_z or self.average_x:
            print("Vorticity requires 'Average z direction' and 'Average x direction' "
                  "to both be off (it needs the full 3D field to differentiate); skipping.")
            return False

        data_dict = {}
        for quantity in self.required_quantities:
            if not data_loader.has(case, quantity, timestep):
                print(f"Missing {quantity} data for {self.name} calculation: {case}, {timestep}")
                return False
            data_dict[quantity] = data_loader.get(case, quantity, timestep)

        grid_info = getattr(data_loader, 'grid_info', None)
        result = self.compute(data_dict, grid_info)
        if result is None:
            return False
        self.raw_results[(case, timestep)] = result
        return True

    def compute(self, data_dict: Dict[str, np.ndarray], grid_info: Optional[Dict] = None) -> Optional[np.ndarray]:
        return op.compute_vorticity(
            {'qx_ccc': data_dict['u1'], 'qy_ccc': data_dict['u2'], 'qz_ccc': data_dict['u3']},
            grid_info, self.component,
        )


class VorticityFluctuationRMS(Profiles):
    """RMS turbulent vorticity fluctuation profile.

    Uses the same Var(X) = <X^2> - <X>^2 identity as compute_normal_stress,
    applied through the curl operator: the mean vorticity comes from the
    curl of the mean velocity (u1, u2, u3), and <omega^2> comes from the
    same curl formula applied to the diagonal Reynolds-stress components
    (uu11, uu22, uu33) in place of (u1, u2, u3) — see
    op.compute_vorticity_fluctuation_rms. Same full-3D requirement as
    MeanVorticity (compute_vorticity needs unaveraged data).
    """

    def __init__(self, component: str, average_z: bool, average_x: bool):
        super().__init__(
            'vorticity_rms', f"Vorticity RMS ($\\omega_{component}'$)",
            ['u1', 'u2', 'u3', 'uu11', 'uu22', 'uu33'],
        )
        self.component = component
        self.average_z = average_z
        self.average_x = average_x

    def compute_for_case(self, case: str, timestep: str, data_loader) -> bool:
        if self.average_z or self.average_x:
            print("Vorticity RMS requires 'Average z direction' and 'Average x direction' "
                  "to both be off (it needs the full 3D field to differentiate); skipping.")
            return False

        data_dict = {}
        for quantity in self.required_quantities:
            if not data_loader.has(case, quantity, timestep):
                print(f"Missing {quantity} data for {self.name} calculation: {case}, {timestep}")
                return False
            data_dict[quantity] = data_loader.get(case, quantity, timestep)

        grid_info = getattr(data_loader, 'grid_info', None)
        result = self.compute(data_dict, grid_info)
        if result is None:
            return False
        self.raw_results[(case, timestep)] = result
        return True

    def compute(self, data_dict: Dict[str, np.ndarray], grid_info: Optional[Dict] = None) -> Optional[np.ndarray]:
        return op.compute_vorticity_fluctuation_rms(data_dict, grid_info, self.component)

# =====================================================================================================================================================
# MHD PROFILE CLASSES
# =====================================================================================================================================================

class MeanCurrentDensityj1(Profiles):
    """Mean streamwise electric current density profile (j1)"""

    def __init__(self):
        super().__init__('j1_mean', '$j_x$', ['j1'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.read_profile(data_dict['j1'])


class MeanCurrentDensityj2(Profiles):
    """Mean wall-normal electric current density profile (j2)"""

    def __init__(self):
        super().__init__('j2_mean', '$j_y$', ['j2'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.read_profile(data_dict['j2'])


class MeanCurrentDensityj3(Profiles):
    """Mean spanwise electric current density profile (j3)"""

    def __init__(self):
        super().__init__('j3_mean', '$j_z$', ['j3'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.read_profile(data_dict['j3'])


class CurrentDensityRMSj1(Profiles):
    """RMS streamwise electric current density fluctuation sqrt(<j1'j1'>)."""

    def __init__(self):
        super().__init__('j1_rms', "$j_x'_{rms}$", ['j1', 'jj11'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return np.sqrt(np.clip(op.compute_normal_stress(data_dict['j1'], data_dict['jj11']), 0, None))


class CurrentDensityRMSj2(Profiles):
    """RMS wall-normal electric current density fluctuation sqrt(<j2'j2'>)."""

    def __init__(self):
        super().__init__('j2_rms', "$j_y'_{rms}$", ['j2', 'jj22'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return np.sqrt(np.clip(op.compute_normal_stress(data_dict['j2'], data_dict['jj22']), 0, None))


class CurrentDensityRMSj3(Profiles):
    """RMS spanwise electric current density fluctuation sqrt(<j3'j3'>)."""

    def __init__(self):
        super().__init__('j3_rms', "$j_z'_{rms}$", ['j3', 'jj33'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return np.sqrt(np.clip(op.compute_normal_stress(data_dict['j3'], data_dict['jj33']), 0, None))


class MeanLorentzForcex(Profiles):
    """Mean Lorentz body force, x-component: F_x = N*(J x B)_x."""

    def __init__(self, mag_field_direction: List[float], stuart_number: float):
        super().__init__('lorentz_force_x', 'Lorentz Force ($F_x$)', ['j1', 'j2', 'j3'])
        self.mag_field_direction = mag_field_direction
        self.stuart_number = stuart_number

    def compute_for_case(self, case: str, timestep: str, data_loader) -> bool:
        data_dict = {}
        for quantity in self.required_quantities:
            if not data_loader.has(case, quantity, timestep):
                print(f"Missing {quantity} data for {self.name} calculation: {case}, {timestep}")
                return False
            data_dict[quantity] = data_loader.get(case, quantity, timestep)

        result = self.compute(data_dict)
        if result is None:
            print(f"Could not compute {self.name} for {case}, {timestep} (stuart_number=0?).")
            return False
        self.raw_results[(case, timestep)] = result
        return True

    def compute(self, data_dict: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        force_dict = {'j': [data_dict['j1'], data_dict['j2'], data_dict['j3']]}
        result = op.compute_lorentz_force(self.mag_field_direction, self.stuart_number, force_dict)
        return result['lorentz_1']


class MeanLorentzForcey(Profiles):
    """Mean Lorentz body force, y-component: F_y = N*(J x B)_y."""

    def __init__(self, mag_field_direction: List[float], stuart_number: float):
        super().__init__('lorentz_force_y', 'Lorentz Force ($F_y$)', ['j1', 'j2', 'j3'])
        self.mag_field_direction = mag_field_direction
        self.stuart_number = stuart_number

    def compute_for_case(self, case: str, timestep: str, data_loader) -> bool:
        data_dict = {}
        for quantity in self.required_quantities:
            if not data_loader.has(case, quantity, timestep):
                print(f"Missing {quantity} data for {self.name} calculation: {case}, {timestep}")
                return False
            data_dict[quantity] = data_loader.get(case, quantity, timestep)

        result = self.compute(data_dict)
        if result is None:
            print(f"Could not compute {self.name} for {case}, {timestep} (stuart_number=0?).")
            return False
        self.raw_results[(case, timestep)] = result
        return True

    def compute(self, data_dict: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        force_dict = {'j': [data_dict['j1'], data_dict['j2'], data_dict['j3']]}
        result = op.compute_lorentz_force(self.mag_field_direction, self.stuart_number, force_dict)
        return result['lorentz_2']


class MeanLorentzForcez(Profiles):
    """Mean Lorentz body force, z-component: F_z = N*(J x B)_z."""

    def __init__(self, mag_field_direction: List[float], stuart_number: float):
        super().__init__('lorentz_force_z', 'Lorentz Force ($F_z$)', ['j1', 'j2', 'j3'])
        self.mag_field_direction = mag_field_direction
        self.stuart_number = stuart_number

    def compute_for_case(self, case: str, timestep: str, data_loader) -> bool:
        data_dict = {}
        for quantity in self.required_quantities:
            if not data_loader.has(case, quantity, timestep):
                print(f"Missing {quantity} data for {self.name} calculation: {case}, {timestep}")
                return False
            data_dict[quantity] = data_loader.get(case, quantity, timestep)

        result = self.compute(data_dict)
        if result is None:
            print(f"Could not compute {self.name} for {case}, {timestep} (stuart_number=0?).")
            return False
        self.raw_results[(case, timestep)] = result
        return True

    def compute(self, data_dict: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        force_dict = {'j': [data_dict['j1'], data_dict['j2'], data_dict['j3']]}
        result = op.compute_lorentz_force(self.mag_field_direction, self.stuart_number, force_dict)
        return result['lorentz_3']

# =====================================================================================================================================================
# THERMO PROFILE CLASSES
# =====================================================================================================================================================

class Temperature(Profiles):
    """Temperature profile"""

    def __init__(self, norm_temp_by_ref_temp: bool, ref_temps: List[float], cases: List[str]):
        super().__init__('temperature', 'Temperature', ['T'])
        self.norm_temp_by_ref_temp = norm_temp_by_ref_temp
        self.ref_temps = ref_temps
        self.cases = cases

    def compute_for_case(self, case: str, timestep: str, data_loader) -> bool:
        """Compute temperature profile with per-case ref_temp."""
        data_dict = {}
        for quantity in self.required_quantities:
            if not data_loader.has(case, quantity, timestep):
                print(f"Missing {quantity} data for {self.name} calculation: {case}, {timestep}")
                return False
            data_dict[quantity] = data_loader.get(case, quantity, timestep)

        ref_temp = float(self.ref_temps[self.cases.index(case)]) if len(self.ref_temps) > 1 else float(self.ref_temps[0])
        result = op.dimensionalize_temperature(op.read_profile(data_dict['T']), ref_temp, self.norm_temp_by_ref_temp)

        self.raw_results[(case, timestep)] = result
        return True

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.dimensionalize_temperature(
            op.read_profile(data_dict['T']),
            float(self.ref_temps[0]),
            self.norm_temp_by_ref_temp,
        )


class HeatTransferCoefficient(Profiles):
    """Wall heat transfer coefficient profile (x-direction only)."""

    def __init__(self, cases: List[str], ref_temp: List[float], wall_heat_flux: List[float],
                 working_fluid: str):
        super().__init__('heat_transfer_coeff', 'Heat Transfer Coefficient', ['T'])
        self.cases = cases
        self.ref_temp = ref_temp
        self.wall_heat_flux = wall_heat_flux
        self.x_profile_only = True
        self.fluid = ut.get_fluid_properties(working_fluid)

    def _case_value(self, values: List[float], case: str) -> float:
        return float(values[self.cases.index(case)]) if len(values) > 1 else float(values[0])

    def _compute_h_profile(self, temp: np.ndarray, ref_temp: float, fuh: np.ndarray, fu: np.ndarray,
                           heat_flux: float, y_coords: Optional[np.ndarray]) -> np.ndarray:
        """Compute heat transfer coefficient profile from thermo variables."""
        return np.asarray(op.compute_wall_heat_transfer_coeff(
            heat_flux,
            temp,
            ref_temp,
            fuh,
            fu,
            y_coords=y_coords,
            fluid=self.fluid,
        ))

    def compute_for_case(self, case: str, timestep: str, data_loader) -> bool:
        if not data_loader.has(case, 'T', timestep):
            print(f"Missing T data for {self.name} calculation: {case}, {timestep}")
            return False

        # Gather all required thermo variables
        temp_data = data_loader.get(case, 'T', timestep)
        fuh_data = data_loader.get(case, 'fuh1', timestep)
        fu_data = data_loader.get(case, 'fu1', timestep)

        if fuh_data is None or fu_data is None:
            print(f"Missing enthalpy data (fuh, fu) for {self.name} calculation: {case}, {timestep}")
            return False

        heat_flux = self._case_value(self.wall_heat_flux, case)
        ref_temp = self._case_value(self.ref_temp, case)
        y_coords = getattr(data_loader, 'y_coords', None)
        self.raw_results[(case, timestep)] = self._compute_h_profile(
            temp_data,
            ref_temp,
            fuh_data,
            fu_data,
            heat_flux,
            y_coords,
        )
        return True

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.read_profile(data_dict['T'])


class NusseltNumber(HeatTransferCoefficient):
    """Wall Nusselt number profile (x-direction only)."""

    def __init__(self, cases: List[str], ref_temp: List[float], wall_heat_flux: List[float],
                 ref_length: List[float], working_fluid: str):
        super().__init__(cases, ref_temp, wall_heat_flux, working_fluid)
        self.name = 'nusselt_number'
        self.label = 'Local Nusselt Number'
        self.ref_length = ref_length
        self.fluid = ut.get_fluid_properties(working_fluid)

    def compute_for_case(self, case: str, timestep: str, data_loader) -> bool:
        if not data_loader.has(case, 'T', timestep):
            print(f"Missing T data for {self.name} calculation: {case}, {timestep}")
            return False

        # Gather all required thermo variables
        temp_data = data_loader.get(case, 'T', timestep)
        fuh_data = data_loader.get(case, 'fuh1', timestep)
        fu_data = data_loader.get(case, 'fu1', timestep)

        if fuh_data is None or fu_data is None:
            print(f"Missing enthalpy data (fuh, fu) for {self.name} calculation: {case}, {timestep}")
            return False

        heat_flux = self._case_value(self.wall_heat_flux, case)
        ref_len = self._case_value(self.ref_length, case)
        ref_temp = self._case_value(self.ref_temp, case)
        y_coords = getattr(data_loader, 'y_coords', None)
        h_profile = self._compute_h_profile(temp_data, ref_temp, fuh_data, fu_data, heat_flux, y_coords)
        k_ref = float(self.fluid.thermal_conductivity(ref_temp)) # this is wrong, should be based on bulk temp
        fluid_props = {'k': k_ref}

        if np.ndim(h_profile) == 0:
            nu_profile = np.asarray(op.compute_wall_Nusselt_number(float(h_profile), ref_len, fluid_props))
        else:
            nu_profile = np.asarray([
                op.compute_wall_Nusselt_number(float(h), ref_len, fluid_props)
                for h in np.ravel(h_profile)
            ])

        self.raw_results[(case, timestep)] = nu_profile
        return True

class TurbulentPrandtlNumber(Profiles):
    """Turbulent Prandtl number: Pr_t = (nu_t / alpha_t)"""

    def __init__(self):
        super().__init__('turb_prandtl', 'Turbulent Prandtl Number', ['u1', 'u2', 'uu12', 'T', 'Tu2'])

    def compute_for_case(self, case: str, timestep: str, data_loader) -> bool:
        for quantity in self.required_quantities:
            if not data_loader.has(case, quantity, timestep):
                print(f"Missing {quantity} data for {self.name} calculation: {case}, {timestep}")
                return False

        data_dict = {q: data_loader.get(case, q, timestep) for q in self.required_quantities}
        y_coords = getattr(data_loader, 'y_coords', None)
        if y_coords is None:
            u1 = data_dict['u1']
            if u1.ndim == 2 and u1.shape[1] == 3:
                y_coords = u1[:, 1]
        result = self.compute(data_dict, y_coords=y_coords)
        self.raw_results[(case, timestep)] = result
        return True

    def compute(self, data_dict: Dict[str, np.ndarray], y_coords: Optional[np.ndarray] = None) -> np.ndarray:
        return op.compute_turb_Prandtl_number(
            data_dict['u1'], data_dict['u2'], data_dict['uu12'],
            data_dict['T'], data_dict['Tu2'], y_coords
        )


# =====================================================================================================================================================
# REYNOLDS STRESS CLASSES
# =====================================================================================================================================================

class ReStresses(ABC):
    """Abstract base class for Reynolds stress statistics"""

    def __init__(self, name: str, label: str, required_quantities: List[str]):
        self.name = name
        self.label = label
        self.required_quantities = required_quantities
        self.raw_results: Dict[Tuple[str, str], np.ndarray] = {}
        self.processed_results: Dict[Tuple[str, str], np.ndarray] = {}

    @abstractmethod
    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute the statistic from required data"""
        pass

    def compute_for_case(self, case: str, timestep: str, data_loader: TurbulenceTextData) -> bool:
        """Compute statistic for a specific case and timestep"""
        # Gather required data
        data_dict = {}
        for quantity in self.required_quantities:
            if not data_loader.has(case, quantity, timestep):
                print(f"Missing {quantity} data for {self.name} calculation: {case}, {timestep}")
                return False
            data_dict[quantity] = data_loader.get(case, quantity, timestep)

        # Compute statistic
        result = self.compute(data_dict)
        self.raw_results[(case, timestep)] = result
        return True

    def get_half_domain(self, values: np.ndarray) -> np.ndarray:
        """Get first half of domain"""
        return values[:(len(values)//2)]


class ReynoldsStressuu11(ReStresses):
    """Reynolds stress u'u'"""

    def __init__(self):
        super().__init__('u_prime_sq', "<u'u'>", ['u1', 'uu11'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.compute_normal_stress(data_dict['u1'], data_dict['uu11'])


class ReynoldsStressuu12(ReStresses):
    """Reynolds stress u'v'"""

    def __init__(self):
        super().__init__('u_prime_v_prime', "<u'v'>", ['u1', 'u2', 'uu12'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.compute_shear_stress(data_dict['u1'], data_dict['u2'], data_dict['uu12'])


class ReynoldsStressuu22(ReStresses):
    """Reynolds stress v'v'"""

    def __init__(self):
        super().__init__('v_prime_sq', "<v'v'>", ['u2', 'uu22'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.compute_normal_stress(data_dict['u2'], data_dict['uu22'])


class ReynoldsStressuu23(ReStresses):
    """Reynolds Stress v'w'"""

    def __init__(self):
        super().__init__('v_prime_w_prime', "<v'w'>", ['u2','u3', 'uu23'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.compute_shear_stress(data_dict['u2'], data_dict['u3'], data_dict['uu23'])


class ReynoldsStressuu33(ReStresses):
    """Reynolds stress w'w'"""

    def __init__(self):
        super().__init__('w_prime_sq', "<w'w'>", ['u3', 'uu33'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.compute_normal_stress(data_dict['u3'], data_dict['uu33'])


# =====================================================================================================================================================
# REYNOLDS STRESS BUDGET CLASSES
# =====================================================================================================================================================

class Budget(ABC):
    """Abstract base class for Reynolds stress budget terms"""

    def __init__(self, name: str, label: str, required_quantities: List[str]):
        self.name = name
        self.label = label
        self.required_quantities = required_quantities
        self.raw_results: Dict[Tuple[str, str], np.ndarray] = {}
        self.processed_results: Dict[Tuple[str, str], np.ndarray] = {}

    @abstractmethod
    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute the budget term from required data"""
        pass

    def compute_for_case(self, case: str, timestep: str, data_loader: TurbulenceTextData) -> bool:
        """Compute budget term for a specific case and timestep"""
        # Gather required data
        data_dict = {}
        for quantity in self.required_quantities:
            if not data_loader.has(case, quantity, timestep):
                print(f"Missing {quantity} data for {self.name} calculation: {case}, {timestep}")
                return False
            data_dict[quantity] = data_loader.get(case, quantity, timestep)

        # Compute budget term
        result = self.compute(data_dict)
        self.raw_results[(case, timestep)] = result
        return True

    def get_half_domain(self, values: np.ndarray) -> np.ndarray:
        """Get first half of domain"""
        return values[:(len(values)//2)]


class BudgetComputer:
    """
    Computes all budget terms by calling op.compute_budget_components once
    and then extracting individual terms via op.compute_* functions.
    """

    # Registry mapping config flags to (op function, key in result dict)
    TERM_REGISTRY = {
        'production_on':          ('production',          'Production'),
        'dissipation_on':         ('dissipation',         'Dissipation'),
        'convection_on':          ('mean_convection',     'Mean Convection'),
        'viscous_diffusion_on':   ('viscous_diffusion',   'Viscous Diffusion'),
        'pressure_transport_on':  ('pressure_transport',  'Pressure Transport'),
        'turbulent_diffusion_on': ('turbulent_convection','Turbulent Diffusion'),
        'pressure_strain_on':     ('pressure_strain',     'Pressure Strain'),
    }

    def __init__(self, config: Config):
        self.config = config
        self.uiuj = config.re_stress_component
        self.average_z = config.average_z_direction
        self.average_x = config.average_x_direction

        # Determine which terms are enabled
        self.enabled_terms = []
        if config.re_stress_budget_on:
            self.enabled_terms = [
                (flag, term_name, label)
                for flag, (term_name, label) in self.TERM_REGISTRY.items()
            ]
            if config.thermo_on:
                self.enabled_terms.append(('buoyancy_on', 'buoyancy', 'Buoyancy'))
            if config.mhd_on:
                self.enabled_terms.append(('mhd_on', 'mhd', 'MHD (Lorentz)'))
            self.enabled_terms.append(('balance', 'balance', 'Balance'))

        # Storage: {term_name: {(case, timestep): array}}
        self.raw_results: Dict[str, Dict[Tuple[str, str], np.ndarray]] = {
            t[1]: {} for t in self.enabled_terms
        }
        self.processed_results: Dict[str, Dict[Tuple[str, str], np.ndarray]] = {
            t[1]: {} for t in self.enabled_terms
        }

    def compute_for_case(self, case: str, timestep: str, data_loader) -> bool:
        """Compute all enabled budget terms for one case/timestep."""
        raw_dict = data_loader.get_raw_dict(case, timestep)
        if raw_dict is None:
            print(f"No data for Reynolds stress budget: {case}, {timestep}")
            return False

        y_coords = getattr(data_loader, 'y_coords', None)
        if y_coords is None:
            print(f"No y_coords for TKE budget: {case}, {timestep}")
            return False

        # Compute all TKE components once
        budget_comp = op.compute_budget_components(
            raw_dict, y_coords,
            average_z=self.average_z, average_x=self.average_x
        )

        # Get the correct Re, reference velocity and length for this case
        Re = float(op.get_ref_Re(case, self.config.cases, self.config.Re))
        u_ref = float(op.get_ref_Re(case, self.config.cases, self.config.ref_bulk_velocity))
        l_ref = float(op.get_ref_Re(case, self.config.cases, self.config.ref_length))

        # Extract each enabled term
        _compute_fns = {
            'production':          lambda d: op.compute_production(d, self.uiuj),
            'dissipation':         lambda d: op.compute_dissipation(Re, d, self.uiuj),
            'mean_convection':     lambda d: op.compute_mean_convection(d, self.uiuj),
            'viscous_diffusion':   lambda d: op.compute_viscous_diffusion(Re, d, self.uiuj),
            'pressure_transport':  lambda d: op.compute_pressure_transport(d, self.uiuj, u_ref),
            'pressure_strain':     lambda d: op.compute_pressure_strain(d, self.uiuj, u_ref),
            'turbulent_convection':lambda d: op.compute_turbulent_convection(d, self.uiuj),
            'buoyancy':            lambda d: op.compute_buoyancy_term(self.config.gravity_direction, u_ref, l_ref, d, self.uiuj),
            'mhd':                 lambda d: op.compute_mhd_term(self.config.mag_field_direction, self.config.stuart_number, d, self.uiuj),
        }

        for _flag, term_name, _label in self.enabled_terms:
            if term_name == 'balance':
                continue
            result = _compute_fns[term_name](budget_comp)
            value = next(iter(result.values()))
            self.raw_results[term_name][(case, timestep)] = value

        balance_val = None
        for _flag, term_name, _label in self.enabled_terms:
            if term_name == 'balance':
                continue
            val = self.raw_results[term_name].get((case, timestep))
            if val is not None:
                balance_val = val if balance_val is None else balance_val + val
        if balance_val is not None:
            self.raw_results['balance'][(case, timestep)] = balance_val

        return True


class BudgetTerm:
    """
    Thin wrapper around a single budget term so it looks like a stat object
    to the pipeline and plotter.
    """

    def __init__(self, term_name: str, label: str, computer: BudgetComputer):
        self.name = term_name
        self.label = label
        self._computer = computer

    @property
    def raw_results(self):
        return self._computer.raw_results[self.name]

    @property
    def processed_results(self):
        return self._computer.processed_results[self.name]

    @processed_results.setter
    def processed_results(self, value):
        self._computer.processed_results[self.name] = value

    def get_half_domain(self, values: np.ndarray) -> np.ndarray:
        return values[:(len(values)//2)]


# =====================================================================================================================================================
# BODY FORCE ANALYSIS CLASSES
# =====================================================================================================================================================

class ForceComputer:
    """
    Computes all enabled body-force terms by calling op.compute_force_components
    once and then extracting individual terms via op.compute_pressure /
    op.compute_buoyancy / op.compute_lorentz_force.

    """

    def __init__(self, config: Config):
        self.config = config
        self.component = config.body_force_component
        self.average_z = config.average_z_direction
        self.average_x = config.average_x_direction

        # Pressure is always available; buoyancy needs thermo data, Lorentz
        # needs MHD data — gated the same way BudgetComputer gates its own
        # buoyancy/mhd terms on config.thermo_on / config.mhd_on.
        self.enabled_terms = []
        if config.body_force_on:
            self.enabled_terms.append(('body_force_on', 'pressure_force', 'Pressure Force'))
            if config.thermo_on:
                self.enabled_terms.append(('thermo_on', 'buoyancy_force', 'Buoyancy Force'))
            if config.mhd_on:
                self.enabled_terms.append(('mhd_on', 'lorentz_force', 'Lorentz Force'))

        # Storage: {term_name: {(case, timestep): array}}
        self.raw_results: Dict[str, Dict[Tuple[str, str], np.ndarray]] = {
            t[1]: {} for t in self.enabled_terms
        }
        self.processed_results: Dict[str, Dict[Tuple[str, str], np.ndarray]] = {
            t[1]: {} for t in self.enabled_terms
        }

    def compute_for_case(self, case: str, timestep: str, data_loader) -> bool:
        """Compute all enabled body-force terms for one case/timestep."""
        raw_dict = data_loader.get_raw_dict(case, timestep)
        if raw_dict is None:
            print(f"No data for body force analysis: {case}, {timestep}")
            return False

        y_coords = getattr(data_loader, 'y_coords', None)
        if y_coords is None:
            print(f"No y_coords for body force analysis: {case}, {timestep}")
            return False

        force_comp = op.compute_force_components(
            raw_dict, y_coords,
            average_z=self.average_z, average_x=self.average_x
        )

        u_ref = float(op.get_ref_Re(case, self.config.cases, self.config.ref_bulk_velocity))
        l_ref = float(op.get_ref_Re(case, self.config.cases, self.config.ref_length))

        component_key = f'_{self.component}'
        _compute_fns = {
            'pressure_force': lambda d: op.compute_pressure(d),
            'buoyancy_force': lambda d: op.compute_buoyancy(
                self.config.gravity_direction, u_ref, l_ref, d),
            'lorentz_force': lambda d: op.compute_lorentz_force(
                self.config.mag_field_direction, self.config.stuart_number, d),
        }
        result_prefix = {
            'pressure_force': 'pressure',
            'buoyancy_force': 'buoyancy',
            'lorentz_force': 'lorentz',
        }

        for _flag, term_name, _label in self.enabled_terms:
            result = _compute_fns[term_name](force_comp)
            value = result.get(f'{result_prefix[term_name]}{component_key}')
            if value is None:
                print(f"  Warning: could not compute {term_name} ({self.component}) for {case}, {timestep}")
                continue
            self.raw_results[term_name][(case, timestep)] = value

        return True


class ForceTerm:
    """
    Thin wrapper around a single body-force term so it looks like a stat
    object to the pipeline and plotter.
    """

    def __init__(self, term_name: str, label: str, computer: ForceComputer):
        self.name = term_name
        self.label = label
        self._computer = computer

    @property
    def raw_results(self):
        return self._computer.raw_results[self.name]

    @property
    def processed_results(self):
        return self._computer.processed_results[self.name]

    @processed_results.setter
    def processed_results(self, value):
        self._computer.processed_results[self.name] = value

    def get_half_domain(self, values: np.ndarray) -> np.ndarray:
        return values[:(len(values)//2)]


# =====================================================================================================================================================
# ANISOTROPY ANALYSIS CLASSES
# =====================================================================================================================================================

class AnisotropyComputer:
    """
    Computes Reynolds-stress and/or vorticity anisotropy-tensor invariants
    (II, III) for Lumley-triangle plotting. Each tensor is built once per
    case/timestep and split into its two invariants — same batched pattern
    as BudgetComputer/ForceComputer, since op.second_invariant/
    op.third_invariant would otherwise redo the (expensive, for vorticity)
    tensor construction twice per case/timestep.

    Reynolds-stress anisotropy works on the native dimensionality of the
    loaded data (no gradients involved). Vorticity anisotropy goes through
    op.compute_vorticity internally, so — like MeanVorticity/
    VorticityFluctuationRMS — it needs the full, unaveraged 3D field.
    """

    def __init__(self, config: Config):
        self.config = config
        self.average_z = config.average_z_direction
        self.average_x = config.average_x_direction

        self.enabled_terms = []
        if config.reynolds_anisotropy_on:
            self.enabled_terms += [
                ('reynolds_anisotropy_on', 'reynolds_ii', '$II$ (Reynolds stress)'),
                ('reynolds_anisotropy_on', 'reynolds_iii', '$III$ (Reynolds stress)'),
            ]
        if config.vorticity_anisotropy_on:
            self.enabled_terms += [
                ('vorticity_anisotropy_on', 'vorticity_ii', '$II$ (vorticity)'),
                ('vorticity_anisotropy_on', 'vorticity_iii', '$III$ (vorticity)'),
            ]

        self.raw_results: Dict[str, Dict[Tuple[str, str], np.ndarray]] = {
            t[1]: {} for t in self.enabled_terms
        }
        self.processed_results: Dict[str, Dict[Tuple[str, str], np.ndarray]] = {
            t[1]: {} for t in self.enabled_terms
        }

    def compute_for_case(self, case: str, timestep: str, data_loader) -> bool:
        """Compute all enabled anisotropy invariants for one case/timestep."""
        raw_dict = data_loader.get_raw_dict(case, timestep)
        if raw_dict is None:
            print(f"No data for anisotropy analysis: {case}, {timestep}")
            return False

        ok = True

        if self.config.reynolds_anisotropy_on:
            b = op.construct_reynolds_stress_anisotropy_tensor(raw_dict)
            if b is None:
                ok = False
            else:
                self.raw_results['reynolds_ii'][(case, timestep)] = op.second_invariant(b)
                self.raw_results['reynolds_iii'][(case, timestep)] = op.third_invariant(b)

        if self.config.vorticity_anisotropy_on:
            if self.average_z or self.average_x:
                print("Vorticity anisotropy requires 'Average z direction' and 'Average x "
                      "direction' to both be off (it needs the full 3D field to "
                      "differentiate); skipping.")
                ok = False
            else:
                grid_info = getattr(data_loader, 'grid_info', None)
                d = op.construct_vorticity_anisotropy_tensor(raw_dict, grid_info)
                if d is None:
                    ok = False
                else:
                    self.raw_results['vorticity_ii'][(case, timestep)] = op.second_invariant(d)
                    self.raw_results['vorticity_iii'][(case, timestep)] = op.third_invariant(d)

        return ok


class AnisotropyTerm:
    """
    Thin wrapper around a single anisotropy invariant so it looks like a
    stat object to the pipeline and plotter.
    """

    def __init__(self, term_name: str, label: str, computer: AnisotropyComputer):
        self.name = term_name
        self.label = label
        self._computer = computer

    @property
    def raw_results(self):
        return self._computer.raw_results[self.name]

    @property
    def processed_results(self):
        return self._computer.processed_results[self.name]

    @processed_results.setter
    def processed_results(self, value):
        self._computer.processed_results[self.name] = value

    def get_half_domain(self, values: np.ndarray) -> np.ndarray:
        return values[:(len(values)//2)]


# =====================================================================================================================================================
# SPECTRAL ANALYSIS CLASSES
# =====================================================================================================================================================

class SpectrumComputer:
    """
    1D energy spectrum of the streamwise velocity fluctuation (qx_ccc)
    along a homogeneous direction (x or z), at one or more wall-normal (y)
    locations.

    A spectrum needs the raw, unaveraged instantaneous flow field — unlike
    the other stats in this pipeline, which work from time/space-averaged
    t_avg/tsp_avg data (already reduced away any spatial structure a
    spectrum would analyze) — so it loads the instantaneous XDMF file
    directly instead of going through the shared data_loader.

    raw_results / processed_results: {(case, timestep): {y_value: (k, E)}}
    — not plain ndarrays, so this is deliberately kept out of
    TurbulenceStatsPipeline.statistics (process_all's generic per-stat
    normalization assumes ndarray values).
    """

    def __init__(self, folder_path: str, direction: str, y_coords_str: str):
        self.folder_path = folder_path
        self.direction = direction if direction in ('x', 'z') else 'z'
        self.y_coords = [float(s) for s in y_coords_str.replace(',', ' ').split() if s.strip()]
        self.raw_results: Dict[Tuple[str, str], Dict[float, Tuple[np.ndarray, np.ndarray]]] = {}
        self.processed_results: Dict[Tuple[str, str], Dict[float, Tuple[np.ndarray, np.ndarray]]] = self.raw_results

    def compute_for_case(self, case: str, timestep: str, data_loader=None) -> bool:
        if not self.y_coords:
            print("Spectral analysis: no y-coordinates specified; skipping.")
            return False

        inst_xdmf = ut.visu_file_paths(self.folder_path, case, timestep)[0]
        if not os.path.isfile(inst_xdmf):
            print(f"Missing instantaneous flow file for spectral analysis: {case}, {timestep}\n"
                  f"  Looked for: {inst_xdmf}")
            return False

        var_meta, grid_info = ut.parse_xdmf_metadata(inst_xdmf)
        data = ut.load_xdmf_variables(var_meta, ['qx_ccc'], grid_info=grid_info)
        if not data or 'qx_ccc' not in data:
            print(f"Failed to load qx_ccc for spectral analysis: {case}, {timestep}")
            return False

        y_nodes = grid_info.get('grid_y')
        x_nodes = grid_info.get('grid_x')
        z_nodes = grid_info.get('grid_z')
        if y_nodes is None or x_nodes is None or z_nodes is None:
            print(f"Missing grid coordinates for spectral analysis: {case}, {timestep}")
            return False
        y_cell = 0.5 * (y_nodes[:-1] + y_nodes[1:])
        x_cell = 0.5 * (x_nodes[:-1] + x_nodes[1:])
        z_cell = 0.5 * (z_nodes[:-1] + z_nodes[1:])

        u = data['qx_ccc']  # (nz, ny, nx)
        results = {}
        for y_val in self.y_coords:
            y_idx = int(np.argmin(np.abs(y_cell - y_val)))
            plane = u[:, y_idx, :]  # (nz, nx), fixed y

            if self.direction == 'x':
                signal = plane - plane.mean(axis=1, keepdims=True)  # (nz, nx)
                dx = float(x_cell[1] - x_cell[0])
            else:
                signal = plane.T - plane.T.mean(axis=1, keepdims=True)  # (nx, nz)
                dx = float(z_cell[1] - z_cell[0])

            k, E = op.compute_1d_spectrum(signal, dx=dx)
            # Average over the other (batched) homogeneous direction for
            # statistical convergence of a single-snapshot spectrum.
            results[float(y_cell[y_idx])] = (k, E.mean(axis=0))

        self.raw_results[(case, timestep)] = results
        return True


# =====================================================================================================================================================
# PIPELINE CLASS
# =====================================================================================================================================================

class TurbulenceStatsPipeline:
    """Orchestrates the computation and processing of turbulence statistics"""

    def __init__(self, config: Config, data_loader: TurbulenceTextData):
        self.config = config
        self.data_loader = data_loader
        self.statistics: List[Union[ReStresses, Profiles, Budget]] = []
        self._register_statistics()

    def _register_statistics(self) -> None:
        """Register all enabled statistics based on configuration"""
        if self.config.ux_velocity_on:
            self.statistics.append(StreamwiseVelocity())

        if self.config.uy_velocity_on:
            self.statistics.append(WallNormalVelocity())

        if self.config.uz_velocity_on:
            self.statistics.append(SpanwiseVelocity())

        if self.config.u_prime_sq_on:
            self.statistics.append(ReynoldsStressuu11())

        if self.config.u_prime_v_prime_on:
            self.statistics.append(ReynoldsStressuu12())

        if self.config.v_prime_sq_on:
            self.statistics.append(ReynoldsStressuu22())

        if self.config.v_prime_w_prime_on:
            self.statistics.append(ReynoldsStressuu23())

        if self.config.w_prime_sq_on:
            self.statistics.append(ReynoldsStressuu33())

        if self.config.tke_on:
            self.statistics.append(TurbulentKineticEnergy())

        if self.config.tke_peak_growth_on:
            # Peak stats are reduced over y at compute time, so the half-channel
            # selection has to be applied there rather than when processing.
            peak_side = (self.config.half_channel_side
                         if self.config.half_channel_plot else None)
            self.statistics.append(PeakTKE(peak_side))
            self.statistics.append(PeakReynoldsStress('u_prime_sq', "<u'u'>", 'u1', 'uu11', peak_side))
            self.statistics.append(PeakReynoldsStress('v_prime_sq', "<v'v'>", 'u2', 'uu22', peak_side))
            self.statistics.append(PeakReynoldsStress('w_prime_sq', "<w'w'>", 'u3', 'uu33', peak_side))

        if self.config.temp_on:
            self.statistics.append(Temperature(self.config.norm_temp_by_ref_temp, self.config.ref_temp, self.config.cases))

        if self.config.heat_transf_coeff_on:
            self.statistics.append(HeatTransferCoefficient(
                self.config.cases,
                self.config.ref_temp,
                self.config.wall_heat_flux,
                self.config.working_fluid,
            ))

        if self.config.Nusselt_number_on:
            self.statistics.append(NusseltNumber(
                self.config.cases,
                self.config.ref_temp,
                self.config.wall_heat_flux,
                self.config.ref_length,
                self.config.working_fluid,
            ))

        if self.config.turb_prandtl_on:
            self.statistics.append(TurbulentPrandtlNumber())

        if self.config.coeff_friction_on:
            self.statistics.append(FrictionCoefficient(self.config.cases, self.config.Re))

        if self.config.mean_vorticity_on:
            self.statistics.append(MeanVorticity(
                self.config.vorticity_component,
                self.config.average_z_direction,
                self.config.average_x_direction,
            ))

        if self.config.vorticity_on:
            self.statistics.append(VorticityFluctuationRMS(
                self.config.vorticity_component,
                self.config.average_z_direction,
                self.config.average_x_direction,
            ))

        if self.config.j1_mean_on:
            self.statistics.append(MeanCurrentDensityj1())
        if self.config.j2_mean_on:
            self.statistics.append(MeanCurrentDensityj2())
        if self.config.j3_mean_on:
            self.statistics.append(MeanCurrentDensityj3())

        if self.config.j1_rms_on:
            self.statistics.append(CurrentDensityRMSj1())
        if self.config.j2_rms_on:
            self.statistics.append(CurrentDensityRMSj2())
        if self.config.j3_rms_on:
            self.statistics.append(CurrentDensityRMSj3())

        if self.config.lorentz_force_x_on:
            self.statistics.append(MeanLorentzForcex(self.config.mag_field_direction, self.config.stuart_number))
        if self.config.lorentz_force_y_on:
            self.statistics.append(MeanLorentzForcey(self.config.mag_field_direction, self.config.stuart_number))
        if self.config.lorentz_force_z_on:
            self.statistics.append(MeanLorentzForcez(self.config.mag_field_direction, self.config.stuart_number))

        re_stress_budget_enabled = self.config.re_stress_budget_on

        self.budget_computer = None
        if re_stress_budget_enabled:
            self.budget_computer = BudgetComputer(self.config)
            for _flag, term_name, label in self.budget_computer.enabled_terms:
                self.statistics.append(BudgetTerm(term_name, label, self.budget_computer))

        self.force_computer = None
        if self.config.body_force_on:
            self.force_computer = ForceComputer(self.config)
            for _flag, term_name, label in self.force_computer.enabled_terms:
                self.statistics.append(ForceTerm(term_name, label, self.force_computer))

        self.anisotropy_computer = None
        if self.config.reynolds_anisotropy_on or self.config.vorticity_anisotropy_on:
            self.anisotropy_computer = AnisotropyComputer(self.config)
            for _flag, term_name, label in self.anisotropy_computer.enabled_terms:
                self.statistics.append(AnisotropyTerm(term_name, label, self.anisotropy_computer))

        # Not appended to self.statistics — see SpectrumComputer's docstring.
        self.spectrum_computer = None
        if self.config.spectrum_on:
            self.spectrum_computer = SpectrumComputer(
                self.config.folder_path,
                self.config.spectrum_direction,
                self.config.spectrum_y_coords,
            )

    def compute_all(self) -> None:
        """Compute all registered statistics for all cases and timesteps"""
        # Use the loader's timestep list — it may have been updated (e.g. to ['avg'])
        # by load_all() when average_over_timesteps is enabled.
        timesteps = self.data_loader.timesteps

        # Separate regular stats from budget/force/anisotropy terms (each of
        # those is computed in one batched call per case/timestep, covering
        # all its terms, rather than per-stat).
        regular_stats = [s for s in self.statistics if not isinstance(s, (BudgetTerm, ForceTerm, AnisotropyTerm))]
        n_budget = len(self.budget_computer.enabled_terms) if self.budget_computer else 0
        n_force = len(self.force_computer.enabled_terms) if self.force_computer else 0
        n_anisotropy = len(self.anisotropy_computer.enabled_terms) if self.anisotropy_computer else 0
        total_tasks = len(regular_stats) * len(self.config.cases) * len(timesteps)
        # Budget/force/anisotropy/spectrum: one compute call per case/timestep (covers all terms)
        total_tasks += len(self.config.cases) * len(timesteps) if n_budget else 0
        total_tasks += len(self.config.cases) * len(timesteps) if n_force else 0
        total_tasks += len(self.config.cases) * len(timesteps) if n_anisotropy else 0
        total_tasks += len(self.config.cases) * len(timesteps) if self.spectrum_computer else 0

        with tqdm(total=total_tasks, desc="Computing statistics", unit="stat") as pbar:
            # Regular stats
            for stat in regular_stats:
                for case in self.config.cases:
                    for timestep in timesteps:
                        stat.compute_for_case(case, timestep, self.data_loader)
                        pbar.update(1)

            # TKE budget: one call per case/timestep computes all terms
            if self.budget_computer:
                for case in self.config.cases:
                    for timestep in timesteps:
                        self.budget_computer.compute_for_case(case, timestep, self.data_loader)
                        pbar.update(1)

            # Body force analysis: one call per case/timestep computes all terms
            if self.force_computer:
                for case in self.config.cases:
                    for timestep in timesteps:
                        self.force_computer.compute_for_case(case, timestep, self.data_loader)
                        pbar.update(1)

            # Anisotropy invariants: one call per case/timestep computes all terms
            if self.anisotropy_computer:
                for case in self.config.cases:
                    for timestep in timesteps:
                        self.anisotropy_computer.compute_for_case(case, timestep, self.data_loader)
                        pbar.update(1)

            # Spectral analysis: one call per case/timestep computes all y-locations
            if self.spectrum_computer:
                for case in self.config.cases:
                    for timestep in timesteps:
                        self.spectrum_computer.compute_for_case(case, timestep, self.data_loader)
                        pbar.update(1)

    def process_all(self) -> None:
        """Apply normalization and averaging to all computed statistics.

        Data is kept in its native dimensionality (1-D, 2-D, or 3-D).
        Plane extraction / x-averaging happens at plot time.
        """
        # Detect native-array mode (XDMF loader with y_coords)
        y_coords = getattr(self.data_loader, 'y_coords', None)

        total_tasks = sum(len(stat.raw_results) for stat in self.statistics)
        printed_info = set()  # only print flow info once per (case, timestep)

        with tqdm(total=total_tasks, desc="Processing statistics", unit="stat") as pbar:
            for stat in self.statistics:
                for (case, timestep), values in stat.raw_results.items():

                    # Get u1 data for normalization
                    ux_data = self.data_loader.get(case, 'u1', timestep)
                    if ux_data is None:
                        print(f'Missing u1 data for normalization: {case}, {timestep}')
                        pbar.update(1)
                        continue

                    ref_Re = op.get_ref_Re(case, self.config.cases, self.config.Re)

                    # Normalize (element-wise — works for any ndim). Anisotropy
                    # invariants are already dimensionless (built from a
                    # normalized tensor), so u_tau^2 normalization doesn't apply.
                    if self.config.norm_by_u_tau_sq and stat.name not in (
                        'temperature', 'coeff_friction', 'heat_transfer_coeff', 'nusselt_number', 'turb_prandtl',
                        'reynolds_ii', 'reynolds_iii', 'vorticity_ii', 'vorticity_iii',
                    ):
                        normed = op.norm_turb_stat_wrt_u_tau_sq(ux_data, values, ref_Re, y_coords=y_coords)
                    else:
                        normed = values

                    # Special normalization for u1 velocity
                    if self.config.norm_ux_by_u_tau and stat.name == 'ux_velocity':
                        normed = op.norm_ux_velocity_wrt_u_tau(ux_data, ref_Re, y_coords=y_coords)
                        print(f'u1 velocity normalised by u_tau for {case}, {timestep}')


                    is_x_profile_only = getattr(stat, 'x_profile_only', False)
                    is_anisotropy_invariant = stat.name in (
                        'reynolds_ii', 'reynolds_iii', 'vorticity_ii', 'vorticity_iii',
                    )
                    if self.config.half_channel_plot and not is_x_profile_only and not is_anisotropy_invariant:

                        half_len = normed.shape[0] - normed.shape[0] // 2
                        side = self.config.half_channel_side

                        if side == 'average' and stat.name not in ('u_prime_v_prime', 'temperature'):
                            stat.processed_results[(case, timestep)] = op.symmetric_average(normed)
                        elif side == 'upper':
                            stat.processed_results[(case, timestep)] = np.flip(normed, axis=0)[:half_len]
                        else:
                            stat.processed_results[(case, timestep)] = normed[:half_len]
                    else:
                        stat.processed_results[(case, timestep)] = normed

                    # Print flow info once per (case, timestep)
                    if (case, timestep) not in printed_info:
                        cur_Re = op.get_Re(case, self.config.cases, self.config.Re, ux_data,
                                           self.config.forcing, y_coords=y_coords)
                        ut.print_flow_info(ux_data, ref_Re, cur_Re, case, timestep, y_coords=y_coords)
                        printed_info.add((case, timestep))
                    pbar.update(1)

    def get_statistic(self, name: str) -> Optional[Union[ReStresses, Profiles, Budget]]:
        """Get a specific statistic by name"""
        for stat in self.statistics:
            if stat.name == name:
                return stat
        return None

    def get_statistics_by_class(self) -> Dict[str, List]:
        """Group statistics by their class type"""
        grouped: Dict[str, List] = {
            'ReStresses': [],
            'Profiles': [],
            'ReStressBudget': [],
            'TkeBudget': [],
            'BodyForce': [],
            'Anisotropy': [],
            'PeakGrowth': [],
        }
        for stat in self.statistics:
            if isinstance(stat, BudgetTerm):
                grouped['ReStressBudget'].append(stat)
                grouped['TkeBudget'].append(stat)
            elif isinstance(stat, ForceTerm):
                grouped['BodyForce'].append(stat)
            elif isinstance(stat, AnisotropyTerm):
                grouped['Anisotropy'].append(stat)
            elif isinstance(stat, PeakGrowth):
                grouped['PeakGrowth'].append(stat)
            elif isinstance(stat, ReStresses):
                grouped['ReStresses'].append(stat)
            elif isinstance(stat, Profiles):
                grouped['Profiles'].append(stat)
        return grouped

# =====================================================================================================================================================
# PLOTTING CLASS
# =====================================================================================================================================================

class TurbulencePlotter:
    """Handles all plotting logic for turbulence statistics.

    Processed data may be 1-D (ny,) or 2-D (ny, nx).  Plane extraction
    happens at plot-time via ``_extract_profiles``:
      * If the data is already 1-D it is used directly.
      * If ``slice_coords`` is set, profiles are extracted at the
        nearest x-index for each requested coordinate.
      * Otherwise the data is averaged over the x-direction (axis 1)
        to produce a single 1-D profile (default behaviour).
    """

    def __init__(self, config: Config, plot_config: PlotConfig, data_loader: TurbulenceTextData):
        self.config = config
        self.plot_config = plot_config
        self.data_loader = data_loader
        # Keep one consistent line style per case across all plots.
        style_cycle = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
        self.case_linestyles = {
            case: style_cycle[i % len(style_cycle)]
            for i, case in enumerate(self.config.cases)
        }
        # Distinct case markers for readability in overlapping lines.
        marker_cycle = ['o', 's', '^', 'D', 'x', 'P', 'v', '*']
        self.case_markers = {
            case: marker_cycle[i % len(marker_cycle)]
            for i, case in enumerate(self.config.cases)
        }
        self._series_color_map: Dict[str, str] = {}
        self._next_color_index = 0
        self._suffix_colors: Optional[Dict[str, str]] = None

    def _reset_color_cycle(self) -> None:
        """Reset per-figure colour assignment so lines are distinct within each plot."""
        self._series_color_map = {}
        self._next_color_index = 0

    def _get_line_color(self, case: str, timestep: str, stat_name: Optional[str], suffix: str) -> str:
        """Colour for one plotted profile line.

        Priority: a fixed budget-term colour (consistent across every plot)
        > a colour shared by every case/stat at the same x-coordinate slice
        (consistent across every figure) > a colour unique to this
        case/timestep/stat (fresh each figure).
        """
        budget_colors = self.plot_config.budget_colors
        if stat_name and stat_name in budget_colors:
            return budget_colors[stat_name]
        if suffix:
            return self._get_suffix_color(suffix)
        return self._get_color(f'{case}|{timestep}|{stat_name}|{suffix}', stat_name)

    def _get_suffix_color(self, suffix: str) -> str:
        """Colour for a slice-coordinate suffix (e.g. ' x=2' from a y-profile
        sliced at x-locations, or ' y=0.5' from an x-profile sliced at
        y-locations), taken from a fixed position in a qualitative colormap
        indexed by the sorted list of requested slice coordinates — so it's
        bounded by the (small, known) number of distinct coordinates rather
        than a running counter, and is the same everywhere that coordinate
        is plotted. x-profiles and y-profiles are never shown in the same
        figure, so both index from the start of the colormap independently
        (slice #1 looks the same colour in either direction).
        """
        if self._suffix_colors is None:
            x_slices = sorted(self._get_slice_indices(), key=lambda pair: pair[1])
            y_slices = sorted(self._get_x_profile_y_indices(), key=lambda pair: pair[1])
            cmap = mpl.colormaps['tab20']
            self._suffix_colors = {}
            for i, (_idx, xv) in enumerate(x_slices):
                self._suffix_colors[f' x={xv:.3g}'] = mcolors.to_hex(cmap(i % cmap.N))
            for i, (_idx, yv) in enumerate(y_slices):
                self._suffix_colors[f' y={yv:.3g}'] = mcolors.to_hex(cmap(i % cmap.N))
        return self._suffix_colors.get(suffix, self.plot_config.visible_palette[0])

    def _format_case_label(self, case: str) -> str:
        """Return a human-readable case label."""
        return case.replace("_", " ")

    def _build_series_context_label(self, case: str, timestep: str, suffix: str) -> str:
        """Build legend context from case/slice info, omitting case for single-case runs."""
        include_case = len(self.config.cases) > 1
        suffix_clean = suffix.strip()

        if include_case and suffix_clean:
            return f'{self._format_case_label(case)} {suffix_clean}'
        if include_case:
            return self._format_case_label(case)
        if suffix_clean:
            return suffix_clean
        if len(self.config.timesteps) > 1:
            return f't={timestep}'
        return ''

    def _build_legend_label(self, stat_label: str, case: str, timestep: str,
                            suffix: str, include_stat_label: bool = False) -> str:
        """Build legend labels with optional statistic prefix."""
        context = self._build_series_context_label(case, timestep, suffix)
        if include_stat_label:
            return f'{stat_label}, {context}' if context else stat_label
        return context if context else stat_label

    def _get_y_profile_xlabel(self) -> str:
        """Return x-axis label for wall-normal profiles."""
        if self.config.norm_y_to_y_plus:
            return '$y^+$'
        return 'Distance from wall' if self.config.half_channel_plot else '$y$'

    def _get_stat_ylabel(self, stat_name: str, stat_label: str) -> str:
        """Return y-axis label matching enabled normalisation options."""
        base_labels = {
            'ux_velocity': '$U_x/U_{bulk}$',
            'uy_velocity': '$U_y/U_{bulk}$',
            'uz_velocity': '$U_z/U_{bulk}$',
            'temperature': '$T$',
            'heat_transfer_coeff': '$h$ (W/(m^2K))',
            'nusselt_number': '$Nu$',
            'turb_prandtl': '$Pr_t$',
            'TKE': '$k/U_{bulk}^2$',
            'u_prime_sq': "$\\langle u'u' \\rangle/U_{bulk}^2$",
            'u_prime_v_prime': "$\\langle u'v' \\rangle/U_{bulk}^2$",
            'v_prime_sq': "$\\langle v'v' \\rangle/U_{bulk}^2$",
            'v_prime_w_prime': "$\\langle v'w' \\rangle/U_{bulk}^2$",
            'w_prime_sq': "$\\langle w'w' \\rangle/U_{bulk}^2$",
        }
        base = base_labels.get(stat_name, stat_label)

        if stat_name == 'temperature':
            return '$\\theta/\\theta_{ref}$' if self.config.norm_temp_by_ref_temp else '$\\theta$ (K)'

        if stat_name == 'ux_velocity' and self.config.norm_ux_by_u_tau:
            return '$U_x/u_\\tau$'

        if self.config.norm_by_u_tau_sq and stat_name not in (
            'coeff_friction', 'heat_transfer_coeff', 'nusselt_number',
            'reynolds_ii', 'reynolds_iii', 'vorticity_ii', 'vorticity_iii',
        ):
            if isinstance(base, str) and base.startswith('$') and base.endswith('$'):
                return base[:-1] + '/u_\\tau^2$'
            return f'{base} / $u_\\tau^2$'

        return base

    def _get_axis_label_fontsize(self) -> Optional[int]:
        """Return axis-label font size when large text mode is enabled."""
        return 18 if self.config.large_text_on else None

    def _get_title_fontsize(self) -> Optional[int]:
        """Return title font size when large text mode is enabled."""
        return 20 if self.config.large_text_on else None

    def _get_legend_fontsize(self) -> str:
        """Return legend font size for normal/large text modes."""
        return 'large' if self.config.large_text_on else 'small'

    def _apply_axis_text_style(self, ax) -> None:
        """Apply larger tick label text when requested."""
        if self.config.large_text_on:
            ax.tick_params(axis='both', which='both', labelsize=16)

    # ------------------------------------------------------------------
    # Plane / profile extraction helpers
    # ------------------------------------------------------------------
    def _parse_slice_coords(self) -> List[float]:
        """Parse the comma-separated ``slice_coords`` config string."""
        if not self.config.slice_coords or not self.config.slice_coords.strip():
            return []
        return [float(s.strip()) for s in self.config.slice_coords.split(',') if s.strip()]

    def _get_slice_indices(self) -> List[Tuple[int, float]]:
        """Return list of (x-index, actual_x_value) for requested slices."""
        x_coords = getattr(self.data_loader, 'x_coords', None)
        requested = self._parse_slice_coords()
        if not requested or x_coords is None:
            return []

        indices = []
        for xc in requested:
            idx = int(np.argmin(np.abs(x_coords - xc)))
            indices.append((idx, float(x_coords[idx])))
        return indices

    def _extract_profiles(self, values: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Extract 1-D wall-normal profiles from (possibly 2-D) data.

        Returns a list of ``(label_suffix, profile_1d)`` tuples.
        For 1-D input the suffix is an empty string.
        """
        if values.ndim == 1:
            return [('', values)]

        slices = self._get_slice_indices()
        if slices:
            return [(f' x={xv:.3g}', values[:, idx]) for idx, xv in slices]
        else:
            # Default: average over x
            return [(' (x-avg)', values.mean(axis=1))]

    # ------------------------------------------------------------------
    # X-direction profile extraction helpers
    # ------------------------------------------------------------------
    def _parse_x_profile_y_coords(self) -> List[float]:
        """Parse the comma-separated ``x_profile_y_coords`` config string."""
        if not self.config.x_profile_y_coords or not self.config.x_profile_y_coords.strip():
            return []
        return [float(s.strip()) for s in self.config.x_profile_y_coords.split(',') if s.strip()]

    def _get_x_profile_y_indices(self) -> List[Tuple[int, float]]:
        """Return list of (y-index, actual_y_value) for requested x-profile locations.

        An empty list means no explicit locations were requested, in which case
        the caller should average over y instead of picking specific rows.
        """
        y_coords = getattr(self.data_loader, 'y_coords', None)
        if y_coords is None:
            return []
        requested = self._parse_x_profile_y_coords()
        if not requested:
            return []
        indices = []
        for yc in requested:
            idx = int(np.argmin(np.abs(y_coords - yc)))
            indices.append((idx, float(y_coords[idx])))
        return indices

    def _extract_x_profiles(self, values: np.ndarray, stat=None) -> List[Tuple[str, np.ndarray]]:
        """Extract 1-D streamwise profiles from native arrays."""
        if values.ndim < 1:
            return []

        if getattr(stat, 'x_profile_only', False):
            if values.ndim == 1:
                return [('', values)]
            if values.ndim == 2:
                return [('', values[0, :])]
            if values.ndim == 3:
                return [('', values[0, :, :].mean(axis=1))]
            return []

        if values.ndim == 1:
            return [('', values)]

        y_indices = self._get_x_profile_y_indices()
        if not y_indices:
            # Default: average over y
            if values.ndim == 2:
                return [(' (y-avg)', values.mean(axis=0))]
            if values.ndim == 3:
                return [(' (y-avg)', values.mean(axis=(0, 2)))]
            return [('', np.ravel(values))]

        if values.ndim == 2:
            return [(f' y={yv:.3g}', values[idx, :]) for idx, yv in y_indices]
        if values.ndim == 3:
            return [(f' y={yv:.3g}', values[idx, :, :].mean(axis=1)) for idx, yv in y_indices]
        return [('', np.ravel(values))]

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def plot(self, statistics: List[Union[ReStresses, Profiles, Budget]], reference_data: Optional[ReferenceData] = None):
        """Main plotting method - delegates to single or multi plot"""
        if len(statistics) == 1 or not self.config.multi_plot:
            return self._plot_single_figure(statistics, reference_data)
        else:
            return self._plot_multi_figure(statistics, reference_data)

    def plot_by_class(self, grouped_statistics: Dict[str, List[Union[ReStresses, Profiles, Budget]]],
                      reference_data: Optional[ReferenceData] = None) -> Dict[str, Any]:
        """Create separate figures for each class type (ReStresses, Profiles, TkeBudget).

        When ``surface_plot_on`` is enabled and data is 2-D, an additional
        set of surface (contour) figures is produced.
        """
        figures = {}

        class_titles = {
            'ReStresses': 'Reynolds Stresses',
            'Profiles': 'Profiles',
            'ReStressBudget': 'TKE Budget',
            'BodyForce': 'Body Force Analysis',
            'Anisotropy': 'Anisotropy (Lumley Triangle)',
            'PeakGrowth': 'Peak Turbulent Kinetic Energy Growth',
        }

        for class_name, stats_list in grouped_statistics.items():
            if not stats_list:
                continue

            # Y-direction (wall-normal) profile figures
            if self.config.profile_direction in ('y', 'both'):
                # Filter out x-profile-only statistics
                y_profile_stats = [s for s in stats_list if not getattr(s, 'x_profile_only', False)]
                if y_profile_stats:
                    class_title = class_titles.get(class_name, class_name)
                    fig = self._plot_class_figure(y_profile_stats, class_title, reference_data)
                    figures[class_name] = fig

            # X-direction (streamwise) profile figures
            if self.config.profile_direction in ('x', 'both'):
                # Include all stats (x-profile-only ones will plot correctly)
                class_title = class_titles.get(class_name, class_name)
                x_fig = self._plot_x_profile_figure(stats_list, class_title)
                if x_fig is not None:
                    figures[f'{class_name}_x_profile'] = x_fig

            # Optional 2-D surface contour figures
            if self.config.surface_plot_on:
                class_title = class_titles.get(class_name, class_name)
                surf_figs = self._plot_surface_figures(stats_list, class_title)
                for key, sfig in surf_figs.items():
                    figures[key] = sfig

        return figures

    def _plot_class_figure(self, statistics, title: str, reference_data: Optional[ReferenceData] = None):
        """Create a figure for a single class type.
        For TkeBudgetTerm stats: all terms on one set of axes.
        For other stats: subplots as before.

        2-D data is reduced to 1-D profiles via ``_extract_profiles`` before
        plotting (either x-averaged or at specific slice coordinates).
        """
        self._reset_color_cycle()

        # ---- TKE Budget: all terms on a single axes ----
        is_budget = all(isinstance(s, BudgetTerm) for s in statistics)
        if is_budget:
            return self._plot_budget_figure(statistics, title)

        # ---- Body force analysis: all terms on a single axes ----
        is_force = all(isinstance(s, ForceTerm) for s in statistics)
        if is_force:
            return self._plot_force_figure(statistics, title)

        # ---- Anisotropy invariants: Lumley triangle(s) ----
        is_anisotropy = all(isinstance(s, AnisotropyTerm) for s in statistics)
        if is_anisotropy:
            return self._plot_lumley_triangle_figure(statistics, title)

        # ---- Standard subplot layout ----
        n_stats = len(statistics)

        if n_stats == 1:
            fig = Figure(figsize=(10, 6), constrained_layout=True)
            ax = fig.add_subplot(111)
            axs = np.array([[ax]])
            nrows, ncols = 1, 1
        else:
            ncols = math.ceil(math.sqrt(n_stats))
            nrows = math.ceil(n_stats / ncols)
            fig = Figure(figsize=(15, 10), constrained_layout=True)
            axs = np.array(fig.subplots(nrows=nrows, ncols=ncols, squeeze=False))

        for i, stat in enumerate(statistics):
            row = i // ncols
            col = i % ncols
            ax = axs[row, col]

            for (case, timestep), values in stat.processed_results.items():
                y_plus = self._get_y_plus(case, timestep)
                if y_plus is None:
                    continue

                profiles = self._extract_profiles(values)
                for suffix, profile in profiles:
                    color = self._get_line_color(case, timestep, stat.name, suffix)
                    label = self._build_legend_label(stat.label, case, timestep, suffix)
                    linestyle = self._get_linestyle(case)
                    marker = self._get_marker(case)
                    self._plot_line(ax, y_plus, profile, label, color, linestyle=linestyle, marker=marker)

                if reference_data:
                    self._plot_reference_data(ax, stat.name, case, reference_data)
                if stat.name == 'ux_velocity' and self.config.ux_velocity_log_ref_on and self.config.log_y_scale:
                    self._plot_log_reference_lines(ax, y_plus)

            ax.set_title(f'{stat.label}', fontsize=self._get_title_fontsize())
            ax.set_ylabel(
                self._get_stat_ylabel(stat.name, stat.label),
                fontsize=self._get_axis_label_fontsize()
            )
            ax.grid(True)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=self._get_legend_fontsize())
            if row == nrows - 1:
                ax.set_xlabel(self._get_y_profile_xlabel(), fontsize=self._get_axis_label_fontsize())
            self._apply_axis_text_style(ax)

        for i in range(n_stats, nrows * ncols):
            row = i // ncols
            col = i % ncols
            axs[row, col].set_visible(False)

        return fig

    def _plot_budget_figure(self, statistics, title: str):
        """Plot all TKE budget terms on a single axes."""
        component = self.config.re_stress_component
        fig = Figure(figsize=(10, 6), constrained_layout=True)
        ax = fig.add_subplot(111)

        for stat in statistics:
            for (case, timestep), values in stat.processed_results.items():
                y_plus = self._get_y_plus(case, timestep)
                if y_plus is None:
                    continue
                profiles = self._extract_profiles(values)
                for suffix, profile in profiles:
                    label = self._build_legend_label(stat.label, case, timestep, suffix, include_stat_label=True)
                    if stat.name == 'balance':
                        self._plot_line(ax, y_plus, profile, label, color='black', linestyle='--')
                    else:
                        color = self._get_line_color(case, timestep, stat.name, suffix)
                        linestyle = self._get_linestyle(case)
                        marker = self._get_marker(case)
                        self._plot_line(ax, y_plus, profile, label, color, linestyle=linestyle, marker=marker)

        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
        ax.set_title(f'TKE Budget ({component})', fontsize=self._get_title_fontsize())
        ax.set_xlabel(self._get_y_profile_xlabel(), fontsize=self._get_axis_label_fontsize())
        if self.config.norm_by_u_tau_sq:
            ax.set_ylabel('Budget term magnitude / $u_\\tau^2$', fontsize=self._get_axis_label_fontsize())
        else:
            ax.set_ylabel('Budget term magnitude', fontsize=self._get_axis_label_fontsize())
        ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=self._get_legend_fontsize())
        self._apply_axis_text_style(ax)
        return fig

    def _plot_force_figure(self, statistics, title: str):
        """Plot all body-force terms (pressure, buoyancy, Lorentz) on a single axes."""
        component_labels = {'1': 'x', '2': 'y', '3': 'z'}
        component = component_labels.get(self.config.body_force_component, self.config.body_force_component)
        fig = Figure(figsize=(10, 6), constrained_layout=True)
        ax = fig.add_subplot(111)

        for stat in statistics:
            for (case, timestep), values in stat.processed_results.items():
                y_plus = self._get_y_plus(case, timestep)
                if y_plus is None:
                    continue
                profiles = self._extract_profiles(values)
                for suffix, profile in profiles:
                    label = self._build_legend_label(stat.label, case, timestep, suffix, include_stat_label=True)
                    color = self._get_line_color(case, timestep, stat.name, suffix)
                    linestyle = self._get_linestyle(case)
                    marker = self._get_marker(case)
                    self._plot_line(ax, y_plus, profile, label, color, linestyle=linestyle, marker=marker)

        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
        ax.set_title(f'Body Force Analysis ({component}-component)', fontsize=self._get_title_fontsize())
        ax.set_xlabel(self._get_y_profile_xlabel(), fontsize=self._get_axis_label_fontsize())
        if self.config.norm_by_u_tau_sq:
            ax.set_ylabel('Force magnitude / $u_\\tau^2$', fontsize=self._get_axis_label_fontsize())
        else:
            ax.set_ylabel('Force magnitude', fontsize=self._get_axis_label_fontsize())
        ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=self._get_legend_fontsize())
        self._apply_axis_text_style(ax)
        return fig

    def _lumley_triangle_boundaries(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Realizability boundary of the Lumley triangle in (III, II) space.

        Built directly from the eigenvalue constraints of a traceless
        symmetric 3x3 tensor (sum(lambda_i) = 0, -1/3 <= lambda_i <= 2/3)
        using the same II = -0.5*sum(lambda_i**2), III = prod(lambda_i)
        definitions as op.second_invariant/op.third_invariant, so the
        boundary curves are guaranteed self-consistent with the plotted
        data rather than risking an algebra slip in an inverted closed form.
        """
        def invariants(l1, l2, l3):
            ii = -0.5 * (l1 ** 2 + l2 ** 2 + l3 ** 2)
            iii = l1 * l2 * l3
            return iii, ii

        lam = np.linspace(0.0, 1.0 / 3.0, 100)

        # Axisymmetric expansion (rod-like): lambda2 = lambda3 = -lambda1/2
        rod = invariants(2 * lam, -lam, -lam)

        # Axisymmetric contraction (disk-like): lambda1 = lambda2 = -lambda3/2
        disk = invariants(lam, lam, -2 * lam)

        # Two-component turbulence: lambda3 = -1/3
        t = np.linspace(-1.0 / 3.0, 1.0 / 6.0, 100)
        two_component = invariants(1.0 / 3.0 - t, t, np.full_like(t, -1.0 / 3.0))

        return {
            'axisymmetric_expansion': rod,
            'axisymmetric_contraction': disk,
            'two_component': two_component,
        }

    def _plot_lumley_triangle_figure(self, statistics, title: str):
        """Plot Reynolds-stress and/or vorticity anisotropy invariants as
        Lumley triangles — one subplot per kind, each case/timestep traced
        as a curve in (III, -II) space (wall to centerline) against the
        realizability boundary.

        Each line gets its own marker shape plus a distinct line/marker-edge
        colour (identity), while the marker fill is colour-graded by
        distance from the wall (physical position, via a colourbar) — two
        separate visual channels rather than relying on marker shape alone.
        """
        by_kind: Dict[str, Dict[str, Any]] = {}
        for stat in statistics:
            kind, _, invariant = stat.name.partition('_')
            by_kind.setdefault(kind, {})[invariant] = stat

        kinds = [k for k, terms in by_kind.items() if 'ii' in terms and 'iii' in terms]
        if not kinds:
            return None

        kind_titles = {'reynolds': 'Reynolds Stress', 'vorticity': 'Vorticity'}
        boundaries = self._lumley_triangle_boundaries()
        marker_cycle = ['o', 's', '^', 'D', 'x', 'P', 'v', '*']

        fig = Figure(figsize=(7 * len(kinds) + 1, 6), constrained_layout=True)
        axs = np.array(fig.subplots(nrows=1, ncols=len(kinds), squeeze=False))[0]

        for col, kind in enumerate(kinds):
            ax = axs[col]
            ii_stat = by_kind[kind]['ii']
            iii_stat = by_kind[kind]['iii']

            for iii_b, ii_b in boundaries.values():
                ax.plot(iii_b, -ii_b, color='black', linewidth=1, zorder=1)

            scatter = None
            marker_index = 0
            for (case, timestep), ii_values in ii_stat.processed_results.items():
                iii_values = iii_stat.processed_results.get((case, timestep))
                if iii_values is None:
                    continue
                # as_wall_distance=True: always 0 at the (selected) wall,
                # increasing toward the centreline — a plain signed y would
                # run the wrong way for 'upper' (+1 at the wall).
                y_plus = self._get_y_plus(case, timestep, as_wall_distance=True)
                if y_plus is None:
                    continue

                iii_profiles = dict(self._extract_profiles(iii_values))
                for suffix, ii_profile in self._extract_profiles(ii_values):
                    iii_profile = iii_profiles.get(suffix)
                    if iii_profile is None:
                        continue

                    # Half-channel: truncate to the near-wall half here (matching
                    # y_plus's already-halved, side-sliced length) rather than
                    # averaging II/III across both walls in process_all — see
                    # the note there on why that would produce unrealizable
                    # (out-of-triangle) points.
                    if self.config.half_channel_plot:
                        if self.config.half_channel_side == 'upper':
                            ii_profile = np.flip(ii_profile)[:len(y_plus)]
                            iii_profile = np.flip(iii_profile)[:len(y_plus)]
                        else:
                            ii_profile = ii_profile[:len(y_plus)]
                            iii_profile = iii_profile[:len(y_plus)]

                    label = self._build_legend_label(kind_titles.get(kind, kind), case, timestep, suffix)
                    marker = marker_cycle[marker_index % len(marker_cycle)]
                    marker_index += 1
                    line_color = self._get_color(f'{case}|{timestep}|{kind}|{suffix}', kind)

                    # Connecting line + marker edge use a distinct colour per
                    # line (identity); marker fill stays colour-graded by
                    # wall distance (physical position) — see docstring.
                    ax.plot(iii_profile, -ii_profile, color=line_color, linewidth=1.0, zorder=1)
                    scatter = ax.scatter(
                        iii_profile, -ii_profile, c=y_plus, cmap='viridis_r',
                        marker=marker, s=32, label=label, zorder=2,
                        edgecolors=line_color, linewidths=1.0,
                    )

            if scatter is not None:
                cbar = fig.colorbar(scatter, ax=ax)
                color_label = self._get_y_profile_xlabel() if self.config.norm_y_to_y_plus else 'Distance from wall'
                cbar.set_label(color_label, fontsize=self._get_axis_label_fontsize())

            ax.set_title(f'{kind_titles.get(kind, kind)} Anisotropy', fontsize=self._get_title_fontsize())
            ax.set_xlabel('$III$', fontsize=self._get_axis_label_fontsize())
            ax.set_ylabel('$-II$', fontsize=self._get_axis_label_fontsize())
            ax.grid(True)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=self._get_legend_fontsize())
            self._apply_axis_text_style(ax)

        return fig

    def plot_spectrum(self, spectrum_computer: Optional[SpectrumComputer]):
        """1D streamwise-velocity-fluctuation energy spectrum: log-log E(k)
        vs k, one curve per (case, timestep, y-location).

        Not part of plot_by_class's grouped-statistics machinery — see
        SpectrumComputer's docstring — so this is called directly by
        main()/the GUI worker and merged into the same figures dict.
        """
        if spectrum_computer is None or not spectrum_computer.processed_results:
            return None

        self._reset_color_cycle()
        fig = Figure(figsize=(8, 6), constrained_layout=True)
        ax = fig.add_subplot(111)

        for (case, timestep), by_y in spectrum_computer.processed_results.items():
            for y_value, (k, E) in by_y.items():
                # k[0] = 0 (DC/mean component) can't be shown on a log-log axis.
                k_plot, E_plot = k[1:], E[1:]
                suffix = f' y={y_value:.3g}'
                label = self._build_legend_label(f'{spectrum_computer.direction}-spectrum', case, timestep, suffix)
                color = self._get_color(f'{case}|{timestep}|{y_value}', 'spectrum')
                linestyle = self._get_linestyle(case)
                ax.loglog(k_plot, E_plot, label=label, color=color, linestyle=linestyle)

        direction_label = f'$k_{spectrum_computer.direction}$'
        ax.set_title(f"Streamwise Velocity Fluctuation Spectrum ({spectrum_computer.direction}-direction)",
                     fontsize=self._get_title_fontsize())
        ax.set_xlabel(direction_label, fontsize=self._get_axis_label_fontsize())
        ax.set_ylabel('$E(k)$', fontsize=self._get_axis_label_fontsize())
        ax.grid(True, which='both')
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=self._get_legend_fontsize())
        self._apply_axis_text_style(ax)

        return fig

    def _plot_single_figure(self, statistics: List[Union[ReStresses, Profiles, Budget]],
                           reference_data: Optional[ReferenceData] = None):
        """Create a single combined plot for all statistics"""
        self._reset_color_cycle()
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        for stat in statistics:
            for (case, timestep), values in stat.processed_results.items():

                y_plus = self._get_y_plus(case, timestep)
                if y_plus is None:
                    continue

                profiles = self._extract_profiles(values)
                for suffix, profile in profiles:
                    color = self._get_line_color(case, timestep, stat.name, suffix)
                    label = self._build_legend_label(stat.label, case, timestep, suffix, include_stat_label=True)
                    linestyle = self._get_linestyle(case)
                    marker = self._get_marker(case)
                    self._plot_line(ax, y_plus, profile, label, color, linestyle=linestyle, marker=marker)

                if reference_data:
                    self._plot_reference_data(ax, stat.name, case, reference_data)

                if stat.name == 'ux_velocity' and self.config.ux_velocity_log_ref_on and self.config.log_y_scale:
                    self._plot_log_reference_lines(ax, y_plus)

        ax.set_xlabel(self._get_y_profile_xlabel(), fontsize=self._get_axis_label_fontsize())

        if len(statistics) == 1:
            ax.set_ylabel(
                self._get_stat_ylabel(statistics[0].name, statistics[0].label),
                fontsize=self._get_axis_label_fontsize()
            )
        else:
            if self.config.norm_by_u_tau_sq:
                ax.set_ylabel('Statistic value / $u_\\tau^2$', fontsize=self._get_axis_label_fontsize())
            else:
                ax.set_ylabel('Statistic value', fontsize=self._get_axis_label_fontsize())

        ax.legend(fontsize=self._get_legend_fontsize())
        if self.config.large_text_on:
            ax.tick_params(axis='x', labelsize=16)
            ax.tick_params(axis='y', labelsize=16)
        ax.grid(True)

        return fig

    def _plot_multi_figure(self, statistics: List[Union[ReStresses, Profiles, Budget]],
                          reference_data: Optional[ReferenceData] = None):
        """Create separate subplots for each statistic"""
        self._reset_color_cycle()
        n_stats = len(statistics)
        ncols = math.ceil(math.sqrt(n_stats))
        nrows = math.ceil(n_stats / ncols)

        fig = Figure(figsize=(15, 10), constrained_layout=True)
        axs = np.array(fig.subplots(nrows=nrows, ncols=ncols, squeeze=False))

        # Plot each statistic
        for i, stat in enumerate(statistics):
            row = i // ncols
            col = i % ncols
            ax = axs[row, col]

            for (case, timestep), values in stat.processed_results.items():

                # Get y coordinates
                y_plus = self._get_y_plus(case, timestep)
                if y_plus is None:
                    continue

                profiles = self._extract_profiles(values)
                for suffix, profile in profiles:
                    # Get plotting aesthetics
                    color = self._get_line_color(case, timestep, stat.name, suffix)
                    label = self._build_legend_label(stat.label, case, timestep, suffix)
                    linestyle = self._get_linestyle(case)
                    marker = self._get_marker(case)

                    # Plot main data
                    self._plot_line(ax, y_plus, profile, label, color, linestyle=linestyle, marker=marker)

                # Plot reference data
                if reference_data:
                    self._plot_reference_data(ax, stat.name, case, reference_data)

                # Add log scale reference lines
                if stat.name == 'ux_velocity' and self.config.ux_velocity_log_ref_on and self.config.log_y_scale:
                    self._plot_log_reference_lines(ax, y_plus)

            # Set subplot properties
            ax.set_title(f'{stat.label}', fontsize=self._get_title_fontsize())
            ax.set_ylabel(
                self._get_stat_ylabel(stat.name, stat.label),
                fontsize=self._get_axis_label_fontsize()
            )
            ax.grid(True)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=self._get_legend_fontsize())

            if row == nrows - 1:
                ax.set_xlabel(self._get_y_profile_xlabel(), fontsize=self._get_axis_label_fontsize())
            self._apply_axis_text_style(ax)

        # Hide unused subplots
        for i in range(n_stats, nrows * ncols):
            row = i // ncols
            col = i % ncols
            axs[row, col].set_visible(False)

        return fig

    # ------------------------------------------------------------------
    # 2-D surface (contour) plotting
    # ------------------------------------------------------------------
    def _plot_surface_figures(self, statistics, title: str) -> Dict[str, Any]:
        """Create filled-contour figures for each statistic that has 2-D data.

        Returns a dict ``{key: fig}`` where *key* encodes the class and
        stat name, suitable for passing to ``save_figure``.
        """
        x_coords = getattr(self.data_loader, 'x_coords', None)
        y_coords = getattr(self.data_loader, 'y_coords', None)
        if x_coords is None or y_coords is None:
            return {}

        figures: Dict[str, Any] = {}

        for stat in statistics:
            for (case, timestep), values in stat.processed_results.items():
                if values.ndim < 2:
                    continue  # nothing to surface-plot

                y = y_coords.copy()
                if self.config.half_channel_plot:
                    # Same wall-first convention as _get_y_plus: distance from
                    # the selected wall, 0 at the wall.
                    n = values.shape[0]
                    y = (1.0 - np.flip(y)[:n] if self.config.half_channel_side == 'upper'
                         else y[:n] + 1.0)

                X, Y = np.meshgrid(x_coords, y)

                fig = Figure(figsize=(12, 5), constrained_layout=True)
                ax = fig.add_subplot(111)
                cf = ax.contourf(X, Y, values, levels=64, cmap='RdBu_r')
                cbar = fig.colorbar(cf, ax=ax)
                cbar.set_label(stat.label, fontsize=self._get_axis_label_fontsize())
                if self.config.large_text_on:
                    cbar.ax.tick_params(labelsize=13)
                ax.set_xlabel('$x$', fontsize=self._get_axis_label_fontsize())
                ax.set_ylabel('Distance from wall' if self.config.half_channel_plot else '$y$',
                              fontsize=self._get_axis_label_fontsize())
                ax.set_title(f'{stat.label}  ({case}, t={timestep})', fontsize=self._get_title_fontsize())
                self._apply_axis_text_style(ax)

                key = f'{stat.name}_surface_{case}_{timestep}'
                figures[key] = fig

        return figures

    # ------------------------------------------------------------------
    # X-direction profile plotting
    # ------------------------------------------------------------------
    def _plot_x_profile_figure(self, statistics, title: str):
        """Create a figure with streamwise (x-direction) profiles for each statistic."""
        self._reset_color_cycle()
        x_coords = getattr(self.data_loader, 'x_coords', None)
        if x_coords is None:
            print('No x_coords available for x-direction profiles.')
            return None

        # Include stats that can produce streamwise profiles:
        # - native 2-D/3-D fields
        # - x_profile_only stats that are already reduced to 1-D x-profiles
        stats_with_x_profiles = [
            s for s in statistics
            if s.processed_results and any(
                (v.ndim >= 2) or (getattr(s, 'x_profile_only', False) and v.ndim == 1)
                for v in s.processed_results.values()
            )
        ]
        if not stats_with_x_profiles:
            return None

        # PeakGrowth stats (peak TKE + its normal-stress components) share
        # units, so they're more useful overlaid on one set of axes than
        # split into separate subplots.
        if all(isinstance(s, PeakGrowth) for s in stats_with_x_profiles):
            return self._plot_x_profile_single_figure(stats_with_x_profiles, title, x_coords)

        n_stats = len(stats_with_x_profiles)
        ncols = math.ceil(math.sqrt(n_stats))
        nrows = math.ceil(n_stats / ncols)

        fig = Figure(figsize=(15, 4 * nrows), constrained_layout=True)
        axs = np.array(fig.subplots(nrows=nrows, ncols=ncols, squeeze=False))

        for i, stat in enumerate(stats_with_x_profiles):
            ax = axs[i // ncols, i % ncols]
            for (case, timestep), values in stat.processed_results.items():
                profiles = self._extract_x_profiles(values, stat=stat)
                for suffix, profile in profiles:
                    color = self._get_line_color(case, timestep, stat.name, suffix)
                    label = self._build_legend_label(stat.label, case, timestep, suffix)
                    linestyle = self._get_linestyle(case)
                    marker = self._get_marker(case)
                    markevery = self._get_markevery(len(x_coords))
                    ax.plot(
                        x_coords,
                        profile,
                        label=label,
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        markersize=4,
                        markevery=markevery
                    )
            ax.set_title(stat.label, fontsize=self._get_title_fontsize())
            ax.set_xlabel('$x$', fontsize=self._get_axis_label_fontsize())
            ax.set_ylabel(
                self._get_stat_ylabel(stat.name, stat.label),
                fontsize=self._get_axis_label_fontsize()
            )
            ax.grid(True)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=self._get_legend_fontsize())
            self._apply_axis_text_style(ax)

        for i in range(n_stats, nrows * ncols):
            axs[i // ncols, i % ncols].set_visible(False)

        return fig

    def _plot_x_profile_single_figure(self, statistics, title: str, x_coords):
        """Plot every statistic's x-profile together on one shared set of axes."""
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        for stat in statistics:
            for (case, timestep), values in stat.processed_results.items():
                profiles = self._extract_x_profiles(values, stat=stat)
                for suffix, profile in profiles:
                    color = self._get_line_color(case, timestep, stat.name, suffix)
                    label = self._build_legend_label(stat.label, case, timestep, suffix, include_stat_label=True)
                    linestyle = self._get_linestyle(case)
                    marker = self._get_marker(case)
                    markevery = self._get_markevery(len(x_coords))
                    ax.plot(
                        x_coords,
                        profile,
                        label=label,
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        markersize=4,
                        markevery=markevery
                    )

        ax.set_title(title, fontsize=self._get_title_fontsize())
        ax.set_xlabel('$x$', fontsize=self._get_axis_label_fontsize())
        if len(statistics) == 1:
            ax.set_ylabel(self._get_stat_ylabel(statistics[0].name, statistics[0].label),
                          fontsize=self._get_axis_label_fontsize())
        elif self.config.norm_by_u_tau_sq:
            ax.set_ylabel('$E_k$ / $u_\\tau^2$', fontsize=self._get_axis_label_fontsize())
        else:
            ax.set_ylabel('$E_k$', fontsize=self._get_axis_label_fontsize())
        ax.grid(True)
        ax.legend(fontsize=self._get_legend_fontsize())
        self._apply_axis_text_style(ax)
        return fig

    def _plot_line(self, ax, x: np.ndarray, y: np.ndarray, label: str, color: str,
                   linestyle='-', marker='') -> None:
        """Plot a single line with appropriate scale"""
        markevery = self._get_markevery(len(x)) if marker else None
        plot_kwargs = {
            'label': label,
            'linestyle': linestyle,
            'marker': marker,
            'color': color
        }
        if marker:
            plot_kwargs['markersize'] = 4
            plot_kwargs['markevery'] = markevery

        ax.plot(x, y, **plot_kwargs)
        self._apply_xscale(ax)

    def _apply_xscale(self, ax) -> None:
        if self.config.log_y_scale:
            ax.set_xscale('log')

    def _get_markevery(self, n_points: int):
        """Return marker spacing that appears visually uniform on the plotted curve."""
        if n_points <= 12:
            return 1
        # Float spacing uses display-space distance, which keeps marker spacing
        # visually even for both linear and semilog plots.
        return (0.0, 1.0 / 11.0)

    def _get_linestyle(self, case: str):
        """Return a consistent line style for each case."""
        return self.case_linestyles.get(case, '-')

    def _get_marker(self, case: str):
        """Return a consistent marker for each case."""
        return self.case_markers.get(case, 'o')

    def _plot_reference_data(self, ax, stat_name: str, case: str,
                            reference_data: ReferenceData) -> None:
        """Plot reference data for a given statistic"""
        # MKM180 reference
        if self.config.mkm180_ch_ref_on and reference_data.mkm180_stats:
            if stat_name in reference_data.mkm180_stats:
                self._plot_line(ax, reference_data.mkm180_y_plus,
                              reference_data.mkm180_stats[stat_name],
                              f'{self.plot_config.stat_labels[stat_name]} MKM180',
                              'black')

        # Noguchi & Kasagi reference — overlays the configured Hartmann-number
        # curve (mhd_NK_ref_case) for every plotted case, regardless of the
        # case's own name.
        if self.config.mhd_NK_ref_on:
            if self.config.mhd_NK_ref_case == 'Ha_4' and reference_data.NK_H4_stats and stat_name in reference_data.NK_H4_stats:
                ax.plot(reference_data.NK_ref_y_H4,
                        reference_data.NK_H4_stats[stat_name],
                        linestyle='', marker='o',
                        label='Ha = 4, Noguchi & Kasagi',
                        color=self.plot_config.colors_1[stat_name], markevery=2)

            elif self.config.mhd_NK_ref_case == 'Ha_6' and reference_data.NK_H6_stats and stat_name in reference_data.NK_H6_stats:
                ax.plot(reference_data.NK_ref_y_H6,
                        reference_data.NK_H6_stats[stat_name],
                        linestyle='', marker='o',
                        label='Ha = 6, Noguchi & Kasagi',
                        color=self.plot_config.colors_2[stat_name], markevery=2)

    def _plot_log_reference_lines(self, ax, y_plus: np.ndarray) -> None:
        """Plot reference lines for log-scale velocity plots"""
        if self.config.log_y_scale:
            ax.plot(y_plus[:15], y_plus[:15], '--', linewidth=1,
                    label='$u^+ = y^+$', color='black', alpha=0.5)
            u_plus_ref = 2.5 * np.log(y_plus) + 5.5
            ax.plot(y_plus, u_plus_ref, '--', linewidth=1,
                    label='$u^+ = 2.5ln(y^+) + 5.5$', color='black', alpha=0.5)
            self._apply_xscale(ax)

    def _get_y_plus(self, case: str, timestep: str, as_wall_distance: bool = False) -> Optional[np.ndarray]:
        """Calculate y, y+, or wall-distance coordinates for a case.

        Full channel: y spans [-1, 1] as loaded.
        Half channel: always distance from the selected wall (0 at the wall,
        increasing toward the centreline), so 'lower' and 'upper' plot on the
        same axis and can be compared directly. 'average' is indexed from the
        lower wall, matching symmetric_average's output.

        `as_wall_distance` forces a 0-at-the-wall, increasing-toward-
        centreline return even when norm_y_to_y_plus is off — for callers
        like the Lumley triangle that colour points by distance from the
        wall, where the raw signed y (which is +1 *at* the wall for
        'upper') would be a direction-flipped, misleading colour scale.
        """
        ux_data = self.data_loader.get(case, 'u1', timestep)
        if ux_data is None:
            print(f'Missing u1 data for plotting: {case}, {timestep}')
            return None

        y_coords = getattr(self.data_loader, 'y_coords', None)
        y = y_coords.copy() if y_coords is not None else ux_data[:, 1]

        if self.config.half_channel_plot:
            half_len = len(y) - len(y) // 2
            side = self.config.half_channel_side
            y = np.flip(y)[:half_len] if side == 'upper' else y[:half_len]
            wall_distance = (1.0 - y) if side == 'upper' else (y + 1.0)
        else:
            wall_distance = y

        if self.config.norm_y_to_y_plus:
            cur_Re = op.get_Re(case, self.config.cases, self.config.Re, ux_data,
                               self.config.forcing, y_coords=y_coords)
            return op.norm_y_to_y_plus(wall_distance, ux_data, cur_Re, y_coords=y_coords)
        elif as_wall_distance or self.config.half_channel_plot:
            return wall_distance
        else:
            return y

    def _get_color(self, key: str, stat_name: Optional[str] = None) -> str:
        """Get a visible colour for a plotted series.

        Budget terms use fixed colours from budget_colors so they are always
        consistent. All other series are assigned sequentially from the visible
        palette so lines are distinct within a plot before any colour reuse.
        """
        if key in self._series_color_map:
            return self._series_color_map[key]

        budget_colors = self.plot_config.budget_colors
        if stat_name and stat_name in budget_colors:
            color = budget_colors[stat_name]
        else:
            palette = self.plot_config.visible_palette
            index = self._next_color_index % len(palette)
            color = palette[index]
            self._next_color_index += 1

        self._series_color_map[key] = color
        return color

    def save_figure(self, fig, suffix: str = '') -> None:
        """Save figure to file"""
        if self.config.plot_name:
            base_name = self.config.plot_name.rsplit('.', 1)[0]
            ext = self.config.plot_name.rsplit('.', 1)[1] if '.' in self.config.plot_name else 'png'
            filename = f'{base_name}{suffix}.{ext}' if suffix else self.config.plot_name
        else:
            filename = f'turb_stats_plot{suffix}.png' if suffix else 'turb_stats_plot.png'
        if self.config.save_to_path and self.config.folder_path:
            os.makedirs(self.config.folder_path, exist_ok=True)
            save_path = os.path.join(self.config.folder_path, filename)
            fig.savefig(save_path,
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.1,
                        facecolor='white',
                        edgecolor='none',
                        transparent=True,
                        orientation='landscape')
            print(f'Figure saved to {save_path}')
        elif self.config.save_to_path and not self.config.folder_path:
            print('save_to_path is enabled but folder_path is empty; using turb_stats_plots fallback.')

        output_dir = os.path.join(os.getcwd(), 'turb_stats_plots')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        fig.savefig(output_path,
                   dpi=300,
                   bbox_inches='tight',
                   pad_inches=0.1,
                   facecolor='white',
                   edgecolor='none',
                   transparent=True,
                   orientation='landscape')
        print(f'Figure saved to {output_path}')

    def save_figures_by_class(self, figures: Dict[str, Any]) -> None:
        """Save multiple figures, one for each class type"""
        for class_name, fig in figures.items():
            suffix = f'_{class_name.lower()}'
            self.save_figure(fig, suffix)

    def display_figure(self) -> None:
        """Display figure"""
        print('Displaying figure...')
        import matplotlib.pyplot as plt
        plt.show()



# =====================================================================================================================================================
# MAIN EXECUTION
# =====================================================================================================================================================

def main():
    """Main execution function"""
    # Import configuration
    import config as config_module
    config = Config.from_module(config_module)

    print("="*120)
    print("TURBULENCE STATISTICS PROCESSING")
    print("="*120)

    # Load turbulence data using the appropriate loader
    print("\nLoading turbulence data...")
    data_loader = create_data_loader(config)
    data_loader.load_all()

    # Load reference data
    print("\nLoading reference data...")
    reference_data = ReferenceData(config)
    reference_data.load_all()

    # Compute statistics
    print("\nComputing turbulence statistics...")
    pipeline = TurbulenceStatsPipeline(config, data_loader)
    pipeline.compute_all()

    # Process statistics (normalize, average)
    print("\nProcessing statistics...")
    print("\nFlow info:\n")
    pipeline.process_all()

    # Plot results
    if config.display_fig or config.save_fig:
        print("\nGenerating plots...")
        plot_config = PlotConfig()
        plotter = TurbulencePlotter(config, plot_config, data_loader)

        # Get statistics grouped by class type
        grouped_stats = pipeline.get_statistics_by_class()

        # Create separate figures for each class
        figures = plotter.plot_by_class(grouped_stats, reference_data)

        # Spectral analysis isn't part of the grouped-statistics machinery
        # (see SpectrumComputer's docstring) — plotted separately.
        spectrum_fig = plotter.plot_spectrum(pipeline.spectrum_computer)
        if spectrum_fig is not None:
            figures['Spectrum'] = spectrum_fig

        if config.save_fig:
            if figures:
                plotter.save_figures_by_class(figures)
            else:
                print('No figures were generated for the current profile_direction/stat configuration.')

        if config.display_fig:
            plotter.display_figure()
    else:
        print('\nNo output option selected')

    print("\n" + "="*120)
    print("PROCESSING COMPLETE")
    print("="*120)


if __name__ == '__main__':
    main()
