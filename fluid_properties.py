"""Fluid properties, as CHAPSim2 computes them.

This is a Python port of the property model in the solver's
``src/input_thermo.f90``, with the coefficients from ``src/modules.f90``.
The point of porting rather than reimplementing is that post-processing
has to agree with the run: a Nusselt number computed from a different
conductivity than the solver used is not a measurement of that run.

The solver has two property states, and so does this:

``IPROPERTY_FUNCS``
    A liquid metal, evaluated from polynomial correlations in T.
    Sodium, lead, bismuth, LBE, lithium, FLiBe and PbLi-17.

``IPROPERTY_TABLE``
    Supercritical water or CO2, interpolated from a NIST table of
    (P, H, T, D, M, K, Cp, Beta). The solver reads the table from its
    working directory - the case folder - so that is where this looks
    first, falling back to the copies bundled under
    ``Reference_Data/thermal_properties/``.

Everything here is dimensional and in SI, with no exceptions:

    temperature             K
    density                 kg/m3
    dynamic viscosity       Pa s
    thermal conductivity    W/(m K)
    specific heat           J/(kg K)
    specific enthalpy       J/kg
    thermal expansion       1/K
    pressure                Pa

The NIST tables state pressure in MPa and the solver keeps it that way;
it is converted on read, so nothing downstream has to remember which.
The solver non-dimensionalises by its reference state afterwards; that is
a separate step and is not done here.

Outside a fluid's valid range the solver stops with "temperature exceeds
specified range". Stopping is not useful in post-processing - a single
bad cell would lose the whole plot - so the equivalent check here returns
NaN for the offending points and the caller can see them as gaps.
"""

import os
import warnings

import numpy as np

__all__ = ['get_fluid_properties', 'FLUIDS', 'TABLE_FLUIDS',
           'FunctionProperties', 'TableProperties', 'PropertyRangeWarning']


#: Directory holding the bundled NIST tables.
REFERENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'Reference_Data', 'thermal_properties')

#: The table filenames the solver looks for, from modules.f90.
INPUT_SCP_WATER = 'NIST_WATER_23.5MP.DAT'
INPUT_SCP_CO2 = 'NIST_CO2_8MP.DAT'


class PropertyRangeWarning(UserWarning):
    """A temperature was asked for outside the fluid's valid range."""


#: (fluid, quantity) pairs already warned about, so a profile of 5000
#: points produces one warning rather than 5000.
_WARNED = set()


# ---------------------------------------------------------------------------
# Correlation coefficients, from CHAPSim2/src/modules.f90
# ---------------------------------------------------------------------------
#
# Transcribed rather than paraphrased, keeping the solver's own names, so
# the two can be diffed when the solver changes. The forms are:
#
#   D  = CoD[0] + CoD[1] * T                       (lithium differs, below)
#   K  = CoK[0] + CoK[1] * T + CoK[2] * T^2
#   B  = 1 / (CoB - T)
#   Cp = CoCp[-2]*T^-2 + CoCp[-1]*T^-1 + CoCp[0] + CoCp[1]*T + CoCp[2]*T^2
#   H  = HM0 + CoH[-1]*(1/T - 1/TM0) + CoH[0] + CoH[1]*(T - TM0)
#                                    + CoH[2]*(T^2 - TM0^2)
#                                    + CoH[3]*(T^3 - TM0^3)
#   M  : per fluid, see viscosity() below
#
# CoCp is stored from index -2, as in the Fortran.
#
# CoH is not stored at all: the solver derives it from CoCp so that
# dH/dT = Cp holds exactly, and _derived_CoH below does the same, for the
# same reason. Transcribing both is how they came to disagree - the
# sodium CoH in modules.f90 was LBE's until October 2026.
#
# TP0max is the top of the interval the correlations are evaluated over.
# It is TB0 - melting to boiling is the single-phase liquid range - except
# where a shipped fit is known not to hold that far; see pbli.

_COEFFICIENTS = {
    'sodium': dict(
        TM0=371.0, TB0=1155.0, HM0=113.0e3,
        CoD=[1014.0, -0.235],
        CoK=[104.0, -0.047, 0.0],
        CoB=4316.0,
        CoCp=[-3.001e6, 0.0, 1658.0, -0.8479, 4.454e-4],
        CoM=[556.835, -6.4406, -0.3958], m_form='exp_ln',
    ),
    'lead': dict(
        TM0=600.6, TB0=2021.0, HM0=23.07e3,
        CoD=[11441.0, -1.2795],
        CoK=[9.2, 0.011, 0.0],
        CoB=8942.0,
        CoCp=[-1.524e6, 0.0, 176.2, -4.923e-2, 1.544e-5],
        CoM=[1069.0, 4.55e-4, 0.0], m_form='arrhenius',
    ),
    'bismuth': dict(
        TM0=544.6, TB0=1831.0, HM0=53.3e3,
        CoD=[10725.0, -1.22],
        CoK=[7.34, 9.5e-3, 0.0],
        CoB=8791.0,
        CoCp=[7.183e6, 0.0, 118.2, 5.934e-3, 0.0],
        CoM=[780.0, 4.456e-4, 0.0], m_form='arrhenius',
    ),
    'lbe': dict(
        TM0=398.0, TB0=1927.0, HM0=38.6e3,
        # Gives 10031 kg/m3 at 800 K, against the OECD/NEA handbook's
        # 10037 from rho = 11096 - 1.3236 T: 0.06%. The solver carried a
        # +1.293 here until October 2026, which made LBE denser as it was
        # heated; runs made before that fix disagree with this.
        CoD=[11065.0, -1.293],
        CoK=[3.284, 1.617e-2, -2.305e-6],
        CoB=8558.0,
        CoCp=[-4.56e5, 0.0, 164.8, -3.94e-2, 1.25e-5],
        # modules.f90 marks this one "check, WRong from literature."
        CoM=[754.1, 4.94e-4, 0.0], m_form='arrhenius',
    ),
    'lithium': dict(
        TM0=453.65, TB0=1615.0, HM0=4.55e5,
        # Lithium alone uses the five-coefficient form
        # D = CoD[0] + CoD[1]*T + CoD[2]*(1 - T/CoD[3])^CoD[4]
        CoD=[278.5, -0.04657, 274.6, 3500.0, 0.467],
        CoK=[22.28, 0.0500, -1.243e-5],
        CoB=5620.0,
        CoCp=[0.0, 0.0, 4754.0, -9.25e-1, 2.91e-4],
        CoM=[-4.164, -6.374e-1, 2.921e2], m_form='lithium',
    ),
    'flibe': dict(
        TM0=732.1, TB0=1703.0, HM0=17.47e5,
        CoD=[2413.03, -0.4884],
        CoK=[1.1, 0.0, 0.0],
        CoB=4940.7,
        CoCp=[0.0, 0.0, 2386.0, 0.0, 0.0],
        CoM=[4022.0, 7.803e-5, 0.0], m_form='arrhenius',
    ),
    'pbli': dict(
        TM0=508.0, TB0=1943.0, HM0=33.9e3,
        CoD=[10520.4, -1.1905],
        CoK=[9.148, 1.963e-2, 0.0],
        CoB=8836.8,
        CoCp=[0.0, 0.0, 195.0, -9.116e-3, 0.0],
        # M = CoM[0] + CoM[1]*T + CoM[2]*T^2 + CoM[3]*T^3
        # The viscosity cubic falls through zero at 858.996 K and is
        # increasingly negative above it. The solver caps the property
        # range at 850 K for this fluid alone rather than let that happen;
        # the bound comes from the fit's own root, not from a literature
        # validity range, because none is recorded for this polynomial.
        TP0max=850.0,
        CoM=[0.0061091, -2.2574e-5, 3.766e-8, -2.2887e-11], m_form='cubic',
    ),
}

#: Table-based fluids and the file the solver reads for each.
TABLE_FLUIDS = {
    'scp_water': INPUT_SCP_WATER,
    'scp_co2': INPUT_SCP_CO2,
}

#: Every fluid this module can produce properties for.
FLUIDS = sorted(list(_COEFFICIENTS) + list(TABLE_FLUIDS))

#: Spellings accepted for each fluid, including the solver's ifluid tokens
#: and the short forms the toolkit's own scripts have always taken.
_ALIASES = {
    'sodium': ('na', 'sodium', 'liquid_sodium'),
    'lead': ('pb', 'lead', 'liquid_lead'),
    'bismuth': ('bi', 'bismuth', 'liquid_bismuth'),
    'lbe': ('lbe', 'pb-bi', 'pbbi', 'liquid_lbe'),
    'lithium': ('li', 'lithium', 'liquid_lithium'),
    'flibe': ('flibe', 'fli-be', '2lif-bef2', 'liquid_flibe'),
    'pbli': ('pbli', 'pb-li', 'pbli17', 'pb17li', 'liquid_pbli'),
    'scp_water': ('scp_water', 'water', 'h2o', 'supercritical_water'),
    'scp_co2': ('scp_co2', 'co2', 'supercritical_co2'),
}

_BY_ALIAS = {alias: name for name, aliases in _ALIASES.items()
             for alias in aliases}


class _Properties:
    """What every property object provides, whichever state it is in.

    The method names are the ones the toolkit already used, so existing
    callers keep working; the derived quantities are the solver's own
    definitions from ftp_get_thermal_properties_dimensional_from_T.
    """

    name = ''
    T_melt = None
    T_boil = None

    # -- the five the rest of the toolkit asks for --------------------------
    def density_mass(self, T):
        raise NotImplementedError

    def viscosity(self, T):
        raise NotImplementedError

    def thermal_conductivity(self, T):
        raise NotImplementedError

    def heat_capacity_p(self, T):
        raise NotImplementedError

    def coeff_vol_exp(self, T):
        raise NotImplementedError

    def enthalpy(self, T):
        raise NotImplementedError

    # -- derived, exactly as the solver derives them ------------------------
    def electrical_conductivity(self, T):
        """The solver sets sigma_e to 1 for every fluid and both states.

        MHD cases are run in terms of the Stuart or Hartmann number, which
        already carries the conductivity, so the property itself is unity
        by construction rather than unknown.
        """
        return np.ones_like(np.asarray(T, dtype=float))

    def prandtl(self, T):
        """Pr = mu * Cp / k."""
        return self.viscosity(T) * self.heat_capacity_p(T) \
            / self.thermal_conductivity(T)

    def thermal_diffusivity(self, T):
        """alpha = k / (rho * Cp)."""
        return self.thermal_conductivity(T) \
            / self.density_mass(T) / self.heat_capacity_p(T)

    def temperature_from_enthalpy(self, h, ref_temp=None):
        """Invert h(T), as the solver's ftp_refresh_thermal_properties_from_H.

        The solver inverts by binary search and linear interpolation over
        the same list it interpolates T with, so the round trip T -> h -> T
        is exact at a tabulated point and consistent between them. This
        interpolates over the same points for the same reason.

        Args:
            h: specific enthalpy in J/kg, or non-dimensional if ref_temp is
                given.
            ref_temp: when set, h is taken to be the solver's
                non-dimensional enthalpy, (h - h0)/(T0 cp0) with the
                reference state at this temperature, and is dimensionalised
                before inversion. This is the form the solver writes to
                file, so it is the form that arrives from the data.

        Returns:
            Temperature in K. NaN where h is outside the range the fluid
            covers, for the same reason the forward direction gives NaN.
        """
        h = np.asarray(h, dtype=float)
        if ref_temp is not None:
            h0 = self.enthalpy(ref_temp)
            cp0 = self.heat_capacity_p(ref_temp)
            h = h * ref_temp * cp0 + h0
        T_points, h_points = self._enthalpy_curve()
        value = np.interp(h, h_points, T_points)
        out = np.where((h >= h_points[0]) & (h <= h_points[-1]), value, np.nan)
        return out if out.ndim else float(out)

    def _enthalpy_curve(self):
        """(T, h) sorted by increasing h, for inverting the relation."""
        raise NotImplementedError

    def phase(self, T, P=None):
        """Solid, Liquid or Vapour by the melting and boiling points.

        Only meaningful for the liquid metals; a supercritical fluid has
        no boiling point and reports Supercritical throughout.
        """
        if self.T_melt is None:
            return 'Supercritical'
        if T < self.T_melt:
            return 'Solid'
        if T < self.T_boil:
            return 'Liquid'
        return 'Vapor'

    def is_in_range(self, T):
        """True where T is inside the range the properties are valid over."""
        T = np.asarray(T, dtype=float)
        return (T >= self.T_min) & (T <= self.T_max)

    def _masked(self, T, values):
        """NaN outside the valid range, as a visible gap.

        The solver stops the run here. Post-processing cannot: one cell
        out of range would cost the whole figure. A NaN plots as a gap and
        propagates into any average, which is the honest outcome.
        """
        T = np.asarray(T, dtype=float)
        out = np.where(self.is_in_range(T), values, np.nan)
        return out if out.ndim else float(out)

    def _positive(self, T, values, quantity):
        """NaN where a correlation returns a value that cannot be real.

        Some of the solver's fits are extrapolated past where they hold:
        the PbLi-17 viscosity cubic goes negative above about 859 K, which
        is inside the range the solver treats as valid. A negative
        viscosity is not a number to carry into a Reynolds or Prandtl
        number, so it becomes a NaN and a warning once per fluid and
        quantity.
        """
        values = self._masked(T, values)
        bad = np.asarray(values) <= 0.0
        if bad.any():
            key = (self.name, quantity)
            if key not in _WARNED:
                _WARNED.add(key)
                T = np.asarray(T, dtype=float)
                where = T[bad] if T.shape == bad.shape else T
                warnings.warn(
                    f'{self.name}: the {quantity} correlation returns a '
                    f'non-physical value at '
                    f'{np.nanmin(where):g}-{np.nanmax(where):g} K; those '
                    f'points are NaN. The correlation is being used outside '
                    f'the range it holds over.',
                    PropertyRangeWarning, stacklevel=3)
            values = np.where(bad, np.nan, values)
        return values if np.ndim(values) else float(values)


class FunctionProperties(_Properties):
    """A liquid metal, from the polynomial correlations (IPROPERTY_FUNCS)."""

    #: The solver turns a correlation fluid into a table of this many
    #: points before it inverts anything (N_FUNC2TABLE in input_thermo.f90).
    N_FUNC2TABLE = 1024

    def __init__(self, name):
        self.name = name
        c = _COEFFICIENTS[name]
        self._c = c
        self.T_melt = self.T_min = c['TM0']
        self.T_boil = c['TB0']
        # Melting to boiling, unless the fluid's fit is known not to reach
        # boiling. T_boil stays the physical value; T_max is how far the
        # correlations may be trusted, and they are not the same question.
        self.T_max = c.get('TP0max', c['TB0'])
        self.H_melt = c['HM0']
        self._curve = None

    @property
    def CoH(self):
        """Enthalpy coefficients, integrated term by term from CoCp.

        The solver derives these rather than transcribing them, so that
        dH/dT = Cp holds exactly; this does the same. CoCp[-1] would
        integrate to a ln(T) term the H polynomial has no slot for, so it
        has to stay zero - it is zero for every fluid here.
        """
        cp = self._c['CoCp']
        if cp[1] != 0.0:
            raise ValueError(
                f'{self.name}: CoCp[-1] is non-zero, which integrates to a '
                f'ln(T) term this enthalpy form cannot represent')
        return [-cp[0], 0.0, cp[2], cp[3] / 2.0, cp[4] / 3.0]

    def __repr__(self):
        return (f'<{type(self).__name__} {self.name} '
                f'{self.T_min:g}-{self.T_max:g} K>')

    def density_mass(self, T):
        T = np.asarray(T, dtype=float)
        d = self._c['CoD']
        if len(d) == 5:
            # Lithium alone: D = CoD0 + CoD1*T + CoD2*(1 - T/CoD3)^CoD4.
            # The solver selects on ifluid; the coefficient count says the
            # same thing without a second place to keep the list of which
            # fluid is special.
            value = d[0] + d[1] * T + d[2] * (1.0 - T / d[3]) ** d[4]
        else:
            value = d[0] + d[1] * T
        return self._positive(T, value, 'density')

    def thermal_conductivity(self, T):
        T = np.asarray(T, dtype=float)
        k = self._c['CoK']
        return self._masked(T, k[0] + k[1] * T + k[2] * T ** 2)

    def coeff_vol_exp(self, T):
        T = np.asarray(T, dtype=float)
        return self._masked(T, 1.0 / (self._c['CoB'] - T))

    def heat_capacity_p(self, T):
        T = np.asarray(T, dtype=float)
        cp = self._c['CoCp']            # indices -2, -1, 0, 1, 2
        value = (cp[0] * T ** -2 + cp[1] * T ** -1 + cp[2]
                 + cp[3] * T + cp[4] * T ** 2)
        return self._masked(T, value)

    def enthalpy(self, T):
        """Specific enthalpy in J/kg, on the solver's own datum.

        The solver's form, not an independent integration of Cp: the two
        differ, and the one that matters is the one the run used.
        """
        T = np.asarray(T, dtype=float)
        h = self.CoH                    # indices -1, 0, 1, 2, 3
        t0 = self._c['TM0']
        value = (self._c['HM0']
                 + h[0] * (1.0 / T - 1.0 / t0)
                 + h[1]
                 + h[2] * (T - t0)
                 + h[3] * (T ** 2 - t0 ** 2)
                 + h[4] * (T ** 3 - t0 ** 3))
        return self._masked(T, value)

    def _enthalpy_curve(self):
        if self._curve is None:
            T = np.linspace(self.T_min, self.T_max, self.N_FUNC2TABLE)
            h = np.asarray(self.enthalpy(T), dtype=float)
            order = np.argsort(h)
            self._curve = (T[order], h[order])
        return self._curve

    def viscosity(self, T):
        T = np.asarray(T, dtype=float)
        m = self._c['CoM']
        form = self._c['m_form']
        if form == 'exp_ln':            # sodium (and the solver's default)
            value = np.exp(m[0] / T + m[1] + m[2] * np.log(T))
        elif form == 'arrhenius':       # lead, bismuth, LBE, FLiBe
            value = m[1] * np.exp(m[0] / T)
        elif form == 'lithium':
            value = np.exp(m[0] + m[1] * np.log(T) + m[2] / T)
        elif form == 'cubic':           # PbLi-17
            value = m[0] + m[1] * T + m[2] * T ** 2 + m[3] * T ** 3
        else:
            raise ValueError(f'unknown viscosity form {form!r}')
        return self._positive(T, value, 'viscosity')


class TableProperties(_Properties):
    """Supercritical water or CO2, from a NIST table (IPROPERTY_TABLE).

    The solver reads the table, sorts it by temperature and interpolates
    linearly between the two bracketing rows. Linear is not an
    approximation chosen here - it is what the run did, and near the
    pseudo-critical peak the choice of interpolation visibly changes Cp.
    """

    #: Column order of the .DAT files, from buildup_property_relations_from_table.
    COLUMNS = ('P', 'h', 't', 'd', 'm', 'k', 'cp', 'b')

    def __init__(self, name, path):
        self.name = name
        self.path = path
        table = np.loadtxt(path, skiprows=1)
        if table.ndim != 2 or table.shape[1] != len(self.COLUMNS):
            raise ValueError(
                f'{path}: expected {len(self.COLUMNS)} columns '
                f'{self.COLUMNS}, found {table.shape[-1] if table.ndim else 0}')
        # Sorted small to big in T, as ftplist_sort_t_small2big does.
        table = table[np.argsort(table[:, 2])]
        self._table = table
        # The file states pressure in MPa; everything this module exposes
        # is SI, so it is converted once here rather than remembered.
        self.pressure = float(table[0, 0]) * 1e6    # Pa
        self.T_min = float(table[0, 2])
        self.T_max = float(table[-1, 2])
        self.T_melt = self.T_boil = None            # supercritical: neither

    def __repr__(self):
        return (f'<{type(self).__name__} {self.name} '
                f'{self.pressure / 1e6:g} MPa {self.T_min:g}-{self.T_max:g} K '
                f'{len(self._table)} rows from {os.path.basename(self.path)}>')

    def _interpolate(self, T, column):
        T = np.asarray(T, dtype=float)
        index = self.COLUMNS.index(column)
        value = np.interp(T, self._table[:, 2], self._table[:, index])
        return self._masked(T, value)

    def density_mass(self, T):
        return self._interpolate(T, 'd')

    def viscosity(self, T):
        return self._interpolate(T, 'm')

    def thermal_conductivity(self, T):
        return self._interpolate(T, 'k')

    def heat_capacity_p(self, T):
        return self._interpolate(T, 'cp')

    def coeff_vol_exp(self, T):
        return self._interpolate(T, 'b')

    def enthalpy(self, T):
        """Specific enthalpy in J/kg, straight from the table."""
        return self._interpolate(T, 'h')

    def _enthalpy_curve(self):
        order = np.argsort(self._table[:, 1])
        return self._table[order, 2], self._table[order, 1]


def find_property_table(filename, case_dir=None):
    """Locate a NIST table: the case's own copy first, then the bundled one.

    The solver opens the table by bare filename from its working
    directory, so the authoritative copy for a run is the one sitting in
    that case folder. Every case in CHAPSim2's test suite carries one, and
    they are identical to the bundled copy today - but a case run against
    a different table should be post-processed against that table, not
    against ours.
    """
    candidates = []
    if case_dir:
        candidates.append(os.path.join(case_dir, filename))
    candidates.append(os.path.join(REFERENCE_DIR, filename))
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f'{filename} not found. Looked in: ' + ', '.join(candidates))


def get_fluid_properties(medium, case_dir=None):
    """Return a property object for a fluid, by any of its accepted names.

    Args:
        medium: a fluid name - the solver's ifluid token ('scp_water',
            'liquid_sodium'), the toolkit's short name ('na') or the plain
            one ('sodium').
        case_dir: the case folder, searched first for a NIST table so a
            supercritical case is post-processed against the table it was
            run with.
    """
    if medium is None:
        raise ValueError(
            'No working fluid given. It is normally read from the case\'s '
            'input_chapsim.ini; set working_fluid explicitly if there is '
            f'no input file. Known fluids: {", ".join(FLUIDS)}.')
    name = _BY_ALIAS.get(str(medium).strip().lower())
    if name is None:
        raise ValueError(
            f'Unknown fluid {medium!r}. Known fluids: {", ".join(FLUIDS)}.')
    if name in TABLE_FLUIDS:
        return TableProperties(
            name, find_property_table(TABLE_FLUIDS[name], case_dir))
    return FunctionProperties(name)
