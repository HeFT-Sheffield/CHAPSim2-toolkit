# Thermal property tables

`NIST_WATER_23.5MP.DAT` and `NIST_CO2_8MP.DAT` are the property tables
CHAPSim2 reads for its two supercritical fluids. They are copied verbatim
from `CHAPSim2/validation/thermal_properties/` so the toolkit can evaluate
properties for a case whose folder does not carry its own copy.

Columns, as `buildup_property_relations_from_table` in the solver's
`src/input_thermo.f90` reads them:

| column | quantity | unit |
|---|---|---|
| 1 | pressure | MPa |
| 2 | enthalpy | J/kg |
| 3 | temperature | K |
| 4 | density | kg/m³ |
| 5 | dynamic viscosity | Pa·s |
| 6 | thermal conductivity | W/(m·K) |
| 7 | specific heat at constant pressure | J/(kg·K) |
| 8 | thermal expansion coefficient | 1/K |

| file | fluid | pressure | range | rows |
|---|---|---|---|---|
| `NIST_WATER_23.5MP.DAT` | water | 23.5 MPa | 573.15–873.15 K | 5001 |
| `NIST_CO2_8MP.DAT` | CO₂ | 8 MPa | 220–520 K | 1501 |

**These are a fallback.** `fluid_properties.find_property_table` looks in the
case folder first, because the solver opens the table by bare filename from
its working directory — so the authoritative table for a run is the one in
that run's folder. A case post-processed against a different table than it
was run with would give properties that never existed in the simulation.

Data from the NIST Chemistry WebBook / REFPROP. Copyright for the source
data remains with NIST; see <https://webbook.nist.gov/chemistry/fluid/>.
