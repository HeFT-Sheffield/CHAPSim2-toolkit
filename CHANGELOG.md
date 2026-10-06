# Changelog

Notable changes to the CHAPSim2 toolkit. Entries that change a number you
might already have published are marked **behaviour change** — those are
the ones worth reading before you regenerate a figure.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [semantic versioning](https://semver.org/spec/v2.0.0.html),
with the caveat that the toolkit is pre-1.0 and the API may still move.

## [Unreleased]

## [0.2.0] — 2026-10-06

The first release intended for people other than its authors, alongside
CHAPSim2. Everything below followed from one problem: the toolkit had been
written against an older CHAPSim2 output layout and no longer read the
current one.

### Added

- **Package layout.** The code lives in `chapsim2_toolkit/`, installable
  with `pip install -e .`, with console scripts (`chapsim2-turbstats`,
  `chapsim2-gui`, …) and module entry points
  (`python -m chapsim2_toolkit.turb_stats`). Running from a clone with
  `python gui.py` is unchanged and still supported — the top-level scripts
  are wrappers onto the package.
- **Figure provenance.** Every saved figure records the toolkit version,
  git revision, whether the tree was dirty, the cases, timesteps and
  `config.py` used, in the file's own metadata. Read it back with
  `python -m chapsim2_toolkit.provenance figure.png`.
- **Supercritical fluid properties.** Water and CO₂ from the same NIST
  tables the solver reads, taken from the case folder when it has one.
  Previously the toolkit had liquid-metal correlations only, so a
  supercritical case had no properties at all.
- **Case consistency checking.** Loading a case rebuilds the mesh from
  `ncx/ncy/ncz`, `istret` and `rstret` and compares it with the grid the
  solver wrote, so an input file that no longer describes the data beside
  it is reported rather than silently believed.
- **A test suite**, 237 tests, run on every push against Python 3.9 and
  3.12, including a wheel built and installed in a clean environment.
- **A Help tab** in the GUI, and one case selector shared by all five tabs.

### Changed

- **behaviour change — Nusselt number.** The thermal conductivity is now
  evaluated at the bulk temperature rather than the reference temperature.
  For supercritical water this changes the answer by up to 77%, because `k`
  falls by a factor of five through the pseudo-critical region.
- **behaviour change — wall shear on a pipe or annulus.** The wall gradient
  was taken at the first wall-normal cell, which for a pipe is the axis,
  where the gradient vanishes. `u_τ` and every profile normalised by it were
  several times too small.
- **behaviour change — fluid properties** now follow the solver's
  `input_thermo.f90` exactly, verified by parsing `modules.f90`. Lithium's
  viscosity was returning µPa·s where every other fluid returned Pa·s, a
  factor of 10⁶. Enthalpy coefficients are derived from the heat-capacity
  ones so `dH/dT = Cp` holds exactly; the previously transcribed sodium
  coefficients were lead-bismuth's.
- **behaviour change — the reference Reynolds number** is no longer
  truncated with `int()` before `u_τ` normalisation. A case Reynolds number
  of 180.9 became 180, a 0.25% error in `u_τ`.
- **Run parameters come from the case.** Reynolds number, reference
  temperature and length, geometry, working fluid, gravity and magnetic
  field directions and the Stuart number are read from the case's
  `input_chapsim.ini`; settings left as `None` in `config.py` are filled
  from it and a disagreement is reported. A Hartmann number is converted as
  `N = Ha²/Re`, which is what unblocked the Lorentz-force statistics.
- **`config.py` is loaded by path**, not by import: `--config`, then
  `./config.py`, then the shipped template, with the file used printed every
  run. Previously whichever `config.py` was first on `sys.path` won, so one
  written beside your data was silently ignored in favour of the toolkit's
  defaults — which matters when several people share a checkout.
- All properties are SI throughout: K, kg/m³, Pa·s, W/(m·K), J/(kg·K), J/kg,
  1/K, Pa.
- The GUI requires ttkbootstrap 2, and therefore Python 3.10. On
  ttkbootstrap 1 the main window could not be constructed at all; the
  declared floor said 1.0.

### Fixed

- Current CHAPSim2 output is read: the `2_visu/{xdmf,data,mesh}` layout,
  XDMF grid collections, packed slice bundles, ASCII `tsp_avg` profile
  tables and the curvilinear meshes pipe and annulus cases write.
- Monitor-point columns are read from each file's own header, rather than
  assuming the pre-2026 six-column layout.
- Budget terms with missing inputs are NaN rather than identically zero, so
  they no longer enter the balance as if they had been computed.
- Gradients follow the data's shape rather than the averaging flags, which
  had broken every MHD statistic on `tsp_avg` input.
- `stitch_domains` produced one cell more than it had data for, and had
  never been run.
- Saved figures are opaque; auto-scaled axes no longer clip a series off the
  plot; an all-NaN series is annotated rather than silently dropped.

### Known issues

- Budget closure is unverified on a converged run. The verification uses
  analytic cases and the solver's short regression output, which exercise
  the formats rather than convergence.
- `LBE`, `PbLi` and `FLiBe` enthalpy data carry the solver's own caveats;
  `CoH_LBE` is marked in `modules.f90` as disagreeing with the literature.
- PbLi-17 properties are valid only over 521–625 K, both ends set by its
  viscosity correlation, against a phase range of 508–1943 K.

## [0.1.0]

The original toolkit by Alex Old, written against the CHAPSim2 output
layout of the time.

[Unreleased]: https://github.com/weiwangstfc/CHAPSim2-toolkit/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/weiwangstfc/CHAPSim2-toolkit/releases/tag/v0.2.0
