"""Run settings read from the case rather than retyped into config.py.

Every number the solver was given is already recorded in the case's
input_chapsim.ini. Asking the user to copy the Reynolds number, the
reference temperature and the magnetic field into a second file invites
exactly one kind of bug: a value that disagrees with the run, silently
rescaling every normalised profile.
"""

import io
import os
import contextlib
import tempfile
import types

from chapsim2_toolkit import turb_stats as ts
from chapsim2_toolkit.turb_stats import Config

from test_mesh_analysis import CASE_INPUT


def _case(root, name, **kwargs):
    fields = dict(icase='channel', x='1,1', istret='twosides', ren='5000.0',
                  ithermo='.true.', imhd='.true.', ifluid='lithium',
                  nstuart='.false., 0.0', nhartmn='.true., 10.0')
    fields.update(kwargs)
    os.makedirs(os.path.join(root, name), exist_ok=True)
    with open(os.path.join(root, name, 'input_chapsim.ini'), 'w') as fh:
        fh.write(CASE_INPUT.format(**fields))


def _config(**kwargs):
    """A Config as config.py would produce it, with these fields overridden."""
    return Config.from_module(types.SimpleNamespace(**kwargs))


def _apply(config, root, quiet=False):
    """Returns (config, what was printed)."""
    config.folder_path = root
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ts.apply_case_inputs(config, quiet=quiet)
    return config, buffer.getvalue()


def test_unset_fields_are_filled_from_the_case():
    with tempfile.TemporaryDirectory() as tmp:
        _case(tmp, 'runA')
        config = _config(cases=['runA'], Re=None, ref_temp=None, ref_length=None,
                        geometry=None, stuart_number=None, working_fluid=None)
        config, _ = _apply(config, tmp)
        assert config.Re == [5000.0]
        assert config.ref_temp == [645.15]
        assert config.ref_length == [0.0015]
        assert config.geometry == 'channel'
        assert config.working_fluid == 'lithium'
        assert config.stuart_number == 100.0 / 5000.0


def test_a_value_the_user_set_is_kept_and_the_disagreement_reported():
    """Overriding is allowed - doing it by accident should not be silent."""
    with tempfile.TemporaryDirectory() as tmp:
        _case(tmp, 'runA')
        config = _config(cases=['runA'], Re=[180.0], ref_temp=None,
                         ref_length=None, geometry=None, working_fluid=None,
                         stuart_number=None)
        config, said = _apply(config, tmp)
        assert config.Re == [180.0]
        assert 'WARNING' in said and '180' in said and '5000' in said


def test_agreeing_with_the_case_is_not_reported():
    with tempfile.TemporaryDirectory() as tmp:
        _case(tmp, 'runA')
        config = _config(cases=['runA'], Re=[5000.0], ref_temp=None,
                         ref_length=None, geometry=None, working_fluid=None,
                         stuart_number=None, mag_field_direction=None,
                         gravity_direction=None)
        config, said = _apply(config, tmp)
        assert 'WARNING' not in said


def test_each_case_gets_its_own_value():
    with tempfile.TemporaryDirectory() as tmp:
        _case(tmp, 'runA', ren='5000.0')
        _case(tmp, 'runB', ren='2650.0')
        config = _config(cases=['runA', 'runB'], Re=None, geometry=None)
        config, _ = _apply(config, tmp)
        assert config.Re == [5000.0, 2650.0]


def test_a_case_with_no_input_file_leaves_the_configuration_alone():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'runA'))
        config = _config(cases=['runA'], Re=None, geometry=None)
        config, said = _apply(config, tmp)
        assert config.Re is None and config.geometry is None
        assert config.thermo_on is False and config.mhd_on is False
        assert said == ''


def test_a_fluid_without_property_data_says_so():
    with tempfile.TemporaryDirectory() as tmp:
        _case(tmp, 'runA', ifluid='scp_water')
        config = _config(cases=['runA'], working_fluid=None, geometry=None)
        config, said = _apply(config, tmp)
        assert 'water' in said.lower()


def test_quiet_says_nothing_but_still_fills_the_values():
    with tempfile.TemporaryDirectory() as tmp:
        _case(tmp, 'runA')
        config = _config(cases=['runA'], Re=None, geometry=None)
        config, said = _apply(config, tmp, quiet=True)
        assert config.Re == [5000.0] and said == ''
