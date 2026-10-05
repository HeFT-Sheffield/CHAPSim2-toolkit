"""Reading input_chapsim.ini.

Current input files name their enumerated settings. Casting them to int
made every one fall back to its default, so the mesh reported was not the
mesh the solver would build - and it was reported confidently.
"""

import os
import tempfile

import numpy as np

import mesh_analysis as ma


MODERN = """\
[decomposition]
nxdomain= 1

[domain]
icase= pipe
lxx= 8.0
lyt= 1.0
lyb= 0.0
lzz= 6.283185

[flow]
ren= 2650

[thermo]
ithermo= .true.
ifluid= scp_water
ref_t0= 645.15

[mesh]
ncx= 80
ncy= 48
ncz= 64
istret= top
rstret= tanh,0.1

[scheme]
dt= 2e-03

[io]
is_record_xoutlet_read_xinlet= .true.,.false.
"""

LEGACY = """\
[domain]
icase= 2
lxx= 8.0

[flow]
ren= 2650

[thermo]
ithermo= .true.
ifluid= 1

[mesh]
ncx= 80
ncy= 48
ncz= 64
istret= 4
rstret= 2,0.1

[scheme]
dt= 2e-03

[io]
is_wrt_read_bc= .true.,.false.
"""


def _config(text):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'input_chapsim.ini')
        with open(path, 'w') as fh:
            fh.write(text)
        return ma.DomainConfig(ma.parse_input_file(path), path)


def test_named_enums_are_understood():
    cfg = _config(MODERN)
    assert cfg.icase == ma.ICASE_PIPE
    assert cfg.istret == ma.ISTRET_TOP
    assert cfg.mstret == ma.MSTRET_TANH
    assert cfg.rstret == 0.1
    assert cfg.ifluid == 1                      # scp_water
    assert cfg.is_thermo is True


def test_legacy_integers_still_mean_the_same_thing():
    """The solver accepts either spelling, so the toolkit must too."""
    modern, legacy = _config(MODERN), _config(LEGACY)
    for field in ('icase', 'istret', 'mstret', 'ifluid',
                  'is_record_xoutlet, '.strip(', ')):
        assert getattr(modern, field) == getattr(legacy, field), field


def test_the_renamed_io_key_is_read_under_either_name():
    assert _config(MODERN).is_record_xoutlet is True
    assert _config(LEGACY).is_record_xoutlet is True
    assert _config(MODERN).is_read_xinlet is False


def test_an_unknown_value_falls_back_rather_than_crashing():
    cfg = _config('[domain]\nicase= hexagon\n[mesh]\nistret= sideways\n')
    assert cfg.icase == ma.ICASE_OTHERS
    assert cfg.istret == ma.ISTRET_NO
    assert cfg.is_wall_bounded is False


def test_case_fixes_the_extents_the_solver_overrides():
    assert _config(MODERN).lyb == 0.0                      # pipe: r in [0, 1]
    channel = _config(MODERN.replace('icase= pipe', 'icase= channel'))
    assert (channel.lyb, channel.lyt) == (-1.0, 1.0)


def test_cylindrical_cases_force_an_even_spanwise_count():
    cfg = _config(MODERN.replace('ncz= 64', 'ncz= 63'))
    assert cfg.nc[2] == 64


def test_uniform_stretching_builds_an_even_grid():
    """MSTRET_NONE had no mapping and raised 'unsupported stretching'."""
    cfg = _config(MODERN.replace('istret= top', 'istret= no')
                        .replace('rstret= tanh,0.1', 'rstret= uniform,0.1'))
    yp, yc = ma.build_y_grid(cfg)
    assert len(yp) == cfg.nc[1] + 1 and len(yc) == cfg.nc[1]
    assert np.allclose(np.diff(yp), np.diff(yp)[0])


def test_grid_is_monotonic_and_spans_the_domain():
    cfg = _config(MODERN)
    yp, yc = ma.build_y_grid(cfg)
    assert np.all(np.diff(yp) > 0)
    assert np.isclose(yp[0], cfg.lyb) and np.isclose(yp[-1], cfg.lyt)
    assert np.all((yc > yp[0]) & (yc < yp[-1]))


def test_top_clustering_puts_the_fine_cells_at_the_wall():
    """A pipe clusters towards r = 1; the axis end must be the coarse one."""
    cfg = _config(MODERN)
    yp, _ = ma.build_y_grid(cfg)
    spacing = np.diff(yp)
    assert spacing[-1] < spacing[0]


def test_written_input_round_trips():
    cfg = _config(MODERN)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'input_chapsim.ini')
        text = ma.write_input_file(cfg, out)
        assert 'icase= pipe' in text          # names, not integers
        assert 'istret= top' in text
        assert 'rstret= tanh,' in text
        back = ma.DomainConfig(ma.parse_input_file(out))
        for field in ('icase', 'istret', 'mstret', 'rstret', 'ifluid',
                      'is_thermo', 'ren', 'dt'):
            assert getattr(back, field) == getattr(cfg, field), field
        assert back.nc == cfg.nc


def test_from_values_rejects_an_unknown_field():
    try:
        ma.DomainConfig.from_values(icase=ma.ICASE_PIPE, nonsense=1)
    except AttributeError:
        return
    raise AssertionError('expected AttributeError for an unknown field')


# ---------------------------------------------------------------------------
# Periodicity. Averaging a direction that is not periodic folds the two ends
# of the domain together, so which directions are homogeneous has to come
# from the case rather than from whoever filled in the config.
# ---------------------------------------------------------------------------

PERIODIC_BC = """\
[domain]
icase= {icase}

[bc]
ifbcx_u= {x}
ifbcx_v= {x}
ifbcx_w= {x}
ifbcy_u= {y}
ifbcy_v= {y}
ifbcy_w= {y}
ifbcz_u= {z}
ifbcz_v= {z}
ifbcz_w= {z}
"""


def _periodicity(icase='channel', x='1,1', y='4,4', z='1,1'):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'input_chapsim.ini')
        with open(path, 'w') as fh:
            fh.write(PERIODIC_BC.format(icase=icase, x=x, y=y, z=z))
        return ma.read_periodicity(path)


def test_doubly_periodic_channel():
    assert _periodicity() == {'x': True, 'y': False, 'z': True}


def test_inlet_outlet_is_not_periodic_in_x():
    """10 is a database inlet and 7 a convective outlet, not periodic."""
    assert _periodicity(x='10,7') == {'x': False, 'y': False, 'z': True}


def test_one_periodic_face_makes_the_direction_periodic():
    """The solver promotes the pair when either face says periodic."""
    assert _periodicity(x='1,4')['x'] is True


def test_a_pipe_is_never_periodic_in_the_radial_direction():
    """y is the radius there and its lower end is the axis, not a face."""
    assert _periodicity(icase='pipe', y='1,1')['y'] is False
    assert _periodicity(icase='channel', y='1,1')['y'] is True


def test_a_case_directory_is_accepted_as_well_as_the_file():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, 'input_chapsim.ini'), 'w') as fh:
            fh.write(PERIODIC_BC.format(icase='channel', x='1,1', y='4,4', z='1,1'))
        assert ma.read_periodicity(tmp) == {'x': True, 'y': False, 'z': True}


def test_no_input_file_gives_no_answer_rather_than_a_guess():
    with tempfile.TemporaryDirectory() as tmp:
        assert ma.read_periodicity(tmp) is None
