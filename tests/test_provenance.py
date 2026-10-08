"""What made a figure, recorded in the figure.

A plot in a paper is evidence. Eight months later the questions are
always the same - which version drew this, was the tree clean, which case
and which config.py - and a PNG answers none of them by itself.
"""

import os
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from _harness import skip

from chapsim2_toolkit import provenance as prov
import chapsim2_toolkit


def _figure():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return fig


def _save(suffix, **kwargs):
    fig = _figure()
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'figure' + suffix)
    fig.savefig(path, metadata=prov.figure_metadata(suffix.lstrip('.'),
                                                    **kwargs))
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# What is recorded
# ---------------------------------------------------------------------------

def test_a_png_carries_the_toolkit_version_and_revision():
    fields = prov.read(_save('.png'))
    assert fields['toolkit_version'] == chapsim2_toolkit.__version__
    assert chapsim2_toolkit.__version__ in fields['toolkit']


def test_a_dirty_tree_is_recorded_as_dirty():
    """The flag that matters: a commit does not describe the code that ran
    if the tree was modified, so a figure from a dirty tree cannot be
    regenerated from that commit alone."""
    sha, dirty = prov.git_revision()
    if sha is None:
        skip('not a git checkout')
    fields = prov.read(_save('.png'))
    assert fields['git_revision'] == sha
    assert fields['git_dirty'] == ('yes' if dirty else 'no')


def test_the_case_and_timestep_are_recorded():
    path = _save('.png', cases=['pipe_iso_periodic'], timesteps=['20'])
    fields = prov.read(path)
    assert fields['cases'] == 'pipe_iso_periodic'
    assert fields['timesteps'] == '20'


def test_the_config_file_is_recorded():
    """Which settings produced the figure, not merely which code."""
    fields = prov.read(_save('.png', config_path='/somewhere/config.py'))
    assert fields['config'] == '/somewhere/config.py'


def test_a_config_object_supplies_the_cases_and_timesteps():
    from chapsim2_toolkit import turb_stats as ts
    config = ts.load_config(ts.BUNDLED_CONFIG, quiet=True)
    config.cases = ['A', 'B']
    config.timesteps = ['100']
    fields = prov.read(_save('.png', config=config))
    assert fields['cases'] == 'A, B'
    assert fields['timesteps'] == '100'


def test_the_creation_time_is_recorded_with_a_timezone():
    """A bare local timestamp is ambiguous between machines."""
    fields = prov.read(_save('.png'))
    assert '+' in fields['created'] or fields['created'].endswith('Z')


# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------

def test_a_pdf_carries_it_too():
    """PDF is what goes into a paper, so it cannot be the one that loses
    the provenance."""
    fields = prov.read(_save('.pdf', cases=['mycase']))
    assert chapsim2_toolkit.__version__ in fields['Creator']
    assert 'mycase' in fields['Keywords']


def test_an_unknown_format_gets_no_metadata_rather_than_an_error():
    """matplotlib raises on unexpected keys for some writers, so returning
    None is how a format opts out."""
    assert prov.figure_metadata('tiff') is None
    assert prov.figure_metadata('') is None


def test_svg_uses_the_keys_its_writer_accepts():
    meta = prov.figure_metadata('svg')
    assert set(meta) <= {'Creator', 'Description', 'Title', 'Date'}


# ---------------------------------------------------------------------------
# Reading it back
# ---------------------------------------------------------------------------

def test_reading_a_figure_with_no_provenance_returns_nothing_not_an_error():
    fig = _figure()
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'plain.png')
    fig.savefig(path)
    plt.close(fig)
    assert 'toolkit' not in prov.read(path)


def test_reading_a_format_it_cannot_parse_says_so():
    try:
        prov.read('/tmp/whatever.tiff')
    except ValueError as exc:
        assert 'tiff' in str(exc)
    else:
        raise AssertionError('an unreadable format was accepted')


def test_the_command_line_prints_what_it_found():
    import io
    import contextlib
    path = _save('.png', cases=['printed'])
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        status = prov.main([path])
    assert status == 0
    assert 'printed' in buffer.getvalue()
    assert chapsim2_toolkit.__version__ in buffer.getvalue()


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

def test_git_is_consulted_once_not_once_per_figure():
    """A run saving thirty figures should not shell out thirty times."""
    prov._GIT_CACHE.clear()
    first = prov.git_revision()
    assert prov._GIT_CACHE
    assert prov.git_revision() == first


def test_it_works_outside_a_git_checkout():
    """An installed copy has no .git, and must still save figures."""
    with tempfile.TemporaryDirectory() as tmp:
        assert prov.git_revision(repo=tmp) == (None, False)
