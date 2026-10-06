"""Which config.py a run uses, and saying so.

turb_stats opened with a bare `import config`, which picked whichever
config.py was first on sys.path. Running `python /toolkit/turb_stats.py`
from a data directory puts the toolkit's directory first, so a config.py
written next to the data was silently ignored and the shipped defaults
(cases = ['Tests']) used instead.

With several people sharing a checkout that is a provenance problem, not
an inconvenience: a figure can come out carrying someone else's settings
with nothing to show for it.
"""

import io
import contextlib
import os
import tempfile

import turb_stats as ts


@contextlib.contextmanager
def _in(directory):
    previous = os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


def _write(path, text):
    with open(path, 'w') as fh:
        fh.write(text)
    return path


def _load(*args, **kwargs):
    """load_config, returning (config, what it printed)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        config = ts.load_config(*args, **kwargs)
    return config, buffer.getvalue()


def test_a_config_beside_the_data_is_the_one_used():
    """The case the old import could not reach."""
    with tempfile.TemporaryDirectory() as tmp:
        _write(os.path.join(tmp, 'config.py'), "cases = ['MINE']\n")
        with _in(tmp):
            config, said = _load()
        assert config.cases == ['MINE']
        assert tmp in said


def test_an_explicit_path_wins_over_the_working_directory():
    with tempfile.TemporaryDirectory() as tmp:
        _write(os.path.join(tmp, 'config.py'), "cases = ['LOCAL']\n")
        other = _write(os.path.join(tmp, 'other.py'), "cases = ['EXPLICIT']\n")
        with _in(tmp):
            config, _ = _load(other)
        assert config.cases == ['EXPLICIT']


def test_with_no_config_anywhere_the_shipped_one_is_used_and_announced():
    """Falling back is fine; falling back silently is not."""
    with tempfile.TemporaryDirectory() as tmp:
        with _in(tmp):
            config, said = _load()
        assert ts.BUNDLED_CONFIG in said
        assert "not one of yours" in said


def test_the_file_actually_used_is_always_reported():
    """So a figure can be traced back to the settings that made it."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(os.path.join(tmp, 'config.py'), "cases = ['MINE']\n")
        with _in(tmp):
            _, said = _load()
        assert os.path.realpath(path) in os.path.realpath(said.split('\n')[0]
                                                          .split(': ', 1)[1])


def test_quiet_suppresses_the_report_but_not_the_choice():
    with tempfile.TemporaryDirectory() as tmp:
        _write(os.path.join(tmp, 'config.py'), "cases = ['MINE']\n")
        with _in(tmp):
            config, said = _load(quiet=True)
        assert config.cases == ['MINE'] and said == ''


def test_a_path_that_does_not_exist_is_an_error():
    """Not a silent fall back to the shipped defaults."""
    try:
        ts.load_config('/definitely/not/here/config.py')
    except FileNotFoundError as exc:
        assert 'config.py' in str(exc)
    else:
        raise AssertionError('a missing config file was accepted')


def test_find_config_prefers_the_working_directory_over_the_bundled_one():
    with tempfile.TemporaryDirectory() as tmp:
        local = _write(os.path.join(tmp, 'config.py'), '')
        with _in(tmp):
            assert os.path.realpath(ts.find_config()) == os.path.realpath(local)


def test_the_bundled_config_is_still_a_usable_template():
    """It is what a new user copies, so it has to load and parse."""
    config = ts.load_config(ts.BUNDLED_CONFIG, quiet=True)
    assert config.cases                      # it names something
    assert config.Re is None                 # and leaves the case to say


def test_main_accepts_a_config_flag():
    import argparse
    try:
        ts.main(['--config', '/definitely/not/here/config.py'])
    except (FileNotFoundError, SystemExit) as exc:
        assert not isinstance(exc, argparse.ArgumentError)
    else:
        raise AssertionError('main ignored --config')
