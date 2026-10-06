"""The package's own metadata, which nothing else checks.

A released toolkit with many users cannot afford a version that means one
thing in the metadata and another in the module, or a dependency floor
that is stated three times and agreed in two. Neither failure shows up
when running the code; both show up when somebody installs it.
"""

import os
import re
import sys

from _harness import skip

import chapsim2_toolkit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name):
    path = os.path.join(ROOT, name)
    if not os.path.isfile(path):
        skip(f'{name} is not in this checkout')
    with open(path) as fh:
        return fh.read()


def _toml():
    try:
        import tomllib
    except ImportError:                       # Python 3.8-3.10
        try:
            import tomli as tomllib
        except ImportError:
            skip('no TOML reader (tomllib needs 3.11, or install tomli)')
    with open(os.path.join(ROOT, 'pyproject.toml'), 'rb') as fh:
        return tomllib.load(fh)


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

def test_the_version_is_stated_in_exactly_one_place():
    """pyproject takes it from the package, so there is nothing to drift."""
    project = _toml()['project']
    assert 'version' not in project, \
        'pyproject hard-codes a version as well as the package declaring one'
    assert 'version' in project.get('dynamic', []), \
        'pyproject should take the version from chapsim2_toolkit.__version__'


def test_the_declared_version_is_what_the_package_reports():
    data = _toml()
    attr = data['tool']['setuptools']['dynamic']['version']['attr']
    assert attr == 'chapsim2_toolkit.__version__'
    assert re.fullmatch(r'\d+\.\d+\.\d+', chapsim2_toolkit.__version__), \
        f'{chapsim2_toolkit.__version__!r} is not a release version'


def test_an_installed_copy_reports_the_same_version():
    """Only meaningful when the toolkit is actually installed."""
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:
        skip('importlib.metadata needs Python 3.8+')
    try:
        installed = version('chapsim2-toolkit')
    except PackageNotFoundError:
        skip('not installed in this environment')
    assert installed == chapsim2_toolkit.__version__


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def _floors(requirements):
    """{name: floor} from a list of PEP 508 requirement strings."""
    found = {}
    for item in requirements:
        match = re.match(r'^([A-Za-z0-9_.-]+)\s*>=\s*([0-9.]+)', item.strip())
        if match:
            found[match.group(1).lower()] = match.group(2)
    return found


def test_requirements_txt_does_not_keep_a_second_dependency_list():
    """It installs the project, so pyproject stays the only list."""
    text = _read('requirements.txt')
    lines = [line.strip() for line in text.splitlines()
             if line.strip() and not line.startswith('#')]
    assert lines == ['-e .[gui,3d]'], \
        f'requirements.txt has grown its own dependency list: {lines}'


def test_the_conda_environment_agrees_with_pyproject():
    """conda cannot read pyproject, so environment.yml restates the floors.
    They have to agree, and nothing but this would notice if they stopped."""
    text = _read('environment.yml')
    conda = _floors(re.findall(r'^\s*-\s*([A-Za-z0-9_.-]+\s*>=\s*[0-9.]+)\s*$',
                               text, re.M))
    declared = _floors(_toml()['project']['dependencies'])
    extras = _toml()['project']['optional-dependencies']
    for group in extras.values():
        declared.update(_floors(group))
    # python is conda's to pin, not pyproject's dependency list
    requires = _toml()['project']['requires-python']
    assert conda.pop('python', None) == requires.lstrip('>='), \
        'environment.yml and requires-python disagree on the Python floor'
    disagree = {name: (floor, declared.get(name))
                for name, floor in conda.items()
                if declared.get(name) != floor}
    assert not disagree, f'environment.yml disagrees with pyproject: {disagree}'


def test_the_gui_extra_carries_the_ttkbootstrap_floor():
    """ttkbootstrap 1 cannot construct the main window at all, so this
    floor is the difference between a working GUI and an unexplained
    TypeError on startup."""
    extras = _toml()['project']['optional-dependencies']
    assert _floors(extras['gui'])['ttkbootstrap'] == '2.0.0'


# ---------------------------------------------------------------------------
# Changelog
# ---------------------------------------------------------------------------

def test_the_changelog_has_an_entry_for_the_current_version():
    """A release whose changes are not written down is one nobody can
    decide whether to upgrade to."""
    text = _read('CHANGELOG.md')
    version = chapsim2_toolkit.__version__
    assert re.search(r'^## \[' + re.escape(version) + r'\]', text, re.M), \
        f'CHANGELOG.md has no section for {version}'


def test_the_changelog_keeps_an_unreleased_section():
    """Somewhere to write the next change down as it is made, rather than
    reconstructing it from git at release time."""
    text = _read('CHANGELOG.md')
    assert re.search(r'^## \[Unreleased\]', text, re.M)


def test_behaviour_changes_are_marked_as_such():
    """The entries that change a number somebody may have published are
    the ones that have to stand out."""
    text = _read('CHANGELOG.md')
    assert 'behaviour change' in text


# ---------------------------------------------------------------------------
# What must ship
# ---------------------------------------------------------------------------

def test_the_data_files_are_declared_as_package_data():
    """They are read by __file__-relative path, which works in a checkout
    whether or not they are packaged - so a checkout cannot catch this."""
    package_data = _toml()['tool']['setuptools']['package-data']['chapsim2_toolkit']
    assert any('Reference_Data' in entry for entry in package_data)
    assert 'config.py' in package_data


def test_every_console_script_points_at_something_importable():
    scripts = _toml()['project']['scripts']
    assert scripts, 'no console scripts declared'
    import importlib
    for command, target in scripts.items():
        module_name, _, function = target.partition(':')
        if module_name.endswith('.gui') or module_name.endswith('.turb_visu'):
            try:
                import ttkbootstrap            # noqa: F401
            except ImportError:
                continue
        module = importlib.import_module(module_name)
        assert callable(getattr(module, function, None)), \
            f'{command} points at {target}, which is not callable'
