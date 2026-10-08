"""Does the data belong to the input file sitting beside it?

Every number the post-processing scales by comes from input_chapsim.ini.
That is only right if the file still describes the run in the folder, and
nothing downstream would notice if it did not - the profiles would simply
come out scaled wrongly, with no sign on the plot.

The grid is what makes the check possible: the input file fully
determines it, and the solver writes it out, so the two can be compared.
"""

import os
import re
import shutil
import tempfile

import numpy as np

from _harness import skip, solver_tests_dir, build_cartesian_case

from chapsim2_toolkit import case_consistency as cc


CASE = ('functional', 'MHD_channel_scp_inout_Tw')


def _solver_case():
    root = solver_tests_dir()
    if root is None:
        skip('CHAPSim2 test output not available (set CHAPSIM2_TESTS)')
    case = os.path.join(root, *CASE)
    if not os.path.isdir(case):
        skip(f'{CASE[1]} is not in this checkout')
    return case


def _copy_with(edits):
    """A copy of the solver case with its input file edited."""
    case = _solver_case()
    tmp = tempfile.mkdtemp()
    target = os.path.join(tmp, 'case')
    shutil.copytree(case, target)
    path = os.path.join(target, 'input_chapsim.ini')
    with open(path) as fh:
        text = fh.read()
    for pattern, replacement in edits:
        text, count = re.subn(pattern, replacement, text, flags=re.M)
        assert count, f'{pattern} did not match the input file'
    with open(path, 'w') as fh:
        fh.write(text)
    return target, tmp


# ---------------------------------------------------------------------------
# A case that agrees with itself
# ---------------------------------------------------------------------------

def test_an_untouched_solver_case_is_consistent():
    result = cc.check_case_consistency(_solver_case())
    assert result['problems'] == []
    assert result['checked'], 'nothing was actually compared'


def test_an_untouched_case_says_nothing():
    """A check that speaks up every run teaches people to ignore it."""
    assert cc.describe_inconsistencies(_solver_case()) == []


def test_every_solver_case_agrees_with_its_own_input_file():
    """The whole regression and functional suite, which is the real test of
    both the mesh port and the checker."""
    import glob
    root = solver_tests_dir()
    if root is None:
        skip('CHAPSim2 test output not available (set CHAPSIM2_TESTS)')
    cases = sorted(set(os.path.dirname(p)
                       for p in glob.glob(os.path.join(root, '*', '*', '2_visu'))))
    if not cases:
        skip('no cases with visualisation output')
    broken = {}
    for case in cases:
        result = cc.check_case_consistency(case)
        if result['problems']:
            broken[os.path.basename(case)] = result['problems']
    assert not broken, f'cases disagreeing with their input file: {broken}'


# ---------------------------------------------------------------------------
# A case that does not
# ---------------------------------------------------------------------------

def test_a_changed_cell_count_is_caught():
    target, tmp = _copy_with([(r'^ncy=.*$', 'ncy= 64')])
    try:
        problems = cc.check_case_consistency(target)['problems']
        assert problems
        assert any('cells' in p or 'nodes' in p for p in problems)
    finally:
        shutil.rmtree(tmp)


def test_a_changed_stretching_is_caught_even_though_the_cell_count_matches():
    """The dangerous one: same number of cells, different spacing, every
    wall-normal profile subtly wrong and nothing else to show for it."""
    target, tmp = _copy_with([(r'^rstret=.*$', 'rstret= 3fmd,0.25')])
    try:
        problems = cc.check_case_consistency(target)['problems']
        assert problems, 'a changed clustering went unnoticed'
        assert any('istret' in p or 'nodes' in p for p in problems)
    finally:
        shutil.rmtree(tmp)


def test_a_changed_stretching_method_is_caught():
    target, tmp = _copy_with([(r'^istret=.*$', 'istret= bottom')])
    try:
        assert cc.check_case_consistency(target)['problems']
    finally:
        shutil.rmtree(tmp)


def test_the_warning_says_why_it_matters():
    target, tmp = _copy_with([(r'^ncy=.*$', 'ncy= 64')])
    try:
        lines = cc.describe_inconsistencies(target, 'mycase')
        assert lines and lines[0].startswith('WARNING')
        assert 'mycase' in lines[0]
        joined = ' '.join(lines)
        assert 'Reynolds' in joined, 'does not say what is at stake'
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# An offset is not a different mesh
# ---------------------------------------------------------------------------

def test_a_whole_domain_offset_is_reported_as_an_offset():
    """The solver's TGV visualisation writes y over [0, 2pi] while it
    computed on [-pi, pi]. Same nodes, shifted - which in a periodic
    direction is immaterial, and must not be reported as a wrong mesh."""
    root = solver_tests_dir()
    if root is None:
        skip('CHAPSim2 test output not available (set CHAPSIM2_TESTS)')
    case = os.path.join(root, 'regression', 'tgv_iso')
    if not os.path.isdir(case):
        skip('tgv_iso is not in this checkout')
    result = cc.check_case_consistency(case)
    assert result['problems'] == [], 'an offset was reported as a bad mesh'
    assert result['notes'], 'the offset was not reported at all'
    assert 'shift' in ' '.join(result['notes'])


# ---------------------------------------------------------------------------
# What it cannot check
# ---------------------------------------------------------------------------

def test_no_input_file_is_a_skip_not_a_pass():
    """An empty problems list with nothing checked is not agreement."""
    with tempfile.TemporaryDirectory() as tmp:
        case = os.path.join(tmp, 'bare')
        build_cartesian_case(case)
        result = cc.check_case_consistency(case)
        assert result['problems'] == []
        assert result['checked'] == []
        assert result['skipped'], 'it did not say why it checked nothing'


def test_no_output_to_compare_against_is_a_skip():
    with tempfile.TemporaryDirectory() as tmp:
        case = os.path.join(tmp, 'inputonly')
        os.makedirs(case)
        with open(os.path.join(case, 'input_chapsim.ini'), 'w') as fh:
            fh.write('[domain]\nicase= channel\n[mesh]\nncx= 4\nncy= 4\nncz= 4\n')
        result = cc.check_case_consistency(case)
        assert result['skipped']
