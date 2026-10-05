#!/usr/bin/env python3
"""Run the toolkit's tests.

Hands over to pytest when it is installed, since that gives better
reporting. Otherwise runs the same tests directly, so the suite is usable
on a machine where installing anything is awkward - which is most clusters.

    python run_tests.py                 every test
    python run_tests.py test_xdmf       one module
    python run_tests.py -k cylindrical  tests whose name contains this

The tests that read CHAPSim2's own output skip themselves when the solver
is not checked out alongside. Point CHAPSIM2_TESTS at its tests directory
to run them from elsewhere.
"""

import argparse
import importlib.util
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(HERE, 'tests')


def _load(path):
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _collect(modules, keyword):
    for path in modules:
        module = _load(path)
        for name in sorted(vars(module)):
            if name.startswith('test_') and callable(getattr(module, name)):
                if keyword and keyword not in f'{module.__name__}.{name}':
                    continue
                yield module.__name__, name, getattr(module, name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('modules', nargs='*', help='test modules to run')
    parser.add_argument('-k', dest='keyword', help='only tests matching this substring')
    parser.add_argument('-q', dest='quiet', action='store_true', help='only show failures')
    parser.add_argument('--no-pytest', action='store_true',
                        help='use the built-in runner even if pytest is installed')
    args = parser.parse_args(argv)

    sys.path.insert(0, HERE)
    sys.path.insert(0, TESTS)

    if not args.no_pytest:
        try:
            import pytest
        except ImportError:
            pass
        else:
            argv = [TESTS, '-q'] if args.quiet else [TESTS]
            if args.keyword:
                argv += ['-k', args.keyword]
            if args.modules:
                argv = [os.path.join(TESTS, m if m.endswith('.py') else m + '.py')
                        for m in args.modules] + argv[1:]
            return pytest.main(argv)

    wanted = args.modules or sorted(
        os.path.splitext(f)[0] for f in os.listdir(TESTS)
        if f.startswith('test_') and f.endswith('.py'))
    paths = [os.path.join(TESTS, m if m.endswith('.py') else m + '.py') for m in wanted]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        parser.error('no such test module: ' + ', '.join(missing))

    from _harness import SKIP_EXCEPTIONS

    passed = failed = skipped = 0
    failures = []
    current = None
    for module_name, test_name, func in _collect(paths, args.keyword):
        if module_name != current and not args.quiet:
            print(f'\n{module_name}')
            current = module_name
        try:
            func()
        except SKIP_EXCEPTIONS as exc:
            skipped += 1
            if not args.quiet:
                print(f'  SKIP  {test_name}  ({exc})')
        except Exception:
            failed += 1
            failures.append((module_name, test_name, traceback.format_exc()))
            print(f'  FAIL  {test_name}')
        else:
            passed += 1
            if not args.quiet:
                print(f'  ok    {test_name}')

    for module_name, test_name, tb in failures:
        print(f'\n{"=" * 70}\nFAILED {module_name}.{test_name}\n{"-" * 70}\n{tb}')

    print(f'\n{passed} passed, {failed} failed, {skipped} skipped')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
