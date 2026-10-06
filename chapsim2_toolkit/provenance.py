"""What made a figure, recorded in the figure.

A plot that goes into a paper is evidence, and evidence that cannot be
traced is weak. Eight months later the questions are always the same:
which version of the toolkit drew this, was the working tree clean at the
time, which case and timestep is it, and which config.py set the
normalisation. None of that is recoverable from a PNG afterwards.

So it is written into the file when the figure is saved. PNG carries
tEXt chunks and PDF carries a document info dictionary; matplotlib
exposes both through ``savefig(metadata=...)``, so the figure itself
carries its provenance and no sidecar file can be separated from it.

Nothing here is displayed on the figure. A watermark would have to be
placed, sized and kept out of the data, and would be cropped off by the
first person who needs the plot to fit a column. Metadata survives that.

Read it back with::

    python -m chapsim2_toolkit.provenance figure.png
"""

import datetime
import os
import subprocess
import sys

__all__ = ['toolkit_version', 'git_revision', 'figure_metadata', 'read']

#: The repository this file lives in, when it lives in one.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Cached so a run saving thirty figures does not call git thirty times.
_GIT_CACHE = {}


def toolkit_version():
    """The installed version string."""
    from chapsim2_toolkit import __version__
    return __version__


def git_revision(repo=None):
    """``(short sha, dirty)`` for the checkout, or ``(None, False)``.

    A release version alone is not enough to identify what drew a figure:
    most work happens between releases, and a dirty tree means the commit
    does not describe the code that ran. Both are recorded, and the dirty
    flag is the one that matters - it is the difference between a figure
    that can be regenerated and one that cannot.
    """
    repo = repo or _REPO
    if repo in _GIT_CACHE:
        return _GIT_CACHE[repo]

    result = (None, False)
    if os.path.isdir(os.path.join(repo, '.git')):
        def git(*args):
            return subprocess.run(
                ('git', '-C', repo) + args,
                capture_output=True, text=True, timeout=10)
        try:
            sha = git('rev-parse', '--short', 'HEAD')
            if sha.returncode == 0:
                status = git('status', '--porcelain')
                dirty = bool(status.stdout.strip()) if status.returncode == 0 \
                    else False
                result = (sha.stdout.strip(), dirty)
        except (OSError, subprocess.SubprocessError):
            pass                      # no git, or not a repository: say nothing

    _GIT_CACHE[repo] = result
    return result


def describe():
    """One line naming the code that is running."""
    sha, dirty = git_revision()
    text = f'CHAPSim2-toolkit {toolkit_version()}'
    if sha:
        text += f' ({sha}{"-dirty" if dirty else ""})'
    return text


def figure_metadata(fmt, config=None, cases=None, timesteps=None,
                    config_path=None, extra=None):
    """Metadata to hand to ``savefig(metadata=...)``.

    Args:
        fmt: output format, 'png' or 'pdf' - their metadata keys differ.
        config: a Config, read for the cases, timesteps and config file
            when those are not given explicitly.
        cases, timesteps, config_path: override what is read from config.
        extra: further key/value pairs to record.

    Returns:
        A dict for savefig, or None for a format that carries no metadata
        (matplotlib raises rather than ignoring unknown keys for some).
    """
    fmt = (fmt or '').lower().lstrip('.')

    if config is not None:
        cases = cases if cases is not None else getattr(config, 'cases', None)
        timesteps = timesteps if timesteps is not None \
            else getattr(config, 'timesteps', None)

    sha, dirty = git_revision()
    fields = {
        'toolkit': describe(),
        'toolkit_version': toolkit_version(),
        'created': datetime.datetime.now().astimezone().isoformat(
            timespec='seconds'),
    }
    if sha:
        fields['git_revision'] = sha
        fields['git_dirty'] = 'yes' if dirty else 'no'
    if cases:
        fields['cases'] = ', '.join(str(c) for c in cases)
    if timesteps:
        fields['timesteps'] = ', '.join(str(t) for t in timesteps)
    if config_path:
        fields['config'] = str(config_path)
    if extra:
        fields.update({str(k): str(v) for k, v in extra.items()})

    if fmt == 'png':
        # PNG tEXt chunks: arbitrary keys, which is what we want.
        return fields
    if fmt in ('pdf', 'eps', 'ps'):
        # The document info dictionary has fixed keys, so everything that
        # does not map onto one goes into Keywords as a single string.
        return {
            'Creator': fields['toolkit'],
            'Producer': fields['toolkit'],
            'CreationDate': None,       # matplotlib fills this in
            'Keywords': '; '.join(f'{k}={v}' for k, v in fields.items()
                                  if k != 'toolkit'),
        }
    if fmt == 'svg':
        # matplotlib's SVG writer accepts only a fixed Dublin Core set.
        return {'Creator': fields['toolkit'],
                'Description': '; '.join(f'{k}={v}' for k, v in fields.items()
                                         if k != 'toolkit')}
    return None


def read(path):
    """The provenance recorded in a saved figure, as a dict.

    Supports PNG and PDF, which is what the toolkit writes.
    """
    path = str(path)
    suffix = os.path.splitext(path)[1].lower()

    if suffix == '.png':
        import struct
        fields = {}
        with open(path, 'rb') as fh:
            if fh.read(8) != b'\x89PNG\r\n\x1a\n':
                raise ValueError(f'{path} is not a PNG')
            while True:
                header = fh.read(8)
                if len(header) < 8:
                    break
                length, kind = struct.unpack('>I4s', header)
                data = fh.read(length)
                fh.read(4)                      # CRC
                if kind == b'tEXt' and b'\x00' in data:
                    key, _, value = data.partition(b'\x00')
                    fields[key.decode('latin-1')] = value.decode('latin-1')
                elif kind == b'IEND':
                    break
        return fields

    if suffix == '.pdf':
        # The trailer's /Info dictionary, read without a PDF library.
        import re
        with open(path, 'rb') as fh:
            blob = fh.read()
        fields = {}
        for key in (b'Creator', b'Producer', b'Keywords'):
            match = re.search(rb'/' + key + rb'\s*\((.*?)(?<!\\)\)', blob, re.S)
            if match:
                fields[key.decode()] = match.group(1).decode(
                    'latin-1', 'replace')
        return fields

    raise ValueError(f'Cannot read provenance from a {suffix or "?"} file')


def main(argv=None):
    """Print the provenance of each figure named on the command line."""
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(f'This is {describe()}')
        print(f'Usage: python -m chapsim2_toolkit.provenance FIGURE '
              f'[FIGURE ...]')
        return 0
    status = 0
    for path in argv:
        print(f'{path}:')
        try:
            fields = read(path)
        except (OSError, ValueError) as exc:
            print(f'  {exc}')
            status = 1
            continue
        if not fields:
            print('  no provenance recorded (saved by an older version?)')
            continue
        for key in sorted(fields):
            print(f'  {key:16s} {fields[key]}')
    return status


if __name__ == '__main__':
    sys.exit(main())
