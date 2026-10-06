#!/usr/bin/env python3
"""Compatibility wrapper so `python slice.py` keeps working from a checkout.

The code lives in chapsim2_toolkit/slice.py. This runs it exactly as
`python -m chapsim2_toolkit.slice` would, so the two cannot diverge.

Running the package module directly is the better habit, because it puts
your working directory first on sys.path and so finds the config.py and
case folders sitting beside your data:

    python -m chapsim2_toolkit.slice
"""

import os
import runpy
import sys

# A checkout needs no install: the package is a subdirectory of this file's.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

runpy.run_module('chapsim2_toolkit.slice', run_name='__main__', alter_sys=True)
