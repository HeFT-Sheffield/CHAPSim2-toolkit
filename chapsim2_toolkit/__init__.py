"""Post-processing and pre-processing for the CHAPSim2 DNS solver.

The modules are importable individually; nothing is re-exported here,
because importing the package should not pull in matplotlib, pyvista or
ttkbootstrap for a caller that only wants to read a mesh.

    from chapsim2_toolkit import utils, operations

Each script also runs as a module, which is the form that puts your
working directory first on sys.path and so finds the config.py and the
case folders sitting beside your data:

    python -m chapsim2_toolkit.turb_stats --config config.py
    python -m chapsim2_toolkit.gui
"""

__version__ = '0.2.0'

__all__ = ['__version__']
