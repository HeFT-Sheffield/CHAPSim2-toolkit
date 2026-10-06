"""Setup script for CHAPSim2 Python Toolkit."""

from setuptools import setup, find_packages

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chapsim2-toolkit",
    version="0.1.0",
    # Alex Old wrote the toolkit and remains its author; the maintainer
    # field is the one that changes hands.
    author="Alex Old",
    maintainer="Wei Wang",
    maintainer_email="wei.wang@stfc.ac.uk",
    description="A Python post-processing toolkit for CHAPSim2 DNS solver",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/weiwangstfc/CHAPSim2-toolkit",
    project_urls={
        "Original repository": "https://github.com/AlexOld1/CHAPSim2_python_toolkit",
        "CHAPSim2 solver": "https://github.com/CHAPSim/CHAPSim2",
    },
    # The modules sit at the top level with no __init__.py, so
    # find_packages() returns nothing and an install places only metadata.
    # The toolkit is run from a checkout; this file exists to pull in the
    # dependencies, `pip install .[gui,3d]`, and for nothing else.
    #
    # Making it importable after an install means either listing every
    # module in py_modules - which would put names as generic as `utils`,
    # `config` and `slice` on the global import path, to collide with
    # whatever else is in the environment - or moving them under a package
    # directory. The second is the right answer; it rewrites 77 imports
    # across 25 files and changes `python gui.py` into a console script,
    # so it is a deliberate change and not a packaging-file tweak.
    #
    # Until then nothing here may promise an importable package. An entry
    # point did, and `chapsim2-turbstats` installed onto the PATH and then
    # died with ModuleNotFoundError - a working-looking install that was
    # not one. Better to install only metadata and say so.
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "pandas>=1.2.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "3d": [
            "pyvista>=0.32.0",  # 3D visualisation: turb_visu.py and the GUI tab
        ],
        "gui": [
            # gui.py calls ttk.Window(theme=...) and uses the pydata themes,
            # neither of which exists before 2.0 - on an older one the window
            # cannot be constructed at all. ttkbootstrap 2 needs Python 3.10,
            # so the GUI has a higher floor than the rest of the toolkit; the
            # scripts still run on 3.8.
            "ttkbootstrap>=2.0.0",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    # No console_scripts and no package_data: both attach to a package
    # that does not exist, so the first shipped a broken command and the
    # second shipped nothing. They come back with the package directory,
    # together with the Reference_Data tables they are meant to carry.
    zip_safe=False,
)
