"""Setup script for CHAPSim2 Python Toolkit."""

from setuptools import setup, find_packages

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chapsim2-toolkit",
    version="0.1.0",
    author="Alex",
    description="A Python post-processing toolkit for CHAPSim2 DNS solver",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlexOld1/CHAPSim2_python_toolkit",
    # NOTE: the modules sit at the top level with no __init__.py, so
    # find_packages() returns nothing and an install places only metadata.
    # The toolkit is therefore run from a checkout; `pip install .[gui,3d]`
    # is useful for pulling in the dependencies. Making it importable after
    # an install means either listing every module in py_modules - which
    # would put names as generic as `utils`, `config` and `slice` on the
    # global import path - or moving them under a package directory. The
    # second is the right answer and is not a packaging-file change.
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
            "ttkbootstrap>=1.0.0",  # gui.py; the scripts do not need it
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "chapsim2-turbstats=turb_stats:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["Reference_Data/**/*"],
    },
    zip_safe=False,
)
