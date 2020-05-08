PyNumero
========

PyNumero: A high-level Python framework for rapid development of
nonlinear optimization algorithms without large sacrifices on
computational performance.

PyNumero dramatically reduces the time required to prototype new NLP
algorithms and parallel decomposition approaches with minimal
performance penalties.

PyNumero libraries
==================

PyNumero relies on C/C++ extensions for expensive computing operations.

If you installed Pyomo using conda (from conda-forge), then you can
obtain precompiled versions of the redistributable interfaces
(pynumero_ASL) using conda.  Through Pyomo 5.6.9 these libraries are
available by installing the `pynumero_libraries` package from
conda-forge.  Beginning in Pyomo 5.7, the redistributable pynumero
libraries (pynumero_ASL) are included in the pyomo conda-forge package.

If you are not using conda or want to build the nonredistributable
interfaces (pynumero_MA27, pynumero_MA57), you can build the extensions
locally one of three ways:

1. By running the `build.py` Python script in this directory.  This
script will automatically drive the `cmake` build harness to compile the
libraries and install them into your local Pyomo configuration
directory. Cmake options may be specified in the command. For example,

    python build.py -DBUILD_ASL=ON

If you have compiled Ipopt, and you would like to link against the
libraries built with Ipopt, you can. For example,

    python build.py -DBUILD_ASL=ON -DBUILD_MA27=ON -DIPOPT_DIR=<path_to_ipopt_build>/lib/

If you do so, you will likely need to update an environment variable
for the path to shared libraries. For example, on Linux,

    export LD_LIBRARY_PATH=<path_to_ipopt_build>/lib/

2. By running `pyomo build-extensions`.  This will build all registered
Pyomo binary extensions, including PyNumero (using the `build.py` script
from option 1).

3. By manually running cmake to build the libraries.  You will need to
ensure that the libraries are then installed into a location that Pyomo
(and PyNumero) can find them (e.g., in the Pyomo configuration
`lib` directory, in a common system location, or in a location included in
the LD_LIBRARY_PATH environment variable).

Prerequisites
-------------

1. `pynumero_ASL`: 
   - cmake
   - a C/C++ compiler
   - ASL library and headers (optionally, the build harness can
     automatically check out and build AMPL/MP from GitHub to obtain
     this library)

2. `pynumero_MA27`:
   - cmake
   - a C/C++ compiler
   - MA27 library, COIN-HSL Archive, or COIN-HSL Full

2. `pynumero_MA57`:
   - cmake
   - a C/C++ compiler
   - MA57 library or COIN-HSL Full
