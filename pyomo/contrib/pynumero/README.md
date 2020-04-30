PyNumero
========

PyNumero: A high-level Python framework for rapid development of
nonlinear optimization algorithms without large sacrifices on
computational performance.

PyNumero dramatically reduces the time required to prototype new NLP
algorithms and parallel decomposition while minimizing the performance
penalty.

PyNumero libraries
==================

PyNumero relies on C/C++ extensions for expensive computing operations.

If you installed Pyomo using Anaconda (from conda-forge), then you can
obtain precompiled versions of the redistributable interfaces
(pynumero_ASL) using conda.  Through Pyomo 5.6.9 these libraries are
available by installing the `pynumero_libraries` package from
conda-forge.  Beginning in Pyomo 5.7, the redistributable pynumero
libraries are included in the pyomo conda-forge package.

If you are not using conda or want to build the nonredistributable
interfaces, you can build the extensions locally one of three ways:

1. By running the `build.py` Python script in this directory.  This
script will automatically drive the `cmake` build harness to compile the
libraries and install them into your local Pyomo configuration
directory.
2. By running `pyomo build-extensions`.  This will build all registered
Pyomo binary extensions, including PyNumero (using the `build.py` script
from option 1).
3. By manually running cmake to build the libraries.  You will need to
ensure that the libraries are then installed into a location that Pyomo
(and PyNumero) can find them (e.g., in the Pyomo configuration
directory, or in a common system location, or in a location included in
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
   - *TODO*
