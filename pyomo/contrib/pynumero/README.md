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

Pynumero relies on C/C++ extensions for expensive computing operations.
If you installed Pyomo through Anaconda, then the redistributable
interfaces (pynumero_ASL) are already present on your system.
Otherwise, you can build the extensions locally one of two ways:

1. By running the `build.py` Python script in this directory.  This
script will automatically drive the `cmake` build harness to compile the
libraries and install them into your local Pyomo configuration
directory.
2. By manually running cmake to build the libraries.  You will need to
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
