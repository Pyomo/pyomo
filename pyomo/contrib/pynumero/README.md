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

Code organization
=================

PyNumero was initially designed around three core components: linear solver
interfaces, an interface for function and derivative callbacks, and block
vector and matrix classes. Since then, it has incorporated additional
functionality in an ad-hoc manner. The original "core functionality" of
PyNumero, as well as the solver interfaces accessible through
`SolverFactory`, should be considered stable and will only change after
appropriate deprecation warnings. Other functionality should be considered
experimental and subject to change without warning.

The following is a rough overview of PyNumero, by directory:

`linalg`
--------

Python interfaces to linear solvers. This is core functionality.

`interfaces`
------------

- Classes that define and implement an API for function and derivative callbacks
required by nonlinear optimization solvers, e.g. `nlp.py` and `pyomo_nlp.py`
- Various wrappers around these NLP classes to support "hybrid" implementations,
e.g. `PyomoNLPWithGreyBoxBlocks`
- The `ExternalGreyBoxBlock` Pyomo modeling component and
`ExternalGreyBoxModel` API
- The `ExternalPyomoModel` implementation of `ExternalGreyBoxModel`, which allows
definition of an external grey box via an implicit function
- The `CyIpoptNLP` class, which wraps an object implementing the NLP API in
the interface required by CyIpopt

Of the above, only `PyomoNLP` and the `NLP` base class should be considered core
functionality.

`src`
-----

C++ interfaces to ASL, MA27, and MA57. The ASL and MA27 interfaces are
core functionality.

`sparse`
--------

Block vector and block matrix classes, including MPI variations.
These are core functionality.

`algorithms`
------------

Originally intended to hold various useful algorithms implemented
on NLP objects rather than Pyomo models. Any files added here should
be considered experimental.

`algorithms/solvers`
--------------------

Interfaces to Python solvers using the NLP API defined in `interfaces`.
Only the solvers accessible through `SolverFactory`, e.g. `PyomoCyIpoptSolver`
and `PyomoFsolveSolver`, should be considered core functionality.
The supported way to access these solvers is via `SolverFactory`. *The locations
of the underlying solver objects are subject to change without warning.*

`examples`
----------

The examples demonstrated in `nlp_interface.py`, `nlp_interface_2.py1`,
`feasibility.py`, `mumps_example.py`, `sensitivity.py`, `sqp.py`,
`parallel_matvec.py`, and `parallel_vector_ops.py` are stable. All other
examples should be considered experimental.
