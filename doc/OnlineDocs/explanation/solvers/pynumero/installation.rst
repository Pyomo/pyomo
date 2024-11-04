PyNumero Installation
=====================

PyNumero is a module within Pyomo. Therefore, Pyomo must be installed
to use PyNumero. PyNumero also has some extensions that need
built. There are many ways to build the PyNumero extensions. Common
use cases are listed below. However, more information can always be
found at
https://github.com/Pyomo/pyomo/blob/main/pyomo/contrib/pynumero/build.py
and
https://github.com/Pyomo/pyomo/blob/main/pyomo/contrib/pynumero/src/CMakeLists.txt.

Note that you will need a C++ compiler and CMake installed to build the
PyNumero libraries.

Method 1
--------

One way to build PyNumero extensions is with the pyomo
`download-extensions` and `build-extensions` subcommands. Note that
this approach will build PyNumero without support for the HSL linear
solvers. ::

  pyomo download-extensions
  pyomo build-extensions

Method 2
--------

If you want PyNumero support for the HSL solvers and you have an IPOPT compilation
for your machine, you can build PyNumero using the build script ::

  python -m pyomo.contrib.pynumero.build -DBUILD_ASL=ON -DBUILD_MA27=ON -DIPOPT_DIR=<path/to/ipopt/build/>

Method 3
--------

You can build the PyNumero libraries from source using `cmake`.  This
generally works best when building from a source distribution of Pyomo.
Assuming that you are starting in the root of the Pyomo source
distribution, you can follow the normal CMake build process ::

  mkdir build
  cd build
  ccmake ../pyomo/contrib/pynumero/src
  make
  make install
