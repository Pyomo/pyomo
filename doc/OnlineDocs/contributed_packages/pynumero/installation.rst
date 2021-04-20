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

Method 1
--------

One way to build PyNumero extensions is with the pyomo
download-extensions and build-extensions subcommands. Note that
this approach will build PyNumero without support for the HSL linear
solvers. ::

  pyomo download-extensions
  pyomo build-extensions

Method 2
--------

If you want PyNumero support for the HSL solvers and you have an IPOPT compilation
for your machine, you can build PyNumero using the build script ::

  cd pyomo/contrib/pynumero/
  python build.py -DBUILD_ASL=ON -DBUILD_MA27=ON -DIPOPT_DIR=<path/to/ipopt/build/>

