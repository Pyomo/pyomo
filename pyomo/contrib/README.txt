This directory is used to support the deployment of third-party
Pyomo libraries in a standard manner.  See the Pyomo online
documentation for a description of these libraries.

The 'example' directory illustrates the layout of a contrib package
that is included in the Pyomo source tree.

The 'simplemodel' directory illustrates the layout of a contrib
package that is installed separately, but which appears as a
subpackage under 'pyomo.contrib'.

NOTE:  Packages that define plugins should be specified in 
the file pyomo/pyomo/environ/__init__.py
