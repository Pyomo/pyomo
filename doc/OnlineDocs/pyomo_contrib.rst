Third-Party Contributions
=========================

Pyomo uses the ``pyomo.contrib`` package to facilitate the integrate
the inclusion of third-party contributions that enhance Pyomo's
core functionaliy.  The following two examples illustrate two ways
that ``pyomo.contrib`` can be used to integrate third-party
contributions.

Including External Packages
---------------------------

The `pyomocontrib_simplemodel
<http://pyomocontrib-simplemodel.readthedocs.io/en/latest/>`_ package
is derived from Pyomo, and it defines the class SimpleModel that
illustrates how Pyomo can be used in a simple, less object-oriented
manner. Specifically, this class mimics the modeling style supported
by `PuLP <https://github.com/coin-or/pulp>`_.

While ``pyomocontrib_simplemodel`` can be installed and used separate from Pyomo, this
package is included in ``pyomo/contrib/simplemodel``.   This allows this package to be 
referenced as if were defined as a subpackage of ``pyomo.contrib``.  For example::

    from pyomo.contrib.simplemodel import *
    from math import pi

    m = SimpleModel()

    r = m.var('r', bounds=(0,None))
    h = m.var('h', bounds=(0,None))

    m += 2*pi*r*(r + h)
    m += pi*h*r**2 == 355

    status = m.solve("ipopt")

This example illustrates that a package can be distributed separate
from Pyomo while appearing to be included in the ``pyomo.contrib``
subpackage.  Pyomo currently requires a separate directory be defined
under ``pyomo/contrib`` for each such package, so the Pyomo developer
team will explicitly approve the inclusion of third-party packages
in this manner.


Contrib Packages within Pyomo
-----------------------------

Third-party contributions can also be included directly within the
``pyomo.contrib`` package.  The ``pyomo/contrib/example`` package
provides an example of how this can be done, including a directory
for plugins and package tests.  For example, this package can be
imported as a subpackage of ``pyomo.contrib``::

    from pyomo.environ import *
    from pyomo.contrib.example import a

    # Print the value of 'a' defined by this package
    print(a)

Although ``pyomo.contrib.example`` is included in the Pyomo source tree, it is
treated as an optional package.  Consequently,

* Plugin classes defined in this package are not loaded when `pyomo.environ` is loaded.  Plugins are loaded when the ``pyomo.contrib.example`` package is imported.

* Tests in this package are skipped unless the ``TEST_PYOMO_CONTRIB`` environment variable is set to ``1``.

Including contrib packages in the Pyomo source tree provides a
convenient mechanism for defining new functionality that can be
optionally deployed by users.  We expect this mechanism to include
Pyomo extensions and experimental modeling capabilities.  However,
contrib packages are treated as optional packages until the Pyomo
developers decide that they should be fully supported in future
releases.

