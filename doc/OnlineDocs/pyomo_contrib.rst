``pyomo.contrib``
=================

Pyomo uses the ``pyomo.contrib`` package to facilitate the inclusion
of third-party contributions that enhance Pyomo's core functionality.
The are two ways that ``pyomo.contrib`` can be used to integrate
third-party packages:

* ``pyomo.contrib`` can provide wrappers for separate Python packages, thereby allowing these packages to be imported as subpackages of pyomo.

* ``pyomo.contrib`` can include contributed packages that are developed and maintained outside of the Pyomo developer team.  

Including contrib packages in the Pyomo source tree provides a
convenient mechanism for defining new functionality that can be
optionally deployed by users.  We expect this mechanism to include
Pyomo extensions and experimental modeling capabilities.  However,
contrib packages are treated as optional packages, which are not
maintained by the Pyomo developer team.  Thus, it is the responsibility
of the code contributor to keep these packages up-to-date.

Contrib package contributions will be considered as pull-requests,
which will be reviewed by the Pyomo developer team.  Specifically,
this review will consider the suitability of the proposed capability,
whether tests are available to check the execution of the code, and
whether documentation is available to describe the capability.
Contrib packages will be tested along with Pyomo.  If test failures
arise, then these packages will be disabled and an issue will be
created to resolve these test failures.

The following two examples illustrate the two ways
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

While ``pyomocontrib_simplemodel`` can be installed and used separate
from Pyomo, this package is included in ``pyomo/contrib/simplemodel``.
This allows this package to be referenced as if were defined as a
subpackage of ``pyomo.contrib``.  For example::

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
subpackage.  Pyomo requires a separate directory be defined under
``pyomo/contrib`` for each such package, and the Pyomo developer
team will approve the inclusion of third-party packages in this
manner.


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

Although ``pyomo.contrib.example`` is included in the Pyomo source
tree, it is treated as an optional package.  Pyomo will attempt to
import this package, but if an import failure occurs, Pyomo will
silently ignore it.  Otherwise, this pyomo package will be treated
like any other.  Specifically:

* Plugin classes defined in this package are loaded when `pyomo.environ` is loaded.

* Tests in this package are run with other Pyomo tests.

