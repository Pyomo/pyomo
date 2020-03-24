Contributing to Pyomo
=====================

We welcome all contributions including bug fixes, feature enhancements,
and documentation improvements. Pyomo manages source code contributions
via GitHub pull requests (PRs). 

Contribution Requirements
-------------------------

A PR should be 1 set of related changes. PRs for large-scale
non-functional changes (i.e. PEP8, comments) should be
separated from functional changes. This simplifies the review process
and ensures that functional changes aren't obscured by large amounts of
non-functional changes.

Coding Standards
++++++++++++++++
    
    * Required: 4 space indentation (no tabs)
    * Desired: PEP8
    * No use of __author__ 
    * Inside ``pyomo.contrib``: Contact information for the contribution
      maintainer (such as a Github ID) should be included in the Sphinx
      documentation

Sphinx-compliant documentation is required for:
    
    * Modules
    * Public and Private Classes
    * Public and Private Functions 

We also encourage you to include examples, especially for new features
and contributions to ``pyomo.contrib``.

Testing
+++++++

Pyomo uses ``unittest``, TravisCI, and Appveyor for testing and
continuous integration. Submitted code should include tests to establish
the validity of its results and/or effects. Unit tests are preferred but
we also accept integration tests. When test are run on a PR, we require
at least 70% coverage of the lines modified in the PR and prefer
coverage closer to 90%. We also require that all tests pass before a PR
will be merged.

At any point in the development cycle, a "work in progress" pull request
may be opened by including '[WIP]' at the beginning of the PR
title. This allows your code changes to be tested by Pyomo's automatic
testing infrastructure. Any pull requests marked '[WIP]' will not be
reviewed or merged by the core development team. In addition, any
'[WIP]' pull request left open for an extended period of time without
active development may be marked 'stale' and closed.

Review Process
--------------

After a PR is opened it will be reviewed by at least two members of the
core development team. The core development team consists of anyone with
write-access to the Pyomo repository. Pull requests opened by a core
developer only require one review. The reviewers will decide if they
think a PR should be merged or if more changes are necessary.

Reviewers look for:
    
    * Outside of ``pyomo.contrib``: Code rigor and standards, edge cases,
      side effects, etc.
    * Inside of ``pyomo.contrib``: No “glaringly obvious” problems with
      the code
    * Documentation and tests

The core development team tries to review pull requests in a timely
manner but we make no guarantees on review timeframes. In addition, PRs
might not be reviewed in the order they are opened in. 

Where to put contributed code 
----------------------------- 

In order to contribute to Pyomo, you must first make a fork of the Pyomo
git repository. Next, you should create a branch on your fork dedicated
to the development of the new feature or bug fix you're interested
in. Once you have this branch checked out, you can start coding. Bug
fixes and minor enhancements to existing Pyomo functionality should be
made in the appropriate files in the Pyomo code base. New examples,
features, and packages built on Pyomo should be placed in
``pyomo.contrib``. Follow the link below to find out if
``pyomo.contrib`` is right for your code.

``pyomo.contrib``
-----------------

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
+++++++++++++++++++++++++++

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
+++++++++++++++++++++++++++++

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

