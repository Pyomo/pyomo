Development Principles
======================

The Pyomo development team follows a set of development principles.
In order to promote overall transparency, this page is intended to document
those principles to the best of our ability, for users and potential
contributors alike. Please also review Pyomo's recent publication
on the history of its development for a holistic view into the changes
of these principles over time [MHJ+25]_.

Backwards Compatibility
-----------------------

Commitment to Published Interfaces
++++++++++++++++++++++++++++++++++

If functionality is published in the most recent edition of the
Pyomo book [PyomoBookIII]_ (The Book) and corresponding examples, we treat that
as a public API commitment. The interfaces and APIs appearing in The Book
will be supported (although possibly in a deprecated form) until 
the next major Pyomo release, which will generally coincide with a
new edition of the book.

This commitment ensures that teaching materials, training resources, and
long-term codebases built following those examples will remain valid across an
entire major release.

Stable, Supported APIs
++++++++++++++++++++++

Functionality that is part of the Pyomo source tree but not explicitly
included in the book is also expected to be stable if it resides outside
the ``contrib`` (or future ``addons`` / ``devel``) directories. This is
referred to as "core" by the Pyomo development team.

When changes to core APIs become necessary, we will endeavor to follow one
(or both) of the following steps:

1. **Deprecation warnings** are added in advance of functionality removal.
   These are visible to users at import or execution time,
   with clear guidance on replacement functionality.
2. **Relocation warnings** are provided for any relocated functionality.
   These modules import from their old locations and print a warning about
   the relocation in order to assist users' transition.

Ideally, changes in this fashion can allow users and downstream packages
to adapt gradually without abrupt breakage.

.. note::

   For detailed guidance on how to mark functionality for deprecation or
   removal within the Pyomo codebase, see
   :doc:`/explanation/developer_utils/deprecation`.

Experimental and Contributed Code
+++++++++++++++++++++++++++++++++

Historically, all contributed and experimental functionality has lived
under ``pyomo.contrib``. These packages are community-driven extensions
that may evolve rapidly as research and experimentation continue. Users
should not rely on API stability within this space.

Pyomo is currently transitioning to a clearer organizational model that
distinguishes between two categories:

* ``pyomo.addons`` – Mostly stable, supported extensions maintained by
 specific contributors.

* ``pyomo.devel`` – Active research and experimental code.

The guiding philosophy is to protect users from surprises in stable
interfaces while continuing to enable innovation and experimentation
in development areas.

.. note::

   For procedural guidance on how to structure and submit new
   contributions to Pyomo, please visit :doc:`contribution_guide`.

Dependency Management
---------------------

Minimal Core Dependencies
+++++++++++++++++++++++++

The core Pyomo codebase is designed to be a *Pure Python* library with minimal
dependencies outside the standard Python library (currently, there is only a
single hard dependency on ``ply``).

This approach simplifies installation, reduces the burden on derived packages,
and lessens the likelihood of triggering dependency conflicts.
Additionally, this allows users to install and run Pyomo in
resource-constrained or isolated environments (such as teaching
containers or HPC systems).

Optional Dependencies
+++++++++++++++++++++

Some extended Pyomo functionality relies on additional optional Python packages.
An optional dependency must not be imported (or required) for the Pyomo
environment. That is:

.. doctest::
   >>> import pyomo.environ

should not raise an ``ImportError`` if the dependency is missing. Further, the
Pyomo test harness (``pytest pyomo``) must run without error/failure if any
optional dependencies are missing (except for the dependencies required by
the ``tests`` :ref:`dependency group <dependency_groups>`).

Pyomo makes extensive use of :py:func:`attempt_import()` to support the
standardized and convenient use of optional dependencies. Further, many
common dependencies are directly importable through
``pyomo.common.dependencies`` without immediately triggering the dependency
import; for example:

.. testcode::
   # Importing numpy from dependencies does not trigger the import
   from pyomo.common.dependencies import numpy as np, numpy_available
   
   # but testing the availability or using the module will trigger the import
   if numpy_available:
      a = np.array([1, 2, 3])
   
.. testoutput::
   : hide:

.. _dependency_groups:
Optional Dependency Groups
++++++++++++++++++++++++++

Pyomo defines three dependency groups to simplify installation of
optional dependencies:

* **tests** – Dependencies needed only to run the automated test suite
  and continuous integration infrastructure (e.g., ``pytest``,
  ``parameterized``, ``coverage``).

* **docs** – Dependencies needed to build the Sphinx-based documentation
  (e.g., ``sphinx``, ``sphinx_rtd_theme``, ``numpydoc``).

* **optional** – Dependencies that enable extended
  functionality throughout the Pyomo codebase, such as numerical
  computations or data handling (e.g., ``numpy``, ``pandas``,
  ``matplotlib``).

These optional dependencies can be installed selectively using standard
``pip`` commands. For example:

::

   pip install pyomo[optional]
   pip install pyomo[tests,docs]

Pyomo's continuous integration infrastructure regularly tests against
the most recent versions of all optional dependencies to ensure that
Pyomo remains compatible with current releases in the Python ecosystem.
When incompatibilities are identified, the setup configuration is
updated accordingly.

.. note::

   For more information on installing Pyomo with optional extras,
   see :doc:`/getting_started/installation`.

Solvers
+++++++

Pyomo does not bundle or directly distribute optimization solvers.
We recognize that solver installation can be challenging for new users.
To assist with this process, see the solver
availability table and installation guidance in
:doc:`/getting_started/solvers`. This table lists solvers that can be
installed via ``pip`` or ``conda`` where available, but Pyomo itself
does not include or require any specific solver as a dependency.

Miscellaneous Conventions
-------------------------

There are a variety of long-standing conventions that have become
standard across the project. This list will be amended as conventions come
up, so please refer to it regularly for updates:

* **Environment imports:** Import the main Pyomo environment as  
  ``import pyomo.environ as pyo``. Avoid all uses of ``import *``.

* **Export lists:** Do not define ``__all__`` in modules.
  Public symbols are determined by naming and documentation, not explicit lists.

* **Circular imports:** Make every effort to avoid circular imports. When
  circular imports are absolutely necessary, leverage :py:func`attempt_import()`
  to explicitly break the cycle. To help with this, some module namespaces
  have additional requirements:

  * ``pyomo.common``: modules within :py:mod:`pyomo.common` *must* not
    import *any* Pyomo modules outside of :py:mod:`pyomo.common`
  * ``pyomo.core.expr``: modules within :py:mod:`pyomo.core.expr` should not
    import modules outside of :py:mod:`pyomo.common`
    or :py:mod:`pyomo.core.expr`.
  * ``pyomo.core.base``: modules within :py:mod:`pyomo.core.base` should not
    import modules outside of :py:mod:`pyomo.common`,
    :py:mod:`pyomo.core.expr`, or :py:mod:`pyomo.core.base`.

* **Print statements:** Avoid printing or writing directly to ``stdout``. Pyomo is a
  library, not an application, and copious output can interfere with downstream
  tools and workflows. Use the appropriate logger instead. Print
  information only when the user has enabled or requested it.

* **Pull Request naming:** Pull Request titles are added to the CHANGELOG
  and the release notes. The Pyomo development team reserves the right to
  alter titles as appropriate to ensure they fit the look and feel of
  other titles in the CHANGELOG.

* **Full commit history:** We do **not** squash-merge Pull Requests,
  preferring to retain the entire commit history.

* **URLs:** All links in code, comments, and documentation must use ``https`` 
  rather than ``http``.

* **File headers:** Every ``.py`` file must begin with the standard Pyomo
  copyright header:

  .. code-block:: text

     #  ___________________________________________________________________________
     #
     #  Pyomo: Python Optimization Modeling Objects
     #  Copyright (c) 2008-2025
     #  National Technology and Engineering Solutions of Sandia, LLC
     #  Under the terms of Contract DE-NA0003525 with National Technology and
     #  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
     #  rights in this software.
     #  This software is distributed under the 3-clause BSD License.
     #  ___________________________________________________________________________

  Update the year range as appropriate when modifying files.
