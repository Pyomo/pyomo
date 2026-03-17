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

Commitment to Published APIs
++++++++++++++++++++++++++++

We treat functionality and examples published in the most recent edition of the
Pyomo book [PyomoBookIII]_ ("The Book") as our public API commitment.
The interfaces and APIs appearing in The Book
will be supported (although possibly in a deprecated form) until
the next major Pyomo release, which will generally coincide with a
new edition of the book.

This commitment ensures that teaching materials, training resources, and
long-term codebases built following those examples will remain valid across an
entire major release.

Core APIs
+++++++++

Functionality that is part of the Pyomo source tree but not explicitly
included in the book is also expected to be stable if it resides outside
the ``pyomo.addons``, ``pyomo.devel``, and ``pyomo.unsupported`` namespaces.
This functionality is referred to as "core" by the Pyomo development team.

When changes to core APIs become necessary, we will endeavor to follow one
(or both) of the following steps:

1. **Deprecation warnings** are added in advance of functionality removal.
   These are visible to users at import or execution time, with clear
   guidance on replacement functionality. For core functionality not
   mentioned in the published Pyomo book, deprecated interfaces are
   generally expected to remain available for **at least two minor Pyomo
   releases** following the introduction of the deprecation warning. For
   example, functionality deprecated in release ``X.Y.Z`` should not be
   removed before release ``X.(Y+2).0``.

2. **Relocation warnings** are provided for any relocated functionality.
   These modules import from their old locations and print a warning about
   the relocation in order to assist users' transition.

Ideally, changes in this fashion allow users and downstream packages to
adapt gradually without abrupt breakage.

.. note::

   For detailed guidance on how to mark functionality for deprecation or
   removal within the Pyomo codebase, see
   :doc:`/explanation/developer_utils/deprecation`.

Stable Extensions
+++++++++++++++++

The ``pyomo.addons`` namespace contains extensions that are intended to be
**mostly stable and reliable for downstream use**, while remaining
outside the Pyomo core.

Functionality in ``pyomo.addons`` is expected to follow Pyomo's coding,
testing, documentation, and backward-compatibility standards. While not
held to the same guarantees as core APIs or Book-published interfaces,
users should be able to rely on ``pyomo.addons`` functionality across
minor Pyomo releases.

Experimental Functionality
++++++++++++++++++++++++++

The ``pyomo.devel`` namespace contains experimental or rapidly evolving
functionality intended for active research, prototyping, and early-stage
development.

APIs under ``pyomo.devel`` may change or be removed between releases
without deprecation warnings. Users should not rely on functionality in
this namespace for production workflows.

Unsupported Functionality
+++++++++++++++++++++++++

The ``pyomo.unsupported`` namespace contains code that no longer has an
active maintainer or future development plans.

Functionality under this namespace may not work and is **NOT** routinely
tested through the standard Pyomo test harness. No compatibility or
stability guarantees are provided.

Historical Note
^^^^^^^^^^^^^^^

Earlier versions of Pyomo placed all experimental and non-core
functionality under a single namespace, ``pyomo.contrib``. While this
approach enabled rapid sharing of new modeling tools and research code,
it made it difficult for users to distinguish between stable,
maintained functionality and experimental or unsupported features.

The new namespace structure (``pyomo.addons``, ``pyomo.devel``, and
``pyomo.unsupported``) has been introduced to provide clearer signals about
stability, maintenance expectations, and compatibility guarantees, while
preserving Pyomo's long-standing support for community-driven development.

This historical namespace is documented here for context only. New
development should follow the current namespace guidelines described
above.


Dependency Management
---------------------

Minimal Core Dependencies
+++++++++++++++++++++++++

The core Pyomo codebase is designed to be a *Pure Python* library with
minimal dependencies outside the standard Python library (currently,
there are no hard external dependencies).

This approach simplifies installation, reduces the burden on derived packages,
and lessens the likelihood of triggering dependency conflicts.
Additionally, this allows users to install and run Pyomo in
resource-constrained or isolated environments (such as teaching
containers or HPC systems).

Optional Dependencies
+++++++++++++++++++++

Some extended Pyomo functionality relies on additional optional Python packages.
An optional dependency must not be imported (or required) for the Pyomo
environment. That is::

   import pyomo.environ

should not raise an ``ImportError`` if the dependency is missing. Further, the
Pyomo test harness (``pytest pyomo``) must run without error/failure if any
optional dependencies are missing (except for the dependencies required by
the ``tests`` :ref:`dependency group <dependency_groups>`).

Pyomo makes extensive use of :py:func:`attempt_import()` to support the
standardized and convenient use of optional dependencies. Further, many
common dependencies are directly importable through
``pyomo.common.dependencies`` without immediately triggering the dependency
import; for example::

   # Importing numpy from dependencies does not trigger the import
   from pyomo.common.dependencies import numpy as np, numpy_available
   
   # but testing the availability or using the module will trigger the import
   if numpy_available:
      a = np.array([1, 2, 3])
   

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

There are a variety of long-standing conventions that have become standard
across the project. This list will be amended as conventions come up, so please
refer to it regularly for updates:

* **Fail loudly:** Silent failure is strongly discouraged. Code should
  defensively guard against unexpected or unsupported cases and raise
  explicit exceptions (e.g., ``NotImplementedError``) rather than silently
  producing incorrect or ambiguous results. Some of the most difficult
  Pyomo bugs to diagnose arise from silently incorrect mathematics.

* **Document and enforce assumptions:** When behavior that a user could
  reasonably expect is ambiguous, code should clearly document the
  assumptions being made and fail loudly when those assumptions are
  violated.

* **Print statements:** Avoid printing or writing directly to ``stdout``. Pyomo is a
  library, not an application, and copious output can interfere with downstream
  tools and workflows. Use the appropriate logger instead. Print
  information only when the user has enabled or requested it.

* **Logging:** Use Python's ``logging`` framework for diagnostic and
  informational output. Loggers should generally be created using
  ``logging.getLogger(__name__)`` to ensure messages are properly scoped
  and can be enabled or filtered at the module or package level.

* **Active components define the model:** Pyomo models are defined as the
  set of *active* components reachable from the root ``Block``. Not all
  modeling objects are active components (notably, ``Var`` objects are
  not). Algorithms and writers should respect this distinction when
  traversing or interpreting models.

* **Avoid iterating over variables directly:** Direct iteration over
  variables using ``m.component_data_objects(Var, ...)`` is rarely
  appropriate, as it returns all variables declared on the model
  hierarchy regardless of their mathematical relevance. When gathering
  variables associated with a model formulation, prefer utilities such as
  ``get_vars_from_components`` with ``Constraint`` or
  ``(Constraint, Objective)`` as the component types.

* **Writers must validate component types:** Writers and solver-facing
  infrastructure should explicitly detect and warn about active
  Pyomo components they do not recognize. Pyomo supports extended
  modeling environments (e.g., DAE and GDP), and silently ignoring
  unexpected structures can result in invalid solver input. The utility
  ``categorize_valid_components`` in ``pyomo.repn.util`` may be used to
  assist with this validation.

* **Do not use names as identifiers:** Component names and strings should
  generally not be used to track Pyomo components within algorithms or
  writers. Names are not guaranteed to be unique or stable across model
  transformations. Prefer data structures from
  ``pyomo.common.collections`` that support using components directly as
  keys.

* **Environment imports:** Import the main Pyomo environment as  
  ``import pyomo.environ as pyo``. Avoid all uses of ``import *``.

* **Export lists:** Do not define ``__all__`` in modules.
  Public symbols are determined by naming and documentation, not explicit lists.

* **Circular imports:** Make every effort to avoid circular imports. When
  circular imports are absolutely necessary, leverage :py:func:`attempt_import()`
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

* **Naming conventions:** Follow PEP 8 naming conventions, including
  descriptive ``snake_case`` for functions and variables and
  ``PascalCase`` for classes. Functions should generally be named as verb
  phrases, while classes should be noun-like representations of concepts.

* **Avoid code duplication:** Repeated or copy-pasted code is strongly
  discouraged. Duplication increases maintenance burden and often leads
  to inconsistent behavior. When code patterns begin to repeat,
  contributors are encouraged to refactor common functionality or
  discuss design alternatives with the core development team.

* **URLs:** All links in code, comments, and documentation must use ``https`` 
  rather than ``http`` wherever possible.

* **File headers:** Every ``.py`` file must begin with the standard Pyomo
  copyright header:

  .. literalinclude:: ../../pyomo/addons/__init__.py
     :language: text

  Update the year range as appropriate when modifying files.

* **Full commit history:** We do **not** squash-merge Pull Requests,
  preferring to retain the entire commit history.

* **Pull Request naming:** Pull Request titles are added to the CHANGELOG
  and the release notes. The Pyomo development team reserves the right to
  alter titles as appropriate to ensure they fit the look and feel of
  other titles in the CHANGELOG.
