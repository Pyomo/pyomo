=================================
The Pyomo Configuration System
=================================

.. py:currentmodule:: pyomo.common.config

The Pyomo configuration system provides a set of three classes
(:py:class:`ConfigDict`, :py:class:`ConfigList`, and
:py:class:`ConfigValue`) for managing and documenting structured
configuration information and user input.  The system is based around
the :class:`ConfigValue` class, which provides storage for a single
configuration entry.  :class:`ConfigValue` objects can be grouped using
two containers (:class:`ConfigDict` and :class:`ConfigList`) that
provide functionality analogous to Python's :class:`dict` and
:class:`list` classes, respectively.

At its simplest, the configuration system allows for developers to specify a
dictionary of documented configuration entries:

.. testcode::

    from pyomo.common.config import (
        ConfigDict, ConfigList, ConfigValue
    )
    config = ConfigDict()
    config.declare('filename', ConfigValue(
        default=None,
        domain=str,
        description="Input file name",
    ))
    config.declare("bound tolerance", ConfigValue(
        default=1E-5,
        domain=float,
        description="Bound tolerance",
        doc="Relative tolerance for bound feasibility checks"
    ))
    config.declare("iteration limit", ConfigValue(
        default=30,
        domain=int,
        description="Iteration limit",
        doc="Number of maximum iterations in the decomposition methods"
    ))

Users can then provide values for those entries, and retrieve the
current values:

.. doctest::

    >>> config['filename'] = 'tmp.txt'
    >>> print(config['filename'])
    tmp.txt
    >>> print(config['iteration limit'])
    30

For convenience, :class:`ConfigDict` objects support read/write access via
attributes (with spaces in the declaration names replaced by
underscores):

.. doctest::

    >>> print(config.filename)
    tmp.txt
    >>> print(config.iteration_limit)
    30
    >>> config.iteration_limit = 20
    >>> print(config.iteration_limit)
    20

Domain validation
=================

All Config objects support a ``domain`` keyword that accepts a callable
object (type, function, or callable instance).  The domain callable
should take a single argument (the incoming data value) and map it onto
the desired domain, optionally
performing domain validation (see :py:class:`ConfigValue`,
:py:class:`ConfigDict`, and :py:class:`ConfigList` for more
information).  This allows client code to accept a very flexible set of
inputs without "cluttering" the code with input validation:

.. doctest::

    >>> config.iteration_limit = 35.5
    >>> print(config.iteration_limit)
    35
    >>> print(type(config.iteration_limit).__name__)
    int

In addition to common types (like :class:`int`, :class:`float`,
:class:`bool`, and :class:`str`), the configuration system provides a
number of custom domain validators for common use cases:

.. autosummary::

   Bool
   Integer
   PositiveInt
   NegativeInt
   NonNegativeInt
   NonPositiveInt
   PositiveFloat
   NegativeFloat
   NonPositiveFloat
   NonNegativeFloat
   In
   InEnum
   IsInstance
   ListOf
   Module
   Path
   PathList
   DynamicImplicitDomain

.. _class_config:

Configuring class hierarchies
=============================

A feature of the configuration system is that the core classes all implement
``__call__``, and can themselves be used as ``domain`` values.  Beyond
providing domain verification for complex hierarchical structures, this
feature allows :class:`ConfigDict` objects to cleanly support extension
and the configuration of
derived classes.  Consider the following example:

.. doctest::

    >>> class Base:
    ...     CONFIG = ConfigDict()
    ...     CONFIG.declare('filename', ConfigValue(
    ...         default='input.txt',
    ...         domain=str,
    ...     ))
    ...     def __init__(self, **kwds):
    ...         self.cfg = self.CONFIG(kwds)
    ...         self.cfg.display()
    ...
    >>> class Derived(Base):
    ...     CONFIG = Base.CONFIG()
    ...     CONFIG.declare('pattern', ConfigValue(
    ...         default=None,
    ...         domain=str,
    ...     ))
    ...
    >>> tmp = Base(filename='foo.txt')
    filename: foo.txt
    >>> tmp = Derived(pattern='.*warning')
    filename: input.txt
    pattern: .*warning

Here, the base class ``Base`` declares a class-level attribute ``CONFIG`` as a
:class:`ConfigDict` containing a single entry (``filename``).  The derived class
(``Derived``) then starts by making a copy of the base class' ``CONFIG``,
and then defines an additional entry (``pattern``).  Instances of the base
class will still create ``cfg`` attributes that only have the single
``filename`` entry, whereas instances of the derived class will have ``cfg``
attributes with two entries: the ``pattern`` entry declared by the derived
class, and the ``filename`` entry "inherited" from the base class.

An extension of this design pattern provides a clean approach for
handling "ephemeral" instance options.  Consider an interface to an
external "solver".  Our class implements a ``solve()`` method that takes a
problem and sends it to the solver along with some solver configuration
options.  We would like to be able to set those options "persistently"
on instances of the interface class, but still override them
"temporarily" for individual calls to ``solve()``.  We implement this by
creating copies of the class's configuration for both specific instances
and for use by each ``solve()`` call:

.. testcode::

   class Solver:
       CONFIG = ConfigDict()
       CONFIG.declare('iterlim', ConfigValue(
           default=10,
           domain=int,
       ))

       def __init__(self, **kwds):
           self.config = self.CONFIG(kwds)

       def solve(self, model, **options):
           config = self.config(options)
           # Solve the model with the specified iterlim
           config.display()

.. doctest::

   >>> solver = Solver()
   >>> solver.solve(None)
   iterlim: 10
   >>> solver.config.iterlim = 20
   >>> solver.solve(None)
   iterlim: 20
   >>> solver.solve(None, iterlim=50)
   iterlim: 50
   >>> solver.solve(None)
   iterlim: 20

This design pattern is widely used across Pyomo; particularly for
configuring solver interfaces and transformations.  We provide a
decorator to simplify the process of documenting these ``CONFIG``
attributes:

.. testcode::

   from pyomo.common.config import document_class_CONFIG

   @document_class_CONFIG(methods=['solve'])
   class MySolver:
       """Interface to My Solver"""
       #
       #: Global class configuration; see :ref:`MySolver_CONFIG`
       CONFIG = ConfigDict()
       CONFIG.declare('iterlim', ConfigValue(
           default=10,
           domain=int,
           doc="Solver iteration limit",
       ))
       #
       def __init__(self, **kwds):
           #: Instance configuration; see :ref:`MySolver_CONFIG`
           self.config = self.CONFIG(kwds)
       #
       def solve(self, model, **options):
           """Solve `model` using My Solver"""
           #
           config = self.config(options)
           # Solve the model with the specified iterlim
           config.display()

.. doctest::

   >>> print(MySolver.__doc__)
   Interface to My Solver
   <BLANKLINE>
   **Class configuration**
   <BLANKLINE>
   This class leverages the Pyomo Configuration System for managing
   configuration options.  See the discussion on :ref:`configuring class
   hierarchies <class_config>` for more information on how configuration
   class attributes, instance attributes, and method keyword arguments
   interact.
   <BLANKLINE>
   .. _MySolver::CONFIG:
   <BLANKLINE>
   CONFIG
   ------
   iterlim: int, default=10
   <BLANKLINE>
       Solver iteration limit

   >>> print(MySolver.solve.__doc__)
   Solve `model` using My Solver
   <BLANKLINE>
   Keyword Arguments
   -----------------
   iterlim: int, default=10
   <BLANKLINE>
       Solver iteration limit


Interacting with argparse
=========================

In addition to basic storage and retrieval, the configuration system provides
hooks to the argparse command-line argument parsing system.  Individual
configuration entries can be declared as :mod:`argparse` arguments using the
:py:meth:`~ConfigBase.declare_as_argument` method.  To make declaration
simpler, the :py:meth:`~ConfigDict.declare` method returns the declared configuration
object so that the argument declaration can be done inline:

.. testcode::

    import argparse
    config = ConfigDict()
    config.declare('iterlim', ConfigValue(
        domain=int,
        default=100,
        description="iteration limit",
    )).declare_as_argument()
    config.declare('lbfgs', ConfigValue(
        domain=bool,
        description="use limited memory BFGS update",
    )).declare_as_argument()
    config.declare('linesearch', ConfigValue(
        domain=bool,
        default=True,
        description="use line search",
    )).declare_as_argument()
    config.declare('relative tolerance', ConfigValue(
        domain=float,
        description="relative convergence tolerance",
    )).declare_as_argument('--reltol', '-r', group='Tolerances')
    config.declare('absolute tolerance', ConfigValue(
        domain=float,
        description="absolute convergence tolerance",
    )).declare_as_argument('--abstol', '-a', group='Tolerances')

The :class:`ConfigDict` can then be used to initialize (or augment) an
:class:`argparse.ArgumentParser` object:

.. testcode::

    parser = argparse.ArgumentParser("tester")
    config.initialize_argparse(parser)


Key information from the :class:`ConfigDict` is automatically transferred over
to the :class:`~argparse.ArgumentParser` object:

.. doctest::
   :hide:

    >>> import os
    >>> original_environ, os.environ = os.environ, os.environ.copy()
    >>> os.environ['COLUMNS'] = '80'

.. doctest::

    >>> print(parser.format_help())
    usage: tester [-h] [--iterlim INT] [--lbfgs] [--disable-linesearch]
                  [--reltol FLOAT] [--abstol FLOAT]
    ...
      -h, --help            show this help message and exit
      --iterlim INT         iteration limit
      --lbfgs               use limited memory BFGS update
      --disable-linesearch  [DON'T] use line search
    <BLANKLINE>
    Tolerances:
      --reltol... -r FLOAT  relative convergence tolerance
      --abstol... -a FLOAT  absolute convergence tolerance
    <BLANKLINE>

..
   NOTE: the text above uses an Ellipsis because the line is rendered as:
      --reltol FLOAT, -r FLOAT  relative convergence tolerance
   through Python 3.12, and beginning in Python 3.13 changed to:
      --reltol, -r FLOAT  relative convergence tolerance

.. doctest::
   :hide:

    >>> os.environ = original_environ

Parsed arguments can then be imported back into the :class:`ConfigDict`:

.. doctest::

    >>> args=parser.parse_args(['--lbfgs', '--reltol', '0.1', '-a', '0.2'])
    >>> args = config.import_argparse(args)
    >>> config.display()
    iterlim: 100
    lbfgs: true
    linesearch: true
    relative tolerance: 0.1
    absolute tolerance: 0.2

Accessing user-specified values
===============================

It is frequently useful to know which values a user explicitly set, and
which values a user explicitly set but have never been retrieved.  The
configuration system provides two generator methods to return the items
that a user explicitly set (:py:meth:`user_values`) and the items that
were set but never retrieved (:py:meth:`unused_user_values`):

.. doctest::

    >>> print([val.name() for val in config.user_values()])
    ['lbfgs', 'relative tolerance', 'absolute tolerance']
    >>> print(config.relative_tolerance)
    0.1
    >>> print([val.name() for val in config.unused_user_values()])
    ['lbfgs', 'absolute tolerance']

Outputting the current state
============================

Configuration objects support two methods for generating output:
:py:meth:`~ConfigBase.display` and
:py:meth:`~ConfigBase.generate_yaml_template`.  The simpler is
:py:meth:`~ConfigBase.display`, which prints out the current values of
the configuration object (and if it is a container type, all of its
children).  :py:meth:`~ConfigBase.generate_yaml_template` is similar to
:py:meth:`~ConfigBase.display`, but also includes the description fields
as formatted comments.

.. testcode::

    solver_config = config
    config = ConfigDict()
    config.declare('output', ConfigValue(
        default='results.yml',
        domain=str,
        description='output results filename'
    ))
    config.declare('verbose', ConfigValue(
        default=0,
        domain=int,
        description='output verbosity',
        doc='This sets the system verbosity.  The default (0) only logs '
        'warnings and errors.  Larger integer values will produce '
        'additional log messages.',
    ))
    config.declare('solvers', ConfigList(
        domain=solver_config,
        description='list of solvers to apply',
    ))

.. doctest::

    >>> config.display()
    output: results.yml
    verbose: 0
    solvers: []
    >>> print(config.generate_yaml_template())
    output: results.yml  # output results filename
    verbose: 0           # output verbosity
    solvers: []          # list of solvers to apply
    <BLANKLINE>

It is important to note that both methods document the current state of
the configuration object.  So, in the example above, since the ``solvers``
list is empty, you will not get any information on the elements in the
list.  Of course, if you add a value to the list, then the data will be
output:

.. doctest::

    >>> tmp = config()
    >>> tmp.solvers.append({})
    >>> tmp.display()
    output: results.yml
    verbose: 0
    solvers:
      -
        iterlim: 100
        lbfgs: true
        linesearch: true
        relative tolerance: 0.1
        absolute tolerance: 0.2
    >>> print(tmp.generate_yaml_template())
    output: results.yml          # output results filename
    verbose: 0                   # output verbosity
    solvers:                     # list of solvers to apply
      -
        iterlim: 100             # iteration limit
        lbfgs: true              # use limited memory BFGS update
        linesearch: true         # use line search
        relative tolerance: 0.1  # relative convergence tolerance
        absolute tolerance: 0.2  # absolute convergence tolerance
    <BLANKLINE>

Generating documentation
========================

One of the most useful features of the Configuration system is the
ability to automatically generate documentation.  To accomplish this, we
rely on a series of formatters derived from :class:`ConfigFormatter`
that implement a visitor pattern for walking the hierarchy of
configuration containers (:class:`ConfigDict` and :class:`ConfigList`)
and documenting the members.  As the :class:`ConfigFormatter` was
designed to generate reference documentation, it behaves differently
from :meth:`~ConfigBase.display` or
:meth:`~ConfigBase.generate_yaml_template`):

    - For each configuration item, the ``doc`` field is output.  If the
      item has no ``doc``, then the ``description`` field is used.

    - List containers have their *domain* documented and not their
      current values.

The simplest interface for generating documentation is to call the
:py:meth:`~ConfigBase.generate_documentation` method.  This method
retrieves the specified formatter, instantiates it, and returns the
result from walking the configuration object.  The documentation format
can be configured through optional arguments.  The defaults generate
LaTeX documentation:

.. doctest::

    >>> print(config.generate_documentation())
    \begin{description}[topsep=0pt,parsep=0.5em,itemsep=-0.4em]
      \item[{output}]\hfill
        \\output results filename
      \item[{verbose}]\hfill
        \\This sets the system verbosity.  The default (0) only logs warnings and
        errors.  Larger integer values will produce additional log messages.
      \item[{solvers}]\hfill
        \\list of solvers to apply
      \begin{description}[topsep=0pt,parsep=0.5em,itemsep=-0.4em]
        \item[{iterlim}]\hfill
          \\iteration limit
        \item[{lbfgs}]\hfill
          \\use limited memory BFGS update
        \item[{linesearch}]\hfill
          \\use line search
        \item[{relative tolerance}]\hfill
          \\relative convergence tolerance
        \item[{absolute tolerance}]\hfill
          \\absolute convergence tolerance
      \end{description}
    \end{description}
    <BLANKLINE>

More useful is actually documenting the source code itself.  To this
end, the Configuration system provides three decorators that append
documentation of the referenced :class:`ConfigDict` (in
`numpydoc format <https://numpydoc.readthedocs.io/en/latest/>`_) for the most
common situations:

.. autosummary::

   document_configdict
   document_class_CONFIG
   document_kwargs_from_configdict
