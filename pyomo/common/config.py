#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
#  This module was originally developed as part of the PyUtilib project
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  ___________________________________________________________________________

import argparse
import os
import platform
import enum
import six
import re
import sys
from textwrap import wrap
import logging
import pickle

if six.PY3:
    import builtins as _builtins
else:
    import __builtin__ as _builtins

from six.moves import xrange

from pyomo.common.deprecation import deprecated

logger = logging.getLogger('pyomo.common.config')

if 'PYOMO_CONFIG_DIR' in os.environ:
    PYOMO_CONFIG_DIR = os.path.abspath(os.environ['PYOMO_CONFIG_DIR'])
elif platform.system().lower().startswith(('windows','cygwin')):
    PYOMO_CONFIG_DIR = os.path.abspath(
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Pyomo'))
else:
    PYOMO_CONFIG_DIR = os.path.abspath(
        os.path.join(os.environ.get('HOME', ''), '.pyomo'))

# Note that alternative platform-independent implementation of the above
# could be to use:
#
#   PYOMO_CONFIG_DIR = os.path.abspath(appdirs.user_data_dir('pyomo'))
#
# But would require re-adding the hard dependency on appdirs.  For now
# (13 Jul 20), the above appears to be sufficiently robust.

USER_OPTION = 0
ADVANCED_OPTION = 1
DEVELOPER_OPTION = 2

def PositiveInt(val):
    """Domain validation function admitting strictly positive integers

    This domain will admit positive integers, as well as any types that are convertible to positive integers.
    """
    ans = int(val)
    # We want to give an error for floating point numbers...
    if ans != float(val) or ans <= 0:
        raise ValueError(
            "Expected positive int, but received %s" % (val,))
    return ans

def NegativeInt(val):
    """Domain validation function admitting strictly negative integers

    This domain will admit negative integers, as well as any types that are convertible to negative integers.
    """
    ans = int(val)
    if ans != float(val) or ans >= 0:
        raise ValueError(
            "Expected negative int, but received %s" % (val,))
    return ans

def NonPositiveInt(val):
    """Domain validation function admitting non-positive integers (smaller than or equal to zero)

    This domain will admit non-positive integers, as well as any types that are convertible to non-positive integers.
    """
    ans = int(val)
    if ans != float(val) or ans > 0:
        raise ValueError(
            "Expected non-positive int, but received %s" % (val,))
    return ans

def NonNegativeInt(val):
    """Domain validation function admitting non-negative integers (greater than or equal to zero)

    This domain will admit non-negative integers, as well as any types that are convertible to non-negative integers.
    """
    ans = int(val)
    if ans != float(val) or ans < 0:
        raise ValueError(
            "Expected non-negative int, but received %s" % (val,))
    return ans

def PositiveFloat(val):
    """Domain validation function admitting strictly positive floating point numbers

    This domain will admit positive floating point numbers, as well as any types that are convertible to positive floating point numbers.
    """
    ans = float(val)
    if ans <= 0:
        raise ValueError(
            "Expected positive float, but received %s" % (val,))
    return ans

def NegativeFloat(val):
    """Domain validation function admitting strictly negative floating point numbers

    This domain will admit negative floating point numbers, as well as any types that are convertible to negative floating point numbers.
    """
    ans = float(val)
    if ans >= 0:
        raise ValueError(
            "Expected negative float, but received %s" % (val,))
    return ans

def NonPositiveFloat(val):
    """Domain validation function admitting strictly non-positive floating point numbers (smaller than or equal to zero)

    This domain will admit non-positive floating point numbers, as well as any types that are convertible to non-positive floating point numbers.
    """
    ans = float(val)
    if ans > 0:
        raise ValueError(
            "Expected non-positive float, but received %s" % (val,))
    return ans

def NonNegativeFloat(val):
    """Domain validation function admitting strictly non-negative floating point numbers (greater than or equal to zero)

    This domain will admit non-negative floating point numbers, as well as any types that are convertible to non-negative floating point numbers.
    """
    ans = float(val)
    if ans < 0:
        raise ValueError(
            "Expected non-negative float, but received %s" % (val,))
    return ans


class In(object):
    """Domain validation function admitting a list of possible values that a variable can be assigned to."""
    def __init__(self, domain, cast=None):
        self._domain = domain
        self._cast = cast

    def __call__(self, value):
        if self._cast is not None:
            v = self._cast(value)
        else:
            v = value
        if v in self._domain:
            return v
        raise ValueError("value %s not in domain %s" % (value, self._domain))


class Path(object):
    BasePath = None
    SuppressPathExpansion = False

    def __init__(self, basePath=None):
        self.basePath = basePath

    def __call__(self, path):
        #print "normalizing path '%s' " % (path,),
        path = str(path)
        if path is None or Path.SuppressPathExpansion:
            return path

        if self.basePath:
            base = self.basePath
        else:
            base = Path.BasePath
        if type(base) is ConfigValue:
            base = base.value()
        if base is None:
            base = ""
        else:
            base = str(base).lstrip()

        # We want to handle the CWD variable ourselves.  It should
        # always be in a known location (the beginning of the string)
        if base and base[:6].lower() == '${cwd}':
            base = os.getcwd() + base[6:]
        if path and path[:6].lower() == '${cwd}':
            path = os.getcwd() + path[6:]

        ans = os.path.normpath(os.path.abspath(os.path.join(
            os.path.expandvars(os.path.expanduser(base)),
            os.path.expandvars(os.path.expanduser(path)))))
        #print "to '%s'" % (ans,)
        return ans

class PathList(Path):
    def __call__(self, data):
        if hasattr(data, "__iter__") and not isinstance(data, six.string_types):
            return [ super(PathList, self).__call__(i) for i in data ]
        else:
            return [ super(PathList, self).__call__(data) ]


def add_docstring_list(docstring, configdict, indent_by=4):
    """Returns the docstring with a formatted configuration arguments listing."""
    return docstring + (" " * indent_by).join(
        configdict.generate_documentation(
            block_start="Keyword Arguments\n-----------------\n",
            block_end="",
            item_start="%s\n",
            item_body="  %s",
            item_end="",
            indent_spacing=0,
            width=256
        ).splitlines(True))


class ConfigEnum(enum.Enum):
    @classmethod
    def from_enum_or_string(cls, arg):
        if type(arg) is str:
            return cls[arg]
        else:
            # Handles enum or integer inputs
            return cls(arg)


"""=================================
The Pyomo Configuration System
=================================

The Pyomo config system provides a set of three classes
(:py:class:`ConfigDict`, :py:class:`ConfigList`, and
:py:class:`ConfigValue`) for managing and documenting structured
configuration information and user input.  The system is based around
the ConfigValue class, which provides storage for a single configuration
entry.  ConfigValue objects can be grouped using two containers
(ConfigDict and ConfigList), which provide functionality analogous to
Python's dict and list classes, respectively.

At its simplest, the Config system allows for developers to specify a
dictionary of documented configuration entries, allow users to provide
values for those entries, and retrieve the current values:

.. doctest::
    :hide:

    >>> import argparse
    >>> from pyomo.common.config import (
    ...     ConfigDict, ConfigList, ConfigValue, In,
    ... )

.. doctest::

    >>> config = ConfigDict()
    >>> config.declare('filename', ConfigValue(
    ...     default=None,
    ...     domain=str,
    ...     description="Input file name",
    ... ))
    <pyomo.common.config.ConfigValue object at ...>
    >>> config.declare("bound tolerance", ConfigValue(
    ...     default=1E-5,
    ...     domain=float,
    ...     description="Bound tolerance",
    ...     doc="Relative tolerance for bound feasibility checks"
    ... ))
    <pyomo.common.config.ConfigValue object at ...>
    >>> config.declare("iteration limit", ConfigValue(
    ...     default=30,
    ...     domain=int,
    ...     description="Iteration limit",
    ...     doc="Number of maximum iterations in the decomposition methods"
    ... ))
    <pyomo.common.config.ConfigValue object at ...>
    >>> config['filename'] = 'tmp.txt'
    >>> print(config['filename'])
    tmp.txt
    >>> print(config['iteration limit'])
    30

For convenience, ConfigDict objects support read/write access via
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

All Config objects support a ``domain`` keyword that accepts a callable
object (type, function, or callable instance).  The domain callable
should take data and map it onto the desired domain, optionally
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

Configuring class hierarchies
=============================

A feature of the Config system is that the core classes all implement
``__call__``, and can themselves be used as ``domain`` values.  Beyond
providing domain verification for complex hierarchical structures, this
feature allows ConfigDicts to cleanly support the configuration of
derived objects.  Consider the following example:

.. doctest::

    >>> class Base(object):
    ...     CONFIG = ConfigDict()
    ...     CONFIG.declare('filename', ConfigValue(
    ...         default='input.txt',
    ...         domain=str,
    ...     ))
    ...     def __init__(self, **kwds):
    ...         c = self.CONFIG(kwds)
    ...         c.display()
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

Here, the base class ``Base`` declares a class-level attribute CONFIG as a
ConfigDict containing a single entry (``filename``).  The derived class
(``Derived``) then starts by making a copy of the base class' ``CONFIG``,
and then defines an additional entry (`pattern`).  Instances of the base
class will still create ``c`` instances that only have the single
``filename`` entry, whereas instances of the derived class will have ``c``
instances with two entries: the ``pattern`` entry declared by the derived
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

.. doctest::

    >>> class Solver(object):
    ...     CONFIG = ConfigDict()
    ...     CONFIG.declare('iterlim', ConfigValue(
    ...         default=10,
    ...         domain=int,
    ...     ))
    ...     def __init__(self, **kwds):
    ...         self.config = self.CONFIG(kwds)
    ...     def solve(self, model, **options):
    ...         config = self.config(options)
    ...         # Solve the model with the specified iterlim
    ...         config.display()
    ...
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


Interacting with argparse
=========================

In addition to basic storage and retrieval, the Config system provides
hooks to the argparse command-line argument parsing system.  Individual
Config entries can be declared as argparse arguments.  To make
declaration simpler, the :py:meth:`declare` method returns the declared Config
object so that the argument declaration can be done inline:

.. doctest::

    >>> config = ConfigDict()
    >>> config.declare('iterlim', ConfigValue(
    ...     domain=int,
    ...     default=100,
    ...     description="iteration limit",
    ... )).declare_as_argument()
    <pyomo.common.config.ConfigValue object at ...>
    >>> config.declare('lbfgs', ConfigValue(
    ...     domain=bool,
    ...     description="use limited memory BFGS update",
    ... )).declare_as_argument()
    <pyomo.common.config.ConfigValue object at ...>
    >>> config.declare('linesearch', ConfigValue(
    ...     domain=bool,
    ...     default=True,
    ...     description="use line search",
    ... )).declare_as_argument()
    <pyomo.common.config.ConfigValue object at ...>
    >>> config.declare('relative tolerance', ConfigValue(
    ...     domain=float,
    ...     description="relative convergence tolerance",
    ... )).declare_as_argument('--reltol', '-r', group='Tolerances')
    <pyomo.common.config.ConfigValue object at ...>
    >>> config.declare('absolute tolerance', ConfigValue(
    ...     domain=float,
    ...     description="absolute convergence tolerance",
    ... )).declare_as_argument('--abstol', '-a', group='Tolerances')
    <pyomo.common.config.ConfigValue object at ...>

The ConfigDict can then be used to initialize (or augment) an argparse
ArgumentParser object:

.. doctest::

    >>> parser = argparse.ArgumentParser("tester")
    >>> config.initialize_argparse(parser)


Key information from the ConfigDict is automatically transferred over
to the ArgumentParser object:

.. doctest::

    >>> print(parser.format_help())
    usage: tester [-h] [--iterlim INT] [--lbfgs] [--disable-linesearch]
                  [--reltol FLOAT] [--abstol FLOAT]
    <BLANKLINE>
    optional arguments:
      -h, --help            show this help message and exit
      --iterlim INT         iteration limit
      --lbfgs               use limited memory BFGS update
      --disable-linesearch  [DON'T] use line search
    <BLANKLINE>
    Tolerances:
      --reltol FLOAT, -r FLOAT
                            relative convergence tolerance
      --abstol FLOAT, -a FLOAT
                            absolute convergence tolerance
    <BLANKLINE>

Parsed arguments can then be imported back into the ConfigDict:

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
which values a user explicitly set, but have never been retrieved.  The
configuration system provides two gemerator methods to return the items
that a user explicitly set (:py:meth:`user_values`) and the items that
were set but never retrieved (:py:meth:`unused_user_values`):

.. doctest::

    >>> print([val.name() for val in config.user_values()])
    ['lbfgs', 'relative tolerance', 'absolute tolerance']
    >>> print(config.relative_tolerance)
    0.1
    >>> print([val.name() for val in config.unused_user_values()])
    ['lbfgs', 'absolute tolerance']

Generating output & documentation
=================================

Configuration objects support three methods for generating output and
documentation: :py:meth:`display()`,
:py:meth:`generate_yaml_template()`, and
:py:meth:`generate_documentation()`.  The simplest is
:py:meth:`display()`, which prints out the current values of the
configuration object (and if it is a container type, all of it's
children).  :py:meth:`generate_yaml_template` is simular to
:py:meth:`display`, but also includes the description fields as
formatted comments.

.. doctest::

    >>> solver_config = config
    >>> config = ConfigDict()
    >>> config.declare('output', ConfigValue(
    ...     default='results.yml',
    ...     domain=str,
    ...     description='output results filename'
    ... ))
    <pyomo.common.config.ConfigValue object at ...>
    >>> config.declare('verbose', ConfigValue(
    ...     default=0,
    ...     domain=int,
    ...     description='output verbosity',
    ...     doc='This sets the system verbosity.  The default (0) only logs '
    ...     'warnings and errors.  Larger integer values will produce '
    ...     'additional log messages.',
    ... ))
    <pyomo.common.config.ConfigValue object at ...>
    >>> config.declare('solvers', ConfigList(
    ...     domain=solver_config,
    ...     description='list of solvers to apply',
    ... ))
    <pyomo.common.config.ConfigList object at ...>
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
the configuration object.  So, in the example above, since the `solvers`
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

The third method (:py:meth:`generate_documentation`) behaves
differently.  This method is designed to generate reference
documentation.  For each configuration item, the `doc` field is output.
If the item has no `doc`, then the `description` field is used.

List containers have their *domain* documented and not their current
values.  The documentation can be configured through optional arguments.
The defaults generate LaTeX documentation:

.. doctest::

    >>> print(config.generate_documentation())
    \\begin{description}[topsep=0pt,parsep=0.5em,itemsep=-0.4em]
      \\item[{output}]\hfill
        \\\\output results filename
      \\item[{verbose}]\hfill
        \\\\This sets the system verbosity.  The default (0) only logs warnings and
        errors.  Larger integer values will produce additional log messages.
      \\item[{solvers}]\hfill
        \\\\list of solvers to apply
      \\begin{description}[topsep=0pt,parsep=0.5em,itemsep=-0.4em]
        \\item[{iterlim}]\hfill
          \\\\iteration limit
        \\item[{lbfgs}]\hfill
          \\\\use limited memory BFGS update
        \\item[{linesearch}]\hfill
          \\\\use line search
        \\item[{relative tolerance}]\hfill
          \\\\relative convergence tolerance
        \\item[{absolute tolerance}]\hfill
          \\\\absolute convergence tolerance
      \\end{description}
    \\end{description}
    <BLANKLINE>

"""

def _dump(*args, **kwds):
    try:
        from yaml import dump
    except ImportError:
        #dump = lambda x,**y: str(x)
        # YAML uses lowercase True/False
        def dump(x, **args):
            if type(x) is bool:
                return str(x).lower()
            return str(x)
    assert '_dump' in globals()
    globals()['_dump'] = dump
    return dump(*args, **kwds)


def _munge_name(name, space_to_dash=True):
    if space_to_dash:
        name = re.sub(r'\s', '-', name)
    name = re.sub(r'_', '-', name)
    return re.sub(r'[^a-zA-Z0-9-_]', '_', name)


_leadingSpace = re.compile('^([ \n\t]*)')

def _strip_indentation(doc):
    if not doc:
        return doc
    lines = doc.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if len(lines) == 1:
        return doc.lstrip()
    minIndent = min(len(_leadingSpace.match(l).group(0)) for l in lines[1:])
    if len(_leadingSpace.match(lines[0]).group(0)) <= minIndent:
        lines[0] = lines[0].strip()
    else:
        lines[0] = lines[0][minIndent:].rstrip()
    for i, l in enumerate(lines[1:]):
        lines[i + 1] = l[minIndent:].rstrip()
    return '\n'.join(lines)


def _value2string(prefix, value, obj):
    _str = prefix
    if value is not None:
        try:
            _data = value._data if value is obj else value
            if getattr(_builtins, _data.__class__.__name__, None
                   ) is not None:
                _str += _dump(_data, default_flow_style=True).rstrip()
                if _str.endswith("..."):
                    _str = _str[:-3].rstrip()
            else:
                _str += str(_data)
        except:
            _str += str(type(_data))
    return _str.rstrip()

def _value2yaml(prefix, value, obj):
    _str = prefix
    if value is not None:
        try:
            _data = value._data if value is obj else value
            _str += _dump(_data, default_flow_style=True).rstrip()
            if _str.endswith("..."):
                _str = _str[:-3].rstrip()
        except:
            _str += str(type(_data))
    return _str.rstrip()


class _UnpickleableDomain(object):
    def __init__(self, obj):
        self._type = type(obj).__name__
        self._name = obj.name(True)

    def __call__(self, arg):
        logging.error(
"""%s %s was pickled with an unpicklable domain.
    The domain was stripped and lost during the pickle process.  Setting
    new values on the restored object cannot be mapped into the correct
    domain.
""" % ( self._type, self._name))
        return arg

def _picklable(field,obj):
    ftype = type(field)
    if ftype in _picklable.known:
        return field if _picklable.known[ftype] else _UnpickleableDomain(obj)
    try:
        pickle.dumps(field)
        _picklable.known[ftype] = True
        return field
    except:
        # Contrary to the documentation, Python is not at all consistent
        # with the exception that is raised when pickling an object
        # fails:
        #
        #    Python 2.6 - 3.4:  pickle.PicklingError
        #    Python 3.5 - 3.6:  AttributeError
        #    Python 2.6 - 2.7 (cPickle):  TypeError
        #
        # What we are concerned about is masking things like recursion
        # errors.  Unfortunately, Python is not quite consistent there,
        # either: exceeding recursion depth raises a RuntimeError
        # through 3.4, then switches to a RecursionError (a derivative
        # of RuntimeError).
        if isinstance(sys.exc_info()[0], RuntimeError):
            raise
        _picklable.known[ftype] = False
        return _UnpickleableDomain(obj)

_picklable.known = {}

class ConfigBase(object):
    __slots__ = ('_parent', '_name', '_userSet', '_userAccessed', '_data',
                 '_default', '_domain', '_description', '_doc', '_visibility',
                 '_argparse')

    # This just needs to be any singleton-like object; we use it so that
    # we can tell if an argument is provided (and we can't use None as
    # None is a valid user-specified argument).  Making it a class helps
    # when Config objects are pickled.
    class NoArgument(object): pass

    def __init__(self,
                 default=None,
                 domain=None,
                 description=None,
                 doc=None,
                 visibility=0):
        self._parent = None
        self._name = None
        self._userSet = False
        self._userAccessed = False

        self._data = None
        self._default = default
        self._domain = domain
        self._description = _strip_indentation(description)
        self._doc = _strip_indentation(doc)
        self._visibility = visibility
        self._argparse = None

    def __getstate__(self):
        # Nominally, __getstate__() should return:
        #
        # state = super(Class, self).__getstate__()
        # for i in Class.__slots__:
        #    state[i] = getattr(self,i)
        # return state
        #
        # Hoewever, in this case, the (nominal) parent class is
        # 'object', and object does not implement __getstate__.  Since
        # super() doesn't actually return a class, we are going to check
        # the *derived class*'s MRO and see if this is the second to
        # last class (the last is always 'object').  If it is, then we
        # can allocate the state dictionary.  If it is not, then we call
        # the super-class's __getstate__ (since that class is NOT
        # 'object').
        if self.__class__.__mro__[-2] is ConfigBase:
            state = {}
        else:
            state = super(ConfigBase, self).__getstate__()
        state.update((key, getattr(self, key)) for key in ConfigBase.__slots__)
        state['_domain'] = _picklable(state['_domain'], self)
        state['_parent'] = None
        return state

    def __setstate__(self, state):
        for key, val in six.iteritems(state):
            # Note: per the Python data model docs, we explicitly
            # set the attribute using object.__setattr__() instead
            # of setting self.__dict__[key] = val.
            object.__setattr__(self, key, val)

    def __call__(self, value=NoArgument, default=NoArgument, domain=NoArgument,
                 description=NoArgument, doc=NoArgument, visibility=NoArgument,
                 implicit=NoArgument, implicit_domain=NoArgument,
                 preserve_implicit=False):
        # We will pass through overriding arguments to the constructor.
        # This way if the constructor does special processing of any of
        # the arguments (like implicit_domain), we don't have to repeat
        # that code here.  Unfortunately, it means we need to do a bit
        # of logic to be sure we only pass through appropriate
        # arguments.
        kwds = {}
        kwds['description'] = ( self._description
                                if description is ConfigBase.NoArgument else
                                description )
        kwds['doc'] = ( self._doc
                        if doc is ConfigBase.NoArgument else
                        doc )
        kwds['visibility'] = ( self._visibility
                               if visibility is ConfigBase.NoArgument else
                               visibility )
        if isinstance(self, ConfigDict):
            kwds['implicit'] = ( self._implicit_declaration
                                 if implicit is ConfigBase.NoArgument else
                                 implicit )
            kwds['implicit_domain'] = (
                self._implicit_domain
                if implicit_domain is ConfigBase.NoArgument else
                implicit_domain )
            if domain is not ConfigBase.NoArgument:
                logger.warn("domain ignored by __call__(): "
                            "class is a ConfigDict" % (type(self),))
            if default is not ConfigBase.NoArgument:
                logger.warn("default ignored by __call__(): "
                            "class is a ConfigDict" % (type(self),))
        else:
            kwds['default'] = ( self.value()
                                if default is ConfigBase.NoArgument else
                                default )
            kwds['domain'] = ( self._domain
                               if domain is ConfigBase.NoArgument else
                               domain )
            if implicit is not ConfigBase.NoArgument:
                logger.warn("implicit ignored by __call__(): "
                            "class %s is not a ConfigDict" % (type(self),))
            if implicit_domain is not ConfigBase.NoArgument:
                logger.warn("implicit_domain ignored by __call__(): "
                            "class %s is not a ConfigDict" % (type(self),))

        # Copy over any other object-specific information (mostly Dict
        # definitions)
        ans = self.__class__(**kwds)
        if isinstance(self, ConfigDict):
            for k in self._decl_order:
                if preserve_implicit or k in self._declared:
                    v = self._data[k]
                    ans._data[k] = _tmp = v(preserve_implicit=preserve_implicit)
                    ans._decl_order.append(k)
                    if k in self._declared:
                        ans._declared.add(k)
                    _tmp._parent = ans
                    _tmp._name = v._name
        else:
            ans.reset()
        # ... and set the value, if appropriate
        if value is not ConfigBase.NoArgument:
            ans.set_value(value)
        return ans

    def name(self, fully_qualified=False):
        # Special case for the top-level dict
        if self._name is None:
            return ""
        elif fully_qualified and self._parent is not None:
            pName = self._parent.name(fully_qualified)
            # Special case for ConfigList indexing and the top-level entries
            if self._name.startswith('[') or not pName:
                return pName + self._name
            else:
                return pName + '.' + self._name
        else:
            return self._name

    def set_default_value(self, default):
        self._default = default

    def set_domain(self, domain):
        self._domain = domain
        self.set_value(self.value(accessValue=False))

    def _cast(self, value):
        if value is None:
            return value
        if self._domain is not None:
            try:
                if value is not ConfigBase.NoArgument:
                    return self._domain(value)
                else:
                    return self._domain()
            except:
                err = sys.exc_info()[1]
                if hasattr(self._domain, '__name__'):
                    _dom = self._domain.__name__
                else:
                    _dom = type(self._domain)
                raise ValueError("invalid value for configuration '%s':\n"
                                 "\tFailed casting %s\n\tto %s\n\tError: %s" %
                                 (self.name(True), value, _dom, err))
        else:
            return value

    def reset(self):
        #
        # This is a dangerous construct, the failure in the first try block
        # can mask a real problem.
        #
        try:
            self.set_value(self._default)
        except:
            if hasattr(self._default, '__call__'):
                self.set_value(self._default())
            else:
                raise
        self._userAccessed = False
        self._userSet = False

    def declare_as_argument(self, *args, **kwds):
        """Map this Config item to an argparse argument.

        Valid arguments include all valid arguments to argparse's
        ArgumentParser.add_argument() with the exception of 'default'.
        In addition, you may provide a group keyword argument can be
        used to either pass in a pre-defined option group or subparser,
        or else pass in the title of a group, subparser, or (subparser,
        group).

        """

        if 'default' in kwds:
            raise TypeError(
                "You cannot specify an argparse default value with "
                "ConfigBase.declare_as_argument().  The default value is "
                "supplied automatically from the Config definition.")

        if 'action' not in kwds and self._domain is bool:
            if not self._default:
                kwds['action'] = 'store_true'
            else:
                kwds['action'] = 'store_false'
                if not args:
                    args = ('--disable-' + _munge_name(self.name()),)
                if 'help' not in kwds:
                    kwds['help'] = "[DON'T] " + self._description
        if 'help' not in kwds:
            kwds['help'] = self._description
        if not args:
            args = ('--' + _munge_name(self.name()),)
        if self._argparse:
            self._argparse = self._argparse + ((args, kwds),)
        else:
            self._argparse = ((args, kwds),)
        return self

    def initialize_argparse(self, parser):

        def _get_subparser_or_group(_parser, name):
            # Note: strings also have a 'title()' method.  We are
            # looking for things that look like argparse
            # groups/subparsers, so just checking for the attribute
            # is insufficient: it needs to be a string attribute as
            # well
            if isinstance(name, argparse._ActionsContainer):
                #hasattr(_group, 'title') and \
                #    isinstance(_group.title, six.string_types):
                return 2, name

            if not isinstance(name, six.string_types):
                raise RuntimeError(
                    'Unknown datatype (%s) for argparse group on '
                    'configuration definition %s' %
                    (type(name).__name__, obj.name(True)))

            try:
                for _grp in _parser._subparsers._group_actions:
                    if name in _grp._name_parser_map:
                        return 1, _grp._name_parser_map[name]
            except AttributeError:
                pass

            for _grp in _parser._action_groups:
                if _grp.title == name:
                    return 0, _grp
            return 0, _parser.add_argument_group(title=name)

        def _process_argparse_def(_args, _kwds):
            _parser = parser
            # shallow copy the dict so we can remove the group flag and
            # add things like documentation, etc.
            _kwds = dict(_kwds)
            if 'group' in _kwds:
                _group = _kwds.pop('group')
                if isinstance(_group, tuple):
                    for _idx, _grp in enumerate(_group):
                        _issub, _parser = _get_subparser_or_group(_parser, _grp)
                        if not _issub and _idx < len(_group) - 1:
                            raise RuntimeError(
                                "Could not find argparse subparser '%s' for "
                                "Config item %s" % (_grp, obj.name(True)))
                else:
                    _issub, _parser = _get_subparser_or_group(_parser, _group)
            if 'dest' not in _kwds:
                _kwds['dest'] = 'CONFIGBLOCK.' + obj.name(True)
                if 'metavar' not in _kwds and \
                   _kwds.get('action','') not in ('store_true','store_false'):
                    if obj._domain is not None and \
                       obj._domain.__class__ is type:
                        _kwds['metavar'] = obj._domain.__name__.upper()
                    else:
                        _kwds['metavar'] = _munge_name(self.name().upper(),
                                                       False)
            _parser.add_argument(*_args, default=argparse.SUPPRESS, **_kwds)

        for level, prefix, value, obj in self._data_collector(None, ""):
            if obj._argparse is None:
                continue
            for _args, _kwds in obj._argparse:
                _process_argparse_def(_args, _kwds)

    def import_argparse(self, parsed_args):
        for level, prefix, value, obj in self._data_collector(None, ""):
            if obj._argparse is None:
                continue
            for _args, _kwds in obj._argparse:
                if 'dest' in _kwds:
                    _dest = _kwds['dest']
                    if _dest in parsed_args:
                        obj.set_value(parsed_args.__dict__[_dest])
                else:
                    _dest = 'CONFIGBLOCK.' + obj.name(True)
                    if _dest in parsed_args:
                        obj.set_value(parsed_args.__dict__[_dest])
                        del parsed_args.__dict__[_dest]
        return parsed_args

    def display(self, content_filter=None, indent_spacing=2, ostream=None,
                visibility=None):
        if content_filter not in ConfigDict.content_filters:
            raise ValueError("unknown content filter '%s'; valid values are %s"
                             % (content_filter, ConfigDict.content_filters))

        _blocks = []
        if ostream is None:
            ostream=sys.stdout

        for lvl, prefix, value, obj in self._data_collector(0, "", visibility):
            if content_filter == 'userdata' and not obj._userSet:
                continue

            _str = _value2string(prefix, value, obj)
            _blocks[lvl:] = [' ' * indent_spacing * lvl + _str + "\n",]

            for i, v in enumerate(_blocks):
                if v is not None:
                    ostream.write(v)
                    _blocks[i] = None

    def generate_yaml_template(self, indent_spacing=2, width=78, visibility=0):
        minDocWidth = 20
        comment = "  # "
        data = list(self._data_collector(0, "", visibility))
        level_info = {}
        for lvl, pre, val, obj in data:
            _str = _value2yaml(pre, val, obj)
            if lvl not in level_info:
                level_info[lvl] = {'data': [], 'off': 0, 'line': 0, 'over': 0}
            level_info[lvl]['data'].append(
                (_str.find(':') + 2, len(_str), len(obj._description or "")))
        for lvl in sorted(level_info):
            indent = lvl * indent_spacing
            _ok = width - indent - len(comment) - minDocWidth
            offset = \
                max( val if val < _ok else key
                     for key,val,doc in level_info[lvl]['data'] )
            offset += indent + len(comment)
            over = sum(1 for key, val, doc in level_info[lvl]['data']
                       if doc + offset > width)
            if len(level_info[lvl]['data']) - over > 0:
                line = max(offset + doc
                           for key, val, doc in level_info[lvl]['data']
                           if offset + doc <= width)
            else:
                line = width
            level_info[lvl]['off'] = offset
            level_info[lvl]['line'] = line
            level_info[lvl]['over'] = over
        maxLvl = 0
        maxDoc = 0
        pad = 0
        for lvl in sorted(level_info):
            _pad = level_info[lvl]['off']
            _doc = level_info[lvl]['line'] - _pad
            if _pad > pad:
                if maxDoc + _pad <= width:
                    pad = _pad
                else:
                    break
            if _doc + pad > width:
                break
            if _doc > maxDoc:
                maxDoc = _doc
            maxLvl = lvl
        os = six.StringIO()
        if self._description:
            os.write(comment.lstrip() + self._description + "\n")
        for lvl, pre, val, obj in data:
            _str = _value2yaml(pre, val, obj)
            if not obj._description:
                os.write(' ' * indent_spacing * lvl + _str + "\n")
                continue
            if lvl <= maxLvl:
                field = pad - len(comment)
            else:
                field = level_info[lvl]['off'] - len(comment)
            os.write(' ' * indent_spacing * lvl)
            if width - len(_str) - minDocWidth >= 0:
                os.write('%%-%ds' % (field - indent_spacing * lvl) % _str)
            else:
                os.write(_str + '\n' + ' ' * field)
            os.write(comment)
            txtArea = max(width - field - len(comment), minDocWidth)
            os.write(("\n" + ' ' * field + comment).join(
                wrap(
                    obj._description, txtArea, subsequent_indent='  ')))
            os.write('\n')
        return os.getvalue()

    def generate_documentation\
            ( self,
              block_start= "\\begin{description}[topsep=0pt,parsep=0.5em,itemsep=-0.4em]\n",
              block_end=   "\\end{description}\n",
              item_start=  "\\item[{%s}]\\hfill\n",
              item_body=   "\\\\%s",
              item_end=    "",
              indent_spacing=2,
              width=78,
              visibility=0
              ):
        os = six.StringIO()
        level = []
        lastObj = self
        indent = ''
        for lvl, pre, val, obj in self._data_collector(1, '', visibility, True):
            #print len(level), lvl, val, obj
            if len(level) < lvl:
                while len(level) < lvl - 1:
                    level.append(None)
                level.append(lastObj)
                if '%s' in block_start:
                    os.write(indent + block_start % lastObj.name())
                elif block_start:
                    os.write(indent + block_start)
                indent += ' ' * indent_spacing
            while len(level) > lvl:
                _last = level.pop()
                if _last is not None:
                    indent = indent[:-1 * indent_spacing]
                    if '%s' in block_end:
                        os.write(indent + block_end % _last.name())
                    elif block_end:
                        os.write(indent + block_end)

            lastObj = obj
            if '%s' in item_start:
                os.write(indent + item_start % obj.name())
            elif item_start:
                os.write(indent + item_start)
            _doc = obj._doc or obj._description or ""
            if _doc:
                _wrapLines = '\n ' not in _doc
                if '%s' in item_body:
                    _doc = item_body % (_doc,)
                elif _doc:
                    _doc = item_body
                if _wrapLines:
                    doc_lines = wrap(
                        _doc,
                        width,
                        initial_indent=indent + ' ' * indent_spacing,
                        subsequent_indent=indent + ' ' * indent_spacing)
                else:
                    doc_lines = (_doc,)
                # Write things out
                os.writelines('\n'.join(doc_lines))
                if not doc_lines[-1].endswith("\n"):
                    os.write('\n')
            if '%s' in item_end:
                os.write(indent + item_end % obj.name())
            elif item_end:
                os.write(indent + item_end)
        while level:
            _last = level.pop()
            if _last is not None:
                indent = indent[:-1 * indent_spacing]
                if '%s' in block_end:
                    os.write(indent + block_end % _last.name())
                else:
                    os.write(indent + block_end)
        return os.getvalue()

    def user_values(self):
        if self._userSet:
            yield self
        for level, prefix, value, obj in self._data_collector(0, ""):
            if obj._userSet:
                yield obj

    def unused_user_values(self):
        if self._userSet and not self._userAccessed:
            yield self
        for level, prefix, value, obj in self._data_collector(0, ""):
            if obj._userSet and not obj._userAccessed:
                yield obj


class ConfigValue(ConfigBase):
    """Store and manipulate a single configuration value.

    Parameters
    ----------
    default: optional
        The default value that this ConfigValue will take if no value is
        provided.

    domain: callable, optional
        The domain can be any callable that accepts a candidate value
        and returns the value converted to the desired type, optionally
        performing any data validation.  The result will be stored into
        the ConfigValue.  Examples include type constructors like `int`
        or `float`.  More complex domain examples include callable
        objects; for example, the :py:class:`In` class that ensures that
        the value falls into an acceptable set or even a complete
        :py:class:`ConfigDict` instance.

    description: str, optional
        The short description of this value

    doc: str, optional
        The long documentation string for this value

    visibility: int, optional
        The visibility of this ConfigValue when generating templates and
        documentation.  Visibility supports specification of "advanced"
        or "developer" options.  ConfigValues with visibility=0 (the
        default) will always be printed / included.  ConfigValues
        with higher visibility values will only be included when the
        generation method specifies a visibility greater than or equal
        to the visibility of this object.

    """

    def __init__(self, *args, **kwds):
        ConfigBase.__init__(self, *args, **kwds)
        self.reset()

    def value(self, accessValue=True):
        if accessValue:
            self._userAccessed = True
        return self._data

    def set_value(self, value):
        self._data = self._cast(value)
        self._userSet = True

    def _data_collector(self, level, prefix, visibility=None, docMode=False):
        if visibility is not None and visibility < self._visibility:
            return
        yield (level, prefix, self, self)


class ImmutableConfigValue(ConfigValue):
    def set_value(self, value):
        if self._cast(value) != self._data:
            raise RuntimeError(str(self) + ' is currently immutable')
        super(ImmutableConfigValue, self).set_value(value)

    def reset(self):
        try:
            super(ImmutableConfigValue, self).set_value(self._default)
        except:
            if hasattr(self._default, '__call__'):
                super(ImmutableConfigValue, self).set_value(self._default())
            else:
                raise
        self._userAccessed = False
        self._userSet = False


class MarkImmutable(object):
    """
    Mark instances of ConfigValue as immutable.

    Parameters
    ----------
    config_value: ConfigValue
        The ConfigValue instances that should be marked immutable.
        Note that multiple instances of ConfigValue can be passed.

    Examples
    --------
    >>> config = ConfigDict()
    >>> config.declare('a', ConfigValue(default=1, domain=int))
    >>> config.declare('b', ConfigValue(default=1, domain=int))
    >>> locker = MarkImmutable(config.get('a'), config.get('b'))

    Now, config.a and config.b cannot be changed. To make them mutable again,

    >>> locker.release_lock()
    """
    def __init__(self, *args):
        self._locked = list()
        try:
            for arg in args:
                if type(arg) is not ConfigValue:
                    raise ValueError('Only ConfigValue instances can be marked immutable.')
                arg.__class__ = ImmutableConfigValue
                self._locked.append(arg)
        except:
            self.release_lock()
            raise

    def release_lock(self):
        for arg in self._locked:
            arg.__class__ = ConfigValue
        self._locked = list()


class ConfigList(ConfigBase):
    """Store and manipulate a list of configuration values.

    Parameters
    ----------
    default: optional
        The default value that this ConfigList will take if no value is
        provided.  If default is a list or ConfigList, then each member
        is cast to the ConfigList's domain to build the default value,
        otherwise the default is cast to the domain and forms a default
        list with a single element.

    domain: callable, optional
        The domain can be any callable that accepts a candidate value
        and returns the value converted to the desired type, optionally
        performing any data validation.  The result will be stored /
        added to the ConfigList.  Examples include type constructors
        like `int` or `float`.  More complex domain examples include
        callable objects; for example, the :py:class:`In` class that
        ensures that the value falls into an acceptable set or even a
        complete :py:class:`ConfigDict` instance.

    description: str, optional
        The short description of this list

    doc: str, optional
        The long documentation string for this list

    visibility: int, optional
        The visibility of this ConfigList when generating templates and
        documentation.  Visibility supports specification of "advanced"
        or "developer" options.  ConfigLists with visibility=0 (the
        default) will always be printed / included.  ConfigLists
        with higher visibility values will only be included when the
        generation method specifies a visibility greater than or equal
        to the visibility of this object.

    """

    def __init__(self, *args, **kwds):
        ConfigBase.__init__(self, *args, **kwds)
        if self._domain is None:
            self._domain = ConfigValue()
        elif isinstance(self._domain, ConfigBase):
            pass
        else:
            self._domain = ConfigValue(None, domain=self._domain)
        self.reset()


    def __setstate__(self, state):
        state = super(ConfigList, self).__setstate__(state)
        for x in self._data:
            x._parent = self

    def __getitem__(self, key):
        val = self._data[key]
        self._userAccessed = True
        if isinstance(val, ConfigValue):
            return val.value()
        else:
            return val

    def get(self, key, default=ConfigBase.NoArgument):
        # Note: get() is borrowed from ConfigDict for cases where we
        # want the raw stored object (and to aviod the implicit
        # conversion of ConfigValue members to their stored data).
        try:
            val = self._data[key]
            self._userAccessed = True
            return val
        except:
            pass
        if default is ConfigBase.NoArgument:
            return None
        if self._domain is not None:
            return self._domain(default)
        else:
            return ConfigValue(default)

    def __setitem__(self, key, val):
        # Note: this will fail if the element doesn't exist in _data.
        # As a result, *this* list doesn't change when someone tries to
        # change an element; instead, the *element* gets its _userSet
        # flag set.
        #self._userSet = True
        self._data[key].set_value(val)

    def __len__(self):
        return self._data.__len__()

    def __iter__(self):
        self._userAccessed = True
        return iter(self[i] for i in xrange(len(self._data)))

    def value(self, accessValue=True):
        if accessValue:
            self._userAccessed = True
        return [config.value(accessValue) for config in self._data]

    def set_value(self, value):
        # If the set_value fails part-way through the list values, we
        # want to restore a deterministic state.  That is, either
        # set_value succeeds completely, or else nothing happens.
        _old = self._data
        self._data = []
        try:
            if (type(value) is list) or \
               isinstance(value, ConfigList):
                for val in value:
                    self.append(val)
            else:
                self.append(value)
        except:
            self._data = _old
            raise
        self._userSet = True

    def reset(self):
        ConfigBase.reset(self)
        # Because the base reset() calls set_value, any deefault list
        # entries will get their userSet flag set.  This is wrong, as
        # reset() should conceptually reset teh object to it's default
        # state (e.g., before the user ever had a chance to mess with
        # things).  As the list could contain a ConfigDict, this is a
        # recursive operation to put the userSet values back.
        for val in self.user_values():
            val._userSet = False

    def append(self, value=ConfigBase.NoArgument):
        val = self._cast(value)
        if val is None:
            return
        self._data.append(val)
        #print self._data[-1], type(self._data[-1])
        self._data[-1]._parent = self
        self._data[-1]._name = '[%s]' % (len(self._data) - 1,)
        self._data[-1]._userSet = True
        self._userSet = True

    @deprecated("ConfigList.add() has been deprecated.  Use append()",
                version='5.7.2')
    def add(self, value=ConfigBase.NoArgument):
        return self.append(value)

    def _data_collector(self, level, prefix, visibility=None, docMode=False):
        if visibility is not None and visibility < self._visibility:
            return
        if docMode:
            # In documentation mode, we do NOT list the documentation
            # for any sub-data, and instead document the *domain*
            # information (as all the entries should share the same
            # domain, potentially duplicating that documentation is
            # somewhat redundant, and worse, if the list is empty, then
            # no documentation is generated at all!)
            yield (level, prefix, None, self)
            subDomain = self._domain._data_collector(level + 1, '- ',
                                                     visibility, docMode)
            # Pop off the (empty) block entry
            six.next(subDomain)
            for v in subDomain:
                yield v
            return
        if prefix:
            if not self._data:
                yield (level, prefix, [], self)
            else:
                yield (level, prefix, None, self)
                if level is not None:
                    level += 1
        for value in self._data:
            for v in value._data_collector(level, '- ', visibility, docMode):
                yield v


class ConfigDict(ConfigBase):
    """Store and manipulate a dictionary of configuration values.

    Parameters
    ----------
    description: str, optional
        The short description of this list

    doc: str, optional
        The long documentation string for this list

    implicit: bool, optional
        If True, the ConfigDict will allow "implicitly" declared
        keys, that is, keys can be stored into the ConfigDict that
        were not prevously declared using :py:meth:`declare` or
        :py:meth:`declare_from`.

    implicit_domain: callable, optional
        The domain that will be used for any implicitly-declared keys.
        Follows the same rules as :py:meth:`ConfigValue`'s `domain`.

    visibility: int, optional
        The visibility of this ConfigDict when generating templates and
        documentation.  Visibility supports specification of "advanced"
        or "developer" options.  ConfigDicts with visibility=0 (the
        default) will always be printed / included.  ConfigDicts
        with higher visibility values will only be included when the
        generation method specifies a visibility greater than or equal
        to the visibility of this object.

    """

    content_filters = (None, 'all', 'userdata')

    __slots__ = ('_decl_order', '_declared', '_implicit_declaration',
                 '_implicit_domain')
    _all_slots = __slots__ + ConfigBase.__slots__

    def __init__(self,
                 description=None,
                 doc=None,
                 implicit=False,
                 implicit_domain=None,
                 visibility=0):
        self._decl_order = []
        self._declared = set()
        self._implicit_declaration = implicit
        if implicit_domain is None or isinstance(implicit_domain, ConfigBase):
            self._implicit_domain = implicit_domain
        else:
            self._implicit_domain = ConfigValue(None, domain=implicit_domain)
        ConfigBase.__init__(self, None, {}, description, doc, visibility)
        self._data = {}

    def __getstate__(self):
        state = super(ConfigDict, self).__getstate__()
        state.update((key, getattr(self, key)) for key in ConfigDict.__slots__)
        state['_implicit_domain'] = _picklable(state['_implicit_domain'], self)
        return state

    def __setstate__(self, state):
        state = super(ConfigDict, self).__setstate__(state)
        for x in six.itervalues(self._data):
            x._parent = self

    def __getitem__(self, key):
        self._userAccessed = True
        key = str(key)
        if isinstance(self._data[key], ConfigValue):
            return self._data[key].value()
        else:
            return self._data[key]

    def get(self, key, default=ConfigBase.NoArgument):
        self._userAccessed = True
        key = str(key)
        if key in self._data:
            return self._data[key]
        if default is ConfigBase.NoArgument:
            return None
        if self._implicit_domain is not None:
            return self._implicit_domain(default)
        else:
            return ConfigValue(default)

    def setdefault(self, key, default=ConfigBase.NoArgument):
        self._userAccessed = True
        key = str(key)
        if key in self._data:
            return self._data[key]
        if default is ConfigBase.NoArgument:
            return self.add(key, None)
        else:
            return self.add(key, default)

    def __setitem__(self, key, val):
        key = str(key)
        if key not in self._data:
            self.add(key, val)
        else:
            self._data[key].set_value(val)
        #self._userAccessed = True

    def __delitem__(self, key):
        # Note that this will produce a KeyError if the key is not valid
        # for this ConfigDict.
        del self._data[key]
        # Clean up the other data structures
        self._decl_order.remove(key)
        self._declared.discard(key)

    def __contains__(self, key):
        key = str(key)
        return key in self._data

    def __len__(self):
        return self._decl_order.__len__()

    def __iter__(self):
        return self._decl_order.__iter__()

    def __getattr__(self, name):
        # Note: __getattr__ is only called after all "usual" attribute
        # lookup methods have failed.  So, if we get here, we already
        # know that key is not a __slot__ or a method, etc...
        #if name in ConfigDict._all_slots:
        #    return super(ConfigDict,self).__getattribute__(name)
        if name not in self._data:
            _name = name.replace('_', ' ')
            if _name not in self._data:
                raise AttributeError("Unknown attribute '%s'" % name)
            name = _name
        return ConfigDict.__getitem__(self, name)

    def __setattr__(self, name, value):
        if name in ConfigDict._all_slots:
            super(ConfigDict, self).__setattr__(name, value)
        else:
            if name not in self._data:
                name = name.replace('_', ' ')
            ConfigDict.__setitem__(self, name, value)

    def iterkeys(self):
        return self._decl_order.__iter__()

    def itervalues(self):
        self._userAccessed = True
        for key in self._decl_order:
            yield self[key]

    def iteritems(self):
        self._userAccessed = True
        for key in self._decl_order:
            yield (key, self[key])

    def keys(self):
        return list(self.iterkeys())

    def values(self):
        return list(self.itervalues())

    def items(self):
        return list(self.iteritems())

    def _add(self, name, config):
        name = str(name)
        if config._parent is not None:
            raise ValueError(
                "config '%s' is already assigned to ConfigDict '%s'; "
                "cannot reassign to '%s'" %
                (name, config._parent.name(True), self.name(True)))
        if name in self._data:
            raise ValueError(
                "duplicate config '%s' defined for ConfigDict '%s'" %
                (name, self.name(True)))
        if '.' in name or '[' in name or ']' in name:
            raise ValueError(
                "Illegal character in config '%s' for ConfigDict '%s': "
                "'.[]' are not allowed." % (name, self.name(True)))
        self._data[name] = config
        self._decl_order.append(name)
        config._parent = self
        config._name = name
        return config

    def declare(self, name, config):
        ans = self._add(name, config)
        self._declared.add(name)
        return ans

    def declare_from(self, other, skip=None):
        if not isinstance(other, ConfigDict):
            raise ValueError(
                "ConfigDict.declare_from() only accepts other ConfigDicts")
        # Note that we duplicate ["other()"] other so that this
        # ConfigDict's entries are independent of the other's
        for key in other.iterkeys():
            if skip and key in skip:
                continue
            if key in self:
                raise ValueError("ConfigDict.declare_from passed a block "
                                 "with a duplicate field, %s" % (key,))
            self.declare(key, other._data[key]())

    def add(self, name, config):
        if not self._implicit_declaration:
            raise ValueError("Key '%s' not defined in ConfigDict '%s'"
                             " and Dict disallows implicit entries" %
                             (name, self.name(True)))

        if self._implicit_domain is None:
            if isinstance(config, ConfigBase):
                ans = self._add(name, config)
            else:
                ans = self._add(name, ConfigValue(config))
        else:
            ans = self._add(name, self._implicit_domain(config))
        self._userSet = True
        return ans

    def value(self, accessValue=True):
        if accessValue:
            self._userAccessed = True
        return dict((name, config.value(accessValue))
                    for name, config in six.iteritems(self._data))

    def set_value(self, value, skip_implicit=False):
        if value is None:
            return self
        if (type(value) is not dict) and \
           (not isinstance(value, ConfigDict)):
            raise ValueError("Expected dict value for %s.set_value, found %s" %
                             (self.name(True), type(value).__name__))
        if not value:
            return self
        _implicit = []
        _decl_map = {}
        for key in value:
            _key = str(key)
            if _key in self._data:
                # str(key) may not be key... store the mapping so that
                # when we later iterate over the _decl_order, we can map
                # the local keys back to the incoming value keys.
                _decl_map[_key] = key
            else:
                _key = _key.replace('_', ' ')
                if _key in self._data:
                    _decl_map[str(_key)] = key
                else:
                    if skip_implicit:
                        pass
                    elif self._implicit_declaration:
                        _implicit.append(key)
                    else:
                        raise ValueError(
                            "key '%s' not defined for ConfigDict '%s' and "
                            "implicit (undefined) keys are not allowed" %
                            (key, self.name(True)))

        # If the set_value fails part-way through the new values, we
        # want to restore a deterministic state.  That is, either
        # set_value succeeds completely, or else nothing happens.
        _old_data = self.value(False)
        try:
            # We want to set the values in declaration order (so that
            # things are deterministic and in case a validation depends
            # on the order)
            for key in self._decl_order:
                if key in _decl_map:
                    #print "Setting", key, " = ", value
                    self[key] = value[_decl_map[key]]
            # implicit data is declared at the end (in sorted order)
            for key in sorted(_implicit):
                self.add(key, value[key])
        except:
            self.reset()
            self.set_value(_old_data)
            raise
        self._userSet = True
        return self

    def reset(self):
        # Reset the values in the order they were declared.  This
        # allows reset functions to have a deterministic ordering.
        def _keep(self, key):
            keep = key in self._declared
            if keep:
                self._data[key].reset()
            else:
                del self._data[key]
            return keep
        # this is an in-place slice of a list...
        self._decl_order[:] = [x for x in self._decl_order if _keep(self, x)]
        self._userAccessed = False
        self._userSet = False

    def _data_collector(self, level, prefix, visibility=None, docMode=False):
        if visibility is not None and visibility < self._visibility:
            return
        if prefix:
            yield (level, prefix, None, self)
            if level is not None:
                level += 1
        for key in self._decl_order:
            for v in self._data[key]._data_collector(level, key + ': ',
                                                     visibility, docMode):
                yield v

# Backwards compatibility: ConfigDict was originally named ConfigBlock.
ConfigBlock = ConfigDict

# In Python3, the items(), etc methods of dict-like things return
# generator-like objects.
if six.PY3:
    ConfigDict.keys = ConfigDict.iterkeys
    ConfigDict.values = ConfigDict.itervalues
    ConfigDict.items = ConfigDict.iteritems
