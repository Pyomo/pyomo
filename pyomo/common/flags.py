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
import pyomo
import inspect
import sys


class FlagType(type):
    """Metaclass to help generate "Flag Types".

    This is useful for defining "flag types" that are default arguments
    in functions so that the Sphinx-generated documentation is
    "cleaner".  These types are not constructable (attempts to construct
    the class return the class) and simplify the repr(type) and
    str(type).

    This metaclass redefines the ``str()`` and ``repr()`` of resulting
    classes.  The str() of the class returns only the class' ``__name__``,
    whereas the repr() returns either the qualified class name
    (``__qualname__``) if Sphinx has been imported, or else the
    fully-qualified class name (``__module__ + '.' + __qualname__``).

    """

    def __new__(mcs, name, bases, dct):
        # Ensure that attempts to construct instances of a Flag type
        # return the type.
        def __new_flag__(cls, *args, **kwargs):
            return cls

        dct["__new__"] = __new_flag__
        return type.__new__(mcs, name, bases, dct)

    def __repr__(cls):
        if building_documentation():
            return cls.__qualname__
        else:
            return cls.__module__ + "." + cls.__qualname__

    def __str__(cls):
        return cls.__name__


class NOTSET(object, metaclass=FlagType):
    """
    Class to be used to indicate that an optional argument
    was not specified, if `None` may be ambiguous. Usage:

    Examples
    --------
    >>> def foo(value=NOTSET):
    ...     if value is NOTSET:
    ...         pass  # no argument was provided to `value`

    """

    pass


def in_testing_environment(state=NOTSET):
    """Return True if we are currently running in a "testing" environment

    This currently includes if ``nose``, ``nose2``, or ``pytest`` are
    running (imported).

    Parameters
    ----------
    state : bool or None
        If provided, sets the current state of the testing environment
        (Setting to None reverts to the normal interrogation of
        ``sys.modules``)

    Returns
    -------
    bool

    """
    if state is not NOTSET:
        in_testing_environment.state = state
    if in_testing_environment.state is not None:
        return bool(in_testing_environment.state)
    return any(mod in sys.modules for mod in ('nose', 'nose2', 'pytest'))


in_testing_environment.state = None


def building_documentation(state=NOTSET):
    """True if we are building the Sphinx documentation

    We detect if we are building the documentation by looking if the
    ``sphinx`` or ``Sphinx`` modules are imported.

    Parameters
    ----------
    state : bool or None
        If provided, sets the current state of the building environment
        flag (Setting to None reverts to the normal interrogation of
        ``sys.modules``)

    Returns
    -------
    bool

    """
    if state is not NOTSET:
        building_documentation.state = state
    if building_documentation.state is not None:
        return bool(building_documentation.state)
    # Note that we previously detected if Sphinx was running by looking
    # for it in sys.modules.  That proved to be fragile as other
    # packages (notably sphinx-jinja2-compat) started causing Sphinx to
    # be imported immediately any time Python started.  We work around
    # this by having Pyomo's Sphinx conf.py add a flag to an
    # easy-to-find location.
    return bool(getattr(pyomo, '__sphinx_build__', False))


building_documentation.state = None


def serializing():
    """True if it looks like we are serializing objects

    This looks through the call stack and returns True if it finds a
    `dump` function anywhere in the call stack.  While not foolproof,
    this should reliably catch most serializers, including ``pickle``
    and `yaml``.

    """
    # Start by skipping this function
    frame = inspect.currentframe().f_back
    while frame is not None:
        if frame.f_code.co_name == 'dump':
            return True
        frame = frame.f_back
    return False
