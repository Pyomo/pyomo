#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys


class FlagType(type):
    """Metaclass to simplify the repr(type) and str(type)

    This metaclass redefines the ``str()`` and ``repr()`` of resulting
    classes.  The str() of the class returns only the class' ``__name__``,
    whereas the repr() returns either the qualified class name
    (``__qualname__``) if Sphinx has been imported, or else the
    fully-qualified class name (``__module__ + '.' + __qualname__``).

    This is useful for defining "flag types" that are default arguments
    in functions so that the Sphinx-generated documentation is "cleaner"

    """

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

      >>> def foo(value=NOTSET):
      >>>     if value is NOTSET:
      >>>         pass  # no argument was provided to `value`

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


def building_documentation(ignore_testing_flag=False):
    """Return True if we are building the Sphinx documentation

    Returns
    -------
    bool

    """
    import sys

    return (ignore_testing_flag or not in_testing_environment()) and (
        'sphinx' in sys.modules or 'Sphinx' in sys.modules
    )
