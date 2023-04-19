#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from random import random
import sys


def randint(a, b):
    """Our implementation of random.randint.

    The Python random.randint is not consistent between python versions
    and produces a series that is different in 3.x than 2.x.  So that we
    can support deterministic testing (i.e., setting the random.seed and
    expecting the same sequence), we will implement a simple, but stable
    version of randint()."""
    return int((b - a + 1) * random())


def unique_component_name(instance, name):
    # test if this name already exists in model. If not, we're good.
    # Else, we add random numbers until it doesn't
    if instance.component(name) is None and not hasattr(instance, name):
        return name
    name += '_%d' % (randint(0, 9),)
    while True:
        if instance.component(name) is None and not hasattr(instance, name):
            return name
        else:
            name += str(randint(0, 9))


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

    if 'sphinx' in sys.modules or 'Sphinx' in sys.modules:

        def __repr__(cls):
            return cls.__qualname__

    else:

        def __repr__(cls):
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


# Backward compatibility with the previous name for this flag
NoArgumentGiven = NOTSET
