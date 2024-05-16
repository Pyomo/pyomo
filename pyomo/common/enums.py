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

"""This module provides standard :py:class:`enum.Enum` definitions used in
Pyomo, along with additional utilities for working with custom Enums

Utilities:

.. autosummary::

   ExtendedEnumType
   NamedIntEnum

Standard Enums:

.. autosummary::

   ObjectiveSense

"""

import enum
import itertools
import sys

if sys.version_info[:2] < (3, 11):
    _EnumType = enum.EnumMeta
else:
    _EnumType = enum.EnumType


class ExtendedEnumType(_EnumType):
    """Metaclass for creating an :py:class:`enum.Enum` that extends another Enum

    In general, :py:class:`enum.Enum` classes are not extensible: that is,
    they are frozen when defined and cannot be the base class of another
    Enum.  This Metaclass provides a workaround for creating a new Enum
    that extends an existing enum.  Members in the base Enum are all
    present as members on the extended enum.

    Example
    -------

    .. testcode::
       :hide:

       import enum
       from pyomo.common.enums import ExtendedEnumType

    .. testcode::

       class ObjectiveSense(enum.IntEnum):
           minimize = 1
           maximize = -1

       class ProblemSense(enum.IntEnum, metaclass=ExtendedEnumType):
           __base_enum__ = ObjectiveSense

           unknown = 0

    .. doctest::

       >>> list(ProblemSense)
       [<ProblemSense.unknown: 0>, <ObjectiveSense.minimize: 1>, <ObjectiveSense.maximize: -1>]
       >>> ProblemSense.unknown
       <ProblemSense.unknown: 0>
       >>> ProblemSense.maximize
       <ObjectiveSense.maximize: -1>
       >>> ProblemSense(0)
       <ProblemSense.unknown: 0>
       >>> ProblemSense(1)
       <ObjectiveSense.minimize: 1>
       >>> ProblemSense('unknown')
       <ProblemSense.unknown: 0>
       >>> ProblemSense('maximize')
       <ObjectiveSense.maximize: -1>
       >>> hasattr(ProblemSense, 'minimize')
       True
       >>> ProblemSense.minimize is ObjectiveSense.minimize
       True
       >>> ProblemSense.minimize in ProblemSense
       True

    """

    def __getattr__(cls, attr):
        try:
            return getattr(cls.__base_enum__, attr)
        except:
            return super().__getattr__(attr)

    def __iter__(cls):
        # The members of this Enum are the base enum members joined with
        # the local members
        return itertools.chain(super().__iter__(), cls.__base_enum__.__iter__())

    def __contains__(cls, member):
        # This enum "contains" both its local members and the members in
        # the __base_enum__ (necessary for good auto-enum[sphinx] docs)
        return super().__contains__(member) or member in cls.__base_enum__

    def __instancecheck__(cls, instance):
        if cls.__subclasscheck__(type(instance)):
            return True
        # Also pretend that members of the extended enum are subclasses
        # of the __base_enum__.  This is needed to circumvent error
        # checking in enum.__new__ (e.g., for `ProblemSense('minimize')`)
        return cls.__base_enum__.__subclasscheck__(type(instance))

    def _missing_(cls, value):
        # Support attribute lookup by value or name
        for attr in ('value', 'name'):
            for member in cls:
                if getattr(member, attr) == value:
                    return member
        return None

    def __new__(metacls, cls, bases, classdict, **kwds):
        # Support lookup by name - but only if the new Enum doesn't
        # specify its own implementation of _missing_
        if '_missing_' not in classdict:
            classdict['_missing_'] = classmethod(ExtendedEnumType._missing_)
        return super().__new__(metacls, cls, bases, classdict, **kwds)


class NamedIntEnum(enum.IntEnum):
    """An extended version of :py:class:`enum.IntEnum` that supports
    creating members by name as well as value.

    """

    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.name == value:
                return member
        return None


class ObjectiveSense(NamedIntEnum):
    """Flag indicating if an objective is minimizing (1) or maximizing (-1).

    While the numeric values are arbitrary, there are parts of Pyomo
    that rely on this particular choice of value.  These values are also
    consistent with some solvers (notably Gurobi).

    """

    minimize = 1
    maximize = -1

    # Overloading __str__ is needed to match the behavior of the old
    # pyutilib.enum class (removed June 2020). There are spots in the
    # code base that expect the string representation for items in the
    # enum to not include the class name. New uses of enum shouldn't
    # need to do this.
    def __str__(self):
        return self.name


minimize = ObjectiveSense.minimize
maximize = ObjectiveSense.maximize
