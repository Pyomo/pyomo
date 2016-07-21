#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = (
  '_VirtualSet', '_AnySet', 'RealSet', 'IntegerSet', 'BooleanSet', 'Any',
  'AnyWithNone', 'Reals', 'PositiveReals', 'NonPositiveReals', 'NegativeReals',
  'NonNegativeReals', 'PercentFraction', 'UnitInterval', 'Integers', 'PositiveIntegers',
  'NonPositiveIntegers', 'NegativeIntegers', 'NonNegativeIntegers', 'Boolean',
  'Binary', 'RealInterval', 'IntegerInterval', 'EmptySet'
)

from weakref import ref as weakref_ref

import pyomo.util.plugin
from pyomo.core.base.sets import SimpleSet
from pyomo.core.base.numvalue import (native_numeric_types,
                                      native_integer_types,
                                      native_boolean_types)
from pyomo.core.base.plugin import *

import logging
logger = logging.getLogger('pyomo.core')

_virtual_sets = []


class _VirtualSet(SimpleSet):
    """
    A set that does not contain elements, but instead overrides the
       __contains__ method to define set membership.
    """

    def __init__(self,*args,**kwds):
        self._class_override=False
        SimpleSet.__init__(self, *args, **kwds)
        self.virtual=True
        self.concrete=False

        global _virtual_sets
        _virtual_sets.append(self)

    def data(self):
        raise TypeError("Cannot access data for a virtual set")


class _AnySet(_VirtualSet):
    """A virtual set that allows any value"""

    def __init__(self,*args,**kwds):
        """Constructor"""
        _VirtualSet.__init__(self,*args,**kwds)

    def __contains__(self, element):
        return True


class _EmptySet(_VirtualSet):
    """A virtual set that allows no values"""

    def __init__(self,*args,**kwds):
        """Constructor"""
        _VirtualSet.__init__(self,*args,**kwds)

    def __contains__(self, element):
        return False


class _AnySetWithNone(_AnySet):
    """A virtual set that allows any value (including None)"""

    def __contains__(self, element):
        logger.warning("DEPRECATION WARNING: Use the Any set instead of AnyWithNone")
        return True


class RealSet(_VirtualSet):
    """A virtual set that represents real values"""

    def __init__(self,*args,**kwds):
        """Constructor"""
        if not 'bounds' in kwds:
            kwds['bounds'] = (None,None)
        _VirtualSet.__init__(self,*args,**kwds)

    def __contains__(self, element):
        """Report whether an element is an 'int', 'long' or 'float' value.

        (Called in response to the expression 'element in self'.)
        """
        return _VirtualSet.__contains__(self, element) and \
            ( element.__class__ in native_numeric_types )


class IntegerSet(_VirtualSet):
    """A virtual set that represents integer values"""

    def __init__(self,*args,**kwds):
        """Constructor"""
        if not 'bounds' in kwds:
            kwds['bounds'] = (None,None)
        _VirtualSet.__init__(self,*args,**kwds)

    def __contains__(self, element):
        """Report whether an element is an 'int'.

        (Called in response to the expression 'element in self'.)
        """
        return _VirtualSet.__contains__(self, element) and \
            ( element.__class__ in native_integer_types )


class BooleanSet(_VirtualSet):
    """A virtual set that represents boolean values"""

    def __init__(self,*args,**kwds):
        """Construct the set of booleans, which contains no explicit values"""
        kwds['bounds'] = (0,1)
        _VirtualSet.__init__(self,*args,**kwds)

    def __contains__(self, element):
        """Report whether an element is a boolean.

        (Called in response to the expression 'element in self'.)
        """
        return _VirtualSet.__contains__(self, element) \
               and ( element.__class__ in native_boolean_types ) \
               and ( element in (0, 1, True, False, 'True', 'False', 'T', 'F') )

# GH 2/2016: I'm doing this to make instances of
#            RealInterval and IntegerInterval pickle-able
#            objects. However, these two classes seem like
#            they could be real memory hogs when used as
#            variable domains (for instance via the
#            relax_integrality transformation). Should we
#            consider reimplementing them as more
#            lightweight objects?
class _validate_interval(object):
    __slots__ = ("_obj",)
    def __init__(self, obj): self._obj = weakref_ref(obj)
    def __getstate__(self): return (self._obj(),)
    def __setstate__(self, state): self._obj = weakref_ref(state[0])
    def __call__(self, model, x):
        obj = self._obj()
        if x is not None:
            return (((obj._bounds[0] is None) or \
                     (x >= obj._bounds[0])) and \
                    ((obj._bounds[1] is None) or \
                     (x <= obj._bounds[1])))
        return False

class RealInterval(RealSet):
    """A virtual set that represents an interval of real values"""

    def __init__(self, *args, **kwds):
        """Constructor"""
        if 'bounds' not in kwds:
            kwds['bounds'] = (None,None)
        kwds['validate'] = _validate_interval(self)
        # GH: Assigning a name here so that var.pprint() does not
        #     output _unknown_ in the book examples
        if 'name' not in kwds:
            kwds['name'] = "RealInterval"+str(kwds['bounds'])
        RealSet.__init__(self, *args, **kwds)

class IntegerInterval(IntegerSet):
    """A virtual set that represents an interval of integer values"""

    def __init__(self, *args, **kwds):
        """Constructor"""
        if 'bounds' not in kwds:
            kwds['bounds'] = (None,None)
        kwds['validate'] = _validate_interval(self)
        # GH: Assigning a name here so that var.pprint() does not
        #     output _unknown_ in the book examples
        if 'name' not in kwds:
            kwds['name'] = "IntegerInterval"+str(kwds['bounds'])
        IntegerSet.__init__(self, *args, **kwds)

#
# Concrete instances of the standard sets
#
Any=_AnySet(name="Any", doc="A set of any data")
EmptySet=_EmptySet(name="EmptySet", doc="A set of no data")
AnyWithNone=_AnySetWithNone(name="AnyWithNone", doc="A set of any data (including None)")

Reals=RealSet(name="Reals", doc="A set of real values")
def validate_PositiveValues(model,x):    return x >  0
def validate_NonPositiveValues(model,x): return x <= 0
def validate_NegativeValues(model,x):    return x <  0
def validate_NonNegativeValues(model,x): return x >= 0
def validate_PercentFraction(model,x):   return x >= 0 and x <= 1.0

PositiveReals    = RealSet(
  name="PositiveReals",
  validate=validate_PositiveValues,
  doc="A set of positive real values",
  bounds=(0, None)
)
NonPositiveReals = RealSet(
  name="NonPositiveReals",
  validate=validate_NonPositiveValues,
  doc="A set of non-positive real values",
  bounds=(None, 0)
)
NegativeReals    = RealSet(
  name="NegativeReals",
  validate=validate_NegativeValues,
  doc="A set of negative real values",
  bounds=(None, 0)
)
NonNegativeReals = RealSet(
  name="NonNegativeReals",
  validate=validate_NonNegativeValues,
  doc="A set of non-negative real values",
  bounds=(0, None)
)
PercentFraction = RealSet(
  name="PercentFraction",
  validate=validate_PercentFraction,
  doc="A set of real values in the interval [0,1]",
  bounds=(0.0,1.0)
)
UnitInterval = RealSet(
  name="UnitInterval",
  validate=validate_PercentFraction,
  doc="A set of real values in the interval [0,1]",
  bounds=(0.0,1.0)
)

Integers            = IntegerSet(
  name="Integers",
  doc="A set of integer values"
)
PositiveIntegers    = IntegerSet(
  name="PositiveIntegers",
  validate=validate_PositiveValues,
  doc="A set of positive integer values",
  bounds=(1, None)
)
NonPositiveIntegers = IntegerSet(
  name="NonPositiveIntegers",
  validate=validate_NonPositiveValues,
  doc="A set of non-positive integer values",
  bounds=(None, 0)
)
NegativeIntegers    = IntegerSet(
  name="NegativeIntegers",
  validate=validate_NegativeValues,
  doc="A set of negative integer values",
  bounds=(None, -1)
)
NonNegativeIntegers = IntegerSet(
  name="NonNegativeIntegers",
  validate=validate_NonNegativeValues,
  doc="A set of non-negative integer values",
  bounds=(0, None)
)

Boolean = BooleanSet( name="Boolean", doc="A set of boolean values")
Binary  = BooleanSet( name="Binary",  doc="A set of boolean values")
