#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
from weakref import ref as weakref_ref

from pyomo.core.expr.numvalue import (native_numeric_types,
                                        native_integer_types,
                                        native_boolean_types)


logger = logging.getLogger('pyomo.core')

_virtual_sets = []

class _VirtualSet(object):
    """
    A set that does not contain elements, but instead overrides the
       __contains__ method to define set membership.
    """

    def __init__(self, name=None, doc=None, bounds=None, validate=None):
        self.name = name
        self.doc = doc
        self._bounds = bounds
        if self._bounds is None:
            self._bounds = (None, None)
        self.validate = validate

        global _virtual_sets
        _virtual_sets.append(self)

    def __lt__(self, other):
        raise TypeError("'<' not supported")

    def __le__(self, other):
        raise TypeError("<=' not supported")

    def __gt__(self, other):
        raise TypeError("'>' not supported")

    def __ge__(self, other):
        raise TypeError("'>=' not supported")

    def __str__(self):
        if self.name is None:
            return super(_VirtualSet, self).__str__()
        else:
            return str(self.name)

    def bounds(self):
        return self._bounds

    def __contains__(self, other):
        valid = True
        if self.validate is not None:
            valid = self.validate(other)
        if valid:
            if (self._bounds is not None):
                if self._bounds[0] is not None:
                    valid &= (other >= self._bounds[0])
                if self._bounds[1] is not None:
                    valid &= (other <= self._bounds[1])
        return valid

class RealSet(_VirtualSet):
    """A virtual set that represents real values"""

    def __init__(self, *args, **kwds):
        """Constructor"""
        _VirtualSet.__init__(self, *args, **kwds)

    def __contains__(self, element):
        """Report whether an element is an 'int', 'long' or 'float' value.

        (Called in response to the expression 'element in self'.)
        """
        return element.__class__ in native_numeric_types and \
            _VirtualSet.__contains__(self, element)

class IntegerSet(_VirtualSet):
    """A virtual set that represents integer values"""

    def __init__(self, *args, **kwds):
        """Constructor"""
        _VirtualSet.__init__(self, *args, **kwds)

    def __contains__(self, element):
        """Report whether an element is an 'int'.

        (Called in response to the expression 'element in self'.)
        """
        return element.__class__ in native_integer_types and \
            _VirtualSet.__contains__(self, element)

class BooleanSet(_VirtualSet):
    """A virtual set that represents boolean values"""

    def __init__(self, *args, **kwds):
        """Construct the set of booleans, which contains no explicit values"""
        assert 'bounds' not in kwds
        kwds['bounds'] = (0,1)
        _VirtualSet.__init__(self, *args, **kwds)

    def __contains__(self, element):
        """Report whether an element is a boolean.

        (Called in response to the expression 'element in self'.)
        """
        return ((element.__class__ in native_boolean_types) or \
                (element.__class__ in native_numeric_types)) and \
            (element in (0, 1, True, False)) and \
            _VirtualSet.__contains__(self, element)
               # where does it end? (i.e., why not 'true', 'TRUE, etc.?)
               #and ( element in (0, 1, True, False, 'True', 'False', 'T', 'F') )

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
    def __call__(self, x):
        assert x is not None
        obj = self._obj()
        return (((obj._bounds[0] is None) or \
                 (x >= obj._bounds[0])) and \
                ((obj._bounds[1] is None) or \
                 (x <= obj._bounds[1])))

class RealInterval(RealSet):
    """A virtual set that represents an interval of real values"""

    def __init__(self, name=None, **kwds):
        """Constructor"""
        if 'bounds' not in kwds:
            kwds['bounds'] = (None,None)
        kwds['validate'] = _validate_interval(self)
        # GH: Assigning a name here so that var.pprint() does not
        #     output _unknown_ in the book examples
        if name is None:
            kwds['name'] = "RealInterval"+str(kwds['bounds'])
        else:
            kwds['name'] = name
        RealSet.__init__(self, **kwds)

class IntegerInterval(IntegerSet):
    """A virtual set that represents an interval of integer values"""

    def __init__(self, name=None, **kwds):
        """Constructor"""
        if 'bounds' not in kwds:
            kwds['bounds'] = (None,None)
        kwds['validate'] = _validate_interval(self)
        # GH: Assigning a name here so that var.pprint() does not
        #     output _unknown_ in the book examples
        if name is None:
            kwds['name'] = "IntegerInterval"+str(kwds['bounds'])
        else:
            kwds['name'] = name
        IntegerSet.__init__(self, **kwds)

Reals=RealSet(name="Reals", doc="A set of real values")
def validate_PositiveValues(x):    return x >  0
def validate_NonPositiveValues(x): return x <= 0
def validate_NegativeValues(x):    return x <  0
def validate_NonNegativeValues(x): return x >= 0
def validate_PercentFraction(x):   return x >= 0 and x <= 1.0

PositiveReals    = RealSet(
  name="PositiveReals",
  doc="A set of positive real values",
  validate=validate_PositiveValues,
  bounds=(0, None)
)
NonPositiveReals = RealSet(
  name="NonPositiveReals",
  doc="A set of non-positive real values",
  validate=validate_NonPositiveValues,
  bounds=(None, 0)
)
NegativeReals    = RealSet(
  name="NegativeReals",
  doc="A set of negative real values",
  validate=validate_NegativeValues,
  bounds=(None, 0)
)
NonNegativeReals = RealSet(
  name="NonNegativeReals",
  doc="A set of non-negative real values",
  validate=validate_NonNegativeValues,
  bounds=(0, None)
)
PercentFraction = RealSet(
  name="PercentFraction",
  doc="A set of real values in the interval [0,1]",
  validate=validate_PercentFraction,
  bounds=(0.0,1.0)
)
UnitInterval = RealSet(
  name="UnitInterval",
  doc="A set of real values in the interval [0,1]",
  validate=validate_PercentFraction,
  bounds=(0.0,1.0)
)

Integers            = IntegerSet(
  name="Integers",
  doc="A set of integer values"
)
PositiveIntegers    = IntegerSet(
  name="PositiveIntegers",
  doc="A set of positive integer values",
  validate=validate_PositiveValues,
  bounds=(1, None)
)
NonPositiveIntegers = IntegerSet(
  name="NonPositiveIntegers",
  doc="A set of non-positive integer values",
  validate=validate_NonPositiveValues,
  bounds=(None, 0)
)
NegativeIntegers    = IntegerSet(
  name="NegativeIntegers",
  doc="A set of negative integer values",
  validate=validate_NegativeValues,
  bounds=(None, -1)
)
NonNegativeIntegers = IntegerSet(
  name="NonNegativeIntegers",
  doc="A set of non-negative integer values",
  validate=validate_NonNegativeValues,
  bounds=(0, None)
)

Boolean = BooleanSet(name="Boolean", doc="A set of boolean values")
Binary  = BooleanSet(name="Binary",  doc="A set of boolean values")
