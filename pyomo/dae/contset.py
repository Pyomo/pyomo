#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['ContinuousSet']

import logging

from pyomo.core import *
from pyomo.core.base.sets import OrderedSimpleSet
from pyomo.core.base.numvalue import native_numeric_types

logger = logging.getLogger('pyomo.core')

class ContinuousSet(OrderedSimpleSet):
    """ ContinuousSet objects are used to represent bounded continuous domains

        Minimally, this set must contain two numeric values defining the
        bounds of a continuous range. Discrete points of interest may
        be added to the continuous set. A continuous set is one
        dimensional and may only contain numerical values.

        Constructor Arguments:
            initialize  Default discretization points to be included

            bounds      Specify bounding points for the continuous domain.
                        The bounds will be included as discrete points in
                        the continuous set but will not be used to restrict
                        points added to the continuous set through the
                        'initialize' argument, a data file, or the add() method

        Private Attributes:
            _changed   This keeps track of whether or not the ContinuousSet was
                       changed during discretization. If the user specifies
                       all of the needed discretization points before the
                       discretization then there is no need to go back through
                       the model and reconstruct things indexed by the ContinuousSet
            _fe        This is a sorted list of the finite element points in the
                       ContinuousSet. i.e. this list contains all the discrete
                       points in the ContinuousSet that are not collocation points.
                       Points that are both finite element points and collocation
                       points will be included in this list.

            _discretization_info
                       This is a dictionary which contains information on the
                       discretization transformation which has been applied to
                       the ContinuousSet.
    """

    def __init__(self, *args, **kwds):
        """ Constructor """
        if kwds.pop("filter", None) is not None:
            raise TypeError("'filter' is not a valid keyword argument for ContinuousSet")
        #if kwds.pop("within", None) is not None:
        #    raise TypeError("'within' is not a valid keyword argument for ContinuousSet")
        if kwds.pop("dimen", None) is not None:
            raise TypeError("'dimen' is not a valid keyword argument for ContinuousSet")
        if kwds.pop("virtual", None) is not None:
            raise TypeError("'virtual' is not a valid keyword argument for ContinuousSet")
        if kwds.pop("validate", None) is not None:
            raise TypeError("'validate' is not a valid keyword argument for ContinuousSet")
        if len(args) != 0:
            raise TypeError("A ContinuousSet expects no arguments")

        kwds.setdefault('ctype',ContinuousSet)
        kwds.setdefault('ordered',Set.SortedOrder)
        self._type=ContinuousSet
        self._changed = False
        self.concrete=True
        self.virtual=False
        self._fe = []
        self._discretization_info = {}
        OrderedSimpleSet.__init__(self, **kwds)

    def get_finite_elements(self):
        return self._fe

    def get_discretization_info(self):
        return self._discretization_info

    def get_changed(self):
        return self._changed

    def set_changed(self,value):
        if value is not True and value is not False:
            raise ValueError("The _changed attribute on a ContinuousSet may only "\
                "be set to True or False")
        self._changed = value

    def get_upper_element_boundary(self, value):
        """
        This function returns the first finite element that is greater
        or equal to the value sent to the function
        """
        if value in self._fe:
            return value
        elif value > max(self._fe):
            print("****WARNING: The value '%s' exceeds the upper bound "\
                  "of the ContinuousSet '%s'. Returning the upper bound" %(str(value),self.name))
            return max(self._fe)
        else :
            for i in self._fe:
                if i>value:     # This works because the list _fe is always sorted
                    return i

    def get_lower_element_boundary(self, value):
        """
        This function returns the first finite element that is less than
        or equal to the value sent to the function
        """
        if value in self._fe:
            if 'scheme' in self._discretization_info:
                if self._discretization_info['scheme'] == 'LAGRANGE-RADAU':
                    # Because Radau Collocation has a collocation point on the
                    # upper finite element bound this if statement ensures that
                    # the desired finite element bound is returned
                    tmp = self._fe.index(value)
                    if tmp != 0:
                        return self._fe[tmp-1]
            return value
        elif value < min(self._fe):
            print("****WARNING: The value '%s' is less than the lower bound "\
                  "of the ContinuousSet '%s'. Returning the lower bound" %(str(value),self.name))
            return min(self._fe)
        else:
            rev_fe = list(self._fe)
            rev_fe.reverse()
            for i in rev_fe:
                if i < value:
                    return i

    def construct(self, values=None):
        OrderedSimpleSet.construct(self, values)

        for val in self.value:
            if type(val) is tuple:
                raise ValueError("ContinuousSet cannot contain tuples")
            if val.__class__ not in native_numeric_types:
                raise ValueError("ContinuousSet can only contain numeric values")

        if self._bounds is None:
            raise ValueError("ContinuousSet '%s' must have at least two values indicating "\
                             "the range over which a differential equation is to be discretized" % (self.name))

        # If bounds were set using pyomo parameters, get their values
        lb = value(self._bounds[0])
        ub = value(self._bounds[1])
        self._bounds = (lb,ub)

        if self._bounds[0].__class__ not in native_numeric_types:
            raise ValueError("Bounds on ContinuousSet must be numeric values")
        if self._bounds[1].__class__ not in native_numeric_types:
            raise ValueError("Bounds on ContinuousSet must be numeric values")

        # TBD: If a user specifies bounds they will be added to the set
        # unless the user specified bounds have been overwritten during
        # OrderedSimpleSet construction. This can lead to some unintuitive
        # behavior when the ContinuousSet is both initialized with values and
        # bounds are specified. The current implementation is consistent
        # with how 'Set' treats this situation.
        if self._bounds[0] not in self.value:
            self.add(self._bounds[0])
            self._sort()
        if self._bounds[1] not in self.value:
            self.add(self._bounds[1])
            self._sort()

        if len(self) < 2:
            raise ValueError("ContinuousSet '%s' must have at least two values indicating "\
                             "the range over which a differential equation is to be discretized" % (self.name))
        self._fe = sorted(self)

register_component(ContinuousSet, "A bounded continuous numerical range optionally containing discrete points of interest.")
