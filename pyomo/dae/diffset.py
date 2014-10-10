#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

__all__ = ['DifferentialSet']

import logging

from pyomo.core import *
from pyomo.core.base.sets import OrderedSimpleSet
from pyomo.core.base.numvalue import native_numeric_types

logger = logging.getLogger('pyomo.core')


class DifferentialSet(OrderedSimpleSet):
    """ DifferentialSet objects are used to index DAE Pyomo objects
    
        This is a set that represents a list of numeric values. Minimally, 
        this set must contain two values defining a range over which a differential
        equations will be discretized and solved. If additional values are specified
        they will be used as finite elements locations in the discretization. 

        Constructor
            Arguments:
                name        The name of the differential set
                rule
                initialize  Default set members, which may be overridden
                            when setting up this set
                doc         Documentation for this object
        Private Attributes:
            _changed   This keeps track of whether or not the diffset was
                       changed during discretization. If the user specifies 
                       all of the needed discretization points before the 
                       discretization then there is no need to go back through
                       the model and reconstruct things indexed by the diffset
    """
    def __init__(self, *args, **kwds):
        """ Constructor """
        if kwds.pop("filter", None) is not None:
            raise TypeError("'filter' is not a valid keyword argument for DifferentialSet")
        if kwds.pop("within", None) is not None:
            raise TypeError("'within' is not a valid keyword argument for DifferentialSet")
        if kwds.pop("dimen", None) is not None:
            raise TypeError("'dimen' is not a valid keyword argument for DifferentialSet")
        if kwds.pop("virtual", None) is not None:
            raise TypeError("'virtual' is not a valid keyword argument for DifferentialSet")
        if kwds.pop("validate", None) is not None:
            raise TypeError("'validate' is not a valid keyword argument for DifferentialSet")
        if len(args) != 0:
            raise TypeError("A DifferentialSet expects no arguments")
        
        kwds.setdefault('ctype',DifferentialSet)
        kwds.setdefault('ordered',Set.SortedOrder)
        self._type=DifferentialSet
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
            raise ValueError("The _changed attribute on a DifferentialSet may only "\
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
                "of the diffset '%s'. Returning the upper bound" %(str(value),self.name))
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
            if self._discretization_info.has_key('scheme'):
                if self._discretization_info['scheme'] == 'Radau Collocation':
                    # Because Radau Collocation has a collocation point on the 
                    # upper finite element bound this if statement ensures that
                    # the desired finite element bound is returned
                    tmp = self._fe.index(value)
                    if tmp != 0:
                        return self._fe[tmp-1]
            return value 
        elif value < min(self._fe):
            print("****WARNING: The value '%s' is less than the lower bound "\
                "of the diffset '%s'. Returning the lower bound" %(str(value),self.name))
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
                raise ValueError("DifferentialSet cannot contain tuples")
            if val.__class__ not in native_numeric_types:
                raise ValueError("DifferentialSet can only contain numeric values")

        if self._bounds is None:
            # This is needed because if a set is initialized using a function
            # the bounds are not updated automatically
            #if self.dimen == 1 and len(self>2):
            #    self._bounds[0] == min(self)
            #    self._bounds[1] == max(self)
            raise ValueError("DifferentialSet '%s' must have at least two values indicating "\
                "the range over which a differential equation is to be discretized" % (self.name))
        if self._bounds[0].__class__ not in native_numeric_types:
            raise ValueError("Bounds on DifferentialSet must be numeric values")
        if self._bounds[1].__class__ not in native_numeric_types:
            raise ValueError("Bounds on DifferentialSet must be numeric values")

        # TBD: If a user specifies bounds they will be added to the set
        # unless the user specified bounds have been overwritten during 
        # OrderedSimpleSet construction. This can lead to some unintuitive
        # behavior when the diffset is both initialized with values and
        # bounds are specified. The current implementation is consistent 
        # with how 'Set' treats this situation. 
        if self._bounds[0] not in self.value:
            self.add(self._bounds[0])
        if self._bounds[1] not in self.value:
            self.add(self._bounds[1])
        
        if len(self) < 2:
            raise ValueError("DifferentialSet '%s' must have at least two values indicating "\
                "the range over which a differential equation is to be discretized" % (self.name))
        self._fe = sorted(self)


register_component(DifferentialSet, "A sequence of numeric values defining the range over which a differential equation is to be discretized and solved.")
