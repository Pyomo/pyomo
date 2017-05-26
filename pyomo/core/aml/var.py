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

from pyomo.core.base import value
from pyomo.core.base.numvalue import NumericValue
from pyomo.core.base.set_types import Reals
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.util import is_functor

from pyomo.core.base.component_interface import IComponent
from pyomo.core.base.component_variable import (variable,
                                                variable_dict)

from pyomo.core.aml.constructed_object import \
    IConstructedObject
from pyomo.core.aml.indexed_component_container import \
    _IndexedComponentContainerMixin

logger = logging.getLogger('pyomo.core')

class Var(IConstructedObject):
    __slots__ = ()

    def __new__(cls, *args, **kwds):
        if cls != Var:
            return super(Var, cls).__new__(cls)
        if args == () or ((type(args[0]) == set) and \
                          (len(args) == 1)):
            return SimpleVar.__new__(SimpleVar)
        else:
            return IndexedVar.__new__(IndexedVar)

    def __init__(self, **kwd):
        #
        # Default keyword values
        #
        initialize = kwd.pop('initialize', None)
        initialize = kwd.pop('rule', initialize)
        domain = kwd.pop('within', Reals)
        domain = kwd.pop('domain', domain)
        bounds = kwd.pop('bounds', None)
        assert len(kwd) == 0
        self._constructed = False

        #
        # Determine if the domain argument is a functor or
        # other object
        #
        self._domain_init_value = None
        self._domain_init_rule = None
        if is_functor(domain):
            self._domain_init_rule = domain
        else:
            self._domain_init_value = domain

        #
        # Allow for functions or functors for value
        # initialization, without confusing with Params, etc
        # (which have a __call__ method).
        #
        self._value_init_value = None
        self._value_init_rule = None
        if  is_functor(initialize) and \
            (not isinstance(initialize,NumericValue)):
            self._value_init_rule = initialize
        else:
            self._value_init_value = initialize
        #
        # Determine if the bound argument is a functor or
        # other object
        #
        self._bounds_init_rule = None
        self._bounds_init_value = None
        if is_functor(bounds):
            self._bounds_init_rule = bounds
        elif type(bounds) is tuple:
            self._bounds_init_value = bounds
        elif bounds is not None:
            raise ValueError("Variable 'bounds' keyword must "
                             "be a tuple or function")

    def construct(self, data=None):
        """Construct this component."""

        if __debug__ and logger.isEnabledFor(logging.DEBUG):   #pragma:nocover
            try:
                name = str(self.name)
            except:
                # Some Var components don't have a name yet,
                # so just use the type
                name = type(self)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Constructing Variable, name=%s, from data=%s"
                    % (name, str(data)))

        if self._constructed:
            return

        self._construct_impl()
        self._constructed = True

    def _construct_impl(self):
        raise NotImplementedError      #pragma:nocover

class SimpleVar(variable, Var):
    """A single variable."""
    __slots__ = ("_constructed",
                 "_domain_init_value",
                 "_domain_init_rule",
                 "_value_init_rule",
                 "_value_init_value",
                 "_bounds_init_rule",
                 "_bounds_init_value")
    def __init__(self, *args, **kwds):
        assert len(args) == 0
        variable.__init__(self)
        Var.__init__(self, **kwds)

    def _construct_impl(self):
        # Initialize domain
        if self._domain_init_rule is not None:
            self.domain = self._domain_init_rule(self._parent())
        else:
            self.domain = self._domain_init_value

        # Initialize values
        if self._value_init_rule is not None:
            val = self._value_init_rule(self._parent())
            self.value = value(val)
        elif self._value_init_value is not None:
            self.value = value(self._value_init_value)

        # Initialize bounds
        if self._bounds_init_rule is not None:
            self.lb, self.ub = self._bounds_init_rule(self._parent())
        elif self._bounds_init_value is not None:
            self.lb, self.ub = self._bounds_init_value

class IndexedVar(_IndexedComponentContainerMixin,
                 variable_dict,
                 Var):
    """An dictionary of variables."""

    def __init__(self, *args, **kwds):
        _IndexedComponentContainerMixin.__init__(self, *args)
        variable_dict.__init__(self)
        Var.__init__(self, **kwds)

    def _initialize_members(self, init_set):
        # Initialize domains
        if self._domain_init_rule is not None:
            for ndx in init_set:
                # because apply_indxed_rule will not pass it
                # to the function
                assert ndx is not None
                self[ndx].domain = \
                    apply_indexed_rule(self,
                                       self._domain_init_rule,
                                       self._parent(),
                                       ndx)
        else:
            # Optimization: It is assumed self._domain_init_value
            #               is used when the _GeneralVarData objects
            #               are created. This avoids an unnecessary
            #               loop over init_set, which can significantly
            #               speed up construction of variables with large
            #               index sets.
            pass

        # Initialize values
        if self._value_init_rule is not None:
            for key in init_set:
                vardata = self[key]
                # because apply_indxed_rule will not pass it
                # to the function
                assert key is not None
                val = apply_indexed_rule(self,
                                         self._value_init_rule,
                                         self._parent(),
                                         key)
                vardata.value = value(val)
        elif self._value_init_value is not None:
            if self._value_init_value.__class__ is dict:
                for key in init_set:
                    # Skip indices that are not in the
                    # dictionary. This arises when
                    # initializing VarList objects with a
                    # dictionary.
                    if not key in self._value_init_value:
                        continue
                    val = self._value_init_value[key]
                    self[key].value = val
            else:
                val = value(self._value_init_value)
                for key in init_set:
                    self[key].value = val

        # Initialize bounds
        if self._bounds_init_rule is not None:
            for key in init_set:
                vardata = self[key]
                # because apply_indxed_rule will not pass it
                # to the function
                assert key is not None
                (vardata.lb, vardata.ub) = \
                    apply_indexed_rule(self,
                                       self._bounds_init_rule,
                                       self._parent(),
                                       key)
        elif self._bounds_init_value is not None:
            (lb, ub) = self._bounds_init_value
            for key in init_set:
                vardata = self[key]
                vardata.lb = lb
                vardata.ub = ub

    def _construct_impl(self):
        # This loop is optimized for speed with pypy.
        # Calling dict.update((...) for ...) is roughly
        # 30% slower
        for ndx in self.index:
            self[ndx] = variable(domain=self._domain_init_value)
        self._initialize_members(self.index)

    # These methods are normally found on the NumericValue
    # interface, but they are here to trick the expression
    # system into reporting errors about trying to use
    # "indexed NumericValue" objects in expressions
    def as_numeric(self): return self
    def is_expression(self): return False
    def is_relational(self): return False
