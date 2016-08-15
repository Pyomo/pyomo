#  _________________________________________________________________________
#
#  PyUtilib: A Python utility library.
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  _________________________________________________________________________

__all__ = ("IVariable",
           "variable",
           "variable_list",
           "variable_dict")

import abc

import pyutilib.math

from pyomo.core.base.component_interface import \
    (IComponent,
     _IActiveComponent,
     _IActiveComponentContainer,
     _abstract_readwrite_property,
     _abstract_readonly_property)
from pyomo.core.base.component_dict import ComponentDict
from pyomo.core.base.component_list import ComponentList
from pyomo.core.base.numvalue import NumericValue
from pyomo.core.base.set_types import (RealSet,
                                       IntegerSet)

import six

_infinity = pyutilib.math.infinity
_negative_infinity = -pyutilib.math.infinity

class IVariable(IComponent, NumericValue):
    """
    The interface for optimization variables.
    """
    __slots__ = ()

    #
    # Implementations can choose to define these
    # properties as using __slots__, __dict__, or
    # by overriding the @property method
    #

    domain_type = _abstract_readwrite_property()
    lb = _abstract_readwrite_property()
    ub = _abstract_readwrite_property()
    value = _abstract_readwrite_property()
    fixed = _abstract_readwrite_property()
    stale = _abstract_readwrite_property()

    #
    # Interface
    #

    @property
    def bounds(self):
        """Returns the tuple (lower bound, upper bound)."""
        return (self.lb, self.ub)
    @bounds.setter
    def bounds(self, bounds_tuple):
        self.lb, self.ub = bounds_tuple

    def is_integer(self):
        """Returns True when the domain class is IntegerSet."""
        return issubclass(self.domain_type, IntegerSet)

    def is_binary(self):
        """Returns True when the domain class is BooleanSet."""
        return self.is_integer() and \
            (self.lb >= 0) and \
            (self.ub <= 1)

# TODO?
#    def is_semicontinuous(self):
#        """
#        Returns True when the domain class is SemiContinuous.
#        """
#        return issubclass(self.domain_type, SemiRealSet)

#    def is_semiinteger(self):
#        """
#        Returns True when the domain class is SemiInteger.
#        """
#        return issubclass(self.domain_type, SemiIntegerSet)

    def is_continuous(self):
        """Returns True when the domain is an instance of RealSet."""
        return issubclass(self.domain_type, RealSet)

    #
    # Implement the NumericValue abstract methods
    #

    def is_fixed(self):
        """Returns True if this variable is fixed, otherwise returns False."""
        return self.fixed

    def is_constant(self):
        """Returns False because this is not a constant in an expression."""
        return False

    def _polynomial_degree(self, result):
        """
        If the variable is fixed, it represents a constant
        is a polynomial with degree 0. Otherwise, it has
        degree 1. This method is used in expressions to
        compute polynomial degree.
        """
        if self.fixed:
            return 0
        return 1

    def __nonzero__(self):
        """
        Return True if the value is defined and non-zero.
        """
        if self.value:
            return True
        if self.value is None:
            raise ValueError("Var value is undefined")
        return False

    def __call__(self, exception=True):
        """Compute the value of this variable."""
        return self.value

    __bool__ = __nonzero__

    def fix(self, *val):
        """
        Set the fixed indicator to True. Value argument is optional,
        indicating the variable should be fixed at its current value.
        """
        raise NotImplementedError

    def unfix(self):
        """Sets the fixed indicator to False."""
        raise NotImplementedError

    free=unfix

class variable(IVariable):
    """A decision variable"""
    # To avoid a circular import, for the time being, this
    # property will be set in var.py
    _ctype = None
    __slots__ = ("_parent",
                 "domain_type",
                 "lb",
                 "ub",
                 "value",
                 "fixed",
                 "stale",
                 "__weakref__")
    def __init__(self,
                 domain_type=RealSet,
                 lb=_negative_infinity,
                 ub=_infinity,
                 value=None,
                 fixed=False):
        self._parent = None
        self.domain_type = domain_type
        self.lb = lb
        self.ub = ub
        self.value = value
        self.fixed = fixed
        self.stale = True

class variable_list(ComponentList):
    """A list-style container for variables."""
    # To avoid a circular import, for the time being, this
    # property will be set in var.py
    _ctype = None
    __slots__ = ("_parent",
                 "_data")
    if six.PY3:
        __slots__ = list(__slots__) + ["__weakref__"]
    def __init__(self, *args, **kwds):
        self._parent = None
        super(variable_list, self).__init__(*args, **kwds)

class variable_dict(ComponentDict):
    """A dict-style container for variables."""
    # To avoid a circular import, for the time being, this
    # property will be set in var.py
    _ctype = None
    __slots__ = ("_parent",
                 "_data")
    if six.PY3:
        __slots__ = list(__slots__) + ["__weakref__"]
    def __init__(self, *args, **kwds):
        self._parent = None
        super(variable_dict, self).__init__(*args, **kwds)

if __name__ == "__main__":
    def _fmt(num, suffix='B'):
        for unit in ['','K','M','G','T','P','E','Z']:
            if abs(num) < 1000.0:
                return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1000.0
        return "%.1f %s%s" % (num, 'Yi', suffix)

    import six
    from six.moves import xrange
    import time
    import pickle
    import pympler.asizeof
    from pyomo.core.base.component_block import (block,
                                                 block_list,
                                                 StaticBlock)
    from pyomo.core.base.var import _GeneralVarData, Var
    from pyomo.core.base import Block, Integers, RangeSet
    pickle_protocol = 4

    pympler_kwds = {'detail': 100,
                    'limit': 1000}

    variable_bytes = pympler.asizeof.asizeof(variable(), **pympler_kwds)
    #variable_bytes = len(pickle.dumps(variable(), protocol=pickle_protocol))
    _GeneralVarData_bytes = pympler.asizeof.asizeof(_GeneralVarData(), **pympler_kwds)
    #_GeneralVarData_bytes = len(pickle.dumps(_GeneralVarData(), protocol=pickle_protocol))
    Var_bytes = pympler.asizeof.asizeof(Var(), **pympler_kwds)
    #Var_bytes = len(pickle.dumps(Var(), protocol=pickle_protocol))
    print("variable: %s" % (_fmt(variable_bytes)))
    print("_GeneralVarData: %s (%.2fx)"
          % (_fmt(_GeneralVarData_bytes),
             _GeneralVarData_bytes/float(variable_bytes)))
    print("Var: %s (%.2fx)"
          % (_fmt(Var_bytes),
             Var_bytes/float(variable_bytes)))

    N = 50
    start = time.time()
    vlist = variable_list(
        variable(domain_type=IntegerSet, lb=0, ub=i, value=i*j)
        for i in xrange(N)
        for j in xrange(N))
    stop = time.time()
    variable_list_seconds = stop-start
    variable_list_bytes = pympler.asizeof.asizeof(vlist, **pympler_kwds)
    #variable_list_bytes = len(pickle.dumps(vlist, protocol=pickle_protocol))
    start = time.time()
    vdict = variable_dict(
        ((i,j), variable(domain_type=IntegerSet, lb=0, ub=i, value=i*j))
        for i in xrange(N)
        for j in xrange(N))
    stop = time.time()
    variable_dict_seconds = stop-start
    variable_dict_bytes = pympler.asizeof.asizeof(vdict, **pympler_kwds)
    #variable_dict_bytes = len(pickle.dumps(vdict, protocol=pickle_protocol))
    start = time.time()
    def _bounds_rule(m, i, j):
        return (0, i)
    def _initialize_rule(m, i, j):
        return i
    model = Block(concrete=True)
    model.index = RangeSet(0, N-1)
    model.indexed_Var = Var(model.index, model.index,
                            domain=Integers,
                            bounds=_bounds_rule,
                            initialize=_initialize_rule)
    stop = time.time()
    indexed_Var_seconds = stop-start
    indexed_Var = model.indexed_Var
    model.del_component(indexed_Var)
    indexed_Var._parent = None
    indexed_Var_bytes = pympler.asizeof.asizeof(indexed_Var, **pympler_kwds)
    #indexed_Var_bytes = len(pickle.dumps(indexed_Var, protocol=pickle_protocol))
    print("")
    #print(_fmt(sum(pympler.asizeof.asizeof(v) for v in indexed_Var.values())))
    #print(_fmt(pympler.asizeof.asized(_GeneralVarData(), detail=1, limit=1000, code=False, ignored=False).size))
    #print(_fmt(pympler.asizeof.asized(indexed_Var, detail=100, limit=1000, code=True, ignored=False).size))
    #print(_fmt(len(pickle.dumps(_GeneralVarData(), protocol=pickle_protocol))))
    #print(_fmt(len(pickle.dumps(indexed_Var, protocol=pickle_protocol))))
    #print(_fmt(pympler.asizeof.asizeof(indexed_Var)))
    #print(_fmt(pympler.asizeof.basicsize(indexed_Var)))
    #print(_fmt(pympler.asizeof.flatsize(indexed_Var)))
    #print(_fmt(pympler.asizeof.itemsize(indexed_Var)))
    #print(_fmt(len(pickle.dumps(indexed_Var,protocol=pickle_protocol))))
    print("variable_list{%d}: %s, %s s"
          % (len(vlist), _fmt(variable_list_bytes), variable_list_seconds))
    print("variable_dict{%d}: %s, %s s"
          % (len(vdict), _fmt(variable_dict_bytes), variable_dict_seconds))
    print("Indexed Var{%d}: %s (%.2fx), %s s (%.2fx)"
          % (len(indexed_Var),
             _fmt(indexed_Var_bytes),
             indexed_Var_bytes/float(variable_list_bytes),
             indexed_Var_seconds,
             indexed_Var_seconds/float(variable_list_seconds)))

    class myblock(StaticBlock):
        __slots__ = ("x1",
                     "x2",
                     "x3",
                     "x4",
                     "x5",
                     "x6",
                     "x7",
                     "x8")
        def __init__(self):
            self.x1 = variable()
            self.x2 = variable()
            self.x3 = variable()
            self.x4 = variable()
            self.x5 = variable()
            self.x6 = variable()
            self.x7 = variable()
            self.x8 = variable()
            super(myblock, self).__init__()

    start = time.time()
    blist_static = block_list()
    for i in xrange(N):
        for j in xrange(N):
            blist_static.append(myblock())
    stop = time.time()
    blist_static_seconds = stop-start
    blist_static_bytes = pympler.asizeof.asizeof(blist_static, **pympler_kwds)
    #blist_static_bytes = len(pickle.dumps(blist_static, protocol=pickle_protocol))
    stop = time.time()

    start = time.time()
    blist = block_list()
    for i in xrange(N):
        for j in xrange(N):
            b = block()
            b.x1 = variable()
            b.x2 = variable()
            b.x3 = variable()
            b.x4 = variable()
            b.x5 = variable()
            b.x6 = variable()
            b.x7 = variable()
            b.x8 = variable()
            blist.append(b)
    stop = time.time()
    blist_seconds = stop-start
    blist_bytes = pympler.asizeof.asizeof(blist, **pympler_kwds)
    #blist_bytes = len(pickle.dumps(blist, protocol=pickle_protocol))
    def _indexed_block_rule(b, i, j):
        b.x1 = Var()
        b.x2 = Var()
        b.x3 = Var()
        b.x4 = Var()
        b.x5 = Var()
        b.x6 = Var()
        b.x7 = Var()
        b.x8 = Var()
        return b
    start = time.time()
    indexed_Block = Block(model.index, model.index, rule=_indexed_block_rule)
    indexed_Block.construct()
    stop = time.time()
    indexed_Block_seconds = stop-start
    indexed_Block_bytes = pympler.asizeof.asizeof(indexed_Block, **pympler_kwds)
    #indexed_Block_bytes = len(pickle.dumps(indexed_Block, protocol=pickle_protocol))
    print("")
    print("static_block_list_with_variables{%d}: %s, %s s"
          % (len(blist_static),
             _fmt(blist_static_bytes),
             blist_static_seconds))
    print("block_list_with_variables{%d}: %s, %s s"
          % (len(blist), _fmt(blist_bytes), blist_seconds))
    print("indexed_Block_with_Vars{%d}: %s (%.2fx), %s s (%.2fx)"
          % (len(indexed_Block),
             _fmt(indexed_Block_bytes),
             indexed_Block_bytes/float(blist_static_bytes),
             indexed_Block_seconds,
             indexed_Block_seconds/float(blist_static_seconds)))

