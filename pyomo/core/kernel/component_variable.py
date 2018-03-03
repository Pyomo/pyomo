#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.kernel.component_interface import \
    (IComponent,
     _abstract_readwrite_property,
     _abstract_readonly_property)
from pyomo.core.kernel.component_dict import (ComponentDict,
                                              create_component_dict)
from pyomo.core.kernel.component_tuple import (ComponentTuple,
                                               create_component_tuple)
from pyomo.core.kernel.component_list import (ComponentList,
                                              create_component_list)
from pyomo.core.kernel.numvalue import (NumericValue,
                                        value)
from pyomo.core.kernel.set_types import (RealSet,
                                         IntegerSet,
                                         BooleanSet,
                                         RealInterval,
                                         IntegerInterval)

import six
from six.moves import xrange

_pos_inf = float('inf')
_neg_inf = float('-inf')

def _extract_domain_type_and_bounds(domain_type,
                                    domain,
                                    lb, ub):
    if domain is not None:
        if domain_type is not None:
            raise ValueError(
                "At most one of the 'domain' and "
                "'domain_type' keywords can be changed "
                "from their default value when "
                "initializing a variable.")
        domain_type = type(domain)
        # handle some edge cases
        if domain_type is BooleanSet:
            domain_type = IntegerSet
        elif domain_type is RealInterval:
            domain_type = RealSet
        elif domain_type is IntegerInterval:
            domain_type = IntegerSet
        domain_lb, domain_ub = domain.bounds()
        if domain_lb is not None:
            if lb is not None:
                raise ValueError(
                    "The 'lb' keyword can not be used "
                    "to initialize a variable when the "
                    "domain lower bound is finite.")
            lb = domain_lb
        if domain_ub is not None:
            if ub is not None:
                raise ValueError(
                    "The 'ub' keyword can not be used "
                    "to initialize a variable when the "
                    "domain upper bound is finite.")
            ub = domain_ub
    elif domain_type is None:
        domain_type = RealSet
    if domain_type not in IVariable._valid_domain_types:
        raise ValueError(
            "Domain type '%s' is not valid. Must be "
            "one of: %s" % (domain_type,
                            IVariable._valid_domain_types))

    return domain_type, lb, ub


class IVariable(IComponent, NumericValue):
    """The interface for decision variables"""
    __slots__ = ()

    _valid_domain_types = (RealSet, IntegerSet)

    #
    # Implementations can choose to define these
    # properties as using __slots__, __dict__, or
    # by overriding the @property method
    #

    domain_type = _abstract_readwrite_property(
        doc=("The domain type of the variable "
             "(:class:`RealSet` or :class:`IntegerSet`)"))
    lb = _abstract_readwrite_property(
        doc="The lower bound of the variable")
    ub = _abstract_readwrite_property(
        doc="The upper bound of the variable")
    value = _abstract_readwrite_property(
        doc="The value of the variable")
    fixed = _abstract_readwrite_property(
        doc="The fixed status of the variable")
    stale = _abstract_readwrite_property(
        doc="The stale status of the variable")

    #
    # Interface
    #

    @property
    def bounds(self):
        """Get/Set the bounds as a tuple (lb, ub)."""
        return (self.lb, self.ub)
    @bounds.setter
    def bounds(self, bounds_tuple):
        self.lb, self.ub = bounds_tuple

    def fix(self, *val):
        """
        Fix the variable. Sets the fixed indicator to
        :const:`True`. An optional value argument will
        update the variable's value before fixing.
        """
        if len(val) == 1:
            self.value = val[0]
        elif len(val) > 1:
            raise TypeError("fix expected at most 1 arguments, "
                            "got %d" % (len(val)))
        self.fixed = True

    def unfix(self):
        """Free the variable. Sets the fixed indicator to
        :const:`False`."""
        self.fixed = False

    free=unfix

    def has_lb(self):
        """Returns :const:`False` when the lower bound is
        :const:`None` or negative infinity"""
        lb = self.lb
        return (lb is not None) and \
            (value(lb) != _neg_inf)

    def has_ub(self):
        """Returns :const:`False` when the upper bound is
        :const:`None` or positive infinity"""
        ub = self.ub
        return (ub is not None) and \
            (value(ub) != _pos_inf)

    @property
    def lslack(self):
        """Lower slack (value - lb). Returns :const:`None` if
        the variable value is :const:`None`."""
        val = self.value
        if val is None:
            return None
        lb = self.lb
        if lb is None:
            lb = _neg_inf
        else:
            lb = value(lb)
        return val - lb

    @property
    def uslack(self):
        """Upper slack (ub - value). Returns :const:`None` if
        the variable value is :const:`None`."""
        val = self.value
        if val is None:
            return None
        ub = self.ub
        if ub is None:
            ub = _pos_inf
        else:
            ub = value(ub)
        return ub - val

    @property
    def slack(self):
        """min(lslack, uslack). Returns :const:`None` if
        the variable value is :const:`None`."""
        # this method is written so that constraint
        # types that build the body expression on the
        # fly do not have to here
        val = self.value
        if val is None:
            return None
        return min(self.lslack, self.uslack)

    #
    # Convenience methods mainly used by the solver
    # interfaces
    #

    def is_continuous(self):
        """Returns :const:`True` when the domain type is
        :class:`RealSet`."""
        return issubclass(self.domain_type, RealSet)

    # this could be expanded to include semi-continuous
    # where as is_integer would not
    def is_discrete(self):
        """Returns :const:`True` when the domain type is
        :class:`IntegerSet`."""
        return issubclass(self.domain_type, IntegerSet)

    def is_integer(self):
        """Returns :const:`True` when the domain type is
        :class:`IntegerSet`."""
        return issubclass(self.domain_type, IntegerSet)

    def is_binary(self):
        """Returns :const:`True` when the domain type is
        :class:`IntegerSet` and the bounds are within
        [0,1]."""
        lb, ub = self.bounds
        return self.is_integer() and \
            (lb is not None) and \
            (ub is not None) and \
            (value(lb) >= 0) and \
            (value(ub) <= 1)

# TODO?
#    def is_semicontinuous(self):
#        """Returns :const:`True` when the domain class is
#        SemiContinuous."""
#        return issubclass(self.domain_type, SemiRealSet)

#    def is_semiinteger(self):
#        """Returns :const:`True` when the domain class is
#        SemiInteger."""
#        return issubclass(self.domain_type, SemiIntegerSet)

    #
    # Implement the NumericValue abstract methods
    #

    def is_fixed(self):
        """Returns :const:`True` if this variable is fixed,
        otherwise returns :const:`False`."""
        return self.fixed

    def is_constant(self):
        """Returns :const:`False` because this is not a
        constant in an expression."""
        return False

    def _potentially_variable(self):
        """Returns :const:`True` because this is a
        variable."""
        return True

    def polynomial_degree(self):
        """Return the polynomial degree of this
        expression"""
        # If the variable is fixed, it represents a constant;
        # otherwise, it has degree 1.
        if self.fixed:
            return 0
        return 1

    def __call__(self, exception=True):
        """Return the value of this variable."""
        if exception and (self.value is None):
            raise ValueError("value is None")
        return self.value

class variable(IVariable):
    """A decision variable

    Decision variables are used in objectives and
    constraints to define an optimization problem.

    Args:
        domain_type: Sets the domain type of the
            variable. Must be one of :const:`RealSet` or
            :const:`IntegerSet`. Can be updated later by
            assigning to the :attr:`domain_type`
            property. The default value of :const:`None` is
            equivalent to :const:`RealSet`, unless the
            :attr:`domain` keyword is used.
        domain: Sets the domain of the variable. This
            updates the :attr:`domain_type`, :attr:`lb`, and
            :attr:`ub` properties of the variable. The
            default value of :const:`None` implies that this
            keyword is ignored. This keyword can not be used
            in combination with the :attr:`domain_type`
            keyword.
        lb: Sets the lower bound of the variable. Can be
            updated later by assigning to the :attr:`lb`
            property on the variable. Default is
            :const:`None`, which is equivalent to
            :const:`-inf`.
        ub: Sets the upper bound of the variable. Can be
            updated later by assigning to the :attr:`ub`
            property on the variable. Default is
            :const:`None`, which is equivalent to
            :const:`+inf`.
        value: Sets the value of the variable. Can be
            updated later by assigning to the :attr:`value`
            property on the variable. Default is
            :const:`None`.
        fixed (bool): Sets the fixed status of the
            variable. Can be updated later by assigning to
            the :attr:`fixed` property or by calling the
            :meth:`fix` method. Default is :const:`False`.

    Examples:
        >>> # A continuous variable with infinite bounds
        >>> x = pmo.variable()
        >>> # A binary variable
        >>> x = pmo.variable(domain=pmo.Binary)
        >>> # Also a binary variable
        >>> x = pmo.variable(domain_type=pmo.IntegerSet, lb=0, ub=1)
    """
    # To avoid a circular import, for the time being, this
    # property will be set externally
    _ctype = None
    __slots__ = ("_parent",
                 "_domain_type",
                 "_lb",
                 "_ub",
                 "_value",
                 "_fixed",
                 "_stale",
                 "__weakref__")

    def __init__(self,
                 domain_type=None,
                 domain=None,
                 lb=None,
                 ub=None,
                 value=None,
                 fixed=False):
        self._parent = None
        self._domain_type = RealSet
        self._lb = lb
        self._ub = ub
        self._value = value
        self._fixed = fixed
        self._stale = True
        if (domain_type is not None) or \
           (domain is not None):
            self._domain_type, self._lb, self._ub = \
                _extract_domain_type_and_bounds(domain_type,
                                                domain,
                                                lb, ub)

    @property
    def lb(self):
        """The lower bound of the variable"""
        return self._lb
    @lb.setter
    def lb(self, lb):
        self._lb = lb

    @property
    def ub(self):
        """The upper bound of the variable"""
        return self._ub
    @ub.setter
    def ub(self, ub):
        self._ub = ub

    @property
    def value(self):
        """The value of the variable"""
        return self._value
    @value.setter
    def value(self, value):
        self._value = value

    @property
    def fixed(self):
        """The fixed status of the variable"""
        return self._fixed
    @fixed.setter
    def fixed(self, fixed):
        self._fixed = fixed

    @property
    def stale(self):
        """The stale status of the variable"""
        return self._stale
    @stale.setter
    def stale(self, stale):
        self._stale = stale

    @property
    def domain_type(self):
        """The domain type of the variable (:class:`RealSet`
        or :class:`IntegerSet`)"""
        return self._domain_type
    @domain_type.setter
    def domain_type(self, domain_type):
        if domain_type not in IVariable._valid_domain_types:
            raise ValueError(
                "Domain type '%s' is not valid. Must be "
                "one of: %s" % (self.domain_type,
                                IVariable._valid_domain_types))
        self._domain_type = domain_type

    def _set_domain(self, domain):
        """Set the domain of the variable. This method
        updates the :attr:`domain_type` property and
        overwrites the :attr:`lb` and :attr:`ub` properties
        with the domain bounds."""
        self.domain_type, self.lb, self.ub = \
            _extract_domain_type_and_bounds(None,
                                            domain,
                                            None, None)
    domain = property(fset=_set_domain,
                      doc=_set_domain.__doc__)

class variable_tuple(ComponentTuple):
    """A tuple-style container for variables."""
    # To avoid a circular import, for the time being, this
    # property will be set externally
    _ctype = None
    __slots__ = ("_parent",
                 "_data")
    if six.PY3:
        # This has to do with a bug in the abc module
        # prior to python3. They forgot to define the base
        # class using empty __slots__, so we shouldn't add a slot
        # for __weakref__ because the base class has a __dict__.
        __slots__ = list(__slots__) + ["__weakref__"]

    def __init__(self, *args, **kwds):
        self._parent = None
        super(variable_tuple, self).__init__(*args, **kwds)

def create_variable_tuple(size, *args, **kwds):
    """
    Generates a full :class:`variable_tuple`.

    Args:
        size (int): The number of objects to place in the
            variable_tuple.
        type_: The object type to populate the container
            with. Must have the same ctype as
            variable_tuple. Default: :class:`variable`
        *args: arguments used to construct the objects
            placed in the container.
        **kwds: keywords used to construct the objects
            placed in the container.

    Returns:
        a fully populated :class:'variable_tuple`
    """
    type_ = kwds.pop('type_', variable)
    return create_component_tuple(variable_tuple,
                                  type_,
                                  size,
                                  *args,
                                  **kwds)

class variable_list(ComponentList):
    """A list-style container for variables."""
    # To avoid a circular import, for the time being, this
    # property will be set externally
    _ctype = None
    __slots__ = ("_parent",
                 "_data")
    if six.PY3:
        # This has to do with a bug in the abc module
        # prior to python3. They forgot to define the base
        # class using empty __slots__, so we shouldn't add a slot
        # for __weakref__ because the base class has a __dict__.
        __slots__ = list(__slots__) + ["__weakref__"]

    def __init__(self, *args, **kwds):
        self._parent = None
        super(variable_list, self).__init__(*args, **kwds)

def create_variable_list(size, *args, **kwds):
    """
    Generates a full :class:`variable_list`.

    Args:
        size (int): The number of objects to place in the
            variable_list.
        type_: The object type to populate the container
            with. Must have the same ctype as
            variable_list. Default: :class:`variable`
        *args: arguments used to construct the objects
            placed in the container.
        **kwds: keywords used to construct the objects
            placed in the container.

    Returns:
        a fully populated :class:`variable_list`
    """
    type_ = kwds.pop('type_', variable)
    return create_component_list(variable_list,
                                 type_,
                                 size,
                                 *args,
                                 **kwds)

class variable_dict(ComponentDict):
    """A dict-style container for variables."""
    # To avoid a circular import, for the time being, this
    # property will be set externally
    _ctype = None
    __slots__ = ("_parent",
                 "_data")
    if six.PY3:
        # This has to do with a bug in the abc module
        # prior to python3. They forgot to define the base
        # class using empty __slots__, so we shouldn't add a slot
        # for __weakref__ because the base class has a __dict__.
        __slots__ = list(__slots__) + ["__weakref__"]

    def __init__(self, *args, **kwds):
        self._parent = None
        super(variable_dict, self).__init__(*args, **kwds)

def create_variable_dict(keys, *args, **kwds):
    """
    Generates a full :class:`variable_dict`.

    Args:
        keys: The set of keys to used to populate the
            variable_dict.
        type_: The object type to populate the container
            with. Must have the same ctype as
            variable_dict. Default: :class:`variable`
        *args: arguments used to construct the objects
            placed in the container.
        **kwds: keywords used to construct the objects
            placed in the container.

    Returns:
        a fully populated :class:`variable_dict`
    """
    type_ = kwds.pop('type_', variable)
    return create_component_dict(variable_dict,
                                 type_,
                                 keys,
                                 *args,
                                 **kwds)
