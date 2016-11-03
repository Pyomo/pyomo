#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['Var', '_VarData', 'VarList']

import logging
from weakref import ref as weakref_ref

from pyomo.core.base.numvalue import NumericValue, value, is_fixed
from pyomo.core.base.set_types import BooleanSet, IntegerSet, RealSet, Reals
from pyomo.core.base.component import ComponentData, register_component
from pyomo.core.base.indexed_component import IndexedComponent, UnindexedComponent_set, normalize_index
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.sets import Set
from pyomo.core.base.util import is_functor

from six import iteritems, itervalues

logger = logging.getLogger('pyomo.core')

class _VarData(ComponentData, NumericValue):
    """
    This class defines the data for a single variable.

    Constructor Arguments:
        component   The Var object that owns this data.

    Public Class Attributes:
        domain      The domain of this variable.
        bounds      A tuple (lower,upper) that defines the variable bounds.
        fixed       If True, then this variable is treated as a
                        fixed constant in the model.
        lb          A lower bound for this variable.  The lower bound can be
                        either numeric constants, parameter values, expressions
                        or any object that can be called with no arguments.
        ub          A upper bound for this variable.  The upper bound can be either
                        numeric constants, parameter values, expressions or any
                        object that can be called with no arguments.
        stale       A Boolean indicating whether the value of this variable is
                        legitimiate.  This value is true if the value should
                        be considered legitimate for purposes of reporting or
                        other interrogation.
        value       The numeric value of this variable.

    The domain, lb, and ub attributes are properties because they
    are too widely accessed directly to enforce explicit getter/setter
    methods and we need to deter directly modifying or accessing
    these attributes in certain cases.
    """

    __slots__ = ()

    def __init__(self, component=None):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - ComponentData
        #   - NumericValue
        self._component = weakref_ref(component) if (component is not None) \
                          else None

    #
    # Interface
    #

    @property
    def bounds(self):
        """Returns the tuple (lower bound, upper bound)."""
        return (self.lb, self.ub)
    @bounds.setter
    def bounds(self, val):
        raise AttributeError("Assignment not allowed. Use the setub and setlb methods")

    def is_integer(self):
        """Returns True when the domain class is IntegerSet."""
        # optimization: this is the most common case
        if self.domain.__class__ is IntegerSet:
            return True
        return isinstance(self.domain, IntegerSet)

    def is_binary(self):
        """Returns True when the domain class is BooleanSet."""
        # optimization: this is the most common case
        if self.domain.__class__ is BooleanSet:
            return True
        return isinstance(self.domain, BooleanSet)

# TODO?
#    def is_semicontinuous(self):
#        """
#        Returns True when the domain class is SemiContinuous.
#        """
#        return self.domain.__class__ is SemiContinuousSet
#    def is_semiinteger(self):
#        """
#        Returns True when the domain class is SemiInteger.
#        """
#        return self.domain.__class__ is SemiIntegerSet

    def is_continuous(self):
        """Returns True when the domain is an instance of RealSet."""
        # optimization: this is the most common case
        if self.domain.__class__ is RealSet:
            return True
        return isinstance(self.domain, RealSet)

    def is_fixed(self):
        """Returns True if this variable is fixed, otherwise returns False."""
        return self.fixed

    def is_constant(self):
        """Returns False because this is not a constant in an expression."""
        return False

    def _potentially_variable(self):
        """Returns True because this is a variable."""
        return True

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

    def set_value(self, val, valid=False):
        """
        Set the value of this numeric object, after
        validating its value. If the 'valid' flag is True,
        then the validation step is skipped.
        """
        if valid or self._valid_value(val):
            self.value = val
            self.stale = False

    def _valid_value(self, val, use_exception=True):
        """
        Validate the value.  If use_exception is True, then raise an
        exception.
        """
        ans = val is None or val in self.domain
        if not ans and use_exception:
            raise ValueError("Numeric value `%s` (%s) is not in "
                             "domain %s" % (val, type(val), self.domain))
        return ans

    def clear(self):
        self.value = None

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


    #
    # Abstract Interface
    #

    @property
    def value(self):
        """Return the value for this variable."""
        raise NotImplementedError

    @property
    def domain(self):
        """Return the domain for this variable."""
        raise NotImplementedError

    @property
    def lb(self):
        """Return the lower bound for this variable."""
        raise NotImplementedError

    @property
    def ub(self):
        """Return the upper bound for this variable."""
        raise NotImplementedError

    @property
    def fixed(self):
        """Return the fixed indicator for this variable."""
        raise NotImplementedError

    @property
    def stale(self):
        """Return the stale indicator for this variable."""
        raise NotImplementedError

    def setlb(self, val):
        """
        Set the lower bound for this variable after validating that
        the value is fixed (or None).
        """
        raise NotImplementedError

    def setub(self, val):
        """
        Set the upper bound for this variable after validating that
        the value is fixed (or None).
        """
        raise NotImplementedError

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

class _GeneralVarData(_VarData):
    """
    This class defines the data for a single variable.

    Constructor Arguments:
        component   The Var object that owns this data.

    Public Class Attributes:
        domain      The domain of this variable.
        bounds      A tuple (lower,upper) that defines the variable bounds.
        fixed       If True, then this variable is treated as a
                        fixed constant in the model.
        lb          A lower bound for this variable.  The lower bound can be
                        either numeric constants, parameter values, expressions
                        or any object that can be called with no arguments.
        ub          A upper bound for this variable.  The upper bound can be either
                        numeric constants, parameter values, expressions or any
                        object that can be called with no arguments.
        stale       A Boolean indicating whether the value of this variable is
                        legitimiate.  This value is true if the value should
                        be considered legitimate for purposes of reporting or
                        other interrogation.
        value       The numeric value of this variable.

    The domain, lb, and ub attributes are properties because they
    are too widely accessed directly to enforce explicit getter/setter
    methods and we need to deter directly modifying or accessing
    these attributes in certain cases.
    """

    __slots__ = ('_value', '_lb', '_ub', '_domain', 'fixed', 'stale')

    def __init__(self, domain=Reals, component=None):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - _VarData
        #   - ComponentData
        #   - NumericValue
        self._component = weakref_ref(component) if (component is not None) \
                          else None
        self._value = None
        #
        # The type of the lower and upper bound attributes can either
        # be atomic numeric types in Python, expressions, etc.
        # Basically, they can be anything that passes an "is_fixed" test.
        #
        self._lb = None
        self._ub = None
        self._domain = None
        self.fixed = False
        self.stale = True
        # don't call the property setter here because
        # the SimplVar constructor will fail
        if hasattr(domain, 'bounds'):
            self._domain = domain
        elif domain is not None:
            raise ValueError(
                "%s is not a valid domain. Variable domains must be an "
                "instance of one of %s, or an object that declares a method "
                "for bounds (like a Pyomo Set). Examples: NonNegativeReals, "
                "Integers, Binary" % (domain, (RealSet, IntegerSet, BooleanSet)))

    def __getstate__(self):
        state = super(_GeneralVarData, self).__getstate__()
        for i in _GeneralVarData.__slots__:
            state[i] = getattr(self, i)
        return state

    # Note: None of the slots on this class need to be edited, so we
    # don't need to implement a specialized __setstate__ method, and
    # can quietly rely on the super() class's implementation.

    #
    # Abstract Interface
    #

    # value is an attribute

    @property
    def value(self):
        """Return the value for this variable."""
        return self._value
    @value.setter
    def value(self, val):
        """Set the value for this variable."""
        self._value = val

    @property
    def domain(self):
        """Return the domain for this variable."""
        return self._domain
    @domain.setter
    def domain(self, domain):
        """Set the domain for this variable."""
        if hasattr(domain, 'bounds'):
            self._domain = domain
        else:
            raise ValueError(
                "%s is not a valid domain. Variable domains must be an "
                "instance of one of %s, or an object that declares a method "
                "for bounds (like a Pyomo Set). Examples: NonNegativeReals, "
                "Integers, Binary" % (domain, (RealSet, IntegerSet, BooleanSet)))

    @property
    def lb(self):
        """Return the lower bound for this variable."""
        dlb, _ = self.domain.bounds()
        if self._lb is None:
            return dlb
        elif dlb is None:
            return value(self._lb)
        return max(value(self._lb), dlb)
    @lb.setter
    def lb(self, val):
        raise AttributeError("Assignment not allowed. Use the setlb method")

    @property
    def ub(self):
        """Return the upper bound for this variable."""
        _, dub = self.domain.bounds()
        if self._ub is None:
            return dub
        elif dub is None:
            return value(self._ub)
        return min(value(self._ub), dub)
    @ub.setter
    def ub(self, val):
        raise AttributeError("Assignment not allowed. Use the setub method")

    # fixed is an attribute

    # stale is an attribute

    def setlb(self, val):
        """
        Set the lower bound for this variable after validating that
        the value is fixed (or None).
        """
        # Note: is_fixed(None) returns True
        if is_fixed(val):
            self._lb = val
        else:
            raise ValueError(
                "Non-fixed input of type '%s' supplied as variable lower "
                "bound - legal types must be fixed expressions or variables."
                % (type(val),))

    def setub(self, val):
        """
        Set the upper bound for this variable after validating that
        the value is fixed (or None).
        """
        # Note: is_fixed(None) returns True
        if is_fixed(val):
            self._ub = val
        else:
            raise ValueError(
                "Non-fixed input of type '%s' supplied as variable upper "
                "bound - legal types are fixed expressions or variables."
                "parameters"
                % (type(val),))

    def fix(self, *val):
        """
        Set the fixed indicator to True. Value argument is optional,
        indicating the variable should be fixed at its current value.
        """
        self.fixed = True
        if len(val) == 1:
            self.value = val[0]
        elif len(val) > 1:
            raise TypeError("fix expected at most 1 arguments, got %d" % (len(val)))

    def unfix(self):
        """Sets the fixed indicator to False."""
        self.fixed = False

    free = unfix

class Var(IndexedComponent):
    """
    A numeric variable, which may be defined over an index.

    Constructor Arguments:
        domain      A set that defines the type of values that
                        each variable must be.
        bounds      A rule for defining bounds values for this
                        variable.
        initialize  A rule for setting up this variable with
                        existing model data
        rule        A function for declaring variables.
        dense       An option to specify that the variables are declared densely.
    """

    def __new__(cls, *args, **kwds):
        if cls != Var:
            return super(Var, cls).__new__(cls)
        if args == () or (type(args[0]) == set and args[0] == UnindexedComponent_set and len(args)==1):
            return SimpleVar.__new__(SimpleVar)
        else:
            return IndexedVar.__new__(IndexedVar)

    def __init__(self, *args, **kwd):
        #
        # Default keyword values
        #
        initialize = kwd.pop('initialize', None)
        initialize = kwd.pop('rule', initialize)
        domain = kwd.pop('within', Reals)
        domain = kwd.pop('domain', domain)
        bounds = kwd.pop('bounds', None)
        self._dense = kwd.pop('dense', True)

        #
        # Initialize the base class
        #
        kwd.setdefault('ctype', Var)
        IndexedComponent.__init__(self, *args, **kwd)
        #
        # Determine if the domain argument is a functor or other object
        #
        self._domain_init_value = None
        self._domain_init_rule = None
        if is_functor(domain):
            self._domain_init_rule = domain
        else:
            self._domain_init_value = domain
        #
        # Allow for functions or functors for value initialization,
        # without confusing with Params, etc (which have a __call__ method).
        #
        self._value_init_value = None
        self._value_init_rule = None
        if  is_functor(initialize) and (not isinstance(initialize,NumericValue)):
            self._value_init_rule = initialize
        else:
            self._value_init_value = initialize
        #
        # Determine if the bound argument is a functor or other object
        #
        self._bounds_init_rule = None
        self._bounds_init_value = None
        if is_functor(bounds):
            self._bounds_init_rule = bounds
        elif type(bounds) is tuple:
            self._bounds_init_value = bounds
        elif bounds is not None:
            raise ValueError("Variable 'bounds' keyword must be a tuple or function")

    def flag_as_stale(self):
        """
        Set the 'stale' attribute of every variable data object to True.
        """
        for var_data in itervalues(self._data):
            var_data.stale = True

    def get_values(self, include_fixed_values=True):
        """
        Return a dictionary of index-value pairs.
        """
        if include_fixed_values:
            return dict((idx, vardata.value)
                            for idx, vardata in iteritems(self._data))
        return dict((idx, vardata.value)
                            for idx, vardata in iteritems(self._data)
                                                if not vardata.fixed)

    def set_values(self, new_values, valid=False):
        """
        Set the values of a dictionary.

        The default behavior is to validate the values in the
        dictionary.
        """
        for index, new_value in iteritems(new_values):
            self[index].set_value(new_value, valid)

    def __setitem__(self, ndx, val):
        """
        Define the setitem operation:
            var[ndx] = val
        """
        #
        # Get the variable data object
        #
        vardata = None
        if ndx in self._data:
            vardata = self._data[ndx]
        elif normalize_index.flatten:
            _ndx = normalize_index(ndx)
            if _ndx in self._data:
                vardata = self._data[_ndx]
        if vardata is None:
            if self.is_indexed():
                vardata = self.add(ndx)
            else:
                msg = "Cannot set the value of a simple variable '%s' with index '%s'"
                raise KeyError(msg % (self.name, str(ndx)))
        #
        # Set the value
        #
        vardata.set_value(val)

    def construct(self, data=None):
        """Construct this component."""

        if __debug__ and logger.isEnabledFor(logging.DEBUG):   #pragma:nocover
            try:
                name = str(self.name)
            except:
                # Some Var components don't have a name yet, so just use the type
                name = type(self)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Constructing Variable, name=%s, from data=%s"
                    % (name, str(data)))

        if self._constructed:
            return
        self._constructed=True

        #
        # Construct _VarData objects for all index values
        #
        if not self.is_indexed():
            self._data[None] = self
            self._initialize_members([None])
        elif self._dense:
            # This loop is optimized for speed with pypy.
            # Calling dict.update((...) for ...) is roughly
            # 30% slower
            self_weakref = weakref_ref(self)
            for ndx in self._index:
                cdata = _GeneralVarData(domain=self._domain_init_value,
                                        component=None)
                cdata._component = self_weakref
                self._data[ndx] = cdata
            self._initialize_members(self._index)

    def add(self, index):
        """Add a variable with a particular index."""
        if not index in self._index:
            raise KeyError("Cannot add variable with index %s" % str(index))
        return self._default(index)

    #
    # This method must be defined on subclasses of
    # IndexedComponent
    #
    def _default(self, idx):
        """Returns the default component data value."""
        vardata = self._data[idx] = _GeneralVarData(self._domain_init_value,
                                                    component=self)
        self._initialize_members([idx])
        return vardata

    def _initialize_members(self, init_set):
        """Initialize variable data for all indices in a set."""
        #
        # Initialize domains
        #
        if self._domain_init_rule is not None:
            #
            # Initialize domains with a rule
            #
            if self.is_indexed():
                for ndx in init_set:
                    self._data[ndx].domain = \
                        apply_indexed_rule(self,
                                           self._domain_init_rule,
                                           self._parent(),
                                           ndx)
            else:
                self.domain = self._domain_init_rule(self._parent())
        else:
            if self.is_indexed():
                # Optimization: It is assumed self._domain_init_value
                #               is used when the _GeneralVarData objects
                #               are created. This avoids an unnecessary
                #               loop over init_set, which can significantly
                #               speed up construction of variables with large
                #               index sets.
                pass
            else:
                # the above optimization does not apply for
                # singleton objects (trying to do so breaks
                # some of the pickle tests)
                self.domain = self._domain_init_value

        #
        # Initialize values
        #
        if self._value_init_rule is not None:
            #
            # Initialize values with a rule
            #
            if self.is_indexed():
                for key in init_set:
                    vardata = self._data[key]
                    val = apply_indexed_rule(self,
                                             self._value_init_rule,
                                             self._parent(),
                                             key)
                    val = value(val)
                    vardata.set_value(val)
            else:
                val = self._value_init_rule(self._parent())
                val = value(val)
                self.set_value(val)
        elif self._value_init_value is not None:
            #
            # Initialize values with a value
            #
            if self._value_init_value.__class__ is dict:
                for key in init_set:
                    # Skip indices that are not in the
                    # dictionary. This arises when
                    # initializing VarList objects with a
                    # dictionary.
                    if not key in self._value_init_value:
                        continue
                    val = self._value_init_value[key]
                    vardata = self._data[key]
                    vardata.set_value(val)
            else:
                val = value(self._value_init_value)
                for key in init_set:
                    vardata = self._data[key]
                    vardata.set_value(val)

        #
        # Initialize bounds
        #
        if self._bounds_init_rule is not None:
            #
            # Initialize bounds with a rule
            #
            if self.is_indexed():
                for key in init_set:
                    vardata = self._data[key]
                    (lb, ub) = apply_indexed_rule(self,
                                                  self._bounds_init_rule,
                                                  self._parent(),
                                                  key)
                    vardata.setlb(lb)
                    vardata.setub(ub)
            else:
                (lb, ub) = self._bounds_init_rule(self._parent())
                self.setlb(lb)
                self.setub(ub)
        elif self._bounds_init_value is not None:
            #
            # Initialize bounds with a value
            #
            (lb, ub) = self._bounds_init_value
            for key in init_set:
                vardata = self._data[key]
                vardata.setlb(lb)
                vardata.setub(ub)

    def _pprint(self):
        """Print component information."""
        return ( [("Size", len(self)),
                  ("Index", self._index \
                       if self._index != UnindexedComponent_set else None),
                  ],
                 iteritems(self._data),
                 ( "Lower","Value","Upper","Fixed","Stale","Domain"),
                 lambda k, v: [ value(v.lb),
                                v.value,
                                value(v.ub),
                                v.fixed,
                                v.stale,
                                v.domain
                                ]
                 )

class SimpleVar(_GeneralVarData, Var):
    """"A single variable."""

    def __init__(self, *args, **kwd):
        _GeneralVarData.__init__(self,
                                 domain=Reals,
                                 component=self)
        Var.__init__(self, *args, **kwd)

    #
    # Since this class derives from Component and Component.__getstate__
    # just packs up the entire __dict__ into the state dict, we do not
    # need to define the __getstate__ or __setstate__ methods.
    # We just defer to the super() get/set state.  Since all of our
    # get/set state methods rely on super() to traverse the MRO, this
    # will automatically pick up both the Component and Data base classes.
    #

    #
    # Override abstract interface methods to first check for
    # construction
    #

    # NOTE: that we can't provide these errors for
    # fixed and stale because they are attributes

    @property
    def value(self):
        """Return the value for this variable."""
        if self._constructed:
            return _GeneralVarData.value.fget(self)
        raise ValueError(
            "Accessing the value of variable '%s' "
            "before the Var has been constructed (there "
            "is currently no value to return)."
            % (self.name))
    @value.setter
    def value(self, val):
        """Set the value for this variable."""
        if self._constructed:
            return _GeneralVarData.value.fset(self, val)
        raise ValueError(
            "Setting the value of variable '%s' "
            "before the Var has been constructed (there "
            "is currently nothing to set."
            % (self.name))

    @property
    def domain(self):
        """Return the domain for this variable."""
        # NOTE: we can't do the right thing here because
        #       of the behavior of block._add_temporary_set
        return _GeneralVarData.domain.fget(self)
        #if self._constructed:
        #    return _GeneralVarData.domain.fget(self)
        #raise ValueError(
        #    "Accessing the domain of variable '%s' "
        #    "before the Var has been constructed (there "
        #    "is currently no domain to return)."
        #    % (self.name))
    @domain.setter
    def domain(self, domain):
        """Set the domain for this variable."""
        if self._constructed:
            return _GeneralVarData.domain.fset(self, domain)
        raise ValueError(
            "Setting the domain of variable '%s' "
            "before the Var has been constructed (there "
            "is currently nothing to set."
            % (self.name))

    @property
    def lb(self):
        """Return the lower bound for this variable."""
        if self._constructed:
            return _GeneralVarData.lb.fget(self)
        raise ValueError(
            "Accessing the lower bound of variable '%s' "
            "before the Var has been constructed (there "
            "is currently nothing to return)."
            % (self.name))
    @lb.setter
    def lb(self, lb):
        raise AttributeError("Assignment not allowed. Use the setlb method")

    @property
    def ub(self):
        """Return the upper bound for this variable."""
        if self._constructed:
            return _GeneralVarData.ub.fget(self)
        raise ValueError(
            "Accessing the upper bound of variable '%s' "
            "before the Var has been constructed (there "
            "is currently nothing to return)."
            % (self.name))
    @ub.setter
    def ub(self, ub):
        raise AttributeError("Assignment not allowed. Use the setub method")

    def setlb(self, val):
        """
        Set the lower bound for this variable after validating that
        the value is fixed (or None).
        """
        if self._constructed:
            return _GeneralVarData.setlb(self, val)
        raise ValueError(
            "Setting the lower bound of variable '%s' "
            "before the Var has been constructed (there "
            "is currently nothing to set)."
            % (self.name))

    def setub(self, val):
        """
        Set the upper bound for this variable after validating that
        the value is fixed (or None).
        """
        if self._constructed:
            return _GeneralVarData.setub(self, val)
        raise ValueError(
            "Setting the upper bound of variable '%s' "
            "before the Var has been constructed (there "
            "is currently nothing to set)."
            % (self.name))

    def fix(self, *val):
        """
        Set the fixed indicator to True. Value argument is optional,
        indicating the variable should be fixed at its current value.
        """
        if self._constructed:
            return _GeneralVarData.fix(self, *val)
        raise ValueError(
            "Fixing variable '%s' "
            "before the Var has been constructed (there "
            "is currently nothing to set)."
            % (self.name))

    def unfix(self):
        """Sets the fixed indicator to False."""
        if self._constructed:
            return _GeneralVarData.unfix(self)
        raise ValueError(
            "Freeing variable '%s' "
            "before the Var has been constructed (there "
            "is currently nothing to set)."
            % (self.name))

    free=unfix

class IndexedVar(Var):
    """An array of variables."""

    # These methods are normally found on the NumericValue
    # interface, but they are here to trick the expression
    # system into reporting errors about trying to use
    # "indexed NumericValue" objects in expressions
    def as_numeric(self): return self
    def is_expression(self): return False
    def is_relational(self): return False

    def fix(self, *val):
        """
        Set the fixed indicator to True. Value argument is optional,
        indicating the variable should be fixed at its current value.
        """
        for vardata in itervalues(self):
            vardata.fix(*val)

    def unfix(self):
        """Sets the fixed indicator to False."""
        for vardata in itervalues(self):
            vardata.unfix()

    @property
    def domain(self):
        raise AttributeError(
            "The domain is not an attribute for IndexedVar. It "
            "can be set for all indices using this property setter, "
            "but must be accessed for individual variables in this container.")
    @domain.setter
    def domain(self, domain):
        """Sets the domain for all variables in this container."""
        for vardata in itervalues(self):
            vardata.domain = domain

    free=unfix

class VarList(IndexedVar):
    """
    Variable-length indexed variable objects used to construct Pyomo models.
    """

    def __init__(self, **kwds):
        kwds['dense'] = False
        args = (Set(),)
        self._nvars = 0
        IndexedVar.__init__(self, *args, **kwds)

    def construct(self, data=None):
        """Construct this component."""
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Constructing variable list %s", self.name)
        IndexedVar.construct(self, data)

    def add(self):
        """Add a variable to this list."""
        self._nvars += 1
        self._index.add(self._nvars)
        vardata = self._data[self._nvars] = \
            _GeneralVarData(domain=self._domain_init_value,
                            component=self)
        self._initialize_members([self._nvars])
        return vardata

register_component(Var, "Decision variables.")
register_component(VarList, "List of decision variables.")

