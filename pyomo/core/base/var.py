#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['Var', '_VarData', '_GeneralVarData', 'VarList', 'SimpleVar',
           'ScalarVar']

import logging
from weakref import ref as weakref_ref

from pyomo.common.deprecation import RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NoArgumentGiven
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.expr.numvalue import (
    NumericValue, value, is_potentially_variable, native_numeric_types,
)
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.indexed_component import (
    IndexedComponent, UnindexedComponent_set, IndexedComponent_NDArrayMixin
)
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.set import Reals, Binary, Set, _SetDataBase
from pyomo.core.base.units_container import units
from pyomo.core.base.util import is_functor

logger = logging.getLogger('pyomo.core')

_no_lower_bound = {None, -float('inf')}
_no_upper_bound = {None, float('inf')}
_VARDATA_API = (
    # including 'domain' runs afoul of logic in Block._add_implicit_sets()
    # 'domain',
    'bounds', 'lower', 'upper', 'lb', 'ub', 'has_lb', 'has_ub',
    'setlb', 'setub',
    'is_integer', 'is_binary', 'is_continuous',
    'is_fixed', 'fix', 'unfix', 'free', 'set_value', 'value',
    # Note: we can't disable fixed / stale as they are public attributes
)

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

    def has_lb(self):
        """Returns :const:`False` when the lower bound is
        :const:`None` or negative infinity"""
        return self.lb not in _no_lower_bound

    def has_ub(self):
        """Returns :const:`False` when the upper bound is
        :const:`None` or positive infinity"""
        return self.ub not in _no_upper_bound

    # TODO: deprecate this?  Properties are generally preferred over "set*()"
    def setlb(self, val):
        """
        Set the lower bound for this variable after validating that
        the value is fixed (or None).
        """
        self.lower = val

    # TODO: deprecate this?  Properties are generally preferred over "set*()"
    def setub(self, val):
        """
        Set the upper bound for this variable after validating that
        the value is fixed (or None).
        """
        self.upper = val

    @property
    def bounds(self):
        """Returns the tuple (lower bound, upper bound).

        This returns the current (numeric) values of the lower and upper
        bounds as a tuple.  If there is no bound, returns None (and not
        +/-inf)

        """
        return (self.lb, self.ub)
    @bounds.setter
    def bounds(self, val):
        self.lower, self.upper = val

    @property
    def lb(self):
        """Return the numeric value of the variable lower bound."""
        return value(self.lower)
    @lb.setter
    def lb(self, val):
        self.lower = val

    @property
    def ub(self):
        """Return the numeric value of the variable upper bound."""
        return value(self.upper)
    @ub.setter
    def ub(self, val):
        self.upper = val

    def is_integer(self):
        """Returns True when the domain is a contiguous integer range."""
        # optimization: Reals and Binary are the most common cases, so
        # we will explicitly test that before generating the interval
        if self.domain is Reals:
            return False
        elif self.domain is Binary:
            return True
        _interval = self.domain.get_interval()
        return _interval is not None and _interval[2] == 1

    def is_binary(self):
        """Returns True when the domain is restricted to Binary values."""
        # optimization: Reals and Binary are the most common cases, so
        # we will explicitly test that before generating the interval
        if self.domain is Reals:
            return False
        elif self.domain is Binary:
            return True
        return self.domain.get_interval() == (0,1,1)

    def is_continuous(self):
        """Returns True when the domain is a continuous real range"""
        # optimization: Reals and Binary are the most common cases, so
        # we will explicitly test that before generating the interval
        if self.domain is Reals:
            return True
        elif self.domain is Binary:
            return False
        _interval = self.domain.get_interval()
        return _interval is not None and _interval[2] == 0

    def is_fixed(self):
        """Returns True if this variable is fixed, otherwise returns False."""
        return self.fixed

    def is_constant(self):
        """Returns False because this is not a constant in an expression."""
        return False

    def is_variable_type(self):
        """Returns True because this is a variable."""
        return True

    def is_potentially_variable(self):
        """Returns True because this is a variable."""
        return True

    def _compute_polynomial_degree(self, result):
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
        if not valid and val is not None:
            # TODO: warn/error: check if this Var has units: assigning
            # a dimensionless value to a united variable should be an error
            if type(val) not in native_numeric_types:
                if self.parent_component()._units is not None:
                    _src_magnitude = value(val)
                    _src_units = units.get_units(val)
                    val = units.convert_value(
                        num_value=_src_magnitude, from_units=_src_units,
                        to_units=self.parent_component()._units)

            if val not in self.domain:
                raise ValueError("Numeric value `%s` (%s) is not in "
                                 "domain %s for Var %s" %
                                 (val, type(val), self.domain, self.name))
                
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
                             "domain %s for Var %s" %
                             (val, type(val), self.domain, self.name))
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
    def lower(self):
        """Return an expression for the vaiable lower bound."""
        raise NotImplementedError

    @property
    def upper(self):
        """Return an expression for the variable upper bound."""
        raise NotImplementedError

    @property
    def fixed(self):
        """Return the fixed indicator for this variable."""
        raise NotImplementedError

    @property
    def stale(self):
        """Return the stale indicator for this variable."""
        raise NotImplementedError

    def fix(self, value=NoArgumentGiven):
        """
        Set the fixed indicator to True. Value argument is optional,
        indicating the variable should be fixed at its current value.
        """
        raise NotImplementedError

    def unfix(self):
        """Sets the fixed indicator to False."""
        raise NotImplementedError

    def free(self):
        return self.unfix()


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
        #
        # TODO: this should be migrated over to using a SetInitializer
        # to handle the checking / conversion of the argument to a
        # proper Pyomo Set and not use isinstance() of a private class.
        if isinstance(domain, _SetDataBase):
            self._domain = domain
        elif domain is not None:
            raise ValueError(
                "%s is not a valid domain. Variable domains must be an "
                "instance of a Pyomo Set.  Examples: NonNegativeReals, "
                "Integers, Binary" % (domain,))

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

    @property
    def value(self):
        """Return the value for this variable."""
        return self._value
    @value.setter
    def value(self, val):
        """Set the value for this variable."""
        if type(val) in native_numeric_types:
            # TODO: warn/error: check if this Var has units: assigning
            # a dimensionless value to a united variable should be an error
            pass
        elif val is not None and self.parent_component()._units is not None:
            _src_magnitude = value(val)
            _src_units = units.get_units(val)
            val = units.convert_value(
                num_value=_src_magnitude, from_units=_src_units,
                to_units=self.parent_component()._units)
        self._value = val

    @property
    def domain(self):
        """Return the domain for this variable."""
        return self._domain
    @domain.setter
    def domain(self, domain):
        """Set the domain for this variable."""
        # TODO: this should be migrated over to using a SetInitializer
        # to handle the checking / conversion of the argument to a
        # proper Pyomo Set and not use isinstance() of a private class.
        if isinstance(domain, _SetDataBase):
            self._domain = domain
        else:
            raise ValueError(
                "%s is not a valid domain. Variable domains must be an "
                "instance of a Pyomo Set.  Examples: NonNegativeReals, "
                "Integers, Binary" % (domain,))

    @property
    def lower(self):
        """Return an expression for the variable lower bound.

        This returns a (non-potentially variable) expression for the
        variable lower bound.  This represents the tighter of the
        current domain and the constant or expression provided to
        setlb().  Note that the expression will NOT automatically
        reflect changes to either the domain or the bound expression
        (e.g., because of a call to setlb()).

        """
        dlb, _ = self.domain.bounds()
        if self._lb is None:
            return dlb
        elif dlb is None:
            return self._lb
        # This is guaranteed by _process_bound():
        # assert not is_potentially_variable(self._lb)
        return NPV_MaxExpression((self._lb, dlb))
    @lower.setter
    def lower(self, val):
        self._lb = self._process_bound(val, 'lower')

    @property
    def upper(self):
        """Return an expression for the variable upper bound.

        This returns a (non-potentially variable) expression for the
        variable upper bound.  This represents the tighter of the
        current domain and the constant or expression provided to
        setub().  Note that the expression will NOT automatically
        reflect changes to either the domain or the bound expression
        (e.g., because of a call to setub()).

        """
        _, dub = self.domain.bounds()
        if self._ub is None:
            return dub
        elif dub is None:
            return self._ub
        # This is guaranteed by _process_bound():
        # assert not is_potentially_variable(self._lb)
        return NPV_MinExpression((self._ub, dub))
    @upper.setter
    def upper(self, val):
        self._ub = self._process_bound(val, 'upper')

    def get_units(self):
        """Return the units for this variable entry."""
        # parent_component() returns self if this is scalar, or the owning
        # component if not scalar
        return self.parent_component()._units

    # fixed is an attribute

    # stale is an attribute

    def fix(self, value=NoArgumentGiven):
        """
        Set the fixed indicator to True. Value argument is optional,
        indicating the variable should be fixed at its current value.
        """
        self.fixed = True
        if value is not NoArgumentGiven:
            self.value = value

    def unfix(self):
        """Sets the fixed indicator to False."""
        self.fixed = False

    def _process_bound(self, val, bound_type):
        # Note: is_potentially_variable(None) returns False
        if is_potentially_variable(val):
            raise ValueError(
                "Potentially variable input of type '%s' supplied as "
                "%s bound for variable '%s' - legal types must be constants "
                "or non-potentially variable expressions."
                % (type(val).__name__, bound_type, self.name))
        if type(val) in native_numeric_types or val is None:
            # TODO: warn/error: check if this Var has units: assigning
            # a dimensionless value to a united variable should be an error
            pass
        else:
            # We want to create an expression and not just convert the
            # current value so that things like mutable Params behave as
            # expected.
            if self.parent_component()._units is not None:
                val = units.convert(
                    val, to_units=self.parent_component()._units)
        return val


@ModelComponentFactory.register("Decision variables.")
class Var(IndexedComponent, IndexedComponent_NDArrayMixin):
    """A numeric variable, which may be defined over an index.

    Args:
        domain (Set or function, optional): A Set that defines valid
            values for the variable (e.g., `Reals`, `NonNegativeReals`,
            `Binary`), or a rule that returns Sets.  Defaults to `Reals`.
        within (Set or function, optional): An alias for `domain`.
        bounds (tuple or function, optional): A tuple of (lower, upper)
            bounds for the variable, or a rule that returns tuples.
            Defaults to (None, None).
        initialize (float or function, optional): The initial value for
            the variable, or a rule that returns initial values.
        rule (float or function, optional): An alias for `initialize`.
        dense (bool, optional): Instantiate all elements from
            `index_set()` when constructing the Var (True) or just the
            variables returned by `initialize`/`rule` (False).  Defaults
            to True.
        units (pyomo units expression, optional): Set the units corresponding
            to the entries in this variable.
    """

    _ComponentDataClass = _GeneralVarData

    def __new__(cls, *args, **kwds):
        if cls is not Var:
            return super(Var, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args)==1):
            return super(Var, cls).__new__(AbstractScalarVar)
        else:
            return super(Var, cls).__new__(IndexedVar)

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
        self._units = kwd.pop('units', None)
        if self._units is not None:
            self._units = units.get_units(self._units)

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
        for var_data in self._data.values():
            var_data.stale = True

    def get_values(self, include_fixed_values=True):
        """
        Return a dictionary of index-value pairs.
        """
        if include_fixed_values:
            return {idx:vardata.value for idx,vardata in self._data.items()}
        return {idx:vardata.value
                            for idx, vardata in self._data.items()
                                                if not vardata.fixed}

    extract_values = get_values

    def set_values(self, new_values, valid=False):
        """
        Set the values of a dictionary.

        The default behavior is to validate the values in the
        dictionary.
        """
        for index, new_value in new_values.items():
            self[index].set_value(new_value, valid)

    def get_units(self):
        """Return the units expression for this Var."""
        return self._units

    def construct(self, data=None):
        """Construct this component."""
        if self._constructed:
            return
        if is_debug_set(logger):   #pragma:nocover
            try:
                name = str(self.name)
            except:
                # Some Var components don't have a name yet, so just use
                # the type
                name = type(self)
            logger.debug(
                "Constructing Variable, name=%s, from data=%s"
                % (name, str(data)))

        timer = ConstructionTimer(self)
        self._constructed=True

        #
        # Construct _VarData objects for all index values
        #
        if not self.is_indexed():
            self._data[None] = self
            self._initialize_members((None,))
        elif self._dense:
            # This loop is optimized for speed with pypy.
            # Calling dict.update((...) for ...) is roughly
            # 30% slower
            self_weakref = weakref_ref(self)
            for ndx in self._index:
                cdata = self._ComponentDataClass(
                    domain=self._domain_init_value, component=None)
                cdata._component = self_weakref
                self._data[ndx] = cdata
                #self._initialize_members((ndx,))
            self._initialize_members(self._index)
        timer.report()

    def add(self, index):
        """Add a variable with a particular index."""
        return self[index]

    #
    # This method must be defined on subclasses of
    # IndexedComponent that support implicit definition
    #
    def _getitem_when_not_present(self, index):
        """Returns the default component data value."""
        if index is None and not self.is_indexed():
            obj = self._data[index] = self
        else:
            obj = self._data[index] = self._ComponentDataClass(
                self._domain_init_value, component=self)
        self._initialize_members((index,))
        return obj

    def _setitem_when_not_present(self, index, value):
        """Perform the fundamental component item creation and storage.

        Var overrides the default implementation from IndexedComponent
        to enforce the call to _initialize_members.

        """
        obj = self._getitem_when_not_present(index)
        try:
            return obj.set_value(value)
        except:
            del self._data[index]
            raise

    def _initialize_members(self, init_set):
        """Initialize variable data for all indices in a set."""
        # TODO: determine if there is any advantage to supporting init_set.
        # Preliminary tests indicate that there isn't a significant speed
        # difference to using the set form (used in dense vector
        # construction).  Getting rid of it could simplify _setitem and
        # this method.
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
        headers = [
            ("Size", len(self)),
            ("Index", self._index if self.is_indexed() else None),
        ]
        if self._units is not None:
            headers.append(('Units', str(self._units)))
        return ( headers,
                 self._data.items(),
                 ( "Lower","Value","Upper","Fixed","Stale","Domain"),
                 lambda k, v: [ value(v.lb),
                                v.value,
                                value(v.ub),
                                v.fixed,
                                v.stale,
                                v.domain
                                ]
                 )


class ScalarVar(_GeneralVarData, Var):
    """A single variable."""

    def __init__(self, *args, **kwd):
        _GeneralVarData.__init__(self, domain=None, component=self)
        Var.__init__(self, *args, **kwd)

    # Since this class derives from Component and Component.__getstate__
    # just packs up the entire __dict__ into the state dict, we do not
    # need to define the __getstate__ or __setstate__ methods.
    # We just defer to the super() get/set state.  Since all of our
    # get/set state methods rely on super() to traverse the MRO, this
    # will automatically pick up both the Component and Data base classes.


@disable_methods(_VARDATA_API)
class AbstractScalarVar(ScalarVar):
    pass


class SimpleVar(metaclass=RenamedClass):
    __renamed__new_class__ = ScalarVar
    __renamed__version__ = '6.0'


class IndexedVar(Var):
    """An array of variables."""

    def setlb(self, val):
        """
        Set the lower bound for this variable.
        """
        for vardata in self.values():
            vardata.setlb(val)

    def setub(self, val):
        """
        Set the upper bound for this variable.
        """
        for vardata in self.values():
            vardata.setub(val)

    def fix(self, value=NoArgumentGiven):
        """
        Set the fixed indicator to True. Value argument is optional,
        indicating the variable should be fixed at its current value.
        """
        for vardata in self.values():
            vardata.fix(value=value)

    def unfix(self):
        """Sets the fixed indicator to False."""
        for vardata in self.values():
            vardata.unfix()

    def free(self):
        return self.unfix()

    @property
    def domain(self):
        raise AttributeError(
            "The domain is not an attribute for IndexedVar. It "
            "can be set for all indices using this property setter, "
            "but must be accessed for individual variables in this container.")
    @domain.setter
    def domain(self, domain):
        """Sets the domain for all variables in this container."""
        for vardata in self.values():
            vardata.domain = domain


@ModelComponentFactory.register("List of decision variables.")
class VarList(IndexedVar):
    """
    Variable-length indexed variable objects used to construct Pyomo models.
    """

    def __init__(self, **kwds):
        #kwds['dense'] = False
        args = (Set(dimen=1),)
        IndexedVar.__init__(self, *args, **kwds)

    def construct(self, data=None):
        """Construct this component."""
        if is_debug_set(logger):
            logger.debug("Constructing variable list %s", self.name)

        if self._constructed:
            return
        # Note: do not set _constructed here, or the super() call will
        # not actually construct the component.
        self.index_set().construct()

        # We need to ensure that the indices needed for initialization are
        # added to the underlying implicit set.  We *could* verify that the
        # indices in the initialization dict are all sequential integers,
        # OR we can just add the correct number of sequential integers and
        # then let _validate_index complain when we set the value.
        if self._value_init_value.__class__ is dict:
            for i in range(len(self._value_init_value)):
                self._index.add(i+1)
        super(VarList,self).construct(data)
        # Note that the current Var initializer silently ignores
        # initialization data that is not in the underlying index set.  To
        # ensure that at least here all initialization data is added to the
        # VarList (so we get potential domain errors), we will re-set
        # everything.
        if self._value_init_value.__class__ is dict:
            for k,v in self._value_init_value.items():
                self[k] = v

    def add(self):
        """Add a variable to this list."""
        next_idx = len(self._index) + 1
        self._index.add(next_idx)
        return self[next_idx]
