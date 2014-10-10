#  _________________________________________________________________________
#
#  Coopr: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Coopr README.txt file.
#  _________________________________________________________________________

__all__ = ['Var', 'VarList']

import logging
import sys
import weakref
from six import iteritems, iterkeys, itervalues
from six.moves import xrange

from coopr.pyomo.base.numvalue import NumericValue, value, is_fixed
from coopr.pyomo.base.set_types import BooleanSet, IntegerSet, RealSet, Reals
from coopr.pyomo.base.component import Component, ComponentData, register_component
from coopr.pyomo.base.sparse_indexed_component import SparseIndexedComponent, UnindexedComponent_set, normalize_index
from coopr.pyomo.base.misc import apply_indexed_rule
from coopr.pyomo.base.sets import Set
from coopr.pyomo.base.util import is_functor

logger = logging.getLogger('coopr.pyomo')


_noarg = object()

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
        initial     The default initial value for this variable.
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

    __pickle_slots__ = ('value','initial','_lb','_ub','fixed','stale')
    __slots__ = __pickle_slots__ + ( '__weakref__', )

    def __init__(self, component):
        """
        Constructor
        """
        #
        # The following is equivalent to calling
        # the base ComponentData constructor, as follows:
        # ComponentData.__init__(self, component)
        #
        self._component = weakref.ref(component)
        #
        # The following is equivalent to calling the
        # base NumericValue constructor, as follows:
        # NumericValue.__init__(self, name, domain, None, False)
        #
        self.value = None
        #
        self.fixed = False
        self.initial = None
        self.stale = True
        #
        # The type of the lower and upper bound attributes can either 
        # be atomic numeric types in Python, expressions, etc. 
        # Basically, they can be anything that passes an "is_fixed" test.
        #
        self._lb = None
        self._ub = None

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        result = super(_VarData, self).__getstate__()
        for i in _VarData.__pickle_slots__:
            result[i] = getattr(self, i)
        return result

    # Note: because NONE of the slots on this class need to be edited,
    # we don't need to implement a specialized __setstate__ method.

    #
    # 'domain' is a property so we can ensure that a 'bounds' attribute exists on the
    # domain object.
    #
    @property
    def domain(self):
        """
        Return the domain attribute.
        """
        return self._component()._domain

    #
    # 'lb' is a property because we need to compare it against the lower bound of 
    # the 'domain' attribute to properly handle domains that can change after a 
    # variable has been constructed.
    # 
    @property
    def lb(self):
        """
        Return the lower bound for this variable.
        """
        dlb, _ = self.domain.bounds()
        if self._lb is None:
            return dlb
        elif dlb is None:
            return value(self._lb)
        return max(value(self._lb), dlb)

    @lb.setter
    def lb(self, val):
        """
        Set the lb attribute to the given value.
        """
        raise AttributeError("Assignment not allowed. Use the setlb method")

    def setlb(self, val):
        """
        Set the lower bound for this variable after validating that 
        the value is fixed (or None).
        """
        if val is None:
            self._lb = None
        else:
            if is_fixed(val):
                self._lb = val
            else:
                raise ValueError(
                    "Non-fixed input of type '%s' supplied as variable lower "
                    "bound - legal types must be fixed expressions or variables."
                    % (type(val),) )

    def setub(self, val):
        """
        Set the upper bound for this variable after validating that 
        the value is fixed (or None).
        """
        if val is None:
            self._ub = None
        else:
            if is_fixed(val):
                self._ub = val
            else:
                raise ValueError(
                    "Non-fixed input of type '%s' supplied as variable upper "
                    "bound - legal types are fixed expressions or variables."
                    "parameters"
                    % (type(val),) )

    #
    # 'ub' is a property because we need to compare it against the upper bound of
    # the 'domain' attribute to properly handle domains that can change after a
    # variable has been constructed.
    # 
    @property
    def ub(self):
        """
        Return the upper bound for this variable.
        """
        _, dub = self.domain.bounds()
        if self._ub is None:
            return dub
        elif dub is None:
            return value(self._ub)
        return min(value(self._ub), dub)

    @ub.setter
    def ub(self, val):
        """
        Set the ub attribute to the given value.
        """
        raise AttributeError("Assignment not allowed. Use the setub method")

    #
    # 'bounds' is a property because we need to compare it against the bounds of 
    # the 'domain' attribute to properly handle domains that can change after a
    # variable has been constructed.
    # 
    @property
    def bounds(self):
        """
        Returns the tuple (lower bound, upper bound).
        """
        return (self.lb, self.ub)

    @bounds.setter
    def bounds(self, val):
        """
        Set the bounds attribute to the given value.
        """
        raise AttributeError("Assignment not allowed. Use the setub and setlb methods")

    def __call__(self, exception=True):
        """
        Return the value of this object.
        """
        return self.value

    def is_integer(self):
        """
        Returns True when the domain is an instance of IntegerSet.
        """
        return self.domain.__class__ is IntegerSet

    def is_binary(self):
        """
        Returns True when the domain is an instance of BooleanSet.
        """
        return self.domain.__class__ is BooleanSet

    def is_continuous(self):
        """
        Returns True when the domain is an instance of RealSet.
        """
        return self.domain.__class__ is RealSet

    def fix(self, val=_noarg):
        """
        Set the fixed indicator to True. Value argument is optional,
        indicating the variable should be fixed at its current value.

        We use the global _noarg object as the default value to allow
        fixing variables to None.
        """
        self.fixed = True
        if val is not _noarg:
            self.value = val

    def unfix(self):
        """
        Sets the fixed indicator to False.
        """
        self.fixed = False

    free=unfix

    def is_fixed(self):
        """
        Returns True if this variable is fixed, otherwise returns False.
        """
        if self.fixed:
            return True
        return False

    def is_constant(self):
        """
        Returns False because this is not a constant in an expression.
        """
        return False

    def polynomial_degree(self):
        """
        If the variable is fixed, it represents a constant is a polynomial with degree 0.
        Otherwise, it has degree 1.

        This method is used in expressions to compute polynomial degree.
        """
        if self.fixed:
            return 0
        return 1

    def set_value(self, val, valid=False):
        """
        Set the value of this numeric object, after validating its value.

        If the 'valid' flag is True, then the validation step is skipped.
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

    __bool__ = __nonzero__


class _VarDataWithDomain(_VarData):
    """
    Variable data that contains a local domain.
    """

    __slots__ = ( '_domain', )

    def __init__(self, component):
        _VarData.__init__(self, component)
        self._domain = Reals

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        result = super(_VarDataWithDomain, self).__getstate__()
        for i in _VarDataWithDomain.__slots__:
            result[i] = getattr(self, i)
        return result

    #
    # 'domain' is a property so we can ensure that a 'bounds' attribute exists on the
    # domain object.
    #
    @property
    def domain(self):
        """
        Return the domain attribute.
        """
        return self._domain

    @domain.setter
    def domain(self, domain):
        """
        Set the domain attribute to the given value.
        """
        if hasattr(domain, 'bounds'):
            self._domain = domain
        else:
            raise ValueError("%s is not a valid domain. Variable domains must be an instance of "
                             "one of %s an object that declares a method for bounds (like a Pyomo Set)."
                             "Examples: NonNegativeReals, Integers, Binary"
                             % (domain, (RealSet, IntegerSet, BooleanSet)))


class Var(SparseIndexedComponent):
    """
    A numeric variable, which may be defined over an index.

    Constructor Arguments:
        name        The name of this variable
        index       The index set that defines the distinct variables.
                        By default, this is None, indicating that there
                        is a single variable.
        domain      A set that defines the type of values that
                        each variable must be.
        bounds      A rule for defining bounds values for this
                        variable.
        rule        A rule for setting up this variable with
                        existing model data
    """

    def __new__(cls, *args, **kwds):
        if cls != Var:
            return super(Var, cls).__new__(cls)
        if args == ():
            return SimpleVar.__new__(SimpleVar)
        else:
            domain = kwds.get('within', Reals )
            domain = kwds.get('domain', domain )
            if is_functor(domain):
                return IndexedVar.__new__(IndexedVar)
            else:
                return IndexedVarWithDomain.__new__(IndexedVarWithDomain)

    def __init__(self, *args, **kwd):
        #
        # Default keyword values
        #
        initialize = kwd.pop('initialize', None )
        domain = kwd.pop('within', Reals )
        domain = kwd.pop('domain', domain )
        bounds = kwd.pop('bounds', None )
        #
        # Initialize the base class
        #
        kwd.setdefault('ctype', Var)
        SparseIndexedComponent.__init__(self, *args, **kwd)
        #
        # Determine if the domain argument is a functor or other object
        #
        self._domain_init_value = None
        self._domain_init_rule = None
        if is_functor(domain):
            self._domain_init_rule = domain
            self._VarData = _VarDataWithDomain
        else:
            self._domain_init_value = domain
            self._VarData = _VarData
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

    def as_numeric(self):
        """
        Return the current object, which we treat as a numeric object.
        """
        return self

    def is_expression(self):
        """
        Return True if this numeric value is an expression.
        """
        return False

    def is_relational(self):
        """
        Return True if this numeric value represents a relational expression.
        """
        return False

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
            self._data[index].set_value(new_value, valid)

    def reset(self):
        """
        Reset the variable values to their initial values.
        """
        for vardata in itervalues(self):
            vardata.set_value(vardata.initial)

    def __setitem__(self, ndx, val):
        """
        Define the setitem operation:
            var[ndx] = val
        """
        #
        # Get the variable data object
        #
        if ndx in self._data:
            vardata = self._data[ndx]
        else:
            _ndx = normalize_index(ndx)
            if _ndx in self._data:
                vardata = self._data[_ndx]
            else:
                msg = "Cannot set the value of variable '%s' with invalid " \
                    "index '%s'"
                raise KeyError(msg % ( self.cname(True), str(ndx) ))
        #  
        # Set the value
        #
        vardata.set_value(val)

    def construct(self, data=None):
        """
        Initialize this component
        """
        if __debug__ and logger.isEnabledFor(logging.DEBUG):   #pragma:nocover
            try:
                name = str(self.cname(True))
            except:
                # Some Var components don't have a name yet, so just use the type
                name = type(self)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Constructing Variable, name=%s, from data=%s", name, str(data))
        #
        if self._constructed:
            return
        self._constructed=True
        #
        # Construct self._VarData objects for all index values
        #
        if self.is_indexed():
            self._add_members(self._index)
        else:
            self._data[None] = self
        self._initialize_members(self._index)

    def add(self, index):
        """
        Add a variable with a particular index
        """
        self._data[index] = self._VarData(self)
        return self._data[index]

    #
    # This method must be defined on subclasses of
    # SparseIndexedComponent
    #
    def _default(self, idx):
        """
        Returns the default component data value
        """
        vardata = self._data[idx] = self._VarData(self)
        self._initialize_members([idx])
        return vardata

    def _add_members(self, init_set):
        """
        Create variable data for all indices in a set
        """
        base_init = self._VarData
        self._data.update((ndx,base_init(self)) for ndx in init_set)

    def _initialize_members(self, init_set):
        """
        Initialize variable data for all indices in a set
        """
        #
        # Initialize domains
        #
        if self._domain_init_rule is not None:
            #
            # Initialize domains with a rule
            #
            if self.is_indexed():
                for ndx in init_set:
                    self._data[ndx].domain = apply_indexed_rule(self, self._domain_init_rule, self._parent(), ndx)
            else:
                self.domain = self._domain_init_rule(self._parent())
        else:
            #
            # Initialize domains with a value
            #
            self._domain = self._domain_init_value
        #
        # Initialize values
        #
        if self._value_init_value is not None:
            #
            # Initialize values with a value
            #
            if self._value_init_value.__class__ is dict:
                for key in init_set:
                    #
                    # Skip indices that are not in the dictionary.  This arises when
                    # initializing VarList objects with a dictionary.
                    #
                    if not key in self._value_init_value:
                        continue
                    val = self._value_init_value[key]
                    vardata = self._data[key]
                    vardata.set_value(val)
                    vardata.initial = val
            else:
                val = value(self._value_init_value)
                for key in init_set:
                    vardata = self._data[key]
                    vardata.set_value(val)
                    vardata.initial = val
        elif self._value_init_rule is not None:
            #
            # Initialize values with a rule
            #
            if self.is_indexed():
                for key in init_set:
                    vardata = self._data[key]
                    val = apply_indexed_rule( self, self._value_init_rule,
                                              self._parent(), key )
                    val = value(val)
                    vardata.set_value(val)
                    vardata.initial = val
            else:
                val = self._value_init_rule(self._parent())
                val = value(val)
                self.set_value(val)
                self.initial = val
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
                    (lb, ub) = apply_indexed_rule( self, self._bounds_init_rule,
                                                   self._parent(), key )
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
        """
        Print component information.
        """
        return ( [("Size", len(self)),
                  ("Index", self._index \
                       if self._index != UnindexedComponent_set else None),
                  ("Domain", None if self._domain_init_value is None else self._domain_init_value.name),
                  ],
                 iteritems(self._data),
                 ( "Key","Lower","Value","Upper","Initial","Fixed","Stale" ),
                 lambda k, v: [ k,
                                value(v.lb),
                                v.value,
                                value(v.ub),
                                v.initial,
                                v.fixed,
                                v.stale,
                                ]
                 )


class SimpleVar(_VarDataWithDomain, Var):
    """
    _VarObject is the implementation representing a single, non-indexed
    variable.
    """

    def __init__(self, *args, **kwd):
        _VarDataWithDomain.__init__(self, self)
        Var.__init__(self, *args, **kwd)

    #
    # Since this class derives from Component and Component.__getstate__
    # just packs up the entire __dict__ into the state dict, we do not
    # need to define the __getstate__ or __setstate__ methods.
    # We just defer to the super() get/set state.  Since all of our 
    # get/set state methods rely on super() to traverse the MRO, this 
    # will automatically pick up both the Component and Data base classes.
    #

    def __call__(self, exception=True):
        """
        Return the value of this variable.
        """
        if self._constructed:
            return _VarDataWithDomain.__call__(self, exception=exception)
        if exception:
            raise ValueError( """Evaluating the numeric value of variable '%s' before the Var has been
            constructed (there is currently no value to return).""" % self.cname(True) )

    def reset(self):
        """
        Reset the variable values to their initial values.
        """
        self.set_value(self.initial)


class IndexedVar(Var):
    """
    An array of variables.
    """
    
    def __call__(self, exception=True):
        """Compute the value of the variable"""
        if exception:
            msg = 'Cannot compute the value of an array of variables'
            raise TypeError(msg)

    def fix(self, val=_noarg):
        """
        Set the fixed indicator to True. Value argument is optional,
        indicating the variable should be fixed at its current value.

        We use the global _noarg object as the default value to allow
        fixing variables to None.
        """
        for vardata in itervalues(self):
            vardata.fix(val)

    def unfix(self):
        """
        Sets the fixed indicator to False.
        """
        for vardata in itervalues(self):
            vardata.unfix()

    free=unfix

    #
    # 'domain' is a property so we can ensure that a 'bounds' attribute exists on the
    # domain object.
    #
    @property
    def domain(self):
        """
        Return the domain attribute.
        """
        return None


class IndexedVarWithDomain(IndexedVar):
    """
    An array of variables.
    """
    
    def __init__(self, *args, **kwds):
        IndexedVar.__init__(self, *args, **kwds)
        self._domain = None

    #
    # 'domain' is a property so we can ensure that a 'bounds' attribute exists on the
    # domain object.
    #
    @property
    def domain(self):
        """
        Return the domain attribute.
        """
        return self._domain

    @domain.setter
    def domain(self, domain):
        """
        Set the domain attribute to the given value.
        """
        if hasattr(domain, 'bounds'):
            self._domain = domain
        else:
            raise ValueError("%s is not a valid domain. Variable domains must be an instance of "
                             "one of %s an object that declares a method for bounds (like a Pyomo Set)."
                             "Examples: NonNegativeReals, Integers, Binary"
                             % (domain, (RealSet, IntegerSet, BooleanSet)))


class VarList(IndexedVar):
    """
    Variable-length indexed variable objects used to construct Pyomo models.
    """

    End = ( 1003, )

    def __new__(cls, *args, **kwds):
        if cls != VarList:
            return super(VarList, cls).__new__(cls)
        domain = kwds.get('within', Reals )
        domain = kwds.get('domain', domain )
        if is_functor(domain):
            return _VarList.__new__(_VarList)
        else:
            return _VarListWithDomain.__new__(_VarListWithDomain)

    def __init__(self, *args, **kwds):
        IndexedVar.__init__(self, *args, **kwds)


class _VarList(VarList):
    """
    Variable-length indexed variable objects, with local domain attribute.
    """

    def __init__(self, *args, **kwds):
        VarList.__init__(self, *args, **kwds)

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            raise ValueError("Cannot specify indices for a VarList object")
        self._hidden_index = Set()
        IndexedVar.__init__(self, self._hidden_index, **kwargs)
        self._nvars = 0

    def construct(self, data=None):
        """
        Initialize this component
        """
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Constructing variable list %s",self.cname(True))
        self._hidden_index.construct()
        IndexedVar.construct(self, data)

    def add(self):
        """
        Add a variable to this list
        """
        index = self._nvars
        self._hidden_index.add(index)
        vardata = self._data[index] = self._VarData(self)
        self._initialize_members([index])
        self._nvars += 1
        return vardata


class _VarListWithDomain(_VarList):
    """
    Variable-length indexed variable objects, with local domain attribute.
    """

    def __init__(self, *args, **kwds):
        _VarList.__init__(self, *args, **kwds)
        self._domain = None

    #
    # 'domain' is a property so we can ensure that a 'bounds' attribute exists on the
    # domain object.
    #
    @property
    def domain(self):
        """
        Return the domain attribute.
        """
        return self._domain

    @domain.setter
    def domain(self, domain):
        """
        Set the domain attribute to the given value.
        """
        if hasattr(domain, 'bounds'):
            self._domain = domain
        else:
            raise ValueError("%s is not a valid domain. Variable domains must be an instance of "
                             "one of %s an object that declares a method for bounds (like a Pyomo Set)."
                             "Examples: NonNegativeReals, Integers, Binary"
                             % (domain, (RealSet, IntegerSet, BooleanSet)))


register_component(Var, "Decision variables.")
register_component(VarList, "List of decision variables.")

