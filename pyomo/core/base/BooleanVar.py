__all__ = ['Var', '_VarData', '_GeneralVarData', 'VarList', 'SimpleVar']

#copied from var.py

import logging
from weakref import ref as weakref_ref

#0-0 what do the fowlling imports do?
from pyomo.common.timing import ConstructionTimer 
from pyomo.core.base.logical import LogicalValue, value, is_fixed
from pyomo.core.base.set_types import BooleanSet, IntegerSet, RealSet, Reals # needed?
from pyomo.core.base.plugin import ModelComponentFactory
from pyomo.core.base.component import ComponentData
from pyomo.core.base.indexed_component import IndexedComponent, UnindexedComponent_set
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.sets import Set
from pyomo.core.base.util import is_functor

#what exactly does logger do?
logger = logging.getLogger('pyomo.core')

class _BooleanVarData(ComponentData, LogicalValue):
	#the following is copied from var.py

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
        # 0-0 kinda confused, is it like a logicalvalue alone can creates _BooleanVarData
        # These lines represent in-lining of the
        # following constructors:
        #   - ComponentData
        #   - LogicalValue
        self._component = weakref_ref(component) if (component is not None) \
                          else None

    #
    # Interface
    #
    '''
    the following functions are deleted
    def has_lb(self):
    def has_ub(self):
    @property
    def bounds(self):
        """Returns the tuple (lower bound, upper bound)."""
        return (self.lb, self.ub)
    @bounds.setter
    def bounds(self, val):
       
    def is_integer(self):
	'''

    def is_binary(self):
        return True
        '''
        if self.domain.__class__ is BooleanSet:
            return True
        return isinstance(self.domain, BooleanSet)

		'''

	'''The following function is deleted
	def is_continuous(self):
	'''

	#The follwing functions are copied from var.py

    def is_fixed(self):
        """Returns True if this variable is fixed, otherwise returns False."""
        return self.fixed

    def is_constant(self):
        """Returns False because this is not a constant in an expression."""
        return False

    def is_parameter_type(self):
        """Returns False because this is not a parameter object."""
        return False

    def is_variable_type(self):
        """Returns True because this is a variable."""
        return True

    def is_expression_type(self):
        """Returns False because this is not an expression"""
        return False

    def is_potentially_variable(self):
        """Returns True because this is a variable."""
        return True

    '''
    #This function is deleted
    def _compute_polynomial_degree(self, result):
	'''

	#0-0 what about the functions that do not have the attribute value
    def set_value(self, val, valid=False):
        """
        Set the value of this numeric object, after
        validating its value. If the 'valid' flag is True,
        then the validation step is skipped.
        """
        if valid or self._valid_value(val):
            self.value = val
            self.stale = False
    #0-0 Not really sure what this do
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

    ''' 
    #The following function is deleted 
    def __nonzero__(self)
    __bool__ = __nonzero__
    '''

    def __call__(self, exception=True):
        """Compute the value of this variable."""
        return self.value

    '''The following functions are copied from var.py'''
    @property
    def value(self):
        """Return the value for this variable."""
        raise NotImplementedError

    @property
    def domain(self):
        """Return the domain for this variable."""
        raise NotImplementedError
    '''
    The following functions are deleted
    @property
    def lb(self):
        """Return the lower bound for this variable."""
        raise NotImplementedError

    @property
    def ub(self):
        """Return the upper bound for this variable."""
        raise NotImplementedError
	'''

    @property
    def fixed(self):
        """Return the fixed indicator for this variable."""
        raise NotImplementedError

    @property
    def stale(self):
        """Return the stale indicator for this variable."""
        raise NotImplementedError
   
    '''
    The following functions are deleted
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
    '''

   	#0-0 why do we need this?
    def fix(self, *val):
        """
        Set the fixed indicator to True. Value argument is optional,
        indicating the variable should be fixed at its current value.
        """
        raise NotImplementedError

    def unfix(self):
        """Sets the fixed indicator to False."""
        raise NotImplementedError

    #copied from var.py
    free=unfix

    def to_string(self, verbose=None, labeler=None, smap=None, compute_values=False):
        """Return the component name"""
        if self.fixed and compute_values:
            try:
                return str(self())
            except:
                pass
        if smap:
            return smap.getSymbol(self, labeler)
        return self.name
  
class _GeneralBooleanVarData(_BooleanVarData):
    
   	'''
   	#0-0
   	Everything related to '_lb' and '_ub' are deleted
   	What should we do about domains, ignore that?
   	'''
    __slots__ = ('_value', '_domain', 'fixed', 'stale')

    def __init__(self, domain=Reals, component=None):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - _VarData
        #   - ComponentData
        #   - LogicalValue
        self._component = weakref_ref(component) if (component is not None) \
                          else None
        self._value = None
        #
        # Basically, they can be anything that passes an "is_fixed" test.
        # 
        self._domain = None
        self.fixed = False
        self.stale = True
        #0-0 don't know what this mean yet.
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

    # fixed is an attribute

    # stale is an attribute

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



@ModelComponentFactory.register("Decision variables.")
class BooleanVar(IndexedComponent):
	_ComponenetDataCalss = GeneralVarData

	#0-0 Not really sure about this
	def __new__(cls, *args, **kwds):
		if cls != Var:
			return super(Var, cls).__new__(cls)
		#0-0 what exactly is UnindexedComponent_set?
        if not args or (args[0] is UnindexedComponent_set and len(args)==1):
            return SimpleBooleanVar.__new__(SimpleBooleanVar)
        else:
            return IndexedBooleanVar.__new__(IndexedBooleanVar)

    def __init__(self, *args, **kwd):
    	#0-0 Is this popping from a dict, also should we keep bounds?
    	initialize = kwd.pop('initialize', None)
        initialize = kwd.pop('rule', initialize)
        domain = kwd.pop('within', Reals)
        domain = kwd.pop('domain', domain)
        #bounds = kwd.pop('bounds', None)
        self._dense = kwd.pop('dense', True)





















