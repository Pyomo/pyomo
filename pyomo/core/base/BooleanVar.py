#Is this correct? #Take domain out
__all__ = ['BooleanVar', '_BooleanVarData', '_GeneralBooleanVarData', 'BooleanVarList', 'SimpleBooleanVar']

#copied from var.py

import logging
from weakref import ref as weakref_ref

#0-0 what do the fowlling imports do?
from pyomo.common.timing import ConstructionTimer 
from pyomo.core.base.logicalvalue import LogicalValue, value, is_fixed
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
    __slots__ = () #0-0 this is the analogous version, following for test only
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

    '''
    The following function is deleted
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
    __slots__ = ('_value', '_domain', 'fixed', 'stale')

    def __init__(self, domain=Reals, component=None):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - _BooleanVarData
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
        # the SimplBooleanVar constructor will fail
        if hasattr(domain, 'bounds'):
            self._domain = domain
        elif domain is not None:
            raise ValueError(
                "%s is not a valid domain. Variable domains must be an "
                "instance of one of %s, or an object that declares a method "
                "for bounds (like a Pyomo Set). Examples: NonNegativeReals, "
                "Integers, Binary" % (domain, (RealSet, IntegerSet, BooleanSet)))

    def __getstate__(self):
        state = super(_GeneralBooleanVarData, self).__getstate__()
        for i in _GeneralBooleanVarData.__slots__:
            state[i] = getattr(self, i)
        return state

    # copied from var.py
    # Note: None of the slots on this class need to be edited, so we
    # don't need to implement a specialized __setstate__ method, and
    # can quietly rely on the super() class's implementation.

    #
    # Abstract Interface
    #

    # value is an attribute
    # 0-0 A mark for debugging, access value
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

    #0-0 Is it just an alias of the function?
    free = unfix



@ModelComponentFactory.register("Decision variables.")
class BooleanVar(IndexedComponent):
    _ComponentDataClass = _GeneralBooleanVarData
    #The following part is rewrite because of indetation error

    def __new__(cls, *args, **kwds):
        if cls != BooleanVar:
            return super(BooleanVar, cls).__new__(cls)
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


        kwd.setdefault('ctype', BooleanVar)
        IndexedComponent.__init__(self, *args, **kwd)
        # Determine if the domain argument is a functor or other object
        self._domain_init_value = None
        self._domain_init_rule = None
        if is_functor(domain):
            self._domain_init_rule = domain
        else:
            self._domain_init_value = domain
        #copied form var.py
        # Allow for functions or functors for value initialization,
        # without confusing with Params, etc (which have a __call__ method).
        #
        self._value_init_value = None
        self._value_init_rule = None
        #0-0 Does _value_init_rule work with LogicalValue?
        #Also O deleted the bound part from line 530-537 in var.py
        if  is_functor(initialize) and (not isinstance(initialize,LogicalValue)):
            self._value_init_rule = initialize
        else:
            self._value_init_value = initialize

    def is_expression_type(self):
        """Returns False because this is not an expression"""
        return False

    def flag_as_stale(self):
        """
        Set the 'stale' attribute of every variable data object to True.
        """
        for boolvar_data in itervalues(self._data):
            boolvar_data.stale = True

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

    #0-0 Again, is this just the python way of aliasing functions
    extract_values = get_values

    def set_values(self, new_values, valid=False):
        """
        copied from var.py
        Set the values of a dictionary.

        The default behavior is to validate the values in the
        dictionary.
        """
        for index, new_value in iteritems(new_values):
            self[index].set_value(new_value, valid)

    #0-0 Really confused. What does this do?
    #Almost a complete copy
    def construct(self, data=None):
        """Construct this component."""
        if __debug__ and logger.isEnabledFor(logging.DEBUG):   #pragma:nocover
            try:
                name = str(self.name)
            except:
                # Some Var components don't have a name yet, so just use
                # the type
                name = type(self)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Constructing Variable, name=%s, from data=%s"
                    % (name, str(data)))

        if self._constructed:
            return
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

    #Do we care about rewriting what's in there already?
    def add(self, index):
        """Add a variable with a particular index."""
        return self[index]

    #
    # This method must be defined on subclasses of
    # IndexedComponent that support implicit definition
    # 0-0 Not really sure in what way this function is useful
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

    #Also not really 
    def _initialize_members(self, init_set):
        """Initialize variable data for all indices in a set."""
        # TODO: determine if there is any advantage to supporting init_set.
        # Preliminary tests indicate that there isn't a significant speed
        # difference to using the set form (used in dense vector
        # construction).  Getting rid of it could simplify _setitem and
        # this method.
        # #0-0 Do I need to care about the above?
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
                # Copied
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
            # 0-0 Is this the Python way of type checking?
            if self._value_init_value.__class__ is dict:
                for key in init_set:
                    # Skip indices that are not in the
                    # dictionary. This arises when
                    # initializing VarList objects with a
                    # dictionary.
                    # What does this continue do here?
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

        # Bounds are deleted.

    def _pprint(self):
        """
            Print component information.
            What do we want to print in this case.
        """
        pass
        '''
        return ( [("Size", len(self)),
                  ("Index", self._index if self.is_indexed() else None),
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
                 '''

class SimpleBooleanVar(_GeneralBooleanVarData, BooleanVar):
    
    """Copied: A single variable."""
    #0-0 Does the following automatically store information in the same 
    #memory block?
    def __init__(self, *args, **kwd):
        _GeneralBooleanVarData.__init__(self,
                                 domain=None,
                                 component=self)
        BooleanVar.__init__(self, *args, **kwd)

    """
    # The following is copied from var.py
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
    """

    @property
    def value(self):
        """Return the value for this variable."""
        if self._constructed:
            return _GeneralBooleanVarData.value.fget(self)
        raise ValueError(
            "Accessing the value of variable '%s' "
            "before the Var has been constructed (there "
            "is currently no value to return)."
            % (self.name))

    @value.setter
    def value(self, val):
        """Set the value for this variable."""
        if self._constructed:
            return _GeneralBooleanVarData.value.fset(self, val)
        raise ValueError(
            "Setting the value of variable '%s' "
            "before the Var has been constructed (there "
            "is currently nothing to set."
            % (self.name))


    @property
    def domain(self):
        return _GeneralBooleanVarData.domain.fget(self)

    @domain.setter
    def domain(self, domain):
        """Set the domain for this variable."""
        if self._constructed:
            return _GeneralBooleanVarData.domain.fset(self, domain)
        raise ValueError(
            "Setting the domain of variable '%s' "
            "before the Var has been constructed (there "
            "is currently nothing to set."
            % (self.name))

    # Delete parts about bounds

    def fix(self, *val):
        """
        Set the fixed indicator to True. Value argument is optional,
        indicating the variable should be fixed at its current value.
        """
        if self._constructed:
            return _GeneralBooleanVarData.fix(self, *val)
        raise ValueError(
            "Fixing variable '%s' "
            "before the Var has been constructed (there "
            "is currently nothing to set)."
            % (self.name))

    def unfix(self):
        """Sets the fixed indicator to False."""
        if self._constructed:
            return _GeneralBooleanVarData.unfix(self)
        raise ValueError(
            "Freeing variable '%s' "
            "before the Var has been constructed (there "
            "is currently nothing to set)."
            % (self.name))

    free=unfix

class IndexedBooleanVar(BooleanVar):
    """An array of variables."""

    def fix(self, *val):
        """
        Set the fixed indicator to True. Value argument is optional,
        indicating the variable should be fixed at its current value.
        """
        for boolean_vardata in itervalues(self):
            boolean_vardata.fix(*val)

    def unfix(self):
        """Sets the fixed indicator to False."""
        for boolean_vardata in itervalues(self):
            boolean_vardata.unfix()

    @property
    def domain(self):
        raise AttributeError(
            "The domain is not an attribute for IndexedBooleanVar. It "
            "can be set for all indices using this property setter, "
            "but must be accessed for individual variables in this container.")
    @domain.setter
    def domain(self, domain):
        """Sets the domain for all variables in this container."""
        for boolean_vardata in itervalues(self):
            boolean_vardata.domain = domain

    free=unfix
    

@ModelComponentFactory.register("List of decision variables.")
class BooleanVarList(IndexedBooleanVar):
    """
    Variable-length indexed variable objects used to construct Pyomo models.
    """

    def __init__(self, **kwds):
        #kwds['dense'] = False
        args = (Set(),)
        IndexedBooleanVar.__init__(self, *args, **kwds)

    def construct(self, data=None):
        """Construct this component."""
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Constructing variable list %s", self.name)

        # We need to ensure that the indices needed for initialization are
        # added to the underlying implicit set.  We *could* verify that the
        # indices in the initialization dict are all sequential integers,
        # OR we can just add the correct number of sequential integers and
        # then let _validate_index complain when we set the value.
        if self._value_init_value.__class__ is dict:
            for i in xrange(len(self._value_init_value)):
                self._index.add(i+1)
        super(BooleanVarList,self).construct(data)
        # Note that the current Var initializer silently ignores
        # initialization data that is not in the underlying index set.  To
        # ensure that at least here all initialization data is added to the
        # VarList (so we get potential domain errors), we will re-set
        # everything.
        if self._value_init_value.__class__ is dict:
            for k,v in iteritems(self._value_init_value):
                self[k] = v

    def add(self):
        """Add a variable to this list."""
        next_idx = len(self._index) + 1
        self._index.add(next_idx)
        return self[next_idx]




