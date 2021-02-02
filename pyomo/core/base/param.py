#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['Param']

import sys
import types
import logging
from weakref import ref as weakref_ref

from pyomo.common.deprecation import deprecation_warning
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NoArgumentGiven
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.plugin import ModelComponentFactory
from pyomo.core.base.component import ComponentData
from pyomo.core.base.indexed_component import IndexedComponent, \
    UnindexedComponent_set
from pyomo.core.base.misc import apply_indexed_rule, apply_parameterized_indexed_rule
from pyomo.core.base.numvalue import NumericValue, native_types, value
from pyomo.core.base.set_types import Any, Reals
from pyomo.core.base.units_container import units

from six import iteritems, iterkeys, next, itervalues

logger = logging.getLogger('pyomo.core')

def _raise_modifying_immutable_error(obj, index):
    if obj.is_indexed():
        name = "%s[%s]" % (obj.name, index)
    else:
        name = obj.name
    raise TypeError(
        "Attempting to set the value of the immutable parameter "
        "%s after the parameter has been constructed.  If you intend "
        "to change the value of this parameter dynamically, please "
        "declare the parameter as mutable [i.e., Param(mutable=True)]"
        % (name,))

class _ImplicitAny(Any.__class__):
    """An Any that issues a deprecation warning for non-Real values.

    This is a helper class to implement the deprecation warnings for the
    change of Param's implicit domain from Any to Reals.

    """
    def __new__(cls, **kwds):
        return super(_ImplicitAny, cls).__new__(cls)

    def __init__(self, owner, **kwds):
        super(_ImplicitAny, self).__init__(**kwds)
        self._owner = weakref_ref(owner)
        self._component = weakref_ref(self)
        self.construct()

    def __getstate__(self):
        state = super(_ImplicitAny, self).__getstate__()
        state['_owner'] = None if self._owner is None else self._owner()
        return state

    def __setstate__(self, state):
        _owner = state.pop('_owner')
        super(_ImplicitAny, self).__setstate__(state)
        self._owner = None if _owner is None else weakref_ref(_owner)

    def __deepcopy__(self, memo):
        return super(Any.__class__, self).__deepcopy__(memo)

    def __contains__(self, val):
        if val not in Reals:
            deprecation_warning(
                "The default domain for Param objects is 'Any'.  However, "
                "we will be changing that default to 'Reals' in the "
                "future.  If you really intend the domain of this Param (%s) "
                "to be 'Any', you can suppress this warning by explicitly "
                "specifying 'within=Any' to the Param constructor."
                % ('Unknown' if self._owner is None else self._owner().name,),
                version='5.6.9', remove_in='6.0')
        return True

class _NotValid(object):
    """A dummy type that is pickle-safe that we can use as the default
    value for Params to indicate that no valid value is present."""
    pass


class _ParamData(ComponentData, NumericValue):
    """
    This class defines the data for a mutable parameter.

    Constructor Arguments:
        owner       The Param object that owns this data.
        value       The value of this parameter.

    Public Class Attributes:
        value       The numeric value of this variable.
    """

    __slots__ = ('_value',)

    def __init__(self, component):
        #
        # The following is equivalent to calling
        # the base ComponentData constructor.
        #
        self._component = weakref_ref(component)
        #
        # The following is equivalent to calling the
        # base NumericValue constructor.
        #
        self._value = _NotValid

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        state = super(_ParamData, self).__getstate__()
        for i in _ParamData.__slots__:
            state[i] = getattr(self, i)
        return state

    # Note: because NONE of the slots on this class need to be edited,
    # we don't need to implement a specialized __setstate__ method.

    def clear(self):
        """Clear the data in this component"""
        self._value = _NotValid

    # FIXME: ComponentData need to have pointers to their index to make
    # operations like validation efficient.  As it stands now, if
    # set_value is called without specifying an index, this call
    # involves a linear scan of the _data dict.
    def set_value(self, value, idx=NoArgumentGiven):
        self._value = value
        if idx is NoArgumentGiven:
            idx = self.index()
        self.parent_component()._validate_value(idx, value)

    def __call__(self, exception=True):
        """
        Return the value of this object.
        """
        if self._value is _NotValid:
            if exception:
                raise ValueError(
                    "Error evaluating Param value (%s):\n\tThe Param value is "
                    "currently set to an invalid value.  This is\n\ttypically "
                    "from a scalar Param or mutable Indexed Param without\n"
                    "\tan initial or default value."
                    % ( self.name, ))
            else:
                return None
        return self._value

    @property
    def value(self):
        """Return the value for this variable."""
        return self()
    @value.setter
    def value(self, val):
        """Set the value for this variable."""
        self.set_value(val)

    def get_units(self):
        """Return the units for this ParamData"""
        return self.parent_component()._units

    def is_fixed(self):
        """
        Returns True because this value is fixed.
        """
        return True

    def is_constant(self):
        """
        Returns False because this is not a constant in an expression.
        """
        return False

    def is_parameter_type(self):
        """
        Returns True because this is a parameter object.
        """
        return True

    def _compute_polynomial_degree(self, result):
        """
        Returns 0 because this object can never reference variables.
        """
        return 0

    def __nonzero__(self):
        """Return True if the value is defined and non-zero."""
        return bool(self())

    __bool__ = __nonzero__


@ModelComponentFactory.register("Parameter data that is used to define a model instance.")
class Param(IndexedComponent):
    """
    A parameter value, which may be defined over an index.

    Constructor Arguments:
        name        
            The name of this parameter
        index       
            The index set that defines the distinct parameters. By default, 
            this is None, indicating that there is a single parameter.
        domain      
            A set that defines the type of values that each parameter must be.
        within      
            A set that defines the type of values that each parameter must be.
        validate    
            A rule for validating this parameter w.r.t. data that exists in 
            the model
        default     
            A scalar, rule, or dictionary that defines default values for 
            this parameter
        initialize  
            A dictionary or rule for setting up this parameter with existing 
            model data
        unit: pyomo unit expression
            An expression containing the units for the parameter
        mutable: `boolean`
            Flag indicating if the value of the parameter may change between
            calls to a solver. Defaults to `False`
    """

    DefaultMutable = False

    def __new__(cls, *args, **kwds):
        if cls != Param:
            return super(Param, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args)==1):
            return SimpleParam.__new__(SimpleParam)
        else:
            return IndexedParam.__new__(IndexedParam)

    def __init__(self, *args, **kwd):
        self._rule          = kwd.pop('rule', _NotValid )
        self._rule          = kwd.pop('initialize', self._rule )
        self._validate      = kwd.pop('validate', None )
        self.domain         = kwd.pop('domain', None )
        self.domain         = kwd.pop('within', self.domain )
        self._mutable       = kwd.pop('mutable', Param.DefaultMutable )
        self._default_val   = kwd.pop('default', _NotValid )
        self._dense_initialize = kwd.pop('initialize_as_dense', False)
        self._units         = kwd.pop('units', None)
        if self._units is not None:
            self._units = units.get_units(self._units)
            self._mutable = True
        #
        if 'repn' in kwd:
            logger.error(
                "The 'repn' keyword is not a validate keyword argument for Param")
        #
        if self.domain is None:
            self.domain = _ImplicitAny(owner=self, name='Any')
        #
        kwd.setdefault('ctype', Param)
        IndexedComponent.__init__(self, *args, **kwd)

    def __len__(self):
        """
        Return the number of component data objects stored by this
        component.  If a default value is specified, then the
        length equals the number of items in the component index.
        """
        if self._default_val is _NotValid:
            return len(self._data)
        return len(self._index)

    def __contains__(self, idx):
        """
        Return true if the index is in the dictionary.  If the default value
        is specified, then all members of the component index are valid.
        """
        if self._default_val is _NotValid:
            return idx in self._data
        return idx in self._index

    def __iter__(self):
        """
        Iterate over the keys in the dictionary.  If the default value is
        specified, then iterate over all keys in the component index.
        """
        if self._default_val is _NotValid:
            return self._data.__iter__()
        return self._index.__iter__()

    @property
    def mutable(self):
        return self._mutable

    def get_units(self):
        """Return the units for this ParamData"""
        return self._units

    #
    # These are "sparse equivalent" access / iteration methods that
    # only loop over the defined data.
    #

    def sparse_keys(self):
        """Return a list of keys in the defined parameters"""
        return list(iterkeys(self._data))

    def sparse_values(self):
        """Return a list of the defined param data objects"""
        return list(itervalues(self._data))

    def sparse_items(self):
        """Return a list (index,data) tuples for defined parameters"""
        return list(iteritems(self._data))

    def sparse_iterkeys(self):
        """Return an iterator for the keys in the defined parameters"""
        return iterkeys(self._data)

    def sparse_itervalues(self):
        """Return an iterator for the defined param data objects"""
        return itervalues(self._data)

    def sparse_iteritems(self):
        """Return an iterator of (index,data) tuples for defined parameters"""
        return iteritems(self._data)

    def extract_values(self):
        """
        A utility to extract all index-value pairs defined for this
        parameter, returned as a dictionary.

        This method is useful in contexts where key iteration and
        repeated __getitem__ calls are too expensive to extract
        the contents of a parameter.
        """
        if self._mutable:
            #
            # The parameter is mutable, parameter data are ParamData types.
            # Thus, we need to create a temporary dictionary that contains the
            # values from the ParamData objects.
            #
            return {key:param_value() for key,param_value in self.iteritems()}
        elif not self.is_indexed():
            #
            # The parameter is a scalar, so we need to create a temporary
            # dictionary using the value for this parameter.
            #
            return { None: self() }
        else:
            #
            # The parameter is not mutable, so iteritems() can be
            # converted into a dictionary containing parameter values.
            #
            return dict( self.iteritems() )

    def extract_values_sparse(self):
        """
        A utility to extract all index-value pairs defined with non-default
        values, returned as a dictionary.

        This method is useful in contexts where key iteration and
        repeated __getitem__ calls are too expensive to extract
        the contents of a parameter.
        """
        if self._mutable:
            #
            # The parameter is mutable, parameter data are ParamData types.
            # Thus, we need to create a temporary dictionary that contains the
            # values from the ParamData objects.
            #
            ans = {}
            for key, param_value in self.sparse_iteritems():
                ans[key] = param_value()
            return ans
        elif not self.is_indexed():
            #
            # The parameter is a scalar, so we need to create a temporary
            # dictionary using the value for this parameter.
            #
            return { None: self() }
        else:
            #
            # The parameter is not mutable, so sparse_iteritems() can be
            # converted into a dictionary containing parameter values.
            #
            return dict( self.sparse_iteritems() )

    def store_values(self, new_values, check=True):
        """
        A utility to update a Param with a dictionary or scalar.

        If check=True, then both the index and value
        are checked through the __getitem__ method.  Using check=False
        should only be used by developers!
        """
        if not self._mutable:
            _raise_modifying_immutable_error(self, '*')
        #
        _srcType = type(new_values)
        _isDict = _srcType is dict or ( \
            hasattr(_srcType, '__getitem__')
            and not isinstance(new_values, NumericValue) )
        #
        if check:
            if _isDict:
                for index, new_value in iteritems(new_values):
                    self[index] = new_value
            else:
                for index in self._index:
                    self[index] = new_values
            return
        #
        # The argument check is False, so we bypass almost all of the
        # Param logic for ensuring data integrity.
        #
        if self.is_indexed():
            if _isDict:
                # It is possible that the Param is sparse and that the
                # index is not already in the _data dict.  As these
                # cases are rare, we will recover from the exception
                # instead of incurring the penalty of checking.
                for index, new_value in iteritems(new_values):
                    if index not in self._data:
                        self._data[index] = _ParamData(self)
                    self._data[index]._value = new_value
            else:
                # For scalars, we will choose an approach based on
                # how "dense" the Param is
                if not self._data: # empty
                    for index in self._index:
                        p = self._data[index] = _ParamData(self)
                        p._value = new_values
                elif len(self._data) == len(self._index):
                    for index in self._index:
                        self._data[index]._value = new_values
                else:
                    for index in self._index:
                        if index not in self._data:
                            self._data[index] = _ParamData(self)
                        self._data[index]._value = new_values
        else:
            #
            # Initialize a scalar
            #
            if _isDict:
                if None not in new_values:
                    raise RuntimeError(
                        "Cannot store value for scalar Param %s:\n\tNo value "
                        "with index None in the new values dict."
                        % (self.name,))
                new_values = new_values[None]
            # scalars have to be handled differently
            self[None] = new_values

    def set_default(self, val):
        """
        Perform error checks and then set the default value for this parameter.

        NOTE: this test will not validate the value of function return values.
        """
        if self._constructed \
                and val is not _NotValid \
                and type(val) in native_types \
                and val not in self.domain:
            raise ValueError(
                "Default value (%s) is not valid for Param %s domain %s" %
                (str(val), self.name, self.domain.name))
        self._default_val = val

    def default(self):
        """
        Return the value of the parameter default.

        Possible values:
            None            
                No default value is provided.
            Numeric         
                A constant value that is the default value for all undefined 
                parameters.
            Function        
                f(model, i) returns the value for the default value for 
                parameter i
        """
        return self._default_val

    def _getitem_when_not_present(self, index):
        """
        Returns the default component data value
        """
        #
        # Local values
        #
        val = self._default_val
        if val is _NotValid:
            # We should allow the creation of mutable params without
            # a default value, as long as *solving* a model without
            # reasonable values produces an informative error.
            if self._mutable:
                # Note: _ParamData defaults to _NotValid
                ans = self._data[index] = _ParamData(self)
                return ans
            if self.is_indexed():
                idx_str = '%s[%s]' % (self.name, index,)
            else:
                idx_str = '%s' % (self.name,)
            raise ValueError(
                "Error retrieving immutable Param value (%s):\n\tThe Param "
                "value is undefined and no default value is specified."
                % ( idx_str,) )

        _default_type = type(val)
        _check_value_domain = True
        if _default_type in native_types:
            #
            # The set_default() method validates the domain of native types, so
            # we can skip the check on the value domain.
            #
            _check_value_domain = False
        elif _default_type is types.FunctionType:
            val = apply_indexed_rule(self, val, self.parent_block(), index)
        elif hasattr(val, '__getitem__') and (
                not isinstance(val, NumericValue) or val.is_indexed() ):
            # Things that look like Dictionaries should be allowable.  This
            # includes other IndexedComponent objects.
            val = val[index]
        else:
            # this is something simple like a non-indexed component
            pass

        #
        # If the user wants to validate values, we need to validate the
        # default value as well. For Mutable Params, this is easy:
        # _setitem_impl will inject the value into _data and
        # then call validate.
        #
        if self._mutable:
            return self._setitem_when_not_present(index, val)
        #
        # For immutable params, we never inject the default into the data
        # dictionary.  This will break validation, as the validation rule is
        # allowed to assume the data is already present (actually, it will
        # die on infinite recursion, as Param.__getitem__() will re-call
        # _getitem_when_not_present).
        #
        # So, we will do something very inefficient: if we are
        # validating, we will inject the value into the dictionary,
        # call validate, and remove it.
        #
        if _check_value_domain or self._validate:
            try:
                self._data[index] = val
                self._validate_value(index, val, _check_value_domain)
            finally:
                del self._data[index]

        return val

    def _setitem_impl(self, index, obj, value):
        """The __setitem__ method performs significant validation around the
        input indices, particularly when the index value is new.  In
        various contexts, we don't need to incur this overhead
        (e.g. during initialization).  The _setitem_impl
        assumes the input value is in the set native_types

        """
        #
        # We need to ensure that users don't override the value for immutable
        # parameters.
        #
        if self._constructed and not self._mutable:
            _raise_modifying_immutable_error(self, index)
        #
        # Params should contain *values*.  Note that if we just call
        # value(), then that forces the value to be a numeric value.
        # Notably, we allow Params with domain==Any to hold strings, tuples,
        # etc.  The following lets us use NumericValues to initialize
        # Params, but is optimized to check for "known" native types to
        # bypass a potentially expensive isinstance()==False call.
        #
        if value.__class__ not in native_types:
            if isinstance(value, NumericValue):
                value = value()
        #
        # Set the value depending on the type of param value.
        #
        if self._mutable:
            obj.set_value(value, index)
            return obj
        else:
            self._data[index] = value
            # Because we do not have a _ParamData, we cannot rely on the
            # validation that occurs in _ParamData.set_value()
            self._validate_value(index, value)
            return value

    def _setitem_when_not_present(self, index, value, _check_domain=True):
        #
        # We need to ensure that users don't override the value for immutable
        # parameters.
        #
        if self._constructed and not self._mutable:
            _raise_modifying_immutable_error(self, index)
        #
        # Params should contain *values*.  Note that if we just call
        # value(), then that forces the value to be a numeric value.
        # Notably, we allow Params with domain==Any to hold strings, tuples,
        # etc.  The following lets us use NumericValues to initialize
        # Params, but is optimized to check for "known" native types to
        # bypass a potentially expensive isinstance()==False call.
        #
        if value.__class__ not in native_types:
            if isinstance(value, NumericValue):
                value = value()

        #
        # Set the value depending on the type of param value.
        #
        try:
            if index is None and not self.is_indexed():
                self._data[None] = self
                self.set_value(value, index)
                return self
            elif self._mutable:
                obj = self._data[index] = _ParamData(self)
                obj.set_value(value, index)
                return obj
            else:
                self._data[index] = value
                # Because we do not have a _ParamData, we cannot rely on the
                # validation that occurs in _ParamData.set_value()
                self._validate_value(index, value, _check_domain)
                return value
        except:
            del self._data[index]
            raise


    def _validate_value(self, index, value, validate_domain=True):
        """
        Validate a given input/value pair.
        """
        #
        # Check if the value is valid within the current domain
        #
        if validate_domain and not value in self.domain:
            raise ValueError(
                "Invalid parameter value: %s[%s] = '%s', value type=%s.\n"
                "\tValue not in parameter domain %s" %
                (self.name, index, value, type(value), self.domain.name))
        if self._validate:
            valid = apply_parameterized_indexed_rule(
                self, self._validate, self.parent_block(), value, index )
            if not valid:
                raise ValueError(
                    "Invalid parameter value: %s[%s] = '%s', value type=%s.\n"
                    "\tValue failed parameter validation rule" %
                    ( self.name, index, value, type(value) ) )

    def _initialize_from(self, _init):
        """
        Initialize data from a rule or data
        """
        _init_type = type(_init)
        _isDict = _init_type is dict

        if _isDict or _init_type in native_types:
            #
            # We skip the other tests if we have a dictionary or constant
            # value, as these are the most common cases.
            #
            pass

        elif _init_type is types.FunctionType:
            #
            # Initializing from a function
            #
            if not self.is_indexed():
                #
                # A scalar value has a single value.
                # We call __setitem__, which does checks on the value.
                #
                self._setitem_when_not_present(None, _init(self.parent_block()))
                return
            else:
                #
                # An indexed parameter, where we call the function for each
                # index.
                #
                self_parent = self.parent_block()
                #
                try:
                    #
                    # Create an iterator for the indices.  We assume that
                    # it returns flattened tuples. Otherwise,
                    # the validation process is far too expensive.
                    #
                    _iter = self._index.__iter__()
                    idx = next(_iter)
                    #
                    # If a function returns a dict (or
                    # dict-like thing), then we initialize the Param object
                    # by reseting _init and _isDict
                    #
                    # Note that this logic allows the user to call a
                    # function without an index
                    #
                    val = apply_indexed_rule(self, _init, self_parent, idx)

                    #
                    # The following is a simplification of the main
                    # _initialize_from logic.  The idea is that if the
                    # function returns a scalar-like thing, use it to
                    # initialize this index and re-call the function for
                    # the next value.  However, if the function returns
                    # something that is dict-like, then use the dict to
                    # initialize everything and do not re-call the
                    # initialize function.
                    #
                    # Note: while scalar components are technically
                    # "dict-like", we will treat them as scalars and
                    # re-call the initialize function.
                    #
                    _dict_like = False
                    if type(val) is dict:
                        _dict_like = True
                    elif isinstance(val, IndexedComponent):
                        _dict_like = val.is_indexed()
                    elif hasattr(val, '__getitem__') \
                            and not isinstance(val, NumericValue):
                        try:
                            for x in _init:
                                _init.__getitem__(x)
                            _dict_like = True
                        except:
                            pass

                    if _dict_like:
                        _init = val
                        _isDict = True
                    else:
                        #
                        # At this point, we know the value is specific
                        # to this index (i.e., not likely to be a
                        # dict-like thing), and that the index is valid;
                        # so, it is safe to use _setitem_impl
                        # (which will perform all the domain /
                        # validation checking)
                        #
                        self._setitem_when_not_present(idx, val)
                        #
                        # Now iterate over the rest of the index set.
                        #
                        for idx in _iter:
                            self._setitem_when_not_present(
                                idx, apply_indexed_rule(
                                    self, _init, self_parent, idx))
                        return
                except StopIteration:
                    #
                    # The index set was empty... The parameter is indexed by
                    # an empty set, or an empty set tuple. Rare, but it has
                    # happened.
                    #
                    return

        elif isinstance(_init, NumericValue):
            #
            # Reduce NumericValues to scalars.  This allows us to treat
            # scalar components as numbers and not
            # as indexed components with a index set of [None]
            #
            _init = _init()

        elif isinstance(_init, IndexedComponent):
            #
            # Ideally, we want to reduce IndexedComponents to
            # a dict, but without "densifying" it.  However, since
            # there is no way to (easily) get the default value, we
            # will take the "less surprising" route of letting the
            # source become dense, so that we get the expected copy.
            #
            # TBD: Are there use-cases where we want to maintain sparsity?
            #
            _init_keys_len = sum(1 for _ in _init.keys())
            sparse_src = len(_init) != _init_keys_len
            tmp = dict( _init.iteritems() )
            if sparse_src and len(_init) == _init_keys_len:
                logger.warning("""
Initializing Param %s using a sparse mutable indexed component (%s).
This has resulted in the conversion of the source to dense form.
""" % (self.name, _init.name))
            _init = tmp
            _isDict = True

        #
        # If the _init is not a native dictionary, but it
        # behaves like one (that is, it could be converted to a
        # dict with "dict((key,_init[key]) for key in _init)"),
        # then we will treat it as such
        #
        # TODO: Establish a use-case for this.  This iteration is
        # expensive.
        #
        if not _isDict and hasattr(_init, '__getitem__'):
            try:
                _isDict = True
                for x in _init:
                    _init.__getitem__(x)
            except:
                _isDict = False
        #
        # Now, we either have a scalar or a dictionary
        #
        if _isDict:
            #
            # Because this is a user-specified dictionary, we
            # must use the normal (expensive) __setitem__ route
            # so that the individual indices are validated.
            #
            for key in _init:
                self[key] = _init[key]
        else:
            try:
                #
                # A constant is being supplied as a default to
                # a parameter.  This happens for indexed parameters,
                # particularly when dealing with mutable parameters.
                #
                # We look at the first iteration index separately to
                # to validate the value against the domain once.
                #
                _iter = self._index.__iter__()
                idx = next(_iter)
                self._setitem_when_not_present(idx, _init)
                #
                # Note: the following is safe for both indexed and
                # non-indexed parameters: for non-indexed, the first
                # idx (above) will be None, and the for-loop below
                # will NOT be called.
                #
                if self._mutable:
                    _init = self[idx]._value
                    for idx in _iter:
                        self._setitem_when_not_present(idx, _init)
                else:
                    _init = self[idx]
                    for idx in _iter:
                        self._setitem_when_not_present(
                            idx, _init, _check_domain=False )
            except StopIteration:
                #
                # The index set was empty...
                # The parameter is indexed by an empty set, or an empty set tuple.
                # Rare, but it has happened.
                #
                pass

    def construct(self, data=None):
        """
        Initialize this component.

        A parameter is constructed using the initial data or
        the data loaded from an external source.  We first
        set all the values based on self._rule, and then
        allow the data dictionary to overwrite anything.

        Note that we allow an undefined Param value to be
        constructed.  We throw an exception if a user tries
        to use an uninitialized Param.
        """
        if is_debug_set(logger):   #pragma:nocover
            logger.debug("Constructing Param, name=%s, from data=%s"
                         % ( self.name, str(data) ))
        #
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        #
        # If the default value is a simple type, we check it versus
        # the domain.
        #
        val = self._default_val
        if val is not _NotValid \
                and type(val) in native_types \
                and val not in self.domain:
            raise ValueError(
                "Default value (%s) is not valid for Param %s domain %s" %
                (str(val), self.name, self.domain.name))
        #
        # Flag that we are in the "during construction" phase
        #
        self._constructed = None
        #
        # Step #1: initialize data from rule value
        #
        if self._rule is not _NotValid:
            self._initialize_from(self._rule)
        #
        # Step #2: allow any user-specified (external) data to override
        # the initialization
        #
        if data is not None:
            try:
                for key, val in iteritems(data):
                    self._setitem_when_not_present(
                        self._validate_index(key), val)
            except Exception:
                msg = sys.exc_info()[1]
                if type(data) is not dict:
                   raise ValueError(
                       "Attempting to initialize parameter=%s with data=%s.\n"
                       "\tData type is not a dictionary, and a dictionary is "
                       "expected." % (self.name, str(data)) )
                else:
                    raise RuntimeError(
                        "Failed to set value for param=%s, index=%s, value=%s."
                        "\n\tsource error message=%s"
                        % (self.name, str(key), str(val), str(msg)) )
        #
        # Flag that things are fully constructed now (and changing an
        # inmutable Param is now an exception).
        #
        self._constructed = True

        # populate all other indices with default data
        # (avoids calling _set_contains on self._index at runtime)
        if self._dense_initialize:
            self.to_dense_data()
        timer.report()

    def reconstruct(self, data=None):
        """
        Reconstruct this parameter object.  This is particularly useful
        for cases where an initialize rule is provided.  An initialize
        rule can return an expression that is a function of other
        parameters, so reconstruction can account for changes in dependent
        parameters.

        Only mutable parameters can be reconstructed.  Otherwise, the
        changes would not be propagated into expressions in objectives
        or constraints.
        """
        if not self._mutable:
            raise RuntimeError(
                "Cannot invoke reconstruct method of immutable Param %s"
                % (self.name,))
        IndexedComponent.reconstruct(self, data=data)

    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        if self._default_val is _NotValid:
            default = "None" # for backwards compatibility in reporting
        elif type(self._default_val) is types.FunctionType:
            default = "(function)"
        else:
            default = str(self._default_val)
        if self._mutable or not self.is_indexed():
            dataGen = lambda k, v: [ v._value, ]
        else:
            dataGen = lambda k, v: [ v, ]
        return ( [("Size", len(self)),
                  ("Index", self._index if self.is_indexed() else None),
                  ("Domain", self.domain.name),
                  ("Default", default),
                  ("Mutable", self._mutable),
                  ],
                 self.sparse_iteritems(),
                 ("Value",),
                 dataGen,
                 )


class SimpleParam(_ParamData, Param):

    def __init__(self, *args, **kwds):
        Param.__init__(self, *args, **kwds)
        _ParamData.__init__(self, component=self)

    #
    # Since this class derives from Component and Component.__getstate__
    # just packs up the entire __dict__ into the state dict, there s
    # nothng special that we need to do here.  We will just defer to the
    # super() get/set state.  Since all of our get/set state methods
    # rely on super() to traverse the MRO, this will automatically pick
    # up both the Component and Data base classes.
    #

    def __call__(self, exception=True):
        """
        Return the value of this parameter.
        """
        if self._constructed:
            if not self._data:
                if self._mutable:
                    # This will trigger populating the _data dict and setting
                    # the _default, if applicable
                    self[None]
                else:
                    # Immutable Param defaults never get added to the
                    # _data dict
                    return self[None]
            return super(SimpleParam, self).__call__(exception=exception)
        if exception:
            raise ValueError(
                "Evaluating the numeric value of parameter '%s' before\n\t"
                "the Param has been constructed (there is currently no "
                "value to return)." % (self.name,) )

    def set_value(self, value, index=NoArgumentGiven):
        if index is NoArgumentGiven:
            index = None
        if self._constructed and not self._mutable:
            _raise_modifying_immutable_error(self, index)
        if not self._data:
            self._data[index] = self
        super(SimpleParam, self).set_value(value, index)

    def is_constant(self):
        """Determine if this SimpleParam is constant (and can be eliminated)

        Returns False if either unconstructed or mutable, as it must be kept
        in expressions (as it either doesn't have a value yet or the value
        can change later.
        """
        return self._constructed and not self._mutable


class IndexedParam(Param):

    def __call__(self, exception=True):
        """Compute the value of the parameter"""
        if exception:
            raise TypeError('Cannot compute the value of an indexed Param (%s)'
                            % (self.name,) )

