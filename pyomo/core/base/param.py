#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['Param']

import sys
import types
import logging
from weakref import ref as weakref_ref

from pyomo.core.base.component import ComponentData, register_component
from pyomo.core.base.indexed_component import IndexedComponent, normalize_index, UnindexedComponent_set
from pyomo.core.base.misc import apply_indexed_rule, apply_parameterized_indexed_rule
from pyomo.core.base.numvalue import NumericValue, native_types, value
from pyomo.core.base.set_types import Any

from six import iteritems, iterkeys, next, itervalues

logger = logging.getLogger('pyomo.core')


class _ParamData(ComponentData, NumericValue):
    """
    This class defines the data for a mutable parameter.

    Constructor Arguments:
        owner       The Param object that owns this data.
        value       The value of this parameter.

    Public Class Attributes:
        value       The numeric value of this variable.
    """

    __slots__ = ('value',)

    def __init__(self, owner, value):
        #
        # The following is equivalent to calling
        # the base ComponentData constructor.
        #
        self._component = weakref_ref(owner)
        #
        # The following is equivalent to calling the
        # base NumericValue constructor.
        #
        self.value = value

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
        self.value = None

    def __call__(self, exception=True):
        """
        Return the value of this object.
        """
        if self.value is None:
            raise ValueError(
                "Error evaluating Param value (%s):\nThe Param value is "
                "undefined and no default value is specified"
                % ( self.name, ))
        return self.value

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

    def _potentially_variable(self):
        """
        Returns False because this object can never reference variables.
        """
        return False

    def _polynomial_degree(self, result):
        """
        Returns 0 because this object can never reference variables.
        """
        return 0

    def __nonzero__(self):
        """Return True if the value is defined and non-zero."""
        if self.value:
            return True
        if self.value is None:
            raise ValueError("Param: value is undefined")
        return False

    __bool__ = __nonzero__

class Param(IndexedComponent):
    """
    A parameter value, which may be defined over an index.

    Constructor Arguments:
       name        The name of this parameter
       index       The index set that defines the distinct parameters.
                     By default, this is None, indicating that there
                     is a single parameter.
       domain      A set that defines the type of values that
                     each parameter must be.
       within      A set that defines the type of values that
                     each parameter must be.
       validate    A rule for validating this parameter w.r.t. data
                     that exists in the model
       default     A scalar, rule, or dictionary that defines default
                     values for this parameter
       initialize  A dictionary or rule for setting up this parameter
                     with existing model data
    """

    DefaultMutable = False

    def __new__(cls, *args, **kwds):
        if cls != Param:
            return super(Param, cls).__new__(cls)
        if args == () or (type(args[0]) is set and args[0] == UnindexedComponent_set and len(args)==1):
            return SimpleParam.__new__(SimpleParam)
        else:
            return IndexedParam.__new__(IndexedParam)

    def __init__(self, *args, **kwd):
        self._rule          = kwd.pop('rule', None )
        self._rule          = kwd.pop('initialize', self._rule )
        self._validate      = kwd.pop('validate', None )
        self.domain         = kwd.pop('domain', Any )
        self.domain         = kwd.pop('within', self.domain )
        self._mutable       = kwd.pop('mutable', Param.DefaultMutable )
        self._default_val   = kwd.pop('default', None )
        self._dense_initialize = kwd.pop('initialize_as_dense', False)
        #
        if 'repn' in kwd:
            logger.error(
                "The 'repn' keyword is not a validate keyword argument for Param")
        #
        if self.domain is None:
            self.domain = Any
        #
        kwd.setdefault('ctype', Param)
        IndexedComponent.__init__(self, *args, **kwd)

    def __len__(self):
        """
        Return the number of component data objects stored by this
        component.  If a default value is specified, then the
        length equals the number of items in the component index.
        """
        if self._default_val is None:
            return len(self._data)
        return len(self._index)

    def __contains__(self, ndx):
        """
        Return true if the index is in the dictionary.  If the default value
        is specified, then all members of the component index are valid.
        """
        if self._default_val is None:
            return ndx in self._data
        return ndx in self._index

    def __iter__(self):
        """
        Iterate over the keys in the dictionary.  If the default value is
        specified, then iterate over all keys in the component index.
        """
        if self._default_val is None:
            return self._data.__iter__()
        return self._index.__iter__()

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
            ans = {}
            for key, param_value in self.iteritems():
                ans[key] = param_value.value
            return ans
        elif not self.is_indexed():
            #
            # The parameter is a scalar, so we need to create a temporary
            # dictionary using the value for this parameter.
            #
            return { None: self.value }
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
                ans[key] = param_value.value
            return ans
        elif not self.is_indexed():
            #
            # The parameter is a scalar, so we need to create a temporary
            # dictionary using the value for this parameter.
            #
            return { None: self.value }
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
            raise RuntimeError("Cannot call store_values method on immutable Param="+ self.name)
        #
        _srcType = type(new_values)
        _isDict = _srcType is dict or ( \
            hasattr(_srcType, '__getitem__')
            and not isinstance(new_values, NumericValue) )
        #
        if check is True:
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
                    try:
                        self._data[index].value = new_value
                    except:
                        self._data[index] = _ParamData(self, new_value)
            else:
                # For scalars, we will choose an approach based on
                # how "dense" the Param is
                if not self._data: # empty
                    for index in self._index:
                        self._data[index] = _ParamData(self, new_values)
                elif len(self._data) == len(self._index):
                    for index in self._index:
                        self._data[index].value = new_values
                else:
                    for index in self._index:
                        if index in self._data:
                            self._data[index].value = new_values
                        else:
                            self._data[index] = _ParamData(self, new_values)
        else:
            #
            # Initialize a scalar
            #
            if _isDict:
                if None not in new_values:
                    raise RuntimeError(
                        "Cannot store value for scalar Param="+
                        self.name+"; no value with index None "
                        "in input new values map.")
                new_values = new_values[None]
            # scalars have to be handled differently
            self._data[None] = new_values

    def _default(self, idx):
        """
        Returns the default component data value
        """
        #
        # Local values
        #
        val = self._default_val
        _default_type = type(val)
        #
        if not self._constructed:
            if idx is None:
                idx_str = '%s' % (self.local_name,)
            else:
                idx_str = '%s[%s]' % (self.local_name, idx,)
            raise ValueError(
                "Error retrieving Param value (%s): This parameter has "
                "not been constructed" % ( idx_str,) )
        if val is None:
            # If the Param is mutable, then it is OK to create a Param
            # implicitly ... the error will be tossed later when someone
            # attempts to evaluate the value of the Param
            if self._mutable:
                if self.is_indexed():
                    self._data[idx] = _ParamData(self, val)
                    #self._raw_setitem(idx, _ParamData(self, val), True)
                else:
                    self._raw_setitem(idx, val)
                return self[idx]

            if self.is_indexed():
                idx_str = '%s[%s]' % (self.name, idx,)
            else:
                idx_str = '%s' % (self.name,)
            raise ValueError(
                    "Error retrieving Param value (%s):\nThe Param value is "
                    "undefined and no default value is specified"
                    % ( idx_str,) )
        #
        # Get the value of the default for this index
        #
        _check_value_domain = True
        if _default_type in native_types:
            #
            # The set_default() method validates the domain of native types, so
            # we can skip the check on the value domain.
            #
            _check_value_domain = False
        elif _default_type is types.FunctionType:
            val = apply_indexed_rule(self, val, self.parent_block(), idx)
        elif hasattr(val, '__getitem__') and not isinstance(val, NumericValue):
            val = val[idx]
        else:
            pass
        if _check_value_domain:
            #
            # Get the value of a numeric value
            #
            if val.__class__ not in native_types:
                if isinstance(val, NumericValue):
                    val = val()
            #
            # Check the domain
            #
            if val not in self.domain:
                raise ValueError(
                    "Invalid default parameter value: %s[%s] = '%s';"
                    " value type=%s.\n\tValue not in parameter domain %s" %
                    (self.name, idx, val, type(val), self.domain.name) )
        #
        # Set the parameter
        #
        if self._mutable:
            if self.is_indexed():
                self._data[idx] = _ParamData(self, val)
                #self._raw_setitem(idx, _ParamData(self, val), True)
            else:
                self._raw_setitem(idx, val)
            return self[idx]
        else:
            #
            # This is kludgy: If the user wants to validate the Param
            # values, we need to validate the default value as well.
            # For Mutable Params, this is easy: setitem will inject the
            # value into _data and then call validate.  For immutable
            # params, we never inject the default into the data
            # dictionary.  This will break validation, as the validation
            # rule is allowed to assume the data is already present
            # (actually, it will die on infinite recursion, as
            # Param.__getitem__() will re-call _default).
            #
            # So, we will do something very inefficient: if we are
            # validating, we will inject the value into the dictionary,
            # call validate, and remove it.
            #
            if self._validate:
                try:
                    self._data[idx] = val
                    self._validateitem(idx, val)
                finally:
                    del self._data[idx]
            return val

    def set_default(self, val):
        """
        Perform error checks and then set the default value for this parameter.

        NOTE: this test will not validate the value of function return values.
        """
        if self._constructed \
                and val is not None \
                and type(val) in native_types \
                and val not in self.domain:
            raise ValueError(
                "Default value (%s) is not valid for Param domain %s" %
                (str(val), self.domain.name))
        self._default_val = val

    def default(self):
        """
        Return the value of the parameter default.

        Possible values:
            None            No default value is provided.
            Numeric         A constant value that is the default value for all
                                undefined parameters.
            Function        f(model, i) returns the value for the default value
                                for parameter i
        """
        return self._default_val

    def _raw_setitem(self, ndx, val):
        """
        The __setitem__ method performs significant
        validation around the input indices, particularly
        when the index value is new.  In various contexts,
        we don't need to incur this overhead (e.g. during
        initialization).  The _raw_setitem assumes the input
        value is in the set native_types
        """
        #
        # Params should contain *values*.  Note that if we just call
        # value(), then that forces the value to be a numeric value.
        # Notably, we allow Params with domain==Any to hold strings, tuples,
        # etc.  The following lets us use NumericValues to initialize
        # Params, but is optimized to check for "known" native types to
        # bypass a potentially expensive isinstance()==False call.
        #
        if val.__class__ not in native_types:
            if isinstance(val, NumericValue):
                val = val()
        #
        # Check if the value is valid within the current domain
        #
        if val not in self.domain:
            raise ValueError(
                "Invalid parameter value: %s[%s] = '%s', value type=%s.\n"
                "\tValue not in parameter domain %s" %
                (self.name, ndx, val, type(val), self.domain.name))
        #
        # Set the value depending on the type of param value.
        #
        if not self.is_indexed():
            self.value = val
        elif self._mutable:
            if ndx in self._data:
                self._data[ndx].value = val
            else:
                self._data[ndx] = _ParamData(self, val)
        else:
            self._data[ndx] = val
        #
        # Execute a validation operation
        #
        if self._validate:
            self._validateitem(ndx, val)

    def _validateitem(self, ndx, val):
        """
        Validate a given input/value pair.
        """
        if not apply_parameterized_indexed_rule(self, self._validate, self.parent_block(), val, ndx):
            raise ValueError(
                "Invalid parameter value: %s[%s] = '%s', value type=%s.\n"
                "\tValue failed parameter validation rule" %
                ( self.name, ndx, val, type(val) ) )

    def __setitem__(self, ndx, val):
        """
        Add a parameter value to the index.
        """
        #
        # TBD: Potential optimization: if we find that updating a Param is
        # more common than setting it in the first place, then first
        # checking the _data and then falling back on the _index *might*
        # be more efficient.
        #
        if self._constructed and not self._mutable:
            raise TypeError(
"""Attempting to set the value of the immutable parameter %s after the
parameter has been constructed.  If you intend to change the value of
this parameter dynamically, please declare the parameter as mutable
[i.e., Param(mutable=True)]""" % (self.name,))
        #
        # Check if we have a valid index.
        # We assume that most calls to this method will send either a
        # scalar or a valid tuple.  So, for efficiency, we will check the
        # index *first*, and only go through the hassle of
        # flattening things if the ndx is not found.
        #
        ndx_ = ()
        if ndx in self._index:
            ndx_ = ndx
        elif normalize_index.flatten:
            ndx = normalize_index(ndx)
            if ndx in self._index:
                ndx_ = ndx
        if ndx_ == ():
            if not self.is_indexed():
                msg = "Error setting parameter value: " \
                      "Cannot treat the scalar Param '%s' as an array" \
                      % ( self.name, )
            else:
                msg = "Error setting parameter value: " \
                      "Index '%s' is not valid for array Param '%s'" \
                      % ( ndx, self.name, )
            raise KeyError(msg)

        # We have a valid index, so do the actual set operation.
        self._raw_setitem(ndx_, val)

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
                self[None] = _init(self.parent_block())
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
                    # somethign that is dict-like, then use the dict to
                    # initialize everything and do not re-vall the
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
                        # At this point, we know the value is specific to
                        # this index (i.e., not likely to be a dict-like
                        # thing), and that the index is valid; so, it is
                        # safe to use _raw_setitem (which will perform all
                        # the domain / validation checking)
                        #
                        self._raw_setitem(idx, val)
                        #
                        # Now iterate over the rest of the index set.
                        #
                        for idx in _iter:
                            self._raw_setitem( idx,
                                apply_indexed_rule( self, _init, self_parent, idx ) )
                        return
                except StopIteration:
                    #
                    # The index set was empty...
                    # The parameter is indexed by an empty set, or an empty set tuple.
                    # Rare, but it has happened.
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
            # TODO: Establish use-cases where we can use default values
            # for sparsity.
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
                self._raw_setitem(idx, _init)
                #
                # Note: the following is safe for both indexed and
                # non-indexed parameters: for non-indexed, the first
                # idx (above) will be None, and the for-loop below
                # will NOT be called.
                #
                if self._mutable:
                    _init = self[idx].value
                    for idx in _iter:
                        self._raw_setitem( idx, _ParamData(self,_init) )
                else:
                    _init = self[idx]
                    for idx in _iter:
                        self._raw_setitem(idx, _init)
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
        if __debug__ and logger.isEnabledFor(logging.DEBUG):   #pragma:nocover
            logger.debug("Constructing Param, name=%s, from data=%s"
                         % ( self.name, str(data) ))
        #
        if self._constructed:
            return
        #
        # If the default value is a simple type, we check it versus
        # the domain.
        #
        val = self._default_val
        if val is not None \
                and type(val) in native_types \
                and val not in self.domain:
            raise ValueError(
                "Default value (%s) is not valid for Param domain %s" %
                (str(val), self.domain.name))
        #
        # Step #1: initialize data from rule value
        #
        if self._rule is not None:
            self._initialize_from(self._rule)
        #
        # Step #2: allow any user-specified (external) data to override
        # the initialization
        #
        if data is not None:
            try:
                for key, val in iteritems(data):
                    self[key] = val
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

        self._constructed = True

        # populate all other indices with default data
        # (avoids calling _set_contains on self._index at runtime)
        if self._dense_initialize:
            self.to_dense_data()

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
            raise RuntimeError("Cannot invoke reconstruct method of immutable param="+self.name)
        IndexedComponent.reconstruct(self, data=data)

    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        return ( [("Size", len(self)),
                  ("Index", self._index \
                       if self._index != UnindexedComponent_set else None),
                  ("Domain", self.domain.name),
                  ("Default", "(function)" if type(self._default_val) \
                       is types.FunctionType else self._default_val),
                  ("Mutable", self._mutable),
                  ],
                 self.sparse_iteritems(),
                 ("Value",),
                 lambda k, v: [ value(v)
                                ]
                 )


class SimpleParam(_ParamData, Param):

    def __init__(self, *args, **kwds):
        Param.__init__(self, *args, **kwds)
        _ParamData.__init__(self, self, kwds.get('default',None))
        self._data[None] = self

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
            return _ParamData.__call__(self, exception=exception)
        if exception:
            raise ValueError( """Evaluating the numeric value of parameter '%s' before the Param has been
            constructed (there is currently no value to return).""" % self.name )

    def set_value(self, value):
        if self._constructed and not self._mutable:
            raise TypeError(
"""Attempting to set the value of the immutable parameter %s after the
parameter has been constructed.  If you intend to change the value of
this parameter dynamically, please declare the parameter as mutable
[i.e., Param(mutable=True)]""" % (self.name,))
        self[None] = value

    def is_constant(self):
        """
        Returns False because this is not a constant in an expression.
        """
        return self._constructed and not self._mutable


class IndexedParam(Param):

    def __call__(self, exception=True):
        """Compute the value of the parameter"""
        if exception:
            msg = 'Cannot compute the value of an array of parameters'
            raise TypeError(msg)

register_component(Param, "Parameter data that is used to define a model instance.")

