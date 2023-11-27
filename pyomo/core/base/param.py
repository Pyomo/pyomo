#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
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
from pyomo.common.pyomo_typing import overload

from pyomo.common.autoslots import AutoSlots
from pyomo.common.deprecation import deprecation_warning, RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.numeric_types import native_types, value as expr_value
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import NumericValue
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import (
    IndexedComponent,
    UnindexedComponent_set,
    IndexedComponent_NDArrayMixin,
)
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.misc import apply_indexed_rule, apply_parameterized_indexed_rule
from pyomo.core.base.set import Reals, _AnySet, SetInitializer
from pyomo.core.base.units_container import units
from pyomo.core.expr import GetItemExpression

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
        "declare the parameter as mutable [i.e., Param(mutable=True)]" % (name,)
    )


class _ImplicitAny(_AnySet):
    """An Any that issues a deprecation warning for non-Real values.

    This is a helper class to implement the deprecation warnings for the
    change of Param's implicit domain from Any to Reals.

    """

    __slots__ = ('_owner',)
    __autoslot_mappers__ = {'_owner': AutoSlots.weakref_mapper}

    def __new__(cls, **kwargs):
        # Strip off owner / kwargs before calling base __new__
        return super().__new__(cls)

    def __init__(self, owner, **kwargs):
        self._owner = weakref_ref(owner)
        super().__init__(**kwargs)
        self._component = weakref_ref(self)
        self.construct()
        # Because this is a "global set", we need to define the _bounds
        # and _interval fields
        object.__setattr__(self, '_parent', None)
        self._bounds = (None, None)
        self._interval = (None, None, None)

    def __contains__(self, val):
        if val not in Reals:
            if self._owner is None or self._owner() is None:
                name = 'Unknown'
            else:
                name = self._owner().name
            deprecation_warning(
                f"Param '{name}' declared with an implicit domain of 'Any'. "
                "The default domain for Param objects is 'Any'.  However, "
                "we will be changing that default to 'Reals' in the "
                "future.  If you really intend the domain of this Param"
                "to be 'Any', you can suppress this warning by explicitly "
                "specifying 'within=Any' to the Param constructor.",
                version='5.6.9',
                remove_in='6.0',
            )
        return True

    # This should "mock up" a global set, so the "name" should always be
    # the local name (without block scope)
    def getname(self, fully_qualified=False, name_buffer=None, relative_to=None):
        return super().getname(False, name_buffer, relative_to)

    # The parent tracks the parent of the owner.  We can't set it
    # directly here because the owner has not been assigned to a block
    # when we create the _ImplicitAny
    @property
    def _parent(self):
        if self._owner is None or self._owner() is None:
            return None
        return self._owner()._parent

    # This is not settable.  However the base classes assume that it is,
    # so we need to define the setter and just ignore the incoming value
    @_parent.setter
    def _parent(self, val):
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
        self._index = NOTSET
        #
        # The following is equivalent to calling the
        # base NumericValue constructor.
        #
        self._value = Param.NoValue

    # Note: because NONE of the slots on this class need to be edited,
    # we don't need to implement a specialized __setstate__ method.

    def clear(self):
        """Clear the data in this component"""
        self._value = Param.NoValue

    # FIXME: ComponentData need to have pointers to their index to make
    # operations like validation efficient.  As it stands now, if
    # set_value is called without specifying an index, this call
    # involves a linear scan of the _data dict.
    def set_value(self, value, idx=NOTSET):
        #
        # If this param has units, then we need to check the incoming
        # value and see if it is "units compatible".  We only need to
        # check here in set_value, because all united Params are
        # required to be mutable.
        #
        _comp = self.parent_component()
        if type(value) in native_types:
            # TODO: warn/error: check if this Param has units: assigning
            # a dimensionless value to a united param should be an error
            pass
        elif _comp._units is not None:
            _src_magnitude = expr_value(value)
            _src_units = units.get_units(value)
            value = units.convert_value(
                num_value=_src_magnitude, from_units=_src_units, to_units=_comp._units
            )

        old_value, self._value = self._value, value
        try:
            _comp._validate_value(idx, value, data=self)
        except:
            self._value = old_value
            raise

    def __call__(self, exception=True):
        """
        Return the value of this object.
        """
        if self._value is Param.NoValue:
            if exception:
                raise ValueError(
                    "Error evaluating Param value (%s):\n\tThe Param value is "
                    "currently set to an invalid value.  This is\n\ttypically "
                    "from a scalar Param or mutable Indexed Param without\n"
                    "\tan initial or default value." % (self.name,)
                )
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


@ModelComponentFactory.register(
    "Parameter data that is used to define a model instance."
)
class Param(IndexedComponent, IndexedComponent_NDArrayMixin):
    """
    A parameter value, which may be defined over an index.

    Constructor Arguments:
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
        name
            Name for this component.
        doc
            Text describing this component.
    """

    DefaultMutable = False
    _ComponentDataClass = _ParamData

    class NoValue(object):
        """A dummy type that is pickle-safe that we can use as the default
        value for Params to indicate that no valid value is present."""

        pass

    def __new__(cls, *args, **kwds):
        if cls != Param:
            return super(Param, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args) == 1):
            return super(Param, cls).__new__(ScalarParam)
        else:
            return super(Param, cls).__new__(IndexedParam)

    @overload
    def __init__(
        self,
        *indexes,
        rule=NOTSET,
        initialize=NOTSET,
        domain=None,
        within=None,
        validate=None,
        mutable=False,
        default=NoValue,
        initialize_as_dense=False,
        units=None,
        name=None,
        doc=None,
    ):
        ...

    def __init__(self, *args, **kwd):
        _init = self._pop_from_kwargs('Param', kwd, ('rule', 'initialize'), NOTSET)
        _domain_rule = self._pop_from_kwargs('Param', kwd, ('domain', 'within'))
        self._validate = kwd.pop('validate', None)
        self._mutable = kwd.pop('mutable', None)
        self._default_val = kwd.pop('default', Param.NoValue)
        self._dense_initialize = kwd.pop('initialize_as_dense', False)
        self._units = kwd.pop('units', None)

        if self._mutable is None:
            if self._units is None:
                self._mutable = Param.DefaultMutable
            else:
                # Params with units *must* be mutable, so that
                # expression simplification does not remove units from
                # the expression.
                self._mutable = True

        kwd.setdefault('ctype', Param)
        IndexedComponent.__init__(self, *args, **kwd)

        # We don't support per-index param domains, so we only need to
        # support constant initializers.
        # (after IndexedComponent.__init__ so we can call parent_block())
        if _domain_rule is None:
            self.domain = _ImplicitAny(owner=self, name='Any')
        else:
            self.domain = SetInitializer(_domain_rule)(self.parent_block(), None)
        # After IndexedComponent.__init__ so we can call is_indexed().
        self._rule = Initializer(
            _init,
            treat_sequences_as_mappings=self.is_indexed(),
            arg_not_specified=NOTSET,
        )

    def __len__(self):
        """
        Return the number of component data objects stored by this
        component.  If a default value is specified, then the
        length equals the number of items in the component index.
        """
        if self._default_val is Param.NoValue:
            return len(self._data)
        return len(self._index_set)

    def __contains__(self, idx):
        """
        Return true if the index is in the dictionary.  If the default value
        is specified, then all members of the component index are valid.
        """
        if self._default_val is Param.NoValue:
            return idx in self._data
        return idx in self._index_set

    # We do not need to override keys(), as the __len__ override will
    # cause the base class keys() to correctly correctly handle default
    # values
    # def keys(self, sort=None):

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
        return list(self._data.keys())

    def sparse_values(self):
        """Return a list of the defined param data objects"""
        return list(self._data.values())

    def sparse_items(self):
        """Return a list (index,data) tuples for defined parameters"""
        return list(self._data.items())

    def sparse_iterkeys(self):
        """Return an iterator for the keys in the defined parameters"""
        return self._data.keys()

    def sparse_itervalues(self):
        """Return an iterator for the defined param data objects"""
        return self._data.values()

    def sparse_iteritems(self):
        """Return an iterator of (index,data) tuples for defined parameters"""
        return self._data.items()

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
            return {key: param_value() for key, param_value in self.items()}
        elif not self.is_indexed():
            #
            # The parameter is a scalar, so we need to create a temporary
            # dictionary using the value for this parameter.
            #
            return {None: self()}
        else:
            #
            # The parameter is not mutable, so iteritems() can be
            # converted into a dictionary containing parameter values.
            #
            return dict(self.items())

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
            return {None: self()}
        else:
            #
            # The parameter is not mutable, so sparse_iteritems() can be
            # converted into a dictionary containing parameter values.
            #
            return dict(self.sparse_iteritems())

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
        _isDict = _srcType is dict or (
            hasattr(_srcType, '__getitem__')
            and not isinstance(new_values, NumericValue)
        )
        #
        if check:
            if _isDict:
                for index, new_value in new_values.items():
                    self[index] = new_value
            else:
                for index in self._index_set:
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
                for index, new_value in new_values.items():
                    if index not in self._data:
                        self._data[index] = _ParamData(self)
                    self._data[index]._value = new_value
            else:
                # For scalars, we will choose an approach based on
                # how "dense" the Param is
                if not self._data:  # empty
                    for index in self._index_set:
                        p = self._data[index] = _ParamData(self)
                        p._value = new_values
                elif len(self._data) == len(self._index_set):
                    for index in self._index_set:
                        self._data[index]._value = new_values
                else:
                    for index in self._index_set:
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
                        "with index None in the new values dict." % (self.name,)
                    )
                new_values = new_values[None]
            # scalars have to be handled differently
            self[None] = new_values

    def set_default(self, val):
        """
        Perform error checks and then set the default value for this parameter.

        NOTE: this test will not validate the value of function return values.
        """
        if (
            self._constructed
            and val is not Param.NoValue
            and type(val) in native_types
            and val not in self.domain
        ):
            raise ValueError(
                "Default value (%s) is not valid for Param %s domain %s"
                % (str(val), self.name, self.domain.name)
            )
        self._default_val = val

    def default(self):
        """
        Return the value of the parameter default.

        Possible values:
            Param.NoValue
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
        if val is Param.NoValue:
            # We should allow the creation of mutable params without
            # a default value, as long as *solving* a model without
            # reasonable values produces an informative error.
            if self._mutable:
                # Note: _ParamData defaults to Param.NoValue
                if self.is_indexed():
                    ans = self._data[index] = _ParamData(self)
                else:
                    ans = self._data[index] = self
                ans._index = index
                return ans
            if self.is_indexed():
                idx_str = '%s[%s]' % (self.name, index)
            else:
                idx_str = '%s' % (self.name,)
            raise ValueError(
                "Error retrieving immutable Param value (%s):\n\tThe Param "
                "value is undefined and no default value is specified." % (idx_str,)
            )

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
            not isinstance(val, NumericValue) or val.is_indexed()
        ):
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
            old_value, self._data[index] = self._data[index], value
            # Because we do not have a _ParamData, we cannot rely on the
            # validation that occurs in _ParamData.set_value()
            try:
                self._validate_value(index, value)
                return value
            except:
                self._data[index] = old_value
                raise

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
                self._index = UnindexedComponent_index
                return self
            elif self._mutable:
                obj = self._data[index] = _ParamData(self)
                obj.set_value(value, index)
                obj._index = index
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

    def _validate_value(self, index, value, validate_domain=True, data=None):
        """
        Validate a given input/value pair.
        """
        #
        # Check if the value is valid within the current domain
        #
        if validate_domain and not value in self.domain:
            if index is NOTSET:
                index = data.index()
            raise ValueError(
                "Invalid parameter value: %s[%s] = '%s', value type=%s.\n"
                "\tValue not in parameter domain %s"
                % (self.name, index, value, type(value), self.domain.name)
            )
        if self._validate:
            if index is NOTSET:
                index = data.index()
            valid = apply_parameterized_indexed_rule(
                self, self._validate, self.parent_block(), value, index
            )
            if not valid:
                raise ValueError(
                    "Invalid parameter value: %s[%s] = '%s', value type=%s.\n"
                    "\tValue failed parameter validation rule"
                    % (self.name, index, value, type(value))
                )

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
        if self._constructed:
            return

        timer = ConstructionTimer(self)
        if is_debug_set(logger):  # pragma:nocover
            logger.debug(
                "Constructing Param, name=%s, from data=%s" % (self.name, str(data))
            )

        if self._units is not None:
            self._units = units.get_units(self._units)
            if not self._mutable:
                logger.warning(
                    "Params with units must be mutable.  "
                    f"Converting Param '{self.name}' to mutable."
                )
                self._mutable = True

        try:
            #
            # If the default value is a simple type, we check it versus
            # the domain.
            #
            val = self._default_val
            if (
                val is not Param.NoValue
                and type(val) in native_types
                and val not in self.domain
            ):
                raise ValueError(
                    "Default value (%s) is not valid for Param %s domain %s"
                    % (str(val), self.name, self.domain.name)
                )
            #
            # Flag that we are in the "during construction" phase
            #
            self._constructed = None
            #
            # Step #1: initialize data from rule value
            #
            self._construct_from_rule_using_setitem()
            #
            # Step #2: allow any user-specified (external) data to override
            # the initialization
            #
            if data is not None:
                try:
                    data_items = data.items()
                except AttributeError:
                    raise ValueError(
                        "Attempting to initialize parameter=%s with data=%s.\n"
                        "\tData type is not a mapping type, and a Mapping is "
                        "expected." % (self.name, str(data))
                    )
            else:
                data_items = iter(())

            try:
                for key, val in data_items:
                    self._setitem_when_not_present(self._validate_index(key), val)
            except:
                msg = sys.exc_info()[1]
                raise RuntimeError(
                    "Failed to set value for param=%s, index=%s, value=%s.\n"
                    "\tsource error message=%s"
                    % (self.name, str(key), str(val), str(msg))
                )
            #
            # Flag that things are fully constructed now (and changing an
            # immutable Param is now an exception).
            #
            self._constructed = True

            # populate all other indices with default data
            # (avoids calling _set_contains on self._index_set at runtime)
            if self._dense_initialize:
                self.to_dense_data()
        finally:
            timer.report()

    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        if self._default_val is Param.NoValue:
            default = "None"  # for backwards compatibility in reporting
        elif type(self._default_val) is types.FunctionType:
            default = "(function)"
        else:
            default = str(self._default_val)
        if self._mutable or not self.is_indexed():
            dataGen = lambda k, v: [v._value]
        else:
            dataGen = lambda k, v: [v]
        headers = [
            ("Size", len(self)),
            ("Index", self._index_set if self.is_indexed() else None),
            ("Domain", self.domain.name),
            ("Default", default),
            ("Mutable", self._mutable),
        ]
        if self._units is not None:
            headers.append(('Units', str(self._units)))
        return (headers, self.sparse_iteritems(), ("Value",), dataGen)


class ScalarParam(_ParamData, Param):
    def __init__(self, *args, **kwds):
        _ParamData.__init__(self, component=self)
        Param.__init__(self, *args, **kwds)
        self._index = UnindexedComponent_index

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
            return super(ScalarParam, self).__call__(exception=exception)
        if exception:
            raise ValueError(
                "Evaluating the numeric value of parameter '%s' before\n\t"
                "the Param has been constructed (there is currently no "
                "value to return)." % (self.name,)
            )

    def set_value(self, value, index=NOTSET):
        if index is NOTSET:
            index = None
        if self._constructed and not self._mutable:
            _raise_modifying_immutable_error(self, index)
        if not self._data:
            self._data[index] = self
        super(ScalarParam, self).set_value(value, index)

    def is_constant(self):
        """Determine if this ScalarParam is constant (and can be eliminated)

        Returns False if either unconstructed or mutable, as it must be kept
        in expressions (as it either doesn't have a value yet or the value
        can change later.
        """
        return self._constructed and not self._mutable


class SimpleParam(metaclass=RenamedClass):
    __renamed__new_class__ = ScalarParam
    __renamed__version__ = '6.0'


class IndexedParam(Param):
    def __call__(self, exception=True):
        """Compute the value of the parameter"""
        if exception:
            raise TypeError(
                'Cannot compute the value of an indexed Param (%s)' % (self.name,)
            )

    # Because IndexedParam can use a non-standard data store (i.e., the
    # values in the _data dict may not be ComponentData objects), we
    # need to override the normal scheme for pre-allocating
    # ComponentData objects during deepcopy.
    def _create_objects_for_deepcopy(self, memo, component_list):
        if self.mutable:
            # Normal indexed object; leverage base implementation
            return super()._create_objects_for_deepcopy(memo, component_list)
        # This is immutable; only add the container (not the _data) to
        # the component_list.
        _new = self.__class__.__new__(self.__class__)
        _ans = memo.setdefault(id(self), _new)
        if _ans is _new:
            component_list.append(self)
        return _ans

    # Because CP supports indirection [the ability to index objects by
    # another (inter) Var] for certain types (including Var), we will
    # catch the normal RuntimeError and return a (variable)
    # GetItemExpression.
    #
    # FIXME: We should integrate this logic into the base implementation
    # of `__getitem__()`, including the recognition / differentiation
    # between potentially variable GetItemExpression objects and
    # "constant" GetItemExpression objects.  That will need to wait for
    # the expression rework [JDS; Nov 22].
    def __getitem__(self, args):
        try:
            return super().__getitem__(args)
        except:
            tmp = args if args.__class__ is tuple else (args,)
            if any(
                hasattr(arg, 'is_potentially_variable')
                and arg.is_potentially_variable()
                for arg in tmp
            ):
                return GetItemExpression((self,) + tmp)
            raise
