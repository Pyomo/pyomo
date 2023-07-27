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

import logging
from weakref import ref as weakref_ref, ReferenceType

from pyomo.common.deprecation import deprecation_warning, RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import unique_component_name, NOTSET
from pyomo.common.timing import ConstructionTimer
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.boolean_value import BooleanValue
from pyomo.core.expr import GetItemExpression
from pyomo.core.expr.numvalue import value
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import IndexedComponent, UnindexedComponent_set
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.set import Set, BooleanSet, Binary
from pyomo.core.base.util import is_functor
from pyomo.core.base.var import Var


logger = logging.getLogger('pyomo.core')

_logical_var_types = {bool, type(None)}


class _DeprecatedImplicitAssociatedBinaryVariable(object):
    __slots__ = ('_boolvar',)

    def __init__(self, boolvar):
        self._boolvar = weakref_ref(boolvar)

    def __call__(self):
        deprecation_warning(
            "Relying on core.logical_to_linear to transform "
            "BooleanVars that do not appear in LogicalConstraints "
            "is deprecated. Please associate your own binaries if "
            "you have BooleanVars not used in logical expressions.",
            version='6.2',
        )

        parent_block = self._boolvar().parent_block()
        new_var = Var(domain=Binary)
        parent_block.add_component(
            unique_component_name(
                parent_block, self._boolvar().local_name + "_asbinary"
            ),
            new_var,
        )
        self._boolvar()._associated_binary = None
        self._boolvar().associate_binary_var(new_var)
        return new_var

    def __getstate__(self):
        return self._boolvar()

    def __setstate__(self, state):
        self._boolvar = weakref_ref(state)


class _BooleanVarData(ComponentData, BooleanValue):
    """
    This class defines the data for a single variable.

    Constructor Arguments:
        component   The BooleanVar object that owns this data.
    Public Class Attributes:
        fixed       If True, then this variable is treated as a
                        fixed constant in the model.
        stale       A Boolean indicating whether the value of this variable is
                        legitimate.  This value is true if the value should
                        be considered legitimate for purposes of reporting or
                        other interrogation.
        value       The numeric value of this variable.
    """

    __slots__ = ()

    def __init__(self, component=None):
        self._component = weakref_ref(component) if (component is not None) else None
        self._index = NOTSET

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

    def set_value(self, val, skip_validation=False):
        """
        Set the value of this numeric object, after
        validating its value. If the 'valid' flag is True,
        then the validation step is skipped.
        """
        # Note that it is basically as fast to check the type as it is
        # to check the skip_validation flag.  Considering that we expect
        # the flag to always be False, we will just ignore it in the
        # name of efficiency.
        if val.__class__ not in _logical_var_types:
            if not skip_validation:
                logger.warning(
                    "implicitly casting '%s' value %s to bool" % (self.name, val)
                )
            val = bool(val)
        self._value = val
        self._stale = StaleFlagManager.get_flag(self._stale)

    def clear(self):
        self.value = None

    def __call__(self, exception=True):
        """Compute the value of this variable."""
        return self.value

    @property
    def value(self):
        """Return the value for this variable."""
        raise NotImplementedError

    @property
    def domain(self):
        """Return the domain for this variable."""
        raise NotImplementedError

    @property
    def fixed(self):
        """Return the fixed indicator for this variable."""
        raise NotImplementedError

    @property
    def stale(self):
        """Return the stale indicator for this variable."""
        raise NotImplementedError

    def fix(self, value=NOTSET, skip_validation=False):
        """Fix the value of this variable (treat as nonvariable)

        This sets the `fixed` indicator to True.  If ``value`` is
        provided, the value (and the ``skip_validation`` flag) are first
        passed to :py:meth:`set_value()`.

        """
        self.fixed = True
        if value is not NOTSET:
            self.set_value(value, skip_validation)

    def unfix(self):
        """Unfix this variable (treat as variable)

        This sets the `fixed` indicator to False.

        """
        self.fixed = False

    def free(self):
        """Alias for :py:meth:`unfix`"""
        return self.unfix()


def _associated_binary_mapper(encode, val):
    if val is None:
        return None
    if encode:
        if val.__class__ is not _DeprecatedImplicitAssociatedBinaryVariable:
            return val()
    else:
        if val.__class__ is not _DeprecatedImplicitAssociatedBinaryVariable:
            return weakref_ref(val)
    return val


class _GeneralBooleanVarData(_BooleanVarData):
    """
    This class defines the data for a single Boolean variable.

    Constructor Arguments:
        component   The BooleanVar object that owns this data.

    Public Class Attributes:
        domain      The domain of this variable.
        fixed       If True, then this variable is treated as a
                        fixed constant in the model.
        stale       A Boolean indicating whether the value of this variable is
                        legitimiate.  This value is true if the value should
                        be considered legitimate for purposes of reporting or
                        other interrogation.
        value       The numeric value of this variable.

    The domain attribute is a property because it is
    too widely accessed directly to enforce explicit getter/setter
    methods and we need to deter directly modifying or accessing
    these attributes in certain cases.
    """

    __slots__ = ('_value', 'fixed', '_stale', '_associated_binary')
    __autoslot_mappers__ = {
        '_associated_binary': _associated_binary_mapper,
        '_stale': StaleFlagManager.stale_mapper,
    }

    def __init__(self, component=None):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - _BooleanVarData
        #   - ComponentData
        #   - BooleanValue
        self._component = weakref_ref(component) if (component is not None) else None
        self._index = NOTSET
        self._value = None
        self.fixed = False
        self._stale = 0  # True

        self._associated_binary = None

    #
    # Abstract Interface
    #

    # value is an attribute

    @property
    def value(self):
        """Return (or set) the value for this variable."""
        return self._value

    @value.setter
    def value(self, val):
        self.set_value(val)

    @property
    def domain(self):
        """Return the domain for this variable."""
        return BooleanSet

    @property
    def stale(self):
        return StaleFlagManager.is_stale(self._stale)

    @stale.setter
    def stale(self, val):
        if val:
            self._stale = 0
        else:
            self._stale = StaleFlagManager.get_flag(0)

    def get_associated_binary(self):
        """Get the binary _VarData associated with this
        _GeneralBooleanVarData"""
        return (
            self._associated_binary() if self._associated_binary is not None else None
        )

    def associate_binary_var(self, binary_var):
        """Associate a binary _VarData to this _GeneralBooleanVarData"""
        if (
            self._associated_binary is not None
            and type(self._associated_binary)
            is not _DeprecatedImplicitAssociatedBinaryVariable
        ):
            raise RuntimeError(
                "Reassociating BooleanVar '%s' (currently associated "
                "with '%s') with '%s' is not allowed"
                % (
                    self.name,
                    self._associated_binary().name
                    if self._associated_binary is not None
                    else None,
                    binary_var.name if binary_var is not None else None,
                )
            )
        if binary_var is not None:
            self._associated_binary = weakref_ref(binary_var)


@ModelComponentFactory.register("Logical decision variables.")
class BooleanVar(IndexedComponent):
    """A logical variable, which may be defined over an index.

    Args:
        initialize (float or function, optional): The initial value for
            the variable, or a rule that returns initial values.
        rule (float or function, optional): An alias for `initialize`.
        dense (bool, optional): Instantiate all elements from
            `index_set()` when constructing the Var (True) or just the
            variables returned by `initialize`/`rule` (False).  Defaults
            to True.
    """

    _ComponentDataClass = _GeneralBooleanVarData

    def __new__(cls, *args, **kwds):
        if cls != BooleanVar:
            return super(BooleanVar, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args) == 1):
            return ScalarBooleanVar.__new__(ScalarBooleanVar)
        else:
            return IndexedBooleanVar.__new__(IndexedBooleanVar)

    def __init__(self, *args, **kwd):
        initialize = kwd.pop('initialize', None)
        initialize = kwd.pop('rule', initialize)
        self._dense = kwd.pop('dense', True)

        kwd.setdefault('ctype', BooleanVar)
        IndexedComponent.__init__(self, *args, **kwd)

        # Allow for functions or functors for value initialization,
        # without confusing with Params, etc (which have a __call__ method).
        #
        self._value_init_value = None
        self._value_init_rule = None

        if is_functor(initialize) and (not isinstance(initialize, BooleanValue)):
            self._value_init_rule = initialize
        else:
            self._value_init_value = initialize

    def flag_as_stale(self):
        """
        Set the 'stale' attribute of every variable data object to True.
        """
        for boolvar_data in self._data.values():
            boolvar_data._stale = 0

    def get_values(self, include_fixed_values=True):
        """
        Return a dictionary of index-value pairs.
        """
        if include_fixed_values:
            return dict((idx, vardata.value) for idx, vardata in self._data.items())
        return dict(
            (idx, vardata.value)
            for idx, vardata in self._data.items()
            if not vardata.fixed
        )

    extract_values = get_values

    def set_values(self, new_values, skip_validation=False):
        """
        Set data values from a dictionary.

        The default behavior is to validate the values in the
        dictionary.
        """
        for index, new_value in new_values.items():
            self[index].set_value(new_value, skip_validation)

    def construct(self, data=None):
        """Construct this component."""
        if is_debug_set(logger):  # pragma:nocover
            try:
                name = str(self.name)
            except:
                name = type(self)
            logger.debug(
                "Constructing Variable, name=%s, from data=%s" % (name, str(data))
            )

        if self._constructed:
            return
        timer = ConstructionTimer(self)
        self._constructed = True

        #
        # Construct _BooleanVarData objects for all index values
        #
        if not self.is_indexed():
            self._data[None] = self
            self._initialize_members((None,))
        elif self._dense:
            # This loop is optimized for speed with pypy.
            # Calling dict.update((...) for ...) is roughly
            # 30% slower
            self_weakref = weakref_ref(self)
            for ndx in self._index_set:
                cdata = self._ComponentDataClass(component=None)
                cdata._component = self_weakref
                self._data[ndx] = cdata
                # NOTE: This is a special case where a key, value pair is added
                # to the _data dictionary without calling
                # _getitem_when_not_present, which is why we need to set the
                # index here.
                cdata._index = ndx
            self._initialize_members(self._index_set)
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
            obj = self._data[index] = self._ComponentDataClass(component=self)
        self._initialize_members((index,))
        obj._index = index
        return obj

    def _setitem_when_not_present(self, index, value):
        """Perform the fundamental component item creation and storage.

        BooleanVar overrides the default implementation from IndexedComponent
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
                    val = apply_indexed_rule(
                        self, self._value_init_rule, self._parent(), key
                    )
                    val = value(val)
                    vardata.set_value(val)
            else:
                val = self._value_init_rule(self._parent())
                val = value(val)
                self.set_value(val)
        elif self._value_init_value is not None:
            #
            # Initialize values with a value
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

    def _pprint(self):
        """
        Print component information.
        """
        return (
            [
                ("Size", len(self)),
                ("Index", self._index_set if self.is_indexed() else None),
            ],
            self._data.items(),
            ("Value", "Fixed", "Stale"),
            lambda k, v: [v.value, v.fixed, v.stale],
        )


class ScalarBooleanVar(_GeneralBooleanVarData, BooleanVar):

    """A single variable."""

    def __init__(self, *args, **kwd):
        _GeneralBooleanVarData.__init__(self, component=self)
        BooleanVar.__init__(self, *args, **kwd)
        self._index = UnindexedComponent_index

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
            return _GeneralBooleanVarData.value.fget(self)
        raise ValueError(
            "Accessing the value of variable '%s' "
            "before the Var has been constructed (there "
            "is currently no value to return)." % (self.name)
        )

    @value.setter
    def value(self, val):
        """Set the value for this variable."""
        if self._constructed:
            return _GeneralBooleanVarData.value.fset(self, val)
        raise ValueError(
            "Setting the value of variable '%s' "
            "before the Var has been constructed (there "
            "is currently nothing to set." % (self.name)
        )

    @property
    def domain(self):
        return _GeneralBooleanVarData.domain.fget(self)

    def fix(self, value=NOTSET, skip_validation=False):
        """
        Set the fixed indicator to True. Value argument is optional,
        indicating the variable should be fixed at its current value.
        """
        if self._constructed:
            return _GeneralBooleanVarData.fix(self, value, skip_validation)
        raise ValueError(
            "Fixing variable '%s' "
            "before the Var has been constructed (there "
            "is currently nothing to set)." % (self.name)
        )

    def unfix(self):
        """Sets the fixed indicator to False."""
        if self._constructed:
            return _GeneralBooleanVarData.unfix(self)
        raise ValueError(
            "Freeing variable '%s' "
            "before the Var has been constructed (there "
            "is currently nothing to set)." % (self.name)
        )


class SimpleBooleanVar(metaclass=RenamedClass):
    __renamed__new_class__ = ScalarBooleanVar
    __renamed__version__ = '6.0'


class IndexedBooleanVar(BooleanVar):
    """An array of variables."""

    def fix(self, value=NOTSET, skip_validation=False):
        """Fix all variables in this IndexedBooleanVar (treat as nonvariable)

        This sets the `fixed` indicator to True for every variable in
        this IndexedBooleanVar.  If ``value`` is provided, the value
        (and the ``skip_validation`` flag) are first passed to
        :py:meth:`set_value()`.

        """
        for boolean_vardata in self.values():
            boolean_vardata.fix(value, skip_validation)

    def unfix(self):
        """Unfix all variables in this IndexedBooleanVar (treat as variable)

        This sets the `fixed` indicator to False for every variable in
        this IndexedBooleanVar.

        """
        for boolean_vardata in self.values():
            boolean_vardata.unfix()

    def free(self):
        """Alias for :py:meth:`unfix`"""
        return self.unfix()

    @property
    def domain(self):
        return BooleanSet

    # Because Emma wants crazy things... (Where crazy things are the ability to
    # index BooleanVars by other (integer) Vars and integer-valued
    # expressions--a thing you can do in Constraint Programming.)
    def __getitem__(self, args):
        tmp = args if args.__class__ is tuple else (args,)
        if any(
            hasattr(arg, 'is_potentially_variable') and arg.is_potentially_variable()
            for arg in tmp
        ):
            return GetItemExpression((self,) + tmp)
        return super().__getitem__(args)


@ModelComponentFactory.register("List of logical decision variables.")
class BooleanVarList(IndexedBooleanVar):
    """
    Variable-length indexed variable objects used to construct Pyomo models.
    """

    def __init__(self, **kwargs):
        self._starting_index = kwargs.pop('starting_index', 1)
        args = (Set(),)
        IndexedBooleanVar.__init__(self, *args, **kwargs)

    def construct(self, data=None):
        """Construct this component."""
        if is_debug_set(logger):
            logger.debug("Constructing variable list %s", self.name)

        # We need to ensure that the indices needed for initialization are
        # added to the underlying implicit set.  We *could* verify that the
        # indices in the initialization dict are all sequential integers,
        # OR we can just add the correct number of sequential integers and
        # then let _validate_index complain when we set the value.
        if self._value_init_value.__class__ is dict:
            for i in range(len(self._value_init_value)):
                self._index_set.add(i + self._starting_index)
        super(BooleanVarList, self).construct(data)
        # Note that the current Var initializer silently ignores
        # initialization data that is not in the underlying index set.  To
        # ensure that at least here all initialization data is added to the
        # VarList (so we get potential domain errors), we will re-set
        # everything.
        if self._value_init_value.__class__ is dict:
            for k, v in self._value_init_value.items():
                self[k] = v

    def add(self):
        """Add a variable to this list."""
        next_idx = len(self._index_set) + self._starting_index
        self._index_set.add(next_idx)
        return self[next_idx]
