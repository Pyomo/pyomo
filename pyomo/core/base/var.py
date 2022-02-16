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
import sys
from typing import overload
from weakref import ref as weakref_ref

from pyomo.common.collections import Sequence
from pyomo.common.deprecation import RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NoArgumentGiven, NOTSET
from pyomo.common.timing import ConstructionTimer
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.core.expr.numvalue import (
    NumericValue, value, is_potentially_variable, native_numeric_types,
)
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.indexed_component import (
    IndexedComponent, UnindexedComponent_set, IndexedComponent_NDArrayMixin
)
from pyomo.core.base.initializer import Initializer, DefaultInitializer
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.set import (
    Reals, Binary, Set, SetInitializer,
    real_global_set_ids, integer_global_set_ids,
)
from pyomo.core.base.units_container import units
from pyomo.core.base.util import is_functor

logger = logging.getLogger('pyomo.core')

_inf = float('inf')
_ninf = -_inf
_no_lower_bound = {None, _ninf}
_no_upper_bound = {None, _inf}
_known_global_real_domains = dict(
    [(_, True) for _ in real_global_set_ids] +
    [(_, False) for _ in integer_global_set_ids]
)
_VARDATA_API = (
    # including 'domain' runs afoul of logic in Block._add_implicit_sets()
    # 'domain',
    'bounds', 'lower', 'upper', 'lb', 'ub', 'has_lb', 'has_ub',
    'setlb', 'setub', 'get_units',
    'is_integer', 'is_binary', 'is_continuous', 'is_fixed',
    'fix', 'unfix', 'free', 'set_value', 'value', 'stale', 'fixed',
)


class _VarData(ComponentData, NumericValue):
    """This class defines the abstract interface for a single variable.

    Note that this "abstract" class is not intended to be directly
    instantiated.

    """

    __slots__ = ()

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
        """Returns (or set) the tuple (lower bound, upper bound).

        This returns the current (numeric) values of the lower and upper
        bounds as a tuple.  If there is no bound, returns None (and not
        +/-inf)

        """
        return self.lb, self.ub
    @bounds.setter
    def bounds(self, val):
        self.lower, self.upper = val

    @property
    def lb(self):
        """Return (or set) the numeric value of the variable lower bound."""
        lb = value(self.lower)
        return None if lb == _ninf else lb
    @lb.setter
    def lb(self, val):
        self.lower = val

    @property
    def ub(self):
        """Return (or set) the numeric value of the variable upper bound."""
        ub = value(self.upper)
        return None if ub == _inf else ub
    @ub.setter
    def ub(self, val):
        self.upper = val

    def is_integer(self):
        """Returns True when the domain is a contiguous integer range."""
        _id = id(self.domain)
        if _id in _known_global_real_domains:
            return not _known_global_real_domains[_id]
        _interval = self.domain.get_interval()
        return _interval is not None and _interval[2] == 1

    def is_binary(self):
        """Returns True when the domain is restricted to Binary values."""
        domain = self.domain
        if domain is Binary:
            return True
        if id(domain) in _known_global_real_domains:
            return False
        return domain.get_interval() == (0, 1, 1)

    def is_continuous(self):
        """Returns True when the domain is a continuous real range"""
        _id = id(self.domain)
        if _id in _known_global_real_domains:
            return _known_global_real_domains[_id]
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

    def clear(self):
        self.value = None

    def __call__(self, exception=True):
        """Compute the value of this variable."""
        return self.value

    #
    # Abstract Interface
    #

    def set_value(self, val, skip_validation=False):
        """Set the current variable value."""
        raise NotImplementedError

    @property
    def value(self):
        """Return (or set) the value for this variable."""
        raise NotImplementedError

    @property
    def domain(self):
        """Return (or set) the domain for this variable."""
        raise NotImplementedError

    @property
    def lower(self):
        """Return (or set) an expression for the variable lower bound."""
        raise NotImplementedError

    @property
    def upper(self):
        """Return (or set) an expression for the variable upper bound."""
        raise NotImplementedError

    @property
    def fixed(self):
        """Return (or set) the fixed indicator for this variable.

        Alias for :meth:`is_fixed` / :meth:`fix` / :meth:`unfix`.

        """
        raise NotImplementedError

    @property
    def stale(self):
        """The stale status for this variable.

        Variables are "stale" if their current value was not updated as
        part of the most recent model update.  A "model update" can be
        one of several things: a solver invocation, loading a previous
        solution, or manually updating a non-stale :class:`Var` value.

        Returns
        -------
        bool

        Notes
        -----
        Fixed :class:`Var` objects will be stale after invoking a solver
        (as their value was not updated by the solver).

        Updating a stale :class:`Var` value will not cause other
        variable values to be come stale.  However, updating the first
        non-stale :class:`Var` value adter a solve or solution load
        *will* cause all other variables to be marked as stale

        """
        raise NotImplementedError

    def fix(self, value=NOTSET, skip_validation=False):
        """Fix the value of this variable (treat as nonvariable)

        This sets the :attr:`fixed` indicator to True.  If ``value`` is
        provided, the value (and the ``skip_validation`` flag) are first
        passed to :meth:`set_value()`.

        """
        self.fixed = True
        if value is not NOTSET:
            self.set_value(value, skip_validation)

    def unfix(self):
        """Unfix this variable (treat as variable in solver interfaces)

        This sets the :attr:`fixed` indicator to False.

        """
        self.fixed = False

    def free(self):
        """Alias for :meth:`unfix`"""
        return self.unfix()


class _GeneralVarData(_VarData):
    """This class defines the data for a single variable.

    """

    __slots__ = ('_value', '_lb', '_ub', '_domain', '_fixed', '_stale')

    def __init__(self, component=None):
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
        # The type of the lower and upper bound attributes can either be
        # atomic numeric types in Python, expressions, etc.  Basically,
        # they can be anything that passes an "not
        # is_potentially_variable" test.
        #
        self._lb = None
        self._ub = None
        self._domain = None
        self._fixed = False
        self._stale = 0 # True

    @classmethod
    def copy(cls, src):
        self = cls.__new__(cls)
        self._component = src._component
        self._value = src._value
        self._lb = src._lb
        self._ub = src._ub
        self._domain = src._domain
        self._fixed = src._fixed
        self._stale = src._stale
        return self

    def __getstate__(self):
        state = super(_GeneralVarData, self).__getstate__()
        for i in _GeneralVarData.__slots__:
            state[i] = getattr(self, i)
        state['_stale'] = StaleFlagManager.is_stale(self._stale)
        return state

    def __setstate__(self, state):
        if state.pop('_stale', True):
            state['_stale'] = 0
        else:
            state['_stale'] = StaleFlagManager.get_flag(0)
        super().__setstate__(state)

    #
    # Abstract Interface
    #

    def set_value(self, val, skip_validation=False):
        """Set the current variable value.

        Set the value of this variable.  The incoming value is converted
        to a numeric value (i.e., expressions are evaluated).  If the
        variable has units, the incoming value is converted to the
        correct units before storing the value.  The final value is
        checked against both the variable domain and bounds, and an
        exception is raised if the value is not valid.  Domain and
        bounds checking can be bypassed by setting the ``skip_validation``
        argument to :const:`True`.

        """
        # Special case: setting a variable to None "clears" the variable.
        if val is None:
            self._value = None
            self._stale = 0 # True
            return
        # TODO: generate a warning/error:
        #
        # Check if this Var has units: assigning dimensionless
        # values to a variable with units should be an error
        if type(val) not in native_numeric_types:
            if self.parent_component()._units is not None:
                _src_magnitude = value(val)
                _src_units = units.get_units(val)
                val = units.convert_value(
                    num_value=_src_magnitude, from_units=_src_units,
                    to_units=self.parent_component()._units)
            else:
                val = value(val)

        if not skip_validation:
            if val not in self.domain:
                logger.warning(
                    "Setting Var '%s' to a value `%s` (%s) not in domain %s." %
                    (self.name, val, type(val).__name__, self.domain),
                    extra={'id':'W1001'},
                )
            elif (self._lb is not None and val < value(self._lb)) or (
                    self._ub is not None and val > value(self._ub)):
                logger.warning(
                    "Setting Var '%s' to a numeric value `%s` "
                    "outside the bounds %s." % (self.name, val, self.bounds),
                    extra={'id':'W1002'},
                )

        self._value = val
        self._stale = StaleFlagManager.get_flag(self._stale)

    @property
    def value(self):
        return self._value
    @value.setter
    def value(self, val):
        self.set_value(val)

    @property
    def domain(self):
        return self._domain
    @domain.setter
    def domain(self, domain):
        try:
            self._domain = SetInitializer(domain)(None, None)
        except:
            logger.error(
                "%s is not a valid domain. Variable domains must be an "
                "instance of a Pyomo Set or convertable to a Pyomo Set."
                % (domain,),
                extra={'id': 'E2001'})
            raise

    @_VarData.bounds.getter
    def bounds(self):
        # Custom implementation of _VarData.bounds to avoid unnecessary
        # expression generation and duplicate calls to domain.bounds()
        domain_bounds = self.domain.bounds()
        if self._lb is None:
            lb = domain_bounds[0]
        elif domain_bounds[0] is None:
            lb = value(self._lb)
        else:
            lb = max(value(self._lb), domain_bounds[0])
        if self._ub is None:
            ub = domain_bounds[1]
        elif domain_bounds[1] is None:
            ub = value(self._ub)
        else:
            ub = min(value(self._ub), domain_bounds[1])
        return None if lb == _ninf else lb, None if ub == _inf else ub

    @_VarData.lb.getter
    def lb(self):
        # Custom implementation of _VarData.lb to avoid unnecessary
        # expression generation
        dlb, _ = self.domain.bounds()
        if self._lb is None:
            lb = dlb
        elif dlb is None:
            lb = value(self._lb)
        else:
            lb = max(value(self._lb), dlb)
        return None if lb == _ninf else lb

    @_VarData.ub.getter
    def ub(self):
        # Custom implementation of _VarData.ub to avoid unnecessary
        # expression generation
        _, dub = self.domain.bounds()
        if self._ub is None:
            ub = dub
        elif dub is None:
            ub = value(self._ub)
        else:
            ub = min(value(self._ub), dub)
        return None if ub == _inf else ub

    @property
    def lower(self):
        """Return (or set) an expression for the variable lower bound.

        This returns a (not potentially variable) expression for the
        variable lower bound.  This represents the tighter of the
        current domain and the constant or expression assigned to
        :attr:`lower`.  Note that the expression will NOT automatically
        reflect changes to either the domain or the bound expression
        (e.g., because of assignment to either :attr:`lower` or
        :attr:`domain`).

        """
        dlb, _ = self.domain.bounds()
        if self._lb is None:
            return dlb
        elif dlb is None:
            return self._lb
        # _process_bound() guarantees _lb is not potentially variable
        return NPV_MaxExpression((self._lb, dlb))
    @lower.setter
    def lower(self, val):
        self._lb = self._process_bound(val, 'lower')

    @property
    def upper(self):
        """Return (or set) an expression for the variable upper bound.

        This returns a (not potentially variable) expression for the
        variable upper bound.  This represents the tighter of the
        current domain and the constant or expression assigned to
        :attr:`upper`.  Note that the expression will NOT automatically
        reflect changes to either the domain or the bound expression
        (e.g., because of assignment to either :attr:`upper` or
        :attr:`domain`).

        """
        _, dub = self.domain.bounds()
        if self._ub is None:
            return dub
        elif dub is None:
            return self._ub
        # _process_bound() guarantees _lb is not potentially variable
        return NPV_MinExpression((self._ub, dub))
    @upper.setter
    def upper(self, val):
        self._ub = self._process_bound(val, 'upper')

    def get_units(self):
        """Return the units for this variable entry."""
        # parent_component() returns self if this is scalar, or the owning
        # component if not scalar
        return self.parent_component()._units

    @property
    def fixed(self):
        return self._fixed
    @fixed.setter
    def fixed(self, val):
        self._fixed = bool(val)

    @property
    def stale(self):
        return StaleFlagManager.is_stale(self._stale)
    @stale.setter
    def stale(self, val):
        if val:
            self._stale = 0 # True
        else:
            self._stale = StaleFlagManager.get_flag(0)

    def _process_bound(self, val, bound_type):
        if type(val) in native_numeric_types or val is None:
            # TODO: warn/error: check if this Var has units: assigning
            # a dimensionless value to a united variable should be an error
            pass
        elif is_potentially_variable(val):
            raise ValueError(
                "Potentially variable input of type '%s' supplied as "
                "%s bound for variable '%s' - legal types must be constants "
                "or non-potentially variable expressions."
                % (type(val).__name__, bound_type, self.name))
        else:
            # We want to create an expression and not just convert the
            # current value so that things like mutable Params behave as
            # expected.
            _units = self.parent_component()._units
            if _units is not None:
                val = units.convert(val, to_units=_units)
        return val


@ModelComponentFactory.register("Decision variables.")
class Var(IndexedComponent, IndexedComponent_NDArrayMixin):
    """A numeric variable, which may be defined over an index.

    Args:
        domain (Set or function, optional): A Set that defines valid
            values for the variable (e.g., ``Reals``, ``NonNegativeReals``,
            ``Binary``), or a rule that returns Sets.  Defaults to ``Reals``.
        within (Set or function, optional): An alias for ``domain``.
        bounds (tuple or function, optional): A tuple of ``(lower, upper)``
            bounds for the variable, or a rule that returns tuples.
            Defaults to ``(None, None)``.
        initialize (float or function, optional): The initial value for
            the variable, or a rule that returns initial values.
        rule (float or function, optional): An alias for ``initialize``.
        dense (bool, optional): Instantiate all elements from
            :meth:`index_set` when constructing the Var (True) or just the
            variables returned by ``initialize``/``rule`` (False).  Defaults
            to ``True``.
        units (pyomo units expression, optional): Set the units corresponding
            to the entries in this variable.
        name (str, optional): Name for this component.
        doc (str, optional): Text describing this component.
    """

    _ComponentDataClass = _GeneralVarData

    def __new__(cls, *args, **kwargs):
        if cls is not Var:
            return super(Var, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args)==1):
            return super(Var, cls).__new__(AbstractScalarVar)
        else:
            return super(Var, cls).__new__(IndexedVar)

    @overload
    def __init__(self, *indexes, domain=Reals, within=Reals, bounds=None,
                 initialize=None, rule=None, dense=True, units=None,
                 name=None, doc=None): ...
    
    def __init__(self, *args, **kwargs):
        #
        # Default keyword values
        #
        self._rule_init = Initializer(self._pop_from_kwargs(
            'Var', kwargs, ('rule', 'initialize'), None))
        self._rule_domain = SetInitializer(self._pop_from_kwargs(
            'Var', kwargs, ('domain', 'within'), Reals))
        _bounds_arg = kwargs.pop('bounds', None)
        self._dense = kwargs.pop('dense', True)
        self._units = kwargs.pop('units', None)
        if self._units is not None:
            self._units = units.get_units(self._units)
        #
        # Initialize the base class
        #
        kwargs.setdefault('ctype', Var)
        IndexedComponent.__init__(self, *args, **kwargs)
        #
        # Now that we can call is_indexed(), process bounds initializer
        #
        if self.is_indexed():
            treat_bounds_sequences_as_mappings = not (
                isinstance(_bounds_arg, Sequence)
                and len(_bounds_arg) == 2
                and not isinstance(_bounds_arg[0], Sequence)
            )
        else:
            treat_bounds_sequences_as_mappings = False
            if not self._dense:
                logger.warning(
                    "ScalarVar object '%s': dense=False is not allowed "
                    "for scalar variables; converting to dense=True"
                    % (self.name,))
                self._dense = True
        self._rule_bounds = Initializer(
            _bounds_arg,
            treat_sequences_as_mappings=treat_bounds_sequences_as_mappings
        )

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

    def set_values(self, new_values, skip_validation=False):
        """
        Set the values of a dictionary.

        The default behavior is to validate the values in the
        dictionary.
        """
        for index, new_value in new_values.items():
            self[index].set_value(new_value, skip_validation)

    def get_units(self):
        """Return the units expression for this Var."""
        return self._units

    # TODO: deprecate this?  __getitem__ is generally preferable"
    def add(self, index):
        """Add a variable with a particular index."""
        return self[index]

    def construct(self, data=None):
        """
        Construct the _VarData objects for this variable
        """
        if self._constructed:
            return
        self._constructed=True

        timer = ConstructionTimer(self)
        if is_debug_set(logger):
            logger.debug("Constructing Variable %s" % (self.name,))

        # Note: define 'index' to avoid 'variable referenced before
        # assignment' in the error message generated in the 'except:'
        # block below.
        index = None
        try:
            # We do not (currently) accept data for constructing Variables
            assert data is None

            if not self.index_set().isfinite() and self._dense:
                # Note: if the index is not finite, then we cannot
                # iterate over it.  This used to be fatal; now we
                # just warn
                logger.warning(
                    "Var '%s' indexed by a non-finite set, but declared "
                    "with 'dense=True'.  Reverting to 'dense=False' as "
                    "it is not possible to make this variable dense.  "
                    "This warning can be suppressed by specifying "
                    "'dense=False'" % (self.name,))
                self._dense = False

            if ( self._rule_init is not None and
                 self._rule_init.contains_indices() ):
                # Historically we have allowed Vars to be initialized by
                # a sparse map (i.e., a dict containing only some of the
                # keys).  We will wrap the incoming initializer to map
                # KeyErrors to None
                self._rule_init = DefaultInitializer(
                    self._rule_init, None, KeyError)
                # The index is coming in externally; we need to validate it
                for index in self._rule_init.indices():
                    self[index]
                # If this is a dense object, we need to ensure that it
                # has been filled in.
                if self._dense:
                    for index in self.index_set():
                        if index not in self._data:
                            self._getitem_when_not_present(index)
            elif not self.is_indexed():
                # As there is only a single VarData to populate, just do
                # so and bypass all special-case testing below
                self._getitem_when_not_present(None)
            elif self._dense:
                # Special case: initialize every VarData.  For the
                # initializers that are constant, we can avoid
                # re-calling (and re-validating) the inputs in certain
                # cases.  To support this, we will create the first
                # _VarData and then use it as a template to initialize
                # (constant portions of) every VarData so as to not
                # repeat all the domain/bounds validation.
                try:
                    ref = self._getitem_when_not_present(
                        next(iter(self.index_set())))
                except StopIteration:
                    # Empty index!
                    return
                call_domain_rule = not self._rule_domain.constant()
                call_bounds_rule = self._rule_bounds is not None and (
                        not self._rule_bounds.constant())
                call_init_rule = self._rule_init is not None and (
                    not self._rule_init.constant()
                    # If either the domain or bounds change, then we
                    # need to re-verify the initial value, even if it is
                    # constant:
                    or call_domain_rule or call_bounds_rule
                )
                # Initialize all the component datas with the common data
                for index in self.index_set():
                    self._data[index] = self._ComponentDataClass.copy(ref)
                # Now go back and initialize any index-specific data
                block = self.parent_block()
                if call_domain_rule:
                    for index, obj in self._data.items():
                        # We can directly set the attribute (not the
                        # property) because the SetInitializer ensures
                        # that the value is a proper Set.
                        obj._domain = self._rule_domain(block, index)
                if call_bounds_rule:
                    for index, obj in self._data.items():
                        obj.lower, obj.upper = self._rule_bounds(block, index)
                if call_init_rule:
                    for index, obj in self._data.items():
                        obj.set_value(self._rule_init(block, index))
            else:
                # non-dense indexed var with generic
                # (non-index-containing) initializer: nothing to do
                pass

        except Exception:
            err = sys.exc_info()[1]
            logger.error(
                "Rule failed when initializing variable for "
                "Var %s with index %s:\n%s: %s"
                % (self.name,
                   str(index),
                   type(err).__name__,
                   err))
            raise
        finally:
            timer.report()

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
        parent = self.parent_block()
        # We can directly set the attribute (not the property) because
        # the SetInitializer ensures that the value is a proper Set.
        obj._domain = self._rule_domain(parent, index)
        if self._rule_bounds is not None:
            obj.lower, obj.upper = self._rule_bounds(parent, index)
        if self._rule_init is not None:
            obj.set_value(self._rule_init(parent, index))
        return obj

    #
    # Because we need to do more initialization than simply calling
    # set_value(), we need to override _setitem_when_not_present
    #
    def _setitem_when_not_present(self, index, value=NOTSET):
        if value is self.Skip:
            return None
        try:
            obj = self._getitem_when_not_present(index)
            if value is not NOTSET:
                obj.set_value(value)
        except:
            self._data.pop(index, None)
            raise
        return obj

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
        _GeneralVarData.__init__(self, component=self)
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
            vardata.lower = val

    def setub(self, val):
        """
        Set the upper bound for this variable.
        """
        for vardata in self.values():
            vardata.upper = val

    def fix(self, value=NOTSET, skip_validation=False):
        """Fix all variables in this :class:`IndexedVar` (treat as nonvariable)

        This sets the :attr:`fixed` indicator to True for every variable
        in this IndexedVar.  If ``value`` is provided, the value (and
        the ``skip_validation`` flag) are first passed to
        :meth:`set_value`.

        """
        for vardata in self.values():
            vardata.fix(value, skip_validation)

    def unfix(self):
        """Unfix all variables in this :class:`IndexedVar` (treat as variable)

        This sets the :attr:`_VarData.fixed` indicator to False for
        every variable in this :class:`IndexedVar`.

        """
        for vardata in self.values():
            vardata.unfix()

    def free(self):
        """Alias for :meth:`unfix`"""
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
        # TODO: Ideally we would pass valid arguments to the initializer
        # that we just created.  However at the moment, getting the
        # index() is expensive (see #1228).  As a result, for the moment
        # we will only support constant initializers
        domain = SetInitializer(domain)(None, None)
        for vardata in self.values():
            vardata.domain = domain


@ModelComponentFactory.register("List of decision variables.")
class VarList(IndexedVar):
    """
    Variable-length indexed variable objects used to construct Pyomo models.
    """

    def __init__(self, **kwargs):
        self._starting_index = kwargs.pop('starting_index', 1)
        args = (Set(dimen=1),)
        IndexedVar.__init__(self, *args, **kwargs)

    def construct(self, data=None):
        """Construct this component."""
        if self._constructed:
            return
        # Note: do not set _constructed here, or the super() call will
        # not actually construct the component.

        if is_debug_set(logger):
            logger.debug("Constructing variable list %s", self.name)

        self.index_set().construct()

        # We need to ensure that the indices needed for initialization are
        # added to the underlying implicit set.  We *could* verify that the
        # indices in the initialization dict are all sequential integers,
        # OR we can just add the correct number of sequential integers and
        # then let _validate_index complain when we set the value.
        if self._rule_init is not None and self._rule_init.contains_indices():
            for i, idx in enumerate(self._rule_init.indices()):
                self._index.add(i + self._starting_index)
        super(VarList,self).construct(data)

    def add(self):
        """Add a variable to this list."""
        next_idx = len(self._index) + self._starting_index
        self._index.add(next_idx)
        return self[next_idx]
