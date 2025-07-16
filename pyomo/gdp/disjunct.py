#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
import sys
import types

from math import fabs
from weakref import ref as weakref_ref

from pyomo.common.autoslots import AutoSlots
from pyomo.common.deprecation import deprecation_warning, RenamedClass
from pyomo.common.errors import PyomoException
from pyomo.common.log import is_debug_set
from pyomo.common.numeric_types import native_logical_types, native_types
from pyomo.common.modeling import unique_component_name, NOTSET
from pyomo.common.timing import ConstructionTimer
from pyomo.core import (
    ModelComponentFactory,
    Binary,
    Block,
    Constraint,
    ConstraintList,
    Any,
    LogicalConstraint,
    LogicalConstraintList,
    BooleanValue,
    ScalarBooleanVar,
    ScalarVar,
    value,
)
from pyomo.core.base.component import (
    ActiveComponent,
    ActiveComponentData,
    ComponentData,
)
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.block import BlockData
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.indexed_component import ActiveIndexedComponent
from pyomo.core.expr.expr_common import ExpressionType

logger = logging.getLogger('pyomo.gdp')

_rule_returned_none_error = """Disjunction '%s': rule returned None.

Disjunction rules must return an iterable containing Disjuncts or
individual expressions, or Disjunction.Skip.  The most common cause of
this error is forgetting to include the "return" statement at the end of
your rule.
"""


class GDP_Error(PyomoException):
    """Exception raised while processing GDP Models"""


class AutoLinkedBinaryVar(ScalarVar):
    """A binary variable implicitly linked to its equivalent Boolean variable.

    Basic operations like setting values and fixing/unfixing this
    variable are also automatically applied to the associated Boolean
    variable.

    As this class is only intended to provide a deprecation path for
    Disjunct.indicator_var, it only supports Scalar instances and does
    not support indexing.
    """

    INTEGER_TOLERANCE = 0.001

    __autoslot_mappers__ = {'_associated_boolean': AutoSlots.weakref_mapper}

    def __init__(self, boolean_var=None):
        super().__init__(domain=Binary)
        self._associated_boolean = weakref_ref(boolean_var)

    def get_associated_boolean(self):
        return self._associated_boolean()

    def set_value(self, val, skip_validation=False, _propagate_value=True):
        super().set_value(val, skip_validation)
        if not _propagate_value:
            return
        # Map the incoming (numeric) value to bool/None
        if val is None:
            bool_val = None
        elif fabs(val - 0.5) < 0.5 - AutoLinkedBinaryVar.INTEGER_TOLERANCE:
            bool_val = None
        else:
            bool_val = bool(int(val + 0.5))
        # (Setting _propagate_value prevents infinite recursion.)
        self.get_associated_boolean().set_value(
            bool_val, skip_validation, _propagate_value=False
        )

    def fix(self, value=NOTSET, skip_validation=False):
        super().fix(value, skip_validation)
        bool_var = self.get_associated_boolean()
        if not bool_var.is_fixed():
            bool_var.fix()

    def unfix(self):
        super().unfix()
        bool_var = self.get_associated_boolean()
        if bool_var.is_fixed():
            bool_var.unfix()


class AutoLinkedBooleanVar(ScalarBooleanVar):
    """A Boolean variable implicitly linked to its equivalent binary variable.

    This class provides a deprecation path for GDP.  Originally,
    Disjunct indicator_var was a binary variable.  This simplified early
    transformations.  However, with the introduction of a proper logical
    expression system, the mathematically correct approach is for the
    Disjunct's indicator_var attribute to be a proper BooleanVar.  As
    part of the transition, indicator_var attributes are instances of
    AutoLinkedBooleanVar, which allow the indicator_var to be used in
    logical expressions, but also implicitly converted (with deprecation
    warning) into their equivalent binary variable.

    Basic operations like setting values and fixing/unfixing this
    variable are also automatically applied to the associated binary
    variable.

    As this class is only intended to provide a deprecation path for
    Disjunct.indicator_var, it only supports Scalar instances and does
    not support indexing.

    """

    def as_numeric(self):
        """Return the binary variable associated with this Boolean variable.

        This method returns the associated binary variable along with a
        deprecation warning about using the Boolean variable in a numeric
        context.

        """
        deprecation_warning(
            "Implicit conversion of the Boolean indicator_var '%s' to a "
            "binary variable is deprecated and will be removed.  "
            "Either express constraints on indicator_var using "
            "LogicalConstraints or work with the associated binary "
            "variable from indicator_var.get_associated_binary()" % (self.name,),
            version='6.0',
        )
        return self.get_associated_binary()

    def as_binary(self):
        return self.as_numeric()

    def set_value(self, val, skip_validation=False, _propagate_value=True):
        # super() does not work as expected for properties; we will call
        # the property setter explicitly.
        super().set_value(val, skip_validation)
        if not _propagate_value:
            return
        # Fetch the current value (so we know it has already been cast
        # to None/bool)
        val = self.value
        if val is not None:
            val = int(val)
        # (Setting _propagate_value prevents infinite recursion.)
        self.get_associated_binary().set_value(
            val, skip_validation, _propagate_value=False
        )

    def fix(self, value=NOTSET, skip_validation=False):
        super().fix(value, skip_validation)
        bin_var = self.get_associated_binary()
        if not bin_var.is_fixed():
            bin_var.fix()

    def unfix(self):
        super().unfix()
        bin_var = self.get_associated_binary()
        if bin_var.is_fixed():
            bin_var.unfix()

    #
    # Duck-type the numeric expression API, but route the conversion to
    # Binary through as_numeric to generate the deprecation warning
    #

    @property
    def bounds(self):
        return self.as_numeric().bounds

    @bounds.setter
    def bounds(self, value):
        self.as_numeric().bounds = value

    @property
    def lb(self):
        return self.as_numeric().lb

    @lb.setter
    def lb(self, value):
        self.as_numeric().lb = value

    @property
    def ub(self):
        return self.as_numeric().ub

    @ub.setter
    def ub(self, value):
        self.as_numeric().ub = value

    def __abs__(self):
        return self.as_numeric().__abs__()

    def __float__(self):
        return self.as_numeric().__float__()

    def __int__(self):
        return self.as_numeric().__int__()

    def __neg__(self):
        return self.as_numeric().__neg__()

    def __bool__(self):
        return self.as_numeric().__bool__()

    def __pos__(self):
        return self.as_numeric().__pos__()

    def get_units(self):
        return self.as_numeric().get_units()

    def has_lb(self):
        return self.as_numeric().has_lb()

    def has_ub(self):
        return self.as_numeric().has_ub()

    def is_binary(self):
        return self.as_numeric().is_binary()

    def is_continuous(self):
        return self.as_numeric().is_continuous()

    def is_integer(self):
        return self.as_numeric().is_integer()

    def polynomial_degree(self):
        return self.as_numeric().polynomial_degree()

    def __le__(self, arg):
        return self.as_numeric().__le__(arg)

    def __lt__(self, arg):
        return self.as_numeric().__lt__(arg)

    def __ge__(self, arg):
        return self.as_numeric().__ge__(arg)

    def __gt__(self, arg):
        return self.as_numeric().__gt__(arg)

    def __eq__(self, arg):
        # If the other operand is a Boolean, then we want to fall back
        # on the "normal" implementation of __eq__ for Boolean values
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return super().__eq__(arg)
        # Otherwise, we will treat this as a binary operand and use the
        # (numeric) relational expression system
        return self.as_numeric().__eq__(arg)

    def __ne__(self, arg):
        # If the other operand is a Boolean, then we want to fall back
        # on the "normal" implementation of __ne__ for Boolean values
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return super().__ne__(arg)
        # Otherwise, we will treat this as a binary operand and use the
        # (numeric) relational expression system
        return self.as_numeric().__ne__(arg)

    def __add__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__add__(arg)

    def __div__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__div__(arg)

    def __mul__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__mul__(arg)

    def __pow__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__pow__(arg)

    def __sub__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__sub__(arg)

    def __truediv__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__truediv__(arg)

    def __iadd__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__iadd__(arg)

    def __idiv__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__idiv__(arg)

    def __imul__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__imul__(arg)

    def __ipow__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__ipow__(arg)

    def __isub__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__isub__(arg)

    def __itruediv__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__itruediv__(arg)

    def __radd__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__radd__(arg)

    def __rdiv__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__rdiv__(arg)

    def __rmul__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__rmul__(arg)

    def __rpow__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__rpow__(arg)

    def __rsub__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__rsub__(arg)

    def __rtruediv__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__rtruediv__(arg)

    def setlb(self, arg):
        return self.as_numeric().setlb(arg)

    def setub(self, arg):
        return self.as_numeric().setub(arg)


# The following should eventually be promoted so that all
# IndexedComponents can use it
class _Initializer(object):
    """A simple function to process an argument to a Component constructor.

    This checks the incoming initializer type and maps it to a static
    identifier so that when constructing indexed Components we can avoid
    a series of isinstance calls.  Eventually this concept should be
    promoted to pyomo.core so that all Components can leverage a
    standardized approach to processing "flexible" arguments (POD data,
    rules, dicts, generators, etc)."""

    value = 0
    deferred_value = 1
    function = 2
    dict_like = 3

    @staticmethod
    def process(arg):
        if type(arg) in native_types:
            return (_Initializer.value, bool(arg))
        elif type(arg) is types.FunctionType:
            return (_Initializer.function, arg)
        elif isinstance(arg, ComponentData):
            return (_Initializer.deferred_value, arg)
        elif hasattr(arg, '__getitem__'):
            return (_Initializer.dict_like, arg)
        else:
            # Hopefully this thing is castable to the type that is desired
            return (_Initializer.deferred_value, arg)


class DisjunctData(BlockData):
    __autoslot_mappers__ = {'_transformation_block': AutoSlots.weakref_mapper}

    _Block_reserved_words = set()

    @property
    def transformation_block(self):
        return (
            None if self._transformation_block is None else self._transformation_block()
        )

    def __init__(self, component):
        BlockData.__init__(self, component)
        with self._declare_reserved_components():
            self.indicator_var = AutoLinkedBooleanVar()
            self.binary_indicator_var = AutoLinkedBinaryVar(self.indicator_var)
        self.indicator_var.associate_binary_var(self.binary_indicator_var)
        # pointer to transformation block if this disjunct has been
        # transformed. None indicates it hasn't been transformed.
        self._transformation_block = None

    def activate(self):
        super(DisjunctData, self).activate()
        self.indicator_var.unfix()

    def deactivate(self):
        super(DisjunctData, self).deactivate()
        self.indicator_var.fix(False)

    def _deactivate_without_fixing_indicator(self):
        super(DisjunctData, self).deactivate()

    def _activate_without_unfixing_indicator(self):
        super(DisjunctData, self).activate()


class _DisjunctData(metaclass=RenamedClass):
    __renamed__new_class__ = DisjunctData
    __renamed__version__ = '6.7.2'


@ModelComponentFactory.register("Disjunctive blocks.")
class Disjunct(Block):
    _ComponentDataClass = DisjunctData

    def __new__(cls, *args, **kwds):
        if cls != Disjunct:
            return super(Disjunct, cls).__new__(cls)
        if args == ():
            return ScalarDisjunct.__new__(ScalarDisjunct)
        else:
            return IndexedDisjunct.__new__(IndexedDisjunct)

    def __init__(self, *args, **kwargs):
        if kwargs.pop('_deep_copying', None):
            # Hack for Python 2.4 compatibility
            # Deep copy will copy all items as necessary, so no need to
            # complete parsing
            return

        kwargs.setdefault('ctype', Disjunct)
        Block.__init__(self, *args, **kwargs)

    # For the time being, this method is not needed.
    #
    # def _deactivate_without_fixing_indicator(self):
    #    # Ideally, this would be a super call from this class.  However,
    #    # doing that would trigger a call to deactivate() on all the
    #    # DisjunctData objects (exactly what we want to avoid!)
    #    #
    #    # For the time being, we will do something bad and directly call
    #    # the base class method from where we would otherwise want to
    #    # call this method.

    def _activate_without_unfixing_indicator(self):
        # Ideally, this would be a super call from this class.  However,
        # doing that would trigger a call to deactivate() on all the
        # DisjunctData objects (exactly what we want to avoid!)
        #
        # For the time being, we will do something bad and directly call
        # the base class method from where we would otherwise want to
        # call this method.
        ActiveComponent.activate(self)
        if self.is_indexed():
            for component_data in self.values():
                component_data._activate_without_unfixing_indicator()


class ScalarDisjunct(DisjunctData, Disjunct):
    def __init__(self, *args, **kwds):
        DisjunctData.__init__(self, self)
        Disjunct.__init__(self, *args, **kwds)
        self._data[None] = self
        self._index = UnindexedComponent_index


class SimpleDisjunct(metaclass=RenamedClass):
    __renamed__new_class__ = ScalarDisjunct
    __renamed__version__ = '6.0'


class IndexedDisjunct(Disjunct):
    #
    # HACK: this should be implemented on ActiveIndexedComponent, but
    # that will take time and a PEP
    #
    @property
    def active(self):
        return any(d.active for d in self._data.values())


DisjunctData._Block_reserved_words = set(dir(Disjunct()))


class DisjunctionData(ActiveComponentData):
    __slots__ = ('disjuncts', 'xor', '_algebraic_constraint', '_transformation_map')
    __autoslot_mappers__ = {'_algebraic_constraint': AutoSlots.weakref_mapper}
    _NoArgument = (0,)

    @property
    def algebraic_constraint(self):
        return (
            None if self._algebraic_constraint is None else self._algebraic_constraint()
        )

    def __init__(self, component=None):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - ConstraintData,
        #   - ActiveComponentData
        #   - ComponentData
        self._component = weakref_ref(component) if (component is not None) else None
        self._index = NOTSET
        self._active = True
        self.disjuncts = []
        self.xor = True
        # pointer to XOR (or OR) constraint if this disjunction has been
        # transformed. None if it has not been transformed
        self._algebraic_constraint = None
        # Dictionary to notate information from partial transformations
        self._transformation_map = {}

    def set_value(self, expr):
        for e in expr:
            # The user gave us a proper Disjunct block
            # [ESJ 06/21/2019] This is really an issue with the reclassifier,
            # but in the case where you are iteratively adding to an
            # IndexedDisjunct indexed by Any which has already been transformed,
            # the new Disjuncts are Blocks already. This catches them for who
            # they are anyway.
            if hasattr(e, 'is_component_type') and e.is_component_type():
                if e.ctype == Disjunct and not e.is_indexed():
                    self.disjuncts.append(e)
                    continue
                e_iter = [e]
            elif hasattr(e, '__iter__'):
                e_iter = e
            else:
                e_iter = [e]
            # The user was lazy and gave us a single constraint
            # expression or an iterable of expressions
            expressions = []
            propositions = []
            for _tmp_e in e_iter:
                try:
                    if _tmp_e.is_expression_type(ExpressionType.RELATIONAL):
                        expressions.append(_tmp_e)
                        continue
                    if _tmp_e.is_expression_type(ExpressionType.LOGICAL):
                        propositions.append(_tmp_e)
                        continue
                except AttributeError:
                    pass
                # Note: Constraint.(In)Feasible is now a full relational
                # expression, so we don't need to check for it.
                # LogicalConstraint.(In)Feasible is a BooleanConstant,
                # so is not an expression and needs to be explicitly
                # checked for.
                if _tmp_e in (LogicalConstraint.Feasible, LogicalConstraint.Infeasible):
                    propositions.append(_tmp_e)
                    continue
                msg = " in '%s'" % (type(e).__name__,) if e_iter is e else ""
                raise ValueError(
                    "Unexpected term for Disjunction '%s'.\n    "
                    "Expected a Disjunct object, relational or logical "
                    "expression, or\n    iterable of relational/logical "
                    "expressions but got '%s'%s"
                    % (self.name, type(_tmp_e).__name__, msg)
                )

            comp = self.parent_component()
            if comp._autodisjuncts is None:
                b = self.parent_block()
                comp._autodisjuncts = Disjunct(Any)
                b.add_component(
                    unique_component_name(b, comp.local_name + "_disjuncts"),
                    comp._autodisjuncts,
                )
                # TODO: I am not at all sure why we need to
                # explicitly construct this block - that should
                # happen automatically.
                comp._autodisjuncts.construct()
            disjunct = comp._autodisjuncts[len(comp._autodisjuncts)]
            disjunct.constraint = ConstraintList(rule=expressions)
            disjunct.propositions = LogicalConstraintList(rule=propositions)
            self.disjuncts.append(disjunct)


class _DisjunctionData(metaclass=RenamedClass):
    __renamed__new_class__ = DisjunctionData
    __renamed__version__ = '6.7.2'


@ModelComponentFactory.register("Disjunction expressions.")
class Disjunction(ActiveIndexedComponent):
    _ComponentDataClass = DisjunctionData

    def __new__(cls, *args, **kwds):
        if cls != Disjunction:
            return super(Disjunction, cls).__new__(cls)
        if args == ():
            return ScalarDisjunction.__new__(ScalarDisjunction)
        else:
            return IndexedDisjunction.__new__(IndexedDisjunction)

    def __init__(self, *args, **kwargs):
        self._init_rule = kwargs.pop('rule', None)
        self._init_expr = kwargs.pop('expr', None)
        self._init_xor = _Initializer.process(kwargs.pop('xor', True))
        self._autodisjuncts = None
        kwargs.setdefault('ctype', Disjunction)
        super(Disjunction, self).__init__(*args, **kwargs)

        if self._init_expr is not None and self._init_rule is not None:
            raise ValueError(
                "Cannot specify both rule= and expr= for Disjunction %s" % (self.name,)
            )

    #
    # TODO: Ideally we would not override these methods and instead add
    # the contents of _check_skip_add to the set_value() method.
    # Unfortunately, until IndexedComponentData objects know their own
    # index, determining the index is a *very* expensive operation.  If
    # we refactor things so that the Data objects have their own index,
    # then we can remove these overloads.
    #

    def _setitem_impl(self, index, obj, value):
        if value is Disjunction.Skip:
            del self[index]
            return None
        else:
            obj.set_value(value)
            return obj

    def _setitem_when_not_present(self, index, value):
        if value is Disjunction.Skip:
            return None
        else:
            ans = super(Disjunction, self)._setitem_when_not_present(
                index=index, value=value
            )
            self._initialize_members((index,))
            return ans

    def _initialize_members(self, init_set):
        if self._init_xor[0] == _Initializer.value:  # POD data
            val = self._init_xor[1]
            for key in init_set:
                self._data[key].xor = val
        elif self._init_xor[0] == _Initializer.deferred_value:  # Param data
            val = bool(value(self._init_xor[1]))
            for key in init_set:
                self._data[key].xor = val
        elif self._init_xor[0] == _Initializer.function:  # rule
            fcn = self._init_xor[1]
            for key in init_set:
                self._data[key].xor = bool(
                    value(apply_indexed_rule(self, fcn, self._parent(), key))
                )
        elif self._init_xor[0] == _Initializer.dict_like:  # dict-like thing
            val = self._init_xor[1]
            for key in init_set:
                self._data[key].xor = bool(value(val[key]))

    def construct(self, data=None):
        if is_debug_set(logger):
            logger.debug("Constructing disjunction %s" % (self.name))
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        self._constructed = True

        if self._anonymous_sets is not None:
            for _set in self._anonymous_sets:
                _set.construct()

        _self_parent = self.parent_block()
        if not self.is_indexed():
            if self._init_rule is not None:
                expr = self._init_rule(_self_parent)
            elif self._init_expr is not None:
                expr = self._init_expr
            else:
                timer.report()
                return

            if expr is None:
                raise ValueError(_rule_returned_none_error % (self.name,))
            if expr is Disjunction.Skip:
                timer.report()
                return
            self._data[None] = self
            self._setitem_when_not_present(None, expr)
        elif self._init_expr is not None:
            raise IndexError(
                "Disjunction '%s': Cannot initialize multiple indices "
                "of a disjunction with a single disjunction list" % (self.name,)
            )
        elif self._init_rule is not None:
            _init_rule = self._init_rule
            for ndx in self._index_set:
                try:
                    expr = apply_indexed_rule(self, _init_rule, _self_parent, ndx)
                except Exception:
                    err = sys.exc_info()[1]
                    logger.error(
                        "Rule failed when generating expression for "
                        "disjunction %s with index %s:\n%s: %s"
                        % (self.name, str(ndx), type(err).__name__, err)
                    )
                    raise
                if expr is None:
                    _name = "%s[%s]" % (self.name, str(ndx))
                    raise ValueError(_rule_returned_none_error % (_name,))
                if expr is Disjunction.Skip:
                    continue
                self._setitem_when_not_present(ndx, expr)
        timer.report()

    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        return (
            [
                ("Size", len(self)),
                ("Index", self._index_set if self.is_indexed() else None),
                ("Active", self.active),
            ],
            self.items(),
            ("Disjuncts", "Active", "XOR"),
            lambda k, v: [[x.name for x in v.disjuncts], v.active, v.xor],
        )


class ScalarDisjunction(DisjunctionData, Disjunction):
    def __init__(self, *args, **kwds):
        DisjunctionData.__init__(self, component=self)
        Disjunction.__init__(self, *args, **kwds)
        self._index = UnindexedComponent_index

    #
    # Singleton disjunctions are strange in that we want them to be
    # both be constructed but have len() == 0 when not initialized with
    # anything (at least according to the unit tests that are
    # currently in place). So during initialization only, we will
    # treat them as "indexed" objects where things like
    # Constraint.Skip are managed. But after that they will behave
    # like DisjunctionData objects where set_value does not handle
    # Disjunction.Skip but expects a valid expression or None.
    #

    def set_value(self, expr):
        """Set the expression on this disjunction."""
        if not self._constructed:
            raise ValueError(
                "Setting the value of disjunction '%s' "
                "before the Disjunction has been constructed (there "
                "is currently no object to set)." % (self.name)
            )

        if len(self._data) == 0:
            self._data[None] = self
        if expr is Disjunction.Skip:
            del self[None]
            return None
        return super(ScalarDisjunction, self).set_value(expr)


class SimpleDisjunction(metaclass=RenamedClass):
    __renamed__new_class__ = ScalarDisjunction
    __renamed__version__ = '6.0'


class IndexedDisjunction(Disjunction):
    #
    # HACK: this should be implemented on ActiveIndexedComponent, but
    # that will take time and a PEP
    #
    @property
    def active(self):
        return any(d.active for d in self._data.values())
