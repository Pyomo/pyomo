#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
import sys
from six import iteritems, itervalues
from weakref import ref as weakref_ref

from pyomo.common.modeling import unique_component_name
from pyomo.common.timing import ConstructionTimer
from pyomo.core import (
    ModelComponentFactory, Binary, Block, Var, ConstraintList, Any
)
from pyomo.core.base.component import (
    ActiveComponent, ActiveComponentData, ComponentData
)
from pyomo.core.base.numvalue import native_types
from pyomo.core.base.block import _BlockData
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.indexed_component import ActiveIndexedComponent

logger = logging.getLogger('pyomo.gdp')

_rule_returned_none_error = """Disjunction '%s': rule returned None.

Disjunction rules must return an iterable containing Disjuncts or
individual expressions, or Disjunction.Skip.  The most common cause of
this error is forgetting to include the "return" statement at the end of
your rule.
"""

class GDP_Error(Exception):
    """Exception raised while processing GDP Models"""


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


class _DisjunctData(_BlockData):

    def __init__(self, component):
        _BlockData.__init__(self, component)
        self.indicator_var = Var(within=Binary)

    def pprint(self, ostream=None, verbose=False, prefix=""):
        _BlockData.pprint(self, ostream=ostream, verbose=verbose, prefix=prefix)

    def set_value(self, val):
        _indicator_var = self.indicator_var
        # Remove everything
        for k in list(getattr(self, '_decl', {})):
            self.del_component(k)
        self._ctypes = {}
        self._decl = {}
        self._decl_order = []
        # Now copy over everything from the other block.  If the other
        # block has an indicator_var, it should override this block's.
        # Otherwise restore this block's indicator_var.
        if val:
            if 'indicator_var' not in val:
                self.add_component('indicator_var', _indicator_var)
            for k in sorted(iterkeys(val)):
                self.add_component(k,val[k])
        else:
            self.add_component('indicator_var', _indicator_var)

    def activate(self):
        super(_DisjunctData, self).activate()
        self.indicator_var.unfix()

    def deactivate(self):
        super(_DisjunctData, self).deactivate()
        self.indicator_var.fix(0)

    def _deactivate_without_fixing_indicator(self):
        super(_DisjunctData, self).deactivate()

    def _activate_without_unfixing_indicator(self):
        super(_DisjunctData, self).activate()


@ModelComponentFactory.register("Disjunctive blocks.")
class Disjunct(Block):

    _ComponentDataClass = _DisjunctData

    def __new__(cls, *args, **kwds):
        if cls != Disjunct:
            return super(Disjunct, cls).__new__(cls)
        if args == ():
            return SimpleDisjunct.__new__(SimpleDisjunct)
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
    #def _deactivate_without_fixing_indicator(self):
    #    # Ideally, this would be a super call from this class.  However,
    #    # doing that would trigger a call to deactivate() on all the
    #    # _DisjunctData objects (exactly what we want to aviod!)
    #    #
    #    # For the time being, we will do something bad and directly call
    #    # the base class method from where we would otherwise want to
    #    # call this method.

    def _activate_without_unfixing_indicator(self):
        # Ideally, this would be a super call from this class.  However,
        # doing that would trigger a call to deactivate() on all the
        # _DisjunctData objects (exactly what we want to aviod!)
        #
        # For the time being, we will do something bad and directly call
        # the base class method from where we would otherwise want to
        # call this method.
        ActiveComponent.activate(self)
        if self.is_indexed():
            for component_data in itervalues(self):
                component_data._activate_without_unfixing_indicator()


class SimpleDisjunct(_DisjunctData, Disjunct):

    def __init__(self, *args, **kwds):
        ## FIXME: This is a HACK to get around a chicken-and-egg issue
        ## where _BlockData creates the indicator_var *before*
        ## Block.__init__ declares the _defer_construction flag.
        self._defer_construction = True
        self._suppress_ctypes = set()

        _DisjunctData.__init__(self, self)
        Disjunct.__init__(self, *args, **kwds)
        self._data[None] = self

    def pprint(self, ostream=None, verbose=False, prefix=""):
        Disjunct.pprint(self, ostream=ostream, verbose=verbose, prefix=prefix)


class IndexedDisjunct(Disjunct):
    pass



class _DisjunctionData(ActiveComponentData):
    __slots__ = ('disjuncts','xor')
    _NoArgument = (0,)

    def __init__(self, component=None):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - _ConstraintData,
        #   - ActiveComponentData
        #   - ComponentData
        self._component = weakref_ref(component) if (component is not None) \
                          else None
        self._active = True
        self.disjuncts = []
        self.xor = True

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        result = super(_DisjunctionData, self).__getstate__()
        for i in _DisjunctionData.__slots__:
            result[i] = getattr(self, i)
        return result

    def set_value(self, expr):
        for e in expr:
            # The user gave us a proper Disjunct block
            if hasattr(e, 'type') and e.type() == Disjunct:
                self.disjuncts.append(e)
                continue
            # The user was lazy and gave us a single constraint
            # expression or an iterable of expressions
            expressions = []
            if hasattr(e, '__iter__'):
                e_iter = e
            else:
                e_iter = [e]
            for _tmpe in e_iter:
                try:
                    isexpr = _tmpe.is_expression_type()
                except AttributeError:
                    isexpr = False
                if not isexpr or not _tmpe.is_relational():
                    msg = "\n\tin %s" % (type(e),) if e_iter is e else ""
                    raise ValueError(
                        "Unexpected term for Disjunction %s.\n"
                        "\tExpected a Disjunct object, relational expression, "
                        "or iterable of\n"
                        "\trelational expressions but got %s%s"
                        % (self.name, type(_tmpe), msg) )
                else:
                    expressions.append(_tmpe)

            comp = self.parent_component()
            if comp._autodisjuncts is None:
                b = self.parent_block()
                comp._autodisjuncts = Disjunct(Any)
                b.add_component(
                    unique_component_name(b, comp.local_name + "_disjuncts"),
                    comp._autodisjuncts )
                # TODO: I am not at all sure why we need to
                # explicitly construct this block - that should
                # happen automatically.
                comp._autodisjuncts.construct()
            disjunct = comp._autodisjuncts[len(comp._autodisjuncts)]
            disjunct.constraint = c = ConstraintList()
            for e in expressions:
                c.add(e)
            self.disjuncts.append(disjunct)


@ModelComponentFactory.register("Disjunction expressions.")
class Disjunction(ActiveIndexedComponent):
    Skip = (0,)
    _ComponentDataClass = _DisjunctionData

    def __new__(cls, *args, **kwds):
        if cls != Disjunction:
            return super(Disjunction, cls).__new__(cls)
        if args == ():
            return SimpleDisjunction.__new__(SimpleDisjunction)
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
                "Cannot specify both rule= and expr= for Disjunction %s"
                % ( self.name, ))

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
                index=index, value=value)
            self._initialize_members((index,))
            return ans

    def _initialize_members(self, init_set):
        if self._init_xor[0] == _Initializer.value: # POD data
            val = self._init_xor[1]
            for key in init_set:
                self._data[key].xor = val
        elif self._init_xor[0] == _Initializer.deferred_value: # Param data
            val = bool(value( self._init_xor[1] ))
            for key in init_set:
                self._data[key].xor = val
        elif self._init_xor[0] == _Initializer.function: # rule
            fcn = self._init_xor[1]
            for key in init_set:
                self._data[key].xor = bool(value(apply_indexed_rule(
                    self, fcn, self._parent(), key)))
        elif self._init_xor[0] == _Initializer.dict_like: # dict-like thing
            val = self._init_xor[1]
            for key in init_set:
                self._data[key].xor = bool(value(val[key]))

    def construct(self, data=None):
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Constructing disjunction %s"
                         % (self.name))
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        self._constructed=True

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
                raise ValueError( _rule_returned_none_error % (self.name,) )
            if expr is Disjunction.Skip:
                timer.report()
                return
            self._data[None] = self
            self._setitem_when_not_present( None, expr )
        elif self._init_expr is not None:
            raise IndexError(
                "Disjunction '%s': Cannot initialize multiple indices "
                "of a disjunction with a single disjunction list" %
                (self.name,) )
        elif self._init_rule is not None:
            _init_rule = self._init_rule
            for ndx in self._index:
                try:
                    expr = apply_indexed_rule(self,
                                             _init_rule,
                                             _self_parent,
                                             ndx)
                except Exception:
                    err = sys.exc_info()[1]
                    logger.error(
                        "Rule failed when generating expression for "
                        "disjunction %s with index %s:\n%s: %s"
                        % (self.name,
                           str(ndx),
                           type(err).__name__,
                           err))
                    raise
                if expr is None:
                    _name = "%s[%s]" % (self.name, str(idx))
                    raise ValueError( _rule_returned_none_error % (_name,) )
                if expr is Disjunction.Skip:
                    continue
                self._setitem_when_not_present(ndx, expr)
        timer.report()

    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        return (
            [("Size", len(self)),
             ("Index", self._index if self.is_indexed() else None),
             ("Active", self.active),
             ],
            iteritems(self),
            ( "Disjuncts", "Active", "XOR" ),
            lambda k, v: [ [x.name for x in v.disjuncts], v.active, v.xor]
            )


class SimpleDisjunction(_DisjunctionData, Disjunction):

    def __init__(self, *args, **kwds):
        _DisjunctionData.__init__(self, component=self)
        Disjunction.__init__(self, *args, **kwds)

    #
    # Singleton disjunctions are strange in that we want them to be
    # both be constructed but have len() == 0 when not initialized with
    # anything (at least according to the unit tests that are
    # currently in place). So during initialization only, we will
    # treat them as "indexed" objects where things like
    # Constraint.Skip are managed. But after that they will behave
    # like _DisjunctionData objects where set_value does not handle
    # Disjunction.Skip but expects a valid expression or None.
    #

    def set_value(self, expr):
        """Set the expression on this disjunction."""
        if not self._constructed:
            raise ValueError(
                "Setting the value of disjunction '%s' "
                "before the Disjunction has been constructed (there "
                "is currently no object to set)."
                % (self.name))

        if len(self._data) == 0:
            self._data[None] = self
        if expr is Disjunction.Skip:
            del self[None]
            return None
        return super(SimpleDisjunction, self).set_value(expr)

class IndexedDisjunction(Disjunction):
    pass

