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
from six import iteritems
from weakref import ref as weakref_ref

#from pyomo.core import *
from pyomo.util.modeling import unique_component_name
from pyomo.util.timing import ConstructionTimer
from pyomo.core import register_component, Binary, Block, Var, Constraint, Any
from pyomo.core.base.component import ( ActiveComponentData, )
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


class _DisjunctData(_BlockData):

    def __init__(self, owner):
        _BlockData.__init__(self, owner)
        self.indicator_var = Var(within=Binary)

    def pprint(self, ostream=None, verbose=False, prefix=""):
        _BlockData.pprint(self, ostream=ostream, verbose=verbose, prefix=prefix)


class Disjunct(Block):

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

    def _default(self, idx):
        return self._data.setdefault(idx, _DisjunctData(self))


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
    __slots__ = ('disjuncts',)
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

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        result = super(_DisjunctionData, self).__getstate__()
        for i in _DisjunctionData.__slots__:
            result[i] = getattr(self, i)
        return result

    def set_value(self, expr):
        return self._set_value_impl(expr)

    def _set_value_impl(self, expr, idx=_NoArgument):
        for e in expr:
            # The user gave us a proper Disjunct block
            if isinstance(e, _DisjunctData):
                self.disjuncts.append(e)
                continue
            # The user was lazy and gave us a single constraint expression
            try:
                isexpr = e.is_expression()
            except AttribureError:
                isexpr = False
            if isexpr and e.is_relational():
                comp = self.parent_component()
                if comp._autodisjuncts is None:
                    b = self.parent_block()
                    comp._autodisjuncts = Disjunct(Any)
                    b.add_component(
                        unique_component_name(b, comp.local_name),
                        comp._autodisjuncts )
                    # TODO: I am not at all sure why we need to
                    # explicitly construct this block - that should
                    # happen automatically.
                    comp._autodisjuncts.construct()
                disjunct = comp._autodisjuncts[len(comp._autodisjuncts)]
                disjunct.constraint = Constraint(expr=e)
                self.disjuncts.append(disjunct)
                continue
            #
            # Anything else is an error
            raise ValueError(
                "Unexpected term for Disjunction %s.\n"
                "\tExpected a Disjunct object or relational expression, "
                "but got %s" % (self.name, type(e)) )


class Disjunction(ActiveIndexedComponent):
    Skip = (0,)

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
        self.xor = kwargs.pop('xor', True)
        self._autodisjuncts = None
        kwargs.setdefault('ctype', Disjunction)
        super(Disjunction, self).__init__(*args, **kwargs)

        if self._init_expr is not None and self._init_rule is not None:
            raise ValueError(
                "Cannot specify both rule= and expr= for Disjunction %s"
                % ( self.name, ))

    def construct(self, data=None):
        # TODO John did something weird with logging
        # if __debug__ and logger.isEnabledFor(logging.DEBUG):
        #     logger.debug("Constructing disjunction %s"
        #                  % (self.name))
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        self._constructed=True

        _self_parent = self.parent_block()
        if not self.is_indexed():
            if self._init_rule is not None:
                expr = self._init_rule(_self_parent)
            else:
                expr = self._init_expr
            if expr is None:
                raise ValueError( _rule_returned_none_error % (self.name,) )
            if expr is Disjunction.Skip:
                timer.report()
                return
            self._data[None] = self
            self._set_value_impl( expr, None )
        else:
            if self._init_expr is not None:
                raise IndexError(
                    "Disjunction '%s': Cannot initialize multiple indices "
                    "of a disjunction with a single disjunction list" %
                    (self.name,) )
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
                self[ndx]._set_value_impl(expr, ndx)
        timer.report()

    #
    # This method must be defined on subclasses of IndexedComponent
    #
    def _default(self, idx):
        """Returns the default component data value."""
        data = self._data[idx] = _DisjunctionData(component=self)
        return data


    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        return (
            [("Size", len(self)),
             ("Index", self._index if self.is_indexed() else None),
             ("XOR", self.xor),
             ("Active", self.active),
             ],
            iteritems(self),
            ( "Disjuncts", "Active" ),
            lambda k, v: [ [x.name for x in v.disjuncts], v.active, ]
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
        if self._constructed:
            if len(self._data) == 0:
                self._data[None] = self
            return _set_value_impl(self, expr, None)
        raise ValueError(
            "Setting the value of disjunction '%s' "
            "before the Disjunction has been constructed (there "
            "is currently no object to set)."
            % (self.name))

class IndexedDisjunction(Disjunction):
    pass


register_component(Disjunct, "Disjunctive blocks.")
register_component(Disjunction, "Disjunction expressions.")
