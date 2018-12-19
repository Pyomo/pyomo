#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys
import inspect
from six import iteritems
from collections import namedtuple

from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr import current as EXPR
from pyomo.core.expr.numvalue import ZeroConstant, _sub, native_numeric_types, as_numeric
from pyomo.core import *
from pyomo.core.base.plugin import ModelComponentFactory
from pyomo.core.base.numvalue import ZeroConstant, _sub
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.block import _BlockData

import logging
logger = logging.getLogger('pyomo.core')


#
# A named 2-tuple that minimizes error checking
#
ComplementarityTuple = namedtuple('ComplementarityTuple', ('arg0', 'arg1'))


def complements(a, b):
    """ Return a named 2-tuple """
    return ComplementarityTuple(a,b)


class _ComplementarityData(_BlockData):

    def _canonical_expression(self, e):
        # Note: as the complimentarity component maintains references to
        # the original expression (e), it is NOT safe or valid to bypass
        # the clone checks: bypassing the check can result in corrupting
        # the original expressions and will result in mind-boggling
        # pprint output.
        e_ = None
        if e.__class__ is EXPR.EqualityExpression:
            if e.arg(1).__class__ in native_numeric_types or e.arg(1).is_fixed():
                _e = (e.arg(1), e.arg(0))
            #
            # The first argument of an equality is never fixed
            #
            else:
                _e = ( ZeroConstant, e.arg(0) - e.arg(1))
        elif e.__class__ is EXPR.InequalityExpression:
            if e.arg(1).__class__ in native_numeric_types or e.arg(1).is_fixed():
                _e = (None, e.arg(0), e.arg(1))
            elif e.arg(0).__class__ in native_numeric_types or e.arg(0).is_fixed():
                _e = (e.arg(0), e.arg(1), None)
            else:
                _e = ( ZeroConstant, e.arg(1) - e.arg(0), None )
        elif e.__class__ is EXPR.RangedExpression:
                _e = (e.arg(0), e.arg(1), e.arg(2))
        else:
            _e = (None, e, None)
        return _e

    def to_standard_form(self):
        #
        # Add auxilliary variables and constraints that ensure
        # a monotone transformation of general complementary constraints to
        # the form:
        #       l1 <= v1 <= u1   OR   l2 <= v2 <= u2
        #
        # Note that this transformation creates more variables and constraints
        # than are strictly necessary.  However, we don't have a complete list of
        # the variables used in a model's complementarity conditions when adding
        # a single condition, so we add additional variables.
        #
        # This has the form:
        #
        #  e:   l1 <= expression <= l2
        #  v:   l3 <= var <= l4
        #
        # where exactly two of l1, l2, l3 and l4 are finite, and with the
        # equality constraint:
        #
        #  c:   v == expression
        #
        _e1 = self._canonical_expression(self._args[0])
        _e2 = self._canonical_expression(self._args[1])
        if len(_e1) == 2:
            # Ignore _e2; _e1 is an equality constraint
            self.c = Constraint(expr=_e1)
            return
        if len(_e2) == 2:
            # Ignore _e1; _e2 is an equality constraint
            self.c = Constraint(expr=_e2)
            return
        #
        if (_e1[0] is None) + (_e1[2] is None) + (_e2[0] is None) + (_e2[2] is None) != 2:
            raise RuntimeError("Complementarity condition %s must have exactly two finite bounds" % self.name)
        #
        if _e1[0] is None and _e1[2] is None:
            # Only e2 will be an unconstrained expression
            _e1, _e2 = _e2, _e1
        #
        if _e2[0] is None and _e2[2] is None:
            self.c = Constraint(expr=(None, _e2[1], None))
            self.c._type = 3
        elif _e2[2] is None:
            self.c = Constraint(expr=_e2[0] <= _e2[1])
            self.c._type = 1
        elif _e2[0] is None:
            self.c = Constraint(expr=- _e2[2] <= - _e2[1])
            self.c._type = 1
        #
        if not _e1[0] is None and not _e1[2] is None:
            if not (_e1[0].__class__ in native_numeric_types or _e1[0].is_constant()):
                raise RuntimeError("Cannot express a complementarity problem of the form L < v < U _|_ g(x) where L is not a constant value")
            if not (_e1[2].__class__ in native_numeric_types or _e1[2].is_constant()):
                raise RuntimeError("Cannot express a complementarity problem of the form L < v < U _|_ g(x) where U is not a constant value")
            self.v = Var(bounds=(_e1[0], _e1[2]))
            self.ve = Constraint(expr=self.v == _e1[1])
        elif _e1[2] is None:
            self.v = Var(bounds=(0, None))
            self.ve = Constraint(expr=self.v == _e1[1] - _e1[0])
        else:
            # _e1[0] is None:
            self.v = Var(bounds=(0, None))
            self.ve = Constraint(expr=self.v == _e1[2] - _e1[1])


@ModelComponentFactory.register("Complementarity conditions.")
class Complementarity(Block):

    Skip = (1000,)

    def __new__(cls, *args, **kwds):
        if cls != Complementarity:
            return super(Complementarity, cls).__new__(cls)
        if args == ():
            return SimpleComplementarity.__new__(SimpleComplementarity)
        else:
            return IndexedComplementarity.__new__(IndexedComplementarity)

    def __init__(self, *args, **kwargs):
        self._expr = kwargs.pop('expr', None )
        #
        kwargs.setdefault('ctype', Complementarity)
        #
        # The attribute _rule is initialized here.
        #
        Block.__init__(self, *args, **kwargs)

    def construct(self, data=None):
        if __debug__ and logger.isEnabledFor(logging.DEBUG):        #pragma:nocover
            logger.debug("Constructing %s '%s', from data=%s",
                         self.__class__.__name__, self.name, str(data))
        if self._constructed:                                       #pragma:nocover
            return
        timer = ConstructionTimer(self)

        #
        _self_rule = self._rule
        self._rule = None
        super(Complementarity, self).construct()
        self._rule = _self_rule
        #
        if _self_rule is None and self._expr is None:
            # No construction rule or expression specified.
            return
        #
        if not self.is_indexed():
            #
            # Scalar component
            #
            if _self_rule is None:
                self.add(None, self._expr)
            else:
                try:
                    tmp = _self_rule(self.parent_block())
                    self.add(None, tmp)
                except Exception:
                    err = sys.exc_info()[1]
                    logger.error(
                        "Rule failed when generating expression for "
                        "complementarity %s:\n%s: %s"
                        % ( self.name, type(err).__name__, err ) )
                    raise
        else:
            if not self._expr is None:
                raise IndexError(
                    "Cannot initialize multiple indices of a Complementarity "
                    "component with a single expression")
            _self_parent = self._parent()
            for idx in self._index:
                try:
                    tmp = apply_indexed_rule( self, _self_rule, _self_parent, idx )
                    self.add(idx, tmp)
                except Exception:
                    err = sys.exc_info()[1]
                    logger.error(
                        "Rule failed when generating expression for "
                        "complementarity %s with index %s:\n%s: %s"
                        % ( self.name, idx, type(err).__name__, err ) )
                    raise
        timer.report()

    def add(self, index, cc):
        """
        Add a complementarity condition with a specified index.
        """
        if cc.__class__ is ComplementarityTuple:
            #
            # The ComplementarityTuple has a fixed length, so we initialize
            # the _args component and return
            #
            self[index]._args = ( as_numeric(cc.arg0), as_numeric(cc.arg1) )
            return self[index]
        #
        if cc.__class__ is tuple:
            if cc is Complementarity.Skip:
                return
            elif len(cc) != 2:
                raise ValueError(
                    "Invalid tuple for Complementarity %s (expected 2-tuple):"
                    "\n\t%s" % (self.name, cc) )
        elif cc.__class__ is list:
            #
            # Call add() recursively to apply the error same error
            # checks.
            #
            return self.add(index, tuple(cc))
        elif cc is None:
                raise ValueError("""
Invalid complementarity condition.  The complementarity condition
is None instead of a 2-tuple.  Please modify your rule to return
Complementarity.Skip instead of None.

Error thrown for Complementarity "%s"
""" % ( self.name, ) )
        else:
            raise ValueError(
                "Unexpected argument declaring Complementarity %s:\n\t%s"
                % (self.name, cc) )
        #
        self[index]._args = tuple( as_numeric(x) for x in cc )
        return self[index]

    def pprint(self, **kwargs):
        if self._type is Complementarity:
            Component.pprint(self, **kwargs)
        else:
            Block.pprint(self, **kwargs)

    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        return (
            [("Size", len(self)),
             ("Index", self._index if self.is_indexed() else None),
             ("Active", self.active),
             ],
            iteritems(self._data),
            ( "Arg0","Arg1","Active" ),
            lambda k, v: [ v._args[0],
                           v._args[1],
                           v.active,
                           ]
            )


class SimpleComplementarity(_ComplementarityData, Complementarity):

    def __init__(self, *args, **kwds):
        _ComplementarityData.__init__(self, self)
        Complementarity.__init__(self, *args, **kwds)
        self._data[None] = self

    def pprint(self, **kwargs):
        Complementarity.pprint(self, **kwargs)


class IndexedComplementarity(Complementarity):

    def _getitem_when_not_present(self, idx):
        return self._data.setdefault(idx, _ComplementarityData(self))


@ModelComponentFactory.register("A list of complementarity conditions.")
class ComplementarityList(IndexedComplementarity):
    """
    A complementarity component that represents a list of complementarity
    conditions.  Each condition can be indexed by its index, but when added
    an index value is not specified.
    """

    End             = (1003,)

    def __init__(self, **kwargs):
        """Constructor"""
        args = (Set(),)
        self._nconditions = 0
        Complementarity.__init__(self, *args, **kwargs)

    def add(self, expr):
        """
        Add a complementarity condition with an implicit index.
        """
        self._nconditions += 1
        self._index.add(self._nconditions)
        return Complementarity.add(self, self._nconditions, expr)

    def construct(self, data=None):
        """
        Construct the expression(s) for this complementarity condition.
        """
        generate_debug_messages = __debug__ and logger.isEnabledFor(logging.DEBUG)
        if generate_debug_messages:         #pragma:nocover
            logger.debug("Constructing complementarity list %s", self.name)
        if self._constructed:               #pragma:nocover
            return
        timer = ConstructionTimer(self)
        _self_rule = self._rule
        self._constructed=True
        if _self_rule is None:
            return
        #
        _generator = None
        _self_parent = self._parent()
        if inspect.isgeneratorfunction(_self_rule):
            _generator = _self_rule(_self_parent)
        elif inspect.isgenerator(_self_rule):
            _generator = _self_rule
        if _generator is None:
            while True:
                val = self._nconditions + 1
                if generate_debug_messages:     #pragma:nocover
                    logger.debug("   Constructing complementarity index "+str(val))
                expr = apply_indexed_rule( self, _self_rule, _self_parent, val )
                if expr is None:
                    raise ValueError( "Complementarity rule returned None "
                                      "instead of ComplementarityList.End" )
                if (expr.__class__ is tuple and expr == ComplementarityList.End):
                    return
                self.add(expr)
        else:
            for expr in _generator:
                if expr is None:
                    raise ValueError( "Complementarity generator returned None "
                                      "instead of ComplementarityList.End" )
                if (expr.__class__ is tuple and expr == ComplementarityList.End):
                    return
                self.add(expr)
        timer.report()

