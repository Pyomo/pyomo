#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the FAST README.txt file.
#  _________________________________________________________________________

import sys
from six import iteritems
from collections import namedtuple

from pyomo.core import *
from pyomo.core.base.numvalue import ZeroConstant, _sub
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.expr import _InequalityExpression, _EqualityExpression, generate_expression_bypassCloneCheck
from pyomo.core.base.block import _BlockData
from pyomo.core.base.sparse_indexed_component import UnindexedComponent_set

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
        e_ = None
        if e.__class__ is _EqualityExpression:
            if e._args[1].is_fixed():
                _e = (e._args[1], e._args[0])
            #
            # The first argument of an equality is never fixed
            #
            #elif e._args[0].is_fixed():
            #    _e = (e._args[0], e._args[1])
            else:
                _e = ( ZeroConstant, generate_expression_bypassCloneCheck(
                        _sub, e._args[0], e._args[1]) )
        elif e.__class__ is _InequalityExpression:
            if len(e._args) == 3:
                _e = (e._args[0], e._args[1], e._args[2])
            else:
                if e._args[1].is_fixed():
                    _e = (None, e._args[0], e._args[1])
                elif e._args[0].is_fixed():
                    _e = (e._args[0], e._args[1], None)
                else:
                    _e = ( ZeroConstant, generate_expression_bypassCloneCheck(
                            _sub, e._args[1], e._args[0]), None )
        else:
            _e = (None, e, None)
        return _e

    def to_standard_form(self, old=False):
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

        if old:
            if len(_e1) == 3 and _e1[0] is None and _e1[2] is None:
                # Only e2 will be an unconstrained expression
                _e1, _e2 = _e2, _e1
            #
            # Define the constraint c such that l == expr or l <= expr
            #
            if len(_e1) == 2:
                self.c = Constraint(expr=_e1[0] == _e1[1])
            elif _e1[2] is None:
                self.c = Constraint(expr=_e1[0] <= _e1[1])
            elif _e1[0] is None:
                self.c = Constraint(expr=- _e1[1] >= - _e1[2])
            else:
                self.c = Constraint(expr=_e1[0] <= _e1[1] <= _e1[2])
            #
            # Define variable v such that v is unconstrained or v >= 0
            #
            if _e2[0] is None and _e2[2] is None:
                if len(_e1) == 3:
                    self.v = Var()
                    self.ve = Constraint(expr=self.v == _e2[1])
                    self._complementarity = 0
                #
                # Else, we have an equality constraint, so we don't need
                # to add additional logic
                #
            elif not _e2[0] is None:
                self.v = Var(bounds=(0, None))
                self.ve = Constraint(expr=self.v == _e2[1] - _e2[0])
                self._complementarity = 1
            else:
                # not _e2[2] is None
                self.v = Var(bounds=(0, None))
                self.ve = Constraint(expr=self.v == _e2[2] - _e2[1])
                self._complementarity = 1
        else:
            if len(_e1) == 3 and _e1[0] is None and _e1[2] is None:
                # Only e2 will be an unconstrained expression
                _e1, _e2 = _e2, _e1
            #
            if len(_e1) == 2:
                # Ignore _e2 is _e1 is an equality constraint
                self.c = Constraint(expr=_e1[0] == _e1[1])
            else:
                if _e2[0] is None and _e2[2] is None:
                    self.c = Constraint(expr=(None, _e2[1], None))
                elif _e2[2] is None:
                    self.c = Constraint(expr=_e2[0] <= _e2[1])
                elif _e2[0] is None:
                    self.c = Constraint(expr=- _e2[1] >= - _e2[2])
                #
                if not _e1[0] is None and not _e1[2] is None:
                    if not _e1[0].is_constant():
                        raise RuntimeError("Cannot express a complementarity problem of the form L < v < U _|_ g(x) where L is not a constant value")
                    if not _e1[2].is_constant():
                        raise RuntimeError("Cannot express a complementarity problem of the form L < v < U _|_ g(x) where U is not a constant value")
                    self.v = Var(bounds=(_e1[0], _e1[2]))
                    self.ve = Constraint(expr=self.v == _e1[1])
                    self.c._complementarity = 3
                elif _e1[2] is None:
                    self.v = Var(bounds=(0, None))
                    self.ve = Constraint(expr=self.v == _e1[1] - _e1[0])
                    self.c._complementarity = 1
                else:
                    # _e1[0] is None:
                    self.v = Var(bounds=(0, None))
                    self.ve = Constraint(expr=self.v == _e1[2] - _e1[1])
                    self.c._complementarity = 2


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
        self._expr = kwargs.pop('initialize', self._expr )
        self._no_rule_init = kwargs.pop('noruleinit', False )
        #
        kwargs.setdefault('ctype', Complementarity)
        #
        # The attribute _rule is initialized here.
        #
        Block.__init__(self, *args, **kwargs)
    
    def construct(self, data=None):
        if __debug__ and logger.isEnabledFor(logging.DEBUG):        #pragma:nocover
            logger.debug( "Constructing %s '%s', from data=%s",
                          self.__class__.__name__, self.cname(), str(data) )
        if self._constructed:                                       #pragma:nocover
            return
        #
        _self_rule = self._rule
        self._rule = None
        super(Complementarity, self).construct()
        self._rule = _self_rule
        #
        if self._no_rule_init and _self_rule is not None:
            logger.warning(
                "The noruleinit keyword is being used in conjunction with " 
                "the rule keyword for complementarity '%s'; defaulting to "
                "rule-based construction", self.cname(True))
        if _self_rule is None and self._expr is None:
            if not self._no_rule_init:
                logger.warning(
                    "No construction rule or expression specified for "
                    "complementarity '%s'", self.cname(True))
            else:
                self._constructed=True
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
                        % ( self.cname(True), type(err).__name__, err ) )
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
                        % ( self.cname(True), idx, type(err).__name__, err ) )
                    raise

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
            return
        #
        if cc.__class__ is tuple:
            if cc is Complementarity.Skip:
                return
            elif len(cc) != 2:
                raise ValueError(
                    "Invalid tuple for Complementarity %s (expected 2-tuple):"
                    "\n\t%s" % (self.cname(True), cc) )
        elif cc.__class__ is list:
            #
            # Call add() recursively to apply the error same error
            # checks.
            #
            self.add(index, tuple(cc))
        elif cc is None:
                raise ValueError("""
Invalid complementarity condition.  The complementarity condition
is None instead of a 2-tuple.  Please modify your rule to return
Complementarity.Skip instead of None.

Error thrown for Complementarity "%s"
""" % ( self.cname(True), ) )
        else:
            raise ValueError(
                "Unexpected argument declaring Complementarity %s:\n\t%s" 
                % (self.cname(True), cc) )
        #
        self[index]._args = tuple( as_numeric(x) for x in cc )

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
             ("Index", self._index \
                  if self._index != UnindexedComponent_set else None),
             ("Active", self.active),
             ],
            iteritems(self._data),
            ( "Key","Arg0","Arg1","Active" ),
            lambda k, v: [ k,
                           v._args[0],
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

    def _default(self, idx):
        return self._data.setdefault(idx, _ComplementarityData(self))


register_component(Complementarity, "Complementarity conditions.")
