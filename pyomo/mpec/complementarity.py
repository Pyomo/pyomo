#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from six import iteritems
from collections import namedtuple

from pyomo.common.log import is_debug_set
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr import current as EXPR
from pyomo.core.expr.numvalue import ZeroConstant, native_numeric_types, as_numeric
from pyomo.core import Constraint, Var, Block, Set
from pyomo.core.base.plugin import ModelComponentFactory
from pyomo.core.base.block import _BlockData
from pyomo.core.base.util import (
    disable_methods, Initializer, IndexedCallInitializer, CountedCallInitializer
)

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
            self.c._complementarity_type = 3
        elif _e2[2] is None:
            self.c = Constraint(expr=_e2[0] <= _e2[1])
            self.c._complementarity_type = 1
        elif _e2[0] is None:
            self.c = Constraint(expr=- _e2[2] <= - _e2[1])
            self.c._complementarity_type = 1
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

    def set_value(self, cc):
        """
        Add a complementarity condition with a specified index.
        """
        if cc.__class__ is ComplementarityTuple:
            #
            # The ComplementarityTuple has a fixed length, so we initialize
            # the _args component and return
            #
            self._args = ( as_numeric(cc.arg0), as_numeric(cc.arg1) )
        #
        elif cc.__class__ is tuple:
            if len(cc) != 2:
                raise ValueError(
                    "Invalid tuple for Complementarity %s (expected 2-tuple):"
                    "\n\t%s" % (self.name, cc) )
            self._args = tuple( as_numeric(x) for x in cc )
        elif cc is Complementarity.Skip:
            del self.parent_component()[self.index()]
        elif cc.__class__ is list:
            #
            # Call set_value() recursively to apply the error same error
            # checks.
            #
            return self.set_value(tuple(cc))
        else:
            raise ValueError(
                "Unexpected value for Complementarity %s:\n\t%s"
                % (self.name, cc) )


@ModelComponentFactory.register("Complementarity conditions.")
class Complementarity(Block):

    _ComponentDataClass = _ComplementarityData

    def __new__(cls, *args, **kwds):
        if cls != Complementarity:
            return super(Complementarity, cls).__new__(cls)
        if args == ():
            return super(Complementarity, cls).__new__(AbstractSimpleComplementarity)
        else:
            return super(Complementarity, cls).__new__(IndexedComplementarity)

    @staticmethod
    def _complementarity_rule(b, *idx):
        _rule = b.parent_component()._init_rule
        if _rule is None:
            return
        cc = _rule(b.parent_block(), idx)
        if cc is None:
            raise ValueError("""
Invalid complementarity condition.  The complementarity condition
is None instead of a 2-tuple.  Please modify your rule to return
Complementarity.Skip instead of None.

Error thrown for Complementarity "%s".""" % ( b.name, ) )
        b.set_value(cc)

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('ctype', Complementarity)
        kwargs.setdefault('dense', False)
        _init = tuple( _arg for _arg in (
            kwargs.pop('initialize', None),
            kwargs.pop('rule', None),
            kwargs.pop('expr', None) ) if _arg is not None )
        if len(_init) > 1:
            raise ValueError(
                "Duplicate initialization: Complementarity() only accepts "
                "one of 'initialize=', 'rule=', and 'expr='")
        elif _init:
            _init = _init[0]
        else:
            _init = None

        self._init_rule = Initializer(
            _init, treat_sequences_as_mappings=False, allow_generators=True
        )

        if self._init_rule is not None:
            kwargs['rule'] = Complementarity._complementarity_rule
        Block.__init__(self, *args, **kwargs)

        # HACK to make the "counted call" syntax work.  We wait until
        # after the base class is set up so that is_indexed() is
        # reliable.
        if self._init_rule is not None \
           and self._init_rule.__class__ is IndexedCallInitializer:
            self._init_rule = CountedCallInitializer(self, self._init_rule)


    def add(self, index, cc):
        """
        Add a complementarity condition with a specified index.
        """
        if cc is Complementarity.Skip:
            return
        _block = self[index]
        _block.set_value(cc)
        return _block

    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        _table_data = lambda k, v: [
            v._args[0], v._args[1], v.active,
        ]

        # This is a bit weird, but is being implemented to preserve
        # backwards compatibility.  The Complementarity transformation
        # is "in place", in that modeling components are added to this
        # block.  If the transformation has been executed (or if any
        # components have been added to the block), we want to output
        # the block components as well as the normal complementarity
        # table.
        #
        # TODO: In the future we should probably reconsider how
        # Complementarity is implemented and move away from this
        # paradigm.
        #
        # FIXME: remove the _transformed check and only invoke
        # _pprint_callback if there are components (requires baseline
        # updates and a check that we do not break anything in the
        # Book).
        _transformed = not issubclass(self.ctype, Complementarity)
        def _conditional_block_printer(ostream, idx, data):
            if _transformed or len(data.component_map()):
                self._pprint_callback(ostream, idx, data)

        return (
            [("Size", len(self)),
             ("Index", self._index if self.is_indexed() else None),
             ("Active", self.active),
             ],
            iteritems(self._data),
            ( "Arg0","Arg1","Active" ),
            (_table_data, _conditional_block_printer),
            )


class SimpleComplementarity(_ComplementarityData, Complementarity):

    def __init__(self, *args, **kwds):
        _ComplementarityData.__init__(self, self)
        Complementarity.__init__(self, *args, **kwds)
        self._data[None] = self


@disable_methods({'add', 'set_value', 'to_standard_form'})
class AbstractSimpleComplementarity(SimpleComplementarity):
    pass


class IndexedComplementarity(Complementarity):
    pass


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
        # disable the implicit rule; construct will exhaust the
        # user-provided rule, and then subsequent attempts to add a CC
        # will bypass the rule
        self._rule = None

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
        if is_debug_set(logger):
            logger.debug("Constructing complementarity list %s", self.name)
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        self._constructed=True

        if self._init_rule is not None:
            _init = self._init_rule(self.parent_block(), ())
            for cc in iter(_init):
                if cc is ComplementarityList.End:
                    break
                if cc is Complementarity.Skip:
                    continue
                self.add(cc)

        timer.report()

