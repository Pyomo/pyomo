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

from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.contrib.cp.interval_var import IntervalVar
from pyomo.contrib.cp.scheduling_expr.sequence_expressions import (
    NoOverlapExpression,
    FirstInSequenceExpression,
    LastInSequenceExpression,
    BeforeInSequenceExpression,
    PredecessorToExpression,
    no_overlap,
    predecessor_to,
    before_in_sequence,
    first_in_sequence,
    last_in_sequence,
)
from pyomo.contrib.cp.sequence_var import SequenceVar, IndexedSequenceVar
from pyomo.environ import ConcreteModel, Integers, LogicalConstraint, Set, value, Var


class TestSequenceVarExpressions(unittest.TestCase):
    def get_model(self):
        m = ConcreteModel()
        m.S = Set(initialize=range(3))
        m.i = IntervalVar(m.S, start=(0, 5))
        m.seq = SequenceVar(expr=[m.i[j] for j in m.S])

        return m

    def test_no_overlap(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=no_overlap(m.seq))
        e = m.c.expr

        self.assertIsInstance(e, NoOverlapExpression)
        self.assertEqual(e.nargs(), 1)
        self.assertEqual(len(e.args), 1)
        self.assertIs(e.args[0], m.seq)

        self.assertEqual(str(e), "no_overlap(seq)")

    def test_first_in_sequence(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=first_in_sequence(m.i[2], m.seq))
        e = m.c.expr

        self.assertIsInstance(e, FirstInSequenceExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(len(e.args), 2)
        self.assertIs(e.args[0], m.i[2])
        self.assertIs(e.args[1], m.seq)

        self.assertEqual(str(e), "first_in(i[2], seq)")

    def test_last_in_sequence(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=last_in_sequence(m.i[0], m.seq))
        e = m.c.expr

        self.assertIsInstance(e, LastInSequenceExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(len(e.args), 2)
        self.assertIs(e.args[0], m.i[0])
        self.assertIs(e.args[1], m.seq)

        self.assertEqual(str(e), "last_in(i[0], seq)")
    
    def test_before_in_sequence(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=before_in_sequence(m.i[1], m.i[0], m.seq))
        e = m.c.expr

        self.assertIsInstance(e, BeforeInSequenceExpression)
        self.assertEqual(e.nargs(), 3)
        self.assertEqual(len(e.args), 3)
        self.assertIs(e.args[0], m.i[1])
        self.assertIs(e.args[1], m.i[0])
        self.assertIs(e.args[2], m.seq)

        self.assertEqual(str(e), "before_in(i[1], i[0], seq)")

    def test_predecessor_in_sequence(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=predecessor_to(m.i[0], m.i[1], m.seq))
        e = m.c.expr
        
        self.assertIsInstance(e, PredecessorToExpression)
        self.assertEqual(e.nargs(), 3)
        self.assertEqual(len(e.args), 3)
        self.assertIs(e.args[0], m.i[0])
        self.assertIs(e.args[1], m.i[1])
        self.assertIs(e.args[2], m.seq)
        
        self.assertEqual(str(e), "predecessor_to(i[0], i[1], seq)")
