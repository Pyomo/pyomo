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

import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.precedence_expressions import (
    BeforeExpression,
    AtExpression,
)
from pyomo.environ import ConcreteModel, LogicalConstraint


class TestPrecedenceRelationships(unittest.TestCase):
    def get_model(self):
        m = ConcreteModel()
        m.a = IntervalVar()
        m.b = IntervalVar()

        return m

    def test_start_before_start(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.a.start_time.before(m.b.start_time))

        self.assertIsInstance(m.c.expr, BeforeExpression)
        self.assertEqual(len(m.c.expr.args), 3)
        self.assertEqual(m.c.expr.nargs(), 3)
        self.assertIs(m.c.expr.args[0], m.a.start_time)
        self.assertIs(m.c.expr.args[1], m.b.start_time)
        self.assertEqual(m.c.expr.delay, 0)

        self.assertEqual(str(m.c.expr), "a.start_time <= b.start_time")

    def test_start_before_end(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.a.start_time.before(m.b.end_time))

        self.assertIsInstance(m.c.expr, BeforeExpression)
        self.assertEqual(len(m.c.expr.args), 3)
        self.assertEqual(m.c.expr.nargs(), 3)
        self.assertIs(m.c.expr.args[0], m.a.start_time)
        self.assertIs(m.c.expr.args[1], m.b.end_time)
        self.assertEqual(m.c.expr.delay, 0)

        self.assertEqual(str(m.c.expr), "a.start_time <= b.end_time")

    def test_start_after_start(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.a.start_time.after(m.b.start_time))

        self.assertIsInstance(m.c.expr, BeforeExpression)
        self.assertEqual(len(m.c.expr.args), 3)
        self.assertIs(m.c.expr.args[0], m.b.start_time)
        self.assertIs(m.c.expr.args[1], m.a.start_time)
        self.assertEqual(m.c.expr.delay, 0)

        self.assertEqual(str(m.c.expr), "b.start_time <= a.start_time")

    def test_start_after_end(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.a.start_time.after(m.b.end_time, delay=2))

        self.assertIsInstance(m.c.expr, BeforeExpression)
        self.assertEqual(len(m.c.expr.args), 3)
        self.assertIs(m.c.expr.args[0], m.b.end_time)
        self.assertIs(m.c.expr.args[1], m.a.start_time)
        self.assertEqual(m.c.expr.delay, 2)

        self.assertEqual(str(m.c.expr), "b.end_time + 2 <= a.start_time")

    def test_start_at_start(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.a.start_time.at(m.b.start_time))

        self.assertIsInstance(m.c.expr, AtExpression)
        self.assertEqual(len(m.c.expr.args), 3)
        self.assertEqual(m.c.expr.nargs(), 3)
        self.assertIs(m.c.expr.args[0], m.a.start_time)
        self.assertIs(m.c.expr.args[1], m.b.start_time)
        self.assertEqual(m.c.expr.delay, 0)

        self.assertEqual(str(m.c.expr), "a.start_time == b.start_time")

    def test_start_at_end(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.a.start_time.at(m.b.end_time, delay=-1))

        self.assertIsInstance(m.c.expr, AtExpression)
        self.assertEqual(len(m.c.expr.args), 3)
        self.assertEqual(m.c.expr.nargs(), 3)
        self.assertIs(m.c.expr.args[0], m.a.start_time)
        self.assertIs(m.c.expr.args[1], m.b.end_time)
        self.assertEqual(m.c.expr.delay, -1)

        self.assertEqual(str(m.c.expr), "a.start_time - 1 == b.end_time")

    def test_end_before_start(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.a.end_time.before(m.b.start_time, delay=3))

        self.assertIsInstance(m.c.expr, BeforeExpression)
        self.assertEqual(len(m.c.expr.args), 3)
        self.assertIs(m.c.expr.args[0], m.a.end_time)
        self.assertIs(m.c.expr.args[1], m.b.start_time)
        self.assertEqual(m.c.expr.delay, 3)

        self.assertEqual(str(m.c.expr), "a.end_time + 3 <= b.start_time")

    def test_end_at_start(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.a.end_time.at(m.b.start_time, delay=4))

        self.assertIsInstance(m.c.expr, AtExpression)
        self.assertEqual(len(m.c.expr.args), 3)
        self.assertIs(m.c.expr.args[0], m.a.end_time)
        self.assertIs(m.c.expr.args[1], m.b.start_time)
        self.assertEqual(m.c.expr.delay, 4)

        self.assertEqual(str(m.c.expr), "a.end_time + 4 == b.start_time")

    def test_end_after_start(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.a.end_time.after(m.b.start_time, delay=-2))

        self.assertIsInstance(m.c.expr, BeforeExpression)
        self.assertEqual(len(m.c.expr.args), 3)
        self.assertIs(m.c.expr.args[0], m.b.start_time)
        self.assertIs(m.c.expr.args[1], m.a.end_time)
        self.assertEqual(m.c.expr.delay, -2)

        self.assertEqual(str(m.c.expr), "b.start_time - 2 <= a.end_time")

    def test_end_before_end(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.a.end_time.before(m.b.end_time, delay=-5))

        self.assertIsInstance(m.c.expr, BeforeExpression)
        self.assertEqual(len(m.c.expr.args), 3)
        self.assertIs(m.c.expr.args[0], m.a.end_time)
        self.assertIs(m.c.expr.args[1], m.b.end_time)
        self.assertEqual(m.c.expr.delay, -5)

        self.assertEqual(str(m.c.expr), "a.end_time - 5 <= b.end_time")

    def test_end_at_end(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.a.end_time.at(m.b.end_time, delay=-3))

        self.assertIsInstance(m.c.expr, AtExpression)
        self.assertEqual(len(m.c.expr.args), 3)
        self.assertIs(m.c.expr.args[0], m.a.end_time)
        self.assertIs(m.c.expr.args[1], m.b.end_time)
        self.assertEqual(m.c.expr.delay, -3)

        self.assertEqual(str(m.c.expr), "a.end_time - 3 == b.end_time")

    def test_end_after_end(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.a.end_time.after(m.b.end_time))

        self.assertIsInstance(m.c.expr, BeforeExpression)
        self.assertEqual(len(m.c.expr.args), 3)
        self.assertIs(m.c.expr.args[0], m.b.end_time)
        self.assertIs(m.c.expr.args[1], m.a.end_time)
        self.assertEqual(m.c.expr.delay, 0)

        self.assertEqual(str(m.c.expr), "b.end_time <= a.end_time")
