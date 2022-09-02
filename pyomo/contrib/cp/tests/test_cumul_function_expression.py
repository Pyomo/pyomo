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
from pyomo.contrib.cp import IntervalVar, CumulativeFunctionExpression
from pyomo.contrib.cp.scheduling_expr import AlwaysIn, Step, Pulse
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
    CumulativeFunction)

from pyomo.environ import ConcreteModel, LogicalConstraint

class TestCumulativeFunctionExpression(unittest.TestCase):
    def get_model(self):
        m = ConcreteModel()
        m.a = IntervalVar()
        m.b = IntervalVar()

        return m

    def test_modify_cumul_function_expr(self):
        m = self.get_model()
        m.resource1 = CumulativeFunctionExpression()
        
        self.assertIsInstance(m.resource1, CumulativeFunctionExpression)
        self.assertIsNone(m.resource1.expr)

        m.resource1.expr = Step(m.a.start_time, height=4) + \
                           Pulse(m.b, height=-1) + Step(0, 1)
        self.assertIsInstance(m.resource1.expr, CumulativeFunction)
        self.assertEqual(len(m.resource1.expr.args), 3)
        self.assertIsInstance(m.resource1.expr.args[0], Step)
        self.assertIsInstance(m.resource1.expr.args[1], Pulse)
        self.assertIsInstance(m.resource1.expr.args[2], Step)
       
        self.assertEqual(str(m.resource1.expr), 
                         "Step(a.start_time, height=4) + "
                         "Pulse(b, height=-1) + Step(0, height=1)")

    def test_scalar_cumul_function_expr(self):
        m = self.get_model()

        m.resource1 = CumulativeFunctionExpression(expr=Step(m.a, height=4) +
                                                   Pulse(m.b, height=-1))
        self.assertIs(m.resource1.ctype, CumulativeFunctionExpression)
        self.assertIsInstance(m.resource1.expr, CumulativeFunctionExpression)
        self.assertEqual(len(m.resource1.expr.args), 2)
        self.assertIsInstance(m.resource1.expr.args[0], StepAtExpression)
        self.assertIsInstance(m.resource1.expr.args[1], PulseExpression)

        self.assertEqual(str(m.resource1.expr), 
                         "Step(a.start_time, height=4) + "
                         "Pulse(b, height=-1)")

    def test_indexed_cumul_function_expr(self):
        # TODO
        pass

    
class TestAlwaysIn(unittest.TestCase):
    def get_model(self):
        m = ConcreteModel()
        m.a = IntervalVar()
        m.b = IntervalVar()

        m.resource1 = CumulativeFunctionExpression(expr=Step(m.a, height=4) +
                                                   Pulse(m.b, height=-1) +
                                                   Step(0,1))
        return m

    def check_always_in_constraint(self, cons):
        self.assertIsInstance(cons.expr, AlwaysIn)

    def test_cumul_func_within(self):
        m = self.get_model()

        m.c = LogicalConstraint(
            expr=m.resource1.within(bounds=range(0, 3),
                                    times=range(0, 24)))
        self.check_always_in_constraint(m.c)

    def test_always_in_logical_constraint_variable_step_func(self):
        m = self.get_model()

        m.c = LogicalConstraint(expr=AlwaysIn([Step(m.a, height=4),
                                               Pulse(m.b, height=-1),
                                               Step(0,1)],
                                               times=range(0, 24),
                                               bounds=range(0, 3)))
        self.check_always_in_constraint(m.c)
