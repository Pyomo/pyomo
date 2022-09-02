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
from pyomo.contrib.cp.scheduling_expr import Step, Pulse
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
    CumulativeFunction)

from pyomo.environ import ConcreteModel

class TestStepFunction(unittest.TestCase):
    def get_model(self):
        m = ConcreteModel()
        m.a = IntervalVar()
        m.b = IntervalVar()

        return m

    def test_sum_step_and_pulse(self):
        m = self.get_model()
        expr = Step(m.a.start_time, height=4) + Pulse(m.b, height=-1)

        self.assertIsInstance(expr, CumulativeFunction)
        self.assertEqual(len(expr.args), 2)
        self.assertEqual(expr.nargs(), 2)
        self.assertIsInstance(expr.args[0], Step)
        self.assertIsInstance(expr.args[1], Pulse)

        self.assertEqual(str(expr), "Step(a.start_time, height=4) + "
                         "Pulse(b, height=-1)")

    def test_sum_in_place(self):
        m = self.get_model()
        expr = Step(m.a.start_time, height=4) + Pulse(m.b, height=-1)        
        expr += Step(0, 1)

        self.assertEqual(len(expr.args), 3)
        self.assertEqual(expr.nargs(), 3)
        self.assertIsInstance(expr.args[0], Step)
        self.assertIsInstance(expr.args[1], Pulse)
        self.assertIsInstance(expr.args[2], Step)

        self.assertEqual(str(expr), "Step(a.start_time, height=4) + "
                         "Pulse(b, height=-1) + Step(0, height=1)")

    def test_sum_pulses_in_place(self):
        m = self.get_model()
        p1 = Pulse(m.a, height=2)
        expr = p1
        
        self.assertEqual(len(expr.args), 1)
        self.assertEqual(expr.nargs(), 1)
        
        p2 = Pulse(m.b, height=3)
        expr += p2
        
        self.assertIsInstance(expr, CumulativeFunction)
        self.assertEqual(len(expr.args), 2)
        self.assertEqual(expr.nargs(), 2)
        self.assertIs(expr.args[0], p1)
        self.assertIs(expr.args[1], p2)

    def test_sum_steps_in_place(self):
        m = self.get_model()
        s1 = Step(m.a.end_time, height=2)
        expr = s1
        
        self.assertEqual(len(expr.args), 1)
        self.assertEqual(expr.nargs(), 1)
        
        s2 = Pulse(m.b.end_time, height=3)
        expr += s2
        
        self.assertIsInstance(expr, CumulativeFunction)
        self.assertEqual(len(expr.args), 2)
        self.assertEqual(expr.nargs(), 2)
        self.assertIs(expr.args[0], s1)
        self.assertIs(expr.args[1], s2)
