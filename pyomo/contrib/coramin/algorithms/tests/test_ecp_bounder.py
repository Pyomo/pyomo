import pyomo.environ as pe
import coramin
from coramin.algorithms.ecp_bounder import ECPBounder
import unittest
import logging
from pyomo.contrib import appsi


logging.basicConfig(level=logging.INFO)


class TestECPBounder(unittest.TestCase):
    def test_ecp_bounder(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=0.5*(m.x**2 + m.y**2))
        m.c1 = pe.Constraint(expr=m.y >= (m.x - 1)**2)
        m.c2 = pe.Constraint(expr=m.y >= pe.exp(m.x))
        coramin.relaxations.relax(m, in_place=True)
        opt = ECPBounder(subproblem_solver=appsi.solvers.Gurobi())
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, appsi.base.TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 1)
