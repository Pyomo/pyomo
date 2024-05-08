import pyomo.environ as pe
from pyomo.contrib import coramin
from pyomo.contrib.coramin.algorithms.ecp_bounder import ECPBounder
from pyomo.common import unittest
from pyomo.contrib import appsi


gurobi_available = appsi.solvers.Gurobi().available()


class TestECPBounder(unittest.TestCase):
    @unittest.skipUnless(gurobi_available, 'gurobi is not available')
    def test_ecp_bounder(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=0.5 * (m.x**2 + m.y**2))
        m.c1 = pe.Constraint(expr=m.y >= (m.x - 1) ** 2)
        m.c2 = pe.Constraint(expr=m.y >= pe.exp(m.x))
        r = coramin.relaxations.relax(m)
        opt = ECPBounder(subproblem_solver=appsi.solvers.Gurobi())
        res = opt.solve(r)
        self.assertEqual(
            res.termination_condition, appsi.base.TerminationCondition.optimal
        )
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 1)
