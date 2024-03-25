from pyomo.common import unittest
import pyomo.environ as pe
from pyomo.contrib.coramin.heuristics.diving import run_diving_heuristic


ipopt_available = pe.SolverFactory('ipopt').available()


@unittest.skipUnless(ipopt_available, 'ipopt is not available')
class TestDiving(unittest.TestCase):
    def test_diving_1(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z1 = pe.Var(domain=pe.Binary)
        m.z2 = pe.Var(domain=pe.Binary)
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pe.Constraint(expr=(m.y - pe.exp(m.x)) * m.z1 >= 0)
        m.c2 = pe.Constraint(expr=(m.y - (m.x - 1) ** 2) * m.z1 >= 0)
        m.c3 = pe.Constraint(expr=(m.y - m.x - 2) * m.z2 >= 0)
        m.c4 = pe.Constraint(expr=(m.y + m.x - 2) * m.z2 >= 0)
        m.c5 = pe.Constraint(expr=m.z1 + m.z2 == 1)
        obj, sol = run_diving_heuristic(m)
        self.assertAlmostEqual(obj, 1)
        self.assertAlmostEqual(sol[m.x], 0)
        self.assertAlmostEqual(sol[m.y], 1)
        self.assertAlmostEqual(sol[m.z1], 1)
        self.assertAlmostEqual(sol[m.z2], 0)
