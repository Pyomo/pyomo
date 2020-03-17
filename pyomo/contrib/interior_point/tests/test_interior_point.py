import pyutilib.th as unittest
import pyomo.environ as pe
from pyomo.common.dependencies import attempt_import


np, numpy_availalbe = attempt_import('numpy', 'Interior point requires numpy', minimum_version='1.13.0')
scipy, scipy_available = attempt_import('scipy', 'Interior point requires scipy')


if not (numpy_availalbe and scipy_available):
    raise unittest.SkipTest('Interior point tests require numpy and scipy')


from pyomo.contrib.interior_point.interior_point import solve_interior_point
from pyomo.contrib.interior_point.interface import InteriorPointInterface
from pyomo.contrib.interior_point.linalg.scipy_interface import ScipyInterface


class TestInteriorPoint(unittest.TestCase):
    def test_solve_1(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pe.Constraint(expr=m.y == pe.exp(m.x))
        m.c2 = pe.Constraint(expr=m.y >= (m.x - 1)**2)
        interface = InteriorPointInterface(m)
        linear_solver = ScipyInterface()
        x, duals_eq, duals_ineq = solve_interior_point(interface, linear_solver)
        self.assertAlmostEqual(x[0], 0)
        self.assertAlmostEqual(x[1], 1)
        self.assertAlmostEqual(duals_eq[0], -1-1.0/3.0)
        self.assertAlmostEqual(duals_ineq[0], 2.0/3.0)
