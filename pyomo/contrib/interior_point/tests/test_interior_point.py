import pyutilib.th as unittest
import pyomo.environ as pe
from pyomo.common.dependencies import attempt_import

np, numpy_availalbe = attempt_import('numpy', 'Interior point requires numpy', minimum_version='1.13.0')
scipy, scipy_available = attempt_import('scipy', 'Interior point requires scipy')
mumps_interface, mumps_available = attempt_import('pyomo.contrib.interior_point.linalg.mumps_interface', 'Interior point requires mumps')
if not (numpy_availalbe and scipy_available):
    raise unittest.SkipTest('Interior point tests require numpy and scipy')

import numpy as np

from pyomo.contrib.pynumero.extensions.asl import AmplInterface
asl_available = AmplInterface.available()

from pyomo.contrib.interior_point.interior_point import (InteriorPointSolver,
                                                         process_init,
                                                         process_init_duals,
                                                         fraction_to_the_boundary,
                                                         _fraction_to_the_boundary_helper_lb,
                                                         _fraction_to_the_boundary_helper_ub)
from pyomo.contrib.interior_point.interface import InteriorPointInterface
from pyomo.contrib.interior_point.linalg.scipy_interface import ScipyInterface
from pyomo.contrib.pynumero.interfaces.utils import build_bounds_mask, build_compression_matrix


class TestSolveInteriorPoint(unittest.TestCase):
    @unittest.skipIf(not asl_available, 'asl is not available')
    def test_solve_interior_point_1(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pe.Constraint(expr=m.y == pe.exp(m.x))
        m.c2 = pe.Constraint(expr=m.y >= (m.x - 1)**2)
        interface = InteriorPointInterface(m)
        linear_solver = ScipyInterface()
        linear_solver.compute_inertia = True
        ip_solver = InteriorPointSolver(linear_solver)
#        x, duals_eq, duals_ineq = solve_interior_point(interface, linear_solver)
        x, duals_eq, duals_ineq = ip_solver.solve(interface)
        self.assertAlmostEqual(x[0], 0)
        self.assertAlmostEqual(x[1], 1)
        self.assertAlmostEqual(duals_eq[0], -1-1.0/3.0)
        self.assertAlmostEqual(duals_ineq[0], 2.0/3.0)

    @unittest.skipIf(not asl_available, 'asl is not available')
    @unittest.skipIf(not mumps_available, 'mumps is not available')
    def test_solve_interior_point_2(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(1, 4))
        m.obj = pe.Objective(expr=m.x**2)
        interface = InteriorPointInterface(m)
        linear_solver = mumps_interface.MumpsInterface()
        ip_solver = InteriorPointSolver(linear_solver)
#        x, duals_eq, duals_ineq = solve_interior_point(interface, linear_solver)
        x, duals_eq, duals_ineq = ip_solver.solve(interface)
        self.assertAlmostEqual(x[0], 1)


class TestProcessInit(unittest.TestCase):
    def testprocess_init(self):
        lb = np.array([-np.inf, -np.inf,     -2, -2], dtype=np.double)
        ub = np.array([ np.inf,       2, np.inf,  2], dtype=np.double)

        x = np.array([       0,       0,      0,  0], dtype=np.double)
        process_init(x, lb, ub)
        self.assertTrue(np.allclose(x, np.array([0, 0, 0, 0], dtype=np.double)))

        x = np.array([      -2,      -2,     -2,  -2], dtype=np.double)
        process_init(x, lb, ub)
        self.assertTrue(np.allclose(x, np.array([-2, -2, -1, 0], dtype=np.double)))

        x = np.array([      -3,      -3,     -3,  -3], dtype=np.double)
        process_init(x, lb, ub)
        self.assertTrue(np.allclose(x, np.array([-3, -3, -1, 0], dtype=np.double)))

        x = np.array([       2,       2,      2,   2], dtype=np.double)
        process_init(x, lb, ub)
        self.assertTrue(np.allclose(x, np.array([2, 1, 2, 0], dtype=np.double)))

        x = np.array([       3,       3,      3,   3], dtype=np.double)
        process_init(x, lb, ub)
        self.assertTrue(np.allclose(x, np.array([3, 1, 3, 0], dtype=np.double)))

    def testprocess_init_duals(self):
        x = np.array([0, 0, 0, 0], dtype=np.double)
        process_init_duals(x)
        self.assertTrue(np.allclose(x, np.array([1, 1, 1, 1], dtype=np.double)))

        x = np.array([-1, -1, -1, -1], dtype=np.double)
        process_init_duals(x)
        self.assertTrue(np.allclose(x, np.array([1, 1, 1, 1], dtype=np.double)))

        x = np.array([2, 2, 2, 2], dtype=np.double)
        process_init_duals(x)
        self.assertTrue(np.allclose(x, np.array([2, 2, 2, 2], dtype=np.double)))

        
class TestFractionToTheBoundary(unittest.TestCase):
    def test_fraction_to_the_boundary_helper_lb(self):
        tau = 0.9
        x = np.array([0, 0, 0, 0], dtype=np.double)
        xl = np.array([-np.inf, -1, -np.inf, -1], dtype=np.double)
        xl_compression_matrix = build_compression_matrix(build_bounds_mask(xl))
        xl_compressed = xl_compression_matrix * xl

        delta_x = np.array([-0.1, -0.1, -0.1, -0.1], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_lb(tau, x, delta_x, xl_compressed, xl_compression_matrix)
        self.assertAlmostEqual(alpha, 1)

        delta_x = np.array([-1, -1, -1, -1], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_lb(tau, x, delta_x, xl_compressed, xl_compression_matrix)
        self.assertAlmostEqual(alpha, 0.9)

        delta_x = np.array([-10, -10, -10, -10], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_lb(tau, x, delta_x, xl_compressed, xl_compression_matrix)
        self.assertAlmostEqual(alpha, 0.09)

        delta_x = np.array([1, 1, 1, 1], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_lb(tau, x, delta_x, xl_compressed, xl_compression_matrix)
        self.assertAlmostEqual(alpha, 1)

        delta_x = np.array([-10, 1, -10, 1], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_lb(tau, x, delta_x, xl_compressed, xl_compression_matrix)
        self.assertAlmostEqual(alpha, 1)

        delta_x = np.array([-10, -1, -10, -1], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_lb(tau, x, delta_x, xl_compressed, xl_compression_matrix)
        self.assertAlmostEqual(alpha, 0.9)

        delta_x = np.array([1, -10, 1, -1], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_lb(tau, x, delta_x, xl_compressed, xl_compression_matrix)
        self.assertAlmostEqual(alpha, 0.09)

    def test_fraction_to_the_boundary_helper_ub(self):
        tau = 0.9
        x = np.array([0, 0, 0, 0], dtype=np.double)
        xu = np.array([np.inf, 1, np.inf, 1], dtype=np.double)
        xu_compression_matrix = build_compression_matrix(build_bounds_mask(xu))
        xu_compressed = xu_compression_matrix * xu

        delta_x = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_ub(tau, x, delta_x, xu_compressed, xu_compression_matrix)
        self.assertAlmostEqual(alpha, 1)

        delta_x = np.array([1, 1, 1, 1], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_ub(tau, x, delta_x, xu_compressed, xu_compression_matrix)
        self.assertAlmostEqual(alpha, 0.9)

        delta_x = np.array([10, 10, 10, 10], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_ub(tau, x, delta_x, xu_compressed, xu_compression_matrix)
        self.assertAlmostEqual(alpha, 0.09)

        delta_x = np.array([-1, -1, -1, -1], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_ub(tau, x, delta_x, xu_compressed, xu_compression_matrix)
        self.assertAlmostEqual(alpha, 1)

        delta_x = np.array([10, -1, 10, -1], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_ub(tau, x, delta_x, xu_compressed, xu_compression_matrix)
        self.assertAlmostEqual(alpha, 1)

        delta_x = np.array([10, 1, 10, 1], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_ub(tau, x, delta_x, xu_compressed, xu_compression_matrix)
        self.assertAlmostEqual(alpha, 0.9)

        delta_x = np.array([-1, 10, -1, 1], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_ub(tau, x, delta_x, xu_compressed, xu_compression_matrix)
        self.assertAlmostEqual(alpha, 0.09)
