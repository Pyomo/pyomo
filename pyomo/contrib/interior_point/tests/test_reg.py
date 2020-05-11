import pyutilib.th as unittest
from pyomo.core.base import ConcreteModel, Var, Constraint, Objective
from pyomo.common.dependencies import attempt_import

np, numpy_available = attempt_import('numpy', 'Interior point requires numpy',
        minimum_version='1.13.0')
scipy, scipy_available = attempt_import('scipy', 'Interior point requires scipy')
mumps_interface, mumps_available = attempt_import(
        'pyomo.contrib.interior_point.linalg.mumps_interface',
        'Interior point requires mumps')
if not (numpy_available and scipy_available):
    raise unittest.SkipTest('Interior point tests require numpy and scipy')

from pyomo.contrib.pynumero.extensions.asl import AmplInterface
asl_available = AmplInterface.available()

from pyomo.contrib.interior_point.interior_point import InteriorPointSolver
from pyomo.contrib.interior_point.interface import InteriorPointInterface
from pyomo.contrib.interior_point.linalg.scipy_interface import ScipyInterface


def make_model():
    m = ConcreteModel()
    m.x = Var([1,2,3], initialize=0)
    m.f = Var([1,2,3], initialize=0)
    m.F = Var(initialize=0)
    m.f[1].fix(1)
    m.f[2].fix(2)

    m.sum_con = Constraint(expr= 
            (1 == m.x[1] + m.x[2] + m.x[3]))
    def bilin_rule(m, i):
        return m.F*m.x[i] == m.f[i]
    m.bilin_con = Constraint([1,2,3], rule=bilin_rule)

    m.obj = Objective(expr=m.F**2)

    return m


class TestRegularization(unittest.TestCase):
    @unittest.skipIf(not asl_available, 'asl is not available')
    @unittest.skipIf(not mumps_available, 'mumps is not available')
    def test_regularize_mumps(self):
        m = make_model()
        interface = InteriorPointInterface(m)

        linear_solver = mumps_interface.MumpsInterface()

        ip_solver = InteriorPointSolver(linear_solver,
                                        regularize_kkt=True)

        interface.set_barrier_parameter(1e-1)

        # Evaluate KKT matrix before any iterations
        kkt = interface.evaluate_primal_dual_kkt_matrix()
        with self.assertRaises(RuntimeError):
            # Should be Mumps error: -10, numerically singular
            # (Really the matrix is structurally singular, but it has 
            # enough symbolic zeros that the symbolic factorization can
            # be performed.
            linear_solver.do_symbolic_factorization(kkt)
            linear_solver.do_numeric_factorization(kkt)

        # Perform one iteration of interior point algorithm
        x, duals_eq, duals_ineq = ip_solver.solve(interface, max_iter=1)

#        # Expected regularization coefficient:
        self.assertAlmostEqual(ip_solver.reg_coef, 1e-4)

        desired_n_neg_evals = (ip_solver.interface._nlp.n_eq_constraints() +
                               ip_solver.interface._nlp.n_ineq_constraints())

        # Expected inertia:
        n_neg_evals = linear_solver.get_infog(12)
        n_null_evals = linear_solver.get_infog(28)
        self.assertEqual(n_null_evals, 0)
        self.assertEqual(n_neg_evals, desired_n_neg_evals)

        # Now perform two iterations of the interior point algorithm.
        # Because of the way the solve routine is written, updates to the
        # interface's variables don't happen until the start of the next
        # next iteration, meaning that the interface has been unaffected
        # by the single iteration performed above.
        x, duals_eq, duals_ineq = ip_solver.solve(interface, max_iter=10)

        # This will be the KKT matrix in iteration 1, without regularization
        kkt = interface.evaluate_primal_dual_kkt_matrix()
        linear_solver.do_symbolic_factorization(kkt)
        linear_solver.do_numeric_factorization(kkt)

        # Assert that one iteration with regularization was enough to get us
        # out of the pointof singularity/incorrect inertia
        n_neg_evals = linear_solver.get_infog(12)
        n_null_evals = linear_solver.get_infog(28)
        self.assertEqual(n_null_evals, 0)
        self.assertEqual(n_neg_evals, desired_n_neg_evals)


    @unittest.skipIf(not asl_available, 'asl is not available')
    @unittest.skipIf(not scipy_available, 'scipy is not available')
    def test_regularize_scipy(self):
        m = make_model()
        interface = InteriorPointInterface(m)

        linear_solver = ScipyInterface(compute_inertia=True)

        ip_solver = InteriorPointSolver(linear_solver,
                                        regularize_kkt=True)

        interface.set_barrier_parameter(1e-1)

        # Evaluate KKT matrix before any iterations
        kkt = interface.evaluate_primal_dual_kkt_matrix()
        with self.assertRaises(RuntimeError):
            linear_solver.do_symbolic_factorization(kkt)
            linear_solver.do_numeric_factorization(kkt)

        # Perform one iteration of interior point algorithm
        x, duals_eq, duals_ineq = ip_solver.solve(interface, max_iter=1)

        # Expected regularization coefficient:
        self.assertAlmostEqual(ip_solver.reg_coef, 1e-4)

        desired_n_neg_evals = (ip_solver.interface._nlp.n_eq_constraints() +
                               ip_solver.interface._nlp.n_ineq_constraints())

        # Expected inertia:
        n_pos_evals, n_neg_evals, n_null_evals = linear_solver.get_inertia()
        self.assertEqual(n_null_evals, 0)
        self.assertEqual(n_neg_evals, desired_n_neg_evals)

        # Now perform two iterations of the interior point algorithm.
        # Because of the way the solve routine is written, updates to the
        # interface's variables don't happen until the start of the next
        # next iteration, meaning that the interface has been unaffected
        # by the single iteration performed above.
        x, duals_eq, duals_ineq = ip_solver.solve(interface, max_iter=15)
        # ^ More iterations are required to get to a region of proper inertia
        # when using scipy. This is not unexpected

        # This will be the KKT matrix in iteration 1, without regularization
        kkt = interface.evaluate_primal_dual_kkt_matrix()
        linear_solver.do_symbolic_factorization(kkt)
        linear_solver.do_numeric_factorization(kkt)

        # Assert that one iteration with regularization was enough to get us
        # out of the point of singularity/incorrect inertia
        n_pos_evals, n_neg_evals, n_null_evals = linear_solver.get_inertia()
        self.assertEqual(n_null_evals, 0)
        self.assertEqual(n_neg_evals, desired_n_neg_evals)
        


if __name__ == '__main__':
    # 
    test_reg = TestRegularization()
    test_reg.test_regularize_mumps()
    test_reg.test_regularize_scipy()
    
