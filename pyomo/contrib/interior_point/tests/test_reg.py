#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.th as unittest
import pyomo.environ as pyo
from pyomo.core.base import ConcreteModel, Var, Constraint, Objective
from pyomo.common.dependencies import attempt_import

np, numpy_available = attempt_import('numpy', 'Interior point requires numpy',
        minimum_version='1.13.0')
scipy, scipy_available = attempt_import('scipy', 'Interior point requires scipy')
mumps, mumps_available = attempt_import('mumps', 'Interior point requires mumps')
if not (numpy_available and scipy_available):
    raise unittest.SkipTest('Interior point tests require numpy and scipy')
if scipy_available:
    from pyomo.contrib.interior_point.linalg.scipy_interface import ScipyInterface
if mumps_available:
    from pyomo.contrib.interior_point.linalg.mumps_interface import MumpsInterface

from pyomo.contrib.pynumero.asl import AmplInterface
asl_available = AmplInterface.available()
if not asl_available:
    raise unittest.SkipTest('Regularization tests require ASL')

from pyomo.contrib.interior_point.interior_point import InteriorPointSolver, InteriorPointStatus
from pyomo.contrib.interior_point.interface import InteriorPointInterface

from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
ma27_available = MA27Interface.available()
if ma27_available:
    from pyomo.contrib.interior_point.linalg.ma27_interface import InteriorPointMA27Interface

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


def make_model_2():
    m = ConcreteModel()
    m.x = Var(initialize=0.1, bounds=(0, 1))
    m.y = Var(initialize=0.1, bounds=(0, 1))
    m.obj = Objective(expr=-m.x**2 - m.y**2)
    m.c = Constraint(expr=m.y <= pyo.exp(-m.x))
    return m


class TestRegularization(unittest.TestCase):
    def _test_regularization(self, linear_solver):
        m = make_model()
        interface = InteriorPointInterface(m)
        ip_solver = InteriorPointSolver(linear_solver)
        ip_solver.set_interface(interface)

        interface.set_barrier_parameter(1e-1)

        # Evaluate KKT matrix before any iterations
        kkt = interface.evaluate_primal_dual_kkt_matrix()
        reg_coef = ip_solver.factorize(kkt)

        # Expected regularization coefficient:
        self.assertAlmostEqual(reg_coef, 1e-4)

        desired_n_neg_evals = (ip_solver.interface.n_eq_constraints() +
                               ip_solver.interface.n_ineq_constraints())

        # Expected inertia:
        n_pos_evals, n_neg_evals, n_null_evals = linear_solver.get_inertia()
        self.assertEqual(n_null_evals, 0)
        self.assertEqual(n_neg_evals, desired_n_neg_evals)

    @unittest.skipIf(not mumps_available, 'Mumps is not available')
    def test_mumps(self):
        solver = MumpsInterface()
        self._test_regularization(solver)

    @unittest.skipIf(not scipy_available, "Scipy is not available")
    def test_scipy(self):
        solver = ScipyInterface(compute_inertia=True)
        self._test_regularization(solver)

    @unittest.skipIf(not ma27_available, 'MA27 is not available')
    def test_ma27(self):
        solver = InteriorPointMA27Interface(icntl_options={1: 0, 2: 0})
        self._test_regularization(solver)

    def _test_regularization_2(self, linear_solver):
        m = make_model_2()
        interface = InteriorPointInterface(m)
        ip_solver = InteriorPointSolver(linear_solver)

        status = ip_solver.solve(interface)
        self.assertEqual(status, InteriorPointStatus.optimal)
        interface.load_primals_into_pyomo_model()
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, pyo.exp(-1))

    @unittest.skipIf(not mumps_available, 'Mumps is not available')
    def test_mumps_2(self):
        solver = MumpsInterface()
        self._test_regularization_2(solver)

    @unittest.skipIf(not scipy_available, "Scipy is not available")
    def test_scipy_2(self):
        solver = ScipyInterface(compute_inertia=True)
        self._test_regularization_2(solver)

    @unittest.skipIf(not ma27_available, 'MA27 is not available')
    def test_ma27_2(self):
        solver = InteriorPointMA27Interface(icntl_options={1: 0, 2: 0})
        self._test_regularization_2(solver)


if __name__ == '__main__':
    #
    unittest.main()
    # test_reg = TestRegularization()
    # test_reg.test_regularize_mumps()
    # test_reg.test_regularize_scipy()
    
