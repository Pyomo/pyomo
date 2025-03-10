#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#
#  Development of this module was conducted as part of the Institute for
#  the Design of Advanced Energy Systems (IDAES) with support through the
#  Simulation-Based Engineering, Crosscutting Research Program within the
#  U.S. Department of Energy’s Office of Fossil Energy and Carbon Management.
#
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
import sys
from io import StringIO

import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.environ import (
    Var,
    ConcreteModel,
    Reals,
    ExternalFunction,
    Objective,
    Constraint,
    sqrt,
    sin,
    cos,
    SolverFactory,
    value,
)
from pyomo.contrib.trustregion.TRF import trust_region_method, _trf_config

logger = logging.getLogger('pyomo.contrib.trustregion')


@unittest.skipIf(
    not SolverFactory('ipopt').available(False), "The IPOPT solver is not available"
)
class TestTrustRegionConfig(unittest.TestCase):
    def setUp(self):
        self.m = ConcreteModel()
        self.m.z = Var(range(3), domain=Reals, initialize=2.0)
        self.m.x = Var(range(2), initialize=2.0)
        self.m.x[1] = 1.0

        def blackbox(a, b):
            return sin(a - b)

        def grad_blackbox(args, fixed):
            a, b = args[:2]
            return [cos(a - b), -cos(a - b)]

        self.m.bb = ExternalFunction(blackbox, grad_blackbox)

        self.m.obj = Objective(
            expr=(self.m.z[0] - 1.0) ** 2
            + (self.m.z[0] - self.m.z[1]) ** 2
            + (self.m.z[2] - 1.0) ** 2
            + (self.m.x[0] - 1.0) ** 4
            + (self.m.x[1] - 1.0) ** 6
        )
        self.m.c1 = Constraint(
            expr=(
                self.m.x[0] * self.m.z[0] ** 2 + self.m.bb(self.m.x[0], self.m.x[1])
                == 2 * sqrt(2.0)
            )
        )
        self.m.c2 = Constraint(
            expr=self.m.z[2] ** 4 * self.m.z[1] ** 2 + self.m.z[1] == 8 + sqrt(2.0)
        )
        self.decision_variables = [self.m.z[0], self.m.z[1], self.m.z[2]]

    def maprule(self, a, b):
        return a**2 + b**2

    def try_solve(self, **kwds):
        status = True
        try:
            self.TRF.solve(self.m, self.decision_variables, **kwds)
        except Exception as e:
            print('error calling TRF.solve: %s' % str(e))
            status = False
        return status

    def test_config_generator(self):
        CONFIG = _trf_config()
        self.assertEqual(CONFIG.solver, 'ipopt')
        self.assertFalse(CONFIG.keepfiles)
        self.assertFalse(CONFIG.tee)
        self.assertFalse(CONFIG.verbose)
        self.assertEqual(CONFIG.trust_radius, 1.0)
        self.assertEqual(CONFIG.minimum_radius, 1e-6)
        self.assertEqual(CONFIG.maximum_radius, 100.0)
        self.assertEqual(CONFIG.maximum_iterations, 50)
        self.assertEqual(CONFIG.feasibility_termination, 1e-5)
        self.assertEqual(CONFIG.step_size_termination, 1e-5)
        self.assertEqual(CONFIG.minimum_feasibility, 1e-4)
        self.assertEqual(CONFIG.switch_condition_kappa_theta, 0.1)
        self.assertEqual(CONFIG.switch_condition_gamma_s, 2.0)
        self.assertEqual(CONFIG.radius_update_param_gamma_c, 0.5)
        self.assertEqual(CONFIG.radius_update_param_gamma_e, 2.5)
        self.assertEqual(CONFIG.ratio_test_param_eta_1, 0.05)
        self.assertEqual(CONFIG.ratio_test_param_eta_2, 0.2)
        self.assertEqual(CONFIG.maximum_feasibility, 50.0)
        self.assertEqual(CONFIG.param_filter_gamma_theta, 0.01)
        self.assertEqual(CONFIG.param_filter_gamma_f, 0.01)

    def test_config_vars(self):
        # Initialized with 1.0
        self.TRF = SolverFactory('trustregion')
        self.assertEqual(self.TRF.config.trust_radius, 1.0)

        # Both persistent and local values should be 1.0
        solve_status = self.try_solve()
        self.assertTrue(solve_status)
        self.assertEqual(self.TRF.config.trust_radius, 1.0)

    def test_solve_with_new_kwdval(self):
        # Initialized with 1.0
        self.TRF = SolverFactory('trustregion')
        self.assertEqual(self.TRF.config.trust_radius, 1.0)

        # Set local to 2.0; persistent should still be 1.0
        solve_status = self.try_solve(trust_radius=2.0)
        self.assertTrue(solve_status)
        self.assertEqual(self.TRF.config.trust_radius, 1.0)

    def test_update_kwdval(self):
        # Initialized with 1.0
        self.TRF = SolverFactory('trustregion')
        self.assertEqual(self.TRF.config.trust_radius, 1.0)

        # Set persistent value to 4.0; local value should also be set to 4.0
        self.TRF.config.trust_radius = 4.0
        solve_status = self.try_solve()
        self.assertTrue(solve_status)
        self.assertEqual(self.TRF.config.trust_radius, 4.0)

    def test_update_kwdval_solve_with_new_kwdval(self):
        # Initialized with 1.0
        self.TRF = SolverFactory('trustregion')
        self.assertEqual(self.TRF.config.trust_radius, 1.0)

        # Set persistent value to 4.0;
        self.TRF.config.trust_radius = 4.0
        self.assertEqual(self.TRF.config.trust_radius, 4.0)

        # Set local to 2.0; persistent should still be 4.0
        solve_status = self.try_solve(trust_radius=2.0)
        self.assertTrue(solve_status)
        self.assertEqual(self.TRF.config.trust_radius, 4.0)

    def test_initialize_with_kwdval(self):
        # Initialized with 3.0
        self.TRF = SolverFactory('trustregion', trust_radius=3.0)
        self.assertEqual(self.TRF.config.trust_radius, 3.0)

        # Both persistent and local values should be set to 3.0
        solve_status = self.try_solve()
        self.assertTrue(solve_status)
        self.assertEqual(self.TRF.config.trust_radius, 3.0)

    def test_initialize_with_kwdval_solve_with_new_kwdval(self):
        # Initialized with 3.0
        self.TRF = SolverFactory('trustregion', trust_radius=3.0)
        self.assertEqual(self.TRF.config.trust_radius, 3.0)

        # Persistent should be 3.0, local should be 2.0
        solve_status = self.try_solve(trust_radius=2.0)
        self.assertTrue(solve_status)
        self.assertEqual(self.TRF.config.trust_radius, 3.0)


@unittest.skipIf(
    not SolverFactory('ipopt').available(False), "The IPOPT solver is not available"
)
class TestTrustRegionMethod(unittest.TestCase):
    def setUp(self):
        self.m = ConcreteModel()
        self.m.z = Var(range(3), domain=Reals, initialize=2.0)
        self.m.x = Var(range(2), initialize=2.0)
        self.m.x[1] = 1.0

        def blackbox(a, b):
            return sin(a - b)

        def grad_blackbox(args, fixed):
            a, b = args[:2]
            return [cos(a - b), -cos(a - b)]

        self.m.bb = ExternalFunction(blackbox, grad_blackbox)

        self.m.obj = Objective(
            expr=(self.m.z[0] - 1.0) ** 2
            + (self.m.z[0] - self.m.z[1]) ** 2
            + (self.m.z[2] - 1.0) ** 2
            + (self.m.x[0] - 1.0) ** 4
            + (self.m.x[1] - 1.0) ** 6
        )
        self.m.c1 = Constraint(
            expr=(
                self.m.x[0] * self.m.z[0] ** 2 + self.m.bb(self.m.x[0], self.m.x[1])
                == 2 * sqrt(2.0)
            )
        )
        self.m.c2 = Constraint(
            expr=self.m.z[2] ** 4 * self.m.z[1] ** 2 + self.m.z[1] == 8 + sqrt(2.0)
        )
        self.config = _trf_config()
        self.ext_fcn_surrogate_map_rule = lambda comp, ef: 0
        self.decision_variables = [self.m.z[0], self.m.z[1], self.m.z[2]]

    def test_solver(self):
        # Check the log contents
        log_OUTPUT = StringIO()
        # Check the printed contents
        print_OUTPUT = StringIO()
        sys.stdout = print_OUTPUT
        with LoggingIntercept(log_OUTPUT, 'pyomo.contrib.trustregion', logging.INFO):
            result = trust_region_method(
                self.m,
                self.decision_variables,
                self.ext_fcn_surrogate_map_rule,
                self.config,
            )
        sys.stdout = sys.__stdout__
        # Check the log to make sure it is capturing
        self.assertIn('Iteration 0', log_OUTPUT.getvalue())
        # Check the printed output
        self.assertIn('EXIT: Optimal solution found.', print_OUTPUT.getvalue())
        # The names of both models should be the same
        self.assertEqual(result.name, self.m.name)
        # The values should not be the same
        self.assertNotEqual(value(result.obj), value(self.m.obj))
