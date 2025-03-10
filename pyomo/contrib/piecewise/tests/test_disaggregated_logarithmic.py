#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise.tests import models
import pyomo.contrib.piecewise.tests.common_tests as ct
from pyomo.core.base import TransformationFactory
from pyomo.environ import SolverFactory, Var, Constraint
from pyomo.core.expr.compare import assertExpressionsEqual


class TestTransformPiecewiseModelToNestedInnerRepnMIP(unittest.TestCase):
    def check_pw_log(self, m):
        z = m.pw_log.get_transformation_var(m.log_expr)
        self.assertIsInstance(z, Var)
        # Now we can use those Vars to check on what the transformation created
        log_block = z.parent_block()

        # We should have three Vars, two of which are indexed, and five
        # Constraints, three of which are indexed

        self.assertEqual(len(log_block.component_map(Var)), 3)
        self.assertEqual(len(log_block.component_map(Constraint)), 5)

        # Constants
        simplex_count = 3
        log_simplex_count = 2
        simplex_point_count = 2

        # Substitute var
        self.assertIsInstance(log_block.substitute_var, Var)
        self.assertIs(m.obj.expr.expr, log_block.substitute_var)
        # Binaries
        self.assertIsInstance(log_block.binaries, Var)
        self.assertEqual(len(log_block.binaries), log_simplex_count)
        # Lambdas
        self.assertIsInstance(log_block.lambdas, Var)
        self.assertEqual(len(log_block.lambdas), simplex_count * simplex_point_count)
        for l in log_block.lambdas.values():
            self.assertEqual(l.lb, 0)
            self.assertEqual(l.ub, 1)

        # Convex combo constraint
        self.assertIsInstance(log_block.convex_combo, Constraint)
        assertExpressionsEqual(
            self,
            log_block.convex_combo.expr,
            log_block.lambdas[0, 0]
            + log_block.lambdas[0, 1]
            + log_block.lambdas[1, 0]
            + log_block.lambdas[1, 1]
            + log_block.lambdas[2, 0]
            + log_block.lambdas[2, 1]
            == 1,
        )

        # Set substitute constraint
        self.assertIsInstance(log_block.set_substitute, Constraint)
        assertExpressionsEqual(
            self,
            log_block.set_substitute.expr,
            log_block.substitute_var
            == log_block.lambdas[0, 0] * m.f1(1)
            + log_block.lambdas[1, 0] * m.f2(3)
            + log_block.lambdas[2, 0] * m.f3(6)
            + log_block.lambdas[0, 1] * m.f1(3)
            + log_block.lambdas[1, 1] * m.f2(6)
            + log_block.lambdas[2, 1] * m.f3(10),
            places=7,
        )

        # x constraint
        self.assertIsInstance(log_block.x_constraint, Constraint)
        # one-dimensional case, so there is only one x variable here
        self.assertEqual(len(log_block.x_constraint), 1)
        assertExpressionsEqual(
            self,
            log_block.x_constraint[0].expr,
            m.x
            == 1 * log_block.lambdas[0, 0]
            + 3 * log_block.lambdas[0, 1]
            + 3 * log_block.lambdas[1, 0]
            + 6 * log_block.lambdas[1, 1]
            + 6 * log_block.lambdas[2, 0]
            + 10 * log_block.lambdas[2, 1],
        )

        # simplex choice 1 constraint enables lambdas when binaries are on
        self.assertEqual(len(log_block.simplex_choice_1), log_simplex_count)
        assertExpressionsEqual(
            self,
            log_block.simplex_choice_1[0].expr,
            log_block.lambdas[2, 0] + log_block.lambdas[2, 1] <= log_block.binaries[0],
        )
        assertExpressionsEqual(
            self,
            log_block.simplex_choice_1[1].expr,
            log_block.lambdas[1, 0] + log_block.lambdas[1, 1] <= log_block.binaries[1],
        )
        # simplex choice 2 constraint enables lambdas when binaries are off
        self.assertEqual(len(log_block.simplex_choice_2), log_simplex_count)
        assertExpressionsEqual(
            self,
            log_block.simplex_choice_2[0].expr,
            log_block.lambdas[0, 0]
            + log_block.lambdas[0, 1]
            + log_block.lambdas[1, 0]
            + log_block.lambdas[1, 1]
            <= 1 - log_block.binaries[0],
        )
        assertExpressionsEqual(
            self,
            log_block.simplex_choice_2[1].expr,
            log_block.lambdas[0, 0]
            + log_block.lambdas[0, 1]
            + log_block.lambdas[2, 0]
            + log_block.lambdas[2, 1]
            <= 1 - log_block.binaries[1],
        )

    def check_pw_paraboloid(self, m):
        # This is a little larger, but at least test that the right numbers of
        # everything are created
        z = m.pw_paraboloid.get_transformation_var(m.paraboloid_expr)
        self.assertIsInstance(z, Var)
        paraboloid_block = z.parent_block()

        self.assertEqual(len(paraboloid_block.component_map(Var)), 3)
        self.assertEqual(len(paraboloid_block.component_map(Constraint)), 5)

        # Constants
        simplex_count = 4
        log_simplex_count = 2
        simplex_point_count = 3

        # Substitute var
        self.assertIsInstance(paraboloid_block.substitute_var, Var)
        # Binaries
        self.assertIsInstance(paraboloid_block.binaries, Var)
        self.assertEqual(len(paraboloid_block.binaries), log_simplex_count)
        # Lambdas
        self.assertIsInstance(paraboloid_block.lambdas, Var)
        self.assertEqual(
            len(paraboloid_block.lambdas), simplex_count * simplex_point_count
        )
        for l in paraboloid_block.lambdas.values():
            self.assertEqual(l.lb, 0)
            self.assertEqual(l.ub, 1)

        # Convex combo constraint
        self.assertIsInstance(paraboloid_block.convex_combo, Constraint)
        assertExpressionsEqual(
            self,
            paraboloid_block.convex_combo.expr,
            paraboloid_block.lambdas[0, 0]
            + paraboloid_block.lambdas[0, 1]
            + paraboloid_block.lambdas[0, 2]
            + paraboloid_block.lambdas[1, 0]
            + paraboloid_block.lambdas[1, 1]
            + paraboloid_block.lambdas[1, 2]
            + paraboloid_block.lambdas[2, 0]
            + paraboloid_block.lambdas[2, 1]
            + paraboloid_block.lambdas[2, 2]
            + paraboloid_block.lambdas[3, 0]
            + paraboloid_block.lambdas[3, 1]
            + paraboloid_block.lambdas[3, 2]
            == 1,
        )

        # Set substitute constraint
        self.assertIsInstance(paraboloid_block.set_substitute, Constraint)
        assertExpressionsEqual(
            self,
            paraboloid_block.set_substitute.expr,
            paraboloid_block.substitute_var
            == paraboloid_block.lambdas[0, 0] * m.g1(0, 1)
            + paraboloid_block.lambdas[1, 0] * m.g1(0, 1)
            + paraboloid_block.lambdas[2, 0] * m.g2(3, 4)
            + paraboloid_block.lambdas[3, 0] * m.g2(0, 7)
            + paraboloid_block.lambdas[0, 1] * m.g1(0, 4)
            + paraboloid_block.lambdas[1, 1] * m.g1(3, 4)
            + paraboloid_block.lambdas[2, 1] * m.g2(3, 7)
            + paraboloid_block.lambdas[3, 1] * m.g2(0, 4)
            + paraboloid_block.lambdas[0, 2] * m.g1(3, 4)
            + paraboloid_block.lambdas[1, 2] * m.g1(3, 1)
            + paraboloid_block.lambdas[2, 2] * m.g2(0, 7)
            + paraboloid_block.lambdas[3, 2] * m.g2(3, 4),
            places=7,
        )

        # x constraint
        self.assertIsInstance(paraboloid_block.x_constraint, Constraint)
        # Here we have two x variables
        self.assertEqual(len(paraboloid_block.x_constraint), 2)
        assertExpressionsEqual(
            self,
            paraboloid_block.x_constraint[0].expr,
            m.x1
            == 0 * paraboloid_block.lambdas[0, 0]
            + 0 * paraboloid_block.lambdas[0, 1]
            + 3 * paraboloid_block.lambdas[0, 2]
            + 0 * paraboloid_block.lambdas[1, 0]
            + 3 * paraboloid_block.lambdas[1, 1]
            + 3 * paraboloid_block.lambdas[1, 2]
            + 3 * paraboloid_block.lambdas[2, 0]
            + 3 * paraboloid_block.lambdas[2, 1]
            + 0 * paraboloid_block.lambdas[2, 2]
            + 0 * paraboloid_block.lambdas[3, 0]
            + 0 * paraboloid_block.lambdas[3, 1]
            + 3 * paraboloid_block.lambdas[3, 2],
        )
        assertExpressionsEqual(
            self,
            paraboloid_block.x_constraint[1].expr,
            m.x2
            == 1 * paraboloid_block.lambdas[0, 0]
            + 4 * paraboloid_block.lambdas[0, 1]
            + 4 * paraboloid_block.lambdas[0, 2]
            + 1 * paraboloid_block.lambdas[1, 0]
            + 4 * paraboloid_block.lambdas[1, 1]
            + 1 * paraboloid_block.lambdas[1, 2]
            + 4 * paraboloid_block.lambdas[2, 0]
            + 7 * paraboloid_block.lambdas[2, 1]
            + 7 * paraboloid_block.lambdas[2, 2]
            + 7 * paraboloid_block.lambdas[3, 0]
            + 4 * paraboloid_block.lambdas[3, 1]
            + 4 * paraboloid_block.lambdas[3, 2],
        )

        # The choices will get long, so let's just assert we have enough
        self.assertEqual(len(paraboloid_block.simplex_choice_1), log_simplex_count)
        self.assertEqual(len(paraboloid_block.simplex_choice_2), log_simplex_count)

    # Test methods using the common_tests.py code.
    def test_transformation_do_not_descend(self):
        ct.check_transformation_do_not_descend(
            self, 'contrib.piecewise.disaggregated_logarithmic'
        )

    def test_transformation_PiecewiseLinearFunction_targets(self):
        ct.check_transformation_PiecewiseLinearFunction_targets(
            self, 'contrib.piecewise.disaggregated_logarithmic'
        )

    def test_descend_into_expressions(self):
        ct.check_descend_into_expressions(
            self, 'contrib.piecewise.disaggregated_logarithmic'
        )

    def test_descend_into_expressions_constraint_target(self):
        ct.check_descend_into_expressions_constraint_target(
            self, 'contrib.piecewise.disaggregated_logarithmic'
        )

    def test_descend_into_expressions_objective_target(self):
        ct.check_descend_into_expressions_objective_target(
            self, 'contrib.piecewise.disaggregated_logarithmic'
        )

    # Check solution of the log(x) model
    @unittest.skipUnless(SolverFactory('gurobi').available(), 'Gurobi is not available')
    @unittest.skipUnless(SolverFactory('gurobi').license_is_valid(), 'No license')
    def test_solve_log_model(self):
        m = models.make_log_x_model()
        TransformationFactory("contrib.piecewise.disaggregated_logarithmic").apply_to(m)
        SolverFactory("gurobi").solve(m)
        ct.check_log_x_model_soln(self, m)
