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

from math import sqrt
from pyomo.common.dependencies import scipy_available
import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise.tests import models
import pyomo.contrib.piecewise.tests.common_tests as ct
from pyomo.core.base import TransformationFactory
from pyomo.core.expr.compare import (
    assertExpressionsEqual,
    assertExpressionsStructurallyEqual,
)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.environ import Constraint, SolverFactory, Var


class TestTransformPiecewiseModelToOuterRepnGDP(unittest.TestCase):
    def check_log_disjunct(self, d, pts, f, substitute_var, x):
        # We can fit both bounds constraints in one constraint, then we have the
        # linear function
        self.assertEqual(len(d.component_map(Constraint)), 2)
        # indicator_var
        self.assertEqual(len(d.component_map(Var)), 1)
        self.assertIsInstance(d.simplex_halfspaces, Constraint)
        self.assertEqual(d.simplex_halfspaces.lower, pts[0])
        self.assertEqual(d.simplex_halfspaces.upper, pts[1])
        self.assertIs(d.simplex_halfspaces.body, x)

        self.assertIsInstance(d.set_substitute, Constraint)
        assertExpressionsEqual(
            self, d.set_substitute.expr, substitute_var == f(x), places=7
        )

    def check_paraboloid_disjunct(self, d, constraint_coefs, f, substitute_var, x1, x2):
        self.assertEqual(len(d.component_map(Constraint)), 2)
        # just indicator_var
        self.assertEqual(len(d.component_map(Var)), 1)
        for i, cons in d.simplex_halfspaces.items():
            coefs = constraint_coefs[i]
            assertExpressionsEqual(
                self, cons.expr, coefs[0] * x1 + coefs[1] * x2 + coefs[2] <= 0, places=6
            )

        self.assertIsInstance(d.set_substitute, Constraint)
        assertExpressionsEqual(
            self, d.set_substitute.expr, substitute_var == f(x1, x2), places=7
        )

    def check_pw_log(self, m):
        ##
        # Check the transformation of the approximation of log(x)
        ##
        z = m.pw_log.get_transformation_var(m.log_expr)
        self.assertIsInstance(z, Var)
        # Now we can use those Vars to check on what the transformation created
        log_block = z.parent_block()
        ct.check_trans_block_structure(self, log_block)

        # Check that all of the Disjuncts have what they should
        self.assertEqual(len(log_block.disjuncts), 3)
        disjuncts_dict = {
            log_block.disjuncts[0]: ((1, 3), m.f1),
            log_block.disjuncts[1]: ((3, 6), m.f2),
            log_block.disjuncts[2]: ((6, 10), m.f3),
        }
        for d, (pts, f) in disjuncts_dict.items():
            self.check_log_disjunct(d, pts, f, log_block.substitute_var, m.x)

        # Check the Disjunction
        self.assertIsInstance(log_block.pick_a_piece, Disjunction)
        self.assertEqual(len(log_block.pick_a_piece.disjuncts), 3)
        for i in range(2):
            self.assertIs(log_block.pick_a_piece.disjuncts[i], log_block.disjuncts[i])

        # And check the substitute Var is in the objective now.
        self.assertIs(m.obj.expr.expr, log_block.substitute_var)

    def check_pw_paraboloid(self, m):
        ##
        # Check the approximation of the transformation of the paraboloid
        ##
        z = m.pw_paraboloid.get_transformation_var(m.paraboloid_expr)
        self.assertIsInstance(z, Var)
        paraboloid_block = z.parent_block()
        ct.check_trans_block_structure(self, paraboloid_block)

        self.assertEqual(len(paraboloid_block.disjuncts), 4)
        disjuncts_dict = {
            # the normal vectors of the faces are normalized when we get
            # them from scipy:
            paraboloid_block.disjuncts[0]: (
                [
                    [sqrt(2) / 2, -sqrt(2) / 2, sqrt(2) / 2],
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, -4.0],
                ],
                m.g1,
            ),
            paraboloid_block.disjuncts[1]: (
                [
                    [-sqrt(2) / 2, sqrt(2) / 2, -sqrt(2) / 2],
                    [0.0, -1.0, 1.0],
                    [1.0, 0.0, -3.0],
                ],
                m.g1,
            ),
            paraboloid_block.disjuncts[2]: (
                [
                    [-sqrt(2) / 2, -sqrt(2) / 2, 7 * sqrt(2) / 2],
                    [0.0, 1.0, -7.0],
                    [1.0, 0.0, -3.0],
                ],
                m.g2,
            ),
            paraboloid_block.disjuncts[3]: (
                [
                    [sqrt(2) / 2, sqrt(2) / 2, -7 * sqrt(2) / 2],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 4.0],
                ],
                m.g2,
            ),
        }
        for d, (constraint_coefs, f) in disjuncts_dict.items():
            self.check_paraboloid_disjunct(
                d, constraint_coefs, f, paraboloid_block.substitute_var, m.x1, m.x2
            )

        # Check the Disjunction
        self.assertIsInstance(paraboloid_block.pick_a_piece, Disjunction)
        self.assertEqual(len(paraboloid_block.pick_a_piece.disjuncts), 4)
        for i in range(3):
            self.assertIs(
                paraboloid_block.pick_a_piece.disjuncts[i],
                paraboloid_block.disjuncts[i],
            )

        # And check the substitute Var is in the objective now.
        self.assertIs(m.indexed_c[0].body.args[0].expr, paraboloid_block.substitute_var)

    @unittest.skipUnless(scipy_available, "Scipy is not available")
    def test_transformation_do_not_descend(self):
        ct.check_transformation_do_not_descend(self, 'contrib.piecewise.outer_repn_gdp')

    def test_transformation_PiecewiseLinearFunction_targets(self):
        ct.check_transformation_PiecewiseLinearFunction_targets(
            self, 'contrib.piecewise.outer_repn_gdp'
        )

    @unittest.skipUnless(scipy_available, "Scipy is not available")
    def test_descend_into_expressions(self):
        ct.check_descend_into_expressions(self, 'contrib.piecewise.outer_repn_gdp')

    @unittest.skipUnless(scipy_available, "Scipy is not available")
    def test_descend_into_expressions_constraint_target(self):
        ct.check_descend_into_expressions_constraint_target(
            self, 'contrib.piecewise.outer_repn_gdp'
        )

    def test_descend_into_expressions_objective_target(self):
        ct.check_descend_into_expressions_objective_target(
            self, 'contrib.piecewise.outer_repn_gdp'
        )

    @unittest.skipUnless(scipy_available, "scipy is not available")
    @unittest.skipUnless(SolverFactory('gurobi').available(), 'Gurobi is not available')
    @unittest.skipUnless(SolverFactory('gurobi').license_is_valid(), 'No license')
    def test_solve_multiple_choice_model(self):
        m = models.make_log_x_model()
        TransformationFactory('contrib.piecewise.multiple_choice').apply_to(m)
        SolverFactory('gurobi').solve(m)

        ct.check_log_x_model_soln(self, m)
