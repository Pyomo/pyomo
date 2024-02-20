#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
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
from pyomo.gdp import Disjunction, Disjunct
from pyomo.core.expr.compare import assertExpressionsEqual


# Test the nested inner repn gdp model using the common_tests code
class TestTransformPiecewiseModelToNestedInnerRepnGDP(unittest.TestCase):
    # Check one disjunct for proper contents. Disjunct structure should be
    # identical to the version for the inner representation gdp
    def check_log_disjunct(self, d, pts, f, substitute_var, x):
        self.assertEqual(len(d.component_map(Constraint)), 3)
        # lambdas and indicator_var
        self.assertEqual(len(d.component_map(Var)), 2)
        self.assertIsInstance(d.lambdas, Var)
        self.assertEqual(len(d.lambdas), 2)
        for lamb in d.lambdas.values():
            self.assertEqual(lamb.lb, 0)
            self.assertEqual(lamb.ub, 1)
        self.assertIsInstance(d.convex_combo, Constraint)
        assertExpressionsEqual(
            self, d.convex_combo.expr, d.lambdas[0] + d.lambdas[1] == 1
        )
        self.assertIsInstance(d.set_substitute, Constraint)
        assertExpressionsEqual(
            self, d.set_substitute.expr, substitute_var == f(x), places=7
        )
        self.assertIsInstance(d.linear_combo, Constraint)
        self.assertEqual(len(d.linear_combo), 1)
        assertExpressionsEqual(
            self,
            d.linear_combo[0].expr,
            x == pts[0] * d.lambdas[0] + pts[1] * d.lambdas[1],
        )

    # Check one disjunct from the paraboloid block for proper contents. This should
    # be identical to the inner_representation_gdp one
    def check_paraboloid_disjunct(self, d, pts, f, substitute_var, x1, x2):
        self.assertEqual(len(d.component_map(Constraint)), 3)
        # lambdas and indicator_var
        self.assertEqual(len(d.component_map(Var)), 2)
        self.assertIsInstance(d.lambdas, Var)
        self.assertEqual(len(d.lambdas), 3)
        for lamb in d.lambdas.values():
            self.assertEqual(lamb.lb, 0)
            self.assertEqual(lamb.ub, 1)
        self.assertIsInstance(d.convex_combo, Constraint)
        assertExpressionsEqual(
            self, d.convex_combo.expr, d.lambdas[0] + d.lambdas[1] + d.lambdas[2] == 1
        )
        self.assertIsInstance(d.set_substitute, Constraint)
        assertExpressionsEqual(
            self, d.set_substitute.expr, substitute_var == f(x1, x2), places=7
        )
        self.assertIsInstance(d.linear_combo, Constraint)
        self.assertEqual(len(d.linear_combo), 2)
        assertExpressionsEqual(
            self,
            d.linear_combo[0].expr,
            x1
            == pts[0][0] * d.lambdas[0]
            + pts[1][0] * d.lambdas[1]
            + pts[2][0] * d.lambdas[2],
        )
        assertExpressionsEqual(
            self,
            d.linear_combo[1].expr,
            x2
            == pts[0][1] * d.lambdas[0]
            + pts[1][1] * d.lambdas[1]
            + pts[2][1] * d.lambdas[2],
        )

    # Check the structure of the log PWLF Block
    def check_pw_log(self, m):
        z = m.pw_log.get_transformation_var(m.log_expr)
        self.assertIsInstance(z, Var)
        # Now we can use those Vars to check on what the transformation created
        log_block = z.parent_block()

        # Not using ct.check_trans_block_structure() because these are slightly
        # different
        # Two top-level disjuncts
        self.assertEqual(len(log_block.component_map(Disjunct)), 2)
        # One disjunction
        self.assertEqual(len(log_block.component_map(Disjunction)), 1)
        # The 'z' var (that we will substitute in for the function being
        # approximated) is here:
        self.assertEqual(len(log_block.component_map(Var)), 1)
        self.assertIsInstance(log_block.substitute_var, Var)

        # Check the tree structure, which should be heavier on the right
        # Parent disjunction
        self.assertIsInstance(log_block.disj, Disjunction)
        self.assertEqual(len(log_block.disj.disjuncts), 2)

        # Left disjunct with constraints
        self.assertIsInstance(log_block.d_l, Disjunct)
        self.check_log_disjunct(
            log_block.d_l, (1, 3), m.f1, log_block.substitute_var, m.x
        )

        # Right disjunct with disjunction
        self.assertIsInstance(log_block.d_r, Disjunct)
        self.assertIsInstance(log_block.d_r.inner_disjunction_r, Disjunction)
        self.assertEqual(len(log_block.d_r.inner_disjunction_r.disjuncts), 2)

        # Left and right child disjuncts with constraints
        self.assertIsInstance(log_block.d_r.d_l, Disjunct)
        self.check_log_disjunct(
            log_block.d_r.d_l, (3, 6), m.f2, log_block.substitute_var, m.x
        )
        self.assertIsInstance(log_block.d_r.d_r, Disjunct)
        self.check_log_disjunct(
            log_block.d_r.d_r, (6, 10), m.f3, log_block.substitute_var, m.x
        )

        # Check that this also became the objective
        self.assertIs(m.obj.expr.expr, log_block.substitute_var)

    # Check the structure of the paraboloid PWLF block
    def check_pw_paraboloid(self, m):
        z = m.pw_paraboloid.get_transformation_var(m.paraboloid_expr)
        self.assertIsInstance(z, Var)
        paraboloid_block = z.parent_block()

        # Two top-level disjuncts
        self.assertEqual(len(paraboloid_block.component_map(Disjunct)), 2)
        # One disjunction
        self.assertEqual(len(paraboloid_block.component_map(Disjunction)), 1)
        # The 'z' var (that we will substitute in for the function being
        # approximated) is here:
        self.assertEqual(len(paraboloid_block.component_map(Var)), 1)
        self.assertIsInstance(paraboloid_block.substitute_var, Var)

        # This one should have an even tree with four leaf disjuncts
        disjuncts_dict = {
            paraboloid_block.d_l.d_l: ([(0, 1), (0, 4), (3, 4)], m.g1),
            paraboloid_block.d_l.d_r: ([(0, 1), (3, 4), (3, 1)], m.g1),
            paraboloid_block.d_r.d_l: ([(3, 4), (3, 7), (0, 7)], m.g2),
            paraboloid_block.d_r.d_r: ([(0, 7), (0, 4), (3, 4)], m.g2),
        }
        for d, (pts, f) in disjuncts_dict.items():
            self.check_paraboloid_disjunct(
                d, pts, f, paraboloid_block.substitute_var, m.x1, m.x2
            )

        # And check the substitute Var is in the objective now.
        self.assertIs(m.indexed_c[0].body.args[0].expr, paraboloid_block.substitute_var)

    # Test methods using the common_tests.py code. Copied in from test_inner_repn_gdp.py.
    def test_transformation_do_not_descend(self):
        ct.check_transformation_do_not_descend(
            self, 'contrib.piecewise.nested_inner_repn_gdp'
        )

    def test_transformation_PiecewiseLinearFunction_targets(self):
        ct.check_transformation_PiecewiseLinearFunction_targets(
            self, 'contrib.piecewise.nested_inner_repn_gdp'
        )

    def test_descend_into_expressions(self):
        ct.check_descend_into_expressions(
            self, 'contrib.piecewise.nested_inner_repn_gdp'
        )

    def test_descend_into_expressions_constraint_target(self):
        ct.check_descend_into_expressions_constraint_target(
            self, 'contrib.piecewise.nested_inner_repn_gdp'
        )

    def test_descend_into_expressions_objective_target(self):
        ct.check_descend_into_expressions_objective_target(
            self, 'contrib.piecewise.nested_inner_repn_gdp'
        )

    # Check the solution of the log(x) model
    @unittest.skipUnless(SolverFactory('gurobi').available(), 'Gurobi is not available')
    @unittest.skipUnless(SolverFactory('gurobi').license_is_valid(), 'No license')
    def test_solve_log_model(self):
        m = models.make_log_x_model()
        TransformationFactory("contrib.piecewise.nested_inner_repn_gdp").apply_to(m)
        TransformationFactory("gdp.bigm").apply_to(m)
        SolverFactory("gurobi").solve(m)
        ct.check_log_x_model_soln(self, m)
