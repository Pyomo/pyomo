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


class TestTransformPiecewiseModelToInnerRepnGDP(unittest.TestCase):
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
            paraboloid_block.disjuncts[0]: ([(0, 1), (0, 4), (3, 4)], m.g1),
            paraboloid_block.disjuncts[1]: ([(0, 1), (3, 4), (3, 1)], m.g1),
            paraboloid_block.disjuncts[2]: ([(3, 4), (3, 7), (0, 7)], m.g2),
            paraboloid_block.disjuncts[3]: ([(0, 7), (0, 4), (3, 4)], m.g2),
        }
        for d, (pts, f) in disjuncts_dict.items():
            self.check_paraboloid_disjunct(
                d, pts, f, paraboloid_block.substitute_var, m.x1, m.x2
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

    def test_transformation_do_not_descend(self):
        ct.check_transformation_do_not_descend(self, 'contrib.piecewise.inner_repn_gdp')

    def test_transformation_PiecewiseLinearFunction_targets(self):
        ct.check_transformation_PiecewiseLinearFunction_targets(
            self, 'contrib.piecewise.inner_repn_gdp'
        )

    def test_descend_into_expressions(self):
        ct.check_descend_into_expressions(self, 'contrib.piecewise.inner_repn_gdp')

    def test_descend_into_expressions_constraint_target(self):
        ct.check_descend_into_expressions_constraint_target(
            self, 'contrib.piecewise.inner_repn_gdp'
        )

    def test_descend_into_expressions_objective_target(self):
        ct.check_descend_into_expressions_objective_target(
            self, 'contrib.piecewise.inner_repn_gdp'
        )

    @unittest.skipUnless(SolverFactory('gurobi').available(), 'Gurobi is not available')
    @unittest.skipUnless(SolverFactory('gurobi').license_is_valid(), 'No license')
    def test_solve_disaggregated_convex_combo_model(self):
        m = models.make_log_x_model()
        TransformationFactory(
            'contrib.piecewise.disaggregated_convex_combination'
        ).apply_to(m)
        SolverFactory('gurobi').solve(m)

        ct.check_log_x_model_soln(self, m)
