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
from pyomo.contrib.piecewise import (
    PiecewiseLinearFunction, PiecewiseLinearExpression)
from pyomo.core.base import TransformationFactory
from pyomo.core.expr.compare import (
    assertExpressionsEqual, assertExpressionsStructurallyEqual)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.environ import (
    ConcreteModel, Constraint, log, SolverFactory, Objective, value, Var)

class TestTransformPiecewiseModelToInnerRepnGDP(unittest.TestCase):
    def make_model(self):
        m = ConcreteModel()
        m.x = Var(bounds=(1, 10))
        def log_function(x):
            return log(x)
        m.log_function = log_function
        m.pw_log = PiecewiseLinearFunction(points=[1, 3, 6, 10],
                                           function=m.log_function)

        # Here are the linear functions, for safe keeping.
        def f1(x):
            return (log(3)/2)*x - log(3)/2
        m.f1 = f1
        def f2(x):
            return (log(2)/3)*x + log(3/2)
        m.f2 = f2
        def f3(x):
            return (log(5/3)/4)*x + log(6/((5/3)**(3/2)))
        m.f3 = f3

        m.obj = Objective(expr=m.pw_log(m.x))

        m.x1 = Var(bounds=(0, 3))
        m.x2 = Var(bounds=(1, 7))
        # Here's a cute paraboloid:
        def f(x, y):
            return x**2 + y**2
        m.parabaloid = f

        def g1(x1, x2):
            return 3*x1 + 5*x2 - 4
        m.g1 = g1
        def g2(x1, x2):
            return 3*x1 + 11*x2 - 28
        m.g2 = g2
        simplices = [[(0, 1), (0, 4), (3, 4)],
                     [(0, 1), (3, 4), (3, 1)],
                     [(3, 4), (3, 7), (0, 7)],
                     [(0, 7), (0, 4), (3, 4)]]
        m.pw_paraboloid = PiecewiseLinearFunction(simplices=simplices,
                                                  linear_functions=[g1, g1, g2,
                                                                    g2])
        def c_rule(m, i):
            if i == 0:
                return m.x >= m.pw_paraboloid(m.x1, m.x2)
            else:
                return (1, m.x1, 2)
        m.indexed_c = Constraint([0, 1], rule=c_rule)

        return m

    def check_trans_block_structure(self, block):
        # One (indexed) disjunct
        self.assertEqual(len(block.component_map(Disjunct)), 1)
        # One disjunction
        self.assertEqual(len(block.component_map(Disjunction)), 1)
        # The 'z' var (that we will substitute in for the function being
        # approximated) is here:
        self.assertEqual(len(block.component_map(Var)), 1)
        self.assertIsInstance(block.substitute_var, Var)

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
        assertExpressionsEqual(self, d.convex_combo.expr,
                               d.lambdas[0] + d.lambdas[1] == 1)
        self.assertIsInstance(d.set_substitute, Constraint)
        assertExpressionsEqual(self, d.set_substitute.expr,
                               substitute_var == f(x), places=7)
        self.assertIsInstance(d.linear_combo, Constraint)
        self.assertEqual(len(d.linear_combo), 1)
        assertExpressionsEqual(
            self, d.linear_combo[0].expr,
            x == pts[0]*d.lambdas[0] + pts[1]*d.lambdas[1])

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
        assertExpressionsEqual(self, d.convex_combo.expr,
                               d.lambdas[0] + d.lambdas[1] + d.lambdas[2] == 1)
        self.assertIsInstance(d.set_substitute, Constraint)
        assertExpressionsEqual(self, d.set_substitute.expr,
                               substitute_var == f(x1, x2), places=7)
        self.assertIsInstance(d.linear_combo, Constraint)
        self.assertEqual(len(d.linear_combo), 2)
        assertExpressionsEqual(
            self, d.linear_combo[0].expr,
            x1 == pts[0][0]*d.lambdas[0] + pts[1][0]*d.lambdas[1] +
            pts[2][0]*d.lambdas[2])
        assertExpressionsEqual(
            self, d.linear_combo[1].expr,
            x2 == pts[0][1]*d.lambdas[0] + pts[1][1]*d.lambdas[1] +
            pts[2][1]*d.lambdas[2])

    def check_pw_log(self, m):
        ##
        # Check the transformation of the approximation of log(x)
        ##
        self.assertIsInstance(m.pw_log._expressions[0].expr, Var)
        # Now we can use those Vars to check on what the transformation created
        z = m.pw_log._expressions[0].expr
        log_block = z.parent_block()
        self.check_trans_block_structure(log_block)

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
            self.assertIs(log_block.pick_a_piece.disjuncts[i],
                          log_block.disjuncts[i])

        # And check the substitute Var is in the objective now.
        self.assertIs(m.obj.expr.expr, log_block.substitute_var)

    def check_pw_paraboloid(self, m):
        ##
        # Check the approximation of the transformation of the paraboloid
        ##
        self.assertIsInstance(m.pw_paraboloid._expressions[0].expr, Var)
        z = m.pw_paraboloid._expressions[0].expr
        paraboloid_block = z.parent_block()
        self.check_trans_block_structure(paraboloid_block)

        self.assertEqual(len(paraboloid_block.disjuncts), 4)
        disjuncts_dict = {
            paraboloid_block.disjuncts[0]: ([(0, 1), (0, 4), (3, 4)], m.g1),
            paraboloid_block.disjuncts[1]: ([(0, 1), (3, 4), (3, 1)], m.g1),
            paraboloid_block.disjuncts[2]: ([(3, 4), (3, 7), (0, 7)], m.g2),
            paraboloid_block.disjuncts[3]: ([(0, 7), (0, 4), (3, 4)], m.g2),
        }
        for d, (pts, f) in disjuncts_dict.items():
            self.check_paraboloid_disjunct(d, pts, f,
                                           paraboloid_block.substitute_var,
                                           m.x1, m.x2)

        # Check the Disjunction
        self.assertIsInstance(paraboloid_block.pick_a_piece, Disjunction)
        self.assertEqual(len(paraboloid_block.pick_a_piece.disjuncts), 4)
        for i in range(3):
            self.assertIs(paraboloid_block.pick_a_piece.disjuncts[i],
                          paraboloid_block.disjuncts[i])

        # And check the substitute Var is in the objective now.
        self.assertIs(m.indexed_c[0].body.args[0].expr,
                      paraboloid_block.substitute_var)

    def test_transformation_do_not_descend(self):
        m = self.make_model()
        inner_repn = TransformationFactory('contrib.inner_repn_gdp')
        inner_repn.apply_to(m)

        self.check_pw_log(m)
        self.check_pw_paraboloid(m)

    def test_transformation_PiecewiseLinearFunction_targets(self):
        m = self.make_model()
        inner_repn = TransformationFactory('contrib.inner_repn_gdp')
        inner_repn.apply_to(m, targets=[m.pw_log])

        self.check_pw_log(m)

        # And check that the paraboloid was *not* transformed.
        self.assertIsInstance(m.pw_paraboloid._expressions[0].expr,
                              PiecewiseLinearExpression)

    def test_descend_into_expressions(self):
        m = self.make_model()
        inner_repn = TransformationFactory('contrib.inner_repn_gdp')
        inner_repn.apply_to(m, descend_into_expressions=True)

        # Everything should be transformed
        self.check_pw_log(m)
        self.check_pw_paraboloid(m)

    def test_descend_into_expressions_constraint_target(self):
        m = self.make_model()
        inner_repn = TransformationFactory('contrib.inner_repn_gdp')
        inner_repn.apply_to(m, descend_into_expressions=True,
                            targets=[m.indexed_c])

        self.check_pw_paraboloid(m)
        # And check that the log was *not* transformed.
        self.assertIsInstance(m.pw_log._expressions[0].expr,
                              PiecewiseLinearExpression)

    def test_descend_into_expressions_objective_target(self):
        m = self.make_model()
        inner_repn = TransformationFactory('contrib.inner_repn_gdp')
        inner_repn.apply_to(m, descend_into_expressions=True,
                            targets=[m.obj])

        self.check_pw_log(m)
        # And check that the paraboloid was *not* transformed.
        self.assertIsInstance(m.pw_paraboloid._expressions[0].expr,
                              PiecewiseLinearExpression)

    @unittest.skipUnless(SolverFactory('gurobi').available(),
                         'Gurobi is not available')
    def test_solve_disaggregated_convex_combo_model(self):
        m = self.make_model()
        TransformationFactory(
            'contrib.disaggregated_convex_combination').apply_to(m)

        SolverFactory('gurobi').solve(m, tee=True)

        self.assertAlmostEqual(value(m.x), 4)
        self.assertAlmostEqual(value(m.x1), 1)
        self.assertAlmostEqual(value(m.x2), 1)
        self.assertAlmostEqual(value(m.obj), m.f2(4))
