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
import pyomo.contrib.piecewise.tests.common_tests as ct
from pyomo.contrib.piecewise.tests.models import make_log_x_model
from pyomo.contrib.piecewise.triangulations import Triangulation
from pyomo.core.base import TransformationFactory
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
    Constraint,
    SolverFactory,
    Var,
    ConcreteModel,
    Objective,
    log,
    value,
    minimize,
)
from pyomo.contrib.piecewise import PiecewiseLinearFunction
import itertools


class TestTransformPiecewiseModelToIncrementalMIP(unittest.TestCase):

    def check_pw_log(self, m):
        z = m.pw_log.get_transformation_var(m.log_expr)
        self.assertIsInstance(z, Var)
        log_block = z.parent_block()

        # Vars: three deltas, two y binaries, one substitute var
        self.assertEqual(len(log_block.component_map(Var)), 3)
        self.assertIsInstance(log_block.delta, Var)
        self.assertEqual(len(log_block.delta), 3)
        self.assertIsInstance(log_block.y_binaries, Var)
        self.assertEqual(len(log_block.y_binaries), 2)
        self.assertIsInstance(log_block.substitute_var, Var)
        self.assertEqual(len(log_block.substitute_var), 1)

        # Constraints: 2 delta below y, 2 y below delta, one each of the three others
        self.assertEqual(len(log_block.component_map(Constraint)), 5)
        self.assertIsInstance(log_block.deltas_below_y, Constraint)
        self.assertEqual(len(log_block.deltas_below_y), 2)
        self.assertIsInstance(log_block.y_below_delta, Constraint)
        self.assertEqual(len(log_block.y_below_delta), 2)
        self.assertIsInstance(log_block.delta_one_constraint, Constraint)
        self.assertEqual(len(log_block.delta_one_constraint), 1)
        self.assertIsInstance(log_block.x_constraint, Constraint)
        self.assertEqual(len(log_block.x_constraint), 1)
        self.assertIsInstance(log_block.set_substitute, Constraint)
        self.assertEqual(len(log_block.set_substitute), 1)

        assertExpressionsEqual(
            self,
            log_block.x_constraint[0].expr,
            m.x
            == 1
            + (
                log_block.delta[0, 1] * (3 - 1)
                + log_block.delta[1, 1] * (6 - 3)
                + log_block.delta[2, 1] * (10 - 6)
            ),
        )
        assertExpressionsEqual(
            self,
            log_block.set_substitute.expr,
            log_block.substitute_var
            == m.f1(1)
            + (
                log_block.delta[0, 1] * (m.f2(3) - m.f1(1))
                + log_block.delta[1, 1] * (m.f3(6) - m.f2(3))
                + log_block.delta[2, 1] * (m.f3(10) - m.f3(6))
            ),
            places=10,
        )
        assertExpressionsEqual(
            self, log_block.delta_one_constraint.expr, log_block.delta[0, 1] <= 1
        )
        assertExpressionsEqual(
            self,
            log_block.deltas_below_y[0].expr,
            log_block.delta[1, 1] <= log_block.y_binaries[0],
        )
        assertExpressionsEqual(
            self,
            log_block.deltas_below_y[1].expr,
            log_block.delta[2, 1] <= log_block.y_binaries[1],
        )
        assertExpressionsEqual(
            self,
            log_block.y_below_delta[0].expr,
            log_block.y_binaries[0] <= log_block.delta[0, 1],
        )
        assertExpressionsEqual(
            self,
            log_block.y_below_delta[1].expr,
            log_block.y_binaries[1] <= log_block.delta[1, 1],
        )

    def check_pw_paraboloid(self, m):
        z = m.pw_paraboloid.get_transformation_var(m.paraboloid_expr)
        self.assertIsInstance(z, Var)
        paraboloid_block = z.parent_block()

        # Vars: 8 deltas (2 per simplex), 3 y binaries, one substitute var
        self.assertEqual(len(paraboloid_block.component_map(Var)), 3)
        self.assertIsInstance(paraboloid_block.delta, Var)
        self.assertEqual(len(paraboloid_block.delta), 8)
        self.assertIsInstance(paraboloid_block.y_binaries, Var)
        self.assertEqual(len(paraboloid_block.y_binaries), 3)
        self.assertIsInstance(paraboloid_block.substitute_var, Var)
        self.assertEqual(len(paraboloid_block.substitute_var), 1)

        # Constraints: 3 delta below y, 3 y below delta, two x constraints (two
        # coordinates), one each of the three others
        self.assertEqual(len(paraboloid_block.component_map(Constraint)), 5)
        self.assertIsInstance(paraboloid_block.deltas_below_y, Constraint)
        self.assertEqual(len(paraboloid_block.deltas_below_y), 3)
        self.assertIsInstance(paraboloid_block.y_below_delta, Constraint)
        self.assertEqual(len(paraboloid_block.y_below_delta), 3)
        self.assertIsInstance(paraboloid_block.delta_one_constraint, Constraint)
        self.assertEqual(len(paraboloid_block.delta_one_constraint), 1)
        self.assertIsInstance(paraboloid_block.x_constraint, Constraint)
        self.assertEqual(len(paraboloid_block.x_constraint), 2)
        self.assertIsInstance(paraboloid_block.set_substitute, Constraint)
        self.assertEqual(len(paraboloid_block.set_substitute), 1)

    ordered_simplices = [
        [(0, 1), (3, 1), (3, 4)],
        [(3, 4), (0, 1), (0, 4)],
        [(0, 4), (0, 7), (3, 4)],
        [(3, 4), (3, 7), (0, 7)],
    ]

    # Test methods using the common_tests.py code.
    def test_transformation_do_not_descend(self):
        ct.check_transformation_do_not_descend(
            self,
            'contrib.piecewise.incremental',
            make_log_x_model(simplices=self.ordered_simplices),
        )

    def test_transformation_PiecewiseLinearFunction_targets(self):
        ct.check_transformation_PiecewiseLinearFunction_targets(
            self,
            'contrib.piecewise.incremental',
            make_log_x_model(simplices=self.ordered_simplices),
        )

    def test_descend_into_expressions(self):
        ct.check_descend_into_expressions(
            self,
            'contrib.piecewise.incremental',
            make_log_x_model(simplices=self.ordered_simplices),
        )

    def test_descend_into_expressions_constraint_target(self):
        ct.check_descend_into_expressions_constraint_target(
            self,
            'contrib.piecewise.incremental',
            make_log_x_model(simplices=self.ordered_simplices),
        )

    def test_descend_into_expressions_objective_target(self):
        ct.check_descend_into_expressions_objective_target(
            self,
            'contrib.piecewise.incremental',
            make_log_x_model(simplices=self.ordered_simplices),
        )

    @unittest.skipUnless(SolverFactory('gurobi').available(), 'Gurobi is not available')
    @unittest.skipUnless(SolverFactory('gurobi').license_is_valid(), 'No license')
    def test_solve_log_model(self):
        m = make_log_x_model(simplices=self.ordered_simplices)
        TransformationFactory('contrib.piecewise.incremental').apply_to(m)
        TransformationFactory('gdp.bigm').apply_to(m)
        SolverFactory('gurobi').solve(m)
        ct.check_log_x_model_soln(self, m)

    # Failed during development when ordered j1 vertex ordering got broken
    @unittest.skipUnless(SolverFactory('gurobi').available(), 'Gurobi is not available')
    @unittest.skipUnless(SolverFactory('gurobi').license_is_valid(), 'No license')
    def test_solve_product_model(self):
        m = ConcreteModel()
        m.x1 = Var(bounds=(0.5, 5))
        m.x2 = Var(bounds=(0.9, 0.95))
        pts = list(itertools.product([0.5, 2.75, 5], [0.9, 0.925, 0.95]))
        m.pwlf = PiecewiseLinearFunction(
            points=pts,
            function=lambda x, y: x * y,
            triangulation=Triangulation.OrderedJ1,
        )
        m.obj = Objective(sense=minimize, expr=m.pwlf(m.x1, m.x2))
        TransformationFactory("contrib.piecewise.incremental").apply_to(m)
        SolverFactory('gurobi').solve(m)
        self.assertAlmostEqual(0.45, value(m.obj))
