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
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.contrib.piecewise.transform.nonlinear_to_pwl import (
    NonlinearToPWL,
    DomainPartitioningMethod,
)
from pyomo.core.expr.compare import (
    assertExpressionsEqual,
    assertExpressionsStructurallyEqual,
)
from pyomo.core.expr.numeric_expr import SumExpression
from pyomo.environ import (
    ConcreteModel,
    Var,
    Constraint,
    TransformationFactory,
    log,
    Objective,
    Reals,
    SolverFactory,
)

## debug
from pytest import set_trace


class TestNonlinearToPWL_1D(unittest.TestCase):
    def make_model(self):
        m = ConcreteModel()
        m.x = Var(bounds=(1, 10))
        m.cons = Constraint(expr=log(m.x) >= 0.35)

        return m

    def check_pw_linear_log_x(self, m, pwlf, x1, x2, x3):
        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')

        points = [(x1,), (x2,), (x3,)]
        self.assertEqual(pwlf._simplices, [(0, 1), (1, 2)])
        self.assertEqual(pwlf._points, points)
        self.assertEqual(len(pwlf._linear_functions), 2)

        assertExpressionsStructurallyEqual(
            self,
            pwlf._linear_functions[0](m.x),
            ((log(x2) - log(x1)) / (x2 - x1)) * m.x
            + (log(x2) - ((log(x2) - log(x1)) / (x2 - x1)) * x2),
            places=7,
        )
        assertExpressionsStructurallyEqual(
            self,
            pwlf._linear_functions[1](m.x),
            ((log(x3) - log(x2)) / (x3 - x2)) * m.x
            + (log(x3) - ((log(x3) - log(x2)) / (x3 - x2)) * x3),
            places=7,
        )

        self.assertEqual(len(pwlf._expressions), 1)
        new_cons = n_to_pwl.get_transformed_component(m.cons)
        self.assertTrue(new_cons.active)
        self.assertIs(new_cons.body, pwlf._expressions[id(new_cons.body.expr)])
        self.assertIsNone(new_cons.ub)
        self.assertEqual(new_cons.lb, 0.35)
        self.assertIs(n_to_pwl.get_src_component(new_cons), m.cons)

        quadratic = n_to_pwl.get_transformed_quadratic_constraints(m)
        self.assertEqual(len(quadratic), 0)
        nonlinear = n_to_pwl.get_transformed_nonlinear_constraints(m)
        self.assertEqual(len(nonlinear), 1)
        self.assertIn(m.cons, nonlinear)

    def test_log_constraint_uniform_grid(self):
        m = self.make_model()

        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        n_to_pwl.apply_to(
            m,
            num_points=3,
            domain_partitioning_method=DomainPartitioningMethod.UNIFORM_GRID,
        )

        # cons is transformed
        self.assertFalse(m.cons.active)

        pwlf = list(
            m.component_data_objects(PiecewiseLinearFunction, descend_into=True)
        )
        self.assertEqual(len(pwlf), 1)
        pwlf = pwlf[0]

        points = [(1.0009,), (5.5,), (9.9991,)]
        (x1, x2, x3) = 1.0009, 5.5, 9.9991
        self.check_pw_linear_log_x(m, pwlf, x1, x2, x3)

    def test_log_constraint_random_grid(self):
        m = self.make_model()

        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        # [ESJ 3/30/24]: The seed is actually set in the function for getting
        # the points right now, so this will be deterministic.
        n_to_pwl.apply_to(
            m,
            num_points=3,
            domain_partitioning_method=DomainPartitioningMethod.RANDOM_GRID,
        )

        # cons is transformed
        self.assertFalse(m.cons.active)

        pwlf = list(
            m.component_data_objects(PiecewiseLinearFunction, descend_into=True)
        )
        self.assertEqual(len(pwlf), 1)
        pwlf = pwlf[0]

        x1 = 4.370861069626263
        x2 = 7.587945476302646
        x3 = 9.556428757689245
        self.check_pw_linear_log_x(m, pwlf, x1, x2, x3)

    def test_do_not_transform_quadratic_constraint(self):
        m = self.make_model()
        m.quad = Constraint(expr=m.x**2 <= 9)
        m.lin = Constraint(expr=m.x >= 2)

        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        n_to_pwl.apply_to(
            m,
            num_points=3,
            domain_partitioning_method=DomainPartitioningMethod.UNIFORM_GRID,
            approximate_quadratic_constraints=False,
        )

        # cons is transformed
        self.assertFalse(m.cons.active)

        pwlf = list(
            m.component_data_objects(PiecewiseLinearFunction, descend_into=True)
        )
        self.assertEqual(len(pwlf), 1)
        pwlf = pwlf[0]

        points = [(1.0009,), (5.5,), (9.9991,)]
        (x1, x2, x3) = 1.0009, 5.5, 9.9991
        self.check_pw_linear_log_x(m, pwlf, x1, x2, x3)

        # quad is not
        self.assertTrue(m.quad.active)
        # neither is the linear one
        self.assertTrue(m.lin.active)

    def test_constraint_target(self):
        m = self.make_model()
        m.quad = Constraint(expr=m.x**2 <= 9)

        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        n_to_pwl.apply_to(
            m,
            num_points=3,
            domain_partitioning_method=DomainPartitioningMethod.UNIFORM_GRID,
            targets=[m.cons],
        )

        # cons is transformed
        self.assertFalse(m.cons.active)

        pwlf = list(
            m.component_data_objects(PiecewiseLinearFunction, descend_into=True)
        )
        self.assertEqual(len(pwlf), 1)
        pwlf = pwlf[0]

        points = [(1.0009,), (5.5,), (9.9991,)]
        (x1, x2, x3) = 1.0009, 5.5, 9.9991
        self.check_pw_linear_log_x(m, pwlf, x1, x2, x3)

        # quad is not
        self.assertTrue(m.quad.active)

    def test_crazy_target_error(self):
        m = self.make_model()

        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        with self.assertRaisesRegex(
            ValueError,
            "Target 'x' is not a Block, Constraint, or Objective. It "
            "is of type '<class 'pyomo.core.base.var.ScalarVar'>' and cannot "
            "be transformed.",
        ):
            n_to_pwl.apply_to(
                m,
                num_points=3,
                domain_partitioning_method=DomainPartitioningMethod.UNIFORM_GRID,
                targets=[m.x],
            )

    def test_cannot_approximate_constraints_with_unbounded_vars(self):
        m = ConcreteModel()
        m.x = Var()
        m.quad = Constraint(expr=m.x**2 <= 9)

        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        with self.assertRaisesRegex(
            ValueError,
            "Cannot automatically approximate constraints with unbounded "
            "variables. Var 'x' appearing in component 'quad' is missing "
            "at least one bound",
        ):
            n_to_pwl.apply_to(
                m,
                num_points=3,
                domain_partitioning_method=DomainPartitioningMethod.UNIFORM_GRID,
            )

    # def test_log_constraint_lmt_uniform_sample(self):
    #     m = self.make_model()

    #     n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
    #     n_to_pwl.apply_to(
    #         m,
    #         num_points=3,
    #         domain_partitioning_method=DomainPartitioningMethod.LINEAR_MODEL_TREE_UNIFORM,
    #     )

    #     # cons is transformed
    #     self.assertFalse(m.cons.active)

    #     pwlf = list(m.component_data_objects(PiecewiseLinearFunction,
    #                                          descend_into=True))
    #     self.assertEqual(len(pwlf), 1)
    #     pwlf = pwlf[0]

    #     set_trace()

    #     # TODO
    #     x1 = 4.370861069626263
    #     x2 = 7.587945476302646
    #     x3 = 9.556428757689245
    #     self.check_pw_linear_log_x(m, pwlf, x1, x2, x3)


class TestNonlinearToPWL_2D(unittest.TestCase):
    def make_paraboloid_model(self):
        m = ConcreteModel()
        m.x1 = Var(bounds=(0, 3))
        m.x2 = Var(bounds=(1, 7))
        m.obj = Objective(expr=m.x1**2 + m.x2**2)

        return m

    def check_pw_linear_paraboloid(self, m, pwlf, x1, x2, y1, y2):
        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        points = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
        self.assertEqual(pwlf._points, points)
        self.assertEqual(pwlf._simplices, [(0, 1, 3), (0, 2, 3)])
        self.assertEqual(len(pwlf._linear_functions), 2)

        # just check that the linear functions make sense--they intersect the
        # paraboloid at the vertices of the simplices.
        self.assertAlmostEqual(pwlf._linear_functions[0](x1, y1), x1**2 + y1**2)
        self.assertAlmostEqual(pwlf._linear_functions[0](x1, y2), x1**2 + y2**2)
        self.assertAlmostEqual(pwlf._linear_functions[0](x2, y2), x2**2 + y2**2)

        self.assertAlmostEqual(pwlf._linear_functions[1](x1, y1), x1**2 + y1**2)
        self.assertAlmostEqual(pwlf._linear_functions[1](x2, y1), x2**2 + y1**2)
        self.assertAlmostEqual(pwlf._linear_functions[1](x2, y2), x2**2 + y2**2)

        self.assertEqual(len(pwlf._expressions), 1)
        new_obj = n_to_pwl.get_transformed_component(m.obj)
        self.assertTrue(new_obj.active)
        self.assertIs(new_obj.expr, pwlf._expressions[id(new_obj.expr.expr)])
        self.assertIs(n_to_pwl.get_src_component(new_obj), m.obj)

        quadratic = n_to_pwl.get_transformed_quadratic_constraints(m)
        self.assertEqual(len(quadratic), 0)
        nonlinear = n_to_pwl.get_transformed_nonlinear_constraints(m)
        self.assertEqual(len(nonlinear), 0)
        quadratic = n_to_pwl.get_transformed_quadratic_objectives(m)
        self.assertEqual(len(quadratic), 1)
        self.assertIn(m.obj, quadratic)
        nonlinear = n_to_pwl.get_transformed_nonlinear_objectives(m)
        self.assertEqual(len(nonlinear), 0)

    def test_paraboloid_objective_uniform_grid(self):
        m = self.make_paraboloid_model()

        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        n_to_pwl.apply_to(
            m,
            num_points=2,
            domain_partitioning_method=DomainPartitioningMethod.UNIFORM_GRID,
        )

        # check obj is transformed
        self.assertFalse(m.obj.active)

        pwlf = list(
            m.component_data_objects(PiecewiseLinearFunction, descend_into=True)
        )
        self.assertEqual(len(pwlf), 1)
        pwlf = pwlf[0]

        x1 = 0.00030000000000000003
        x2 = 2.9997
        y1 = 1.0006
        y2 = 6.9994

        self.check_pw_linear_paraboloid(m, pwlf, x1, x2, y1, y2)

    def test_objective_target(self):
        m = self.make_paraboloid_model()

        m.some_other_nonlinear_constraint = Constraint(expr=m.x1**3 + m.x2 <= 6)

        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        n_to_pwl.apply_to(
            m,
            num_points=2,
            domain_partitioning_method=DomainPartitioningMethod.UNIFORM_GRID,
            targets=[m.obj],
        )

        # check obj is transformed
        self.assertFalse(m.obj.active)

        pwlf = list(
            m.component_data_objects(PiecewiseLinearFunction, descend_into=True)
        )
        self.assertEqual(len(pwlf), 1)
        pwlf = pwlf[0]

        x1 = 0.00030000000000000003
        x2 = 2.9997
        y1 = 1.0006
        y2 = 6.9994

        self.check_pw_linear_paraboloid(m, pwlf, x1, x2, y1, y2)

        # and check that the constraint isn't transformed
        self.assertTrue(m.some_other_nonlinear_constraint.active)

    def test_do_not_transform_quadratic_objective(self):
        m = self.make_paraboloid_model()

        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        n_to_pwl.apply_to(
            m,
            num_points=2,
            domain_partitioning_method=DomainPartitioningMethod.UNIFORM_GRID,
            approximate_quadratic_objectives=False,
        )

        # check obj is *not* transformed
        self.assertTrue(m.obj.active)

        quadratic = n_to_pwl.get_transformed_quadratic_constraints(m)
        self.assertEqual(len(quadratic), 0)
        nonlinear = n_to_pwl.get_transformed_nonlinear_constraints(m)
        self.assertEqual(len(nonlinear), 0)
        quadratic = n_to_pwl.get_transformed_quadratic_objectives(m)
        self.assertEqual(len(quadratic), 0)
        nonlinear = n_to_pwl.get_transformed_nonlinear_objectives(m)
        self.assertEqual(len(nonlinear), 0)


class TestNonlinearToPWLIntegration(unittest.TestCase):
    def test_additively_decompose(self):
        m = ConcreteModel()
        m.x1 = Var(within=Reals, bounds=(0, 2), initialize=1.745)
        m.x4 = Var(within=Reals, bounds=(0, 5), initialize=3.048)
        m.x7 = Var(within=Reals, bounds=(0.9, 0.95), initialize=0.928)
        m.obj = Objective(expr=-6.3 * m.x4 * m.x7 + 5.04 * m.x1)
        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        n_to_pwl.apply_to(
            m,
            num_points=4,
            domain_partitioning_method=DomainPartitioningMethod.LINEAR_MODEL_TREE_RANDOM,
            additively_decompose=True,
        )

        self.assertFalse(m.obj.active)
        new_obj = n_to_pwl.get_transformed_component(m.obj)
        self.assertIs(n_to_pwl.get_src_component(new_obj), m.obj)
        self.assertTrue(new_obj.active)
        # two terms
        self.assertIsInstance(new_obj.expr, SumExpression)
        self.assertEqual(len(new_obj.expr.args), 2)
        first = new_obj.expr.args[0]
        pwlf = first.expr.pw_linear_function
        all_pwlf = list(
            m.component_data_objects(PiecewiseLinearFunction, descend_into=True)
        )
        self.assertEqual(len(all_pwlf), 1)
        # It is on the active tree.
        self.assertIs(pwlf, all_pwlf[0])

        second = new_obj.expr.args[1]
        assertExpressionsEqual(self, second, 5.04 * m.x1)

        objs = n_to_pwl.get_transformed_nonlinear_objectives(m)
        self.assertEqual(len(objs), 0)
        objs = n_to_pwl.get_transformed_quadratic_objectives(m)
        self.assertEqual(len(objs), 1)
        self.assertIn(m.obj, objs)
        self.assertEqual(len(n_to_pwl.get_transformed_nonlinear_constraints(m)), 0)
        self.assertEqual(len(n_to_pwl.get_transformed_quadratic_constraints(m)), 0)

        TransformationFactory('contrib.piecewise.outer_repn_gdp').apply_to(m)
        TransformationFactory('gdp.bigm').apply_to(m)
        SolverFactory('gurobi').solve(m)


#     def test_Ali_example(self):
#         m = ConcreteModel()
#         m.flow_super_heated_vapor = Var()
#         m.flow_super_heated_vapor.fix(0.4586949988166174)
#         m.super_heated_vapor_temperature = Var(bounds=(31, 200), initialize=45)
#         m.evaporator_condensate_temperature = Var(
#             bounds=(29, 120.8291392028045), initialize=30
#         )
#         m.LMTD = Var(bounds=(0, 130.61608989795093), initialize=1)
#         m.evaporator_condensate_enthalpy = Var(
#             bounds=(-15836.847, -15510.210751855624), initialize=100
#         )
#         m.evaporator_condensate_vapor_enthalpy = Var(
#             bounds=(-13416.64, -13247.674383866839), initialize=100
#         )
#         m.heat_transfer_coef = Var(
#             bounds=(1.9936854577372858, 5.995319594088982), initialize=0.1
#         )
#         m.evaporator_brine_temperature = Var(
#             bounds=(27, 118.82913920280366), initialize=35
#         )
#         m.each_evaporator_area = Var()

#         m.c = Constraint(
#             expr=m.each_evaporator_area
#             == (
#                 1.873
#                 * m.flow_super_heated_vapor
#                 * (
#                     m.super_heated_vapor_temperature
#                     - m.evaporator_condensate_temperature
#                 )
#                 / (100 * m.LMTD)
#                 + m.flow_super_heated_vapor
#                 * (
#                     m.evaporator_condensate_vapor_enthalpy
#                     - m.evaporator_condensate_enthalpy
#                 )
#                 / (
#                     m.heat_transfer_coef
#                     * (
#                         m.evaporator_condensate_temperature
#                         - m.evaporator_brine_temperature
#                     )
#                 )
#             )
#         )

#         n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
#         n_to_pwl.apply_to(
#             m,
#             num_points=3,
#             domain_partitioning_method=DomainPartitioningMethod.UNIFORM_GRID,
#         )

#         m.pprint()

#         from pyomo.environ import SolverFactory
#         SolverFactory('gurobi').solve(m, tee=True)
