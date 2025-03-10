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

from io import StringIO
import logging

from pyomo.common.dependencies import attempt_import, scipy_available, numpy_available
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.contrib.piecewise.transform.nonlinear_to_pwl import (
    NonlinearToPWL,
    DomainPartitioningMethod,
)
from pyomo.core.base.expression import _ExpressionData
from pyomo.core.expr.compare import (
    assertExpressionsEqual,
    assertExpressionsStructurallyEqual,
)
from pyomo.core.expr.numeric_expr import SumExpression
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Var,
    Constraint,
    Integers,
    TransformationFactory,
    log,
    Objective,
    Reals,
    SolverFactory,
    TerminationCondition,
    value,
)

gurobi_available = (
    SolverFactory('gurobi').available(exception_flag=False)
    and SolverFactory('gurobi').license_is_valid()
)
lineartree_available = attempt_import('lineartree')[1]
sklearn_available = attempt_import('sklearn.linear_model')[1]


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
        self.assertIs(
            new_cons.body, pwlf._expressions[pwlf._expression_ids[new_cons.body.expr]]
        )
        self.assertIsNone(new_cons.ub)
        self.assertEqual(new_cons.lb, 0.35)
        self.assertIs(n_to_pwl.get_src_component(new_cons), m.cons)

        quadratic = n_to_pwl.get_transformed_quadratic_constraints(m)
        self.assertEqual(len(quadratic), 0)
        nonlinear = n_to_pwl.get_transformed_nonlinear_constraints(m)
        self.assertEqual(len(nonlinear), 1)
        self.assertIn(m.cons, nonlinear)

    @unittest.skipUnless(numpy_available, "Numpy is not available")
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

    @unittest.skipUnless(numpy_available, "Numpy is not available")
    def test_clone_transformed_model(self):
        m = self.make_model()

        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        n_to_pwl.apply_to(
            m,
            num_points=3,
            domain_partitioning_method=DomainPartitioningMethod.UNIFORM_GRID,
        )

        twin = m.clone()

        # cons is transformed
        self.assertFalse(twin.cons.active)

        pwlf = list(
            twin.component_data_objects(PiecewiseLinearFunction, descend_into=True)
        )
        self.assertEqual(len(pwlf), 1)
        pwlf = pwlf[0]

        points = [(1.0009,), (5.5,), (9.9991,)]
        (x1, x2, x3) = 1.0009, 5.5, 9.9991

        self.check_pw_linear_log_x(twin, pwlf, x1, x2, x3)

    @unittest.skipUnless(numpy_available, "Numpy is not available")
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

    @unittest.skipUnless(numpy_available, "Numpy is not available")
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

    @unittest.skipUnless(numpy_available, "Numpy is not available")
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

    def test_error_for_non_separable_exceeding_max_dimension(self):
        m = ConcreteModel()
        m.x = Var([0, 1, 2, 3, 4], bounds=(-4, 5))
        m.ick = Constraint(expr=m.x[0] ** (m.x[1] * m.x[2] * m.x[3] * m.x[4]) <= 8)

        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        with self.assertRaisesRegex(
            ValueError,
            "Not approximating expression for component 'ick' as "
            "it exceeds the maximum dimension of 4. Try increasing "
            "'max_dimension' or additively separating the expression.",
        ):
            n_to_pwl.apply_to(
                m,
                num_points=3,
                domain_partitioning_method=DomainPartitioningMethod.UNIFORM_GRID,
                max_dimension=4,
            )

    @unittest.skipUnless(numpy_available, "Numpy is not available")
    @unittest.skipUnless(scipy_available, "Scipy is not available")
    def test_do_not_additively_decompose_below_min_dimension(self):
        m = ConcreteModel()
        m.x = Var([0, 1, 2, 3, 4], bounds=(-4, 5))
        m.c = Constraint(expr=m.x[0] * m.x[1] + m.x[3] <= 4)

        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        n_to_pwl.apply_to(
            m,
            num_points=3,
            additively_decompose=True,
            min_dimension_to_additively_decompose=4,
            domain_partitioning_method=DomainPartitioningMethod.UNIFORM_GRID,
        )

        transformed_c = n_to_pwl.get_transformed_component(m.c)
        # This is only approximated by one pwlf:
        self.assertIsInstance(transformed_c.body, _ExpressionData)

    @unittest.skipUnless(numpy_available, "Numpy is not available")
    def test_uniform_sampling_discrete_vars(self):
        m = ConcreteModel()
        m.x = Var(['rocky', 'bullwinkle'], domain=Binary)
        m.y = Var(domain=Integers, bounds=(0, 5))
        m.c = Constraint(expr=m.x['rocky'] * m.x['bullwinkle'] + m.y <= 4)

        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core', logging.WARNING):
            n_to_pwl.apply_to(
                m,
                num_points=3,
                additively_decompose=False,
                domain_partitioning_method=DomainPartitioningMethod.UNIFORM_GRID,
            )
        # No warnings (this is to check that we aren't emitting a bunch of
        # warnings about setting variables outside of their domains)
        self.assertEqual(output.getvalue().strip(), "")

        transformed_c = n_to_pwl.get_transformed_component(m.c)
        pwlf = transformed_c.body.expr.pw_linear_function

        # should sample 0, 1 for th m.x's
        # should sample 0, 2, 5 for m.y (because of half to even rounding (*sigh*))
        points = set(pwlf._points)
        self.assertEqual(len(points), 12)
        for x in [0, 1]:
            for y in [0, 1]:
                for z in [0, 2, 5]:
                    self.assertIn((x, y, z), points)

    @unittest.skipUnless(numpy_available, "Numpy is not available")
    @unittest.skipUnless(scipy_available, "Scipy is not available")
    def test_uniform_sampling_discrete_vars(self):
        m = ConcreteModel()
        m.x = Var(['rocky', 'bullwinkle'], domain=Binary)
        m.y = Var(domain=Integers, bounds=(0, 5))
        m.c = Constraint(expr=m.x['rocky'] * m.x['bullwinkle'] + m.y <= 4)

        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core', logging.WARNING):
            n_to_pwl.apply_to(
                m,
                num_points=3,
                additively_decompose=False,
                domain_partitioning_method=DomainPartitioningMethod.UNIFORM_GRID,
            )
        # No warnings (this is to check that we aren't emitting a bunch of
        # warnings about setting variables outside of their domains)
        self.assertEqual(output.getvalue().strip(), "")

        transformed_c = n_to_pwl.get_transformed_component(m.c)
        pwlf = transformed_c.body.expr.pw_linear_function

        # should sample 0, 1 for th m.x's
        # should sample 0, 2, 5 for m.y (because of half to even rounding (*sigh*))
        points = set(pwlf._points)
        self.assertEqual(len(points), 12)
        for x in [0, 1]:
            for y in [0, 1]:
                for z in [0, 2, 5]:
                    self.assertIn((x, y, z), points)

    @unittest.skipUnless(numpy_available, "Numpy is not available")
    @unittest.skipUnless(scipy_available, "Scipy is not available")
    def test_random_sampling_discrete_vars(self):
        m = ConcreteModel()
        m.x = Var(['rocky', 'bullwinkle'], domain=Binary)
        m.y = Var(domain=Integers, bounds=(0, 5))
        m.c = Constraint(expr=m.x['rocky'] * m.x['bullwinkle'] + m.y <= 4)

        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core', logging.WARNING):
            n_to_pwl.apply_to(
                m,
                num_points=3,
                additively_decompose=False,
                domain_partitioning_method=DomainPartitioningMethod.RANDOM_GRID,
            )
        # No warnings (this is to check that we aren't emitting a bunch of
        # warnings about setting variables outside of their domains)
        self.assertEqual(output.getvalue().strip(), "")

        transformed_c = n_to_pwl.get_transformed_component(m.c)
        pwlf = transformed_c.body.expr.pw_linear_function

        # should sample 0, 1 for th m.x's
        # Happen to get 0, 1, 5 for m.y
        points = set(pwlf._points)
        self.assertEqual(len(points), 12)
        for x in [0, 1]:
            for y in [0, 1]:
                for z in [0, 1, 5]:
                    self.assertIn((x, y, z), points)


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
        self.assertIs(
            new_obj.expr, pwlf._expressions[pwlf._expression_ids[new_obj.expr.expr]]
        )
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

    @unittest.skipUnless(numpy_available, "Numpy is not available")
    @unittest.skipUnless(scipy_available, "Scipy is not available")
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

    @unittest.skipUnless(numpy_available, "Numpy is not available")
    @unittest.skipUnless(scipy_available, "Scipy is not available")
    def test_multivariate_clone(self):
        m = self.make_paraboloid_model()

        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        n_to_pwl.apply_to(
            m,
            num_points=2,
            domain_partitioning_method=DomainPartitioningMethod.UNIFORM_GRID,
        )

        twin = m.clone()

        # check obj is transformed
        self.assertFalse(twin.obj.active)

        pwlf = list(
            twin.component_data_objects(PiecewiseLinearFunction, descend_into=True)
        )
        self.assertEqual(len(pwlf), 1)
        pwlf = pwlf[0]

        x1 = 0.00030000000000000003
        x2 = 2.9997
        y1 = 1.0006
        y2 = 6.9994

        self.check_pw_linear_paraboloid(twin, pwlf, x1, x2, y1, y2)

    @unittest.skipUnless(numpy_available, "Numpy is not available")
    @unittest.skipUnless(scipy_available, "Scipy is not available")
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


@unittest.skipUnless(lineartree_available, "lineartree not available")
@unittest.skipUnless(sklearn_available, "sklearn not available")
class TestLinearTreeDomainPartitioning(unittest.TestCase):
    def make_absolute_value_model(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-10, 10))
        m.obj = Objective(expr=abs(m.x))

        return m

    def test_linear_model_tree_uniform(self):
        m = self.make_absolute_value_model()
        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        n_to_pwl.apply_to(
            m,
            num_points=301,  # sample a lot so we train a good tree
            domain_partitioning_method=DomainPartitioningMethod.LINEAR_MODEL_TREE_UNIFORM,
            linear_tree_max_depth=1,  # force parsimony
        )

        transformed_obj = n_to_pwl.get_transformed_component(m.obj)
        pwlf = transformed_obj.expr.expr.pw_linear_function

        self.assertEqual(len(pwlf._simplices), 2)
        self.assertEqual(pwlf._simplices, [(0, 1), (1, 2)])
        self.assertEqual(pwlf._points, [(-10,), (-0.08402,), (10,)])
        self.assertEqual(len(pwlf._linear_functions), 2)
        assertExpressionsEqual(self, pwlf._linear_functions[0](m.x), -1.0 * m.x)
        assertExpressionsStructurallyEqual(
            self,
            pwlf._linear_functions[1](m.x),
            # pretty close to m.x, but we're a bit off because we don't have 0
            # as a breakpoint.
            0.9833360108369479 * m.x + 0.16663989163052034,
            places=7,
        )

    def test_linear_model_tree_random(self):
        m = self.make_absolute_value_model()
        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        n_to_pwl.apply_to(
            m,
            num_points=300,  # sample a lot so we train a good tree
            domain_partitioning_method=DomainPartitioningMethod.LINEAR_MODEL_TREE_RANDOM,
            linear_tree_max_depth=1,  # force parsimony
        )

        transformed_obj = n_to_pwl.get_transformed_component(m.obj)
        pwlf = transformed_obj.expr.expr.pw_linear_function

        self.assertEqual(len(pwlf._simplices), 2)
        self.assertEqual(pwlf._simplices, [(0, 1), (1, 2)])
        self.assertEqual(pwlf._points, [(-10,), (-0.03638,), (10,)])
        self.assertEqual(len(pwlf._linear_functions), 2)
        assertExpressionsEqual(self, pwlf._linear_functions[0](m.x), -1.0 * m.x)
        assertExpressionsStructurallyEqual(
            self,
            pwlf._linear_functions[1](m.x),
            # pretty close to m.x, but we're a bit off because we don't have 0
            # as a breakpoint.
            0.9927503741388829 * m.x + 0.07249625861117256,
            places=7,
        )

    def test_linear_model_tree_random_auto_depth_tree(self):
        m = self.make_absolute_value_model()
        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        n_to_pwl.apply_to(
            m,
            num_points=100,  # sample a lot but not too many because this one is
            # more prone to overfitting
            domain_partitioning_method=DomainPartitioningMethod.LINEAR_MODEL_TREE_RANDOM,
        )

        transformed_obj = n_to_pwl.get_transformed_component(m.obj)
        pwlf = transformed_obj.expr.expr.pw_linear_function

        print(pwlf._simplices)
        print(pwlf._points)
        for f in pwlf._linear_functions:
            print(f(m.x))

        # We end up with 8, which is just what happens, but it's not a terrible
        # approximation
        self.assertEqual(len(pwlf._simplices), 8)
        self.assertEqual(
            pwlf._simplices,
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)],
        )
        self.assertEqual(
            pwlf._points,
            [
                (-10,),
                (-9.24119,),
                (-8.71428,),
                (-8.11135,),
                (0.06048,),
                (0.70015,),
                (1.9285,),
                (2.15597,),
                (10,),
            ],
        )
        self.assertEqual(len(pwlf._linear_functions), 8)
        for i in range(3):
            assertExpressionsEqual(self, pwlf._linear_functions[i](m.x), -1.0 * m.x)
        assertExpressionsStructurallyEqual(
            self,
            pwlf._linear_functions[3](m.x),
            # pretty close to - m.x, but we're a bit off because we don't have 0
            # as a breakpoint.
            -0.9851979299618323 * m.x + 0.12006477080409184,
            places=7,
        )
        for i in range(4, 8):
            assertExpressionsEqual(self, pwlf._linear_functions[i](m.x), m.x)


class TestNonlinearToPWLIntegration(unittest.TestCase):
    @unittest.skipUnless(gurobi_available, "Gurobi is not available")
    @unittest.skipUnless(scipy_available, "Scipy is not available")
    def test_transform_and_solve_additively_decomposes_model(self):
        # A bit of an integration test to make sure that we build additively
        # decomposed pw-linear approximations in such a way that they are
        # transformed to MILP and solved correctly. (Largely because we have to
        # be careful to make sure that we don't ever directly insert
        # PiecewiseLinearExpression objects into expressions and are instead
        # using the ExpressionData that points to them (and will eventually be
        # replaced in transformation))
        m = ConcreteModel()
        m.x1 = Var(within=Reals, bounds=(0, 2), initialize=1.745)
        m.x4 = Var(within=Reals, bounds=(0, 5), initialize=3.048)
        m.x7 = Var(within=Reals, bounds=(0.9, 0.95), initialize=0.928)
        m.obj = Objective(expr=-6.3 * m.x4 * m.x7 + 5.04 * m.x1)
        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        xm = n_to_pwl.create_using(
            m,
            num_points=4,
            domain_partitioning_method=DomainPartitioningMethod.UNIFORM_GRID,
            additively_decompose=True,
        )

        self.assertFalse(xm.obj.active)
        new_obj = n_to_pwl.get_transformed_component(xm.obj)
        self.assertIs(n_to_pwl.get_src_component(new_obj), xm.obj)
        self.assertTrue(new_obj.active)
        # two terms
        self.assertIsInstance(new_obj.expr, SumExpression)
        self.assertEqual(len(new_obj.expr.args), 2)
        first = new_obj.expr.args[0]
        pwlf = first.expr.pw_linear_function
        all_pwlf = list(
            xm.component_data_objects(PiecewiseLinearFunction, descend_into=True)
        )
        self.assertEqual(len(all_pwlf), 1)
        # It is on the active tree.
        self.assertIs(pwlf, all_pwlf[0])

        second = new_obj.expr.args[1]
        assertExpressionsEqual(self, second, 5.04 * xm.x1)

        objs = n_to_pwl.get_transformed_nonlinear_objectives(xm)
        self.assertEqual(len(objs), 0)
        objs = n_to_pwl.get_transformed_quadratic_objectives(xm)
        self.assertEqual(len(objs), 1)
        self.assertIn(xm.obj, objs)
        self.assertEqual(len(n_to_pwl.get_transformed_nonlinear_constraints(xm)), 0)
        self.assertEqual(len(n_to_pwl.get_transformed_quadratic_constraints(xm)), 0)

        TransformationFactory('contrib.piecewise.outer_repn_gdp').apply_to(xm)
        TransformationFactory('gdp.bigm').apply_to(xm)
        opt = SolverFactory('gurobi')
        results = opt.solve(xm)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )

        # solve the original
        opt.options['NonConvex'] = 2
        results = opt.solve(m)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )

        # Not a bad approximation:
        self.assertAlmostEqual(value(xm.obj), value(m.obj), places=2)

        self.assertAlmostEqual(value(xm.x4), value(m.x4), places=3)
        self.assertAlmostEqual(value(xm.x7), value(m.x7), places=4)
        self.assertAlmostEqual(value(xm.x1), value(m.x1), places=7)
