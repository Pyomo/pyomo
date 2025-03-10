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
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
    Any,
    Binary,
    Block,
    ConcreteModel,
    Constraint,
    Objective,
    Set,
    TransformationFactory,
    Var,
)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.common.log import LoggingIntercept


class TestBoundPretransformation(unittest.TestCase):
    def create_nested_structure(self):
        """
        Creates a two-term Disjunction with on nested two-term Disjunction on
        the first Disjunct
        """
        m = ConcreteModel()
        m.outer_d1 = Disjunct()
        m.outer_d1.inner_d1 = Disjunct()
        m.outer_d1.inner_d2 = Disjunct()
        m.outer_d1.inner = Disjunction(expr=[m.outer_d1.inner_d1, m.outer_d1.inner_d2])
        m.outer_d2 = Disjunct()
        m.outer = Disjunction(expr=[m.outer_d1, m.outer_d2])
        return m

    def create_nested_model(self):
        """
        -100 <= x <= 102
        [-10 <= x <= 11, [x <= 3] v [x >= -7]] v [x == 0]
        """
        m = self.create_nested_structure()
        m.x = Var(bounds=(-100, 102))
        m.outer_d1.c = Constraint(expr=(-10, m.x, 11))
        m.outer_d1.inner_d1.c = Constraint(expr=m.x <= 3)
        m.outer_d1.inner_d2.c = Constraint(expr=m.x >= -7)
        m.outer_d2.c = Constraint(expr=m.x == 0)

        return m

    def check_nested_model_disjunction(self, m, bt):
        # We expect: -10w_1 -7w_2 <= x <= 3w_1 + 11w_2

        cons = bt.get_transformed_constraints(m.x, m.outer)
        self.assertEqual(len(cons), 2)
        lb = cons[0]
        ub = cons[1]
        assertExpressionsEqual(
            self,
            lb.expr,
            -10.0 * m.outer_d1.binary_indicator_var
            + 0.0 * m.outer_d2.binary_indicator_var
            <= m.x,
        )
        assertExpressionsEqual(
            self,
            ub.expr,
            11.0 * m.outer_d1.binary_indicator_var
            + 0.0 * m.outer_d2.binary_indicator_var
            >= m.x,
        )

        cons = bt.get_transformed_constraints(m.x, m.outer_d1.inner)
        self.assertEqual(len(cons), 2)
        lb = cons[0]
        ub = cons[1]
        assertExpressionsEqual(
            self,
            lb.expr,
            -10.0 * m.outer_d1.inner_d1.binary_indicator_var
            - 7.0 * m.outer_d1.inner_d2.binary_indicator_var
            <= m.x,
        )
        assertExpressionsEqual(
            self,
            ub.expr,
            3.0 * m.outer_d1.inner_d1.binary_indicator_var
            + 11.0 * m.outer_d1.inner_d2.binary_indicator_var
            >= m.x,
        )

        # All the disjunctive constraints were transformed
        self.assertFalse(m.outer_d1.c.active)
        self.assertFalse(m.outer_d1.inner_d1.c.active)
        self.assertFalse(m.outer_d1.inner_d2.c.active)
        self.assertFalse(m.outer_d2.c.active)

    def test_transform_nested_model(self):
        m = self.create_nested_model()

        bt = TransformationFactory('gdp.bound_pretransformation')
        bt.apply_to(m)
        self.check_nested_model_disjunction(m, bt)

        # There aren't any other constraints on the model other than what we
        # added
        self.assertEqual(
            len(
                list(
                    m.component_data_objects(
                        Constraint, active=True, descend_into=(Block, Disjunct)
                    )
                )
            ),
            4,
        )

    def test_transform_nested_model_no_0_terms(self):
        m = self.create_nested_model()
        m.outer_d2.c.deactivate()
        m.outer_d2.c2 = Constraint(expr=m.x == 101)

        bt = TransformationFactory('gdp.bound_pretransformation')
        bt.apply_to(m)

        # We expect: -10w_1 -7w_2 + 101 y_2 <= x <= 3w_1 + 11w_2 + 101y_2

        cons = bt.get_transformed_constraints(m.x, m.outer)
        self.assertEqual(len(cons), 2)
        lb = cons[0]
        ub = cons[1]
        assertExpressionsEqual(
            self,
            lb.expr,
            -10.0 * m.outer_d1.binary_indicator_var
            + 101.0 * m.outer_d2.binary_indicator_var
            <= m.x,
        )
        assertExpressionsEqual(
            self,
            ub.expr,
            11.0 * m.outer_d1.binary_indicator_var
            + 101.0 * m.outer_d2.binary_indicator_var
            >= m.x,
        )

        cons = bt.get_transformed_constraints(m.x, m.outer_d1.inner)
        self.assertEqual(len(cons), 2)
        lb = cons[0]
        ub = cons[1]
        assertExpressionsEqual(
            self,
            lb.expr,
            -10.0 * m.outer_d1.inner_d1.binary_indicator_var
            - 7.0 * m.outer_d1.inner_d2.binary_indicator_var
            <= m.x,
        )
        assertExpressionsEqual(
            self,
            ub.expr,
            3.0 * m.outer_d1.inner_d1.binary_indicator_var
            + 11.0 * m.outer_d1.inner_d2.binary_indicator_var
            >= m.x,
        )
        self.assertIs(_parent_disjunct(lb), m.outer_d1)
        self.assertIs(_parent_disjunct(ub), m.outer_d1)

        # All the disjunctive constraints were transformed
        self.assertFalse(m.outer_d1.c.active)
        self.assertFalse(m.outer_d1.inner_d1.c.active)
        self.assertFalse(m.outer_d1.inner_d2.c.active)
        self.assertFalse(m.outer_d2.c2.active)

        # There aren't any other constraints on the model other than what we
        # added
        self.assertEqual(
            len(
                list(
                    m.component_data_objects(
                        Constraint, active=True, descend_into=(Block, Disjunct)
                    )
                )
            ),
            4,
        )

    def test_transformation_gives_up_without_enough_bound_info(self):
        """
        If we have unbounded variables and not enough bounding constraints,
        we want the transformation to just leave the bounding constraints
        be to be transformed later.
        """
        m = self.create_nested_structure()
        m.x = Var()
        m.y = Var(bounds=(4, 67))
        m.outer_d1.c = Constraint(Any)
        m.outer_d1.c[1] = m.x >= 3
        m.outer_d1.c[2] = 5 <= m.y
        m.outer_d1.inner_d1.c = Constraint(Any)
        m.outer_d1.inner_d1.c[1] = m.x >= 4
        m.outer_d1.inner_d2.c = Constraint(Any)
        m.outer_d1.inner_d2.c[1] = m.x >= 17
        m.outer_d2.c = Constraint(Any)
        m.outer_d2.c[1] = m.x <= 1
        m.outer_d2.c[2] = m.y <= 66
        m.outer_d2.c[3] = m.x >= 2

        # y constraints should be fully transformed, and x can do lower but not
        # upper.

        bt = TransformationFactory('gdp.bound_pretransformation')
        bt.apply_to(m)

        # We expect: 3d_1 + 2d_2 <= x
        #            5d_1 + 4d_2 <= y <= 67d_1 + 66d_2
        # in the global scope, and
        # x >= 4a_1 + 17a_2 on outer_d1

        cons = bt.get_transformed_constraints(m.x, m.outer)
        self.assertEqual(len(cons), 1)
        lb = cons[0]
        assertExpressionsEqual(
            self,
            lb.expr,
            3.0 * m.outer_d1.binary_indicator_var
            + 2.0 * m.outer_d2.binary_indicator_var
            <= m.x,
        )

        cons = bt.get_transformed_constraints(m.y, m.outer)
        self.assertEqual(len(cons), 2)
        lb = cons[0]
        assertExpressionsEqual(
            self,
            lb.expr,
            5.0 * m.outer_d1.binary_indicator_var + 4 * m.outer_d2.binary_indicator_var
            <= m.y,
        )
        ub = cons[1]
        assertExpressionsEqual(
            self,
            ub.expr,
            67 * m.outer_d1.binary_indicator_var
            + 66.0 * m.outer_d2.binary_indicator_var
            >= m.y,
        )

        cons = bt.get_transformed_constraints(m.x, m.outer_d1.inner)
        self.assertEqual(len(cons), 1)
        lb = cons[0]
        assertExpressionsEqual(
            self,
            lb.expr,
            4.0 * m.outer_d1.inner_d1.binary_indicator_var
            + 17.0 * m.outer_d1.inner_d2.binary_indicator_var
            <= m.x,
        )
        self.assertIs(_parent_disjunct(cons[0]), m.outer_d1)

        # check that all the y constraints are deactivated, and that the
        # lower bound ones for x are, but not the upper bound ones
        self.assertFalse(m.outer_d1.c[1].active)
        self.assertFalse(m.outer_d1.c[2].active)
        self.assertFalse(m.outer_d1.inner_d1.c[1].active)
        self.assertFalse(m.outer_d1.inner_d2.c[1].active)
        self.assertTrue(m.outer_d2.c[1].active)
        self.assertFalse(m.outer_d2.c[2].active)
        self.assertFalse(m.outer_d2.c[3].active)

        # and check that there are only four active constraints, the ones we
        # made and the remaining upper bound for x:
        self.assertEqual(
            len(
                list(
                    m.component_data_objects(
                        Constraint, active=True, descend_into=(Block, Disjunct)
                    )
                )
            ),
            5,
        )

    def test_partially_deactivating_constraint_lb_transformed_but_ub_not(self):
        m = ConcreteModel()
        m.x = Var()
        m.x.setlb(3.0)
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=(4.5, m.x, 6))
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=5.6 <= m.x)
        m.disj = Disjunction(expr=[m.d1, m.d2])

        bt = TransformationFactory('gdp.bound_pretransformation')
        bt.apply_to(m)

        # We expect: 4.5*y_1 + 5.6*y_2 <= x

        cons = bt.get_transformed_constraints(m.x, m.disj)
        self.assertEqual(len(cons), 1)
        lb = cons[0]
        assertExpressionsEqual(
            self,
            lb.expr,
            4.5 * m.d1.binary_indicator_var + 5.6 * m.d2.binary_indicator_var <= m.x,
        )

        self.assertFalse(m.d1.c.active)
        self.assertFalse(m.d2.c.active)
        c_ub = m.d1.component('c_ub')
        self.assertIsInstance(c_ub, Constraint)
        self.assertTrue(c_ub.active)
        assertExpressionsEqual(self, c_ub.expr, m.x <= 6.0)

        self.assertEqual(
            len(
                list(
                    m.component_data_objects(
                        Constraint, active=True, descend_into=(Block, Disjunct)
                    )
                )
            ),
            2,
        )

    def test_partially_deactivating_constraint_ub_transformed_but_lb_not(self):
        m = ConcreteModel()
        m.w = Var()
        m.d = Disjunct([1, 2, 3])
        m.disjunction = Disjunction(expr=[m.d[1], m.d[2], m.d[3]])

        m.d[1].c = Constraint(expr=m.w == 45)
        m.d[2].c = Constraint(expr=m.w <= 36)
        m.d[3].c = Constraint(expr=m.w <= 232)

        bt = TransformationFactory('gdp.bound_pretransformation')
        bt.apply_to(m)

        # We expect: 45*y_1 + 36y_2 + 232y_3 >= w

        cons = bt.get_transformed_constraints(m.w, m.disjunction)
        self.assertEqual(len(cons), 1)
        ub = cons[0]
        assertExpressionsEqual(
            self,
            ub.expr,
            45.0 * m.d[1].binary_indicator_var
            + 36.0 * m.d[2].binary_indicator_var
            + 232.0 * m.d[3].binary_indicator_var
            >= m.w,
        )

        self.assertFalse(m.d[1].c.active)
        self.assertFalse(m.d[2].c.active)
        self.assertFalse(m.d[3].c.active)
        c_lb = m.d[1].component('c_lb')
        self.assertIsInstance(c_lb, Constraint)
        self.assertTrue(c_lb.active)
        assertExpressionsEqual(self, c_lb.expr, m.w >= 45.0)

        self.assertEqual(
            len(
                list(
                    m.component_data_objects(
                        Constraint, active=True, descend_into=(Block, Disjunct)
                    )
                )
            ),
            2,
        )

    def create_two_disjunction_model(self):
        m = self.create_nested_model()
        m.y = Var()
        m.d1 = Disjunct()
        m.d2 = Disjunct()
        m.d3 = Disjunct()
        m.disjunction = Disjunction(expr=[m.d1, m.d2, m.d3])

        m.d1.c = Constraint(expr=m.y == 7.8)
        m.d1.c_x = Constraint(expr=m.x <= 27)
        m.d2.c = Constraint(expr=m.y == 8.9)
        m.d2.c_x = Constraint(expr=m.x >= 34)
        m.d3.c = Constraint(expr=m.y <= 45.7)
        return m

    def test_transform_multiple_disjunctions(self):
        m = self.create_two_disjunction_model()

        bt = TransformationFactory('gdp.bound_pretransformation')
        bt.apply_to(m)

        self.check_nested_model_disjunction(m, bt)

        cons = bt.get_transformed_constraints(m.x, m.disjunction)
        self.assertEqual(len(cons), 2)
        lb = cons[0]
        assertExpressionsEqual(
            self,
            lb.expr,
            -100 * m.d1.binary_indicator_var
            + 34.0 * m.d2.binary_indicator_var
            + -100 * m.d3.binary_indicator_var
            <= m.x,
        )
        ub = cons[1]
        assertExpressionsEqual(
            self,
            ub.expr,
            27.0 * m.d1.binary_indicator_var
            + 102 * m.d2.binary_indicator_var
            + 102 * m.d3.binary_indicator_var
            >= m.x,
        )

        cons = bt.get_transformed_constraints(m.y, m.disjunction)
        self.assertEqual(len(cons), 1)
        ub = cons[0]
        assertExpressionsEqual(
            self,
            ub.expr,
            7.8 * m.d1.binary_indicator_var
            + 8.9 * m.d2.binary_indicator_var
            + 45.7 * m.d3.binary_indicator_var
            >= m.y,
        )

        self.assertFalse(m.d1.c.active)
        self.assertFalse(m.d1.c_x.active)
        self.assertFalse(m.d2.c.active)
        self.assertFalse(m.d2.c_x.active)
        self.assertFalse(m.d3.c.active)

        c_lb = m.d1.component('c_lb')
        self.assertIsInstance(c_lb, Constraint)
        self.assertTrue(c_lb.active)
        assertExpressionsEqual(self, c_lb.expr, 7.8 <= m.y)
        c_lb = m.d2.component('c_lb')
        self.assertIsInstance(c_lb, Constraint)
        self.assertTrue(c_lb.active)
        assertExpressionsEqual(self, c_lb.expr, 8.9 <= m.y)

        self.assertEqual(
            len(
                list(
                    m.component_data_objects(
                        Constraint, active=True, descend_into=(Block, Disjunct)
                    )
                )
            ),
            9,
        )

    def test_disjunction_target(self):
        m = self.create_two_disjunction_model()

        bt = TransformationFactory('gdp.bound_pretransformation')
        bt.apply_to(m, targets=m.outer)

        self.check_nested_model_disjunction(m, bt)

        self.assertTrue(m.d1.c.active)
        self.assertTrue(m.d1.c_x.active)
        self.assertTrue(m.d2.c.active)
        self.assertTrue(m.d2.c_x.active)
        self.assertTrue(m.d3.c.active)

        self.assertEqual(
            len(
                list(
                    m.component_data_objects(
                        Constraint, active=True, descend_into=(Block, Disjunct)
                    )
                )
            ),
            9,
        )

    def test_get_transformed_constraint_errors(self):
        m = self.create_two_disjunction_model()
        m.z = Var()

        bt = TransformationFactory('gdp.bound_pretransformation')
        bt.apply_to(m, targets=m.outer)

        out = StringIO()
        with LoggingIntercept(
            out, 'pyomo.gdp.plugins.bound_pretransformation', logging.DEBUG
        ):
            nothing = bt.get_transformed_constraints(m.z, m.outer)
        self.assertEqual(len(nothing), 0)
        # ...And we log that we're a bit confused.
        self.assertEqual(
            out.getvalue(),
            "Constraint bounding variable 'z' on Disjunction 'outer' was "
            "not transformed by the 'gdp.bound_pretransformation' "
            "transformation\n",
        )

        out = StringIO()
        with LoggingIntercept(
            out, 'pyomo.gdp.plugins.bound_pretransformation', logging.DEBUG
        ):
            nothing = bt.get_transformed_constraints(m.x, m.disjunction)
        self.assertEqual(len(nothing), 0)
        self.assertEqual(
            out.getvalue(),
            "No variable on Disjunction 'disjunction' was transformed with the "
            "gdp.bound_pretransformation transformation\n",
        )

    def test_univariate_constraints_with_expression_bodies(self):
        m = self.create_nested_structure()

        # This is a convoluted way to write the same model as the nested model
        m.x = Var(bounds=(-100, 102))
        m.outer_d1.c = Constraint(expr=-20 <= 2 * m.x)
        m.outer_d1.c2 = Constraint(expr=m.x - 1 <= 10)
        m.outer_d1.inner_d1.c = Constraint(expr=3 * m.x - 7 <= 2)
        m.outer_d1.inner_d2.c = Constraint(expr=m.x >= -7)
        m.outer_d2.c = Constraint(expr=m.x + 4 == 4)

        bt = TransformationFactory('gdp.bound_pretransformation')
        bt.apply_to(m)

        self.check_nested_model_disjunction(m, bt)

        self.assertFalse(m.outer_d1.c.active)
        self.assertFalse(m.outer_d1.c2.active)
        self.assertFalse(m.outer_d1.inner_d1.c.active)
        self.assertFalse(m.outer_d1.inner_d2.c.active)
        self.assertFalse(m.outer_d2.c.active)

        self.assertEqual(
            len(
                list(
                    m.component_data_objects(
                        Constraint, descend_into=(Block, Disjunct), active=True
                    )
                )
            ),
            4,
        )

    def test_bound_constraints_skip_levels_in_hierarchy(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.y = Var()
        m.Y = Disjunct([1, 2])
        m.Z = Disjunct([1, 2, 3])
        m.W = Disjunct([1, 2])
        m.W[1].c = Constraint(expr=m.x <= 7)
        m.W[2].c = Constraint(expr=m.x <= 9)
        m.Z[1].c = Constraint(expr=m.y == 0)
        m.Z[1].w_disj = Disjunction(expr=[m.W[i] for i in [1, 2]])
        m.Z[2].c = Constraint(expr=m.y == 1)
        m.Z[3].c = Constraint(expr=m.y == 2)
        m.Y[1].c = Constraint(expr=m.x >= 2)
        m.Y[1].z_disj = Disjunction(expr=[m.Z[i] for i in [1, 2, 3]])
        m.Y[2].c1 = Constraint(expr=m.x == 0)
        m.Y[2].c2 = Constraint(expr=(3, m.y, 17))
        m.y_disj = Disjunction(expr=[m.Y[i] for i in [1, 2]])

        bt = TransformationFactory('gdp.bound_pretransformation')
        bt.apply_to(m)

        cons = bt.get_transformed_constraints(m.x, m.y_disj)
        self.assertEqual(len(cons), 2)
        x_lb = cons[0]
        assertExpressionsEqual(
            self,
            x_lb.expr,
            2.0 * m.Y[1].binary_indicator_var + 0 * m.Y[2].binary_indicator_var <= m.x,
        )
        x_ub = cons[1]
        assertExpressionsEqual(
            self,
            x_ub.expr,
            10 * m.Y[1].binary_indicator_var + 0.0 * m.Y[2].binary_indicator_var >= m.x,
        )
        self.assertIsNone(_parent_disjunct(x_lb))
        self.assertIsNone(_parent_disjunct(x_ub))

        cons = bt.get_transformed_constraints(m.y, m.y_disj)
        self.assertEqual(len(cons), 0)

        cons = bt.get_transformed_constraints(m.y, m.Y[1].z_disj)
        self.assertEqual(len(cons), 2)
        y_lb = cons[0]
        assertExpressionsEqual(
            self,
            y_lb.expr,
            0.0 * m.Z[1].binary_indicator_var
            + 1.0 * m.Z[2].binary_indicator_var
            + 2.0 * m.Z[3].binary_indicator_var
            <= m.y,
        )
        y_ub = cons[1]
        assertExpressionsEqual(
            self,
            y_ub.expr,
            0.0 * m.Z[1].binary_indicator_var
            + 1.0 * m.Z[2].binary_indicator_var
            + 2.0 * m.Z[3].binary_indicator_var
            >= m.y,
        )

        cons = bt.get_transformed_constraints(m.x, m.Y[1].z_disj)
        self.assertEqual(len(cons), 0)

        cons = bt.get_transformed_constraints(m.y, m.Z[1].w_disj)
        self.assertEqual(len(cons), 0)

        cons = bt.get_transformed_constraints(m.x, m.Z[1].w_disj)
        self.assertEqual(len(cons), 2)
        x_lb = cons[0]
        assertExpressionsEqual(
            self,
            x_lb.expr,
            2.0 * m.W[1].binary_indicator_var + 2.0 * m.W[2].binary_indicator_var
            <= m.x,
        )
        x_ub = cons[1]
        assertExpressionsEqual(
            self,
            x_ub.expr,
            7.0 * m.W[1].binary_indicator_var + 9.0 * m.W[2].binary_indicator_var
            >= m.x,
        )

        self.assertFalse(m.W[1].c.active)
        self.assertFalse(m.W[2].c.active)
        self.assertFalse(m.Z[1].c.active)
        self.assertFalse(m.Z[2].c.active)
        self.assertFalse(m.Z[3].c.active)
        self.assertFalse(m.Y[1].c.active)
        self.assertFalse(m.Y[2].c1.active)
        self.assertTrue(m.Y[2].c2.active)

        self.assertEqual(
            len(
                list(
                    m.component_data_objects(
                        Constraint, descend_into=(Block, Disjunct), active=True
                    )
                )
            ),
            7,
        )

    def test_skip_nonlinear_and_multivariate_constraints(self):
        m = self.create_nested_model()
        m.y = Var()
        m.z = Var()
        m.outer_d1.nonlinear = Constraint(expr=m.y**2 <= 7)
        m.outer_d1.inner_d2.multivariate = Constraint(expr=m.x + m.y <= m.z)
        m.outer_d2.leave_it = Constraint(expr=m.z == 7)

        bt = TransformationFactory('gdp.bound_pretransformation')
        bt.apply_to(m)

        self.check_nested_model_disjunction(m, bt)

        self.assertTrue(m.outer_d1.nonlinear.active)
        self.assertTrue(m.outer_d1.inner_d2.multivariate.active)
        self.assertTrue(m.outer_d2.leave_it.active)

        self.assertFalse(m.outer_d1.c.active)
        self.assertFalse(m.outer_d1.inner_d1.c.active)
        self.assertFalse(m.outer_d1.inner_d2.c.active)
        self.assertFalse(m.outer_d2.c.active)

        self.assertEqual(
            len(
                list(
                    m.component_data_objects(
                        Constraint, descend_into=(Block, Disjunct), active=True
                    )
                )
            ),
            7,
        )

    def test_tightest_bound_is_at_root(self):
        """
        x >= 60
        [ [x >= 55, [ ] v [x >= 66] ] ] v [x >= 5]
        """
        m = ConcreteModel()
        m.x = Var()
        m.x.setlb(4)
        m.c = Constraint(expr=m.x >= 60)
        m.d = Disjunct([1, 2])
        m.inner1 = Disjunct([1, 2])
        m.inner2 = Disjunct([1, 2])
        m.disjunction = Disjunction(expr=[m.d[1], m.d[2]])
        m.d[1].disjunction = Disjunction(expr=[m.inner1[1], m.inner1[2]])
        m.inner1[1].disjunction = Disjunction(expr=[m.inner2[1], m.inner2[2]])

        m.d[2].c = Constraint(expr=m.x >= 5)
        m.inner1[1].c = Constraint(expr=m.x >= 55)
        m.inner2[2].c = Constraint(expr=m.x >= 66)

        bt = TransformationFactory('gdp.bound_pretransformation')
        bt.apply_to(m)

        cons = bt.get_transformed_constraints(m.x, m.disjunction)
        self.assertEqual(len(cons), 1)
        lb = cons[0]
        print(lb.expr)
        assertExpressionsEqual(
            self,
            lb.expr,
            60.0 * m.d[1].binary_indicator_var + 60.0 * m.d[2].binary_indicator_var
            <= m.x,
        )
        self.assertIsNone(_parent_disjunct(lb))

        cons = bt.get_transformed_constraints(m.x, m.d[1].disjunction)
        self.assertEqual(len(cons), 1)
        lb = cons[0]
        assertExpressionsEqual(
            self,
            lb.expr,
            60.0 * m.inner1[1].binary_indicator_var
            + 60.0 * m.inner1[2].binary_indicator_var
            <= m.x,
        )
        self.assertIs(_parent_disjunct(lb), m.d[1])

        cons = bt.get_transformed_constraints(m.x, m.inner1[1].disjunction)
        self.assertEqual(len(cons), 1)
        lb = cons[0]
        assertExpressionsEqual(
            self,
            lb.expr,
            60.0 * m.inner2[1].binary_indicator_var
            + 66.0 * m.inner2[2].binary_indicator_var
            <= m.x,
        )
        self.assertIs(_parent_disjunct(lb), m.inner1[1])

        # We shouldn't deactivate global constraints. Reason 1 being that we
        # don't deactivate bounds and Reason 2 being that generally the global
        # part of the model is none of pyomo.gdp's beeswax.
        self.assertEqual(
            len(
                list(
                    m.component_data_objects(
                        Constraint, descend_into=(Block, Disjunct), active=True
                    )
                )
            ),
            4,
        )

    def test_bounds_on_disjuncts_with_block_hierarchies(self):
        m = ConcreteModel()
        m.x = Var()
        m.b = Block()
        m.b.c = Constraint(expr=m.x <= 4)
        m.d = Disjunct([1, 2])
        m.d[1].b = Block()
        m.d[1].b.c = Constraint(expr=m.x <= 5)
        m.d[2].b = Block()
        m.d[2].b.c = Constraint(expr=m.x <= 3)
        m.d[2].c = Constraint(expr=m.x <= 4.1)
        m.disjunction = Disjunction(expr=[m.d[1], m.d[2]])

        bt = TransformationFactory('gdp.bound_pretransformation')
        bt.apply_to(m)

        cons = bt.get_transformed_constraints(m.x, m.disjunction)
        self.assertEqual(len(cons), 1)
        ub = cons[0]

        assertExpressionsEqual(
            self,
            ub.expr,
            4.0 * m.d[1].binary_indicator_var + 3.0 * m.d[2].binary_indicator_var
            >= m.x,
        )
        # just the one we made and the global one are active.
        self.assertEqual(
            len(
                list(
                    m.component_data_objects(
                        Constraint, descend_into=(Block, Disjunct), active=True
                    )
                )
            ),
            2,
        )

    def test_indexed_disjunction_target(self):
        m = ConcreteModel()
        m.x = Var()
        m.d = Disjunct([1, 2, 3, 4, 5])
        m.d[1].c = Constraint(expr=m.x <= 1)
        m.d[2].c = Constraint(expr=m.x <= 2)
        m.d[3].c = Constraint(expr=m.x <= 3)
        m.d[4].c = Constraint(expr=m.x >= -5)
        m.d[5].c = Constraint(expr=m.x >= -8)
        m.disjunction = Disjunction(['pos', 'neg'])
        m.disjunction['pos'] = [m.d[1], m.d[2], m.d[3]]
        m.disjunction['neg'] = [m.d[4], m.d[5]]

        bt = TransformationFactory('gdp.bound_pretransformation')
        bt.apply_to(m, targets=m.disjunction)

        cons = bt.get_transformed_constraints(m.x, m.disjunction['pos'])
        self.assertEqual(len(cons), 1)
        ub = cons[0]
        assertExpressionsEqual(
            self,
            ub.expr,
            1.0 * m.d[1].binary_indicator_var
            + 2.0 * m.d[2].binary_indicator_var
            + 3.0 * m.d[3].binary_indicator_var
            >= m.x,
        )
        cons = bt.get_transformed_constraints(m.x, m.disjunction['neg'])
        self.assertEqual(len(cons), 1)
        lb = cons[0]
        assertExpressionsEqual(
            self,
            lb.expr,
            -5.0 * m.d[4].binary_indicator_var - 8.0 * m.d[5].binary_indicator_var
            <= m.x,
        )

        self.assertEqual(
            len(
                list(
                    m.component_data_objects(
                        Constraint, descend_into=(Block, Disjunct), active=True
                    )
                )
            ),
            2,
        )

    def test_nested_target(self):
        m = self.create_nested_model()

        bt = TransformationFactory('gdp.bound_pretransformation')
        bt.apply_to(m, targets=[m.outer_d1.inner])

        cons = bt.get_transformed_constraints(m.x, m.outer_d1.inner)
        self.assertEqual(len(cons), 2)
        lb = cons[0]
        ub = cons[1]
        assertExpressionsEqual(
            self,
            lb.expr,
            -100 * m.outer_d1.inner_d1.binary_indicator_var
            - 7.0 * m.outer_d1.inner_d2.binary_indicator_var
            <= m.x,
        )
        self.assertIs(lb.parent_block().parent_block(), m.outer_d1)
        assertExpressionsEqual(
            self,
            ub.expr,
            3.0 * m.outer_d1.inner_d1.binary_indicator_var
            + 102 * m.outer_d1.inner_d2.binary_indicator_var
            >= m.x,
        )
        self.assertIs(ub.parent_block().parent_block(), m.outer_d1)

        self.assertTrue(m.outer_d1.c.active)
        self.assertTrue(m.outer_d2.c.active)
        self.assertTrue(lb.active)
        self.assertTrue(ub.active)

        self.assertEqual(
            len(
                list(
                    m.component_data_objects(
                        Constraint, active=True, descend_into=(Block, Disjunct)
                    )
                )
            ),
            4,
        )

    def test_targets_nested_in_each_other(self):
        m = self.create_nested_model()

        bt = TransformationFactory('gdp.bound_pretransformation')
        bt.apply_to(m, targets=[m.outer_d1.inner, m.outer])

        # This should do the outermost disjunctions only--we should
        # get the same result as if we had transformed the whole
        # model.

        self.check_nested_model_disjunction(m, bt)
        # There aren't any other constraints on the model other than what we
        # added
        self.assertEqual(
            len(
                list(
                    m.component_data_objects(
                        Constraint, active=True, descend_into=(Block, Disjunct)
                    )
                )
            ),
            4,
        )

    def test_variables_not_in_any_leaves(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.disjunct1 = Disjunct()
        m.disjunct1.c = Constraint(expr=m.x <= 9.7)
        m.disjunct1.disjunct1 = Disjunct()
        m.disjunct1.disjunct1.c = Constraint(expr=m.x + m.y <= 4)
        m.disjunct1.disjunct2 = Disjunct()
        m.disjunct1.disjunct2.c = Constraint(expr=m.y <= 9)
        m.disjunct1.disjunction = Disjunction(
            expr=[m.disjunct1.disjunct1, m.disjunct1.disjunct2]
        )
        m.disjunct2 = Disjunct()
        m.disjunct2.c = Constraint(expr=m.x <= 9)
        m.disjunction = Disjunction(expr=[m.disjunct1, m.disjunct2])

        bt = TransformationFactory('gdp.bound_pretransformation')
        bt.apply_to(m)

        cons = bt.get_transformed_constraints(m.x, m.disjunction)
        self.assertEqual(len(cons), 1)
        ub = cons[0]
        assertExpressionsEqual(
            self,
            ub.expr,
            9.7 * m.disjunct1.binary_indicator_var
            + 9.0 * m.disjunct2.binary_indicator_var
            >= m.x,
        )

        cons = bt.get_transformed_constraints(m.y, m.disjunction)
        self.assertEqual(len(cons), 0)

        self.assertFalse(m.disjunct1.c.active)
        self.assertFalse(m.disjunct2.c.active)

        # two leftover and the one we added
        self.assertEqual(
            len(
                list(
                    m.component_data_objects(
                        Constraint, active=True, descend_into=(Block, Disjunct)
                    )
                )
            ),
            3,
        )

    def test_fixed_vars_handled_correctly(self):
        m = ConcreteModel()
        m.x = Var()
        m.x.setub(78)
        m.y = Var()
        m.y.fix(1)
        m.z = Var()

        m.disjunction = Disjunction(expr=[[m.x + m.y <= 5], [m.x <= 17], [m.z == 0]])

        bt = TransformationFactory('gdp.bound_pretransformation')
        bt.apply_to(m)

        cons = bt.get_transformed_constraints(m.x, m.disjunction)
        self.assertEqual(len(cons), 1)
        ub = cons[0]
        assertExpressionsEqual(
            self,
            ub.expr,
            4.0 * m.disjunction.disjuncts[0].binary_indicator_var
            + 17.0 * m.disjunction.disjuncts[1].binary_indicator_var
            + 78 * m.disjunction.disjuncts[2].binary_indicator_var
            >= m.x,
        )

        self.assertFalse(m.disjunction.disjuncts[0].constraint[1].active)
        self.assertFalse(m.disjunction.disjuncts[1].constraint[1].active)
        # Only have the one we added and the one on z.
        self.assertEqual(
            len(
                list(
                    m.component_data_objects(
                        Constraint, active=True, descend_into=(Block, Disjunct)
                    )
                )
            ),
            2,
        )

    def test_bounds_flipped_for_negative_coefs(self):
        m = ConcreteModel()
        m.Time = Set(initialize=range(4))
        m.makespan = Var(bounds=(0, 4))
        m.act_time = Var(m.Time, domain=Binary)
        m.cost = Var(bounds=(1, 6))

        @m.Disjunct()
        def d1(d):
            d.c = Constraint(expr=m.cost == 6)

        @m.Disjunct()
        def d2(d):
            d.c = Constraint(expr=m.cost == 5)
            d.do_act = Constraint(expr=sum(m.act_time[t] for t in m.Time) == 1)

            @d.Constraint(m.Time)
            def ms(d, t):
                return t * m.act_time[t] + 2 <= m.makespan

        m.disjunction = Disjunction(expr=[m.d1, m.d2])

        m.obj = Objective(expr=m.cost)

        m.act_time.fix(0)
        m.act_time[2].unfix()

        bt = TransformationFactory('gdp.bound_pretransformation')
        bt.apply_to(m)

        cons = bt.get_transformed_constraints(m.act_time[2], m.disjunction)
        self.assertEqual(len(cons), 2)

        lb = cons[0]
        assertExpressionsEqual(
            self,
            lb.expr,
            0 * m.disjunction.disjuncts[0].binary_indicator_var
            + m.disjunction.disjuncts[1].binary_indicator_var
            <= m.act_time[2],
        )
        ub = cons[1]
        assertExpressionsEqual(
            self,
            ub.expr,
            m.disjunction.disjuncts[0].binary_indicator_var
            + m.disjunction.disjuncts[1].binary_indicator_var
            >= m.act_time[2],
        )

        cons = bt.get_transformed_constraints(m.act_time[0], m.disjunction)
        self.assertEqual(len(cons), 0)
        cons = bt.get_transformed_constraints(m.act_time[1], m.disjunction)
        self.assertEqual(len(cons), 0)
        cons = bt.get_transformed_constraints(m.act_time[3], m.disjunction)
        self.assertEqual(len(cons), 0)

        cons = bt.get_transformed_constraints(m.makespan, m.disjunction)
        self.assertEqual(len(cons), 2)
        lb = cons[0]
        assertExpressionsEqual(
            self,
            lb.expr,
            0 * m.disjunction.disjuncts[0].binary_indicator_var
            + 2.0 * m.disjunction.disjuncts[1].binary_indicator_var
            <= m.makespan,
        )
        ub = cons[1]
        assertExpressionsEqual(
            self,
            ub.expr,
            4 * m.disjunction.disjuncts[0].binary_indicator_var
            + 4 * m.disjunction.disjuncts[1].binary_indicator_var
            >= m.makespan,
        )

        cons = bt.get_transformed_constraints(m.cost, m.disjunction)
        self.assertEqual(len(cons), 2)
        lb = cons[0]
        assertExpressionsEqual(
            self,
            lb.expr,
            6.0 * m.disjunction.disjuncts[0].binary_indicator_var
            + 5.0 * m.disjunction.disjuncts[1].binary_indicator_var
            <= m.cost,
        )
        ub = cons[1]
        assertExpressionsEqual(
            self,
            ub.expr,
            6 * m.disjunction.disjuncts[0].binary_indicator_var
            + 5.0 * m.disjunction.disjuncts[1].binary_indicator_var
            >= m.cost,
        )

    def test_bounds_correct_for_multiple_nested_disjunctions(self):
        m = ConcreteModel()
        m.c = Var(bounds=(3, 9))
        m.x = Var(bounds=(0, 10))
        m.y = Var(bounds=(-10, 2))

        m.d1 = Disjunct()
        m.d1.cons = Constraint(expr=m.c == 4)
        m.d1.disjunction = Disjunction(expr=[[m.x + m.y >= 8], [m.x + m.y <= 3]])
        m.d1.disjunction2 = Disjunction(
            expr=[[m.x + 2 * m.y <= 4], [m.y + 2 * m.x >= 7]]
        )

        m.d2 = Disjunct()
        m.d2.cons = Constraint(expr=m.c == 5)
        m.d2.disjunction = Disjunction(expr=[[m.x + m.y >= 10], [m.x + m.y <= 0]])
        m.d2.disjunction2 = Disjunction(
            expr=[[m.x + 3 * m.y <= 2], [m.y + 2 * m.x >= 9]]
        )
        m.disjunction = Disjunction(expr=[m.d1, m.d2])

        # will be 4 if you solve
        m.obj = Objective(expr=m.c)

        bt = TransformationFactory('gdp.bound_pretransformation')
        bt.apply_to(m)

        cons = bt.get_transformed_constraints(m.x, m.disjunction)
        self.assertEqual(len(cons), 0)
        cons = bt.get_transformed_constraints(m.y, m.disjunction)
        self.assertEqual(len(cons), 0)
        cons = bt.get_transformed_constraints(m.c, m.disjunction)
        self.assertEqual(len(cons), 2)
        lb = cons[0]
        assertExpressionsEqual(
            self,
            lb.expr,
            4.0 * m.d1.binary_indicator_var + 5.0 * m.d2.binary_indicator_var <= m.c,
        )
        ub = cons[1]
        assertExpressionsEqual(
            self,
            ub.expr,
            4.0 * m.d1.binary_indicator_var + 5.0 * m.d2.binary_indicator_var >= m.c,
        )

        cons = bt.get_transformed_constraints(m.x, m.d1.disjunction)
        self.assertEqual(len(cons), 0)
        cons = bt.get_transformed_constraints(m.y, m.d1.disjunction)
        self.assertEqual(len(cons), 0)
        cons = bt.get_transformed_constraints(m.c, m.d1.disjunction)
        self.assertEqual(len(cons), 0)

        cons = bt.get_transformed_constraints(m.x, m.d1.disjunction2)
        self.assertEqual(len(cons), 0)
        cons = bt.get_transformed_constraints(m.y, m.d1.disjunction2)
        self.assertEqual(len(cons), 0)
        cons = bt.get_transformed_constraints(m.c, m.d1.disjunction2)
        self.assertEqual(len(cons), 0)
