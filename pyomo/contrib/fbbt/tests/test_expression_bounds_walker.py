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

import math
from pyomo.environ import exp, log, log10, sin, cos, tan, asin, acos, atan, sqrt
import pyomo.common.unittest as unittest
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor
from pyomo.core import Any, ConcreteModel, Expression, Param, Var


class TestExpressionBoundsWalker(unittest.TestCase):
    def make_model(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-2, 4))
        m.y = Var(bounds=(3, 5))
        m.z = Var(bounds=(0.5, 0.75))
        return m

    def test_sum_bounds(self):
        m = self.make_model()
        visitor = ExpressionBoundsVisitor()
        lb, ub = visitor.walk_expression(m.x + m.y)
        self.assertEqual(lb, 1)
        self.assertEqual(ub, 9)

        self.assertEqual(len(visitor.leaf_bounds), 2)
        self.assertIn(m.x, visitor.leaf_bounds)
        self.assertIn(m.y, visitor.leaf_bounds)
        self.assertEqual(visitor.leaf_bounds[m.x], (-2, 4))
        self.assertEqual(visitor.leaf_bounds[m.y], (3, 5))

    def test_fixed_var(self):
        m = self.make_model()
        m.x.fix(3)

        visitor = ExpressionBoundsVisitor()
        lb, ub = visitor.walk_expression(m.x + m.y)
        self.assertEqual(lb, 1)
        self.assertEqual(ub, 9)

        self.assertEqual(len(visitor.leaf_bounds), 2)
        self.assertIn(m.x, visitor.leaf_bounds)
        self.assertIn(m.y, visitor.leaf_bounds)
        self.assertEqual(visitor.leaf_bounds[m.x], (-2, 4))
        self.assertEqual(visitor.leaf_bounds[m.y], (3, 5))

    def test_fixed_var_value_used_for_bounds(self):
        m = self.make_model()
        m.x.fix(3)

        visitor = ExpressionBoundsVisitor(use_fixed_var_values_as_bounds=True)
        lb, ub = visitor.walk_expression(m.x + m.y)
        self.assertEqual(lb, 6)
        self.assertEqual(ub, 8)

        self.assertEqual(len(visitor.leaf_bounds), 2)
        self.assertIn(m.x, visitor.leaf_bounds)
        self.assertIn(m.y, visitor.leaf_bounds)
        self.assertEqual(visitor.leaf_bounds[m.x], (3, 3))
        self.assertEqual(visitor.leaf_bounds[m.y], (3, 5))

    def test_product_bounds(self):
        m = self.make_model()
        visitor = ExpressionBoundsVisitor()
        lb, ub = visitor.walk_expression(m.x * m.y)
        self.assertEqual(lb, -10)
        self.assertEqual(ub, 20)

    def test_division_bounds(self):
        m = self.make_model()
        visitor = ExpressionBoundsVisitor()
        lb, ub = visitor.walk_expression(m.x / m.y)
        self.assertAlmostEqual(lb, -2 / 3)
        self.assertAlmostEqual(ub, 4 / 3)

    def test_power_bounds(self):
        m = self.make_model()
        visitor = ExpressionBoundsVisitor()
        lb, ub = visitor.walk_expression(m.y**m.x)
        self.assertEqual(lb, 5 ** (-2))
        self.assertEqual(ub, 5**4)

    def test_sums_of_squares_bounds(self):
        m = ConcreteModel()
        m.x = Var([1, 2], bounds=(-2, 6))
        visitor = ExpressionBoundsVisitor()
        lb, ub = visitor.walk_expression(m.x[1] * m.x[1] + m.x[2] * m.x[2])
        self.assertEqual(lb, 0)
        self.assertEqual(ub, 72)

    def test_negation_bounds(self):
        m = self.make_model()
        visitor = ExpressionBoundsVisitor()
        lb, ub = visitor.walk_expression(-(m.y + 3 * m.x))
        self.assertEqual(lb, -17)
        self.assertEqual(ub, 3)

    def test_exp_bounds(self):
        m = self.make_model()
        visitor = ExpressionBoundsVisitor()
        lb, ub = visitor.walk_expression(exp(m.y))
        self.assertAlmostEqual(lb, math.e**3)
        self.assertAlmostEqual(ub, math.e**5)

    def test_log_bounds(self):
        m = self.make_model()
        visitor = ExpressionBoundsVisitor()
        lb, ub = visitor.walk_expression(log(m.y))
        self.assertAlmostEqual(lb, log(3))
        self.assertAlmostEqual(ub, log(5))

    def test_log10_bounds(self):
        m = self.make_model()
        visitor = ExpressionBoundsVisitor()
        lb, ub = visitor.walk_expression(log10(m.y))
        self.assertAlmostEqual(lb, log10(3))
        self.assertAlmostEqual(ub, log10(5))

    def test_sin_bounds(self):
        m = self.make_model()
        visitor = ExpressionBoundsVisitor()
        lb, ub = visitor.walk_expression(sin(m.y))
        self.assertAlmostEqual(lb, -1)  # reaches -1 at 3*pi/2 \approx 4.712
        self.assertAlmostEqual(ub, sin(3))  # it's positive here

    def test_cos_bounds(self):
        m = self.make_model()
        visitor = ExpressionBoundsVisitor()
        lb, ub = visitor.walk_expression(cos(m.y))
        self.assertAlmostEqual(lb, -1)  # reaches -1 at pi
        self.assertAlmostEqual(ub, cos(5))  # it's positive here

    def test_tan_bounds(self):
        m = self.make_model()
        visitor = ExpressionBoundsVisitor()
        lb, ub = visitor.walk_expression(tan(m.y))
        self.assertEqual(lb, -float('inf'))
        self.assertEqual(ub, float('inf'))

    def test_asin_bounds(self):
        m = self.make_model()
        visitor = ExpressionBoundsVisitor()
        lb, ub = visitor.walk_expression(asin(m.z))
        self.assertAlmostEqual(lb, asin(0.5))
        self.assertAlmostEqual(ub, asin(0.75))

    def test_acos_bounds(self):
        m = self.make_model()
        visitor = ExpressionBoundsVisitor()
        lb, ub = visitor.walk_expression(acos(m.z))
        self.assertAlmostEqual(lb, acos(0.75))
        self.assertAlmostEqual(ub, acos(0.5))

    def test_atan_bounds(self):
        m = self.make_model()
        visitor = ExpressionBoundsVisitor()
        lb, ub = visitor.walk_expression(atan(m.z))
        self.assertAlmostEqual(lb, atan(0.5))
        self.assertAlmostEqual(ub, atan(0.75))

    def test_sqrt_bounds(self):
        m = self.make_model()
        visitor = ExpressionBoundsVisitor()
        lb, ub = visitor.walk_expression(sqrt(m.y))
        self.assertAlmostEqual(lb, sqrt(3))
        self.assertAlmostEqual(ub, sqrt(5))

    def test_abs_bounds(self):
        m = self.make_model()
        visitor = ExpressionBoundsVisitor()
        lb, ub = visitor.walk_expression(abs(m.x))
        self.assertEqual(lb, 0)
        self.assertEqual(ub, 4)

    def test_leaf_bounds_cached(self):
        m = self.make_model()
        visitor = ExpressionBoundsVisitor()
        lb, ub = visitor.walk_expression(m.x - m.y)
        self.assertEqual(lb, -7)
        self.assertEqual(ub, 1)

        self.assertIn(m.x, visitor.leaf_bounds)
        self.assertEqual(visitor.leaf_bounds[m.x], m.x.bounds)
        self.assertIn(m.y, visitor.leaf_bounds)
        self.assertEqual(visitor.leaf_bounds[m.y], m.y.bounds)

        # This should exercise the code that uses the cache.
        lb, ub = visitor.walk_expression(m.x**2 + 3)
        self.assertEqual(lb, 3)
        self.assertEqual(ub, 19)

    def test_var_fixed_to_None(self):
        m = self.make_model()
        m.x.fix(None)

        visitor = ExpressionBoundsVisitor(use_fixed_var_values_as_bounds=True)
        with self.assertRaisesRegex(
            ValueError,
            "Var 'x' is fixed to None. This value cannot be "
            "used to calculate bounds.",
        ):
            lb, ub = visitor.walk_expression(m.x - m.y)

    def test_var_with_no_lb(self):
        m = self.make_model()
        m.x.setlb(None)

        visitor = ExpressionBoundsVisitor()
        lb, ub = visitor.walk_expression(m.x - m.y)
        self.assertEqual(lb, -float('inf'))
        self.assertEqual(ub, 1)

    def test_var_with_no_ub(self):
        m = self.make_model()
        m.y.setub(None)

        visitor = ExpressionBoundsVisitor()
        lb, ub = visitor.walk_expression(m.x - m.y)
        self.assertEqual(lb, -float('inf'))
        self.assertEqual(ub, 1)

    def test_param(self):
        m = self.make_model()
        m.p = Param(initialize=6)

        visitor = ExpressionBoundsVisitor()
        lb, ub = visitor.walk_expression(m.p**m.y)
        self.assertEqual(lb, 6**3)
        self.assertEqual(ub, 6**5)

    def test_mutable_param(self):
        m = self.make_model()
        m.p = Param(initialize=6, mutable=True)

        visitor = ExpressionBoundsVisitor()
        lb, ub = visitor.walk_expression(m.p**m.y)
        self.assertEqual(lb, 6**3)
        self.assertEqual(ub, 6**5)

    def test_named_expression(self):
        m = self.make_model()
        m.e = Expression(expr=sqrt(m.x**2 + m.y**2))
        visitor = ExpressionBoundsVisitor()

        lb, ub = visitor.walk_expression(m.e + 4)
        self.assertEqual(lb, 7)
        self.assertAlmostEqual(ub, sqrt(41) + 4)

        self.assertIn(m.e, visitor.leaf_bounds)
        self.assertEqual(visitor.leaf_bounds[m.e][0], 3)
        self.assertAlmostEqual(visitor.leaf_bounds[m.e][1], sqrt(41))

        # exercise the using of the cached bounds
        lb, ub = visitor.walk_expression(m.e)
        self.assertEqual(lb, 3)
        self.assertAlmostEqual(ub, sqrt(41))

    def test_npv_expression(self):
        m = self.make_model()
        m.p = Param(initialize=4, mutable=True)
        visitor = ExpressionBoundsVisitor()
        lb, ub = visitor.walk_expression(1 / m.p)
        self.assertEqual(lb, 0.25)
        self.assertEqual(ub, 0.25)

    def test_invalid_numeric_type(self):
        m = self.make_model()
        m.p = Param(initialize=True, domain=Any)
        visitor = ExpressionBoundsVisitor()
        with self.assertRaisesRegex(
            ValueError,
            r"True \(<class 'bool'>\) is not a valid numeric type. "
            r"Cannot compute bounds on expression.",
        ):
            lb, ub = visitor.walk_expression(m.p + m.y)

    def test_invalid_string(self):
        m = self.make_model()
        m.p = Param(initialize='True', domain=Any)
        visitor = ExpressionBoundsVisitor()
        with self.assertRaisesRegex(
            ValueError,
            r"'True' \(<class 'str'>\) is not a valid numeric type. "
            r"Cannot compute bounds on expression.",
        ):
            lb, ub = visitor.walk_expression(m.p + m.y)

    def test_invalid_complex(self):
        m = self.make_model()
        m.p = Param(initialize=complex(4, 5), domain=Any)
        visitor = ExpressionBoundsVisitor()
        with self.assertRaisesRegex(
            ValueError,
            r"Cannot compute bounds on expressions containing "
            r"complex numbers. Encountered when processing \(4\+5j\)",
        ):
            lb, ub = visitor.walk_expression(m.p + m.y)
