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
#
# Unit Tests for Elements of a Model
#
# TestSimpleCon                Class for testing single constraint
# TestArrayCon                Class for testing array of constraint
#

import sys
import os
from os.path import abspath, dirname

currdir = dirname(abspath(__file__)) + os.sep

import pyomo.common.unittest as unittest

from pyomo.environ import (
    ConcreteModel,
    AbstractModel,
    Var,
    Constraint,
    ConstraintList,
    Param,
    RangeSet,
    Set,
    Expression,
    value,
    simple_constraintlist_rule,
    simple_constraint_rule,
    inequality,
)
from pyomo.core.expr import (
    SumExpression,
    EqualityExpression,
    InequalityExpression,
    RangedExpression,
)
from pyomo.core.base.constraint import _GeneralConstraintData


class TestConstraintCreation(unittest.TestCase):
    def create_model(self, abstract=False):
        if abstract is True:
            model = AbstractModel()
        else:
            model = ConcreteModel()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        return model

    def test_tuple_construct_equality(self):
        model = self.create_model()

        def rule(model):
            return (0.0, model.x)

        model.c = Constraint(rule=rule)

        self.assertEqual(model.c.equality, True)
        self.assertEqual(model.c.lower, 0)
        self.assertIs(model.c.body, model.x)
        self.assertEqual(model.c.upper, 0)

        model = self.create_model()

        def rule(model):
            return (model.x, 0.0)

        model.c = Constraint(rule=rule)

        self.assertEqual(model.c.equality, True)
        self.assertEqual(model.c.lower, 0)
        self.assertIs(model.c.body, model.x)
        self.assertEqual(model.c.upper, 0)

    def test_tuple_construct_inf_equality(self):
        model = self.create_model(abstract=True)

        def rule(model):
            return (model.x, float('inf'))

        model.c = Constraint(rule=rule)
        self.assertRaises(ValueError, model.create_instance)

        model = self.create_model(abstract=True)

        def rule(model):
            return (float('inf'), model.x)

        model.c = Constraint(rule=rule)
        self.assertRaises(ValueError, model.create_instance)

    def test_tuple_construct_1sided_inequality(self):
        model = self.create_model()

        def rule(model):
            return (None, model.y, 1)

        model.c = Constraint(rule=rule)

        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, None)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, 1)

        model = self.create_model()

        def rule(model):
            return (0, model.y, None)

        model.c = Constraint(rule=rule)

        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, 0)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, None)

    def test_tuple_construct_1sided_inf_inequality(self):
        model = self.create_model()

        def rule(model):
            return (float('-inf'), model.y, 1)

        model.c = Constraint(rule=rule)

        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, None)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, 1)

        model = self.create_model()

        def rule(model):
            return (0, model.y, float('inf'))

        model.c = Constraint(rule=rule)

        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, 0)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, None)

    def test_tuple_construct_unbounded_inequality(self):
        model = self.create_model()

        def rule(model):
            return (None, model.y, None)

        model.c = Constraint(rule=rule)

        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, None)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, None)

        model = self.create_model()

        def rule(model):
            return (float('-inf'), model.y, float('inf'))

        model.c = Constraint(rule=rule)

        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, None)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, None)

    def test_tuple_construct_invalid_1sided_inequality(self):
        model = self.create_model(abstract=True)

        def rule(model):
            return (model.x, model.y, None)

        model.c = Constraint(rule=rule)
        # We now recognize this as a valid inequality
        # self.assertRaises(ValueError, model.create_instance)
        instance = model.create_instance()
        self.assertEqual(instance.c.lower, None)
        self.assertIsInstance(instance.c.body, SumExpression)
        self.assertEqual(instance.c.upper, 0)

        model = self.create_model(abstract=True)

        def rule(model):
            return (None, model.y, model.z)

        model.c = Constraint(rule=rule)
        # We now recognize this as a valid inequality
        # self.assertRaises(ValueError, model.create_instance)
        instance = model.create_instance()
        self.assertEqual(instance.c.lower, None)
        self.assertIsInstance(instance.c.body, SumExpression)
        self.assertEqual(instance.c.upper, 0)

    def test_tuple_construct_2sided_inequality(self):
        model = self.create_model()

        def rule(model):
            return (0, model.y, 1)

        model.c = Constraint(rule=rule)

        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, 0)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, 1)

    def test_tuple_construct_invalid_2sided_inequality(self):
        model = self.create_model(abstract=True)

        def rule(model):
            return (model.x, model.y, 1)

        model.c = Constraint(rule=rule)
        instance = model.create_instance()
        with self.assertRaisesRegex(
            ValueError,
            "Constraint 'c' is a Ranged Inequality with a variable lower bound",
        ):
            instance.c.lower
        self.assertIs(instance.c.body, instance.y)
        self.assertEqual(instance.c.upper, 1)
        instance.x.fix(3)
        self.assertEqual(value(instance.c.lower), 3)

        model = self.create_model(abstract=True)

        def rule(model):
            return (0, model.y, model.z)

        model.c = Constraint(rule=rule)
        instance = model.create_instance()
        self.assertEqual(instance.c.lower, 0)
        self.assertIs(instance.c.body, instance.y)
        with self.assertRaisesRegex(
            ValueError,
            "Constraint 'c' is a Ranged Inequality with a variable upper bound",
        ):
            instance.c.upper
        instance.z.fix(3)
        self.assertEqual(value(instance.c.upper), 3)

    def test_expr_construct_equality(self):
        model = self.create_model()

        def rule(model):
            return 0.0 == model.x

        model.c = Constraint(rule=rule)

        self.assertEqual(model.c.equality, True)
        self.assertEqual(model.c.lower, 0)
        self.assertIs(model.c.body, model.x)
        self.assertEqual(model.c.upper, 0)

        model = self.create_model()

        def rule(model):
            return model.x == 0.0

        model.c = Constraint(rule=rule)

        self.assertEqual(model.c.equality, True)
        self.assertEqual(model.c.lower, 0)
        self.assertIs(model.c.body, model.x)
        self.assertEqual(model.c.upper, 0)

    def test_expr_construct_inf_equality(self):
        model = self.create_model(abstract=True)

        def rule(model):
            return model.x == float('inf')

        model.c = Constraint(rule=rule)
        self.assertRaises(ValueError, model.create_instance)

        model = self.create_model(abstract=True)

        def rule(model):
            return float('inf') == model.x

        model.c = Constraint(rule=rule)
        self.assertRaises(ValueError, model.create_instance)

    def test_expr_construct_1sided_inequality(self):
        model = self.create_model()

        def rule(model):
            return model.y <= 1

        model.c = Constraint(rule=rule)

        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, None)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, 1)

        model = self.create_model()

        def rule(model):
            return 0 <= model.y

        model.c = Constraint(rule=rule)

        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, 0)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, None)

        model = self.create_model()

        def rule(model):
            return model.y >= 1

        model.c = Constraint(rule=rule)

        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, 1)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, None)

        model = self.create_model()

        def rule(model):
            return 0 >= model.y

        model.c = Constraint(rule=rule)

        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, None)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, 0)

    def test_expr_construct_unbounded_inequality(self):
        model = self.create_model()

        def rule(model):
            return model.y <= float('inf')

        model.c = Constraint(rule=rule)

        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, None)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, None)

        model = self.create_model()

        def rule(model):
            return float('-inf') <= model.y

        model.c = Constraint(rule=rule)

        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, None)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, None)

        model = self.create_model()

        def rule(model):
            return model.y >= float('-inf')

        model.c = Constraint(rule=rule)

        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, None)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, None)

        model = self.create_model()

        def rule(model):
            return float('inf') >= model.y

        model.c = Constraint(rule=rule)

        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, None)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, None)

    def test_expr_construct_invalid_unbounded_inequality(self):
        model = self.create_model(abstract=True)

        def rule(model):
            return model.y <= float('-inf')

        model.c = Constraint(rule=rule)
        self.assertRaises(ValueError, model.create_instance)

        model = self.create_model(abstract=True)

        def rule(model):
            return float('inf') <= model.y

        model.c = Constraint(rule=rule)
        self.assertRaises(ValueError, model.create_instance)

        model = self.create_model(abstract=True)

        def rule(model):
            return model.y >= float('inf')

        model.c = Constraint(rule=rule)
        self.assertRaises(ValueError, model.create_instance)

        model = self.create_model(abstract=True)

        def rule(model):
            return float('-inf') >= model.y

        model.c = Constraint(rule=rule)
        self.assertRaises(ValueError, model.create_instance)

    def test_expr_construct_invalid(self):
        m = ConcreteModel()
        c = Constraint(rule=lambda m: None)
        self.assertRaisesRegex(
            ValueError, ".*rule returned None", m.add_component, 'c', c
        )

        m = ConcreteModel()
        c = Constraint([1], rule=lambda m, i: None)
        self.assertRaisesRegex(
            ValueError, ".*rule returned None", m.add_component, 'c', c
        )

        m = ConcreteModel()
        c = Constraint(rule=lambda m: True)
        self.assertRaisesRegex(
            ValueError,
            r".*resolved to a trivial Boolean \(True\).*Constraint\.Feasible",
            m.add_component,
            'c',
            c,
        )

        m = ConcreteModel()
        c = Constraint([1], rule=lambda m, i: True)
        self.assertRaisesRegex(
            ValueError,
            r".*resolved to a trivial Boolean \(True\).*Constraint\.Feasible",
            m.add_component,
            'c',
            c,
        )

        m = ConcreteModel()
        c = Constraint(rule=lambda m: False)
        self.assertRaisesRegex(
            ValueError,
            r".*resolved to a trivial Boolean \(False\).*" r"Constraint\.Infeasible",
            m.add_component,
            'c',
            c,
        )

        m = ConcreteModel()
        c = Constraint([1], rule=lambda m, i: False)
        self.assertRaisesRegex(
            ValueError,
            r".*resolved to a trivial Boolean \(False\).*" r"Constraint\.Infeasible",
            m.add_component,
            'c',
            c,
        )

    def test_nondata_bounds(self):
        model = ConcreteModel()
        model.c = Constraint()
        model.v = Var([1, 2, 3])
        model.e1 = Expression()
        model.e2 = Expression()
        model.e3 = Expression()
        model.c.set_value((model.e1, model.e2, model.e3))
        self.assertIsNone(model.c._lower)
        self.assertIsNone(model.c._body)
        self.assertIsNone(model.c._upper)
        self.assertIs(model.c.lower, model.e1)
        self.assertIs(model.c.body, model.e2)
        self.assertIs(model.c.upper, model.e3)
        model.e1.expr = 1
        model.e2.expr = 2
        model.e3.expr = 3
        self.assertEqual(value(model.c.lower), 1)
        self.assertEqual(value(model.c.body), 2)
        self.assertEqual(value(model.c.upper), 3)
        model.e1 = model.v[1]
        model.e2 = model.v[2]
        model.e3 = model.v[3]
        with self.assertRaisesRegex(
            ValueError,
            "Constraint 'c' is a Ranged Inequality with a variable lower bound",
        ):
            model.c.lower
        self.assertIs(model.c.body.expr, model.v[2])
        with self.assertRaisesRegex(
            ValueError,
            "Constraint 'c' is a Ranged Inequality with a variable upper bound",
        ):
            model.c.upper

    # make sure we can use a mutable param that
    # has not been given a value in the upper bound
    # of an inequality constraint
    def test_mutable_novalue_param_lower_bound(self):
        model = ConcreteModel()
        model.x = Var()
        model.p = Param(mutable=True)
        model.p.value = None

        model.c = Constraint(expr=0 <= model.x - model.p)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=model.p <= model.x)
        self.assertTrue(model.c.lower is model.p)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=model.p <= model.x + 1)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=model.p + 1 <= model.x)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=(model.p + 1) ** 2 <= model.x)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=(model.p, model.x, model.p + 1))
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=model.x - model.p >= 0)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=model.x >= model.p)
        self.assertTrue(model.c.lower is model.p)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=model.x + 1 >= model.p)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=model.x >= model.p + 1)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=model.x >= (model.p + 1) ** 2)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=(model.p, model.x, None))
        self.assertTrue(model.c.lower is model.p)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=(model.p, model.x + 1, None))
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=(model.p + 1, model.x, None))
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=(model.p, model.x, 1))
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

    # make sure we can use a mutable param that
    # has not been given a value in the lower bound
    # of an inequality constraint
    def test_mutable_novalue_param_upper_bound(self):
        model = ConcreteModel()
        model.x = Var()
        model.p = Param(mutable=True)
        model.p.value = None

        model.c = Constraint(expr=model.x - model.p <= 0)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=model.x <= model.p)
        self.assertTrue(model.c.upper is model.p)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=model.x + 1 <= model.p)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=model.x <= model.p + 1)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=model.x <= (model.p + 1) ** 2)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=(model.p + 1, model.x, model.p))
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=0 >= model.x - model.p)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=model.p >= model.x)
        self.assertTrue(model.c.upper is model.p)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=model.p >= model.x + 1)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=model.p + 1 >= model.x)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=(model.p + 1) ** 2 >= model.x)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=(None, model.x, model.p))
        self.assertTrue(model.c.upper is model.p)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=(None, model.x + 1, model.p))
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=(None, model.x, model.p + 1))
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

        model.c = Constraint(expr=(1, model.x, model.p))
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

    # make sure we can use a mutable param that
    # has not been given a value in the rhs of
    # of an equality constraint
    def test_mutable_novalue_param_equality(self):
        model = ConcreteModel()
        model.x = Var()
        model.p = Param(mutable=True)
        model.p.value = None

        model.c = Constraint(expr=model.x - model.p == 0)
        self.assertEqual(model.c.equality, True)
        model.del_component(model.c)

        model.c = Constraint(expr=model.x == model.p)
        self.assertTrue(model.c.upper is model.p)
        self.assertEqual(model.c.equality, True)
        model.del_component(model.c)

        model.c = Constraint(expr=model.x + 1 == model.p)
        self.assertEqual(model.c.equality, True)
        model.del_component(model.c)

        model.c = Constraint(expr=model.x + 1 == (model.p + 1) ** 2)
        self.assertEqual(model.c.equality, True)
        model.del_component(model.c)

        model.c = Constraint(expr=model.x == model.p + 1)
        self.assertEqual(model.c.equality, True)
        model.del_component(model.c)

        model.c = Constraint(expr=inequality(model.p, model.x, model.p))
        self.assertTrue(model.c.upper is model.p)
        # GH: Not sure if we are supposed to detect equality
        #     in this situation. I would rather us not, for
        #     the sake of making the code less complicated.
        #     Either way, I am not going to test for it here.
        # self.assertEqual(model.c.equality, <blah>)
        model.del_component(model.c)

        model.c = Constraint(expr=(model.x, model.p))
        self.assertTrue(model.c.upper is model.p)
        self.assertEqual(model.c.equality, True)
        model.del_component(model.c)

        model.c = Constraint(expr=(model.p, model.x))
        self.assertTrue(model.c.upper is model.p)
        self.assertEqual(model.c.equality, True)
        model.del_component(model.c)

    def test_inequality(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(expr=inequality(lower=-1, body=m.x))
        self.assertEqual(m.c.lower.value, -1)
        self.assertIs(m.c.body, m.x)
        self.assertIs(m.c.upper, None)

        del m.c
        m.c = Constraint(expr=inequality(body=m.x, upper=1))
        self.assertIs(m.c.lower, None)
        self.assertIs(m.c.body, m.x)
        self.assertEqual(m.c.upper.value, 1)

        del m.c
        m.c = Constraint(expr=inequality(lower=-1, body=m.x, upper=1))
        self.assertEqual(m.c.lower.value, -1)
        self.assertIs(m.c.body, m.x)
        self.assertEqual(m.c.upper.value, 1)


class TestSimpleCon(unittest.TestCase):
    def test_set_expr_explicit_multivariate(self):
        """Test expr= option (multivariate expression)"""
        model = ConcreteModel()
        model.A = RangeSet(1, 4)
        model.x = Var(model.A, initialize=2)
        ans = 0
        for i in model.A:
            ans = ans + model.x[i]
        ans = ans >= 0
        ans = ans <= 1
        model.c = Constraint(expr=ans)

        self.assertEqual(model.c(), 8)
        self.assertEqual(model.c.body(), 8)
        self.assertEqual(value(model.c.body), 8)

    def test_set_expr_explicit_univariate(self):
        """Test expr= option (univariate expression)"""
        model = ConcreteModel()
        model.x = Var(initialize=2)
        ans = model.x >= 0
        ans = ans <= 1
        model.c = Constraint(expr=ans)

        self.assertEqual(model.c(), 2)
        self.assertEqual(model.c.body(), 2)
        self.assertEqual(value(model.c.body), 2)

    def test_set_expr_undefined_univariate(self):
        """Test expr= option (univariate expression)"""
        model = ConcreteModel()
        model.x = Var()
        ans = model.x >= 0
        ans = ans <= 1
        model.c = Constraint(expr=ans)

        with self.assertRaisesRegex(
            ValueError, "No value for uninitialized NumericValue object x"
        ):
            value(model.c)
        model.x = 2
        self.assertEqual(model.c(), 2)
        self.assertEqual(value(model.c.body), 2)

    def test_set_expr_inline(self):
        """Test expr= option (inline expression)"""
        model = ConcreteModel()
        model.A = RangeSet(1, 4)
        model.x = Var(model.A, initialize=2)
        model.c = Constraint(expr=(0, sum(model.x[i] for i in model.A), 1))

        self.assertEqual(model.c(), 8)
        self.assertEqual(value(model.c.body), 8)

    def test_rule1(self):
        """Test rule option"""
        model = ConcreteModel()
        model.B = RangeSet(1, 4)

        def f(model):
            ans = 0
            for i in model.B:
                ans = ans + model.x[i]
            ans = ans >= 0
            ans = ans <= 1
            return ans

        model.x = Var(model.B, initialize=2)
        model.c = Constraint(rule=f)

        self.assertEqual(model.c(), 8)
        self.assertEqual(value(model.c.body), 8)

    def test_rule2(self):
        """Test rule option"""
        model = ConcreteModel()
        model.B = RangeSet(1, 4)

        def f(model):
            ans = 0
            for i in model.B:
                ans = ans + model.x[i]
            return (0, ans, 1)

        model.x = Var(model.B, initialize=2)
        model.c = Constraint(rule=f)

        self.assertEqual(model.c(), 8)
        self.assertEqual(value(model.c.body), 8)

    def test_rule3(self):
        """Test rule option"""
        model = ConcreteModel()
        model.B = RangeSet(1, 4)

        def f(model):
            ans = 0
            for i in model.B:
                ans = ans + model.x[i]
            return (0, ans, None)

        model.x = Var(model.B, initialize=2)
        model.c = Constraint(rule=f)

        self.assertEqual(model.c(), 8)
        self.assertEqual(value(model.c.body), 8)

    def test_rule4(self):
        """Test rule option"""
        model = ConcreteModel()
        model.B = RangeSet(1, 4)

        def f(model):
            ans = 0
            for i in model.B:
                ans = ans + model.x[i]
            return (None, ans, 1)

        model.x = Var(model.B, initialize=2)
        model.c = Constraint(rule=f)

        self.assertEqual(model.c(), 8)
        self.assertEqual(value(model.c.body), 8)

    def test_rule5(self):
        """Test rule option"""
        model = ConcreteModel()
        model.B = RangeSet(1, 4)

        def f(model):
            ans = 0
            for i in model.B:
                ans = ans + model.x[i]
            return (ans, 1)

        model.x = Var(model.B, initialize=2)
        model.c = Constraint(rule=f)

        self.assertEqual(model.c(), 8)
        self.assertEqual(value(model.c.body), 8)

    def test_dim(self):
        """Test dim method"""
        model = ConcreteModel()
        model.c = Constraint()

        self.assertEqual(model.c.dim(), 0)

    def test_keys_empty(self):
        """Test keys method"""
        model = ConcreteModel()
        model.c = Constraint()

        self.assertEqual(list(model.c.keys()), [])

    def test_len_empty(self):
        """Test len method"""
        model = ConcreteModel()
        model.c = Constraint()

        self.assertEqual(len(model.c), 0)

    def test_None_key(self):
        """Test keys method"""
        model = ConcreteModel()
        model.x = Var()
        model.c = Constraint(expr=model.x == 1)
        self.assertEqual(list(model.c.keys()), [None])
        self.assertEqual(id(model.c), id(model.c[None]))

    def test_len(self):
        """Test len method"""
        model = AbstractModel()
        model.x = Var()
        model.c = Constraint(rule=lambda m: m.x == 1)
        self.assertEqual(len(model.c), 0)
        inst = model.create_instance()
        self.assertEqual(len(inst.c), 1)

    def test_setitem(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint()
        self.assertEqual(len(m.c), 0)

        m.c = m.x**2 <= 4
        self.assertEqual(len(m.c), 1)
        self.assertEqual(list(m.c.keys()), [None])
        self.assertEqual(m.c.upper, 4)

        m.c = Constraint.Skip
        self.assertEqual(len(m.c), 0)


class TestArrayCon(unittest.TestCase):
    def create_model(self):
        model = ConcreteModel()
        model.A = Set(initialize=[1, 2, 3, 4])
        return model

    def test_rule_option1(self):
        model = self.create_model()
        model.B = RangeSet(1, 4)

        def f(model, i):
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            ans = ans <= 0
            ans = ans >= 0
            return ans

        model.x = Var(model.B, initialize=2)
        model.c = Constraint(model.A, rule=f)

        self.assertEqual(model.c[1](), 8)
        self.assertEqual(model.c[2](), 16)
        self.assertEqual(len(model.c), 4)

    def test_rule_option2(self):
        model = self.create_model()
        model.B = RangeSet(1, 4)

        def f(model, i):
            if i % 2 == 0:
                return Constraint.Skip
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            ans = ans <= 0
            ans = ans >= 0
            return ans

        model.x = Var(model.B, initialize=2)
        model.c = Constraint(model.A, rule=f)

        self.assertEqual(model.c[1](), 8)
        self.assertEqual(len(model.c), 2)

    def test_rule_option3(self):
        model = self.create_model()
        model.B = RangeSet(1, 4)

        def f(model, i):
            if i % 2 == 0:
                return Constraint.Skip
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            ans = ans <= 0
            ans = ans >= 0
            return ans

        model.x = Var(model.B, initialize=2)
        model.c = Constraint(model.A, rule=f)

        self.assertEqual(model.c[1](), 8)
        self.assertEqual(len(model.c), 2)

    def test_rule_option2a(self):
        model = self.create_model()
        model.B = RangeSet(1, 4)

        @simple_constraint_rule
        def f(model, i):
            if i % 2 == 0:
                return None
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            ans = ans <= 0
            ans = ans >= 0
            return ans

        model.x = Var(model.B, initialize=2)
        model.c = Constraint(model.A, rule=f)

        self.assertEqual(model.c[1](), 8)
        self.assertEqual(len(model.c), 2)

    def test_rule_option3a(self):
        model = self.create_model()
        model.B = RangeSet(1, 4)

        @simple_constraint_rule
        def f(model, i):
            if i % 2 == 0:
                return None
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            ans = ans <= 0
            ans = ans >= 0
            return ans

        model.x = Var(model.B, initialize=2)
        model.c = Constraint(model.A, rule=f)

        self.assertEqual(model.c[1](), 8)
        self.assertEqual(len(model.c), 2)

    def test_dim(self):
        model = self.create_model()
        model.c = Constraint(model.A)

        self.assertEqual(model.c.dim(), 1)

    def test_keys(self):
        model = self.create_model()
        model.c = Constraint(model.A)

        self.assertEqual(len(list(model.c.keys())), 0)

    def test_len(self):
        model = self.create_model()
        model.c = Constraint(model.A)
        self.assertEqual(len(model.c), 0)

        model = self.create_model()
        model.B = RangeSet(1, 4)
        """Test rule option"""

        def f(model):
            ans = 0
            for i in model.B:
                ans = ans + model.x[i]
            ans = ans == 2
            return ans

        model.x = Var(model.B, initialize=2)
        model.c = Constraint(rule=f)

        self.assertEqual(len(model.c), 1)

    def test_setitem(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(range(5))
        self.assertEqual(len(m.c), 0)

        m.c[2] = m.x**2 <= 4
        self.assertEqual(len(m.c), 1)
        self.assertEqual(list(m.c.keys()), [2])
        self.assertIsInstance(m.c[2], _GeneralConstraintData)
        self.assertEqual(m.c[2].upper, 4)

        m.c[3] = Constraint.Skip
        self.assertEqual(len(m.c), 1)
        self.assertRaisesRegex(KeyError, "3", m.c.__getitem__, 3)

        self.assertRaisesRegex(
            ValueError, r"'c\[3\]': rule returned None", m.c.__setitem__, 3, None
        )
        self.assertEqual(len(m.c), 1)

        m.c[2] = Constraint.Skip
        self.assertEqual(len(m.c), 0)


class TestConList(unittest.TestCase):
    def create_model(self):
        model = ConcreteModel()
        model.A = Set(initialize=[1, 2, 3, 4])
        return model

    #
    # Tests that adding Constraint.Skip increments
    # the internal counter but does not create an object
    #
    def test_conlist_skip(self):
        model = ConcreteModel()
        model.x = Var()
        model.c = ConstraintList()
        self.assertTrue(1 not in model.c)
        self.assertEqual(len(model.c), 0)
        model.c.add(Constraint.Skip)
        self.assertTrue(1 not in model.c)
        self.assertEqual(len(model.c), 0)
        model.c.add(model.x >= 1)
        self.assertTrue(1 not in model.c)
        self.assertTrue(2 in model.c)
        self.assertEqual(len(model.c), 1)

    def test_rule_option1(self):
        model = self.create_model()
        model.B = RangeSet(1, 4)

        def f(model, i):
            if i > 4:
                return ConstraintList.End
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            ans = ans <= 0
            ans = ans >= 0
            return ans

        model.x = Var(model.B, initialize=2)
        model.c = ConstraintList(rule=f)

        self.assertEqual(model.c[1](), 8)
        self.assertEqual(model.c[2](), 16)
        self.assertEqual(len(model.c), 4)

    def test_rule_option2(self):
        model = self.create_model()
        model.B = RangeSet(1, 4)

        def f(model, i):
            if i > 2:
                return ConstraintList.End
            i = 2 * i - 1
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            ans = ans <= 0
            ans = ans >= 0
            return ans

        model.x = Var(model.B, initialize=2)
        model.c = ConstraintList(rule=f)

        self.assertEqual(model.c[1](), 8)
        self.assertEqual(len(model.c), 2)

    def test_rule_option1a(self):
        model = self.create_model()
        model.B = RangeSet(1, 4)

        @simple_constraintlist_rule
        def f(model, i):
            if i > 4:
                return None
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            ans = ans <= 0
            ans = ans >= 0
            return ans

        model.x = Var(model.B, initialize=2)
        model.c = ConstraintList(rule=f)

        self.assertEqual(model.c[1](), 8)
        self.assertEqual(model.c[2](), 16)
        self.assertEqual(len(model.c), 4)

    def test_rule_option2a(self):
        model = self.create_model()
        model.B = RangeSet(1, 4)

        @simple_constraintlist_rule
        def f(model, i):
            if i > 2:
                return None
            i = 2 * i - 1
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            ans = ans <= 0
            ans = ans >= 0
            return ans

        model.x = Var(model.B, initialize=2)
        model.c = ConstraintList(rule=f)

        self.assertEqual(model.c[1](), 8)
        self.assertEqual(len(model.c), 2)

    def test_rule_option3(self):
        model = self.create_model()
        model.y = Var(initialize=2)

        def f(model):
            yield model.y <= 0
            yield 2 * model.y <= 0
            yield 2 * model.y <= 0
            yield ConstraintList.End

        model.c = ConstraintList(rule=f)
        self.assertEqual(len(model.c), 3)
        self.assertEqual(model.c[1](), 2)
        model.d = ConstraintList(rule=f(model))
        self.assertEqual(len(model.d), 3)
        self.assertEqual(model.d[1](), 2)

    def test_rule_option4(self):
        model = self.create_model()
        model.y = Var(initialize=2)
        model.c = ConstraintList(rule=((i + 1) * model.y >= 0 for i in range(3)))
        self.assertEqual(len(model.c), 3)
        self.assertEqual(model.c[1](), 2)

    def test_dim(self):
        model = self.create_model()
        model.c = ConstraintList()

        self.assertEqual(model.c.dim(), 1)

    def test_keys(self):
        model = self.create_model()
        model.c = ConstraintList()

        self.assertEqual(len(list(model.c.keys())), 0)

    def test_len(self):
        model = self.create_model()
        model.c = ConstraintList()

        self.assertEqual(len(model.c), 0)

    def test_0based_add(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = ConstraintList(starting_index=0)
        m.c.add(m.x <= 0)
        self.assertEqual(list(m.c.keys()), [0])
        m.c.add(m.x >= 0)
        self.assertEqual(list(m.c.keys()), [0, 1])


class Test2DArrayCon(unittest.TestCase):
    def create_model(self):
        model = ConcreteModel()
        model.A = Set(initialize=[1, 2])
        return model

    def test_rule_option(self):
        """Test rule option"""
        model = self.create_model()
        model.B = RangeSet(1, 4)

        def f(model, i, j):
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            ans = ans <= 0
            ans = ans >= 0
            return ans

        model.x = Var(model.B, initialize=2)
        model.c = Constraint(model.A, model.A, rule=f)

        self.assertEqual(model.c[1, 1](), 8)
        self.assertEqual(model.c[2, 1](), 16)

    def test_dim(self):
        """Test dim method"""
        model = self.create_model()
        model.c = Constraint(model.A, model.A)

        self.assertEqual(model.c.dim(), 2)

    def test_keys(self):
        """Test keys method"""
        model = self.create_model()
        model.c = Constraint(model.A, model.A)

        self.assertEqual(len(list(model.c.keys())), 0)

    def test_len(self):
        """Test len method"""
        model = self.create_model()
        model.c = Constraint(model.A, model.A)
        self.assertEqual(len(model.c), 0)

        model = self.create_model()
        model.B = RangeSet(1, 4)
        """Test rule option"""

        def f(model):
            ans = 0
            for i in model.B:
                ans = ans + model.x[i]
            ans = ans == 2
            return ans

        model.x = Var(model.B, initialize=2)
        model.c = Constraint(rule=f)

        self.assertEqual(len(model.c), 1)


class MiscConTests(unittest.TestCase):
    def test_infeasible(self):
        m = ConcreteModel()
        with self.assertRaisesRegex(ValueError, "Constraint 'c' is always infeasible"):
            m.c = Constraint(expr=Constraint.Infeasible)
        self.assertEqual(m.c._data, {})

        with self.assertRaisesRegex(ValueError, "Constraint 'c' is always infeasible"):
            m.c = Constraint.Infeasible
        self.assertEqual(m.c._data, {})
        self.assertIsNone(m.c.expr)

        m.c = (0, 1, 2)
        self.assertIn(None, m.c._data)
        self.assertEqual(m.c.lb, 0)
        self.assertEqual(m.c.ub, 2)

        with self.assertRaisesRegex(ValueError, "Constraint 'c' is always infeasible"):
            m.c = Constraint.Infeasible
        self.assertEqual(m.c._data, {})
        self.assertIsNone(m.c.expr)
        self.assertEqual(m.c.lb, None)
        self.assertEqual(m.c.ub, None)

    def test_slack_methods(self):
        model = ConcreteModel()
        model.x = Var(initialize=2.0)
        L = -1.0
        U = 5.0
        model.cL = Constraint(expr=model.x**2 >= L)
        self.assertEqual(model.cL.lslack(), 5.0)
        self.assertEqual(model.cL.uslack(), float('inf'))
        self.assertEqual(model.cL.slack(), 5.0)
        model.cU = Constraint(expr=model.x**2 <= U)
        self.assertEqual(model.cU.lslack(), float('inf'))
        self.assertEqual(model.cU.uslack(), 1.0)
        self.assertEqual(model.cU.slack(), 1.0)
        model.cR = Constraint(expr=(L, model.x**2, U))
        self.assertEqual(model.cR.lslack(), 5.0)
        self.assertEqual(model.cR.uslack(), 1.0)
        self.assertEqual(model.cR.slack(), 1.0)

    def test_constructor(self):
        a = Constraint(name="b")
        self.assertEqual(a.local_name, "b")
        try:
            a = Constraint(foo="bar")
            self.fail("Can't specify an unexpected constructor option")
        except ValueError:
            pass

    def test_contains(self):
        model = ConcreteModel()
        model.a = Set(initialize=[1, 2, 3])
        model.b = Constraint(model.a)

        self.assertEqual(2 in model.b, False)
        tmp = []
        for i in model.b:
            tmp.append(i)
        self.assertEqual(len(tmp), 0)

    def test_empty_singleton(self):
        a = Constraint()
        a.construct()
        #
        # Even though we construct a ScalarConstraint,
        # if it is not initialized that means it is "empty"
        # and we should encounter errors when trying to access the
        # _ConstraintData interface methods until we assign
        # something to the constraint.
        #
        self.assertEqual(a._constructed, True)
        self.assertEqual(len(a), 0)
        try:
            a()
            self.fail("Component is empty")
        except ValueError:
            pass
        try:
            a.body
            self.fail("Component is empty")
        except ValueError:
            pass
        try:
            a.lower
            self.fail("Component is empty")
        except ValueError:
            pass
        try:
            a.upper
            self.fail("Component is empty")
        except ValueError:
            pass
        try:
            a.equality
            self.fail("Component is empty")
        except ValueError:
            pass
        try:
            a.strict_lower
            self.fail("Component is empty")
        except ValueError:
            pass
        try:
            a.strict_upper
            self.fail("Component is empty")
        except ValueError:
            pass
        x = Var(initialize=1.0)
        x.construct()
        a.set_value((0, x, 2))
        self.assertEqual(len(a), 1)
        self.assertEqual(a(), 1)
        self.assertEqual(a.body(), 1)
        self.assertEqual(a.lower(), 0)
        self.assertEqual(a.upper(), 2)
        self.assertEqual(a.equality, False)
        self.assertEqual(a.strict_lower, False)
        self.assertEqual(a.strict_upper, False)

    def test_unconstructed_singleton(self):
        a = Constraint()
        self.assertEqual(a._constructed, False)
        self.assertEqual(len(a), 0)
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot access .* on AbstractScalarConstraint"
            ".*before it has been constructed",
        ):
            a()
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot access .* on AbstractScalarConstraint"
            ".*before it has been constructed",
        ):
            a.body
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot access .* on AbstractScalarConstraint"
            ".*before it has been constructed",
        ):
            a.lower
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot access .* on AbstractScalarConstraint"
            ".*before it has been constructed",
        ):
            a.upper
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot access .* on AbstractScalarConstraint"
            ".*before it has been constructed",
        ):
            a.equality
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot access .* on AbstractScalarConstraint"
            ".*before it has been constructed",
        ):
            a.strict_lower
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot access .* on AbstractScalarConstraint"
            ".*before it has been constructed",
        ):
            a.strict_upper

        x = Var(initialize=1.0)
        x.construct()
        a.construct()
        a.set_value((0, x, 2))
        self.assertEqual(len(a), 1)
        self.assertEqual(a(), 1)
        self.assertEqual(a.body(), 1)
        self.assertEqual(a.lower(), 0)
        self.assertEqual(a.upper(), 2)
        self.assertEqual(a.equality, False)
        self.assertEqual(a.strict_lower, False)
        self.assertEqual(a.strict_upper, False)

    def test_rule(self):
        def rule1(model):
            return Constraint.Skip

        model = ConcreteModel()
        try:
            model.o = Constraint(rule=rule1)
        except Exception:
            e = sys.exc_info()[1]
            self.fail("Failure to create empty constraint: %s" % str(e))

        #
        def rule1(model):
            return (0.0, model.x, 2.0)

        model = ConcreteModel()
        model.x = Var(initialize=1.1)
        model.o = Constraint(rule=rule1)

        self.assertEqual(model.o(), 1.1)

        #
        def rule1(model, i):
            return Constraint.Skip

        model = ConcreteModel()
        model.a = Set(initialize=[1, 2, 3])
        try:
            model.o = Constraint(model.a, rule=rule1)
        except Exception:
            self.fail("Error generating empty constraint")

        #
        def rule1(model):
            return (0.0, 1.1, 2.0, None)

        model = ConcreteModel()
        try:
            model.o = Constraint(rule=rule1)
            self.fail("Can only return tuples of length 2 or 3")
        except ValueError:
            pass

    def test_tuple_constraint_create(self):
        def rule1(model):
            return (0.0, model.x)

        model = ConcreteModel()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        model.c = Constraint(rule=rule1)
        self.assertEqual(model.c.lower, 0)
        self.assertIs(model.c.body, model.x)
        self.assertEqual(model.c.upper, 0)

        #
        def rule1(model):
            return (model.y, model.x, model.z)

        model = AbstractModel()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        model.c = Constraint(rule=rule1)
        instance = model.create_instance()
        with self.assertRaisesRegex(
            ValueError,
            "Constraint 'c' is a Ranged Inequality with a variable lower bound",
        ):
            instance.c.lower
        self.assertIs(instance.c.body, instance.x)
        with self.assertRaisesRegex(
            ValueError,
            "Constraint 'c' is a Ranged Inequality with a variable upper bound",
        ):
            instance.c.upper
        #

    def test_expression_constructor_coverage(self):
        def rule1(model):
            expr = model.x
            expr = expr == 0.0
            expr = expr >= 1.0
            return expr

        model = AbstractModel()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        model.o = Constraint(rule=rule1)
        self.assertRaises(TypeError, model.create_instance)

        #
        def rule1(model):
            expr = model.U >= model.x
            expr = expr >= model.L
            return expr

        model = ConcreteModel()
        model.x = Var()
        model.L = Param(initialize=0)
        model.U = Param(initialize=1)
        model.o = Constraint(rule=rule1)

        #
        def rule1(model):
            expr = model.x <= model.z
            expr = expr >= model.y
            return expr

        model = AbstractModel()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        model.o = Constraint(rule=rule1)

        # self.assertRaises(ValueError, model.create_instance)
        #
        def rule1(model):
            expr = model.x >= model.z
            expr = model.y >= expr
            return expr

        model = AbstractModel()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        model.o = Constraint(rule=rule1)

        # self.assertRaises(ValueError, model.create_instance)
        #
        def rule1(model):
            expr = model.y <= model.x
            expr = model.y >= expr
            return expr

        model = AbstractModel()
        model.x = Var()
        model.y = Var()
        model.o = Constraint(rule=rule1)

        # self.assertRaises(ValueError, model.create_instance)
        #
        def rule1(model):
            expr = model.x >= model.L
            return expr

        model = ConcreteModel()
        model.x = Var()
        model.L = Param(initialize=0)
        model.o = Constraint(rule=rule1)

        #
        def rule1(model):
            expr = model.U >= model.x
            return expr

        model = ConcreteModel()
        model.x = Var()
        model.U = Param(initialize=0)
        model.o = Constraint(rule=rule1)

        #
        def rule1(model):
            expr = model.x
            expr = expr == 0.0
            expr = expr <= 1.0
            return expr

        model = AbstractModel()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        model.o = Constraint(rule=rule1)
        self.assertRaises(TypeError, model.create_instance)

        #
        def rule1(model):
            expr = model.U <= model.x
            expr = expr <= model.L
            return expr

        model = ConcreteModel()
        model.x = Var()
        model.L = Param(initialize=0)
        model.U = Param(initialize=1)
        model.o = Constraint(rule=rule1)

        #
        def rule1(model):
            expr = model.x >= model.z
            expr = expr <= model.y
            return expr

        model = AbstractModel()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        model.o = Constraint(rule=rule1)

        # self.assertRaises(ValueError, model.create_instance)
        #
        def rule1(model):
            expr = model.x <= model.z
            expr = model.y <= expr
            return expr

        model = AbstractModel()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        model.o = Constraint(rule=rule1)

        # self.assertRaises(ValueError, model.create_instance)
        #
        def rule1(model):
            expr = model.x <= model.L
            return expr

        model = ConcreteModel()
        model.x = Var()
        model.L = Param(initialize=0)
        model.o = Constraint(rule=rule1)

        #
        def rule1(model):
            expr = model.y >= model.x
            expr = model.y <= expr
            return expr

        model = AbstractModel()
        model.x = Var()
        model.y = Var()
        model.o = Constraint(rule=rule1)

        # self.assertRaises(ValueError, model.create_instance)
        #
        def rule1(model):
            expr = model.U <= model.x
            return expr

        model = ConcreteModel()
        model.x = Var()
        model.U = Param(initialize=0)
        model.o = Constraint(rule=rule1)

        #
        def rule1(model):
            return model.x + model.x

        model = ConcreteModel()
        model.x = Var()
        try:
            model.o = Constraint(rule=rule1)
            self.fail("Cannot return an unbounded expression")
        except ValueError:
            pass
        #

    def test_abstract_index(self):
        model = AbstractModel()
        model.A = Set()
        model.B = Set()
        model.C = model.A | model.B
        model.x = Constraint(model.C)

    def test_ranged_inequality_expr(self):
        model = ConcreteModel()
        model.v = Var()
        model.l = Param(initialize=1, mutable=True)
        model.u = Param(initialize=3, mutable=True)
        model.con = Constraint(expr=inequality(model.l, model.v, model.u))
        self.assertIs(model.con.expr.args[0], model.l)
        self.assertIs(model.con.expr.args[1], model.v)
        self.assertIs(model.con.expr.args[2], model.u)

    def test_potentially_variable_bounds(self):
        m = ConcreteModel()
        m.x = Var()
        m.l = Expression()
        m.u = Expression()
        m.c = Constraint(expr=inequality(m.l, m.x, m.u))
        self.assertIs(m.c.lower, m.l)
        self.assertIs(m.c.upper, m.u)
        with self.assertRaisesRegex(
            ValueError, 'No value for uninitialized NumericValue object l'
        ):
            m.c.lb
        with self.assertRaisesRegex(
            ValueError, 'No value for uninitialized NumericValue object u'
        ):
            m.c.ub

        m.l = 5
        m.u = 10
        self.assertIs(m.c.lower, m.l)
        self.assertIs(m.c.upper, m.u)
        self.assertEqual(m.c.lb, 5)
        self.assertEqual(m.c.ub, 10)

        m.l.expr = m.x
        with self.assertRaisesRegex(
            ValueError,
            r"Constraint 'c' is a Ranged Inequality with a variable lower bound",
        ):
            m.c.lower
        self.assertIs(m.c.upper, m.u)
        with self.assertRaisesRegex(
            ValueError,
            r"Constraint 'c' is a Ranged Inequality with a variable lower bound",
        ):
            m.c.lb
        self.assertEqual(m.c.ub, 10)

        m.l = 15
        m.u.expr = m.x
        self.assertIs(m.c.lower, m.l)
        with self.assertRaisesRegex(
            ValueError,
            r"Constraint 'c' is a Ranged Inequality with a variable upper bound",
        ):
            m.c.upper
        self.assertEqual(m.c.lb, 15)
        with self.assertRaisesRegex(
            ValueError,
            r"Constraint 'c' is a Ranged Inequality with a variable upper bound",
        ):
            m.c.ub

        m.l = -float('inf')
        m.u = float('inf')
        self.assertIs(m.c.lower, m.l)
        self.assertIs(m.c.upper, m.u)
        self.assertIsNone(m.c.lb)
        self.assertIsNone(m.c.ub)

        m.l = float('inf')
        m.u = -float('inf')
        self.assertIs(m.c.lower, m.l)
        self.assertIs(m.c.upper, m.u)
        with self.assertRaisesRegex(
            ValueError,
            r"Constraint 'c' created with an invalid "
            r"non-finite lower bound \(inf\)",
        ):
            m.c.lb
        with self.assertRaisesRegex(
            ValueError,
            r"Constraint 'c' created with an invalid "
            r"non-finite upper bound \(-inf\)",
        ):
            m.c.ub

        m.l = float('nan')
        m.u = -float('nan')
        self.assertIs(m.c.lower, m.l)
        self.assertIs(m.c.upper, m.u)
        with self.assertRaisesRegex(
            ValueError,
            r"Constraint 'c' created with an invalid "
            r"non-finite lower bound \(nan\)",
        ):
            m.c.lb
        with self.assertRaisesRegex(
            ValueError,
            r"Constraint 'c' created with an invalid "
            r"non-finite upper bound \(nan\)",
        ):
            m.c.ub

    def test_tuple_expression(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.p = Param(mutable=True, initialize=0)
        m.c = Constraint()

        m.c = (m.x, m.y)
        self.assertTrue(m.c.equality)
        self.assertIs(type(m.c.expr), EqualityExpression)

        with self.assertRaisesRegex(
            ValueError,
            "Constraint 'c' does not have a proper value. "
            "Equality Constraints expressed as 2-tuples cannot "
            "contain None",
        ):
            m.c = (m.x, None)

        with self.assertRaisesRegex(
            ValueError,
            r"Constraint 'c' created with an invalid "
            r"non-finite lower bound \(inf\)",
        ):
            m.c = (m.x, float('inf'))

        with self.assertRaisesRegex(
            ValueError, r"Equality constraint 'c' defined with non-finite term"
        ):
            m.c = EqualityExpression((m.x, None))


if __name__ == "__main__":
    unittest.main()
