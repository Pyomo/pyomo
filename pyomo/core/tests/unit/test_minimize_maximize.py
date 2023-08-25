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
# TestScalarObj                Class for testing single objective
# TestArrayObj                Class for testing array of objective
#

import os
from os.path import abspath, dirname

currdir = dirname(abspath(__file__)) + os.sep

import pyomo.common.unittest as unittest

from pyomo.environ import (
    ConcreteModel,
    AbstractModel,
    Objective,
    ObjectiveList,
    Var,
    Param,
    Set,
    RangeSet,
    value,
    maximize,
    minimize,
    Maximize,
    Minimize,
    simple_objective_rule,
    simple_objectivelist_rule,
)


class TestScalarObj(unittest.TestCase):
    def test_singleton_get_set(self):
        model = ConcreteModel()
        model.o = Minimize(expr=1)
        self.assertEqual(len(model.o), 1)
        self.assertEqual(model.o.expr, 1)
        model.o.expr = 2
        self.assertEqual(model.o.expr, 2)
        model.o.expr += 2
        self.assertEqual(model.o.expr, 4)

    def test_singleton_get_set_value(self):
        model = ConcreteModel()
        model.o = Minimize(expr=1)
        self.assertEqual(len(model.o), 1)
        self.assertEqual(model.o.expr, 1)
        model.o.expr = 2
        self.assertEqual(model.o.expr, 2)
        model.o.expr += 2
        self.assertEqual(model.o.expr, 4)

    def test_scalar_invalid_expr(self):
        m = ConcreteModel()
        m.x = Var()
        with self.assertRaisesRegex(
            ValueError,
            "Cannot assign InequalityExpression to 'obj': "
            "ScalarObjective components only allow numeric expression "
            "types.",
        ):
            m.obj = Minimize(expr=m.x <= 0)

    def test_empty_singleton(self):
        a = Minimize()
        a.construct()
        #
        # Even though we construct a ScalarObjective,
        # if it is not initialized that means it is "empty"
        # and we should encounter errors when trying to access the
        # _ObjectiveData interface methods until we assign
        # something to the objective.
        #
        self.assertEqual(a._constructed, True)
        self.assertEqual(len(a), 0)
        try:
            a()
            self.fail("Component is empty")
        except ValueError:
            pass
        try:
            a.expr
            self.fail("Component is empty")
        except ValueError:
            pass
        try:
            a.sense
            self.fail("Component is empty")
        except ValueError:
            pass
        x = Var(initialize=1.0)
        x.construct()
        a.set_value(x + 1)
        self.assertEqual(len(a), 1)
        self.assertEqual(a(), 2)
        self.assertEqual(a.expr(), 2)
        self.assertEqual(a.sense, minimize)

    def test_unconstructed_singleton(self):
        a = Minimize()
        self.assertEqual(a._constructed, False)
        self.assertEqual(len(a), 0)
        try:
            a()
            self.fail("Component is unconstructed")
        except ValueError:
            pass
        try:
            a.expr
            self.fail("Component is unconstructed")
        except ValueError:
            pass
        try:
            a.sense
            self.fail("Component is unconstructed")
        except ValueError:
            pass
        a.construct()
        a.set_sense(minimize)
        self.assertEqual(len(a), 1)
        self.assertEqual(a(), None)
        self.assertEqual(a.expr, None)
        self.assertEqual(a.sense, minimize)
        a.sense = maximize
        self.assertEqual(len(a), 1)
        self.assertEqual(a(), None)
        self.assertEqual(a.expr, None)
        self.assertEqual(a.sense, maximize)

    def test_numeric_expr(self):
        """Test expr option with a single numeric constant"""
        model = ConcreteModel()
        model.obj = Minimize(expr=0.0)
        self.assertEqual(model.obj(), 0.0)
        self.assertEqual(value(model.obj), 0.0)
        self.assertEqual(value(model.obj._data[None]), 0.0)

    def test_mutable_param_expr(self):
        """Test expr option with a single mutable param"""
        model = ConcreteModel()
        model.p = Param(initialize=1.0, mutable=True)
        model.obj = Minimize(expr=model.p)

        self.assertEqual(model.obj(), 1.0)
        self.assertEqual(value(model.obj), 1.0)
        self.assertEqual(value(model.obj._data[None]), 1.0)

    def test_immutable_param_expr(self):
        """Test expr option a single immutable param"""
        model = ConcreteModel()
        model.p = Param(initialize=1.0, mutable=False)
        model.obj = Minimize(expr=model.p)

        self.assertEqual(model.obj(), 1.0)
        self.assertEqual(value(model.obj), 1.0)
        self.assertEqual(value(model.obj._data[None]), 1.0)

    def test_var_expr(self):
        """Test expr option with a single var"""
        model = ConcreteModel()
        model.x = Var(initialize=1.0)
        model.obj = Minimize(expr=model.x)

        self.assertEqual(model.obj(), 1.0)
        self.assertEqual(value(model.obj), 1.0)
        self.assertEqual(value(model.obj._data[None]), 1.0)

    def test_expr1_option(self):
        """Test expr option"""
        model = ConcreteModel()
        model.B = RangeSet(1, 4)
        model.x = Var(model.B, initialize=2)
        ans = 0
        for i in model.B:
            ans = ans + model.x[i]
        model.obj = Minimize(expr=ans)

        self.assertEqual(model.obj(), 8)
        self.assertEqual(value(model.obj), 8)
        self.assertEqual(value(model.obj._data[None]), 8)

    def test_expr2_option(self):
        """Test expr option"""
        model = ConcreteModel()
        model.x = Var(initialize=2)
        model.obj = Minimize(expr=model.x)

        self.assertEqual(model.obj(), 2)
        self.assertEqual(value(model.obj), 2)
        self.assertEqual(value(model.obj._data[None]), 2)

    def test_rule_option(self):
        """Test rule option"""
        model = ConcreteModel()

        def f(model):
            ans = 0
            for i in [1, 2, 3, 4]:
                ans = ans + model.x[i]
            return ans

        model.x = Var(RangeSet(1, 4), initialize=2)
        model.obj = Minimize(rule=f)

        self.assertEqual(model.obj(), 8)
        self.assertEqual(value(model.obj), 8)
        self.assertEqual(value(model.obj._data[None]), 8)

    def test_arguments(self):
        """Test that arguments notare of type ScalarSet"""
        model = ConcreteModel()

        def rule(model):
            return 1

        try:
            model.obj = Minimize(model, rule=rule)
        except TypeError:
            pass
        else:
            self.fail("Minimize should only accept ScalarSets")

    def test_sense_option(self):
        """Test sense option"""
        model = ConcreteModel()

        def rule(model):
            return 1.0

        model.obj = Maximize(rule=rule)

        self.assertEqual(model.obj.sense, maximize)
        self.assertEqual(model.obj.is_minimizing(), False)

    def test_dim(self):
        """Test dim method"""
        model = ConcreteModel()

        def rule(model):
            return 1

        model.obj = Minimize(rule=rule)

        self.assertEqual(model.obj.dim(), 0)

    def test_keys(self):
        """Test keys method"""
        model = ConcreteModel()

        def rule(model):
            return 1

        model.obj = Minimize(rule=rule)

        self.assertEqual(list(model.obj.keys()), [None])
        self.assertEqual(id(model.obj), id(model.obj[None]))

    def test_len(self):
        """Test len method"""
        model = AbstractModel()

        def rule(model):
            return 1.0

        model.obj = Minimize(rule=rule)
        self.assertEqual(len(model.obj), 0)
        inst = model.create_instance()
        self.assertEqual(len(inst.obj), 1)

        model = AbstractModel()
        """Test rule option"""

        def f(model):
            ans = 0
            for i in model.x.keys():
                ans = ans + model.x[i]
            return ans

        model = AbstractModel()
        model.x = Var(RangeSet(1, 4), initialize=2)
        model.obj = Minimize(rule=f)

        self.assertEqual(len(model.obj), 0)
        inst = model.create_instance()
        self.assertEqual(len(inst.obj), 1)

    def test_keys_empty(self):
        """Test keys method"""
        model = ConcreteModel()
        model.o = Minimize()

        self.assertEqual(list(model.o.keys()), [])

    def test_len_empty(self):
        """Test len method"""
        model = ConcreteModel()
        model.o = Minimize()
        self.assertEqual(len(model.o), 0)


class TestArrayObj(unittest.TestCase):
    def create_model(self):
        #
        # Create Model
        #
        model = ConcreteModel()
        model.A = Set(initialize=[1, 2])
        return model

    def test_objdata_get_set(self):
        model = ConcreteModel()
        model.o = Minimize([1], rule=lambda m, i: 1)
        self.assertEqual(len(model.o), 1)
        self.assertEqual(model.o[1].expr, 1)
        model.o[1].expr = 2
        self.assertEqual(model.o[1].expr, 2)
        model.o[1].expr += 2
        self.assertEqual(model.o[1].expr, 4)

    def test_objdata_get_set_value(self):
        model = ConcreteModel()
        model.o = Minimize([1], rule=lambda m, i: 1)
        self.assertEqual(len(model.o), 1)
        self.assertEqual(model.o[1].expr, 1)
        model.o[1].expr = 2
        self.assertEqual(model.o[1].expr, 2)
        model.o[1].expr += 2
        self.assertEqual(model.o[1].expr, 4)

    def test_objdata_get_set_sense(self):
        model = ConcreteModel()
        model.o = Maximize([1], rule=lambda m, i: 1)
        self.assertEqual(len(model.o), 1)
        self.assertEqual(model.o[1].expr, 1)
        self.assertEqual(model.o[1].sense, maximize)
        model.o[1].set_sense(minimize)
        self.assertEqual(model.o[1].sense, minimize)
        model.o[1].sense = maximize
        self.assertEqual(model.o[1].sense, maximize)

    def test_maximize(self):
        model = ConcreteModel()
        model.o = Maximize([1], rule=lambda m, i: 1)
        self.assertEqual(len(model.o), 1)
        self.assertEqual(model.o[1].expr, 1)
        self.assertEqual(model.o[1].sense, maximize)

    def test_maximize_decorator(self):
        model = ConcreteModel()

        @model.Maximize([1])
        def o(m, i):
            return 1

        self.assertEqual(len(model.o), 1)
        self.assertEqual(model.o[1].expr, 1)
        self.assertEqual(model.o[1].sense, maximize)

    def test_minimize(self):
        model = ConcreteModel()
        model.o = Minimize([1], rule=lambda m, i: 1)
        self.assertEqual(len(model.o), 1)
        self.assertEqual(model.o[1].expr, 1)
        self.assertEqual(model.o[1].sense, minimize)

    def test_minimize_decorator(self):
        model = ConcreteModel()

        @model.Minimize([1])
        def o(m, i):
            return 1

        self.assertEqual(len(model.o), 1)
        self.assertEqual(model.o[1].expr, 1)
        self.assertEqual(model.o[1].sense, minimize)

    def test_rule_option1(self):
        """Test rule option"""
        model = self.create_model()
        model.B = RangeSet(1, 4)

        def f(model, i):
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            return ans

        model.x = Var(model.B, initialize=2)
        model.obj = Minimize(model.A, rule=f)

        self.assertEqual(model.obj[1](), 8)
        self.assertEqual(model.obj[2](), 16)
        self.assertEqual(value(model.obj[1]), 8)
        self.assertEqual(value(model.obj[2]), 16)

    def test_rule_option2(self):
        """Test rule option"""
        model = self.create_model()
        model.B = RangeSet(1, 4)

        def f(model, i):
            if i == 1:
                return Objective.Skip
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            return ans

        model.x = Var(model.B, initialize=2)
        model.obj = Minimize(model.A, rule=f)

        self.assertEqual(model.obj[2](), 16)
        self.assertEqual(value(model.obj[2]), 16)

    def test_rule_option3(self):
        """Test rule option"""
        model = self.create_model()
        model.B = RangeSet(1, 4)

        @simple_objective_rule
        def f(model, i):
            if i == 1:
                return None
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            return ans

        model.x = Var(model.B, initialize=2)
        model.obj = Minimize(model.A, rule=f)

        self.assertEqual(model.obj[2](), 16)
        self.assertEqual(value(model.obj[2]), 16)

    def test_rule_numeric_expr(self):
        """Test rule option with returns a single numeric constant for the expression"""
        model = self.create_model()

        def f(model, i):
            return 1.0

        model.obj = Minimize(model.A, rule=f)

        self.assertEqual(model.obj[2](), 1.0)
        self.assertEqual(value(model.obj[2]), 1.0)

    def test_rule_immutable_param_expr(self):
        """Test rule option that returns a single immutable param for the expression"""
        model = self.create_model()

        def f(model, i):
            return model.p[i]

        model.p = Param(RangeSet(1, 4), initialize=1.0, mutable=False)
        model.x = Var()
        model.obj = Minimize(model.A, rule=f)

        self.assertEqual(model.obj[2](), 1.0)
        self.assertEqual(value(model.obj[2]), 1.0)

    def test_rule_mutable_param_expr(self):
        """Test rule option that returns a single mutable param for the expression"""
        model = self.create_model()

        def f(model, i):
            return model.p[i]

        model.r = RangeSet(1, 4)
        model.p = Param(model.r, initialize=1.0, mutable=True)
        model.x = Var()
        model.obj = Minimize(model.A, rule=f)

        self.assertEqual(model.obj[2](), 1.0)
        self.assertEqual(value(model.obj[2]), 1.0)

    def test_rule_var_expr(self):
        """Test rule option that returns a single var for the expression"""
        model = self.create_model()

        def f(model, i):
            return model.x[i]

        model.r = RangeSet(1, 4)
        model.x = Var(model.r, initialize=1.0)
        model.obj = Minimize(model.A, rule=f)

        self.assertEqual(model.obj[2](), 1.0)
        self.assertEqual(value(model.obj[2]), 1.0)

    def test_sense_option(self):
        """Test sense option"""
        model = self.create_model()
        model.obj1 = Maximize(model.A, rule=lambda m, i: 1.0)
        model.obj2 = Minimize(model.A, rule=lambda m, i: 1.0)
        self.assertTrue(len(model.A) > 0)
        self.assertEqual(len(model.obj1), len(model.A))
        self.assertEqual(len(model.obj2), len(model.A))
        for i in model.A:
            self.assertEqual(model.obj1[i].sense, maximize)
            self.assertEqual(model.obj1[i].is_minimizing(), False)
            self.assertEqual(model.obj2[i].sense, minimize)
            self.assertEqual(model.obj2[i].is_minimizing(), True)

    def test_dim(self):
        """Test dim method"""
        model = self.create_model()
        model.obj = Minimize(model.A)

        self.assertEqual(model.obj.dim(), 1)

    def test_keys(self):
        """Test keys method"""
        model = self.create_model()

        def A_rule(model, i):
            return model.x

        model.x = Var()
        model.obj = Minimize(model.A, rule=A_rule)

        self.assertEqual(len(list(model.obj.keys())), 2)

    def test_len(self):
        """Test len method"""
        model = self.create_model()
        model.obj = Minimize(model.A)
        self.assertEqual(len(model.obj), 0)

        model = self.create_model()
        """Test rule option"""

        def f(model):
            ans = 0
            for i in model.x.keys():
                ans = ans + model.x[i]
            return ans

        model.x = Var(RangeSet(1, 4), initialize=2)
        model.obj = Minimize(rule=f)

        self.assertEqual(len(model.obj), 1)


class Test2DArrayObj(unittest.TestCase):
    def create_model(self):
        model = ConcreteModel()
        model.A = Set(initialize=[1, 2])
        return model

    def test_rule_option1(self):
        """Test rule option"""
        model = self.create_model()
        model.B = RangeSet(1, 4)

        def f(model, i, k):
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            return ans

        model.x = Var(model.B, initialize=2)
        model.obj = Minimize(model.A, model.A, rule=f)

        try:
            self.assertEqual(model.obj(), None)
            self.fail("Expected TypeError")
        except TypeError:
            pass

        self.assertEqual(model.obj[1, 1](), 8)
        self.assertEqual(model.obj[2, 1](), 16)
        self.assertEqual(value(model.obj[1, 1]), 8)
        self.assertEqual(value(model.obj[2, 1]), 16)

    def test_sense_option(self):
        """Test sense option"""
        model = self.create_model()
        model.obj1 = Maximize(model.A, model.A, rule=lambda m, i, j: 1.0)
        model.obj2 = Minimize(model.A, model.A, rule=lambda m, i, j: 1.0)
        self.assertTrue(len(model.A) > 0)
        self.assertEqual(len(model.obj1), len(model.A) * len(model.A))
        self.assertEqual(len(model.obj2), len(model.A) * len(model.A))
        for i in model.A:
            for j in model.A:
                self.assertEqual(model.obj1[i, j].sense, maximize)
                self.assertEqual(model.obj1[i, j].is_minimizing(), False)
                self.assertEqual(model.obj2[i, j].sense, minimize)
                self.assertEqual(model.obj2[i, j].is_minimizing(), True)

    def test_dim(self):
        """Test dim method"""
        model = self.create_model()
        model.obj = Minimize(model.A, model.A)

        self.assertEqual(model.obj.dim(), 2)

    def test_keys(self):
        """Test keys method"""
        model = self.create_model()

        def A_rule(model, i, j):
            return model.x

        model.x = Var()
        model.obj = Minimize(model.A, model.A, rule=A_rule)

        self.assertEqual(len(list(model.obj.keys())), 4)

    def test_len(self):
        """Test len method"""
        model = self.create_model()
        model.obj = Minimize(model.A, model.A)
        self.assertEqual(len(model.obj), 0)

        model = self.create_model()
        """Test rule option"""

        def f(model):
            ans = 0
            for i in model.x.keys():
                ans = ans + model.x[i]
            return ans

        model.x = Var(RangeSet(1, 4), initialize=2)
        model.obj = Minimize(rule=f)

        self.assertEqual(len(model.obj), 1)


class MiscObjTests(unittest.TestCase):
    def test_constructor(self):
        a = Minimize(name="b")
        self.assertEqual(a.local_name, "b")
        try:
            a = Minimize(foo="bar")
            self.fail("Can't specify an unexpected constructor option")
        except ValueError:
            pass

    def test_rule(self):
        def rule1(model):
            return []

        model = ConcreteModel()
        try:
            model.o = Minimize(rule=rule1)
            self.fail("Error generating objective")
        except Exception:
            pass
        #
        model = ConcreteModel()

        def rule1(model):
            return 1.1

        model = ConcreteModel()
        model.o = Minimize(rule=rule1)
        self.assertEqual(model.o(), 1.1)
        #
        model = ConcreteModel()

        def rule1(model, i):
            return 1.1

        model = ConcreteModel()
        model.a = Set(initialize=[1, 2, 3])
        try:
            model.o = Minimize(model.a, rule=rule1)
        except Exception:
            self.fail("Error generating objective")

    def test_abstract_index(self):
        model = AbstractModel()
        model.A = Set()
        model.B = Set()
        model.C = model.A | model.B
        model.x = Minimize(model.C)


if __name__ == "__main__":
    unittest.main()
