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
#
# Unit Tests for Elements of a Model
#
# TestSimpleVar                Class for testing single variables
# TestArrayVar                Class for testing array of variables
#

import os
from os.path import abspath, dirname

currdir = dirname(abspath(__file__)) + os.sep

from io import StringIO

import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept

from pyomo.core.base import IntegerSet
from pyomo.core.expr.numeric_expr import (
    NPV_ProductExpression,
    NPV_MaxExpression,
    NPV_MinExpression,
)
from pyomo.core.staleflag import StaleFlagManager
from pyomo.environ import (
    AbstractModel,
    ConcreteModel,
    Set,
    Param,
    Var,
    VarList,
    RangeSet,
    Suffix,
    Expression,
    NonPositiveReals,
    PositiveReals,
    Reals,
    RealSet,
    NonNegativeReals,
    Integers,
    Binary,
    value,
)
from pyomo.core.base.units_container import units, pint_available, UnitsError


class TestVarData(unittest.TestCase):
    def test_lower_bound(self):
        m = ConcreteModel()
        m.x = Var()
        m.p = Param(mutable=True, initialize=2)
        self.assertIsNone(m.x.lower)
        m.x.domain = NonNegativeReals
        self.assertIs(type(m.x.lower), int)
        self.assertEqual(value(m.x.lower), 0)
        m.x.domain = Reals
        m.x.setlb(5 * m.p)
        self.assertIs(type(m.x.lower), NPV_ProductExpression)
        self.assertEqual(value(m.x.lower), 10)
        m.x.domain = NonNegativeReals
        self.assertIs(type(m.x.lower), NPV_MaxExpression)
        self.assertEqual(value(m.x.lower), 10)
        with self.assertRaisesRegex(
            ValueError,
            "Potentially variable input of type 'ScalarVar' "
            "supplied as lower bound for variable 'x'",
        ):
            m.x.setlb(m.x)

    def test_lower_bound_setter(self):
        m = ConcreteModel()
        m.x = Var()
        self.assertIsNone(m.x.lb)
        m.x.lb = 1
        self.assertEqual(m.x.lb, 1)
        m.x.lower = 2
        self.assertEqual(m.x.lb, 2)
        m.x.setlb(3)
        self.assertEqual(m.x.lb, 3)

        m.y = Var([1])
        self.assertIsNone(m.y[1].lb)
        m.y[1].lb = 1
        self.assertEqual(m.y[1].lb, 1)
        m.y[1].lower = 2
        self.assertEqual(m.y[1].lb, 2)
        m.y[1].setlb(3)
        self.assertEqual(m.y[1].lb, 3)

    def test_upper_bound(self):
        m = ConcreteModel()
        m.x = Var()
        m.p = Param(mutable=True, initialize=2)
        self.assertIsNone(m.x.upper)
        m.x.domain = NonPositiveReals
        self.assertIs(type(m.x.upper), int)
        self.assertEqual(value(m.x.upper), 0)
        m.x.domain = Reals
        m.x.setub(-5 * m.p)
        self.assertIs(type(m.x.upper), NPV_ProductExpression)
        self.assertEqual(value(m.x.upper), -10)
        m.x.domain = NonPositiveReals
        self.assertIs(type(m.x.upper), NPV_MinExpression)
        self.assertEqual(value(m.x.upper), -10)
        with self.assertRaisesRegex(
            ValueError,
            "Potentially variable input of type 'ScalarVar' "
            "supplied as upper bound for variable 'x'",
        ):
            m.x.setub(m.x)

    def test_upper_bound_setter(self):
        m = ConcreteModel()
        m.x = Var()
        self.assertIsNone(m.x.ub)
        m.x.ub = 1
        self.assertEqual(m.x.ub, 1)
        m.x.upper = 2
        self.assertEqual(m.x.ub, 2)
        m.x.setub(3)
        self.assertEqual(m.x.ub, 3)

        m.y = Var([1])
        self.assertIsNone(m.y[1].ub)
        m.y[1].ub = 1
        self.assertEqual(m.y[1].ub, 1)
        m.y[1].upper = 2
        self.assertEqual(m.y[1].ub, 2)
        m.y[1].setub(3)
        self.assertEqual(m.y[1].ub, 3)

    def test_lb(self):
        m = ConcreteModel()
        m.x = Var()
        self.assertEqual(m.x.lb, None)
        m.x.domain = NonNegativeReals
        self.assertEqual(m.x.lb, 0)
        m.x.lb = float('inf')
        with self.assertRaisesRegex(
            ValueError, r'invalid non-finite lower bound \(inf\)'
        ):
            m.x.lb
        m.x.lb = float('nan')
        with self.assertRaisesRegex(
            ValueError, r'invalid non-finite lower bound \(nan\)'
        ):
            m.x.lb

    def test_ub(self):
        m = ConcreteModel()
        m.x = Var()
        self.assertEqual(m.x.ub, None)
        m.x.domain = NonPositiveReals
        self.assertEqual(m.x.ub, 0)
        m.x.ub = float('-inf')
        with self.assertRaisesRegex(
            ValueError, r'invalid non-finite upper bound \(-inf\)'
        ):
            m.x.ub
        m.x.ub = float('nan')
        with self.assertRaisesRegex(
            ValueError, r'invalid non-finite upper bound \(nan\)'
        ):
            m.x.ub

    def test_bounds(self):
        m = ConcreteModel()
        m.x = Var()
        lb, ub = m.x.bounds
        self.assertEqual(lb, None)
        self.assertEqual(ub, None)
        m.x.domain = NonNegativeReals
        lb, ub = m.x.bounds
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)
        m.x.lb = float('inf')
        with self.assertRaisesRegex(
            ValueError, r'invalid non-finite lower bound \(inf\)'
        ):
            lb, ub = m.x.bounds
        m.x.lb = float('nan')
        with self.assertRaisesRegex(
            ValueError, r'invalid non-finite lower bound \(nan\)'
        ):
            lb, ub = m.x.bounds

        m.x.lb = None
        m.x.domain = NonPositiveReals
        lb, ub = m.x.bounds
        self.assertEqual(lb, None)
        self.assertEqual(ub, 0)
        m.x.ub = float('-inf')
        with self.assertRaisesRegex(
            ValueError, r'invalid non-finite upper bound \(-inf\)'
        ):
            lb, ub = m.x.bounds
        m.x.ub = float('nan')
        with self.assertRaisesRegex(
            ValueError, r'invalid non-finite upper bound \(nan\)'
        ):
            lb, ub = m.x.bounds


class PyomoModel(unittest.TestCase):
    def setUp(self):
        self.model = AbstractModel()
        self.instance = None

    def tearDown(self):
        self.model = None
        self.instance = None

    def construct(self, filename=None):
        if filename is not None:
            self.instance = self.model.create_instance(filename)
        else:
            self.instance = self.model.create_instance()


class TestSimpleVar(PyomoModel):
    def setUp(self):
        #
        # Create Model
        #
        PyomoModel.setUp(self)

    def test_fixed_attr(self):
        """Test fixed attribute"""
        self.model.x = Var()
        self.instance = self.model.create_instance()
        self.instance.x.fixed = True
        self.assertEqual(self.instance.x.fixed, True)

    def Xtest_setlb_nondata_expression(self):
        model = ConcreteModel()
        model.x = Var()
        model.e = Expression()
        with self.assertRaises(ValueError):
            model.x.setlb(model.e)
        model.e.expr = 1.0
        with self.assertRaises(ValueError):
            model.x.setlb(model.e)
        model.y = Var()
        with self.assertRaises(ValueError):
            model.x.setlb(model.y)
        model.y.value = 1.0
        with self.assertRaises(ValueError):
            model.x.setlb(model.y)
        model.y.fix()
        with self.assertRaises(ValueError):
            model.x.setlb(model.y + 1)

    def Xtest_setub_nondata_expression(self):
        model = ConcreteModel()
        model.x = Var()
        model.e = Expression()
        with self.assertRaises(ValueError):
            model.x.setub(model.e)
        model.e.expr = 1.0
        with self.assertRaises(ValueError):
            model.x.setub(model.e)
        model.y = Var()
        with self.assertRaises(ValueError):
            model.x.setub(model.y)
        model.y.value = 1.0
        with self.assertRaises(ValueError):
            model.x.setub(model.y)
        model.y.fix()
        with self.assertRaises(ValueError):
            model.x.setub(model.y + 1)

    def Xtest_setlb_data_expression(self):
        model = ConcreteModel()
        model.x = Var()
        model.p = Param(mutable=True)
        model.x.setlb(model.p)
        model.x.setlb(model.p**2 + 1)
        model.p.value = 1.0
        model.x.setlb(model.p)
        model.x.setlb(model.p**2)
        model.x.setlb(1.0)

    def Xtest_setub_data_expression(self):
        model = ConcreteModel()
        model.x = Var()
        model.p = Param(mutable=True)
        model.x.setub(model.p)
        model.x.setub(model.p**2 + 1)
        model.p.value = 1.0
        model.x.setub(model.p)
        model.x.setub(model.p**2)
        model.x.setub(1.0)

    def test_setlb_indexed(self):
        """Test setlb variables method"""
        self.model.B = RangeSet(4)
        self.model.y = Var(self.model.B, dense=True)

        self.instance = self.model.create_instance()
        self.assertEqual(len(self.instance.y) > 0, True)
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].lb, None)
        self.instance.y.setlb(1)
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].lb, 1)
        self.instance.y.setlb(None)
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].lb, None)

    def test_setub_indexed(self):
        """Test setub variables method"""
        self.model.B = RangeSet(4)
        self.model.y = Var(self.model.B, dense=True)

        self.instance = self.model.create_instance()
        self.assertEqual(len(self.instance.y) > 0, True)
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].ub, None)
        self.instance.y.setub(1)
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].ub, 1)
        self.instance.y.setub(None)
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].ub, None)

    def test_fix_all(self):
        """Test fix all variables method"""
        self.model.B = RangeSet(4)
        self.model.x = Var()
        self.model.y = Var(self.model.B, dense=True)

        self.instance = self.model.create_instance()
        self.instance.fix_all_vars()
        self.assertEqual(self.instance.x.fixed, True)

        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].fixed, True)

    def test_unfix_all(self):
        """Test unfix all variables method"""
        self.model.B = RangeSet(4)
        self.model.x = Var()
        self.model.y = Var(self.model.B)

        self.instance = self.model.create_instance()
        self.instance.x.fixed = True
        for a in self.instance.B:
            self.instance.y[a].fixed = True
        self.instance.unfix_all_vars()

        self.assertEqual(self.instance.x.fixed, False)
        for a in self.instance.B:
            self.assertEqual(self.instance.y[a].fixed, False)

    def test_fix_indexed(self):
        """Test fix variables method"""
        self.model.B = RangeSet(4)
        self.model.y = Var(self.model.B, dense=True)

        self.instance = self.model.create_instance()
        self.assertEqual(len(self.instance.y) > 0, True)
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].value, None)
            self.assertEqual(self.instance.y[a].fixed, False)
        self.instance.y.fix()
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].value, None)
            self.assertEqual(self.instance.y[a].fixed, True)
        self.instance.y.free()
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].value, None)
            self.assertEqual(self.instance.y[a].fixed, False)
        self.instance.y.fix(1)
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].value, 1)
            self.assertEqual(self.instance.y[a].fixed, True)
        self.instance.y.unfix()
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].value, 1)
            self.assertEqual(self.instance.y[a].fixed, False)
        self.instance.y.fix(None)
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].value, None)
            self.assertEqual(self.instance.y[a].fixed, True)
        self.instance.y.unfix()
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].value, None)
            self.assertEqual(self.instance.y[a].fixed, False)

        self.instance.y[1].fix()
        self.assertEqual(self.instance.y[1].value, None)
        self.assertEqual(self.instance.y[1].fixed, True)
        self.instance.y[1].free()
        self.assertEqual(self.instance.y[1].value, None)
        self.assertEqual(self.instance.y[1].fixed, False)
        self.instance.y[1].fix(value=1)
        self.assertEqual(self.instance.y[1].value, 1)
        self.assertEqual(self.instance.y[1].fixed, True)
        self.instance.y[1].unfix()
        self.assertEqual(self.instance.y[1].value, 1)
        self.assertEqual(self.instance.y[1].fixed, False)
        self.instance.y[1].fix(value=None)
        self.assertEqual(self.instance.y[1].value, None)
        self.assertEqual(self.instance.y[1].fixed, True)

    def test_unfix_indexed(self):
        """Test unfix variables method"""
        self.model.B = RangeSet(4)
        self.model.y = Var(self.model.B)

        self.instance = self.model.create_instance()
        for a in self.instance.B:
            self.instance.y[a].fixed = True
        self.instance.unfix_all_vars()

        for a in self.instance.B:
            self.assertEqual(self.instance.y[a].fixed, False)

    def test_fix_nonindexed(self):
        """Test fix variables method"""
        self.model.B = RangeSet(4)
        self.model.x = Var()

        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x.value, None)
        self.assertEqual(self.instance.x.fixed, False)
        self.instance.x.fix()
        self.assertEqual(self.instance.x.value, None)
        self.assertEqual(self.instance.x.fixed, True)
        self.instance.x.free()
        self.assertEqual(self.instance.x.value, None)
        self.assertEqual(self.instance.x.fixed, False)
        self.instance.x.fix(1)
        self.assertEqual(self.instance.x.value, 1)
        self.assertEqual(self.instance.x.fixed, True)
        self.instance.x.unfix()
        self.assertEqual(self.instance.x.value, 1)
        self.assertEqual(self.instance.x.fixed, False)
        self.instance.x.fix(None)
        self.assertEqual(self.instance.x.value, None)
        self.assertEqual(self.instance.x.fixed, True)

    def test_unfix_nonindexed(self):
        """Test unfix variables method"""
        self.model.B = RangeSet(4)
        self.model.x = Var()
        self.model.y = Var(self.model.B)

        self.instance = self.model.create_instance()
        self.instance.x.fixed = True
        self.instance.x.unfix()

        self.assertEqual(self.instance.x.fixed, False)

    def test_value_attr(self):
        """Test value attribute"""
        self.model.x = Var()
        self.instance = self.model.create_instance()
        self.instance.x.value = 3.5
        self.assertEqual(self.instance.x.value, 3.5)

    def test_domain_attr(self):
        """Test domain attribute"""
        self.model.x = Var()
        self.instance = self.model.create_instance()
        self.instance.x.domain = Integers
        self.assertEqual(type(self.instance.x.domain), IntegerSet)
        self.assertEqual(self.instance.x.is_integer(), True)
        self.assertEqual(self.instance.x.is_binary(), False)
        self.assertEqual(self.instance.x.is_continuous(), False)

    def test_name_attr(self):
        """Test name attribute"""
        #
        # A user would never need to do this, but this
        # attribute is needed within Pyomo
        #
        self.model.x = Var()
        self.model.x._name = "foo"
        self.assertEqual(self.model.x.name, "foo")

    def test_lb_attr1(self):
        """Test lb attribute"""
        self.model.x = Var()
        self.instance = self.model.create_instance()
        self.instance.x.setlb(-1.0)
        self.assertEqual(value(self.instance.x.lb), -1.0)

    def test_lb_attr2(self):
        """Test lb attribute"""
        self.model.x = Var(within=NonNegativeReals, bounds=(-1, 2))
        self.instance = self.model.create_instance()
        self.assertEqual(value(self.instance.x.lb), 0.0)
        self.assertEqual(value(self.instance.x.ub), 2.0)

    def test_lb_attr3(self):
        """Test lb attribute"""
        self.model.p = Param(mutable=True, initialize=1)
        self.model.x = Var(within=NonNegativeReals, bounds=(self.model.p, None))
        self.instance = self.model.create_instance()
        self.assertEqual(value(self.instance.x.lb), 1.0)
        self.instance.p = 2
        self.assertEqual(value(self.instance.x.lb), 2.0)

    def test_ub_attr1(self):
        """Test ub attribute"""
        self.model.x = Var()
        self.instance = self.model.create_instance()
        self.instance.x.setub(1.0)
        self.assertEqual(value(self.instance.x.ub), 1.0)

    def test_ub_attr2(self):
        """Test ub attribute"""
        self.model.x = Var(within=NonPositiveReals, bounds=(-2, 1))
        self.instance = self.model.create_instance()
        self.assertEqual(value(self.instance.x.lb), -2.0)
        self.assertEqual(value(self.instance.x.ub), 0.0)

    def test_within_option(self):
        """Test within option"""
        self.model.x = Var(within=Reals)
        self.construct()
        self.assertEqual(type(self.instance.x.domain), RealSet)
        self.assertEqual(self.instance.x.is_integer(), False)
        self.assertEqual(self.instance.x.is_binary(), False)
        self.assertEqual(self.instance.x.is_continuous(), True)

    def test_bounds_option1(self):
        """Test bounds option"""

        def x_bounds(model):
            return (-1.0, 1.0)

        self.model.x = Var(bounds=x_bounds)
        self.instance = self.model.create_instance()
        self.assertEqual(value(self.instance.x.lb), -1.0)
        self.assertEqual(value(self.instance.x.ub), 1.0)

    def test_bounds_option2(self):
        """Test bounds option"""
        self.model.x = Var(bounds=(-1.0, 1.0))
        self.instance = self.model.create_instance()
        self.assertEqual(value(self.instance.x.lb), -1.0)
        self.assertEqual(value(self.instance.x.ub), 1.0)

    def test_rule_option(self):
        """Test rule option"""

        def x_init(model):
            return 1.3

        self.model.x = Var(initialize=x_init)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x.value, 1.3)

    def test_initialize_with_function(self):
        """Test initialize option with an initialization rule"""

        def init_rule(model):
            return 1.3

        self.model.x = Var(initialize=init_rule)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x.value, 1.3)
        self.instance.x = 1
        self.assertEqual(self.instance.x.value, 1)

    def test_initialize_with_dict(self):
        """Test initialize option with a dictionary"""
        self.model.x = Var(initialize={None: 1.3})
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x.value, 1.3)
        self.instance.x = 1
        self.assertEqual(self.instance.x.value, 1)

    def test_initialize_with_const(self):
        """Test initialize option with a constant"""
        self.model.x = Var(initialize=1.3)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x.value, 1.3)
        self.instance.x = 1
        self.assertEqual(self.instance.x.value, 1)

    def test_without_initial_value(self):
        """Test default initial value"""
        self.model.x = Var()
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x.value, None)
        self.instance.x = 6
        self.assertEqual(self.instance.x.value, 6)

    def test_dim(self):
        """Test dim method"""
        self.model.x = Var()
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x.dim(), 0)

    def test_keys(self):
        """Test keys method"""
        self.model.x = Var()
        self.instance = self.model.create_instance()
        self.assertEqual(list(self.instance.x.keys()), [None])
        self.assertEqual(id(self.instance.x), id(self.instance.x[None]))

    def test_len(self):
        """Test len method"""
        self.model.x = Var()
        self.assertEqual(len(self.model.x), 0)
        self.instance = self.model.create_instance()
        self.assertEqual(len(self.instance.x), 1)

    def test_value(self):
        """Check the value of the variable"""
        self.model.x = Var(initialize=3.3)
        self.instance = self.model.create_instance()
        tmp = value(self.instance.x.value)
        self.assertEqual(type(tmp), float)
        self.assertEqual(tmp, 3.3)
        tmp = float(self.instance.x.value)
        self.assertEqual(type(tmp), float)
        self.assertEqual(tmp, 3.3)
        tmp = int(self.instance.x.value)
        self.assertEqual(type(tmp), int)
        self.assertEqual(tmp, 3)


class TestArrayVar(TestSimpleVar):
    def setUp(self):
        #
        # Create Model
        #
        PyomoModel.setUp(self)
        self.model.A = Set(initialize=[1, 2])

    def test_fixed_attr(self):
        """Test fixed attribute"""
        self.model.x = Var(self.model.A)
        self.model.y = Var(self.model.A)
        self.instance = self.model.create_instance()
        self.instance.x.fixed = True
        self.assertEqual(self.instance.x[1].fixed, False)
        self.instance.y[1].fixed = True
        self.assertEqual(self.instance.y[1].fixed, True)

    def test_value_attr(self):
        """Test value attribute"""
        self.model.x = Var(self.model.A, dense=True)
        self.model.y = Var(self.model.A, dense=True)
        self.instance = self.model.create_instance()
        try:
            self.instance.x = 3.5
            self.fail("Expected ValueError")
        except ValueError:
            pass
        self.instance.y[1] = 3.5
        self.assertEqual(self.instance.y[1].value, 3.5)

    # def test_lb_attr(self):
    # """Test lb attribute"""
    # self.model.x = Var(self.model.A)
    # self.instance = self.model.create_instance()
    # self.instance.x.setlb(-1.0)
    # self.assertEqual(value(self.instance.x[1].lb), -1.0)

    # def test_ub_attr(self):
    # """Test ub attribute"""
    # self.model.x = Var(self.model.A)
    # self.instance = self.model.create_instance()
    # self.instance.x.setub(1.0)
    # self.assertEqual(value(self.instance.x[1].ub), 1.0)

    def test_initialize_with_function(self):
        """Test initialize option with an initialization rule"""

        def init_rule(model, key):
            i = key + 11
            return key == 1 and 1.3 or 2.3

        self.model.x = Var(self.model.A, initialize=init_rule)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1].value, 1.3)
        self.assertEqual(self.instance.x[2].value, 2.3)
        self.instance.x[1] = 1
        self.instance.x[2] = 2
        self.assertEqual(self.instance.x[1].value, 1)
        self.assertEqual(self.instance.x[2].value, 2)

    def test_initialize_with_dict(self):
        """Test initialize option with a dictionary"""
        self.model.x = Var(self.model.A, initialize={1: 1.3, 2: 2.3})
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1].value, 1.3)
        self.assertEqual(self.instance.x[2].value, 2.3)
        self.instance.x[1] = 1
        self.instance.x[2] = 2
        self.assertEqual(self.instance.x[1].value, 1)
        self.assertEqual(self.instance.x[2].value, 2)

    def test_initialize_with_subdict(self):
        """Test initialize option method with a dictionary of subkeys"""
        self.model.x = Var(self.model.A, initialize={1: 1.3})
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1].value, 1.3)
        self.assertEqual(self.instance.x[2].value, None)
        self.instance.x[1] = 1
        self.instance.x[2] = 2
        self.assertEqual(self.instance.x[1].value, 1)
        self.assertEqual(self.instance.x[2].value, 2)

    def test_initialize_with_const(self):
        """Test initialize option with a constant"""
        self.model.x = Var(self.model.A, initialize=3)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1].value, 3)
        self.assertEqual(self.instance.x[2].value, 3)
        self.instance.x[1] = 1
        self.instance.x[2] = 2
        self.assertEqual(self.instance.x[1].value, 1)
        self.assertEqual(self.instance.x[2].value, 2)

    def test_without_initial_value(self):
        """Test default initial value"""
        self.model.x = Var(self.model.A)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1].value, None)
        self.assertEqual(self.instance.x[2].value, None)
        self.instance.x[1] = 5
        self.instance.x[2] = 6
        self.assertEqual(self.instance.x[1].value, 5)
        self.assertEqual(self.instance.x[2].value, 6)

    def test_bounds_option1(self):
        """Test bounds option"""

        def x_bounds(model, i):
            return (-1.0, 1.0)

        self.model.x = Var(self.model.A, bounds=x_bounds)
        self.instance = self.model.create_instance()
        self.assertEqual(value(self.instance.x[1].lb), -1.0)
        self.assertEqual(value(self.instance.x[1].ub), 1.0)

    def test_bounds_option2(self):
        """Test bounds option"""
        self.model.x = Var(self.model.A, bounds=(-1.0, 1.0))
        self.instance = self.model.create_instance()
        self.assertEqual(value(self.instance.x[1].lb), -1.0)
        self.assertEqual(value(self.instance.x[1].ub), 1.0)

    def test_rule_option(self):
        """Test rule option"""

        def x_init(model, i):
            return 1.3

        self.model.x = Var(self.model.A, initialize=x_init)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1].value, 1.3)

    def test_dim(self):
        """Test dim method"""
        self.model.x = Var(self.model.A)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x.dim(), 1)

    def test_keys(self):
        """Test keys method"""
        self.model.x = Var(self.model.A, dense=False)
        self.model.y = Var(self.model.A, dense=True)
        self.model.z = Var(self.model.A)
        self.instance = self.model.create_instance()
        self.assertEqual(set(self.instance.x.keys()), set())
        self.assertEqual(set(self.instance.y.keys()), set([1, 2]))
        self.assertEqual(set(self.instance.z.keys()), set([1, 2]))

    def test_len(self):
        """Test len method"""
        self.model.x = Var(self.model.A, dense=False)
        self.model.y = Var(self.model.A, dense=True)
        self.model.z = Var(self.model.A)
        self.instance = self.model.create_instance()
        self.assertEqual(len(self.instance.x), 0)
        self.assertEqual(len(self.instance.y), 2)
        self.assertEqual(len(self.instance.z), 2)

    def test_value(self):
        """Check the value of the variable"""
        self.model.x = Var(self.model.A, initialize=3.3)
        self.instance = self.model.create_instance()
        tmp = value(self.instance.x[1].value)
        self.assertEqual(type(tmp), float)
        self.assertEqual(tmp, 3.3)
        tmp = float(self.instance.x[1].value)
        self.assertEqual(type(tmp), float)
        self.assertEqual(tmp, 3.3)
        tmp = int(self.instance.x[1].value)
        self.assertEqual(type(tmp), int)
        self.assertEqual(tmp, 3)

    def test_var_domain_setter(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3])
        self.assertIs(m.x[1].domain, Reals)
        self.assertIs(m.x[2].domain, Reals)
        self.assertIs(m.x[3].domain, Reals)
        m.x.domain = Integers
        self.assertIs(m.x[1].domain, Integers)
        self.assertIs(m.x[2].domain, Integers)
        self.assertIs(m.x[3].domain, Integers)
        m.x.domain = lambda m, i: PositiveReals
        self.assertIs(m.x[1].domain, PositiveReals)
        self.assertIs(m.x[2].domain, PositiveReals)
        self.assertIs(m.x[3].domain, PositiveReals)
        m.x.domain = {1: Reals, 2: NonPositiveReals, 3: NonNegativeReals}
        self.assertIs(m.x[1].domain, Reals)
        self.assertIs(m.x[2].domain, NonPositiveReals)
        self.assertIs(m.x[3].domain, NonNegativeReals)
        m.x.domain = {2: Integers}
        self.assertIs(m.x[1].domain, Reals)
        self.assertIs(m.x[2].domain, Integers)
        self.assertIs(m.x[3].domain, NonNegativeReals)
        with (
            LoggingIntercept() as LOG,
            self.assertRaisesRegex(
                TypeError,
                'Cannot create a Set from data that does not support __contains__.  '
                'Expected set-like object supporting collections.abc.Collection '
                "interface, but received 'NoneType'",
            ),
        ):
            m.x.domain = {1: None, 2: None, 3: None}
        self.assertIn(
            '{1: None, 2: None, 3: None} is not a valid domain. '
            'Variable domains must be an instance of a Pyomo Set or '
            'convertible to a Pyomo Set.',
            LOG.getvalue(),
        )


class TestVarList(PyomoModel):
    def setUp(self):
        #
        # Create Model
        #
        PyomoModel.setUp(self)

    def test_fixed_attr(self):
        """Test fixed attribute"""
        self.model.x = VarList()
        self.model.y = VarList()
        self.instance = self.model.create_instance()
        self.instance.x.add()
        self.instance.x.add()
        self.instance.x.add()
        self.instance.y.add()
        self.instance.y.add()
        self.instance.y.add()
        self.instance.x.fixed = True
        self.assertEqual(self.instance.x[1].fixed, False)
        self.instance.y[1].fixed = True
        self.assertEqual(self.instance.y[1].fixed, True)

    def test_value_attr(self):
        """Test value attribute"""
        self.model.x = VarList()
        self.model.y = VarList()
        self.instance = self.model.create_instance()
        self.instance.x.add()
        self.instance.x.add()
        self.instance.x.add()
        self.instance.y.add()
        self.instance.y.add()
        self.instance.y.add()
        try:
            self.instance.x = 3.5
            self.fail("Expected ValueError")
        except ValueError:
            pass
        self.instance.y[1] = 3.5
        self.assertEqual(self.instance.y[1].value, 3.5)

    def test_initialize_with_function(self):
        """Test initialize option with an initialization rule"""

        def init_rule(model, key):
            i = key + 11
            return key == 1 and 1.3 or 2.3

        self.model.x = VarList(initialize=init_rule)
        self.instance = self.model.create_instance()
        self.instance.x.add()
        self.instance.x.add()
        self.instance.x.add()
        self.assertEqual(self.instance.x[1].value, 1.3)
        self.assertEqual(self.instance.x[2].value, 2.3)
        self.instance.x[1] = 1
        self.instance.x[2] = 2
        self.assertEqual(self.instance.x[1].value, 1)
        self.assertEqual(self.instance.x[2].value, 2)

    def test_initialize_with_dict(self):
        """Test initialize option with a dictionary"""
        self.model.x = VarList(initialize={1: 1.3, 2: 2.3})
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1].value, 1.3)
        self.assertEqual(self.instance.x[2].value, 2.3)
        self.instance.x[1] = 1
        self.instance.x[2] = 2
        self.assertEqual(self.instance.x[1].value, 1)
        self.assertEqual(self.instance.x[2].value, 2)

    def test_initialize_with_bad_dict(self):
        """Test initialize option with a dictionary of subkeys"""
        self.model.x = VarList(initialize={0: 1.3})
        self.assertRaisesRegex(
            KeyError,
            ".*Index '0' is not valid for indexed component 'x'",
            self.model.create_instance,
        )

    def test_initialize_with_const(self):
        """Test initialize option with a constant"""
        self.model.x = VarList(initialize=3)
        self.instance = self.model.create_instance()
        self.instance.x.add()
        self.instance.x.add()
        self.instance.x.add()
        self.assertEqual(self.instance.x[1].value, 3)
        self.assertEqual(self.instance.x[2].value, 3)
        self.instance.x[1] = 1
        self.instance.x[2] = 2
        self.assertEqual(self.instance.x[1].value, 1)
        self.assertEqual(self.instance.x[2].value, 2)

    def test_without_initial_value(self):
        """Test default initialization method"""
        self.model.x = VarList()
        self.instance = self.model.create_instance()
        self.instance.x.add()
        self.instance.x.add()
        self.instance.x.add()
        self.assertEqual(self.instance.x[1].value, None)
        self.assertEqual(self.instance.x[2].value, None)
        self.instance.x[1] = 5
        self.instance.x[2] = 6
        self.assertEqual(self.instance.x[1].value, 5)
        self.assertEqual(self.instance.x[2].value, 6)

    def test_bounds_option1(self):
        """Test bounds option"""

        def x_bounds(model, i):
            return (-1.0, 1.0)

        self.model.x = VarList(bounds=x_bounds)
        self.instance = self.model.create_instance()
        self.instance.x.add()
        self.instance.x.add()
        self.assertEqual(value(self.instance.x[1].lb), -1.0)
        self.assertEqual(value(self.instance.x[1].ub), 1.0)

    def test_bounds_option2(self):
        """Test bounds option"""
        self.model.x = VarList(bounds=(-1.0, 1.0))
        self.instance = self.model.create_instance()
        self.instance.x.add()
        self.instance.x.add()
        self.assertEqual(value(self.instance.x[1].lb), -1.0)
        self.assertEqual(value(self.instance.x[1].ub), 1.0)

    def test_rule_option(self):
        """Test rule option"""

        def x_init(model, i):
            return 1.3

        self.model.x = VarList(initialize=x_init)
        self.instance = self.model.create_instance()
        self.instance.x.add()
        self.instance.x.add()
        self.assertEqual(self.instance.x[1].value, 1.3)

    def test_domain1(self):
        self.model.x = VarList(domain=NonNegativeReals)
        self.instance = self.model.create_instance()
        self.instance.x.add()
        self.instance.x.add()
        self.assertEqual(str(self.instance.x[1].domain), str(NonNegativeReals))
        self.assertEqual(str(self.instance.x[2].domain), str(NonNegativeReals))
        self.instance.x[1].domain = Integers
        self.assertEqual(str(self.instance.x[1].domain), str(Integers))

    def test_domain2(self):
        def x_domain(model, i):
            if i == 1:
                return NonNegativeReals
            elif i == 2:
                return Reals
            elif i == 3:
                return Integers

        self.model.x = VarList(domain=x_domain)
        self.instance = self.model.create_instance()
        self.instance.x.add()
        self.instance.x.add()
        self.instance.x.add()
        self.assertEqual(str(self.instance.x[1].domain), str(NonNegativeReals))
        self.assertEqual(str(self.instance.x[2].domain), str(Reals))
        self.assertEqual(str(self.instance.x[3].domain), str(Integers))
        try:
            self.instance.x.domain
        except AttributeError:
            pass
        # test the property setter
        self.instance.x.domain = Binary
        self.assertEqual(str(self.instance.x[1].domain), str(Binary))
        self.assertEqual(str(self.instance.x[2].domain), str(Binary))
        self.assertEqual(str(self.instance.x[3].domain), str(Binary))

    # VarList doesn't handle generators yet
    @unittest.expectedFailure
    def test_domain3(self):
        def x_domain(model):
            yield NonNegativeReals
            yield Reals
            yield Integers

        self.model.x = VarList(domain=x_domain)
        self.instance = self.model.create_instance()
        self.instance.x.add()
        self.instance.x.add()
        self.instance.x.add()
        self.assertEqual(self.instance.x.domain, None)
        self.assertEqual(str(self.instance.x[0].domain), str(NonNegativeReals))
        self.assertEqual(str(self.instance.x[1].domain), str(Reals))
        self.assertEqual(str(self.instance.x[2].domain), str(Integers))
        try:
            self.instance.x.domain = Reals
        except AttributeError:
            pass
        self.assertEqual(self.instance.x.domain, None)

    def test_dim(self):
        """Test dim method"""
        self.model.x = VarList()
        self.instance = self.model.create_instance()
        self.instance.x.add()
        self.assertEqual(self.instance.x.dim(), 1)

    def test_keys(self):
        """Test keys method"""
        self.model.x = VarList()
        self.instance = self.model.create_instance()
        self.instance.x.add()
        self.instance.x.add()
        self.assertEqual(set(self.instance.x.keys()), set([1, 2]))

    def test_len(self):
        """Test len method"""
        self.model.x = VarList()
        self.instance = self.model.create_instance()
        self.instance.x.add()
        self.instance.x.add()
        self.assertEqual(len(self.instance.x), 2)

    def test_value(self):
        """Check the value of the variable"""
        self.model.x = VarList(initialize=3.3)
        self.instance = self.model.create_instance()
        self.instance.x.add()
        self.instance.x.add()
        tmp = value(self.instance.x[1].value)
        self.assertEqual(type(tmp), float)
        self.assertEqual(tmp, 3.3)
        tmp = float(self.instance.x[1].value)
        self.assertEqual(type(tmp), float)
        self.assertEqual(tmp, 3.3)
        tmp = int(self.instance.x[1].value)
        self.assertEqual(type(tmp), int)
        self.assertEqual(tmp, 3)

    def test_0based_add(self):
        m = ConcreteModel()
        m.x = VarList(starting_index=0)
        m.x.add()
        self.assertEqual(list(m.x.keys()), [0])
        m.x.add()
        self.assertEqual(list(m.x.keys()), [0, 1])
        m.x.add()
        self.assertEqual(list(m.x.keys()), [0, 1, 2])

    def test_0based_initialize_with_dict(self):
        """Test initialize option with a dictionary"""
        self.model.x = VarList(initialize={1: 1.3, 2: 2.3}, starting_index=0)
        self.assertRaisesRegex(
            KeyError,
            ".*Index '2' is not valid for indexed component 'x'",
            self.model.create_instance,
        )

    def test_0based_initialize_with_bad_dict(self):
        """Test initialize option with a dictionary of subkeys"""
        self.model.x = VarList(initialize={0: 1.3, 1: 2.3}, starting_index=0)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[0].value, 1.3)
        self.assertEqual(self.instance.x[1].value, 2.3)
        self.instance.x[0] = 1
        self.instance.x[1] = 2
        self.assertEqual(self.instance.x[0].value, 1)
        self.assertEqual(self.instance.x[1].value, 2)
        self.instance.x.add()
        self.assertEqual(list(self.instance.x.keys()), [0, 1, 2])


class Test2DArrayVar(TestSimpleVar):
    def setUp(self):
        #
        # Create Model
        #
        PyomoModel.setUp(self)
        self.model.A = Set(initialize=[1, 2])

    def test_fixed_attr(self):
        """Test fixed attribute"""
        self.model.x = Var(self.model.A, self.model.A)
        self.model.y = Var(self.model.A, self.model.A)
        self.instance = self.model.create_instance()
        self.instance.x.fixed = True
        self.assertEqual(self.instance.x[1, 2].fixed, False)
        self.instance.y[1, 2].fixed = True
        self.assertEqual(self.instance.y[1, 2].fixed, True)

    def test_value_attr(self):
        """Test value attribute"""
        self.model.x = Var(self.model.A, self.model.A, dense=True)
        self.model.y = Var(self.model.A, self.model.A, dense=True)
        self.instance = self.model.create_instance()
        try:
            self.instance.x = 3.5
            self.fail("Expected ValueError")
        except ValueError:
            pass
        self.instance.y[1, 2] = 3.5
        self.assertEqual(self.instance.y[1, 2].value, 3.5)

    # def test_lb_attr(self):
    # """Test lb attribute"""
    # self.model.x = Var(self.model.A,self.model.A)
    # self.instance = self.model.create_instance()
    # self.instance.x.setlb(-1.0)
    # self.assertEqual(value(self.instance.x[2,1].lb), -1.0)

    # def test_ub_attr(self):
    # """Test ub attribute"""
    # self.model.x = Var(self.model.A,self.model.A)
    # self.instance = self.model.create_instance()
    # self.instance.x.setub(1.0)
    # self.assertEqual(value(self.instance.x[2,1].ub), 1.0)

    def test_initialize_with_function(self):
        """Test initialize option with an initialization rule"""

        def init_rule(model, key1, key2):
            i = key1 + 1
            return key1 == 1 and 1.3 or 2.3

        self.model.x = Var(self.model.A, self.model.A, initialize=init_rule)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1, 1].value, 1.3)
        self.assertEqual(self.instance.x[2, 2].value, 2.3)
        self.instance.x[1, 1] = 1
        self.instance.x[2, 2] = 2
        self.assertEqual(self.instance.x[1, 1].value, 1)
        self.assertEqual(self.instance.x[2, 2].value, 2)

    def test_initialize_with_dict(self):
        """Test initialize option with a dictionary"""
        self.model.x = Var(
            self.model.A, self.model.A, initialize={(1, 1): 1.3, (2, 2): 2.3}
        )
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1, 1].value, 1.3)
        self.assertEqual(self.instance.x[2, 2].value, 2.3)
        self.instance.x[1, 1] = 1
        self.instance.x[2, 2] = 2
        self.assertEqual(self.instance.x[1, 1].value, 1)
        self.assertEqual(self.instance.x[2, 2].value, 2)

    def test_initialize_with_const(self):
        """Test initialize option with a constant"""
        self.model.x = Var(self.model.A, self.model.A, initialize=3)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1, 1].value, 3)
        self.assertEqual(self.instance.x[2, 2].value, 3)
        self.instance.x[1, 1] = 1
        self.instance.x[2, 2] = 2
        self.assertEqual(self.instance.x[1, 1].value, 1)
        self.assertEqual(self.instance.x[2, 2].value, 2)

    def test_without_initial_value(self):
        """Test default initialization"""
        self.model.x = Var(self.model.A, self.model.A)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1, 1].value, None)
        self.assertEqual(self.instance.x[2, 2].value, None)
        self.instance.x[1, 1] = 5
        self.instance.x[2, 2] = 6
        self.assertEqual(self.instance.x[1, 1].value, 5)
        self.assertEqual(self.instance.x[2, 2].value, 6)

    def test_initialize_option(self):
        """Test initialize option"""
        self.model.x = Var(
            self.model.A, self.model.A, initialize={(1, 1): 1.3, (2, 2): 2.3}
        )
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1, 1].value, 1.3)
        self.assertEqual(self.instance.x[2, 2].value, 2.3)
        try:
            value(self.instance.x[1, 2])
            self.fail("Expected ValueError")
        except ValueError:
            pass

    def test_bounds_option1(self):
        """Test bounds option"""

        def x_bounds(model, i, j):
            return (-1.0 * (i + j), 1.0 * (i + j))

        self.model.x = Var(self.model.A, self.model.A, bounds=x_bounds)
        self.instance = self.model.create_instance()
        self.assertEqual(value(self.instance.x[1, 1].lb), -2.0)
        self.assertEqual(value(self.instance.x[1, 2].ub), 3.0)

    def test_bounds_option2(self):
        """Test bounds option"""
        self.model.x = Var(self.model.A, self.model.A, bounds=(-1.0, 1.0))
        self.instance = self.model.create_instance()
        self.assertEqual(value(self.instance.x[1, 1].lb), -1.0)
        self.assertEqual(value(self.instance.x[1, 1].ub), 1.0)

    def test_rule_option(self):
        """Test rule option"""

        def x_init(model, i, j):
            return 1.3

        self.model.x = Var(self.model.A, self.model.A, initialize=x_init)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1, 2].value, 1.3)

    def test_dim(self):
        """Test dim method"""
        self.model.x = Var(self.model.A, self.model.A)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x.dim(), 2)

    def test_keys(self):
        """Test keys method"""
        self.model.x = Var(self.model.A, self.model.A, dense=True)
        self.instance = self.model.create_instance()
        ans = [(1, 1), (1, 2), (2, 1), (2, 2)]
        self.assertEqual(list(sorted(self.instance.x.keys())), ans)

    def test_len(self):
        """Test len method"""
        self.model.x = Var(self.model.A, self.model.A, dense=True)
        self.instance = self.model.create_instance()
        self.assertEqual(len(self.instance.x), 4)

    def test_value(self):
        """Check the value of the variable"""
        self.model.x = Var(self.model.A, self.model.A, initialize=3.3)
        self.instance = self.model.create_instance()
        tmp = value(self.instance.x[1, 1].value)
        self.assertEqual(type(tmp), float)
        self.assertEqual(tmp, 3.3)
        tmp = float(self.instance.x[1, 1].value)
        self.assertEqual(type(tmp), float)
        self.assertEqual(tmp, 3.3)
        tmp = int(self.instance.x[1, 1].value)
        self.assertEqual(type(tmp), int)
        self.assertEqual(tmp, 3)


class TestVarComplexArray(PyomoModel):
    def test_index1(self):
        self.model.A = Set(initialize=range(0, 4))

        def B_index(model):
            for i in model.A:
                if i % 2 == 0:
                    yield i

        def B_init(model, i, j):
            if j:
                return 2 + i
            return -(2 + i)

        self.model.B = Var(B_index, [True, False], initialize=B_init, dense=True)
        self.instance = self.model.create_instance()
        self.assertEqual(
            set(self.instance.B.keys()),
            set([(0, True), (2, True), (0, False), (2, False)]),
        )
        self.assertEqual(self.instance.B[0, True].value, 2)
        self.assertEqual(self.instance.B[0, False].value, -2)
        self.assertEqual(self.instance.B[2, True].value, 4)
        self.assertEqual(self.instance.B[2, False].value, -4)

    def test_index2(self):
        self.model.A = Set(initialize=range(0, 4))

        def B_index(model):
            for i in model.A:
                if i % 2 == 0:
                    yield i - 1, i

        B_index.dimen = 2

        def B_init(model, k, i, j):
            if j:
                return (2 + i) * k
            return -(2 + i) * k

        self.model.B = Var(B_index, [True, False], initialize=B_init, dense=True)
        self.instance = self.model.create_instance()
        # self.instance.pprint()
        self.assertEqual(
            set(self.instance.B.keys()),
            set([(-1, 0, True), (1, 2, True), (-1, 0, False), (1, 2, False)]),
        )
        self.assertEqual(self.instance.B[-1, 0, True].value, -2)
        self.assertEqual(self.instance.B[-1, 0, False].value, 2)
        self.assertEqual(self.instance.B[1, 2, True].value, 4)
        self.assertEqual(self.instance.B[1, 2, False].value, -4)


class MiscVarTests(unittest.TestCase):
    def test_error1(self):
        a = Var(name="a")
        try:
            a = Var(foo=1)
            self.fail("test_error1")
        except ValueError:
            pass

    def test_getattr1(self):
        """
        Verify the behavior of non-standard suffixes with simple variable
        """
        model = AbstractModel()
        model.a = Var()
        model.suffix = Suffix(datatype=Suffix.INT)
        instance = model.create_instance()
        self.assertEqual(instance.suffix.get(instance.a), None)
        instance.suffix.set_value(instance.a, True)
        self.assertEqual(instance.suffix.get(instance.a), True)

    def test_getattr2(self):
        """
        Verify the behavior of non-standard suffixes with an array of variables
        """
        model = AbstractModel()
        model.X = Set(initialize=[1, 3, 5])
        model.a = Var(model.X)
        model.suffix = Suffix(datatype=Suffix.INT)
        try:
            self.assertEqual(model.a.suffix, None)
            self.fail("Expected AttributeError")
        except AttributeError:
            pass
        instance = model.create_instance()
        self.assertEqual(instance.suffix.get(instance.a[1]), None)
        instance.suffix.set_value(instance.a[1], True)
        self.assertEqual(instance.suffix.get(instance.a[1]), True)

    def test_error2(self):
        try:
            model = AbstractModel()
            model.a = Var(initialize=[1, 2, 3])
            model.b = Var(model.a)
            self.fail("test_error2")
        except TypeError:
            pass

    def test_contains(self):
        model = AbstractModel()
        model.a = Set(initialize=[1, 2, 3])
        model.b = Var(model.a, dense=True)
        instance = model.create_instance()
        self.assertEqual(1 in instance.b, True)

    def test_float_int(self):
        model = AbstractModel()
        model.a = Set(initialize=[1, 2, 3])
        model.b = Var(model.a, initialize=1.1)
        model.c = Var(initialize=2.1)
        model.d = Var()
        instance = model.create_instance()
        self.assertEqual(float(value(instance.b[1])), 1.1)
        self.assertEqual(int(value(instance.b[1])), 1)
        self.assertEqual(float(value(instance.c)), 2.1)
        self.assertEqual(int(value(instance.c)), 2)
        try:
            float(instance.d)
            self.fail("expected TypeError")
        except TypeError:
            pass
        try:
            int(instance.d)
            self.fail("expected TypeError")
        except TypeError:
            pass
        try:
            float(instance.b)
            self.fail("expected TypeError")
        except TypeError:
            pass
        try:
            int(instance.b)
            self.fail("expected TypeError")
        except TypeError:
            pass

    def test_set_get(self):
        model = AbstractModel()
        model.a = Set(initialize=[1, 2, 3])
        model.b = Var(model.a, initialize=1.1, within=PositiveReals)
        model.c = Var(initialize=2.1, within=PositiveReals, bounds=(1, 10))
        with self.assertRaisesRegex(
            ValueError,
            "Cannot set the value for the indexed "
            "component 'b' without specifying an index value",
        ):
            model.b = 2.2

        instance = model.create_instance()
        with self.assertRaisesRegex(
            KeyError, "Cannot treat the scalar component 'c' as an indexed component"
        ):
            instance.c[1] = 2.2

        instance.b[1] = 2.2
        with self.assertRaisesRegex(
            KeyError, "Index '4' is not valid for indexed component 'b'"
        ):
            instance.b[4] = 2.2

        with LoggingIntercept() as LOG:
            instance.b[3] = -2.2
        self.assertEqual(
            LOG.getvalue().strip(),
            "Setting Var 'b[3]' to a value `-2.2` (float) "
            "not in domain PositiveReals.",
        )

        with self.assertRaisesRegex(
            KeyError, "Cannot treat the scalar component 'c' as an indexed component"
        ):
            tmp = instance.c[3]
        with LoggingIntercept() as LOG:
            instance.c = 'a'
        self.assertEqual(
            LOG.getvalue().strip(),
            "Setting Var 'c' to a value `a` (str) not in domain PositiveReals.",
        )
        with LoggingIntercept() as LOG:
            instance.c = -2.2
        self.assertEqual(
            LOG.getvalue().strip(),
            "Setting Var 'c' to a value `-2.2` (float) not in domain PositiveReals.",
        )
        with LoggingIntercept() as LOG:
            instance.c = 11
        self.assertEqual(
            LOG.getvalue().strip(),
            "Setting Var 'c' to a numeric value `11` outside the bounds (1, 10).",
        )

        with LoggingIntercept() as LOG:
            instance.c.set_value('a', skip_validation=True)
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(instance.c.value, 'a')
        with LoggingIntercept() as LOG:
            instance.c.set_value(-1, skip_validation=True)
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(instance.c.value, -1)

        # try:
        # instance.c.ub = 'a'
        # self.fail("can't set a bad ub for variable c")
        # except ValueError:
        # pass
        # try:
        # instance.c.ub = -1.0
        # self.fail("can't set a bad ub for variable c")
        # except ValueError:
        # pass

        # try:
        # instance.c.fixed = 'a'
        # self.fail("can't fix a variable with a non-boolean")
        # except ValueError:
        # pass

    def test_set_index(self):
        model = ConcreteModel()
        model.s = Set(initialize=[1, 2, 3])
        model.x = Var(model.s, initialize=0, dense=True)

        # test proper instantiation
        self.assertEqual(len(model.x), 3)
        for i in model.s:
            self.assertEqual(value(model.x[i]), 0)

        # test mutability of index set
        model.s.add(4)
        self.assertEqual(len(model.x), 3)
        for i in model.s:
            self.assertEqual(value(model.x[i]), 0)
        self.assertEqual(len(model.x), 4)

    def test_simple_default_domain(self):
        model = ConcreteModel()
        model.x = Var()
        self.assertIs(model.x.domain, Reals)

    def test_simple_nondefault_domain_value(self):
        model = ConcreteModel()
        model.x = Var(domain=Integers)
        self.assertIs(model.x.domain, Integers)

    def test_simple_bad_nondefault_domain_value(self):
        model = ConcreteModel()
        with self.assertRaises(TypeError):
            model.x = Var(domain=25)

    def test_simple_nondefault_domain_rule(self):
        model = ConcreteModel()
        model.x = Var(domain=lambda m: Integers)
        self.assertIs(model.x.domain, Integers)

    def test_simple_bad_nondefault_domain_rule(self):
        model = ConcreteModel()
        with self.assertRaises(TypeError):
            model.x = Var(domain=lambda m: 25)

    def test_indexed_default_domain(self):
        model = ConcreteModel()
        model.s = Set(initialize=[1])
        model.x = Var(model.s)
        self.assertIs(model.x[1].domain, Reals)

    def test_indexed_nondefault_domain_value(self):
        model = ConcreteModel()
        model.s = Set(initialize=[1])
        model.x = Var(model.s, domain=Integers)
        self.assertIs(model.x[1].domain, Integers)

    def test_indexed_bad_nondefault_domain_value(self):
        model = ConcreteModel()
        model.s = Set(initialize=[1])
        with self.assertRaises(TypeError):
            model.x = Var(model.s, domain=25)

    def test_indexed_nondefault_domain_rule(self):
        model = ConcreteModel()
        model.s = Set(initialize=[1])
        model.x = Var(model.s, domain=lambda m, i: Integers)
        self.assertIs(model.x[1].domain, Integers)

    def test_indexed_bad_nondefault_domain_rule(self):
        model = ConcreteModel()
        model.s = Set(initialize=[1])
        with self.assertRaises(TypeError):
            model.x = Var(model.s, domain=lambda m, i: 25)

    def test_list_default_domain(self):
        model = ConcreteModel()
        model.x = VarList()
        model.x.add()
        self.assertIs(model.x[1].domain, Reals)

    def test_list_nondefault_domain_value(self):
        model = ConcreteModel()
        model.x = VarList(domain=Integers)
        model.x.add()
        self.assertIs(model.x[1].domain, Integers)

    def test_list_bad_nondefault_domain_value(self):
        model = ConcreteModel()
        model.x = VarList(domain=25)
        with self.assertRaises(TypeError):
            model.x.add()

    def test_list_nondefault_domain_rule(self):
        model = ConcreteModel()
        model.x = VarList(domain=lambda m, i: Integers)
        model.x.add()
        self.assertIs(model.x[1].domain, Integers)

    def test_list_bad_nondefault_domain_rule(self):
        model = ConcreteModel()
        model.x = VarList(domain=lambda m, i: 25)
        with self.assertRaises(TypeError):
            model.x.add()

    def test_setdata_index(self):
        model = ConcreteModel()
        model.sindex = Set(initialize=[1])
        model.s = Set(model.sindex, initialize=[1, 2, 3])
        model.x = Var(model.s[1], initialize=0, dense=True)

        # test proper instantiation
        self.assertEqual(len(model.x), 3)
        for i in model.s[1]:
            self.assertEqual(value(model.x[i]), 0)

        # test mutability of index set
        newIdx = 4
        self.assertFalse(newIdx in model.s[1])
        self.assertFalse(newIdx in model.x)

        model.s[1].add(newIdx)
        self.assertTrue(newIdx in model.s[1])
        self.assertFalse(newIdx in model.x)

        self.assertEqual(len(model.x), 3)
        for i in model.s[1]:
            self.assertEqual(value(model.x[i]), 0)
        self.assertEqual(len(model.x), 4)

        self.assertTrue(newIdx in model.s[1])
        self.assertTrue(newIdx in model.x)

    def test_setdata_multidimen_index(self):
        model = ConcreteModel()
        model.sindex = Set(initialize=[1])
        model.s = Set(model.sindex, dimen=2, initialize=[(1, 1), (1, 2), (1, 3)])
        model.x = Var(model.s[1], initialize=0, dense=True)

        # test proper instantiation
        self.assertEqual(len(model.x), 3)
        for i in model.s[1]:
            self.assertEqual(value(model.x[i]), 0)

        # test mutability of index set
        newIdx = (1, 4)
        self.assertFalse(newIdx in model.s[1])
        self.assertFalse(newIdx in model.x)

        model.s[1].add(newIdx)
        self.assertTrue(newIdx in model.s[1])
        self.assertFalse(newIdx in model.x)

        self.assertEqual(len(model.x), 3)
        for i in model.s[1]:
            self.assertEqual(value(model.x[i]), 0)
        self.assertEqual(len(model.x), 4)

        self.assertTrue(newIdx in model.s[1])
        self.assertTrue(newIdx in model.x)

    def test_abstract_index(self):
        model = AbstractModel()
        model.A = Set()
        model.B = Set()
        model.C = model.A | model.B
        model.x = Var(model.C)

    @unittest.skipUnless(pint_available, "units test requires pint module")
    def test_set_value_units(self):
        m = ConcreteModel()
        m.x = Var(units=units.g)
        m.x = 5
        self.assertEqual(value(m.x), 5)
        m.x = 6 * units.g
        self.assertEqual(value(m.x), 6)
        m.x = None
        self.assertIsNone(m.x.value, None)
        m.x = 7 * units.kg
        self.assertEqual(value(m.x), 7000)
        with self.assertRaises(UnitsError):
            m.x = 1 * units.s

        out = StringIO()
        m.pprint(ostream=out)
        self.assertEqual(
            out.getvalue().strip(),
            """
1 Var Declarations
    x : Size=1, Index=None, Units=g
        Key  : Lower : Value  : Upper : Fixed : Stale : Domain
        None :  None : 7000.0 :  None : False : False :  Reals

1 Declarations: x
        """.strip(),
        )

    @unittest.skipUnless(pint_available, "units test requires pint module")
    def test_set_bounds_units(self):
        m = ConcreteModel()
        m.x = Var(units=units.g)
        m.p = Param(mutable=True, initialize=1, units=units.kg)
        m.x.setlb(5)
        self.assertEqual(m.x.lb, 5)
        m.x.setlb(6 * units.g)
        self.assertEqual(m.x.lb, 6)
        m.x.setlb(7 * units.kg)
        self.assertEqual(m.x.lb, 7000)
        with self.assertRaises(UnitsError):
            m.x.setlb(1 * units.s)
        m.x.setlb(m.p)
        self.assertEqual(m.x.lb, 1000)
        m.p = 2 * units.kg
        self.assertEqual(m.x.lb, 2000)

        m.x.setub(2)
        self.assertEqual(m.x.ub, 2)
        m.x.setub(3 * units.g)
        self.assertEqual(m.x.ub, 3)
        m.x.setub(4 * units.kg)
        self.assertEqual(m.x.ub, 4000)
        with self.assertRaises(UnitsError):
            m.x.setub(1 * units.s)
        m.x.setub(m.p)
        self.assertEqual(m.x.ub, 2000)
        m.p = 3 * units.kg
        self.assertEqual(m.x.ub, 3000)

    def test_stale(self):
        m = ConcreteModel()
        m.x = Var(initialize=0)
        self.assertFalse(m.x.stale)
        m.y = Var()
        self.assertTrue(m.y.stale)

        StaleFlagManager.mark_all_as_stale(delayed=False)
        self.assertTrue(m.x.stale)
        self.assertTrue(m.y.stale)
        m.x = 1
        self.assertFalse(m.x.stale)
        self.assertTrue(m.y.stale)
        m.y = 2
        self.assertFalse(m.x.stale)
        self.assertFalse(m.y.stale)

        StaleFlagManager.mark_all_as_stale(delayed=True)
        self.assertFalse(m.x.stale)
        self.assertFalse(m.y.stale)
        m.x = 1
        self.assertFalse(m.x.stale)
        self.assertTrue(m.y.stale)
        m.y = 2
        self.assertFalse(m.x.stale)
        self.assertFalse(m.y.stale)

    def test_stale_clone(self):
        m = ConcreteModel()
        m.x = Var(initialize=0)
        self.assertFalse(m.x.stale)
        m.y = Var()
        self.assertTrue(m.y.stale)
        m.z = Var(initialize=0)
        self.assertFalse(m.z.stale)

        i = m.clone()
        self.assertFalse(i.x.stale)
        self.assertTrue(i.y.stale)
        self.assertFalse(i.z.stale)

        StaleFlagManager.mark_all_as_stale(delayed=True)
        m.z = 5
        i = m.clone()
        self.assertTrue(i.x.stale)
        self.assertTrue(i.y.stale)
        self.assertFalse(i.z.stale)

    def test_domain_categories(self):
        """Test domain attribute"""
        x = Var()
        x.construct()
        self.assertEqual(x.is_integer(), False)
        self.assertEqual(x.is_binary(), False)
        self.assertEqual(x.is_continuous(), True)
        self.assertEqual(x.bounds, (None, None))
        x.domain = Integers
        self.assertEqual(x.is_integer(), True)
        self.assertEqual(x.is_binary(), False)
        self.assertEqual(x.is_continuous(), False)
        self.assertEqual(x.bounds, (None, None))
        x.domain = Binary
        self.assertEqual(x.is_integer(), True)
        self.assertEqual(x.is_binary(), True)
        self.assertEqual(x.is_continuous(), False)
        self.assertEqual(x.bounds, (0, 1))
        x.domain = RangeSet(0, 10, 0)
        self.assertEqual(x.is_integer(), False)
        self.assertEqual(x.is_binary(), False)
        self.assertEqual(x.is_continuous(), True)
        self.assertEqual(x.bounds, (0, 10))
        x.domain = RangeSet(0, 10, 1)
        self.assertEqual(x.is_integer(), True)
        self.assertEqual(x.is_binary(), False)
        self.assertEqual(x.is_continuous(), False)
        self.assertEqual(x.bounds, (0, 10))
        x.domain = RangeSet(0.5, 10, 1)
        self.assertEqual(x.is_integer(), False)
        self.assertEqual(x.is_binary(), False)
        self.assertEqual(x.is_continuous(), False)
        self.assertEqual(x.bounds, (0.5, 9.5))
        x.domain = RangeSet(0, 1, 1)
        self.assertEqual(x.is_integer(), True)
        self.assertEqual(x.is_binary(), True)
        self.assertEqual(x.is_continuous(), False)
        self.assertEqual(x.bounds, (0, 1))


if __name__ == "__main__":
    unittest.main()
