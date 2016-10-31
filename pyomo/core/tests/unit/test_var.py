#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Unit Tests for Elements of a Model
#
# TestSimpleVar                Class for testing single variables
# TestArrayVar                Class for testing array of variables
#

import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest

from pyomo.core.base import IntegerSet
from pyomo.environ import *

class PyomoModel(unittest.TestCase):

    def setUp(self):
        self.model = AbstractModel()
        self.instance = None

    def tearDown(self):
        self.model = None
        self.instance = None

    def construct(self,filename=None):
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
        self.instance.y[1].fix(1)
        self.assertEqual(self.instance.y[1].value, 1)
        self.assertEqual(self.instance.y[1].fixed, True)
        self.instance.y[1].unfix()
        self.assertEqual(self.instance.y[1].value, 1)
        self.assertEqual(self.instance.y[1].fixed, False)
        self.instance.y[1].fix(None)
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
        self.model.x = Var(within=NonNegativeReals, bounds=(-1,2))
        self.instance = self.model.create_instance()
        self.assertEqual(value(self.instance.x.lb), 0.0)
        self.assertEqual(value(self.instance.x.ub), 2.0)

    def test_lb_attr3(self):
        """Test lb attribute"""
        self.model.p = Param(mutable=True, initialize=1)
        self.model.x = Var(within=NonNegativeReals, bounds=(self.model.p,None))
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
        self.model.x = Var(within=NonPositiveReals, bounds=(-2,1))
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
            return (-1.0,1.0)
        self.model.x = Var(bounds=x_bounds)
        self.instance = self.model.create_instance()
        self.assertEqual(value(self.instance.x.lb), -1.0)
        self.assertEqual(value(self.instance.x.ub), 1.0)

    def test_bounds_option2(self):
        """Test bounds option"""
        self.model.x = Var(bounds=(-1.0,1.0))
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
        self.assertEqual(self.instance.x, 1.3)
        self.instance.x = 1
        self.assertEqual(self.instance.x, 1)

    def test_initialize_with_dict(self):
        """Test initialize option with a dictionary"""
        self.model.x = Var(initialize={None:1.3})
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x, 1.3)
        self.instance.x = 1
        self.assertEqual(self.instance.x, 1)

    def test_initialize_with_const(self):
        """Test initialize option with a constant"""
        self.model.x = Var(initialize=1.3)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x, 1.3)
        self.instance.x = 1
        self.assertEqual(self.instance.x, 1)

    def test_without_initial_value(self):
        """Test default initial value"""
        self.model.x = Var()
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x.value,None)
        self.instance.x = 6
        self.assertEqual(self.instance.x.value,6)

    def test_dim(self):
        """Test dim method"""
        self.model.x = Var()
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x.dim(),0)

    def test_keys(self):
        """Test keys method"""
        self.model.x = Var()
        self.instance = self.model.create_instance()
        self.assertEqual(list(self.instance.x.keys()),[None])
        self.assertEqual(id(self.instance.x),id(self.instance.x[None]))

    def test_len(self):
        """Test len method"""
        self.model.x = Var()
        self.assertEqual(len(self.model.x),0)
        self.instance = self.model.create_instance()
        self.assertEqual(len(self.instance.x),1)

    def test_value(self):
        """Check the value of the variable"""
        self.model.x = Var(initialize=3.3)
        self.instance = self.model.create_instance()
        tmp = value(self.instance.x.value)
        self.assertEqual( type(tmp), float)
        self.assertEqual( tmp, 3.3 )
        tmp = float(self.instance.x.value)
        self.assertEqual( type(tmp), float)
        self.assertEqual( tmp, 3.3 )
        tmp = int(self.instance.x.value)
        self.assertEqual( type(tmp), int)
        self.assertEqual( tmp, 3 )


class TestArrayVar(TestSimpleVar):

    def setUp(self):
        #
        # Create Model
        #
        PyomoModel.setUp(self)
        self.model.A = Set(initialize=[1,2])

    def test_fixed_attr(self):
        """Test fixed attribute"""
        self.model.x = Var(self.model.A)
        self.model.y = Var(self.model.A)
        self.instance = self.model.create_instance()
        self.instance.x.fixed = True
        self.assertEqual(self.instance.x[1].fixed, False)
        self.instance.y[1].fixed=True
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
        self.assertEqual(self.instance.y[1], 3.5)

    #def test_lb_attr(self):
        #"""Test lb attribute"""
        #self.model.x = Var(self.model.A)
        #self.instance = self.model.create_instance()
        #self.instance.x.setlb(-1.0)
        #self.assertEqual(value(self.instance.x[1].lb), -1.0)

    #def test_ub_attr(self):
        #"""Test ub attribute"""
        #self.model.x = Var(self.model.A)
        #self.instance = self.model.create_instance()
        #self.instance.x.setub(1.0)
        #self.assertEqual(value(self.instance.x[1].ub), 1.0)

    def test_initialize_with_function(self):
        """Test initialize option with an initialization rule"""
        def init_rule(model, key):
            i = key+11
            return key == 1 and 1.3 or 2.3
        self.model.x = Var(self.model.A,initialize=init_rule)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1].value, 1.3)
        self.assertEqual(self.instance.x[2].value, 2.3)
        self.instance.x[1] = 1
        self.instance.x[2] = 2
        self.assertEqual(self.instance.x[1], 1)
        self.assertEqual(self.instance.x[2], 2)

    def test_initialize_with_dict(self):
        """Test initialize option with a dictionary"""
        self.model.x = Var(self.model.A,initialize={1:1.3,2:2.3})
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1], 1.3)
        self.assertEqual(self.instance.x[2], 2.3)
        self.instance.x[1] = 1
        self.instance.x[2] = 2
        self.assertEqual(self.instance.x[1], 1)
        self.assertEqual(self.instance.x[2], 2)

    def test_initialize_with_subdict(self):
        """Test initialize option method with a dictionary of subkeys"""
        self.model.x = Var(self.model.A,initialize={1:1.3})
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1], 1.3)
        self.assertEqual(self.instance.x[2].value, None)
        self.instance.x[1] = 1
        self.instance.x[2] = 2
        self.assertEqual(self.instance.x[1], 1)
        self.assertEqual(self.instance.x[2], 2)

    def test_initialize_with_const(self):
        """Test initialize option with a constant"""
        self.model.x = Var(self.model.A,initialize=3)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1], 3)
        self.assertEqual(self.instance.x[2], 3)
        self.instance.x[1] = 1
        self.instance.x[2] = 2
        self.assertEqual(self.instance.x[1], 1)
        self.assertEqual(self.instance.x[2], 2)

    def test_without_initial_value(self):
        """Test default initial value"""
        self.model.x = Var(self.model.A)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1].value,None)
        self.assertEqual(self.instance.x[2].value,None)
        self.instance.x[1] = 5
        self.instance.x[2] = 6
        self.assertEqual(self.instance.x[1].value,5)
        self.assertEqual(self.instance.x[2].value,6)

    def test_bounds_option1(self):
        """Test bounds option"""
        def x_bounds(model, i):
            return (-1.0,1.0)
        self.model.x = Var(self.model.A, bounds=x_bounds)
        self.instance = self.model.create_instance()
        self.assertEqual(value(self.instance.x[1].lb), -1.0)
        self.assertEqual(value(self.instance.x[1].ub), 1.0)

    def test_bounds_option2(self):
        """Test bounds option"""
        self.model.x = Var(self.model.A, bounds=(-1.0,1.0))
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
        self.assertEqual(self.instance.x.dim(),1)

    def test_keys(self):
        """Test keys method"""
        self.model.x = Var(self.model.A, dense=False)
        self.model.y = Var(self.model.A, dense=True)
        self.model.z = Var(self.model.A)
        self.instance = self.model.create_instance()
        self.assertEqual(set(self.instance.x.keys()),set())
        self.assertEqual(set(self.instance.y.keys()),set([1,2]))
        self.assertEqual(set(self.instance.z.keys()),set([1,2]))

    def test_len(self):
        """Test len method"""
        self.model.x = Var(self.model.A, dense=False)
        self.model.y = Var(self.model.A, dense=True)
        self.model.z = Var(self.model.A)
        self.instance = self.model.create_instance()
        self.assertEqual(len(self.instance.x),0)
        self.assertEqual(len(self.instance.y),2)
        self.assertEqual(len(self.instance.z),2)

    def test_value(self):
        """Check the value of the variable"""
        self.model.x = Var(self.model.A,initialize=3.3)
        self.instance = self.model.create_instance()
        tmp = value(self.instance.x[1].value)
        self.assertEqual( type(tmp), float)
        self.assertEqual( tmp, 3.3 )
        tmp = float(self.instance.x[1].value)
        self.assertEqual( type(tmp), float)
        self.assertEqual( tmp, 3.3 )
        tmp = int(self.instance.x[1].value)
        self.assertEqual( type(tmp), int)
        self.assertEqual( tmp, 3 )


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
        self.instance.y[1].fixed=True
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
        self.assertEqual(self.instance.y[1], 3.5)

    def test_initialize_with_function(self):
        """Test initialize option with an initialization rule"""
        def init_rule(model, key):
            i = key+11
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
        self.assertEqual(self.instance.x[1], 1)
        self.assertEqual(self.instance.x[2], 2)

    def test_initialize_with_dict(self):
        """Test initialize option with a dictionary"""
        self.model.x = VarList(initialize={1:1.3,2:2.3})
        self.model.x.add()
        self.model.x.add()
        self.model.x.add()
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1], 1.3)
        self.assertEqual(self.instance.x[2], 2.3)
        self.instance.x[1] = 1
        self.instance.x[2] = 2
        self.assertEqual(self.instance.x[1], 1)
        self.assertEqual(self.instance.x[2], 2)

    def test_initialize_with_subdict(self):
        """Test initialize option with a dictionary of subkeys"""
        self.model.x = VarList(initialize={1:1.3})
        self.model.x.add()
        self.model.x.add()
        self.model.x.add()
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1], 1.3)
        self.assertEqual(self.instance.x[2].value, None)
        self.instance.x[1] = 1
        self.instance.x[2] = 2
        self.assertEqual(self.instance.x[1], 1)
        self.assertEqual(self.instance.x[2], 2)

    def test_initialize_with_const(self):
        """Test initialize option with a constant"""
        self.model.x = VarList(initialize=3)
        self.instance = self.model.create_instance()
        self.instance.x.add()
        self.instance.x.add()
        self.instance.x.add()
        self.assertEqual(self.instance.x[1], 3)
        self.assertEqual(self.instance.x[2], 3)
        self.instance.x[1] = 1
        self.instance.x[2] = 2
        self.assertEqual(self.instance.x[1], 1)
        self.assertEqual(self.instance.x[2], 2)

    def test_without_initial_value(self):
        """Test default initialization method"""
        self.model.x = VarList()
        self.instance = self.model.create_instance()
        self.instance.x.add()
        self.instance.x.add()
        self.instance.x.add()
        self.assertEqual(self.instance.x[1].value,None)
        self.assertEqual(self.instance.x[2].value,None)
        self.instance.x[1] = 5
        self.instance.x[2] = 6
        self.assertEqual(self.instance.x[1].value,5)
        self.assertEqual(self.instance.x[2].value,6)

    def test_bounds_option1(self):
        """Test bounds option"""
        def x_bounds(model, i):
            return (-1.0,1.0)
        self.model.x = VarList(bounds=x_bounds)
        self.instance = self.model.create_instance()
        self.instance.x.add()
        self.instance.x.add()
        self.assertEqual(value(self.instance.x[1].lb), -1.0)
        self.assertEqual(value(self.instance.x[1].ub), 1.0)

    def test_bounds_option2(self):
        """Test bounds option"""
        self.model.x = VarList(bounds=(-1.0,1.0))
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
        self.assertEqual(self.instance.x.dim(),1)

    def test_keys(self):
        """Test keys method"""
        self.model.x = VarList()
        self.instance = self.model.create_instance()
        self.instance.x.add()
        self.instance.x.add()
        self.assertEqual(set(self.instance.x.keys()),set([1,2]))

    def test_len(self):
        """Test len method"""
        self.model.x = VarList()
        self.instance = self.model.create_instance()
        self.instance.x.add()
        self.instance.x.add()
        self.assertEqual(len(self.instance.x),2)

    def test_value(self):
        """Check the value of the variable"""
        self.model.x = VarList(initialize=3.3)
        self.instance = self.model.create_instance()
        self.instance.x.add()
        self.instance.x.add()
        tmp = value(self.instance.x[1].value)
        self.assertEqual( type(tmp), float)
        self.assertEqual( tmp, 3.3 )
        tmp = float(self.instance.x[1].value)
        self.assertEqual( type(tmp), float)
        self.assertEqual( tmp, 3.3 )
        tmp = int(self.instance.x[1].value)
        self.assertEqual( type(tmp), int)
        self.assertEqual( tmp, 3 )


class Test2DArrayVar(TestSimpleVar):

    def setUp(self):
        #
        # Create Model
        #
        PyomoModel.setUp(self)
        self.model.A = Set(initialize=[1,2])

    def test_fixed_attr(self):
        """Test fixed attribute"""
        self.model.x = Var(self.model.A,self.model.A)
        self.model.y = Var(self.model.A,self.model.A)
        self.instance = self.model.create_instance()
        self.instance.x.fixed = True
        self.assertEqual(self.instance.x[1,2].fixed, False)
        self.instance.y[1,2].fixed=True
        self.assertEqual(self.instance.y[1,2].fixed, True)

    def test_value_attr(self):
        """Test value attribute"""
        self.model.x = Var(self.model.A,self.model.A, dense=True)
        self.model.y = Var(self.model.A,self.model.A, dense=True)
        self.instance = self.model.create_instance()
        try:
            self.instance.x = 3.5
            self.fail("Expected ValueError")
        except ValueError:
            pass
        self.instance.y[1,2] = 3.5
        self.assertEqual(self.instance.y[1,2], 3.5)

    #def test_lb_attr(self):
        #"""Test lb attribute"""
        #self.model.x = Var(self.model.A,self.model.A)
        #self.instance = self.model.create_instance()
        #self.instance.x.setlb(-1.0)
        #self.assertEqual(value(self.instance.x[2,1].lb), -1.0)

    #def test_ub_attr(self):
        #"""Test ub attribute"""
        #self.model.x = Var(self.model.A,self.model.A)
        #self.instance = self.model.create_instance()
        #self.instance.x.setub(1.0)
        #self.assertEqual(value(self.instance.x[2,1].ub), 1.0)

    def test_initialize_with_function(self):
        """Test initialize option with an initialization rule"""
        def init_rule(model, key1, key2):
            i = key1+1
            return key1 == 1 and 1.3 or 2.3
        self.model.x = Var(self.model.A,self.model.A,initialize=init_rule)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1,1], 1.3)
        self.assertEqual(self.instance.x[2,2], 2.3)
        self.instance.x[1,1] = 1
        self.instance.x[2,2] = 2
        self.assertEqual(self.instance.x[1,1], 1)
        self.assertEqual(self.instance.x[2,2], 2)

    def test_initialize_with_dict(self):
        """Test initialize option with a dictionary"""
        self.model.x = Var(self.model.A,self.model.A,
                           initialize={(1,1):1.3,(2,2):2.3})
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1,1], 1.3)
        self.assertEqual(self.instance.x[2,2], 2.3)
        self.instance.x[1,1] = 1
        self.instance.x[2,2] = 2
        self.assertEqual(self.instance.x[1,1], 1)
        self.assertEqual(self.instance.x[2,2], 2)

    def test_initialize_with_const(self):
        """Test initialize option with a constant"""
        self.model.x = Var(self.model.A,self.model.A,initialize=3)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1,1], 3)
        self.assertEqual(self.instance.x[2,2], 3)
        self.instance.x[1,1] = 1
        self.instance.x[2,2] = 2
        self.assertEqual(self.instance.x[1,1], 1)
        self.assertEqual(self.instance.x[2,2], 2)

    def test_without_initial_value(self):
        """Test default initialization"""
        self.model.x = Var(self.model.A,self.model.A)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1,1].value,None)
        self.assertEqual(self.instance.x[2,2].value,None)
        self.instance.x[1,1] = 5
        self.instance.x[2,2] = 6
        self.assertEqual(self.instance.x[1,1].value,5)
        self.assertEqual(self.instance.x[2,2].value,6)

    def test_initialize_option(self):
        """Test initialize option"""
        self.model.x = Var(self.model.A,self.model.A,initialize={(1,1):1.3,(2,2):2.3})
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1,1], 1.3)
        self.assertEqual(self.instance.x[2,2], 2.3)
        try:
            value(self.instance.x[1,2])
            self.fail("Expected ValueError")
        except ValueError:
            pass

    def test_bounds_option1(self):
        """Test bounds option"""
        def x_bounds(model, i, j):
            return (-1.0*(i+j),1.0*(i+j))
        self.model.x = Var(self.model.A, self.model.A, bounds=x_bounds)
        self.instance = self.model.create_instance()
        self.assertEqual(value(self.instance.x[1,1].lb), -2.0)
        self.assertEqual(value(self.instance.x[1,2].ub), 3.0)

    def test_bounds_option2(self):
        """Test bounds option"""
        self.model.x = Var(self.model.A, self.model.A, bounds=(-1.0,1.0))
        self.instance = self.model.create_instance()
        self.assertEqual(value(self.instance.x[1,1].lb), -1.0)
        self.assertEqual(value(self.instance.x[1,1].ub), 1.0)

    def test_rule_option(self):
        """Test rule option"""
        def x_init(model, i, j):
            return 1.3
        self.model.x = Var(self.model.A, self.model.A, initialize=x_init)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1,2].value, 1.3)

    def test_dim(self):
        """Test dim method"""
        self.model.x = Var(self.model.A,self.model.A)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x.dim(),2)

    def test_keys(self):
        """Test keys method"""
        self.model.x = Var(self.model.A,self.model.A, dense=True)
        self.instance = self.model.create_instance()
        ans = [(1,1),(1,2),(2,1),(2,2)]
        self.assertEqual(list(sorted(self.instance.x.keys())),ans)

    def test_len(self):
        """Test len method"""
        self.model.x = Var(self.model.A,self.model.A, dense=True)
        self.instance = self.model.create_instance()
        self.assertEqual(len(self.instance.x),4)

    def test_value(self):
        """Check the value of the variable"""
        self.model.x = Var(self.model.A,self.model.A,initialize=3.3)
        self.instance = self.model.create_instance()
        tmp = value(self.instance.x[1,1].value)
        self.assertEqual( type(tmp), float)
        self.assertEqual( tmp, 3.3 )
        tmp = float(self.instance.x[1,1].value)
        self.assertEqual( type(tmp), float)
        self.assertEqual( tmp, 3.3 )
        tmp = int(self.instance.x[1,1].value)
        self.assertEqual( type(tmp), int)
        self.assertEqual( tmp, 3 )


class TestVarComplexArray(PyomoModel):

    def test_index1(self):
        self.model.A = Set(initialize=range(0,4))
        def B_index(model):
            for i in model.A:
                if i%2 == 0:
                    yield i
        def B_init(model, i, j):
            if j:
                return 2+i
            return -(2+i)
        self.model.B = Var(B_index, [True,False], initialize=B_init, dense=True)
        self.instance = self.model.create_instance()
        self.assertEqual(set(self.instance.B.keys()),set([(0,True),(2,True),(0,False),(2,False)]))
        self.assertEqual(self.instance.B[0,True],2)
        self.assertEqual(self.instance.B[0,False],-2)
        self.assertEqual(self.instance.B[2,True],4)
        self.assertEqual(self.instance.B[2,False],-4)

    def test_index2(self):
        self.model.A = Set(initialize=range(0,4))
        def B_index(model):
            for i in model.A:
                if i%2 == 0:
                    yield i-1, i
        B_index.dimen=2
        def B_init(model, k, i, j):
            if j:
                return (2+i)*k
            return -(2+i)*k
        self.model.B = Var(B_index, [True,False], initialize=B_init, dense=True)
        self.instance = self.model.create_instance()
        #self.instance.pprint()
        self.assertEqual(set(self.instance.B.keys()),set([(-1,0,True),(1,2,True),(-1,0,False),(1,2,False)]))
        self.assertEqual(self.instance.B[-1,0,True],-2)
        self.assertEqual(self.instance.B[-1,0,False],2)
        self.assertEqual(self.instance.B[1,2,True],4)
        self.assertEqual(self.instance.B[1,2,False],-4)


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
        self.assertEqual(instance.suffix.get(instance.a),None)
        instance.suffix.set_value(instance.a,True)
        self.assertEqual(instance.suffix.get(instance.a),True)

    def test_getattr2(self):
        """
        Verify the behavior of non-standard suffixes with an array of variables
        """
        model = AbstractModel()
        model.X = Set(initialize=[1,3,5])
        model.a = Var(model.X)
        model.suffix = Suffix(datatype=Suffix.INT)
        try:
            self.assertEqual(model.a.suffix,None)
            self.fail("Expected AttributeError")
        except AttributeError:
            pass
        instance = model.create_instance()
        self.assertEqual(instance.suffix.get(instance.a[1]),None)
        instance.suffix.set_value(instance.a[1], True)
        self.assertEqual(instance.suffix.get(instance.a[1]),True)

    def test_error2(self):
        try:
            model=AbstractModel()
            model.a = Var(initialize=[1,2,3])
            model.b = Var(model.a)
            self.fail("test_error2")
        except TypeError:
            pass

    def test_contains(self):
        model=AbstractModel()
        model.a = Set(initialize=[1,2,3])
        model.b = Var(model.a, dense=True)
        instance = model.create_instance()
        self.assertEqual(1 in instance.b,True)

    def test_float_int(self):
        model=AbstractModel()
        model.a = Set(initialize=[1,2,3])
        model.b = Var(model.a,initialize=1.1)
        model.c = Var(initialize=2.1)
        model.d = Var()
        instance = model.create_instance()
        self.assertEqual(float(value(instance.b[1])),1.1)
        self.assertEqual(int(value(instance.b[1])),1)
        self.assertEqual(float(value(instance.c)),2.1)
        self.assertEqual(int(value(instance.c)),2)
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
        model=AbstractModel()
        model.a = Set(initialize=[1,2,3])
        model.b = Var(model.a,initialize=1.1,within=PositiveReals)
        model.c = Var(initialize=2.1, within=PositiveReals,dense=True)
        try:
            model.b = 2.2
            self.fail("can't set the value of an array variable")
        except ValueError:
            pass
        instance = model.create_instance()
        try:
            instance.c[1]=2.2
            self.fail("can't use an index to set a scalar variable")
        except KeyError:
            pass
        instance.b[1]=2.2
        try:
            instance.b[4]=2.2
            self.fail("can't set an array variable with a bad index")
        except KeyError:
            pass
        try:
            instance.b[3] = -2.2
            #print "HERE",type(instance)
            #print "HERE",type(instance.b[3])
            self.fail("can't set an array variable with a bad value")
        except ValueError:
            pass
        try:
            tmp = instance.c[3]
            self.fail("can't index a scalar variable")
        except KeyError:
            pass

        try:
            instance.c.set_value('a')
            self.fail("can't set a bad value for variable c")
        except ValueError:
            pass
        try:
            instance.c.set_value(-1.0)
            self.fail("can't set a bad value for variable c")
        except ValueError:
            pass

        #try:
            #instance.c.ub = 'a'
            #self.fail("can't set a bad ub for variable c")
        #except ValueError:
            #pass
        #try:
            #instance.c.ub = -1.0
            #self.fail("can't set a bad ub for variable c")
        #except ValueError:
            #pass

        #try:
            #instance.c.fixed = 'a'
            #self.fail("can't fix a variable with a non-boolean")
        #except ValueError:
            #pass

    def test_set_index(self):

        model = ConcreteModel()
        model.s = Set(initialize=[1,2,3])
        model.x = Var(model.s,initialize=0, dense=True)

        # test proper instantiation
        self.assertEqual(len(model.x),3)
        for i in model.s:
            self.assertEqual(value(model.x[i]),0)

        # test mutability of index set
        model.s.add(4)
        self.assertEqual(len(model.x),3)
        for i in model.s:
            self.assertEqual(value(model.x[i]),0)
        self.assertEqual(len(model.x),4)

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
        with self.assertRaises(ValueError):
            model.x = Var(domain=25)

    def test_simple_nondefault_domain_rule(self):
        model = ConcreteModel()
        model.x = Var(domain=lambda m: Integers)
        self.assertIs(model.x.domain, Integers)

    def test_simple_bad_nondefault_domain_rule(self):
        model = ConcreteModel()
        with self.assertRaises(ValueError):
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
        with self.assertRaises(ValueError):
            model.x = Var(model.s, domain=25)

    def test_indexed_nondefault_domain_rule(self):
        model = ConcreteModel()
        model.s = Set(initialize=[1])
        model.x = Var(model.s, domain=lambda m, i: Integers)
        self.assertIs(model.x[1].domain, Integers)

    def test_indexed_bad_nondefault_domain_rule(self):
        model = ConcreteModel()
        model.s = Set(initialize=[1])
        with self.assertRaises(ValueError):
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
        with self.assertRaises(ValueError):
            model.x.add()

    def test_list_nondefault_domain_rule(self):
        model = ConcreteModel()
        model.x = VarList(domain=lambda m, i: Integers)
        model.x.add()
        self.assertIs(model.x[1].domain, Integers)

    def test_list_bad_nondefault_domain_rule(self):
        model = ConcreteModel()
        model.x = VarList(domain=lambda m, i: 25)
        with self.assertRaises(ValueError):
            model.x.add()

    def test_setdata_index(self):

        model = ConcreteModel()
        model.sindex = Set(initialize=[1])
        model.s = Set(model.sindex,initialize=[1,2,3])
        model.x = Var(model.s[1],initialize=0, dense=True)

        # test proper instantiation
        self.assertEqual(len(model.x),3)
        for i in model.s[1]:
            self.assertEqual(value(model.x[i]),0)

        # test mutability of index set
        newIdx = 4
        self.assertFalse( newIdx in model.s[1] )
        self.assertFalse( newIdx in model.x )

        model.s[1].add(newIdx)
        self.assertTrue( newIdx in model.s[1] )
        self.assertFalse( newIdx in model.x )

        self.assertEqual(len(model.x),3)
        for i in model.s[1]:
            self.assertEqual(value(model.x[i]),0)
        self.assertEqual(len(model.x),4)

        self.assertTrue( newIdx in model.s[1] )
        self.assertTrue( newIdx in model.x )

    def test_setdata_multidimen_index(self):

        model = ConcreteModel()
        model.sindex = Set(initialize=[1])
        model.s = Set(model.sindex,dimen=2,initialize=[(1,1),(1,2),(1,3)])
        model.x = Var(model.s[1],initialize=0, dense=True)

        # test proper instantiation
        self.assertEqual(len(model.x),3)
        for i in model.s[1]:
            self.assertEqual(value(model.x[i]),0)

        # test mutability of index set
        newIdx = (1,4)
        self.assertFalse( newIdx in model.s[1] )
        self.assertFalse( newIdx in model.x )

        model.s[1].add(newIdx)
        self.assertTrue( newIdx in model.s[1] )
        self.assertFalse( newIdx in model.x )

        self.assertEqual(len(model.x),3)
        for i in model.s[1]:
            self.assertEqual(value(model.x[i]),0)
        self.assertEqual(len(model.x),4)

        self.assertTrue( newIdx in model.s[1] )
        self.assertTrue( newIdx in model.x )

    def test_abstract_index(self):
        model = AbstractModel()
        model.A = Set()
        model.B = Set()
        model.C = model.A | model.B
        model.x = Var(model.C)


if __name__ == "__main__":
    unittest.main()
