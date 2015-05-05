#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Unit Tests for Utility Functions
#

import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest

from pyomo.environ import *

def obj_rule(model):
    return sum(model.x[a] + model.y[a] for a in model.A)
def constr_rule(model,a):
    return model.x[a] >= model.y[a]


class Test(unittest.TestCase):

    def test_expr1(self):
        model = AbstractModel()
        model.A = Set(initialize=[1,2,3])
        model.B = Param(model.A,initialize={1:100,2:200,3:300}, mutable=True)
        model.x = Var(model.A)
        model.y = Var(model.A)
        instance=model.create_instance()
        expr = dot_product(instance.x,instance.B,instance.y)
        self.assertEqual(
            str(expr),
            "x[1] * B[1] * y[1] + x[2] * B[2] * y[2] + x[3] * B[3] * y[3]" )

    def test_expr2(self):
        model = AbstractModel()
        model.A = Set(initialize=[1,2,3])
        model.B = Param(model.A,initialize={1:100,2:200,3:300}, mutable=True)
        model.x = Var(model.A)
        model.y = Var(model.A)
        instance=model.create_instance()
        expr = dot_product(instance.x,instance.B,instance.y, index=[1,3])
        self.assertEqual(
            str(expr),
            "x[1] * B[1] * y[1] + x[3] * B[3] * y[3]" )

    def test_expr3(self):
        model = AbstractModel()
        model.A = Set(initialize=[1,2,3])
        model.B = Param(model.A,initialize={1:100,2:200,3:300}, mutable=True)
        model.x = Var(model.A)
        model.y = Var(model.A)
        instance=model.create_instance()
        expr = dot_product(instance.x,instance.B,denom=instance.y, index=[1,3])
        self.assertEqual(
            str(expr),
            "x[1] * B[1] / y[1] + x[3] * B[3] / y[3]" )

    def test_expr4(self):
        model = AbstractModel()
        model.A = Set(initialize=[1,2,3])
        model.B = Param(model.A,initialize={1:100,2:200,3:300}, mutable=True)
        model.x = Var(model.A)
        model.y = Var(model.A)
        instance=model.create_instance()
        expr = dot_product(denom=[instance.y,instance.x])
        self.assertEqual(
            str(expr),
            "1 / ( y[1] * x[1] ) + 1 / ( y[2] * x[2] ) + 1 / ( y[3] * x[3] )" )

    def test_expr5(self):
        model = ConcreteModel()
        model.A = Set(initialize=[1,2,3], doc='set A')
        model.B = Param(model.A, initialize={1:100,2:200,3:300}, doc='param B', mutable=True)
        model.C = Param(initialize=3, doc='param C', mutable=True)
        model.x = Var(model.A, doc='var x')
        model.y = Var(doc='var y')
        model.o = Objective(expr=model.y, doc='obj o')
        model.c1 = Constraint(expr=model.x[1] >= 0, doc='con c1')
        def c2_rule(model, a):
            return model.B[a] * model.x[a] <= 1
        model.c2 = Constraint(model.A, doc='con c2', rule=c2_rule)
        model.c3 = ConstraintList(doc='con c3')
        model.c3.add(model.y <= 0)
        #
        OUTPUT=open(currdir+"test_expr5.out","w")
        model.pprint(ostream=OUTPUT)
        OUTPUT.close()
        self.assertFileEqualsBaseline(currdir+"test_expr5.out",currdir+"test_expr5.txt")

    def test_prod(self):
        self.assertEqual(prod([1,2,3,5]),30)

    def test_summation_error1(self):
        try:
            dot_product()
            self.fail("Expected ValueError")
        except ValueError:
            pass

    def test_summation_error2(self):
        model = AbstractModel()
        model.A = Set(initialize=[1,2,3])
        model.B = Param(model.A,initialize={1:100,2:200,3:300}, mutable=True)
        model.x = Var(model.A)
        instance=model.create_instance()
        try:
            expr = dot_product(instance.x,instance.B)
            self.fail("Expected ValueError")
        except ValueError:
            pass

    def test_summation_error3(self):
        model = AbstractModel()
        model.A = Set(initialize=[1,2,3])
        model.B = Param(model.A,initialize={1:100,2:200,3:300}, mutable=True)
        model.x = Var(model.A)
        instance=model.create_instance()
        try:
            expr = dot_product(denom=(instance.x,instance.B))
            self.fail("Expected ValueError")
        except ValueError:
            pass

    def test_sequence_error1(self):
        try:
            sequence()
            self.fail("Expected ValueError")
        except ValueError:
            pass
        try:
            sequence(1,2,3,4)
            self.fail("Expected ValueError")
        except ValueError:
            pass

    def test_sequence(self):
        self.assertEqual(list(sequence(10)), list(range(1,11)))
        self.assertEqual(list(sequence(8,10)), [8,9,10])
        self.assertEqual(list(sequence(1,10,3)), [1,4,7,10])


if __name__ == "__main__":
    unittest.main()
