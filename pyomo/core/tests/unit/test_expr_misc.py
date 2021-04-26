#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Unit Tests for Utility Functions
#

import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest

from pyomo.environ import AbstractModel, ConcreteModel, ConstraintList, Set, Param, Var, Constraint, Objective, sum_product, quicksum, sequence, prod

def obj_rule(model):
    return sum(model.x[a] + model.y[a] for a in model.A)
def constr_rule(model,a):
    return model.x[a] >= model.y[a]


class Test(unittest.TestCase):

    def test_expr0(self):
        model = AbstractModel()
        model.A = Set(initialize=[1,2,3])
        model.B = Param(model.A,initialize={1:100,2:200,3:300}, mutable=True)
        model.C = Param(model.A,initialize={1:100,2:200,3:300}, mutable=False)
        model.x = Var(model.A)
        model.y = Var(model.A)
        instance=model.create_instance()
        expr = sum_product(instance.B,instance.y)
        baseline = "B[1]*y[1] + B[2]*y[2] + B[3]*y[3]"
        self.assertEqual( str(expr), baseline )
        expr = sum_product(instance.C,instance.y)
        self.assertEqual( str(expr), "100*y[1] + 200*y[2] + 300*y[3]" )

    def test_expr1(self):
        model = AbstractModel()
        model.A = Set(initialize=[1,2,3])
        model.B = Param(model.A,initialize={1:100,2:200,3:300}, mutable=True)
        model.C = Param(model.A,initialize={1:100,2:200,3:300}, mutable=False)
        model.x = Var(model.A)
        model.y = Var(model.A)
        instance=model.create_instance()
        expr = sum_product(instance.x,instance.B,instance.y)
        baseline = "B[1]*x[1]*y[1] + B[2]*x[2]*y[2] + B[3]*x[3]*y[3]"
        self.assertEqual( str(expr), baseline )
        expr = sum_product(instance.x,instance.C,instance.y)
        self.assertEqual( str(expr), "100*x[1]*y[1] + 200*x[2]*y[2] + 300*x[3]*y[3]" )

    def test_expr2(self):
        model = AbstractModel()
        model.A = Set(initialize=[1,2,3])
        model.B = Param(model.A,initialize={1:100,2:200,3:300}, mutable=True)
        model.C = Param(model.A,initialize={1:100,2:200,3:300}, mutable=False)
        model.x = Var(model.A)
        model.y = Var(model.A)
        instance=model.create_instance()
        expr = sum_product(instance.x,instance.B,instance.y, index=[1,3])
        baseline = "B[1]*x[1]*y[1] + B[3]*x[3]*y[3]"
        self.assertEqual( str(expr), baseline )
        expr = sum_product(instance.x,instance.C,instance.y, index=[1,3])
        self.assertEqual( str(expr), "100*x[1]*y[1] + 300*x[3]*y[3]" )

    def test_expr3(self):
        model = AbstractModel()
        model.A = Set(initialize=[1,2,3])
        model.B = Param(model.A,initialize={1:100,2:200,3:300}, mutable=True)
        model.C = Param(model.A,initialize={1:100,2:200,3:300}, mutable=False)
        model.x = Var(model.A)
        model.y = Var(model.A)
        instance=model.create_instance()
        expr = sum_product(instance.x,instance.B,denom=instance.y, index=[1,3])
        baseline = "B[1]*x[1]/y[1] + B[3]*x[3]/y[3]"
        self.assertEqual( str(expr), baseline )
        expr = sum_product(instance.x,instance.C,denom=instance.y, index=[1,3])
        self.assertEqual( str(expr), "100*x[1]/y[1] + 300*x[3]/y[3]" )

    def test_expr4(self):
        model = AbstractModel()
        model.A = Set(initialize=[1,2,3])
        model.B = Param(model.A,initialize={1:100,2:200,3:300}, mutable=True)
        model.x = Var(model.A)
        model.y = Var(model.A)
        instance=model.create_instance()
        expr = sum_product(denom=[instance.y,instance.x])
        baseline = "1/(y[1]*x[1]) + 1/(y[2]*x[2]) + 1/(y[3]*x[3])"
        self.assertEqual( str(expr), baseline )

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

    def test_prod1(self):
        self.assertEqual(prod([1,2,3,5]),30)

    def test_prod2(self):
        model = ConcreteModel()
        model.A = Set(initialize=[1,2,3], doc='set A')
        model.x = Var(model.A)
        expr = prod(model.x[i] for i in model.x)
        baseline = "x[1]*x[2]*x[3]"
        self.assertEqual( str(expr), baseline )
        expr = prod(model.x)
        self.assertEqual( expr, 6)

    def test_sum1(self):
        self.assertEqual(quicksum([1,2,3,5]),11)

    def test_sum2(self):
        model = ConcreteModel()
        model.A = Set(initialize=[1,2,3], doc='set A')
        model.x = Var(model.A)
        expr = quicksum(model.x[i] for i in model.x)
        baseline = "x[1] + x[2] + x[3]"
        self.assertEqual( str(expr), baseline )

    def test_sum3(self):
        model = ConcreteModel()
        model.A = Set(initialize=[1,2,3], doc='set A')
        model.x = Var(model.A)
        expr = quicksum(model.x)
        self.assertEqual( expr, 6)

    def test_summation_error1(self):
        try:
            sum_product()
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
            expr = sum_product(instance.x,instance.B)
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
            expr = sum_product(denom=(instance.x,instance.B))
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
