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

import pickle
import os
import sys
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest

from pyomo.environ import *

def obj_rule(model):
    return sum(model.x[a] + model.y[a] for a in model.A)
def constr_rule(model,a):
    return model.x[a] >= model.y[a]
def simple_con_rule(model, i):
    return model.x <= i

class Test(unittest.TestCase):

    def verifyModel(self, ref, new):
        # Verify the block indices
        self.assertEqual(sorted(ref._data.keys()), sorted(new._data.keys()))
        for idx in ref._data.keys():
            self.assertEqual(type(ref._data[idx]),  type(new._data[idx]))
            if idx is not None:
                self.assertNotEqual(id(ref._data[idx]),  id(new._data[idx]))
        self.assertEqual( id(ref.solutions._instance()), id(ref) )
        self.assertEqual( id(new.solutions._instance()), id(new) )
            
        # Verify the block attributes
        for idx in ref._data.keys():
            ref_c = ref[idx].component_map()
            new_c = new[idx].component_map()
            self.assertEqual( sorted(ref_c.keys()), sorted(new_c.keys()) )
            for a in ref_c.keys():
                self.assertEqual(type(ref_c[a]),  type(new_c[a]))
                self.assertNotEqual(id(ref_c[a]),  id(new_c[a]))
                

    def test_pickle_empty_abstract_model(self):
        model = AbstractModel()
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    def test_pickle_abstract_model_set(self):
        model = AbstractModel()
        model.A = Set(initialize=[1,2,3])
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    def test_pickle_abstract_model_virtual_set(self):
        model = AbstractModel()
        model._a = Set(initialize=[1,2,3])
        model.A = model._a * model._a
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    def test_pickle_abstract_model_param(self):
        model = AbstractModel()
        model.A = Param(initialize=1)
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    def test_pickle_abstract_model_indexed_param(self):
        model = AbstractModel()
        model.A = Param([1,2,3], initialize={1:100,2:200,3:300})
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    def test_pickle_abstract_model_mutable_param(self):
        model = AbstractModel()
        model.A = Param(initialize=1, mutable=True)
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    def test_pickle_abstract_model_mutable_indexed_param(self):
        model = AbstractModel()
        model.A = Param([1,2,3], initialize={1:100,3:300}, mutable=True)
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    def test_pickle_abstract_model_var(self):
        model = AbstractModel()
        model.A = Var(initialize=1)
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    def test_pickle_abstract_model_indexed_var(self):
        model = AbstractModel()
        model.A = Var([1,2,3], initialize={1:100,2:200,3:300})
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    def test_pickle_abstract_model_constant_objective(self):
        model = AbstractModel()
        model.A = Objective(expr=1)
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    def test_pickle_abstract_model_objective(self):
        model = AbstractModel()
        model.x = Var()
        model.A = Objective(expr=model.x <= 0)
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    def test_pickle_abstract_model_constraint(self):
        model = AbstractModel()
        model.x = Var()
        model.A = Constraint(expr=model.x <= 0)
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    def test_pickle_abstract_model_indexed_constraint(self):
        model = AbstractModel()
        model.x = Var()
        model.A = Constraint([1,2,3], rule=simple_con_rule)
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    #########

    def test_pickle_empty_concrete_model(self):
        model = ConcreteModel()
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    def test_pickle_concrete_model_set(self):
        model = ConcreteModel()
        model.A = Set(initialize=[1,2,3])
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    def test_pickle_concrete_model_virtual_set(self):
        model = ConcreteModel()
        model._a = Set(initialize=[1,2,3])
        model.A = model._a * model._a
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    def test_pickle_concrete_model_param(self):
        model = ConcreteModel()
        model.A = Param(initialize=1)
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    def test_pickle_concrete_model_indexed_param(self):
        model = ConcreteModel()
        model.A = Param([1,2,3], initialize={1:100,2:200,3:300})
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    def test_pickle_concrete_model_mutable_param(self):
        model = ConcreteModel()
        model.A = Param(initialize=1, mutable=True)
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    def test_pickle_concrete_model_mutable_indexed_param(self):
        model = ConcreteModel()
        model.A = Param([1,2,3], initialize={1:100,3:300}, mutable=True)
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    def test_pickle_concrete_model_var(self):
        model = ConcreteModel()
        model.A = Var(initialize=1)
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    def test_pickle_concrete_model_indexed_var(self):
        model = ConcreteModel()
        model.A = Var([1,2,3], initialize={1:100,2:200,3:300})
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    def test_pickle_concrete_model_constant_objective(self):
        model = ConcreteModel()
        model.A = Objective(expr=1)
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    def test_pickle_concrete_model_objective(self):
        model = ConcreteModel()
        model.x = Var()
        model.A = Objective(expr=model.x <= 0)
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    def test_pickle_concrete_model_constraint(self):
        model = ConcreteModel()
        model.x = Var()
        model.A = Constraint(expr=model.x <= 0)
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    def test_pickle_concrete_model_indexed_constraint(self):
        model = ConcreteModel()
        model.x = Var()
        model.A = Constraint([1,2,3], rule=simple_con_rule)
        str = pickle.dumps(model)
        tmodel = pickle.loads(str)
        self.verifyModel(model, tmodel)

    ##########


    # tests the ability to pickle an abstract model prior to construction,
    # read it back it, and create an instance from it. validation is relatively
    # weak, in that it only tests the validity of an expression constructed
    # using the resulting model.
    def test_pickle1(self):
        model = AbstractModel()
        model.A = Set(initialize=[1,2,3])
        model.B = Param(model.A,initialize={1:100,2:200,3:300}, mutable=True)
        model.x = Var(model.A)
        model.y = Var(model.A)
        model.obj = Objective(rule=obj_rule)
        model.constr = Constraint(model.A,rule=constr_rule)
        pickle_str = pickle.dumps(model)
        tmodel = pickle.loads(pickle_str)
        instance=tmodel.create_instance()
        expr = dot_product(instance.x,instance.B,instance.y)
        self.assertEquals( 
            str(expr), 
            "x[1] * B[1] * y[1] + x[2] * B[2] * y[2] + x[3] * B[3] * y[3]" )

    # same as above, but pickles the constructed AbstractModel and 
    # then operates on the unpickled instance.
    def test_pickle2(self):
        model = AbstractModel()
        model.A = Set(initialize=[1,2,3])
        model.B = Param(model.A,initialize={1:100,2:200,3:300}, mutable=True)
        model.x = Var(model.A)
        model.y = Var(model.A)
        model.obj = Objective(rule=obj_rule)
        model.constr = Constraint(model.A,rule=constr_rule)
        tmp=model.create_instance()
        pickle_str = pickle.dumps(tmp)
        instance = pickle.loads(pickle_str)
        expr = dot_product(instance.x,instance.B,instance.y)
        self.assertEquals( 
            str(expr), 
            "x[1] * B[1] * y[1] + x[2] * B[2] * y[2] + x[3] * B[3] * y[3]" )

    # verifies that the use of lambda expressions as rules yields model instances
    # that are not pickle'able.
    @unittest.skipIf(sys.version_info[:2] < (2,6), "Skipping test because the sparse_dict repn is not supported")
    def test_pickle3(self):
        def rule1(model):
            return (1,model.x+model.y[1],2)
        def rule2(model, i):
            return (1,model.x+model.y[1]+i,2)

        model = AbstractModel()
        model.a = Set(initialize=[1,2,3])
        model.A = Param(initialize=1, mutable=True)
        model.B = Param(model.a, mutable=True)
        model.x = Var(initialize=1,within=Reals)
        model.y = Var(model.a, initialize=1,within=Reals)
        model.obj = Objective(rule=lambda model: model.x+model.y[1])
        model.obj2 = Objective(model.a,rule=lambda model,i: i+model.x+model.y[1])
        model.con = Constraint(rule=rule1)
        model.con2 = Constraint(model.a, rule=rule2)
        instance = model.create_instance()
        try:
            str = pickle.dumps(instance)
            self.fail("Expected pickling error due to the use of lambda expressions - did not generate one!")
        except pickle.PicklingError:
            pass
        except TypeError:
            pass
        except AttributeError:
            pass

    # verifies that we can print a constructed model and obtain identical results before and after 
    # pickling. introduced due to a test case by Gabe that illustrated __getstate__ of various 
    # modeling components was incorrectly and unexpectedly modifying object state.
    def test_pickle4(self):
    
        model = ConcreteModel()
        model.s = Set(initialize=[1,2])
        model.x = Var(within=NonNegativeReals)
        model.x_indexed = Var(model.s, within=NonNegativeReals)
        model.obj = Objective(expr=model.x + model.x_indexed[1] + model.x_indexed[2])
        model.con = Constraint(expr=model.x >= 1)
        model.con2 = Constraint(expr=model.x_indexed[1] + model.x_indexed[2] >= 4)

        OUTPUT=open(currdir+"test_pickle4_baseline.out","w")
        model.pprint(ostream=OUTPUT)
        OUTPUT.close()
        self.assertFileEqualsBaseline(currdir+"test_pickle4_baseline.out",currdir+"test_pickle4_baseline.txt")

        str = pickle.dumps(model)

        OUTPUT=open(currdir+"test_pickle4_after.out","w")
        model.pprint(ostream=OUTPUT)
        OUTPUT.close()
        self.assertFileEqualsBaseline(currdir+"test_pickle4_after.out",currdir+"test_pickle4_baseline.txt")
        


if __name__ == "__main__":
    unittest.main()
