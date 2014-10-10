import pyutilib.th as unittest

import os
import pickle

from pyomo.solvers.tests.io.writer_test_cases import testCases
from pyomo.solvers.tests.pickling import trivial_model

currdir = os.path.dirname(os.path.abspath(__file__))

problem_list = ['trivial_model']

def createTestMethod(pName, test_case):

    # Skip this test if the solver is not available on the system
    if test_case.available is False:
        def skipping_test(self):
            return self.skipTest('Solver unavailable: '+test_case.name+' ('+test_case.io+')')
        return skipping_test

    def testMethod(self):
        
        files_to_delete = []
        test_case.initialize()
        opt = test_case.solver
        model = trivial_model.define_model()
        model.preprocess()

        instance1 = model.clone()

        # try to pickle the instance
        filename = os.path.join(currdir,pName+'.pickle1')
        files_to_delete.append(filename)
        with open(filename,'wb') as f:
            pickle.dump(instance1,f)

        # try to unpickle the instance 
        instance2 = None
        with open(filename,'rb') as f:
            instance2 = pickle.load(f)
        self.assertNotEqual(id(instance1),id(instance2))

        # try to solve the original instance
        results1 = opt.solve(instance1)
        instance1.load(results1)
        # try to solve the unpickled instance
        results2 = opt.solve(instance2)
        instance2.load(results2)

        # try to pickle the instance and results
        filename = os.path.join(currdir,pName+'.pickle2')
        files_to_delete.append(filename)
        with open(filename,'wb') as f: 
            pickle.dump([instance1,results1],f)

        # try to pickle the instance and results
        filename = os.path.join(currdir,pName+'.pickle3')
        files_to_delete.append(filename)
        with open(filename,'wb') as f: 
            pickle.dump([instance2,results2],f)

        for fname in files_to_delete:
            try:
                os.remove(fname)
            except OSError:
                pass

    return testMethod

def assignTests(cls):
    for case in testCases:
        attrName = "test_pickle_{0}_{1}".format(case.name,case.io)
        setattr(cls,attrName,createTestMethod(attrName,case))

class PickleTest(unittest.TestCase):
    
    @classmethod
    def tearDownClass(self):
        try:
            os.unlink('junk.pickle')
        except Exception:
            pass

assignTests(PickleTest)

if __name__ == "__main__":
    unittest.main()

"""
    # Test that an unpickled instance can be sent through the LP writer
    @unittest.skipIf(solver['cplex'] is None, "Can't find cplex.")
    def test_pickle5(self):
        model = ConcreteModel()
        model.s = Set(initialize=[1,2])
        model.x = Var(within=NonNegativeReals)
        model.x_indexed = Var(model.s, within=NonNegativeReals)
        model.obj = Objective(expr=model.x + model.x_indexed[1] + model.x_indexed[2])
        model.con = Constraint(expr=model.x >= 1)
        model.con2 = Constraint(expr=model.x_indexed[1] + model.x_indexed[2] >= 4)

        inst = model.create()
        strng = pickle.dumps(inst)
        up_inst = pickle.loads(strng)
        opt = solver['cplex']
        results = opt.solve(up_inst)
        up_inst.load(results)
        assert(abs(up_inst.x.value - 1.0) < .001)
        assert(abs(up_inst.x_indexed[1].value - 0.0) < .001)
        assert(abs(up_inst.x_indexed[2].value - 4.0) < .001)

    # Test that an unpickled instance can be sent through the NL writer
    @unittest.skipIf(solver['asl:cplexamp'] is None, "Can't find cplexamp.")
    def test_pickle6(self):
        model = ConcreteModel()
        model.s = Set(initialize=[1,2])
        model.x = Var(within=NonNegativeReals)
        model.x_indexed = Var(model.s, within=NonNegativeReals)
        model.obj = Objective(expr=model.x + model.x_indexed[1] + model.x_indexed[2])
        model.con = Constraint(expr=model.x >= 1)
        model.con2 = Constraint(expr=model.x_indexed[1] + model.x_indexed[2] >= 4)

        inst = model.create()
        strng = pickle.dumps(inst)
        up_inst = pickle.loads(strng)
        opt = solver['asl:cplexamp']
        results = opt.solve(up_inst)
        up_inst.load(results)
        assert(abs(up_inst.x.value - 1.0) < .001)
        assert(abs(up_inst.x_indexed[1].value - 0.0) < .001)
        assert(abs(up_inst.x_indexed[2].value - 4.0) < .001)

    # Test that an unpickled instance can be sent through the cplex python interface
    @unittest.skipIf(solver['py:cplex'] is None, "Can't cplex python interface.")
    def test_pickle7(self):
        model = ConcreteModel()
        model.s = Set(initialize=[1,2])
        model.x = Var(within=NonNegativeReals)
        model.x_indexed = Var(model.s, within=NonNegativeReals)
        model.obj = Objective(expr=model.x + model.x_indexed[1] + model.x_indexed[2])
        model.con = Constraint(expr=model.x >= 1)
        model.con2 = Constraint(expr=model.x_indexed[1] + model.x_indexed[2] >= 4)

        inst = model.create()
        strng = pickle.dumps(inst)
        up_inst = pickle.loads(strng)
        opt = solver['py:cplex']
        results = opt.solve(up_inst)
        up_inst.load(results)
        assert(abs(up_inst.x.value - 1.0) < .001)
        assert(abs(up_inst.x_indexed[1].value - 0.0) < .001)
        assert(abs(up_inst.x_indexed[2].value - 4.0) < .001)
"""
