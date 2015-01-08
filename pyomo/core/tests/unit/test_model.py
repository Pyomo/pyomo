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
# Test             Class to test the Model class
#

import os
import sys
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
import pyutilib.services
from pyomo.core.base import IntegerSet
from pyomo.environ import *
from pyomo.opt import *

solver = load_solvers('glpk')


class Test(unittest.TestCase):

    def tearDown(self):
        if os.path.exists("unknown.lp"):
            os.unlink("unknown.lp")
        pyutilib.services.TempfileManager.clear_tempfiles()


    def test_clone_concrete_model(self):
        def _populate(b, *args):
            b.A = RangeSet(1,3)
            b.v = Var()
            b.vv = Var(m.A)
            b.p = Param()
            
        m = ConcreteModel()
        _populate(m)
        m.b = Block()
        _populate(m.b)
        m.b.c = Block()
        _populate(m.b.c)
        m.bb = Block(m.A, rule=_populate)

        n = m.clone()
        self.assertNotEqual(id(m), id(n))

        self.assertEqual(id(m), id(m.b.parent_block()))
        self.assertEqual(id(m), id(m.bb.parent_block()))
        self.assertEqual(id(m), id(m.bb[1].parent_block()))
        self.assertEqual(id(m), id(m.bb[2].parent_block()))
        self.assertEqual(id(m), id(m.bb[3].parent_block()))

        self.assertEqual(id(n), id(n.b.parent_block()))
        self.assertEqual(id(n), id(n.bb.parent_block()))
        self.assertEqual(id(n), id(n.bb[1].parent_block()))
        self.assertEqual(id(n), id(n.bb[2].parent_block()))
        self.assertEqual(id(n), id(n.bb[3].parent_block()))

        for x,y in ((m, n), (m.b, n.b), (m.b.c, n.b.c), (m.bb[2], n.bb[2])):
            self.assertNotEqual(id(x), id(y))
            self.assertNotEqual(id(x.parent_block()), id(x))
            self.assertNotEqual(id(y.parent_block()), id(y))

            self.assertEqual(id(x.A.parent_block()), id(x))
            self.assertEqual(id(x.v.parent_block()), id(x))
            self.assertEqual(id(x.vv.parent_block()), id(x))
            self.assertEqual(id(x.vv[1].parent_block()), id(x))
            self.assertEqual(id(x.p.parent_block()), id(x))

            self.assertEqual(id(y.A.parent_block()), id(y))
            self.assertEqual(id(y.v.parent_block()), id(y))
            self.assertEqual(id(y.vv.parent_block()), id(y))
            self.assertEqual(id(y.vv[1].parent_block()), id(y))
            self.assertEqual(id(y.p.parent_block()), id(y))

    def test_clone_abstract_model(self):
        def _populate(b, *args):
            b.A = RangeSet(1,3)
            b.v = Var()
            b.vv = Var(m.A)
            b.p = Param()
            
        m = AbstractModel()
        _populate(m)
        m.b = Block()
        _populate(m.b)
        m.b.c = Block()
        _populate(m.b.c)
        m.bb = Block(m.A, rule=_populate)

        n = m.clone()
        self.assertNotEqual(id(m), id(n))

        self.assertEqual(id(m), id(m.b.parent_block()))
        self.assertEqual(id(m), id(m.bb.parent_block()))

        self.assertEqual(id(n), id(n.b.parent_block()))
        self.assertEqual(id(n), id(n.bb.parent_block()))

        for x,y in ((m, n), (m.b, n.b), (m.b.c, n.b.c)):
            self.assertNotEqual(id(x), id(y))
            self.assertNotEqual(id(x.parent_block()), id(x))
            self.assertNotEqual(id(y.parent_block()), id(y))

            self.assertEqual(id(x.A.parent_block()), id(x))
            self.assertEqual(id(x.v.parent_block()), id(x))
            self.assertEqual(id(x.vv.parent_block()), id(x))
            self.assertEqual(id(x.p.parent_block()), id(x))

            self.assertEqual(id(y.A.parent_block()), id(y))
            self.assertEqual(id(y.v.parent_block()), id(y))
            self.assertEqual(id(y.vv.parent_block()), id(y))
            self.assertEqual(id(y.p.parent_block()), id(y))

    def test_clear_attribute(self):
        """ Coverage of the _clear_attribute method """
        model = ConcreteModel()
        obj = Set()
        model.A = obj
        self.assertEqual(model.A.name,"A")
        self.assertEqual(obj.name,"A")
        self.assertIs(obj, model.A)

        obj = Var()
        model.A = obj
        self.assertEqual(model.A.name,"A")
        self.assertEqual(obj.name,"A")
        self.assertIs(obj, model.A)

        obj = Param()
        model.A = obj
        self.assertEqual(model.A.name,"A")
        self.assertEqual(obj.name,"A")
        self.assertIs(obj, model.A)

        obj = Objective()
        model.A = obj
        self.assertEqual(model.A.name,"A")
        self.assertEqual(obj.name,"A")
        self.assertIs(obj, model.A)

        obj = Constraint()
        model.A = obj
        self.assertEqual(model.A.name,"A")
        self.assertEqual(obj.name,"A")
        self.assertIs(obj, model.A)

        obj = Set()
        model.A = obj
        self.assertEqual(model.A.name,"A")
        self.assertEqual(obj.name,"A")
        self.assertIs(obj, model.A)

    def test_set_attr(self):
        model = ConcreteModel()
        model.x = Param(mutable=True)
        model.x = 5
        self.assertEqual(value(model.x), 5)
        model.x = 6
        self.assertEqual(value(model.x), 6)
        try:
            model.x = None
            self.fail("Expected exception assigning None to domain Any")
        except ValueError:
            pass

    def test_write(self):
        model = ConcreteModel()
        model.A = RangeSet(1,4)
        model.x = Var(model.A, bounds=(-1,1))
        def obj_rule(model):
            return summation(model.x)
        model.obj = Objective(rule=obj_rule)
        instance = model.create()
        instance.write()

    def test_write2(self):
        model = ConcreteModel()
        model.A = RangeSet(1,4)
        model.x = Var(model.A, bounds=(-1,1))
        def obj_rule(model):
            return summation(model.x)
        model.obj = Objective(rule=obj_rule)
        def c_rule(model):
            return (1, model.x[1]+model.x[2], 2)
        model.c = Constraint(rule=c_rule)
        instance = model.create()
        instance.write()

    def test_write3(self):
        """Test that the summation works correctly, even though param 'w' has a default value"""
        model = ConcreteModel()
        model.J = RangeSet(1,4)
        model.w=Param(model.J, default=4)
        model.x=Var(model.J)
        def obj_rule(instance):
            return summation(instance.w, instance.x)
        model.obj = Objective(rule=obj_rule)
        instance = model.create()
        self.assertEqual(len(instance.obj[None].expr._args), 4)

    @unittest.skipIf(solver['glpk'] is None, "glpk solver is not available")
    def test_solve1(self):
        model = ConcreteModel()
        model.A = RangeSet(1,4)
        model.x = Var(model.A, bounds=(-1,1))
        def obj_rule(model):
            return summation(model.x)
        model.obj = Objective(rule=obj_rule)
        def c_rule(model):
            expr = 0
            for i in model.A:
                expr += i*model.x[i]
            return expr == 0
        model.c = Constraint(rule=c_rule)
        instance = model.create()
        opt = solver['glpk']
        results = opt.solve(instance, keepfiles=True, symbolic_solver_labels=True)
        results.write(filename=currdir+"solve1.out", format='json')
        self.assertMatchesJsonBaseline(currdir+"solve1.out",currdir+"solve1.txt", tolerance=1e-4)
        #
        def d_rule(model):
            return model.x[1] >= 0
        model.d = Constraint(rule=d_rule)
        model.d.deactivate()
        instance = model.create()
        results = opt.solve(instance, keepfiles=True)
        results.write(filename=currdir+"solve1x.out", format='json')
        self.assertMatchesJsonBaseline(currdir+"solve1x.out",currdir+"solve1.txt", tolerance=1e-4)
        #
        model.d.activate()
        instance = model.create()
        results = opt.solve(instance, keepfiles=True)
        results.write(filename=currdir+"solve1a.out", format='json')
        self.assertMatchesJsonBaseline(currdir+"solve1a.out",currdir+"solve1a.txt", tolerance=1e-4)
        #
        model.d.deactivate()
        def e_rule(model, i):
            return model.x[i] >= 0
        model.e = Constraint(model.A, rule=e_rule)
        instance = model.create()
        for i in instance.A:
            instance.e[i].deactivate()
        results = opt.solve(instance, keepfiles=True)
        results.write(filename=currdir+"solve1b.out", format='json')
        self.assertMatchesJsonBaseline(currdir+"solve1b.out",currdir+"solve1b.txt", tolerance=1e-4)

    def Xtest_load1(self):
        """Testing loading of vector solutions"""
        model = ConcreteModel()
        model.A = RangeSet(1,4)
        model.x = Var(model.A, bounds=(-1,1))
        def obj_rule(model):
            return summation(model.x)
        model.obj = Objective(rule=obj_rule)
        def c_rule(model):
            expr = 0
            for i in model.A:
                expr += i*model.x[i]
            return expr == 0
        model.c = Constraint(rule=c_rule)
        instance = model.create()
        ans = [0.75]*4
        instance.load(ans)
        instance.display(currdir+"solve1.out")
        self.assertFileEqualsBaseline(currdir+"solve1.out",currdir+"solve1c.txt")

    
    @unittest.skipIf(solver['glpk'] is None, "glpk solver is not available")
    def Xtest_solve2(self):
        """
        WEH - this is disabled because glpk appears to work fine
        on this example.  I'm not quite sure what has changed that has
        impacted this test...
        """
        model = ConcreteModel()
        model.A = RangeSet(1,4)
        model.x = Var(model.A, bounds=(-1,1))
        def obj_rule(model):
            expr = 0
            for i in model.A:
                expr += model.x[i]
            return expr
        model.obj = Objective(rule=obj_rule)
        instance = model.create()
        #instance.pprint()
        opt = solvers.GLPK(keepfiles=True)
        solutions = opt.solve(instance)
        solutions.write()
        sys.exit(1)
        try:
            instance.load(solutions)
            self.fail("Cannot load a solution with a bad solver status")
        except ValueError:
            pass

    def test_solve3(self):
        model = ConcreteModel()
        model.A = RangeSet(1,4)
        model.x = Var(model.A, bounds=(-1,1))
        def obj_rule(model):
            expr = 0
            for i in model.A:
                expr += model.x[i]
            return expr
        model.obj = Objective(rule=obj_rule)
        instance = model.create()
        instance.display(currdir+"solve3.out")
        self.assertFileEqualsBaseline(currdir+"solve3.out",currdir+"solve3.txt")

    @unittest.skipIf(solver['glpk'] is None, "glpk solver is not available")
    def test_solve4(self):
        model = ConcreteModel()
        model.A = RangeSet(1,4)
        model.x = Var(model.A, bounds=(-1,1))
        def obj_rule(model):
            return summation(model.x)
        model.obj = Objective(rule=obj_rule)
        def c_rule(model):
            expr = 0
            for i in model.A:
                expr += i*model.x[i]
            return expr == 0
        model.c = Constraint(rule=c_rule)
        instance = model.create()
        opt = solver['glpk']
        results = opt.solve(instance, symbolic_solver_labels=True)
        results.write(filename=currdir+'solve4.out', format='json')
        self.assertMatchesJsonBaseline(currdir+"solve4.out",currdir+"solve1.txt", tolerance=1e-4)

    @unittest.skipIf(solver['glpk'] is None, "glpk solver is not available")
    def Xtest_solve5(self):
        """ A draft test for the option to select an objective """
        model = ConcreteModel()
        model.A = RangeSet(1,4)
        model.x = Var(model.A, bounds=(-1,1))
        def obj1_rule(model):
            expr = 0
            for i in model.A:
                expr += model.x[i]
            return expr
        model.obj1 = Objective(rule=obj1_rule)
        def obj2_rule(model):
            expr = 0
            tmp=-1
            for i in model.A:
                expr += tmp*i*model.x[i]
                tmp *= -1
            return expr
        model.obj2 = Objective(rule=obj2_rule)
        instance = model.create()
        opt = solver['glpk']
        results = opt.solve(instance, objective='obj2')
        results.write(filename=currdir+"solve5.out", format='json')
        self.assertMatchesJsonBaseline(currdir+"solve5.out",currdir+"solve5a.txt", tolerance=1e-4)

    @unittest.expectedFailure
    @unittest.skipIf(solver['glpk'] is None, "glpk solver is not available")
    def test_solve6(self):
        #
        # Test that solution values have complete block names:
        #   b.obj
        #   b.x
        #
        model = ConcreteModel()
        model.y = Var(bounds=(-1,1))
        model.b = Block()
        model.b.A = RangeSet(1,4)
        model.b.x = Var(model.b.A, bounds=(-1,1))
        def obj_rule(block):
            return summation(block.x)
        model.b.obj = Objective(rule=obj_rule)
        def c_rule(model):
            expr = model.y
            for i in model.b.A:
                expr += i*model.b.x[i]
            return expr == 0
        model.c = Constraint(rule=c_rule)
        instance = model.create()
        opt = solver['glpk']
        results = opt.solve(instance, symbolic_solver_labels=True)
        results.write(filename=currdir+'solve6.out', format='json')
        self.assertMatchesJsonBaseline(currdir+"solve6.out", currdir+"solve6.txt", tolerance=1e-4)

    @unittest.skipIf(solver['glpk'] is None, "glpk solver is not available")
    def test_solve7(self):
        #
        # Test that solution values are writen with appropriate quotations in results
        #
        model = ConcreteModel()
        model.y = Var(bounds=(-1,1))
        model.A = RangeSet(1,4)
        model.B = Set(initialize=['A B', 'C,D', 'E'])
        model.x = Var(model.A, model.B, bounds=(-1,1))
        def obj_rule(model):
            return summation(model.x)
        model.obj = Objective(rule=obj_rule)
        def c_rule(model):
            expr = model.y
            for i in model.A:
                for j in model.B:
                    expr += i*model.x[i,j]
            return expr == 0
        model.c = Constraint(rule=c_rule)
        instance = model.create()
        opt = solver['glpk']
        results = opt.solve(instance, symbolic_solver_labels=True)
        instance.load(results)
        instance.display()
        results = instance.update_results(results)
        results.write(filename=currdir+'solve7.out', format='json')
        self.assertMatchesJsonBaseline(currdir+"solve7.out", currdir+"solve7.txt", tolerance=1e-4)

    def test_stats1(self):
        model = ConcreteModel()
        model.x = Var([1,2])
        def obj_rule(model, i):
            return summation(model.x)
        model.obj = Objective([1,2], rule=obj_rule)
        def c_rule(model, i):
            expr = 0
            for j in [1,2]:
                expr += j*model.x[j]
            return expr == 0
        model.c = Constraint([1,2], rule=c_rule)
        instance = model.create()
        self.assertEquals(instance.nvariables(), 2)
        self.assertEquals(instance.nobjectives(), 2)
        self.assertEquals(instance.nconstraints(), 2)

    def test_stats2(self):
        model = ConcreteModel()
        #
        model.x = Var([1,2])
        def obj_rule(model, i):
            return summation(model.x)
        model.y = VarList()
        model.y.add()
        model.y.add()
        #
        model.obj = Objective([1,2], rule=obj_rule)
        model.o = ObjectiveList()
        model.o.add(model.y[0])
        model.o.add(model.y[1])
        #
        def c_rule(model, i):
            expr = 0
            for j in [1,2]:
                expr += j*model.x[j]
            return expr == 0
        model.c = Constraint([1,2], rule=c_rule)
        model.C = ConstraintList()
        model.C.add(model.y[0] == 0)
        model.C.add(model.y[1] == 0)
        #
        instance = model.create()
        self.assertEquals(instance.nvariables(), 4)
        self.assertEquals(instance.nobjectives(), 4)
        self.assertEquals(instance.nconstraints(), 4)

    def test_stats3(self):
        model = ConcreteModel()
        model.x = Var([1,2])
        def obj_rule(model, i):
            return summation(model.x)
        model.obj = Objective([1,2], rule=obj_rule)
        def c_rule(model, i):
            expr = 0
            for j in [1,2]:
                expr += j*model.x[j]
            return expr == 0
        model.c = Constraint([1,2], rule=c_rule)
        #
        model.B = Block()
        model.B.x = Var([1,2])
        model.B.o = ObjectiveList()
        model.B.o.add(model.B.x[1])
        model.B.o.add(model.B.x[2])
        model.B.c = ConstraintList()
        model.B.c.add(model.x[1] == 0)
        model.B.c.add(model.x[2] == 0)
        instance = model.create()
        self.assertEquals(instance.nvariables(), 4)
        self.assertEquals(instance.nobjectives(), 4)
        self.assertEquals(instance.nconstraints(), 4)

if __name__ == "__main__":
    unittest.main()
