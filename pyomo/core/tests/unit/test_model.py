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
import pickle
import pyutilib.th as unittest
import pyutilib.services
import pyomo.opt
from pyomo.opt import SolutionStatus
from pyomo.opt.parallel.local import SolverManager_Serial
from pyomo.environ import *
from pyomo.core.base.expr import identify_variables

solvers = pyomo.opt.check_available_solvers('glpk')

try:
    import yaml
    yaml_available=True
except ImportError:
    yaml_available=False


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
        # Test coverage of the _clear_attribute method
        model = ConcreteModel()
        obj = Set()
        model.A = obj
        self.assertEqual(model.A.local_name, "A")
        self.assertEqual(obj.local_name, "A")
        self.assertIs(obj, model.A)

        obj = Var()
        model.A = obj
        self.assertEqual(model.A.local_name, "A")
        self.assertEqual(obj.local_name, "A")
        self.assertIs(obj, model.A)

        obj = Param()
        model.A = obj
        self.assertEqual(model.A.local_name, "A")
        self.assertEqual(obj.local_name, "A")
        self.assertIs(obj, model.A)

        obj = Objective()
        model.A = obj
        self.assertEqual(model.A.local_name, "A")
        self.assertEqual(obj.local_name, "A")
        self.assertIs(obj, model.A)

        obj = Constraint()
        model.A = obj
        self.assertEqual(model.A.local_name, "A")
        self.assertEqual(obj.local_name, "A")
        self.assertIs(obj, model.A)

        obj = Set()
        model.A = obj
        self.assertEqual(model.A.local_name, "A")
        self.assertEqual(obj.local_name, "A")
        self.assertIs(obj, model.A)

    def test_set_attr(self):
        model = ConcreteModel()
        model.x = Param(mutable=True)
        model.x = 5
        self.assertEqual(value(model.x), 5)
        model.x = 6
        self.assertEqual(value(model.x), 6)
        model.x = None
        self.assertEqual(model.x.value, None)

    def test_write(self):
        model = ConcreteModel()
        model.A = RangeSet(1,4)
        model.x = Var(model.A, bounds=(-1,1))
        def obj_rule(model):
            return summation(model.x)
        model.obj = Objective(rule=obj_rule)
        model.write()

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
        model.write()

    def test_write3(self):
        # Test that the summation works correctly, even though param 'w' has a default value
        model = ConcreteModel()
        model.J = RangeSet(1,4)
        model.w=Param(model.J, default=4)
        model.x=Var(model.J)
        def obj_rule(instance):
            return summation(instance.w, instance.x)
        model.obj = Objective(rule=obj_rule)
        self.assertEqual(len(model.obj[None].expr._args), 4)

    @unittest.skipIf(not 'glpk' in solvers, "glpk solver is not available")
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
        opt = SolverFactory('glpk')
        results = opt.solve(model, keepfiles=True, symbolic_solver_labels=True)
        model.solutions.store_to(results)
        results.write(filename=currdir+"solve1.out", format='json')
        self.assertMatchesJsonBaseline(currdir+"solve1.out",currdir+"solve1.txt", tolerance=1e-4)
        #
        def d_rule(model):
            return model.x[1] >= 0
        model.d = Constraint(rule=d_rule)
        model.d.deactivate()
        results = opt.solve(model, keepfiles=True)
        model.solutions.store_to(results)
        results.write(filename=currdir+"solve1x.out", format='json')
        self.assertMatchesJsonBaseline(currdir+"solve1x.out",currdir+"solve1.txt", tolerance=1e-4)
        #
        model.d.activate()
        results = opt.solve(model, keepfiles=True)
        model.solutions.store_to(results)
        results.write(filename=currdir+"solve1a.out", format='json')
        self.assertMatchesJsonBaseline(currdir+"solve1a.out",currdir+"solve1a.txt", tolerance=1e-4)
        #
        model.d.deactivate()
        def e_rule(model, i):
            return model.x[i] >= 0
        model.e = Constraint(model.A, rule=e_rule)
        for i in model.A:
            model.e[i].deactivate()
        results = opt.solve(model, keepfiles=True)
        model.solutions.store_to(results)
        results.write(filename=currdir+"solve1b.out", format='json')
        self.assertMatchesJsonBaseline(currdir+"solve1b.out",currdir+"solve1b.txt", tolerance=1e-4)

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
        model.display(currdir+"solve3.out")
        self.assertFileEqualsBaseline(currdir+"solve3.out",currdir+"solve3.txt")

    @unittest.skipIf(not 'glpk' in solvers, "glpk solver is not available")
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
        opt = SolverFactory('glpk')
        results = opt.solve(model, symbolic_solver_labels=True)
        model.solutions.store_to(results)
        results.write(filename=currdir+'solve4.out', format='json')
        self.assertMatchesJsonBaseline(currdir+"solve4.out",currdir+"solve1.txt", tolerance=1e-4)

    @unittest.skipIf(not 'glpk' in solvers, "glpk solver is not available")
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
        opt = SolverFactory('glpk')
        results = opt.solve(model, symbolic_solver_labels=True)
        model.solutions.store_to(results)
        results.write(filename=currdir+'solve6.out', format='json')
        self.assertMatchesJsonBaseline(currdir+"solve6.out", currdir+"solve6.txt", tolerance=1e-4)

    @unittest.skipIf(not 'glpk' in solvers, "glpk solver is not available")
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
        opt = SolverFactory('glpk')
        results = opt.solve(model, symbolic_solver_labels=True)
        #model.display()
        model.solutions.store_to(results)
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
        self.assertEqual(model.nvariables(), 2)
        self.assertEqual(model.nobjectives(), 2)
        self.assertEqual(model.nconstraints(), 2)

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
        model.o.add(model.y[1])
        model.o.add(model.y[2])
        #
        def c_rule(model, i):
            expr = 0
            for j in [1,2]:
                expr += j*model.x[j]
            return expr == 0
        model.c = Constraint([1,2], rule=c_rule)
        model.C = ConstraintList()
        model.C.add(model.y[1] == 0)
        model.C.add(model.y[2] == 0)
        #
        self.assertEqual(model.nvariables(), 4)
        self.assertEqual(model.nobjectives(), 4)
        self.assertEqual(model.nconstraints(), 4)

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
        self.assertEqual(model.nvariables(), 4)
        self.assertEqual(model.nobjectives(), 4)
        self.assertEqual(model.nconstraints(), 4)

    @unittest.skipIf(not 'glpk' in solvers, "glpk solver is not available")
    def test_solve_with_pickle(self):
        model = ConcreteModel()
        model.A = RangeSet(1,4)
        model.b = Block()
        model.b.x = Var(model.A, bounds=(-1,1))
        model.b.obj = Objective(expr=summation(model.b.x))
        model.c = Constraint(expr=model.b.x[1] >= 0)
        opt = SolverFactory('glpk')
        self.assertEqual(len(model.solutions), 0)
        results = opt.solve(model, symbolic_solver_labels=True)
        self.assertEqual(len(model.solutions), 1)
        #
        self.assertEqual(model.solutions[0].gap, 0.0)
        self.assertEqual(model.solutions[0].status, SolutionStatus.feasible)
        self.assertEqual(model.solutions[0].message, None)
        #
        buf = pickle.dumps(model)
        tmodel = pickle.loads(buf)
        self.assertEqual(len(tmodel.solutions), 1)
        self.assertEqual(tmodel.solutions[0].gap, 0.0)
        self.assertEqual(tmodel.solutions[0].status, SolutionStatus.feasible)
        self.assertEqual(tmodel.solutions[0].message, None)

    @unittest.skipIf(not 'glpk' in solvers, "glpk solver is not available")
    @unittest.skipIf(not yaml_available, "YAML not available available")
    def test_solve_with_store1(self):
        # With symbolic solver labels
        model = ConcreteModel()
        model.A = RangeSet(1,4)
        model.b = Block()
        model.b.x = Var(model.A, bounds=(-1,1))
        model.b.obj = Objective(expr=summation(model.b.x))
        model.c = Constraint(expr=model.b.x[1] >= 0)
        opt = SolverFactory('glpk')
        results = opt.solve(model, symbolic_solver_labels=True)
        #
        results.write(filename=currdir+'solve_with_store1.out', format='yaml')
        self.assertMatchesYamlBaseline(currdir+"solve_with_store1.out", currdir+"solve_with_store1.txt")
        model.solutions.store_to(results)
        #
        results.write(filename=currdir+'solve_with_store2.out', format='yaml')
        self.assertMatchesYamlBaseline(currdir+"solve_with_store2.out", currdir+"solve_with_store2.txt")
        #
        # Load results with string indices
        #
        tmodel = ConcreteModel()
        tmodel.A = RangeSet(1,4)
        tmodel.b = Block()
        tmodel.b.x = Var(tmodel.A, bounds=(-1,1))
        tmodel.b.obj = Objective(expr=summation(tmodel.b.x))
        tmodel.c = Constraint(expr=tmodel.b.x[1] >= 0)
        self.assertEqual(len(tmodel.solutions), 0)
        tmodel.solutions.load_from(results)
        self.assertEqual(len(tmodel.solutions), 1)

    @unittest.skipIf(not 'glpk' in solvers, "glpk solver is not available")
    @unittest.skipIf(not yaml_available, "YAML not available available")
    def test_solve_with_store2(self):
        # Without symbolic solver labels
        model = ConcreteModel()
        model.A = RangeSet(1,4)
        model.b = Block()
        model.b.x = Var(model.A, bounds=(-1,1))
        model.b.obj = Objective(expr=summation(model.b.x))
        model.c = Constraint(expr=model.b.x[1] >= 0)
        opt = SolverFactory('glpk')
        results = opt.solve(model, symbolic_solver_labels=False)
        #
        results.write(filename=currdir+'solve_with_store1.out', format='yaml')
        self.assertMatchesYamlBaseline(currdir+"solve_with_store1.out", currdir+"solve_with_store1.txt")
        model.solutions.store_to(results)
        #
        results.write(filename=currdir+'solve_with_store2.out', format='yaml')
        self.assertMatchesYamlBaseline(currdir+"solve_with_store2.out", currdir+"solve_with_store2.txt")
        #
        # Load results with string indices
        #
        tmodel = ConcreteModel()
        tmodel.A = RangeSet(1,4)
        tmodel.b = Block()
        tmodel.b.x = Var(tmodel.A, bounds=(-1,1))
        tmodel.b.obj = Objective(expr=summation(tmodel.b.x))
        tmodel.c = Constraint(expr=tmodel.b.x[1] >= 0)
        self.assertEqual(len(tmodel.solutions), 0)
        tmodel.solutions.load_from(results)
        self.assertEqual(len(tmodel.solutions), 1)

    @unittest.skipIf(not 'glpk' in solvers, "glpk solver is not available")
    @unittest.skipIf(not yaml_available, "YAML not available available")
    def test_solve_with_store2(self):
        model = ConcreteModel()
        model.A = RangeSet(1,4)
        model.b = Block()
        model.b.x = Var(model.A, bounds=(-1,1))
        model.b.obj = Objective(expr=summation(model.b.x))
        model.c = Constraint(expr=model.b.x[1] >= 0)
        opt = SolverFactory('glpk')
        results = opt.solve(model)
        #
        results.write(filename=currdir+'solve_with_store3.out', format='json')
        self.assertMatchesYamlBaseline(currdir+"solve_with_store3.out", currdir+"solve_with_store3.txt")
        #
        model.solutions.store_to(results)
        results.write(filename=currdir+'solve_with_store4.out', format='json')
        self.assertMatchesYamlBaseline(currdir+"solve_with_store4.out", currdir+"solve_with_store4.txt")
        #
        # Test that we can pickle the results object
        #
        buf = pickle.dumps(results)
        results_ = pickle.loads(buf)
        results.write(filename=currdir+'solve_with_store4.out', format='json')
        self.assertMatchesYamlBaseline(currdir+"solve_with_store4.out", currdir+"solve_with_store4.txt")
        #
        # Load results with string indices
        #
        tmodel = ConcreteModel()
        tmodel.A = RangeSet(1,3)
        tmodel.b = Block()
        tmodel.b.x = Var(tmodel.A, bounds=(-1,1))
        tmodel.b.obj = Objective(expr=summation(tmodel.b.x))
        tmodel.c = Constraint(expr=tmodel.b.x[1] >= 0)
        self.assertEqual(len(tmodel.solutions), 0)
        tmodel.solutions.load_from(results, ignore_invalid_labels=True)
        self.assertEqual(len(tmodel.solutions), 1)

    @unittest.skipIf(not 'glkp' in solvers, "glpk solver is not available")
    @unittest.skipIf(not yaml_available, "YAML not available available")
    def test_solve_with_store3(self):
        model = ConcreteModel()
        model.A = RangeSet(1,4)
        model.b = Block()
        model.b.x = Var(model.A, bounds=(-1,1))
        model.b.obj = Objective(expr=summation(model.b.x))
        model.c = Constraint(expr=model.b.x[1] >= 0)
        opt = SolverFactory('glpk')
        results = opt.solve(model)
        #
        model.solutions.store_to(results)
        results.write(filename=currdir+'solve_with_store5.out', format='json')
        self.assertMatchesYamlBaseline(currdir+"solve_with_store5.out", currdir+"solve_with_store4.txt")
        #
        model.solutions.store_to(results, cuid=True)
        buf = pickle.dumps(results)
        results_ = pickle.loads(buf)
        model.solutions.load_from(results_)
        model.solutions.store_to(results_)
        results_.write(filename=currdir+'solve_with_store6.out', format='json')
        self.assertMatchesYamlBaseline(currdir+"solve_with_store6.out", currdir+"solve_with_store4.txt")
        #
        # Load results with string indices
        #
        tmodel = ConcreteModel()
        tmodel.A = RangeSet(1,4)
        tmodel.b = Block()
        tmodel.b.x = Var(tmodel.A, bounds=(-1,1))
        tmodel.b.obj = Objective(expr=summation(tmodel.b.x))
        tmodel.c = Constraint(expr=tmodel.b.x[1] >= 0)
        self.assertEqual(len(tmodel.solutions), 0)
        tmodel.solutions.load_from(results)
        self.assertEqual(len(tmodel.solutions), 1)
        tmodel.solutions.store_to(results)
        results.write(filename=currdir+'solve_with_store7.out', format='json')
        self.assertMatchesYamlBaseline(currdir+"solve_with_store7.out", currdir+"solve_with_store4.txt")

    @unittest.skipIf(not 'glpk' in solvers, "glpk solver is not available")
    @unittest.skipIf(not yaml_available, "YAML not available available")
    def test_solve_with_store4(self):
        model = ConcreteModel()
        model.A = RangeSet(1,4)
        model.b = Block()
        model.b.x = Var(model.A, bounds=(-1,1))
        model.b.obj = Objective(expr=summation(model.b.x))
        model.c = Constraint(expr=model.b.x[1] >= 0)
        opt = SolverFactory('glpk')
        results = opt.solve(model, load_solutions=False)
        self.assertEqual(len(model.solutions), 0)
        self.assertEqual(len(results.solution), 1)
        model.solutions.load_from(results)
        self.assertEqual(len(model.solutions), 1)
        self.assertEqual(len(results.solution), 1)
        #
        model.solutions.store_to(results)
        results.write(filename=currdir+'solve_with_store8.out', format='json')
        self.assertMatchesYamlBaseline(currdir+"solve_with_store8.out", currdir+"solve_with_store4.txt")

    @unittest.skipIf(not 'glpk' in solvers, "glpk solver is not available")
    @unittest.skipIf(not yaml_available, "YAML not available available")
    def test_solve_with_store5(self):
        model = ConcreteModel()
        model.A = RangeSet(1,4)
        model.b = Block()
        model.b.x = Var(model.A, bounds=(-1,1))
        model.b.obj = Objective(expr=summation(model.b.x))
        model.c = Constraint(expr=model.b.x[1] >= 0)

        smanager = SolverManager_Serial()
        ah = smanager.queue(model, solver='glpk', load_solutions=False)
        results = smanager.wait_for(ah)
        self.assertEqual(len(model.solutions), 0)
        self.assertEqual(len(results.solution), 1)
        model.solutions.load_from(results)
        self.assertEqual(len(model.solutions), 1)
        self.assertEqual(len(results.solution), 1)
        #
        model.solutions.store_to(results)
        results.write(filename=currdir+'solve_with_store8.out', format='json')
        self.assertMatchesYamlBaseline(currdir+"solve_with_store8.out", currdir+"solve_with_store4.txt")


    def test_create_concrete_from_rule(self):
        def make(m):
            m.I = RangeSet(3)
            m.x = Var(m.I)
            m.c = Constraint( expr=sum(m.x[i] for i in m.I) >= 0 )
        model = ConcreteModel(rule=make)
        self.assertEqual( [x.local_name for x in model.component_objects()],
                          ['I','x','c'] )
        self.assertEqual( len(list(identify_variables(model.c.body))), 3 )


    def test_create_abstract_from_rule(self):
        def make_invalid(m):
            m.I = RangeSet(3)
            m.x = Var(m.I)
            m.c = Constraint( expr=sum(m.x[i] for i in m.I) >= 0 )

        def make(m):
            m.I = RangeSet(3)
            m.x = Var(m.I)
            def c(b):
                return sum(m.x[i] for i in m.I) >= 0
            m.c = Constraint( rule=c )

        model = AbstractModel(rule=make_invalid)
        self.assertRaises(RuntimeError, model.create_instance)

        model = AbstractModel(rule=make)
        instance = model.create_instance()
        self.assertEqual( [x.local_name for x in model.component_objects()],
                          [] )
        self.assertEqual( [x.local_name for x in instance.component_objects()],
                          ['I','x','c'] )
        self.assertEqual( len(list(identify_variables(instance.c.body))), 3 )

        model = AbstractModel(rule=make)
        model.y = Var()
        instance = model.create_instance()
        self.assertEqual( [x.local_name for x in instance.component_objects()],
                          ['y','I','x','c'] )
        self.assertEqual( len(list(identify_variables(instance.c.body))), 3 )

    def test_error1(self):
        model = ConcreteModel()
        model.x = Var()
        instance = model.create_instance()

if __name__ == "__main__":
    unittest.main()

