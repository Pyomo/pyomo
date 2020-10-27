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
#  SOS tests
#

import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep
from six.moves import xrange
import pyutilib.th as unittest
from pyomo.environ import ConcreteModel, AbstractModel, SOSConstraint, Var, Set


class TestErrors(unittest.TestCase):

    def test_arg1(self):
        M = ConcreteModel()
        try:
            M.c = SOSConstraint()
            self.fail("Expected TypeError")
        except TypeError:
            pass

    def test_arg2(self):
        M = ConcreteModel()
        M.v = Var()
        try:
            M.c = SOSConstraint(var=M.v, sos=1, level=1)
            self.fail("Expected TypeError")
        except TypeError:
            pass

    def test_arg3(self):
        M = ConcreteModel()
        M.v = Var()
        try:
            M.c = SOSConstraint(var=M.v)
            self.fail("Expected TypeError")
        except TypeError:
            pass

    def test_negative_weights(self):
        M = ConcreteModel()
        M.v = Var()
        try:
            M.c = SOSConstraint(var=M.v, weights={None:-1}, sos=1)
            self.fail("Expected ValueError")
        except ValueError:
            pass

    def test_ordered(self):
        M = ConcreteModel()
        M.v = Var({1,2,3})
        try:
            M.c = SOSConstraint(var=M.v, sos=2)
            self.fail("Expected ValueError")
        except ValueError:
            pass
        M = ConcreteModel()
        M.s = Set(initialize=[1,2,3], ordered=True)
        M.v = Var(M.s)
        M.c = SOSConstraint(var=M.v, sos=2)


class TestSimple(unittest.TestCase):

    def setUp(self):
        #
        # Create Model
        #
        self.M = ConcreteModel()

    def tearDown(self):
        self.M = None

    def test_num_vars(self):
        # Test the number of variables
        self.M.x = Var([1,2,3])
        self.M.c = SOSConstraint(var=self.M.x, sos=1)
        self.assertEqual(self.M.c.num_variables(), 3)

    def test_level(self):
        # Test level property
        self.M.x = Var([1,2,3])
        self.M.c = SOSConstraint(var=self.M.x, sos=1)
        self.assertEqual(self.M.c.level, 1)
        self.M.c.level = 2
        self.assertEqual(self.M.c.level, 2)
        try:
            self.M.c.level = -1
            self.fail("Expected ValueError")
        except ValueError:
            pass

    def test_get_variables(self):
        # Test that you get the correct variables
        self.M.x = Var([1,2,3])
        self.M.c = SOSConstraint(var=self.M.x, sos=1)
        self.assertEqual(set(id(v) for v in self.M.c.get_variables()),
                         set(id(v) for v in self.M.x.values()))


class TestExamples(unittest.TestCase):

    def test1(self):
        M = ConcreteModel()
        M.x = Var(xrange(20))
        M.c = SOSConstraint(var=M.x, sos=1)
        self.assertEqual(set((v.name,w) for v,w in M.c.get_items()),
                         set((M.x[i].name, i+1) for i in xrange(20)))

    def test2(self):
        # Use an index set, which is a subset of M.x.index_set()
        M = ConcreteModel()
        M.x = Var(xrange(20))
        M.c = SOSConstraint(var=M.x, index=list(xrange(10)), sos=1)
        self.assertEqual(set((v.name,w) for v,w in M.c.get_items()),
                         set((M.x[i].name, i+1) for i in xrange(10)))

    def test3(self):
        # User-specified weights
        w = {1:10, 2:2, 3:30}
        M = ConcreteModel()
        M.x = Var([1,2,3])
        M.c = SOSConstraint(var=M.x, weights=w, sos=1)
        self.assertEqual(set((v.name,w) for v,w in M.c.get_items()),
                         set((M.x[i].name, w[i]) for i in [1,2,3]))

    def test4(self):
        # User-specified weights
        w = {1:10, 2:2, 3:30}
        def rule(model):
            return list(M.x[i] for i in M.x), [10, 2, 30]
        M = ConcreteModel()
        M.x = Var([1,2,3], dense=True)
        M.c = SOSConstraint(rule=rule, sos=1)
        self.assertEqual(set((v.name,w) for v,w in M.c.get_items()),
                         set((M.x[i].name, w[i]) for i in [1,2,3]))

    def test10(self):
        M = ConcreteModel()
        M.x = Var([1,2,3])
        M.c = SOSConstraint([0,1], var=M.x, sos=1, index={0:[1,2], 1:[2,3]})
        self.assertEqual(set((v.name,w) for v,w in M.c[0].get_items()),
                         set((M.x[i].name, i) for i in [1,2]))
        self.assertEqual(set((v.name,w) for v,w in M.c[1].get_items()),
                         set((M.x[i].name, i-1) for i in [2,3]))

    def test11(self):
        w = {1:10, 2:2, 3:30}
        M = ConcreteModel()
        M.x = Var([1,2,3], dense=True)
        M.c = SOSConstraint([0,1], var=M.x, weights=w, sos=1, index={0:[1,2], 1:[2,3]})
        self.assertEqual(set((v.name,w) for v,w in M.c[0].get_items()),
                         set((M.x[i].name, w[i]) for i in [1,2]))
        self.assertEqual(set((v.name,w) for v,w in M.c[1].get_items()),
                         set((M.x[i].name, w[i]) for i in [2,3]))

    def test12(self):
        def rule(model, i):
            if i == 0:
                return list(M.x[i] for i in M.x), [10, 2, 30]
            else:
                return list(M.x[i] for i in M.x), [1, 20, 3]
        w = {0:{1:10, 2:2, 3:30}, 1:{1:1, 2:20, 3:3}}
        M = ConcreteModel()
        M.x = Var([1,2,3], dense=True)
        M.c = SOSConstraint([0,1], rule=rule, sos=1)
        self.assertEqual(set((v.name,w) for v,w in M.c[0].get_items()),
                         set((M.x[i].name, w[0][i]) for i in [1,2,3]))
        self.assertEqual(set((v.name,w) for v,w in M.c[1].get_items()),
                         set((M.x[i].name, w[1][i]) for i in [1,2,3]))

    def test13(self):
        I = {0:[1,2], 1:[2,3]}
        M = ConcreteModel()
        M.x = Var([1,2,3], dense=True)
        M.c = SOSConstraint([0,1], var=M.x, index=I, sos=1)
        self.assertEqual(set((v.name,w) for v,w in M.c[0].get_items()),
                         set((M.x[i].name, i) for i in I[0]))
        self.assertEqual(set((v.name,w) for v,w in M.c[1].get_items()),
                         set((M.x[i].name, i-1) for i in I[1]))

    def test14(self):
        def rule(model, i):
            if i == 0:
                return SOSConstraint.Skip
            else:
                return list(M.x[i] for i in M.x), [1, 20, 3]
        w = {0:{1:10, 2:2, 3:30}, 1:{1:1, 2:20, 3:3}}
        M = ConcreteModel()
        M.x = Var([1,2,3], dense=True)
        M.c = SOSConstraint([0,1], rule=rule, sos=1)
        self.assertEqual(list(M.c.keys()), [1])
        self.assertEqual(set((v.name,w) for v,w in M.c[1].get_items()),
                         set((M.x[i].name, w[1][i]) for i in [1,2,3]))

    def test_abstract_index(self):
        model = AbstractModel()
        model.A = Set(initialize=[0])
        model.B = Set(initialize=[1])
        model.C = model.A | model.B
        M = ConcreteModel()
        M.x = Var([1,2,3])
        M.c = SOSConstraint(model.C, var=M.x, sos=1, index={0:[1,2], 1:[2,3]})


if __name__ == "__main__":
    unittest.main()
