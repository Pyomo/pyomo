#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
#  SOS tests
#

import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest

from pyomo.core.base import IntegerSet
from pyomo.environ import *


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


class TestSimple(unittest.TestCase):

    def setUp(self):
        #
        # Create Model
        #
        self.M = ConcreteModel()

    def tearDown(self):
        pass

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
        self.assertEqual(set(id(v) for v in self.M.c.get_variables()), set(id(v) for v in self.M.x.values()))


class TestExamples(unittest.TestCase):

    def test1(self):
        M = ConcreteModel()
        M.x = Var(xrange(20))
        M.c = SOSConstraint(var=M.x, sos=1)
        self.assertEqual(set((v.cname(True),w) for v,w in M.c.get_items()), set((M.x[i].cname(True), i+1) for i in xrange(20)))

    def test2(self):
        # Use an index set, which is a subset of M.x.index_set()
        M = ConcreteModel()
        M.x = Var(xrange(20))
        M.c = SOSConstraint(var=M.x, index=xrange(10), sos=1)
        self.assertEqual(set((v.cname(True),w) for v,w in M.c.get_items()), set((M.x[i].cname(True), i+1) for i in xrange(10)))

    def test3(self):
        # User-specified weights
        w = {1:-1, 2:2, 3:-3}
        M = ConcreteModel()
        M.x = Var([1,2,3])
        M.c = SOSConstraint(var=M.x, weights=w, sos=1)
        self.assertEqual(set((v.cname(True),w) for v,w in M.c.get_items()), set((M.x[i].cname(True), w[i]) for i in [1,2,3]))

    def test10(self):
        M = ConcreteModel()
        M.x = Var([1,2,3])
        M.c = SOSConstraint([0,1], var=M.x, sos=1)
        self.assertEqual(set((v.cname(True),w) for v,w in M.c[0].get_items()), set((M.x[i].cname(True), i) for i in [1,2,3]))
        self.assertEqual(set((v.cname(True),w) for v,w in M.c[1].get_items()), set((M.x[i].cname(True), i) for i in [1,2,3]))

    def test11(self):
        w = {1:-1, 2:2, 3:-3}
        M = ConcreteModel()
        M.x = Var([1,2,3])
        M.c = SOSConstraint([0,1], var=M.x, weights=w, sos=1)
        self.assertEqual(set((v.cname(True),w) for v,w in M.c[0].get_items()), set((M.x[i].cname(True), w[i]) for i in [1,2,3]))
        self.assertEqual(set((v.cname(True),w) for v,w in M.c[1].get_items()), set((M.x[i].cname(True), w[i]) for i in [1,2,3]))


if __name__ == "__main__":
    unittest.main()
