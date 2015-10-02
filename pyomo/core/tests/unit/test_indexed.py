#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Unit Tests for indexed components
#

import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest

from pyomo.environ import *


class TestSimpleVar(unittest.TestCase):

    def test0(self):
        # Test fixed attribute - 1D
        m = ConcreteModel()
        m.x = Var()

        names = set()
        for var in m.x[:]:
            names.add(var.cname(True))
        self.assertEqual(names, set(['x']))

    def test1(self):
        # Test fixed attribute - 1D
        m = ConcreteModel()
        m.x = Var(range(3), dense=True)

        names = set()
        for var in m.x[:]:
            names.add(var.cname(True))
        self.assertEqual(names, set(['x[0]', 'x[1]', 'x[2]']))

    def test2a(self):
        # Test fixed attribute - 2D
        m = ConcreteModel()
        m.x = Var(range(3), range(3), dense=True)

        names = set()
        for var in m.x[:, 1]:
            names.add(var.cname(True))
        self.assertEqual(names, set(['x[0,1]', 'x[1,1]', 'x[2,1]']))

    def test2b(self):
        # Test fixed attribute - 2D
        m = ConcreteModel()
        m.x = Var(range(3), range(3), dense=True)

        names = set()
        for var in m.x[2, :]:
            names.add(var.cname(True))
        self.assertEqual(names, set(['x[2,0]', 'x[2,1]', 'x[2,2]']))

    def test2c(self):
        # Test fixed attribute - 2D
        m = ConcreteModel()
        m.x = Var(range(3), range(3), dense=True)

        names = set()
        for var in m.x[3, :]:
            names.add(var.cname(True))
        self.assertEqual(names, set())

    def test3a(self):
        # Test fixed attribute - 3D
        m = ConcreteModel()
        m.x = Var(range(3), range(3), range(3), dense=True)

        names = set()
        for var in m.x[:, 1, :]:
            names.add(var.cname(True))
        self.assertEqual(names, set(['x[0,1,0]', 'x[0,1,1]', 'x[0,1,2]', 'x[1,1,0]', 'x[1,1,1]', 'x[1,1,2]', 'x[2,1,0]', 'x[2,1,1]', 'x[2,1,2]' ]))

    def test3b(self):
        # Test fixed attribute - 3D
        m = ConcreteModel()
        m.x = Var(range(3), range(3), range(3), dense=True)

        names = set()
        for var in m.x[0, :, 2]:
            names.add(var.cname(True))
        self.assertEqual(names, set(['x[0,0,2]', 'x[0,1,2]', 'x[0,2,2]']))


if __name__ == "__main__":
    unittest.main()
