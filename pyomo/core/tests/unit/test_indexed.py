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
# Unit Tests for indexed components
#

import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyomo.common.unittest as unittest

from pyomo.environ import ConcreteModel, Var, Param, Set, value
from pyomo.core.base.indexed_component import normalize_index


class TestSimpleVar(unittest.TestCase):

    def test0(self):
        # Test fixed attribute - 1D
        m = ConcreteModel()
        m.x = Var()

        names = set()
        for var in m.x[:]:
            names.add(var.name)
        self.assertEqual(names, set(['x']))

    def test1(self):
        # Test fixed attribute - 1D
        m = ConcreteModel()
        m.x = Var(range(3), dense=True)

        names = set()
        for var in m.x[:]:
            names.add(var.name)
        self.assertEqual(names, set(['x[0]', 'x[1]', 'x[2]']))

    def test2a(self):
        # Test fixed attribute - 2D
        m = ConcreteModel()
        m.x = Var(range(3), range(3), dense=True)

        names = set()
        for var in m.x[:, 1]:
            names.add(var.name)
        self.assertEqual(names, set(['x[0,1]', 'x[1,1]', 'x[2,1]']))

    def test2b(self):
        # Test fixed attribute - 2D
        m = ConcreteModel()
        m.x = Var(range(3), range(3), dense=True)

        names = set()
        for var in m.x[2, :]:
            names.add(var.name)
        self.assertEqual(names, set(['x[2,0]', 'x[2,1]', 'x[2,2]']))

    def test2c(self):
        # Test fixed attribute - 2D
        m = ConcreteModel()
        m.x = Var(range(3), range(3), dense=True)

        names = set()
        for var in m.x[3, :]:
            names.add(var.name)
        self.assertEqual(names, set())

    def test3a(self):
        # Test fixed attribute - 3D
        m = ConcreteModel()
        m.x = Var(range(3), range(3), range(3), dense=True)

        names = set()
        for var in m.x[:, 1, :]:
            names.add(var.name)
        self.assertEqual(names, set(['x[0,1,0]', 'x[0,1,1]', 'x[0,1,2]', 'x[1,1,0]', 'x[1,1,1]', 'x[1,1,2]', 'x[2,1,0]', 'x[2,1,1]', 'x[2,1,2]' ]))

    def test3b(self):
        # Test fixed attribute - 3D
        m = ConcreteModel()
        m.x = Var(range(3), range(3), range(3), dense=True)

        names = set()
        for var in m.x[0, :, 2]:
            names.add(var.name)
        self.assertEqual(names, set(['x[0,0,2]', 'x[0,1,2]', 'x[0,2,2]']))


class TestIndexedComponent(unittest.TestCase):
    def test_normalize_index(self):
        # Test that normalize_index works correctly
        self.assertEqual("abc", normalize_index("abc"))
        self.assertEqual(1, normalize_index(1))
        self.assertEqual(1, normalize_index([1]))
        self.assertEqual((1, 2, 3), normalize_index((1, 2, 3)))
        self.assertEqual((1, 2, 3), normalize_index([1, 2, 3]))
        self.assertEqual((1, 2, 3, 4), normalize_index((1, 2, [3, 4])))
        self.assertEqual((1, 2, 'abc'), normalize_index((1, 2, 'abc')))
        self.assertEqual((1, 2, 'abc'), normalize_index((1, 2, ('abc',))))
        a = [0, 9, 8]
        self.assertEqual((1, 2, 0, 9, 8), normalize_index((1, 2, a)))
        self.assertEqual((1, 2, 3, 4, 5), normalize_index(
            [[], 1, [], 2, [[], 3, [[], 4, []], []], 5, []]))
        self.assertEqual((), normalize_index([[[[], []], []], []]))
        self.assertEqual((), normalize_index([[], [[], [[],]]]))

        # Test that normalize_index doesn't expand component-like things
        m = ConcreteModel()
        m.x = Var()
        m.y = Var([1])
        m.i = Set(initialize=[1])
        m.j = Set([1], initialize=[1])
        self.assertIs(m, normalize_index(m))
        self.assertIs(m.x, normalize_index(m.x))
        self.assertIs(m.y, normalize_index(m.y))
        self.assertIs(m.y[1], normalize_index(m.y[1]))
        self.assertIs(m.i, normalize_index(m.i))
        self.assertIs(m.j, normalize_index(m.j))
        self.assertIs(m.j[1], normalize_index(m.j[1]))

    def test_index_by_constant_simpleComponent(self):
        m = ConcreteModel()
        m.i = Param(initialize=2)
        m.x = Var([1,2,3], initialize=lambda m,x: 2*x)
        self.assertEqual(value(m.x[2]), 4)
        self.assertEqual(value(m.x[m.i]), 4)
        self.assertIs(m.x[2], m.x[m.i])

    def test_index_by_multiple_constant_simpleComponent(self):
        m = ConcreteModel()
        m.i = Param(initialize=2)
        m.j = Param(initialize=3)
        m.x = Var([1,2,3], [1,2,3], initialize=lambda m,x,y: 2*x*y)
        self.assertEqual(value(m.x[2,3]), 12)
        self.assertEqual(value(m.x[m.i,3]), 12)
        self.assertEqual(value(m.x[m.i,m.j]), 12)
        self.assertEqual(value(m.x[2,m.j]), 12)
        self.assertIs(m.x[2,3], m.x[m.i,3])
        self.assertIs(m.x[2,3], m.x[m.i,m.j])
        self.assertIs(m.x[2,3], m.x[2,m.j])

    def test_index_by_fixed_simpleComponent(self):
        m = ConcreteModel()
        m.i = Param(initialize=2, mutable=True)
        m.x = Var([1,2,3], initialize=lambda m,x: 2*x)
        self.assertEqual(value(m.x[2]), 4)
        self.assertRaisesRegex(
            RuntimeError, 'is a fixed but not constant value',
            m.x.__getitem__, m.i)

    def test_index_by_variable_simpleComponent(self):
        m = ConcreteModel()
        m.i = Var(initialize=2)
        m.x = Var([1,2,3], initialize=lambda m,x: 2*x)
        self.assertEqual(value(m.x[2]), 4)
        self.assertRaisesRegex(
            RuntimeError, 'is not a constant value',
            m.x.__getitem__, m.i)

    def test_index_by_unhashable_type(self):
        m = ConcreteModel()
        m.x = Var([1,2,3], initialize=lambda m,x: 2*x)
        self.assertRaisesRegex(
            TypeError, '.*',
            m.x.__getitem__, {})

    def test_ordered_keys(self):
        m = ConcreteModel()
        # Pick a set whose unordered iteration order should never match
        # the "ordered" iteration order.
        init_keys = [2, 1, (1, 2), (1, 'a'), (1, 1)]
        m.I = Set(ordered=False, dimen=None, initialize=init_keys)
        ordered_keys = [1, 2, (1, 1), (1, 2), (1, 'a')]
        m.x = Var(m.I)
        self.assertNotEqual(list(m.x.keys()), list(m.x.keys(True)))
        self.assertEqual(set(m.x.keys()), set(m.x.keys(True)))
        self.assertEqual(ordered_keys, list(m.x.keys(True)))

        m.P = Param(m.I, initialize={k:v for v,k in enumerate(init_keys)})
        self.assertNotEqual(list(m.P.keys()), list(m.P.keys(True)))
        self.assertEqual(set(m.P.keys()), set(m.P.keys(True)))
        self.assertEqual(ordered_keys, list(m.P.keys(True)))
        self.assertEqual([1, 0, 4, 2, 3], list(m.P.values(True)))
        self.assertEqual(list(zip(ordered_keys, [1, 0, 4, 2, 3])),
                         list(m.P.items(True)))

        m.P = Param(m.I, initialize={(1,2): 30, 1:10, 2:20}, default=1)
        self.assertNotEqual(list(m.P.keys()), list(m.P.keys(True)))
        self.assertEqual(set(m.P.keys()), set(m.P.keys(True)))
        self.assertEqual(ordered_keys, list(m.P.keys(True)))
        self.assertEqual([10, 20, 1, 30, 1], list(m.P.values(True)))
        self.assertEqual(list(zip(ordered_keys, [10, 20, 1, 30, 1])),
                         list(m.P.items(True)))


if __name__ == "__main__":
    unittest.main()
