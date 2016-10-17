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
from pyomo.core.base.block import _BlockData
from pyomo.core.base.indexed_component import _IndexedComponent_slicer


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
    def test_index_by_constant_simpleComponent(self):
        m = ConcreteModel()
        m.i = Param(initialize=2)
        m.x = Var([1,2,3], initialize=lambda m,x: 2*x)
        self.assertEqual(m.x[2], 4)
        self.assertEqual(m.x[m.i], 4)

    def test_index_by_multiple_constant_simpleComponent(self):
        m = ConcreteModel()
        m.i = Param(initialize=2)
        m.j = Param(initialize=3)
        m.x = Var([1,2,3], [1,2,3], initialize=lambda m,x,y: 2*x*y)
        self.assertEqual(m.x[2,3], 12)
        self.assertEqual(m.x[m.i,3], 12)
        self.assertEqual(m.x[m.i,m.j], 12)
        self.assertEqual(m.x[2,m.j], 12)

    def test_index_by_fixed_simpleComponent(self):
        m = ConcreteModel()
        m.i = Param(initialize=2, mutable=True)
        m.x = Var([1,2,3], initialize=lambda m,x: 2*x)
        self.assertEqual(m.x[2], 4)
        self.assertRaisesRegexp(
            RuntimeError, 'is a fixed but not constant value',
            m.x.__getitem__, m.i)

    def test_index_by_variable_simpleComponent(self):
        m = ConcreteModel()
        m.i = Var(initialize=2)
        m.x = Var([1,2,3], initialize=lambda m,x: 2*x)
        self.assertEqual(m.x[2], 4)
        self.assertRaisesRegexp(
            RuntimeError, 'is not a constant value',
            m.x.__getitem__, m.i)

    def test_index_by_unhashable_type(self):
        m = ConcreteModel()
        m.i = Var(initialize=2)
        m.x = Var([1,2,3], initialize=lambda m,x: 2*x)
        self.assertEqual(m.x[2], 4)
        self.assertRaisesRegexp(
            TypeError, 'found when trying to retrieve index for component',
            m.x.__getitem__, {})


class TestComponentSlices(unittest.TestCase):
    def setUp(self):
        def _c(b, i, j):
            b.x = Var(b.model().K)

        def _b(b, i, j):
            _c(b,i,j)
            b.c = Block(b.model().I, b.model().J, rule=_c)
        def _bb(b, i, j, k):
            _c(b,i,j)
            b.c = Block(b.model().I, b.model().J, rule=_c)

        self.m = m = ConcreteModel()
        m.I = RangeSet(1,3)
        m.J = RangeSet(4,6)
        m.K = RangeSet(7,9)
        m.b = Block(m.I, m.J, rule=_b)
        m.bb = Block(m.I, m.J, m.K, rule=_bb)

    def tearDown(self):
        self.m = None

    def test_simple_getitem(self):
        self.assertTrue(isinstance(self.m.b[1,4], _BlockData))

    def test_simple_getslice(self):
        _slicer = self.m.b[:,4]
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, ['b[1,4]', 'b[2,4]', 'b[3,4]'] )

        _slicer = self.m.b[1,4].c[:,4]
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, ['b[1,4].c[1,4]', 'b[1,4].c[2,4]', 'b[1,4].c[3,4]'] )

    def test_wildcard_slice(self):
        _slicer = self.m.b[:]
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        ans = [ str(x) for x in _slicer ]
        self.assertEqual( ans, [] )

        _slicer = self.m.b[...]
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, [ 'b[1,4]', 'b[1,5]', 'b[1,6]',
                   'b[2,4]', 'b[2,5]', 'b[2,6]',
                   'b[3,4]', 'b[3,5]', 'b[3,6]',
               ] )

        _slicer = self.m.b[1,...]
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, [ 'b[1,4]', 'b[1,5]', 'b[1,6]',
               ] )

        _slicer = self.m.b[...,5]
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, [ 'b[1,5]',
                   'b[2,5]',
                   'b[3,5]',
               ] )

        _slicer = self.m.bb[2,...,8]
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, [ 'bb[2,4,8]', 'bb[2,5,8]', 'bb[2,6,8]',
               ] )

        _slicer = self.m.bb[:,...,8]
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, [ 'bb[1,4,8]', 'bb[1,5,8]', 'bb[1,6,8]',
                   'bb[2,4,8]', 'bb[2,5,8]', 'bb[2,6,8]',
                   'bb[3,4,8]', 'bb[3,5,8]', 'bb[3,6,8]',
               ] )

        _slicer = self.m.bb[:,:,...,8]
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, [ 'bb[1,4,8]', 'bb[1,5,8]', 'bb[1,6,8]',
                   'bb[2,4,8]', 'bb[2,5,8]', 'bb[2,6,8]',
                   'bb[3,4,8]', 'bb[3,5,8]', 'bb[3,6,8]',
               ] )

        _slicer = self.m.bb[:,...,:,8]
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, [ 'bb[1,4,8]', 'bb[1,5,8]', 'bb[1,6,8]',
                   'bb[2,4,8]', 'bb[2,5,8]', 'bb[2,6,8]',
                   'bb[3,4,8]', 'bb[3,5,8]', 'bb[3,6,8]',
               ] )

        _slicer = self.m.b[1,4,...]
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, [ 'b[1,4]',
               ] )

        _slicer = self.m.b[1,2,3,...]
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        ans = [ str(x) for x in _slicer ]
        self.assertEqual( ans, [] )

        _slicer = self.m.b[1,:,2]
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        ans = [ str(x) for x in _slicer ]
        self.assertEqual( ans, [] )

        self.assertRaisesRegexp(
            IndexError, 'wildcard slice .* can only appear once',
            self.m.b.__getitem__, (Ellipsis,Ellipsis) )


    def test_nonterminal_slice(self):
        _slicer = self.m.b[:,4].x
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, ['b[1,4].x', 'b[2,4].x', 'b[3,4].x'] )

        _slicer = self.m.b[:,4].x[7]
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, ['b[1,4].x[7]', 'b[2,4].x[7]', 'b[3,4].x[7]'] )

    def test_nested_slices(self):
        _slicer = self.m.b[1,:].c[:,4].x
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, ['b[1,4].c[1,4].x', 'b[1,4].c[2,4].x', 'b[1,4].c[3,4].x',
                  'b[1,5].c[1,4].x', 'b[1,5].c[2,4].x', 'b[1,5].c[3,4].x',
                  'b[1,6].c[1,4].x', 'b[1,6].c[2,4].x', 'b[1,6].c[3,4].x',
              ] )

        _slicer = self.m.b[1,:].c[:,4].x[8]
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans,
            [ 'b[1,4].c[1,4].x[8]', 'b[1,4].c[2,4].x[8]', 'b[1,4].c[3,4].x[8]',
              'b[1,5].c[1,4].x[8]', 'b[1,5].c[2,4].x[8]', 'b[1,5].c[3,4].x[8]',
              'b[1,6].c[1,4].x[8]', 'b[1,6].c[2,4].x[8]', 'b[1,6].c[3,4].x[8]',
               ] )

    def test_function_slices(self):
        _slicer = self.m.component('b')[1,:].component('c')[:,4].component('x')
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, ['b[1,4].c[1,4].x', 'b[1,4].c[2,4].x', 'b[1,4].c[3,4].x',
                  'b[1,5].c[1,4].x', 'b[1,5].c[2,4].x', 'b[1,5].c[3,4].x',
                  'b[1,6].c[1,4].x', 'b[1,6].c[2,4].x', 'b[1,6].c[3,4].x',
              ] )


    def test_empty_slices(self):
        _slicer = self.m.b[1,:].c[:,1].x
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        ans = [ str(x) for x in _slicer ]
        self.assertEqual( ans, [] )

        _slicer = self.m.b[1,:].c[:,4].x[1]
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        _slicer.key_errors_generate_exceptions = False
        ans = [ str(x) for x in _slicer ]
        self.assertEqual( ans, [] )

        _slicer = self.m.b[1,:].c[:,4].y
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        _slicer.attribute_errors_generate_exceptions = False
        ans = [ str(x) for x in _slicer ]
        self.assertEqual( ans, [] )

        _slicer = self.m.b[1,:].c[:,4].component('y', False)
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        _slicer.call_errors_generate_exceptions = False
        ans = [ str(x) for x in _slicer ]
        self.assertEqual( ans, [] )

        _slicer = self.m.b[1,:].c[:,4].x[1]
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        _slicer.key_errors_generate_exceptions = True
        self.assertRaises( KeyError, _slicer.next )

        _slicer = self.m.b[1,:].c[:,4].y
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        _slicer.attribute_errors_generate_exceptions = True
        self.assertRaises( AttributeError, _slicer.next )

        _slicer = self.m.b[1,:].c[:,4].component('y', False)
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        _slicer.call_errors_generate_exceptions = True
        self.assertRaises( TypeError,_slicer.next )

        _slicer = self.m.b[1,:].c[:,4].component()
        self.assertTrue(isinstance(_slicer, _IndexedComponent_slicer))
        _slicer.call_errors_generate_exceptions = True
        self.assertRaises( TypeError, _slicer.next )


if __name__ == "__main__":
    unittest.main()
