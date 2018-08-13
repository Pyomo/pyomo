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

import pyutilib.th as unittest
from six import itervalues

from pyomo.environ import *
from pyomo.core.base.block import _BlockData
from pyomo.core.base.indexed_component import _IndexedComponent_slice
from pyomo.core.base.indexed_component_slice import _ReferenceDict


class TestComponentSlices(unittest.TestCase):
    def setUp(self):
        def _c(b, i, j):
            b.x = Var(b.model().K, initialize=lambda m,k: i*100+j*10+k)

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
        m.x = Var(m.K, initialize=lambda m,k: k)
        m.y = Var(m.I, m.J, initialize=lambda m,i,j: i*10+j)
        m.b = Block(m.I, m.J, rule=_b)
        m.bb = Block(m.I, m.J, m.K, rule=_bb)

    def tearDown(self):
        self.m = None

    def test_simple_getitem(self):
        self.assertIsInstance(self.m.b[1,4], _BlockData)

    def test_simple_getslice(self):
        _slicer = self.m.b[:,4]
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, ['b[1,4]', 'b[2,4]', 'b[3,4]'] )

        _slicer = self.m.b[1,4].c[:,4]
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, ['b[1,4].c[1,4]', 'b[1,4].c[2,4]', 'b[1,4].c[3,4]'] )

    def test_wildcard_slice(self):
        _slicer = self.m.b[:]
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        ans = [ str(x) for x in _slicer ]
        self.assertEqual( ans, [] )

        _slicer = self.m.b[...]
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, [ 'b[1,4]', 'b[1,5]', 'b[1,6]',
                   'b[2,4]', 'b[2,5]', 'b[2,6]',
                   'b[3,4]', 'b[3,5]', 'b[3,6]',
               ] )

        _slicer = self.m.b[1,...]
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, [ 'b[1,4]', 'b[1,5]', 'b[1,6]',
               ] )

        _slicer = self.m.b[...,5]
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, [ 'b[1,5]',
                   'b[2,5]',
                   'b[3,5]',
               ] )

        _slicer = self.m.bb[2,...,8]
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, [ 'bb[2,4,8]', 'bb[2,5,8]', 'bb[2,6,8]',
               ] )

        _slicer = self.m.bb[:,...,8]
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, [ 'bb[1,4,8]', 'bb[1,5,8]', 'bb[1,6,8]',
                   'bb[2,4,8]', 'bb[2,5,8]', 'bb[2,6,8]',
                   'bb[3,4,8]', 'bb[3,5,8]', 'bb[3,6,8]',
               ] )

        _slicer = self.m.bb[:,:,...,8]
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, [ 'bb[1,4,8]', 'bb[1,5,8]', 'bb[1,6,8]',
                   'bb[2,4,8]', 'bb[2,5,8]', 'bb[2,6,8]',
                   'bb[3,4,8]', 'bb[3,5,8]', 'bb[3,6,8]',
               ] )

        _slicer = self.m.bb[:,...,:,8]
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, [ 'bb[1,4,8]', 'bb[1,5,8]', 'bb[1,6,8]',
                   'bb[2,4,8]', 'bb[2,5,8]', 'bb[2,6,8]',
                   'bb[3,4,8]', 'bb[3,5,8]', 'bb[3,6,8]',
               ] )

        _slicer = self.m.b[1,4,...]
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, [ 'b[1,4]',
               ] )

        _slicer = self.m.b[1,2,3,...]
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        ans = [ str(x) for x in _slicer ]
        self.assertEqual( ans, [] )

        _slicer = self.m.b[1,:,2]
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        ans = [ str(x) for x in _slicer ]
        self.assertEqual( ans, [] )

        self.assertRaisesRegexp(
            IndexError, 'wildcard slice .* can only appear once',
            self.m.b.__getitem__, (Ellipsis,Ellipsis) )


    def test_nonterminal_slice(self):
        _slicer = self.m.b[:,4].x
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, ['b[1,4].x', 'b[2,4].x', 'b[3,4].x'] )

        _slicer = self.m.b[:,4].x[7]
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, ['b[1,4].x[7]', 'b[2,4].x[7]', 'b[3,4].x[7]'] )

    def test_nested_slices(self):
        _slicer = self.m.b[1,:].c[:,4].x
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, ['b[1,4].c[1,4].x', 'b[1,4].c[2,4].x', 'b[1,4].c[3,4].x',
                  'b[1,5].c[1,4].x', 'b[1,5].c[2,4].x', 'b[1,5].c[3,4].x',
                  'b[1,6].c[1,4].x', 'b[1,6].c[2,4].x', 'b[1,6].c[3,4].x',
              ] )

        _slicer = self.m.b[1,:].c[:,4].x[8]
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans,
            [ 'b[1,4].c[1,4].x[8]', 'b[1,4].c[2,4].x[8]', 'b[1,4].c[3,4].x[8]',
              'b[1,5].c[1,4].x[8]', 'b[1,5].c[2,4].x[8]', 'b[1,5].c[3,4].x[8]',
              'b[1,6].c[1,4].x[8]', 'b[1,6].c[2,4].x[8]', 'b[1,6].c[3,4].x[8]',
               ] )

    def test_component_function_slices(self):
        _slicer = self.m.component('b')[1,:].component('c')[:,4].component('x')
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        ans = [ str(x) for x in _slicer ]
        self.assertEqual(
            ans, ['b[1,4].c[1,4].x', 'b[1,4].c[2,4].x', 'b[1,4].c[3,4].x',
                  'b[1,5].c[1,4].x', 'b[1,5].c[2,4].x', 'b[1,5].c[3,4].x',
                  'b[1,6].c[1,4].x', 'b[1,6].c[2,4].x', 'b[1,6].c[3,4].x',
              ] )

    def test_noncomponent_function_slices(self):
        ans = self.m.component('b')[1,:].component('c')[:,4].x.fix(5)
        self.assertIsInstance(ans, list)
        self.assertEqual( ans, [None]*9 )

        ans = self.m.component('b')[1,:].component('c')[:,4].x[:].is_fixed()
        self.assertIsInstance(ans, list)
        self.assertEqual( ans, [True]*(9*3) )

        ans = self.m.component('b')[1,:].component('c')[:,5].x[:].is_fixed()
        self.assertIsInstance(ans, list)
        self.assertEqual( ans, [False]*(9*3) )

    def test_setattr_slices(self):
        init_sum = sum(self.m.b[:,:].c[:,:].x[:].value)
        init_vals = list(self.m.b[1,:].c[:,4].x[:].value)
        self.m.b[1,:].c[:,4].x[:].value = 0
        new_sum = sum(self.m.b[:,:].c[:,:].x[:].value)
        new_vals = list(self.m.b[1,:].c[:,4].x[:].value)
        # nothing got deleted
        self.assertEqual(len(init_vals), len(new_vals))
        # the lists values were changes
        self.assertNotEqual(init_vals, new_vals)
        # the set values are all now zero
        self.assertEqual(sum(new_vals), 0)
        # nothing outside the set values changed
        self.assertEqual(init_sum-sum(init_vals), new_sum)

    def test_setitem_slices(self):
        init_sum = sum(self.m.b[:,:].c[:,:].x[:].value)
        init_vals = list(self.m.b[1,:].c[:,4].x[:].value)
        self.m.b[1,:].c[:,4].x[:] = 0
        new_sum = sum(self.m.b[:,:].c[:,:].x[:].value)
        new_vals = list(self.m.b[1,:].c[:,4].x[:].value)
        # nothing got deleted
        self.assertEqual(len(init_vals), len(new_vals))
        # the lists values were changes
        self.assertNotEqual(init_vals, new_vals)
        # the set values are all now zero
        self.assertEqual(sum(new_vals), 0)
        # nothing outside the set values changed
        self.assertEqual(init_sum-sum(init_vals), new_sum)

    def test_setitem_component(self):
        init_sum = sum(self.m.x[:].value)
        init_vals = list(self.m.x[:].value)
        self.m.x[:] = 0
        new_sum = sum(self.m.x[:].value)
        new_vals = list(self.m.x[:].value)
        # nothing got deleted
        self.assertEqual(len(init_vals), len(new_vals))
        # the lists values were changes
        self.assertNotEqual(init_vals, new_vals)
        # the set values are all now zero
        self.assertEqual(sum(new_vals), 0)
        # nothing outside the set values changed
        self.assertEqual(init_sum-sum(init_vals), new_sum)

        init_sum = sum(self.m.y[:,:].value)
        init_vals = list(self.m.y[1,:].value)
        self.m.y[1,:] = 0
        new_sum = sum(self.m.y[:,:].value)
        new_vals = list(self.m.y[1,:].value)
        # nothing got deleted
        self.assertEqual(len(init_vals), len(new_vals))
        # the lists values were changes
        self.assertNotEqual(init_vals, new_vals)
        # the set values are all now zero
        self.assertEqual(sum(new_vals), 0)
        # nothing outside the set values changed
        self.assertEqual(init_sum-sum(init_vals), new_sum)

    def test_delitem_slices(self):
        init_all = list(self.m.b[:,:].c[:,:].x[:])
        init_tgt = list(self.m.b[1,:].c[:,4].x[:])
        del self.m.b[1,:].c[:,4].x[:]
        new_all = list(self.m.b[:,:].c[:,:].x[:])
        new_tgt = list(self.m.b[1,:].c[:,4].x[:])

        self.assertEqual(len(init_tgt), 3*3*3)
        self.assertEqual(len(init_all), (3*3)*(3*3)*3)
        self.assertEqual(len(new_tgt), 0)
        self.assertEqual(len(new_all), (3*3)*(3*3)*3 - 3*3*3)

    def test_delitem_component(self):
        init_all = list(self.m.bb[:,:,:])
        del self.m.bb[:,:,:]
        new_all = list(self.m.bb[:,:,:])
        self.assertEqual(len(init_all), 3*3*3)
        self.assertEqual(len(new_all), 0)

        init_all = list(self.m.b[:,:])
        init_tgt = list(self.m.b[1,:])
        del self.m.b[1,:]
        new_all = list(self.m.b[:,:])
        new_tgt = list(self.m.b[1,:])
        self.assertEqual(len(init_tgt), 3)
        self.assertEqual(len(init_all), 3*3)
        self.assertEqual(len(new_tgt), 0)
        self.assertEqual(len(new_all), 2*3)

    def test_empty_slices(self):
        _slicer = self.m.b[1,:].c[:,1].x
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        ans = [ str(x) for x in _slicer ]
        self.assertEqual( ans, [] )

        _slicer = self.m.b[1,:].c[:,4].x[1]
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        _slicer.key_errors_generate_exceptions = False
        ans = [ str(x) for x in _slicer ]
        self.assertEqual( ans, [] )

        _slicer = self.m.b[1,:].c[:,4].y
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        _slicer.attribute_errors_generate_exceptions = False
        ans = [ str(x) for x in _slicer ]
        self.assertEqual( ans, [] )

        _slicer = self.m.b[1,:].c[:,4].component('y', False)
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        _slicer.call_errors_generate_exceptions = False
        ans = [ str(x) for x in _slicer ]
        self.assertEqual( ans, [] )

        _slicer = self.m.b[1,:].c[:,4].x[1]
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        _slicer.key_errors_generate_exceptions = True
        self.assertRaises( KeyError, _slicer.next )

        _slicer = self.m.b[1,:].c[:,4].y
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        _slicer.attribute_errors_generate_exceptions = True
        self.assertRaises( AttributeError, _slicer.next )

        _slicer = self.m.b[1,:].c[:,4].component('y', False)
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        _slicer.call_errors_generate_exceptions = True
        self.assertRaises( TypeError,_slicer.next )

        _slicer = self.m.b[1,:].c[:,4].component()
        self.assertIsInstance(_slicer, _IndexedComponent_slice)
        _slicer.call_errors_generate_exceptions = True
        self.assertRaises( TypeError, _slicer.next )


class TestComponentReferences(unittest.TestCase):
    def setUp(self):
        self.m = m = ConcreteModel()
        @m.Block([1,2], [4,5])
        def b(b,i,j):
            b.x = Var([7,8],[10,11], initialize=0)

    def test_simple_lookup(self):
        m = self.m

        rd = _ReferenceDict(m.b[:,:].x[:,:])
        self.assertIs(rd[1,5,7,10], m.b[1,5].x[7,10])

        rd = _ReferenceDict(m.b[:,4].x[8,:])
        self.assertIs(rd[1,10], m.b[1,4].x[8,10])

    def test_len(self):
        m = self.m

        rd = _ReferenceDict(m.b[:,:].x[:,:])
        self.assertEqual(len(rd), 2*2*2*2)

        rd = _ReferenceDict(m.b[:,4].x[8,:])
        self.assertEqual(len(rd), 2*2)

    def test_simple_assignment(self):
        m = self.m

        rd = _ReferenceDict(m.b[:,:].x[:,:])
        self.assertEqual( sum(x.value for x in itervalues(rd)), 0 )
        rd[1,5,7,10] = 10
        self.assertEqual( m.b[1,5].x[7,10].value, 10 )
        self.assertEqual( sum(x.value for x in itervalues(rd)), 10 )

        rd = _ReferenceDict(m.b[:,4].x[8,:])
        self.assertEqual( sum(x.value for x in itervalues(rd)), 0 )
        rd[1,10] = 20
        self.assertEqual( m.b[1,4].x[8,10].value, 20 )
        self.assertEqual( sum(x.value for x in itervalues(rd)), 20 )

    def test_simple_attribute_assignment(self):
        m = self.m

        rd = _ReferenceDict(m.b[:,:].x[:,:])
        self.assertEqual( sum(x.value for x in itervalues(rd)), 0 )
        rd[1,5,7,10].value = 10
        self.assertEqual( m.b[1,5].x[7,10].value, 10 )
        self.assertEqual( sum(x.value for x in itervalues(rd)), 10 )

        rd = _ReferenceDict(m.b[:,4].x[8,:])
        self.assertEqual( sum(x.value for x in itervalues(rd)), 0 )
        rd[1,10].value = 20
        self.assertEqual( m.b[1,4].x[8,10].value, 20 )
        self.assertEqual( sum(x.value for x in itervalues(rd)), 20 )

    def test_single_deletion(self):
        m = self.m

        rd = _ReferenceDict(m.b[1,5].x[:,:])
        self.assertEqual(len(list(x.value for x in itervalues(rd))), 2*2)
        self.assertTrue((7,10) in rd)
        del rd[7,10]
        self.assertFalse((7,10) in rd)
        self.assertEqual(len(list(x.value for x in itervalues(rd))), 2*2*2*2-1)

        rd = _ReferenceDict(m.b[1,4].x[8,:])
        self.assertEqual(len(list(x.value for x in itervalues(rd))), 2)
        self.assertTrue((10) in rd)
        del rd[10]
        self.assertFalse(10 in rd)
        self.assertEqual(len(list(x.value for x in itervalues(rd))), 2-1)

    def test_nested_deletion(self):
        m = self.m

        rd = _ReferenceDict(m.b[:,:].x[:,:])
        self.assertEqual(len(list(x.value for x in itervalues(rd))), 2*2*2*2)
        self.assertTrue((1,5,7,10) in rd)
        del rd[1,5,7,10]
        self.assertFalse((1,5,7,10) in rd)
        self.assertEqual(len(list(x.value for x in itervalues(rd))), 2*2*2*2-1)

        rd = _ReferenceDict(m.b[:,4].x[8,:])
        self.assertEqual(len(list(x.value for x in itervalues(rd))), 2*2)
        self.assertTrue((1,10) in rd)
        del rd[1,10]
        self.assertFalse((1,10) in rd)
        self.assertEqual(len(list(x.value for x in itervalues(rd))), 2*2-1)


if __name__ == "__main__":
    unittest.main()
