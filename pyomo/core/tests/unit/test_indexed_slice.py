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
import pickle

import pyutilib.th as unittest

from pyomo.environ import *
from pyomo.core.base.block import _BlockData
from pyomo.core.base.indexed_component import _IndexedComponent_slice

def _x_init(m, k):
    return k

def _y_init(m, i, j):
    return i*10+j

def _cx_init(b, k):
    i, j = b.index()[:2]
    return i*100+j*10+k

def _c(b, i, j):
    b.x = Var(b.model().K, initialize=_cx_init)

def _b(b, i, j):
    _c(b,i,j)
    b.c = Block(b.model().I, b.model().J, rule=_c)

def _bb(b, i, j, k):
    _c(b,i,j)
    b.c = Block(b.model().I, b.model().J, rule=_c)

class TestComponentSlices(unittest.TestCase):
    def setUp(self):
        self.m = m = ConcreteModel()
        m.I = RangeSet(1,3)
        m.J = RangeSet(4,6)
        m.K = RangeSet(7,9)
        m.x = Var(m.K, initialize=_x_init)
        m.y = Var(m.I, m.J, initialize=_y_init)
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

        # Test error on invalid attribute
        _slice = self.m.b[...].c[...].x[:]
        with self.assertRaisesRegexp(
                AttributeError, ".*VarData' object has no attribute 'bogus'"):
            _slice.duplicate().bogus = 0
        # but disabling the exception flag will run without error
        _slice.attribute_errors_generate_exceptions = False
        # This doesn't do anything ... simply not raising an exception
        # is sufficient to verify the desired behavior
        _slice.bogus = 0

    def test_delattr_slices(self):
        self.m.b[1,:].c[:,4].x.foo = 10
        # check that the attribute was added
        self.assertEqual(len(list(self.m.b[1,:].c[:,4].x)), 3*3)
        self.assertEqual(sum(list(self.m.b[1,:].c[:,4].x.foo)), 10*3*3)
        self.assertEqual(sum(list(1 if hasattr(x,'foo') else 0
                                  for x in self.m.b[:,:].c[:,:].x)), 3*3)

        _slice = self.m.b[1,:].c[:,4].x.foo
        _slice._call_stack[-1] = (
            _IndexedComponent_slice.del_attribute,
            _slice._call_stack[-1][1] )
        # call the iterator to delete the attributes
        list(_slice.duplicate())
        self.assertEqual(sum(list(1 if hasattr(x,'foo') else 0
                                  for x in self.m.b[:,:].c[:,:].x)), 0)
        # calling the iterator again will raise an exception
        with self.assertRaisesRegexp(AttributeError, 'foo'):
            list(_slice.duplicate())
        # but disabling the exception flag will run without error
        _slice.attribute_errors_generate_exceptions = False
        # This doesn't do anything ... simply not raising an exception
        # is sufficient to verify the desired behavior
        list(_slice)

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

        _slice = self.m.b[1,:].c[:,4].x
        with self.assertRaisesRegexp(
                KeyError, "Index 'bogus' is not valid for indexed "
                "component 'b\[1,4\]\.c\[1,4\]\.x'"):
            _slice.duplicate()['bogus'] = 0
        # but disabling the exception flag will run without error
        _slice.key_errors_generate_exceptions = False
        # This doesn't do anything ... simply not raising an exception
        # is sufficient to verify the desired behavior
        _slice['bogus'] = 0


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

        _slice = self.m.b[2,:].c[:,4].x
        with self.assertRaisesRegexp(
                KeyError, "Index 'bogus' is not valid for indexed "
                "component 'b\[2,4\]\.c\[1,4\]\.x'"):
            del _slice.duplicate()['bogus']
        # but disabling the exception flag will run without error
        _slice.key_errors_generate_exceptions = False
        # This doesn't do anything ... simply not raising an exception
        # is sufficient to verify the desired behavior
        del _slice['bogus']
        # Nothing additional should have been deleted
        final_all = list(self.m.b[:,:].c[:,:].x[:])
        self.assertEqual(len(new_all), len(final_all))

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

    def test_iterators(self):
        m = self.m

        _slice = self.m.x[...]
        self.assertEqual(
            list(_slice.wildcard_keys()),
            [7,8,9]
        )
        self.assertEqual(
            list(_slice.wildcard_items()),
            [(7, m.x[7]), (8, m.x[8]), (9, m.x[9])]
        )
        self.assertEqual(
            list(_slice.expanded_keys()),
            [7,8,9]
        )
        self.assertEqual(
            list(_slice.expanded_items()),
            [(7, m.x[7]), (8, m.x[8]), (9, m.x[9])]
        )

        _slice = self.m.b[...]
        self.assertEqual(
            list(_slice.wildcard_keys()),
            [(1,4), (1,5), (1,6), (2,4), (2,5), (2,6), (3,4), (3,5), (3,6)]
        )
        self.assertEqual(
            list(_slice.wildcard_items()),
            [((1,4), m.b[1,4]), ((1,5), m.b[1,5]), ((1,6), m.b[1,6]),
             ((2,4), m.b[2,4]), ((2,5), m.b[2,5]), ((2,6), m.b[2,6]),
             ((3,4), m.b[3,4]), ((3,5), m.b[3,5]), ((3,6), m.b[3,6]),]
        )
        self.assertEqual(
            list(_slice.expanded_keys()),
            [(1,4), (1,5), (1,6), (2,4), (2,5), (2,6), (3,4), (3,5), (3,6)]
        )
        self.assertEqual(
            list(_slice.expanded_items()),
            [((1,4), m.b[1,4]), ((1,5), m.b[1,5]), ((1,6), m.b[1,6]),
             ((2,4), m.b[2,4]), ((2,5), m.b[2,5]), ((2,6), m.b[2,6]),
             ((3,4), m.b[3,4]), ((3,5), m.b[3,5]), ((3,6), m.b[3,6]),]
        )

        _slice = self.m.b[1,:]
        self.assertEqual(
            list(_slice.wildcard_keys()),
            [4, 5, 6]
        )
        self.assertEqual(
            list(_slice.wildcard_items()),
            [(4, m.b[1,4]), (5, m.b[1,5]), (6, m.b[1,6]),]
        )
        self.assertEqual(
            list(_slice.expanded_keys()),
            [(1,4), (1,5), (1,6)]
        )
        self.assertEqual(
            list(_slice.expanded_items()),
            [((1,4), m.b[1,4]), ((1,5), m.b[1,5]), ((1,6), m.b[1,6])]
        )

    def test_pickle_slices(self):
        m = self.m
        _slicer = m.b[1,:].c[:,4].x
        _new_slicer = pickle.loads(pickle.dumps(_slicer))

        self.assertIsNot(_slicer, _new_slicer)
        self.assertIsNot(_slicer._call_stack, _new_slicer._call_stack)
        self.assertIs(type(_slicer._call_stack), type(_new_slicer._call_stack))
        self.assertEqual(len(_slicer._call_stack), len(_new_slicer._call_stack))

        ref = ['b[1,4].c[1,4].x', 'b[1,4].c[2,4].x', 'b[1,4].c[3,4].x',
               'b[1,5].c[1,4].x', 'b[1,5].c[2,4].x', 'b[1,5].c[3,4].x',
               'b[1,6].c[1,4].x', 'b[1,6].c[2,4].x', 'b[1,6].c[3,4].x',
               ]
        self.assertEqual([str(x) for x in _slicer], ref )
        self.assertEqual([str(x) for x in _new_slicer], ref )
        for x,y in zip(iter(_slicer), iter(_new_slicer)):
            self.assertIs(type(x), type(y))
            self.assertEqual(x.name, y.name)
            self.assertIsNot(x, y)

    def test_clone_on_model(self):
        m = self.m
        m.slicer = m.b[1,:].c[:,4].x
        n = m.clone()

        self.assertIsNot(m, n)
        self.assertIsNot(m.slicer, n.slicer)
        self.assertIsNot(m.slicer._call_stack, n.slicer._call_stack)
        self.assertIs(type(m.slicer._call_stack), type(n.slicer._call_stack))
        self.assertEqual(len(m.slicer._call_stack), len(n.slicer._call_stack))

        ref = ['b[1,4].c[1,4].x', 'b[1,4].c[2,4].x', 'b[1,4].c[3,4].x',
               'b[1,5].c[1,4].x', 'b[1,5].c[2,4].x', 'b[1,5].c[3,4].x',
               'b[1,6].c[1,4].x', 'b[1,6].c[2,4].x', 'b[1,6].c[3,4].x',
               ]
        self.assertEqual([str(x) for x in m.slicer], ref )
        self.assertEqual([str(x) for x in n.slicer], ref )
        for x,y in zip(iter(m.slicer), iter(n.slicer)):
            self.assertIs(type(x), type(y))
            self.assertEqual(x.name, y.name)
            self.assertIsNot(x, y)
            self.assertIs(x.model(), m)
            self.assertIs(y.model(), n)

if __name__ == "__main__":
    unittest.main()
