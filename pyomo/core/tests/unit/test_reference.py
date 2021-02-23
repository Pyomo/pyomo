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
from six import itervalues, StringIO, iterkeys, iteritems

from pyomo.environ import (
    ConcreteModel, Block, Var, Set, RangeSet, Param, value,
)
from pyomo.common.collections import OrderedDict, ComponentSet
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.set import SetProduct, UnorderedSetOf
from pyomo.core.base.indexed_component import (
    UnindexedComponent_set, IndexedComponent
)
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.reference import (
    _ReferenceDict, _ReferenceSet, Reference
)


class TestReferenceDict(unittest.TestCase):
    def setUp(self):
        self.m = m = ConcreteModel()
        @m.Block([1,2], [4,5])
        def b(b,i,j):
            b.x = Var([7,8],[10,11], initialize=0)
            b.y = Var([7,8], initialize=0)
            b.z = Var()

        @m.Block([1,2])
        def c(b,i):
            b.x = Var([7,8],[10,11], initialize=0)
            b.y = Var([7,8], initialize=0)
            b.z = Var()

    def _lookupTester(self, _slice, key, ans):
        rd = _ReferenceDict(_slice)
        self.assertIn(key, rd)
        self.assertIs(rd[key], ans)

        if len(key) == 1:
            self.assertIn(key[0], rd)
            self.assertIs(rd[key[0]], ans)

        self.assertNotIn(None, rd)
        with self.assertRaises(KeyError):
            rd[None]

        for i in range(len(key)):
            _ = tuple([0]*i)
            self.assertNotIn(_, rd)
            with self.assertRaises(KeyError):
                rd[_]

    def test_simple_lookup(self):
        m = self.m

        self._lookupTester(m.b[:,:].x[:,:],  (1,5,7,10), m.b[1,5].x[7,10])
        self._lookupTester(m.b[:,4].x[8,:],  (1,10),     m.b[1,4].x[8,10])
        self._lookupTester(m.b[:,4].x[8,10], (1,),       m.b[1,4].x[8,10])
        self._lookupTester(m.b[1,4].x[8,:],  (10,),      m.b[1,4].x[8,10])

        self._lookupTester(m.b[:,:].y[:], (1,5,7), m.b[1,5].y[7])
        self._lookupTester(m.b[:,4].y[:], (1,7),   m.b[1,4].y[7])
        self._lookupTester(m.b[:,4].y[8], (1,),    m.b[1,4].y[8])

        self._lookupTester(m.b[:,:].z, (1,5), m.b[1,5].z)
        self._lookupTester(m.b[:,4].z, (1,),  m.b[1,4].z)


        self._lookupTester(m.c[:].x[:,:],  (1,7,10), m.c[1].x[7,10])
        self._lookupTester(m.c[:].x[8,:],  (1,10),   m.c[1].x[8,10])
        self._lookupTester(m.c[:].x[8,10], (1,),     m.c[1].x[8,10])
        self._lookupTester(m.c[1].x[:,:],  (8,10),   m.c[1].x[8,10])
        self._lookupTester(m.c[1].x[8,:],  (10,),    m.c[1].x[8,10])

        self._lookupTester(m.c[:].y[:], (1,7), m.c[1].y[7])
        self._lookupTester(m.c[:].y[8], (1,),  m.c[1].y[8])
        self._lookupTester(m.c[1].y[:], (8,),  m.c[1].y[8])

        self._lookupTester(m.c[:].z, (1,), m.c[1].z)

        m.jagged_set = Set(initialize=[1,(2,3)], dimen=None)
        m.jb = Block(m.jagged_set)
        m.jb[1].x = Var([1,2,3])
        m.jb[2,3].x = Var([1,2,3])
        self._lookupTester(m.jb[...], (1,), m.jb[1])
        self._lookupTester(m.jb[...].x[:], (1,2), m.jb[1].x[2])
        self._lookupTester(m.jb[...].x[:], (2,3,2), m.jb[2,3].x[2])

        rd = _ReferenceDict(m.jb[:,:,:].x[:])
        with self.assertRaises(KeyError):
            rd[2,3,4,2]
        rd = _ReferenceDict(m.b[:,4].x[:])
        with self.assertRaises(KeyError):
            rd[1,0]

    def test_len(self):
        m = self.m

        rd = _ReferenceDict(m.b[:,:].x[:,:])
        self.assertEqual(len(rd), 2*2*2*2)

        rd = _ReferenceDict(m.b[:,4].x[8,:])
        self.assertEqual(len(rd), 2*2)

    def test_iterators(self):
        m = self.m
        rd = _ReferenceDict(m.b[:,4].x[8,:])

        self.assertEqual(
            list(iterkeys(rd)),
            [(1,10), (1,11), (2,10), (2,11)]
        )
        self.assertEqual(
            list(itervalues(rd)),
            [m.b[1,4].x[8,10], m.b[1,4].x[8,11],
             m.b[2,4].x[8,10], m.b[2,4].x[8,11]]
        )
        self.assertEqual(
            list(iteritems(rd)),
            [((1,10), m.b[1,4].x[8,10]),
             ((1,11), m.b[1,4].x[8,11]),
             ((2,10), m.b[2,4].x[8,10]),
             ((2,11), m.b[2,4].x[8,11])]
        )

    def test_nested_assignment(self):
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

    def test_attribute_assignment(self):
        m = self.m

        rd = _ReferenceDict(m.b[:,:].x[:,:].value)
        self.assertEqual( sum(x for x in itervalues(rd)), 0 )
        rd[1,5,7,10] = 10
        self.assertEqual( m.b[1,5].x[7,10].value, 10 )
        self.assertEqual( sum(x for x in itervalues(rd)), 10 )

        rd = _ReferenceDict(m.b[:,4].x[8,:].value)
        self.assertEqual( sum(x for x in itervalues(rd)), 0 )
        rd[1,10] = 20
        self.assertEqual( m.b[1,4].x[8,10].value, 20 )
        self.assertEqual( sum(x for x in itervalues(rd)), 20 )

        m.x = Var([1,2], initialize=0)
        rd = _ReferenceDict(m.x[:])
        self.assertEqual( sum(x.value for x in itervalues(rd)), 0 )
        rd[2] = 10
        self.assertEqual( m.x[1].value, 0 )
        self.assertEqual( m.x[2].value, 10 )
        self.assertEqual( sum(x.value for x in itervalues(rd)), 10 )

    def test_single_attribute_assignment(self):
        m = self.m

        rd = _ReferenceDict(m.b[1,5].x[:,:])
        self.assertEqual( sum(x.value for x in itervalues(rd)), 0 )
        rd[7,10].value = 10
        self.assertEqual( m.b[1,5].x[7,10].value, 10 )
        self.assertEqual( sum(x.value for x in itervalues(rd)), 10 )

        rd = _ReferenceDict(m.b[1,4].x[8,:])
        self.assertEqual( sum(x.value for x in itervalues(rd)), 0 )
        rd[10].value = 20
        self.assertEqual( m.b[1,4].x[8,10].value, 20 )
        self.assertEqual( sum(x.value for x in itervalues(rd)), 20 )

    def test_nested_attribute_assignment(self):
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
        self.assertEqual(len(list(x.value for x in itervalues(rd))), 3)

        rd = _ReferenceDict(m.b[1,4].x[8,:])
        self.assertEqual(len(list(x.value for x in itervalues(rd))), 2)
        self.assertTrue((10) in rd)
        del rd[10]
        self.assertFalse(10 in rd)
        self.assertEqual(len(list(x.value for x in itervalues(rd))), 2-1)

        with self.assertRaisesRegexp(
                KeyError,
                r"\(8, 10\) is not valid for indexed component 'b\[1,4\].x'"):
            del rd[10]

        rd = _ReferenceDict(m.b[1,:].x[8,0])
        with self.assertRaisesRegexp(
                KeyError,
                r"'\(8, 0\)' is not valid for indexed component 'b\[1,4\].x'"):
            del rd[4]


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

    def test_attribute_deletion(self):
        m = self.m

        rd = _ReferenceDict(m.b[:,:].z)
        rd._slice.attribute_errors_generate_exceptions = False
        self.assertEqual(len(list(x.value for x in itervalues(rd))), 2*2)
        self.assertTrue((1,5) in rd)
        self.assertTrue( hasattr(m.b[1,5], 'z') )
        self.assertTrue( hasattr(m.b[2,5], 'z') )
        del rd[1,5]
        self.assertFalse((1,5) in rd)
        self.assertFalse( hasattr(m.b[1,5], 'z') )
        self.assertTrue( hasattr(m.b[2,5], 'z') )
        self.assertEqual(len(list(x.value for x in itervalues(rd))), 3)

        rd = _ReferenceDict(m.b[2,:].z)
        rd._slice.attribute_errors_generate_exceptions = False
        self.assertEqual(len(list(x.value for x in itervalues(rd))), 2)
        self.assertTrue(5 in rd)
        self.assertTrue( hasattr(m.b[2,4], 'z') )
        self.assertTrue( hasattr(m.b[2,5], 'z') )
        del rd[5]
        self.assertFalse(5 in rd)
        self.assertTrue( hasattr(m.b[2,4], 'z') )
        self.assertFalse( hasattr(m.b[2,5], 'z') )
        self.assertEqual(len(list(x.value for x in itervalues(rd))), 2-1)

class TestReferenceSet(unittest.TestCase):
    def test_lookup_and_iter_dense_data(self):
        m = ConcreteModel()
        @m.Block([1,2], [4,5])
        def b(b,i,j):
            b.x = Var([7,8],[10,11], initialize=0)
            b.y = Var([7,8], initialize=0)
            b.z = Var()

        rs = _ReferenceSet(m.b[:,5].z)
        self.assertNotIn((0,), rs)
        self.assertIn(1, rs)
        self.assertIn((1,), rs)
        self.assertEqual(len(rs), 2)
        self.assertEqual(list(rs), [1,2])

        rs = _ReferenceSet(m.b[:,5].bad)
        self.assertNotIn((0,), rs)
        self.assertNotIn((1,), rs)
        self.assertEqual(len(rs), 0)
        self.assertEqual(list(rs), [])

        @m.Block([1,2,3])
        def d(b, i):
            if i % 2:
                b.x = Var(range(i))

        rs = _ReferenceSet(m.d[:].x[:])
        self.assertIn((1,0), rs)
        self.assertIn((3,0), rs)
        self.assertNotIn((2,0), rs)
        self.assertEqual(len(rs), 4)
        self.assertEqual(list(rs), [(1,0), (3,0), (3,1), (3,2)])

        rs = _ReferenceSet(m.d[...].x[...])
        self.assertIn((1,0), rs)
        self.assertIn((3,0), rs)
        self.assertNotIn((2,0), rs)
        self.assertEqual(len(rs), 4)
        self.assertEqual(list(rs), [(1,0), (3,0), (3,1), (3,2)])

        # Test the SliceEllipsisError case (lookup into a jagged set
        # with an ellipsis)

        m.e_index = Set(initialize=[2,(2,3)], dimen=None)
        @m.Block(m.e_index)
        def e(b, *args):
            b.x_index = Set(initialize=[1,(3,4)], dimen=None)
            b.x = Var(b.x_index)
        rs = _ReferenceSet(m.e[...].x[...])
        self.assertIn((2,1), rs)
        self.assertIn((2,3,1), rs)
        self.assertIn((2,3,4), rs)
        self.assertNotIn((2,3,5), rs)
        self.assertEqual(len(rs), 4)
        self.assertEqual(list(rs), [(2,1), (2,3,4), (2,3,1), (2,3,3,4)])

        # Make sure scalars and tuples work for jagged sets
        rs = _ReferenceSet(m.e[...])
        self.assertIn(2, rs)
        self.assertIn((2,), rs)
        self.assertNotIn(0, rs)
        self.assertNotIn((0,), rs)

    def test_lookup_and_iter_sparse_data(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.x = Var(m.I, m.I, dense=False)

        rd = _ReferenceDict(m.x[...])
        rs = _ReferenceSet(m.x[...])
        self.assertEqual(len(rd), 0)
        # Note: we will periodically re-check the dict to ensure
        # iteration doesn't accidentally declare data
        self.assertEqual(len(rd), 0)

        self.assertEqual(len(rs), 9)
        self.assertEqual(len(rd), 0)

        self.assertIn((1,1), rs)
        self.assertEqual(len(rd), 0)
        self.assertEqual(len(rs), 9)



class TestReference(unittest.TestCase):
    def test_constructor_error(self):
        m = ConcreteModel()
        m.x = Var([1,2])
        class Foo(object): pass
        self.assertRaisesRegexp(
            TypeError,
            "First argument to Reference constructors must be a "
            "component, component slice, Sequence, or Mapping \(received Foo",
            Reference, Foo()
            )
        self.assertRaisesRegexp(
            TypeError,
            "First argument to Reference constructors must be a "
            "component, component slice, Sequence, or Mapping \(received int",
            Reference, 5
            )
        self.assertRaisesRegexp(
            TypeError,
            "First argument to Reference constructors must be a "
            "component, component slice, Sequence, or Mapping \(received None",
            Reference, None
            )

    def test_component_reference(self):
        m = ConcreteModel()
        m.x = Var()
        m.r = Reference(m.x)

        self.assertIs(m.r.ctype, Var)
        self.assertIsNot(m.r.index_set(), m.x.index_set())
        self.assertIs(m.x.index_set(), UnindexedComponent_set)
        self.assertIs(type(m.r.index_set()), UnorderedSetOf)
        self.assertEqual(len(m.r), 1)
        self.assertTrue(m.r.is_indexed())
        self.assertIn(None, m.r)
        self.assertNotIn(1, m.r)
        self.assertIs(m.r[None], m.x)
        with self.assertRaises(KeyError):
            m.r[1]

        m.s = Reference(m.x[:])

        self.assertIs(m.s.ctype, Var)
        self.assertIsNot(m.s.index_set(), m.x.index_set())
        self.assertIs(m.x.index_set(), UnindexedComponent_set)
        self.assertIs(type(m.s.index_set()), UnorderedSetOf)
        self.assertEqual(len(m.s), 1)
        self.assertTrue(m.s.is_indexed())
        self.assertIn(None, m.s)
        self.assertNotIn(1, m.s)
        self.assertIs(m.s[None], m.x)
        with self.assertRaises(KeyError):
            m.s[1]

        m.y = Var([1,2])
        m.t = Reference(m.y)

        self.assertIs(m.t.ctype, Var)
        self.assertIs(m.t.index_set(), m.y.index_set())
        self.assertEqual(len(m.t), 2)
        self.assertTrue(m.t.is_indexed())
        self.assertNotIn(None, m.t)
        self.assertIn(1, m.t)
        self.assertIn(2, m.t)
        self.assertIs(m.t[1], m.y[1])
        with self.assertRaises(KeyError):
            m.t[3]

    def test_reference_indexedcomponent_pprint(self):
        m = ConcreteModel()
        m.x = Var([1,2], initialize={1:4,2:8})
        m.r = Reference(m.x, ctype=IndexedComponent)
        buf = StringIO()
        m.r.pprint(ostream=buf)
        self.assertEqual(buf.getvalue(),
"""r : Size=2, Index=x_index
    Key : Object
      1 : <class 'pyomo.core.base.var._GeneralVarData'>
      2 : <class 'pyomo.core.base.var._GeneralVarData'>
""")

    def test_single_reference(self):
        m = ConcreteModel()
        m.b = Block([1,2])
        m.b[1].x = Var(bounds=(1,None))
        m.b[2].x = Var(bounds=(2,None))
        m.r = Reference(m.b[:].x)

        self.assertIs(m.r.ctype, Var)
        self.assertIs(m.r.index_set(), m.b.index_set())
        self.assertEqual(len(m.r), 2)
        self.assertEqual(m.r[1].lb, 1)
        self.assertEqual(m.r[2].lb, 2)
        self.assertIn(1, m.r)
        self.assertIn(2, m.r)
        self.assertNotIn(3, m.r)
        with self.assertRaises(KeyError):
            m.r[3]

    def test_nested_reference(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1,2])
        m.J = Set(initialize=[3,4])
        @m.Block(m.I)
        def b(b,i):
            b.x = Var(b.model().J, bounds=(i,None))

        m.r = Reference(m.b[:].x[:])

        self.assertIs(m.r.ctype, Var)
        self.assertIsInstance(m.r.index_set(), SetProduct)
        self.assertIs(m.r.index_set().set_tuple[0], m.I)
        self.assertIs(m.r.index_set().set_tuple[1], m.J)
        self.assertEqual(len(m.r), 2*2)
        self.assertEqual(m.r[1,3].lb, 1)
        self.assertEqual(m.r[2,4].lb, 2)
        self.assertIn((1,3), m.r)
        self.assertIn((2,4), m.r)
        self.assertNotIn(0, m.r)
        self.assertNotIn((1,0), m.r)
        self.assertNotIn((1,3,0), m.r)
        with self.assertRaises(KeyError):
            m.r[0]

    def test_nested_reference_multidim_set(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1,2])
        m.J = Set(initialize=[(3,3),(4,4)])
        @m.Block(m.I)
        def b(b,i):
            b.x = Var(b.model().J, bounds=(i,None))

        m.r = Reference(m.b[:].x[:,:])

        self.assertIs(m.r.ctype, Var)
        self.assertIsInstance(m.r.index_set(), SetProduct)
        self.assertIs(m.r.index_set().set_tuple[0], m.I)
        self.assertIs(m.r.index_set().set_tuple[1], m.J)
        self.assertEqual(len(m.r), 2*2)
        self.assertEqual(m.r[1,3,3].lb, 1)
        self.assertEqual(m.r[2,4,4].lb, 2)
        self.assertIn((1,3,3), m.r)
        self.assertIn((2,4,4), m.r)
        self.assertNotIn(0, m.r)
        self.assertNotIn((1,0), m.r)
        self.assertNotIn((1,3,0), m.r)
        self.assertNotIn((1,3,3,0), m.r)
        with self.assertRaises(KeyError):
            m.r[0]

    def test_nested_reference_partial_multidim_set(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1,2])
        m.J = Set(initialize=[(3,3),(4,4)])
        @m.Block(m.I)
        def b(b,i):
            b.x = Var(b.model().J, bounds=(i,None))

        m.r = Reference(m.b[:].x[3,:])

        self.assertIs(m.r.ctype, Var)
        self.assertIs(type(m.r.index_set()), UnorderedSetOf)
        self.assertEqual(len(m.r), 2*1)
        self.assertEqual(m.r[1,3].lb, 1)
        self.assertEqual(m.r[2,3].lb, 2)
        self.assertIn((1,3), m.r)
        self.assertNotIn((2,4), m.r)
        self.assertNotIn(0, m.r)
        self.assertNotIn((1,0), m.r)
        self.assertNotIn((1,3,0), m.r)
        with self.assertRaises(KeyError):
            m.r[0]

    def test_nested_reference_nonuniform_indexes(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1,2])
        m.J = Set(initialize=[3,4])
        @m.Block(m.I)
        def b(b,i):
            b.x = Var([3,4], bounds=(i,None))

        m.r = Reference(m.b[:].x[:])

        self.assertIs(m.r.ctype, Var)
        self.assertIs(type(m.r.index_set()), UnorderedSetOf)
        self.assertEqual(len(m.r), 2*2)
        self.assertEqual(m.r[1,3].lb, 1)
        self.assertEqual(m.r[2,4].lb, 2)
        self.assertIn((1,3), m.r)
        self.assertIn((2,4), m.r)
        self.assertNotIn(0, m.r)
        self.assertNotIn((1,0), m.r)
        self.assertNotIn((1,3,0), m.r)
        with self.assertRaises(KeyError):
            m.r[0]

    def test_nested_reference_nondimen_set(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1,2])
        m.J = Set(initialize=[3,4], dimen=None)
        @m.Block(m.I)
        def b(b,i):
            b.x = Var(b.model().J, bounds=(i,None))

        m.r = Reference(m.b[:].x[:])

        self.assertIs(m.r.ctype, Var)
        self.assertIs(type(m.r.index_set()), UnorderedSetOf)
        self.assertEqual(len(m.r), 2*2)
        self.assertEqual(m.r[1,3].lb, 1)
        self.assertEqual(m.r[2,4].lb, 2)
        self.assertIn((1,3), m.r)
        self.assertIn((2,4), m.r)
        self.assertNotIn(0, m.r)
        self.assertNotIn((1,0), m.r)
        self.assertNotIn((1,3,0), m.r)
        with self.assertRaises(KeyError):
            m.r[0]

    def test_nested_reference_nonuniform_index_size(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1,2])
        m.J = Set(initialize=[3,4])
        m.b = Block(m.I)
        m.b[1].x = Var([(3,3),(3,4),(4,3),(4,4)], bounds=(1,None))
        m.b[2].x = Var(m.J, m.J, bounds=(2,None))

        m.r = Reference(m.b[:].x[:,:])

        self.assertIs(m.r.ctype, Var)
        self.assertIs(type(m.r.index_set()), UnorderedSetOf)
        self.assertEqual(len(m.r), 2*2*2)
        self.assertEqual(m.r[1,3,3].lb, 1)
        self.assertEqual(m.r[2,4,3].lb, 2)
        self.assertIn((1,3,3), m.r)
        self.assertIn((2,4,4), m.r)
        self.assertNotIn(0, m.r)
        self.assertNotIn((1,0), m.r)
        self.assertNotIn((1,3,0), m.r)
        self.assertNotIn((1,3,3,0), m.r)
        with self.assertRaises(KeyError):
            m.r[0]

    def test_nested_scalars(self):
        m = ConcreteModel()
        m.b = Block()
        m.b.x = Var()
        m.r = Reference(m.b[:].x[:])
        self.assertEqual(len(m.r), 1)
        self.assertEqual(m.r.index_set().dimen, 2)
        base_sets = list(m.r.index_set().subsets())
        self.assertEqual(len(base_sets), 2)
        self.assertIs(type(base_sets[0]), UnorderedSetOf)
        self.assertIs(type(base_sets[1]), UnorderedSetOf)

    def test_ctype_detection(self):
        m = ConcreteModel()
        m.js = Set(initialize=[1, (2,3)], dimen=None)
        m.b = Block([1,2])
        m.b[1].x = Var(m.js)
        m.b[1].y = Var()
        m.b[1].z = Var([1,2])
        m.b[2].x = Param(initialize=0)
        m.b[2].y = Var()
        m.b[2].z = Var([1,2])

        m.x = Reference(m.b[:].x[...])
        self.assertIs(type(m.x), IndexedComponent)

        m.y = Reference(m.b[:].y[...])
        self.assertIs(type(m.y), IndexedVar)
        self.assertIs(m.y.ctype, Var)
        m.y1 = Reference(m.b[:].y[...], ctype=None)
        self.assertIs(type(m.y1), IndexedComponent)
        self.assertIs(m.y1.ctype, IndexedComponent)

        m.z = Reference(m.b[:].z)
        self.assertIs(type(m.z), IndexedComponent)
        self.assertIs(m.z.ctype, IndexedComponent)

    def test_reference_to_sparse(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.x = Var(m.I, m.I, dense=False)
        m.xx = Reference(m.x[...], ctype=Var)

        self.assertEqual(len(m.x), 0)
        self.assertNotIn((1,1), m.x)
        self.assertNotIn((1,1), m.xx)
        self.assertIn((1,1), m.x.index_set())
        self.assertIn((1,1), m.xx.index_set())
        self.assertEqual(len(m.x), 0)

        m.xx[1,2]
        self.assertEqual(len(m.x), 1)
        self.assertIs(m.xx[1,2], m.x[1,2])
        self.assertEqual(len(m.x), 1)

        m.xx[1,3] = 5
        self.assertEqual(len(m.x), 2)
        self.assertIs(m.xx[1,3], m.x[1,3])
        self.assertEqual(len(m.x), 2)
        self.assertEqual(value(m.x[1,3]), 5)

        m.xx.add((1,1))
        self.assertEqual(len(m.x), 3)
        self.assertIs(m.xx[1,1], m.x[1,1])
        self.assertEqual(len(m.x), 3)

    def test_nested_reference_to_sparse(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1])
        @m.Block(m.I)
        def b(b, i):
            b.x = Var(b.model().I, dense=False)
        m.xx = Reference(m.b[:].x[:], ctype=Var)
        m.I.add(2)
        m.I.add(3)

        self.assertEqual(len(m.b), 1)
        self.assertEqual(len(m.b[1].x), 0)
        self.assertIn(1, m.b)
        self.assertNotIn((1,1), m.xx)
        self.assertIn(1, m.b[1].x.index_set())
        self.assertIn((1,1), m.xx.index_set())
        self.assertEqual(len(m.b), 1)
        self.assertEqual(len(m.b[1].x), 0)

        m.xx[1,2]
        self.assertEqual(len(m.b), 1)
        self.assertEqual(len(m.b[1].x), 1)
        self.assertIs(m.xx[1,2], m.b[1].x[2])
        self.assertEqual(len(m.b), 1)
        self.assertEqual(len(m.b[1].x), 1)

        m.xx[1,3] = 5
        self.assertEqual(len(m.b), 1)
        self.assertEqual(len(m.b[1].x), 2)
        self.assertIs(m.xx[1,3], m.b[1].x[3])
        self.assertEqual(value(m.b[1].x[3]), 5)
        self.assertEqual(len(m.b), 1)
        self.assertEqual(len(m.b[1].x), 2)

        m.xx.add((1,1))
        self.assertEqual(len(m.b), 1)
        self.assertEqual(len(m.b[1].x), 3)
        self.assertIs(m.xx[1,1], m.b[1].x[1])
        self.assertEqual(len(m.b), 1)
        self.assertEqual(len(m.b[1].x), 3)

        # While (2,2) appears to be a valid member of the slice, because
        # 2 was not in the Set when the Block rule fired, there is no
        # m.b[2] block data.  Accessing m.xx[2,1] will construct the
        # b[2] block data, fire the rule, and then add the new value to
        # the Var x.
        self.assertEqual(len(m.xx), 3)
        m.xx[2,2] = 10
        self.assertEqual(len(m.b), 2)
        self.assertEqual(len(list(m.b[2].component_objects())), 1)
        self.assertEqual(len(m.xx), 4)
        self.assertIs(m.xx[2,2], m.b[2].x[2])
        self.assertEqual(value(m.b[2].x[2]), 10)

    def test_insert_var(self):
        m = ConcreteModel()
        m.T = Set(initialize=[1,5])
        m.x = Var(m.T, initialize=lambda m,i: i)
        @m.Block(m.T)
        def b(b, i):
            b.y = Var(initialize=lambda b: 10*b.index())
        ref_x = Reference(m.x[:])
        ref_y = Reference(m.b[:].y)

        self.assertEqual(len(m.x), 2)
        self.assertEqual(len(ref_x), 2)
        self.assertEqual(len(m.b), 2)
        self.assertEqual(len(ref_y), 2)
        self.assertEqual(value(ref_x[1]), 1)
        self.assertEqual(value(ref_x[5]), 5)
        self.assertEqual(value(ref_y[1]), 10)
        self.assertEqual(value(ref_y[5]), 50)

        m.T.add(2)
        _x = ref_x[2]
        self.assertEqual(len(m.x), 3)
        self.assertIs(_x, m.x[2])
        self.assertEqual(value(_x), 2)
        self.assertEqual(value(m.x[2]), 2)
        self.assertEqual(value(ref_x[2]), 2)

        _y = ref_y[2]
        self.assertEqual(len(m.b), 3)
        self.assertIs(_y, m.b[2].y)
        self.assertEqual(value(_y), 20)
        self.assertEqual(value(ref_y[2]), 20)
        self.assertEqual(value(m.b[2].y), 20)

    def test_reference_to_dict(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var([1,2,3])
        m.r = Reference({1: m.x, 'a': m.y[2], 3: m.y[1]})
        self.assertFalse(m.r.index_set().isordered())
        self.assertEqual(len(m.r), 3)
        self.assertEqual(set(m.r.keys()), {1,3,'a'})
        self.assertEqual( ComponentSet(m.r.values()),
                          ComponentSet([m.x, m.y[2], m.y[1]]) )
        # You can delete something from the reference
        del m.r[1]
        self.assertEqual(len(m.r), 2)
        self.assertEqual(set(m.r.keys()), {3,'a'})
        self.assertEqual( ComponentSet(m.r.values()),
                          ComponentSet([m.y[2], m.y[1]]) )
        # But not add it back
        with self.assertRaisesRegex(
                KeyError, "Index '1' is not valid for indexed component 'r'"):
            m.r[1] = m.x

    def test_reference_to_list(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var([1,2,3])
        m.r = Reference([m.x, m.y[2], m.y[1]])
        self.assertTrue(m.r.index_set().isordered())
        self.assertEqual(len(m.r), 3)
        self.assertEqual(list(m.r.keys()), [0,1,2])
        self.assertEqual(list(m.r.values()), [m.x, m.y[2], m.y[1]])
        # You can delete something from the reference
        del m.r[1]
        self.assertEqual(len(m.r), 2)
        self.assertEqual(list(m.r.keys()), [0,2])
        self.assertEqual(list(m.r.values()), [m.x, m.y[1]])
        # But not add it back
        with self.assertRaisesRegex(
                KeyError, "Index '1' is not valid for indexed component 'r'"):
            m.r[1] = m.x

    def test_is_reference(self):
        m = ConcreteModel()
        m.v0 = Var()
        m.v1 = Var([1,2,3])

        m.ref0 = Reference(m.v0)
        m.ref1 = Reference(m.v1)

        self.assertFalse(m.v0.is_reference())
        self.assertFalse(m.v1.is_reference())

        self.assertTrue(m.ref0.is_reference())
        self.assertTrue(m.ref1.is_reference())

        unique_vars = list(
                v for v in m.component_objects(Var) if not v.is_reference())
        self.assertEqual(len(unique_vars), 2)

    def test_referent(self):
        m = ConcreteModel()
        m.v0 = Var()
        m.v2 = Var([1, 2, 3],['a', 'b'])

        varlist = [m.v2[1, 'a'], m.v2[1, 'b']]

        vardict = {
                0: m.v0, 
                1: m.v2[1, 'a'],
                2: m.v2[2, 'a'],
                3: m.v2[3, 'a'],
                }

        scalar_ref = Reference(m.v0)
        self.assertIs(scalar_ref.referent, m.v0)

        sliced_ref = Reference(m.v2[:,'a'])
        referent = sliced_ref.referent
        self.assertIs(type(referent), IndexedComponent_slice)
        self.assertEqual(len(referent._call_stack), 1)
        call, info = referent._call_stack[0]
        self.assertEqual(call, IndexedComponent_slice.slice_info)
        self.assertIs(info[0], m.v2)
        self.assertEqual(info[1], {1: 'a'}) # Fixed
        self.assertEqual(info[2], {0: slice(None)}) # Sliced
        self.assertIs(info[3], None) # Ellipsis

        list_ref = Reference(varlist)
        self.assertIs(list_ref.referent, varlist)

        dict_ref = Reference(vardict)
        self.assertIs(dict_ref.referent, vardict)

if __name__ == "__main__":
    unittest.main()
