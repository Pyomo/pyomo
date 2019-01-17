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
from six import itervalues, StringIO

from pyomo.environ import *
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.sets import _SetProduct, SetOf
from pyomo.core.base.indexed_component import (
    UnindexedComponent_set, IndexedComponent
)
from pyomo.core.base.reference import (
    _ReferenceDict, _ReferenceSet, Reference, _get_base_sets
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

    def test_reference_set_wrapper(self):
        m = self.m
        rs = _ReferenceSet(_ReferenceDict(m.b[:,5].z))
        self.assertIn((1,), rs)
        self.assertEqual(len(rs), 2)
        self.assertEqual(list(rs), [1,2])


class TestReference(unittest.TestCase):
    def test_constructor_error(self):
        m = ConcreteModel()
        m.x = Var([1,2])
        class Foo(object): pass
        self.assertRaisesRegexp(
            TypeError,
            "First argument to Reference constructors must be a "
            "component or component slice \(received Foo",
            Reference, Foo()
            )
        self.assertRaisesRegexp(
            TypeError,
            "First argument to Reference constructors must be a "
            "component or component slice \(received int",
            Reference, 5
            )
        self.assertRaisesRegexp(
            TypeError,
            "First argument to Reference constructors must be a "
            "component or component slice \(received None",
            Reference, None
            )

    def test_component_reference(self):
        m = ConcreteModel()
        m.x = Var()
        m.r = Reference(m.x)

        self.assertIs(m.r.type(), Var)
        self.assertIsNot(m.r.index_set(), m.x.index_set())
        self.assertIs(m.x.index_set(), UnindexedComponent_set)
        self.assertIs(type(m.r.index_set()), SetOf)
        self.assertEqual(len(m.r), 1)
        self.assertTrue(m.r.is_indexed())
        self.assertIn(None, m.r)
        self.assertNotIn(1, m.r)
        self.assertIs(m.r[None], m.x)
        with self.assertRaises(KeyError):
            m.r[1]

        m.s = Reference(m.x[:])

        self.assertIs(m.s.type(), Var)
        self.assertIsNot(m.s.index_set(), m.x.index_set())
        self.assertIs(m.x.index_set(), UnindexedComponent_set)
        self.assertIs(type(m.s.index_set()), SetOf)
        self.assertEqual(len(m.s), 1)
        self.assertTrue(m.s.is_indexed())
        self.assertIn(None, m.s)
        self.assertNotIn(1, m.s)
        self.assertIs(m.s[None], m.x)
        with self.assertRaises(KeyError):
            m.s[1]

        m.y = Var([1,2])
        m.t = Reference(m.y)

        self.assertIs(m.t.type(), Var)
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

        self.assertIs(m.r.type(), Var)
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

        self.assertIs(m.r.type(), Var)
        self.assertIs(type(m.r.index_set()), _SetProduct)
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

        self.assertIs(m.r.type(), Var)
        self.assertIs(type(m.r.index_set()), _SetProduct)
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

        self.assertIs(m.r.type(), Var)
        self.assertIs(type(m.r.index_set()), SetOf)
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

        self.assertIs(m.r.type(), Var)
        self.assertIs(type(m.r.index_set()), SetOf)
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

        self.assertIs(m.r.type(), Var)
        self.assertIs(type(m.r.index_set()), SetOf)
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

        self.assertIs(m.r.type(), Var)
        self.assertIs(type(m.r.index_set()), SetOf)
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
        base_sets = list(_get_base_sets(m.r.index_set()))
        self.assertEqual(len(base_sets), 2)
        self.assertIs(type(base_sets[0]), SetOf)
        self.assertIs(type(base_sets[1]), SetOf)
        
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
        self.assertIs(m.y.type(), Var)
        m.y1 = Reference(m.b[:].y[...], ctype=None)
        self.assertIs(type(m.y1), IndexedComponent)
        self.assertIs(m.y1.type(), IndexedComponent)

        m.z = Reference(m.b[:].z)
        self.assertIs(type(m.z), IndexedComponent)
        self.assertIs(m.z.type(), IndexedComponent)

if __name__ == "__main__":
    unittest.main()
