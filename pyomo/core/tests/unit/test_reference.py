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
from pyomo.core.base.sets import _SetProduct, SetOf
from pyomo.core.base.reference import _ReferenceDict, Reference


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


    def test_len(self):
        m = self.m

        rd = _ReferenceDict(m.b[:,:].x[:,:])
        self.assertEqual(len(rd), 2*2*2*2)

        rd = _ReferenceDict(m.b[:,4].x[8,:])
        self.assertEqual(len(rd), 2*2)

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

class TestReference(unittest.TestCase):
    def test_constructor_error(self):
        m = ConcreteModel()
        m.x = Var([1,2])
        self.assertRaisesRegexp(
            TypeError,
            "First argument to Reference constructors must be a "
            "component slice.*received IndexedVar",
            Reference, m.x
            )
        self.assertRaisesRegexp(
            TypeError,
            "First argument to Reference constructors must be a "
            "component slice.*received None",
            Reference, None
            )

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



if __name__ == "__main__":
    unittest.main()
