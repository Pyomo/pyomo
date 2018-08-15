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
from pyomo.core.base.reference import _ReferenceDict


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
                r"'\(8, 10\)' is not valid for indexed component 'b\[1,4\].x'"):
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



if __name__ == "__main__":
    unittest.main()
