#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.th as unittest

from pyomo.core.base.set import _ClosedNumericRange


class TestNumericRange(unittest.TestCase):
    def test_init(self):
        CNR = _ClosedNumericRange

        a = CNR(None, None, 0)
        self.assertIsNone(a.start)
        self.assertIsNone(a.end)
        self.assertEqual(a.step, 0)

        a = CNR(0, None, 0)
        self.assertEqual(a.start, 0)
        self.assertIsNone(a.end)
        self.assertEqual(a.step, 0)

        a = CNR(0, 0, 0)
        self.assertEqual(a.start, 0)
        self.assertEqual(a.end, 0)
        self.assertEqual(a.step, 0)

        with self.assertRaisesRegexp(
                ValueError, '.*start must be <= end for continuous ranges'):
            CNR(0, -1, 0)


        with self.assertRaisesRegexp(ValueError, '.*start must not be None'):
            CNR(None, None, 1)

        with self.assertRaisesRegexp(ValueError, '.*step must be int'):
            CNR(None, None, 1.5)

        a = CNR(0, None, 1)
        self.assertEqual(a.start, 0)
        self.assertEqual(a.end, None)
        self.assertEqual(a.step, 1)

        a = CNR(0, 5, 1)
        self.assertEqual(a.start, 0)
        self.assertEqual(a.end, 5)
        self.assertEqual(a.step, 1)

        a = CNR(0, 5, 2)
        self.assertEqual(a.start, 0)
        self.assertEqual(a.end, 4)
        self.assertEqual(a.step, 2)

        a = CNR(0, 5, 10)
        self.assertEqual(a.start, 0)
        self.assertEqual(a.end, 0)
        self.assertEqual(a.step, 0)

        with self.assertRaisesRegexp(
                ValueError, '.*start, end ordering incompatible with step'):
            CNR(0, -1, 1)

        with self.assertRaisesRegexp(
                ValueError, '.*start, end ordering incompatible with step'):
            CNR(0, 1, -2)
        
    def test_str(self):
        CNR = _ClosedNumericRange

        self.assertEqual(str(CNR(1,10,0)), "[1,10]")
        self.assertEqual(str(CNR(1,10,1)), "[1:10]")
        self.assertEqual(str(CNR(1,10,3)), "[1:10:3]")
        self.assertEqual(str(CNR(1,1,1)), "[1,1]")

    def test_eq(self):
        CNR = _ClosedNumericRange

        self.assertEqual(CNR(1,1,1), CNR(1,1,1))
        self.assertEqual(CNR(1,None,0), CNR(1,None,0))
        self.assertEqual(CNR(0,10,3), CNR(0,9,3))

        self.assertNotEqual(CNR(1,1,1), CNR(1,None,1))
        self.assertNotEqual(CNR(1,None,0), CNR(1,None,1))
        self.assertNotEqual(CNR(0,10,3), CNR(0,8,3))

    def test_contains(self):
        CNR = _ClosedNumericRange

        self.assertIn(0, CNR(0,10,0))
        self.assertIn(0, CNR(None,10,0))
        self.assertIn(0, CNR(0,None,0))
        self.assertIn(1, CNR(0,10,0))
        self.assertIn(1, CNR(None,10,0))
        self.assertIn(1, CNR(0,None,0))
        self.assertIn(10, CNR(0,10,0))
        self.assertIn(10, CNR(None,10,0))
        self.assertIn(10, CNR(0,None,0))
        self.assertNotIn(-1, CNR(0,10,0))
        self.assertNotIn(-1, CNR(0,None,0))
        self.assertNotIn(11, CNR(0,10,0))
        self.assertNotIn(11, CNR(None,10,0))

        self.assertIn(0, CNR(0,10,1))
        self.assertIn(0, CNR(10,None,-1))
        self.assertIn(0, CNR(0,None,1))
        self.assertIn(1, CNR(0,10,1))
        self.assertIn(1, CNR(10,None,-1))
        self.assertIn(1, CNR(0,None,1))
        self.assertIn(10, CNR(0,10,1))
        self.assertIn(10, CNR(10,None,-1))
        self.assertIn(10, CNR(0,None,1))
        self.assertNotIn(-1, CNR(0,10,1))
        self.assertNotIn(-1, CNR(0,None,1))
        self.assertNotIn(11, CNR(0,10,1))
        self.assertNotIn(11, CNR(10,None,-1))
        self.assertNotIn(1.1, CNR(0,10,2))
        self.assertNotIn(1.1, CNR(10,None,-2))
        self.assertNotIn(1.1, CNR(0,None,2))

        self.assertIn(0, CNR(0,10,2))
        self.assertIn(0, CNR(0,-10,-2))
        self.assertIn(0, CNR(10,None,-2))
        self.assertIn(0, CNR(0,None,2))
        self.assertIn(2, CNR(0,10,2))
        self.assertIn(-2, CNR(0,-10,-2))
        self.assertIn(2, CNR(10,None,-2))
        self.assertIn(2, CNR(0,None,2))
        self.assertIn(10, CNR(0,10,2))
        self.assertIn(-10, CNR(0,-10,-2))
        self.assertIn(10, CNR(10,None,-2))
        self.assertIn(10, CNR(0,None,2))
        self.assertNotIn(1, CNR(0,10,2))
        self.assertNotIn(-1, CNR(0,-10,-2))
        self.assertNotIn(1, CNR(10,None,-2))
        self.assertNotIn(1, CNR(0,None,2))
        self.assertNotIn(-2, CNR(0,10,2))
        self.assertNotIn(2, CNR(0,-10,-2))
        self.assertNotIn(-2, CNR(0,None,2))
        self.assertNotIn(12, CNR(0,10,2))
        self.assertNotIn(-12, CNR(0,-10,-2))
        self.assertNotIn(12, CNR(10,None,-2))
        self.assertNotIn(1.1, CNR(0,10,2))
        self.assertNotIn(1.1, CNR(0,-10,-2))
        self.assertNotIn(-1.1, CNR(10,None,-2))
        self.assertNotIn(1.1, CNR(0,None,2))
