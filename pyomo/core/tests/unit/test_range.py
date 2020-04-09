#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pickle

import pyutilib.th as unittest

from pyomo.core.base.range import (
    NumericRange as NR, NonNumericRange as NNR, RangeProduct as RP,
    AnyRange, RangeDifferenceError
)
from pyomo.core.base.set import (
    Any
)

class TestNumericRange(unittest.TestCase):
    def test_init(self):
        a = NR(None, None, 0)
        self.assertIsNone(a.start)
        self.assertIsNone(a.end)
        self.assertEqual(a.step, 0)

        a = NR(-float('inf'), float('inf'), 0)
        self.assertIsNone(a.start)
        self.assertIsNone(a.end)
        self.assertEqual(a.step, 0)

        a = NR(0, None, 0)
        self.assertEqual(a.start, 0)
        self.assertIsNone(a.end)
        self.assertEqual(a.step, 0)

        a = NR(0, 0, 0)
        self.assertEqual(a.start, 0)
        self.assertEqual(a.end, 0)
        self.assertEqual(a.step, 0)

        with self.assertRaisesRegexp(
                ValueError, '.*start must be <= end for continuous ranges'):
            NR(0, -1, 0)


        with self.assertRaisesRegexp(ValueError, '.*start must not be None'):
            NR(None, None, 1)

        with self.assertRaisesRegexp(ValueError, '.*step must be int'):
            NR(None, None, 1.5)

        with self.assertRaisesRegexp(
                ValueError,
                '.*start, end ordering incompatible with step direction'):
            NR(0, 1, -1)

        with self.assertRaisesRegexp(
                ValueError,
                '.*start, end ordering incompatible with step direction'):
            NR(1, 0, 1)

        with self.assertRaisesRegexp(
                ValueError,
                '\[0:1\] is discrete, but passed closed=\(False, True\)'):
            NR(0, 1, 1, "(]")

        a = NR(0, None, 1)
        self.assertEqual(a.start, 0)
        self.assertEqual(a.end, None)
        self.assertEqual(a.step, 1)

        a = NR(0, 5, 1)
        self.assertEqual(a.start, 0)
        self.assertEqual(a.end, 5)
        self.assertEqual(a.step, 1)

        a = NR(0, 5, 2)
        self.assertEqual(a.start, 0)
        self.assertEqual(a.end, 4)
        self.assertEqual(a.step, 2)

        a = NR(0, 5, 10)
        self.assertEqual(a.start, 0)
        self.assertEqual(a.end, 0)
        self.assertEqual(a.step, 0)

        a = NR(0, 5.5, 1)
        self.assertEqual(a.start, 0)
        self.assertEqual(a.end, 5)
        self.assertEqual(a.step, 1)

        a = NR(0.5, 5.5, 1)
        self.assertEqual(a.start, 0.5)
        self.assertEqual(a.end, 5.5)
        self.assertEqual(a.step, 1)

        with self.assertRaisesRegexp(
                ValueError, '.*start, end ordering incompatible with step'):
            NR(0, -1, 1)

        with self.assertRaisesRegexp(
                ValueError, '.*start, end ordering incompatible with step'):
            NR(0, 1, -2)

    def test_str(self):
        self.assertEqual(str(NR(1, 10, 0)), "[1..10]")
        self.assertEqual(str(NR(1, 10, 1)), "[1:10]")
        self.assertEqual(str(NR(1, 10, 3)), "[1:10:3]")
        self.assertEqual(str(NR(1, 1, 1)), "[1]")

    def test_eq(self):
        self.assertEqual(NR(1, 1, 1), NR(1, 1, 1))
        self.assertEqual(NR(1, None, 0), NR(1, None, 0))
        self.assertEqual(NR(0, 10, 3), NR(0, 9, 3))

        self.assertNotEqual(NR(1, 1, 1), NR(1, None, 1))
        self.assertNotEqual(NR(1, None, 0), NR(1, None, 1))
        self.assertNotEqual(NR(0, 10, 3), NR(0, 8, 3))

    def test_contains(self):
        # Test non-numeric values
        self.assertNotIn(None, NR(None, None, 0))
        self.assertNotIn(None, NR(0, 10, 0))
        self.assertNotIn(None, NR(0, None, 1))
        self.assertNotIn(None, NR(0, 10, 1))

        self.assertNotIn('1', NR(None, None, 0))
        self.assertNotIn('1', NR(0, 10, 0))
        self.assertNotIn('1', NR(0, None, 1))
        self.assertNotIn('1', NR(0, 10, 1))

        # Test continuous ranges
        self.assertIn(0, NR(0, 10, 0))
        self.assertIn(0, NR(None, 10, 0))
        self.assertIn(0, NR(0, None, 0))
        self.assertIn(1, NR(0, 10, 0))
        self.assertIn(1, NR(None, 10, 0))
        self.assertIn(1, NR(0, None, 0))
        self.assertIn(10, NR(0, 10, 0))
        self.assertIn(10, NR(None, 10, 0))
        self.assertIn(10, NR(0, None, 0))
        self.assertNotIn(-1, NR(0, 10, 0))
        self.assertNotIn(-1, NR(0, None, 0))
        self.assertNotIn(11, NR(0, 10, 0))
        self.assertNotIn(11, NR(None, 10, 0))

        self.assertNotIn(0, NR(0.5, 10.5, 0))
        self.assertIn(0, NR(None, 10.5, 0))
        self.assertNotIn(0, NR(0.5, None, 0))
        self.assertNotIn(11, NR(0.5, 10.5, 0))
        self.assertNotIn(11, NR(None, 10.5, 0))
        self.assertIn(11, NR(0.5, None, 0))
        self.assertIn(1.5, NR(0.5, 10.5, 0))
        self.assertIn(1.5, NR(None, 10.5, 0))
        self.assertIn(1.5, NR(0.5, None, 0))

        # test discrete ranges (both increasing & decreasing)
        self.assertIn(0, NR(0, 10, 1))
        self.assertIn(0, NR(10, None, -1))
        self.assertIn(0, NR(0, None, 1))
        self.assertIn(1, NR(0, 10, 1))
        self.assertIn(1, NR(10, None, -1))
        self.assertIn(1, NR(0, None, 1))
        self.assertIn(10, NR(0, 10, 1))
        self.assertIn(10, NR(10, None, -1))
        self.assertIn(10, NR(0, None, 1))
        self.assertNotIn(-1, NR(0, 10, 1))
        self.assertNotIn(-1, NR(0, None, 1))
        self.assertNotIn(11, NR(0, 10, 1))
        self.assertNotIn(11, NR(10, None, -1))
        self.assertNotIn(1.1, NR(0, 10, 1))
        self.assertNotIn(1.1, NR(10, None, -1))
        self.assertNotIn(1.1, NR(0, None, 1))

        self.assertNotIn(0, NR(0.5, 10.5, 1))
        self.assertNotIn(0, NR(10.5, None, -1))
        self.assertNotIn(0, NR(0.5, None, 1))
        self.assertNotIn(11, NR(0.5, 10.5, 1))
        self.assertNotIn(11, NR(10.5, None, -1))
        self.assertNotIn(11, NR(0.5, None, 1))
        self.assertIn(1.5, NR(0.5, 10.5, 1))
        self.assertIn(1.5, NR(10.5, None, -1))
        self.assertIn(1.5, NR(0.5, None, 1))

        # test discrete ranges (increasing/decreasing by 2)
        self.assertIn(0, NR(0, 10, 2))
        self.assertIn(0, NR(0, -10, -2))
        self.assertIn(0, NR(10, None, -2))
        self.assertIn(0, NR(0, None, 2))
        self.assertIn(2, NR(0, 10, 2))
        self.assertIn(-2, NR(0, -10, -2))
        self.assertIn(2, NR(10, None, -2))
        self.assertIn(2, NR(0, None, 2))
        self.assertIn(10, NR(0, 10, 2))
        self.assertIn(-10, NR(0, -10, -2))
        self.assertIn(10, NR(10, None, -2))
        self.assertIn(10, NR(0, None, 2))
        self.assertNotIn(1, NR(0, 10, 2))
        self.assertNotIn(-1, NR(0, -10, -2))
        self.assertNotIn(1, NR(10, None, -2))
        self.assertNotIn(1, NR(0, None, 2))
        self.assertNotIn(-2, NR(0, 10, 2))
        self.assertNotIn(2, NR(0, -10, -2))
        self.assertNotIn(-2, NR(0, None, 2))
        self.assertNotIn(12, NR(0, 10, 2))
        self.assertNotIn(-12, NR(0, -10, -2))
        self.assertNotIn(12, NR(10, None, -2))
        self.assertNotIn(1.1, NR(0, 10, 2))
        self.assertNotIn(1.1, NR(0, -10, -2))
        self.assertNotIn(-1.1, NR(10, None, -2))
        self.assertNotIn(1.1, NR(0, None, 2))

    def test_isdisjoint(self):
        def _isdisjoint(expected_result, a, b):
            self.assertIs(expected_result, a.isdisjoint(b))
            self.assertIs(expected_result, b.isdisjoint(a))

        #
        # Simple continuous ranges
        _isdisjoint(True, NR(0, 1, 0), NR(2, 3, 0))
        _isdisjoint(True, NR(2, 3, 0), NR(0, 1, 0))

        _isdisjoint(False, NR(0, 1, 0), NR(1, 2, 0))
        _isdisjoint(False, NR(0, 1, 0), NR(0.5, 2, 0))
        _isdisjoint(False, NR(0, 1, 0), NR(0, 2, 0))
        _isdisjoint(False, NR(0, 1, 0), NR(-1, 2, 0))

        _isdisjoint(False, NR(0, 1, 0), NR(-1, 0, 0))
        _isdisjoint(False, NR(0, 1, 0), NR(-1, 0.5, 0))
        _isdisjoint(False, NR(0, 1, 0), NR(-1, 1, 0))
        _isdisjoint(False, NR(0, 1, 0), NR(-1, 2, 0))

        _isdisjoint(True, NR(0, 1, 0, (True,False)), NR(1, 2, 0))
        _isdisjoint(True, NR(0, 1, 0, (False,True)), NR(-1, 0, 0))

        #
        # Continuous to discrete ranges (positive step)
        #
        _isdisjoint(True, NR(0, 1, 0), NR(2, 3, 1))
        _isdisjoint(True, NR(2, 3, 0), NR(0, 1, 1))

        _isdisjoint(False, NR(0, 1, 0), NR(-1, 2, 1))

        _isdisjoint(False, NR(0.25, 1, 0), NR(1, 2, 1))
        _isdisjoint(False, NR(0.25, 1, 0), NR(0.5, 2, 1))
        _isdisjoint(False, NR(0.25, 1, 0), NR(0, 2, 1))
        _isdisjoint(False, NR(0.25, 1, 0), NR(-1, 2, 1))

        _isdisjoint(False, NR(0, 0.75, 0), NR(-1, 0, 1))
        _isdisjoint(False, NR(0, 0.75, 0), NR(-1, 0.5, 1))
        _isdisjoint(False, NR(0, 0.75, 0), NR(-1, 1, 1))
        _isdisjoint(False, NR(0, 0.75, 0), NR(-1, 2, 1))

        _isdisjoint(True, NR(0.1, 0.9, 0), NR(-1, 0, 1))
        _isdisjoint(True, NR(0.1, 0.9, 0), NR(-1, 0.5, 1))
        _isdisjoint(True, NR(0.1, 0.9, 0), NR(-1, 1, 1))
        _isdisjoint(True, NR(0.1, 0.9, 0), NR(-1, 2, 1))

        _isdisjoint(False, NR(-.1, 1.1, 0), NR(-1, 2, 1))
        _isdisjoint(False, NR(-.1, 1.1, 0), NR(-2, 0, 2))
        _isdisjoint(True, NR(-.1, 1.1, 0), NR(-1, -1, 1))
        _isdisjoint(True, NR(-.1, 1.1, 0), NR(-2, -1, 1))

        # (additional edge cases)
        _isdisjoint(False, NR(0, 1, 0, closed=(True,True)), NR(-1, 2, 1))
        _isdisjoint(False, NR(0, 1, 0, closed=(True,False)), NR(-1, 2, 1))
        _isdisjoint(False, NR(0, 1, 0, closed=(False,True)), NR(-1, 2, 1))
        _isdisjoint(True, NR(0, 1, 0, closed=(False,False)), NR(-1, 2, 1))
        _isdisjoint(True, NR(0.1, 1, 0, closed=(True,False)), NR(-1, 2, 1))
        _isdisjoint(True, NR(0, 0.9, 0, closed=(False,True)), NR(-1, 2, 1))
        _isdisjoint(False, NR(0, 0.99, 0), NR(-1, 1, 1))
        _isdisjoint(True, NR(0.001, 0.99, 0), NR(-1, 1, 1))

        #
        # Continuous to discrete ranges (negative step)
        #
        _isdisjoint(True, NR(0, 1, 0), NR(3, 2, -1))
        _isdisjoint(True, NR(2, 3, 0), NR(1, 0, -1))

        _isdisjoint(False, NR(0, 1, 0), NR(2, -1, -1))

        _isdisjoint(False, NR(0.25, 1, 0), NR(2, 1, -1))
        _isdisjoint(False, NR(0.25, 1, 0), NR(2, 0.5, -1))
        _isdisjoint(False, NR(0.25, 1, 0), NR(2, 0, -1))
        _isdisjoint(False, NR(0.25, 1, 0), NR(2, -1, -1))

        _isdisjoint(False, NR(0, 0.75, 0), NR(0, -1, -1))
        _isdisjoint(False, NR(0, 0.75, 0), NR(0.5, -1, -1))
        _isdisjoint(False, NR(0, 0.75, 0), NR(1, -1, -1))
        _isdisjoint(False, NR(0, 0.75, 0), NR(2, -1, -1))

        # (additional edge cases)
        _isdisjoint(False, NR(0, 0.99, 0), NR(1, -1, -1))
        _isdisjoint(True, NR(0.01, 0.99, 0), NR(1, -1, -1))

        #
        # Discrete to discrete sets
        #
        _isdisjoint(False, NR(0,10,2), NR(2,10,2))
        _isdisjoint(True, NR(0,10,2), NR(1,10,2))

        _isdisjoint(False, NR(0,50,5), NR(0,50,7))
        _isdisjoint(False, NR(0,34,5), NR(0,34,7))
        _isdisjoint(False, NR(5,50,5), NR(7,50,7))
        _isdisjoint(True, NR(5,34,5), NR(7,34,7))
        _isdisjoint(False, NR(5,50,5), NR(49,7,-7))
        _isdisjoint(True, NR(5,34,5), NR(28,7,-7))

        _isdisjoint(True, NR(0.25, 10, 1), NR(0.5, 20, 1))
        _isdisjoint(True, NR(0.25, 10, 1), NR(0.5, 20, 2))
        _isdisjoint(True, NR(0, 100, 2), NR(1, 100, 4))
        _isdisjoint(True, NR(0, None, 2), NR(1, None, 4))
        _isdisjoint(True, NR(0.25, None, 1), NR(0.5, None, 1))

        # 1, 8, 15, 22, 29, 36
        _isdisjoint(False, NR(0, None, 5), NR(1, None, 7))
        _isdisjoint(False, NR(0, None, -5), NR(1, None, -7))
        # 2, 9, 16, 23, 30, 37
        _isdisjoint(True, NR(0, None, 5), NR(23, None, -7))
        # 0, 7, 14, 21, 28, 35
        _isdisjoint(False, NR(0, None, 5), NR(28, None, -7))

    def test_issubset(self):
        # Continuous-continuous
        self.assertTrue(NR(0, 10, 0).issubset(NR(0, 10, 0)))
        self.assertTrue(NR(1, 10, 0).issubset(NR(0, 10, 0)))
        self.assertTrue(NR(0, 9, 0).issubset(NR(0, 10, 0)))
        self.assertTrue(NR(1, 9, 0).issubset(NR(0, 10, 0)))
        self.assertFalse(NR(0, 11, 0).issubset(NR(0, 10, 0)))
        self.assertFalse(NR(-1, 10, 0).issubset(NR(0, 10, 0)))

        self.assertTrue(NR(0, 10, 0).issubset(NR(0, None, 0)))
        self.assertTrue(NR(1, 10, 0).issubset(NR(0, None, 0)))
        self.assertFalse(NR(-1, 10, 0).issubset(NR(0, None, 0)))

        self.assertTrue(NR(0, 10, 0).issubset(NR(None, 10, 0)))
        self.assertTrue(NR(0, 9, 0).issubset(NR(None, 10, 0)))
        self.assertFalse(NR(0, 11, 0).issubset(NR(None, 10, 0)))

        self.assertTrue(NR(0, None, 0).issubset(NR(None, None, 0)))
        self.assertTrue(NR(0, None, 0).issubset(NR(-1, None, 0)))
        self.assertTrue(NR(0, None, 0).issubset(NR(0, None, 0)))
        self.assertFalse(NR(0, None, 0).issubset(NR(1, None, 0)))
        self.assertFalse(NR(0, None, 0).issubset(NR(None, 1, 0)))

        self.assertTrue(NR(None, 0, 0).issubset(NR(None, None, 0)))
        self.assertTrue(NR(None, 0, 0).issubset(NR(None, 1, 0)))
        self.assertTrue(NR(None, 0, 0).issubset(NR(None, 0, 0)))
        self.assertFalse(NR(None, 0, 0).issubset(NR(None, -1, 0)))
        self.assertFalse(NR(None, 0, 0).issubset(NR(0, None, 0)))

        B = True,True
        self.assertTrue(NR(0,1,0,(True,True)).issubset(NR(0,1,0,B)))
        self.assertTrue(NR(0,1,0,(True,False)).issubset(NR(0,1,0,B)))
        self.assertTrue(NR(0,1,0,(False,True)).issubset(NR(0,1,0,B)))
        self.assertTrue(NR(0,1,0,(False,False)).issubset(NR(0,1,0,B)))

        B = True,False
        self.assertFalse(NR(0,1,0,(True,True)).issubset(NR(0,1,0,B)))
        self.assertTrue(NR(0,1,0,(True,False)).issubset(NR(0,1,0,B)))
        self.assertFalse(NR(0,1,0,(False,True)).issubset(NR(0,1,0,B)))
        self.assertTrue(NR(0,1,0,(False,False)).issubset(NR(0,1,0,B)))

        B = False,True
        self.assertFalse(NR(0,1,0,(True,True)).issubset(NR(0,1,0,B)))
        self.assertFalse(NR(0,1,0,(True,False)).issubset(NR(0,1,0,B)))
        self.assertTrue(NR(0,1,0,(False,True)).issubset(NR(0,1,0,B)))
        self.assertTrue(NR(0,1,0,(False,False)).issubset(NR(0,1,0,B)))

        B = False,False
        self.assertFalse(NR(0,1,0,(True,True)).issubset(NR(0,1,0,B)))
        self.assertFalse(NR(0,1,0,(True,False)).issubset(NR(0,1,0,B)))
        self.assertFalse(NR(0,1,0,(False,True)).issubset(NR(0,1,0,B)))
        self.assertTrue(NR(0,1,0,(False,False)).issubset(NR(0,1,0,B)))

        # Continuous - discrete
        self.assertTrue(NR(0, None, 1).issubset(NR(None, None, 0)))
        self.assertTrue(NR(0, None, 1).issubset(NR(0, None, 0)))
        self.assertFalse(NR(0, None, 1).issubset(NR(None, 0, 0)))

        self.assertTrue(NR(0, None, -1).issubset(NR(None, None, 0)))
        self.assertFalse(NR(0, None, -1).issubset(NR(0, None, 0)))
        self.assertTrue(NR(0, None, -1).issubset(NR(None, 0, 0)))

        self.assertTrue(NR(0, 10, 1).issubset(NR(None, None, 0)))
        self.assertTrue(NR(0, 10, 1).issubset(NR(0, None, 0)))
        self.assertTrue(NR(0, 10, 1).issubset(NR(0, 10, 0)))

        self.assertFalse(NR(0, None, 0).issubset(NR(0, None, 1)))
        self.assertFalse(NR(None, 0, 0).issubset(NR(0, None, -1)))
        self.assertFalse(NR(0, 10, 0).issubset(NR(0, 10, 1)))

        # Discrete - discrete
        self.assertTrue(NR(0, 10, 2).issubset(NR(0, 10, 2)))
        self.assertTrue(NR(0, 10, 2).issubset(NR(-2, 10, 2)))
        self.assertTrue(NR(0, 10, 2).issubset(NR(0, 12, 2)))
        self.assertFalse(NR(0, 10, 3).issubset(NR(0, 10, 2)))
        self.assertTrue(NR(0, 11, 2).issubset(NR(0, 10, 2)))
        self.assertFalse(NR(1, 10, 2).issubset(NR(0, 10, 2)))
        self.assertFalse(NR(0, 10, 2).issubset(NR(0, 10, 4)))
        self.assertTrue(NR(0, 10, 2).issubset(NR(0, 10, 1)))

        self.assertTrue(NR(10, 0, -2).issubset(NR(10, 0, -2)))
        self.assertTrue(NR(10, 0, -2).issubset(NR(10, -2, -2)))
        self.assertTrue(NR(10, 0, -2).issubset(NR(12, 0, -2)))
        self.assertFalse(NR(10, 0, -3).issubset(NR(10, 0, -2)))
        self.assertTrue(NR(10, 1, -2).issubset(NR(10, 0, -2)))
        self.assertTrue(NR(8, 0, -2).issubset(NR(10, 0, -2)))
        self.assertFalse(NR(10, 0, -2).issubset(NR(10, 0, -4)))
        self.assertTrue(NR(10, 0, -2).issubset(NR(10, 0, -1)))

        # Scalar-discrete
        self.assertTrue(NR(5, 5, 0).issubset(NR(0, 10, 1)))
        self.assertFalse(NR(15, 15, 0).issubset(NR(0, 10, 1)))

    def test_lcm(self):
        self.assertEqual(
            NR(None,None,0)._step_lcm((NR(0,1,0),)),
            0
        )
        self.assertEqual(
            NR(None,None,0)._step_lcm((NR(0,0,0),)),
            1
        )
        self.assertEqual(
            NR(0,None,3)._step_lcm((NR(0,None,1),)),
            3
        )
        self.assertEqual(
            NR(0,None,3)._step_lcm((NR(0,None,0),)),
            3
        )
        self.assertEqual(
            NR(0,None,0)._step_lcm((NR(0,None,1),)),
            1
        )
        self.assertEqual(
            NR(0,None,3)._step_lcm((NR(0,None,2),)),
            6
        )
        self.assertEqual(
            NR(0,None,18)._step_lcm((NR(0,None,12),)),
            36
        )
        self.assertEqual(
            NR(0,None,3)._step_lcm((NR(0,None,2),NR(0,None,5))),
            30
        )
        self.assertEqual(
            NR(0,None,3)._step_lcm((NR(0,None,2),NR(0,None,10))),
            30
        )

    def test_range_difference(self):
        self.assertEqual(
            NR(0,None,1).range_difference([NR(1,None,0)]),
            [NR(0,0,0)],
        )
        self.assertEqual(
            NR(0,None,1).range_difference([NR(0,0,0)]),
            [NR(1,None,1)],
        )
        self.assertEqual(
            NR(0,None,2).range_difference([NR(10,None,3)]),
            [NR(0,None,6), NR(2,None,6), NR(4,4,0)],
        )
        with self.assertRaisesRegexp(ValueError, "Unknown range type, list"):
            NR(0,None,0).range_difference([[0]])

        # test relatively prime ranges that don't expand to all offsets
        self.assertEqual(
            NR(0,7,2).range_difference([NR(6,None,10)]),
            [NR(0,0,0), NR(2,2,0), NR(4,4,0)],
        )

        # test ranges running in the other direction
        self.assertEqual(
            NR(10,0,-1).range_difference([NR(7,4,-2)]),
            [NR(10,0,-2), NR(1,3,2), NR(9,9,0)],
        )
        self.assertEqual(
            NR(0,None,-1).range_difference([NR(-10,10,0)]),
            [NR(-11,None,-1)],
        )

        # Test non-overlapping ranges
        self.assertEqual(
            NR(0,4,0).range_difference([NR(5,10,0)]),
            [NR(0,4,0)],
        )
        self.assertEqual(
            NR(5,10,0).range_difference([NR(0,4,0)]),
            [NR(5,10,0)],
        )

        # Test continuous ranges

        # Subtracting a closed range from a closed range should
        # result in an open range.
        self.assertEqual(
            NR(0,None,0).range_difference([NR(5,None,0)]),
            [NR(0,5,0,'[)')],
        )
        self.assertEqual(
            NR(0,None,0).range_difference([NR(5,10,0)]),
            [NR(0,5,0,'[)'), NR(10,None,0,'(]')],
        )
        self.assertEqual(
            NR(None,0,0).range_difference([NR(-5,None,0)]),
            [NR(None,-5,0,'[)')],
        )
        self.assertEqual(
            NR(None,0,0).range_difference([NR(-5,0,0,'[)')]),
            [NR(None,-5,0,'[)')],
        )
        self.assertEqual(
            NR(0,10,0).range_difference([NR(None,5,0,'[)')]),
            [NR(5,10,0,'[]')],
        )
        # Subtracting an open range from a closed range gives a closed
        # range
        self.assertEqual(
            NR(0,None,0).range_difference([NR(5,10,0,'()')]),
            [NR(0,5,0,'[]'), NR(10,None,0,'[]')],
        )
        # Subtracting a discrete range from a continuous range gives a
        # set of open continuous ranges
        self.assertEqual(
            NR(None,None,0).range_difference([NR(5,10,5)]),
            [NR(None,5,0,'[)'), NR(5,10,0,'()'), NR(10,None,0,'(]')],
        )
        self.assertEqual(
            NR(-10,20,0).range_difference([NR(5,10,5)]),
            [NR(-10,5,0,'[)'), NR(5,10,0,'()'), NR(10,20,0,'(]')],
        )
        self.assertEqual(
            NR(-10,20,0,"()").range_difference([NR(5,10,5)]),
            [NR(-10,5,0,'()'), NR(5,10,0,'()'), NR(10,20,0,'()')],
        )
        self.assertEqual(
            NR(-3,3,0).range_difference([NR(0,None,5),NR(0,None,-5)]),
            [NR(-3,0,0,'[)'), NR(0,3,0,'(]')],
        )

        # Disjoint ranges...
        a = NR(0.25, 10, 1)
        self.assertEqual(a.range_difference([NR(0.5, 20, 1)]), [a])
        self.assertEqual(a.range_difference([NR(0.5, 20, 2)]),
                         [NR(0.25, 8.25, 2), NR(1.25, 9.25, 2)])
        a = NR(0, 100, 2)
        self.assertEqual(a.range_difference([NR(1, 100, 4)]),
                         [NR(0, 100, 4), NR(2, 98, 4)])
        a = NR(0, None, 2)
        self.assertEqual(a.range_difference([NR(1, None, 4)]),
                         [NR(0, None, 4), NR(2, None, 4)])
        a = NR(0.25, None, 1)
        self.assertEqual(a.range_difference([NR(0.5, None, 1)]), [a])

        # And the onee thing we don't support:
        with self.assertRaisesRegex(
                RangeDifferenceError, 'We do not support subtracting an '
                'infinite discrete range \[0:None\] from an infinite '
                'continuous range \[None..None\]'):
            NR(None,None,0).range_difference([NR(0,None,1)])

    def test_range_intersection(self):
        self.assertEqual(
            NR(0,None,1).range_intersection([NR(1,None,0)]),
            [NR(1,None,1)],
        )
        self.assertEqual(
            NR(0,None,1).range_intersection([NR(0,0,0)]),
            [NR(0,0,0)],
        )
        self.assertEqual(
            NR(0,None,1).range_intersection([NR(0.5,1.5,0)]),
            [NR(1,1,0)],
        )
        self.assertEqual(
            NR(0,None,2).range_intersection([NR(1,None,3)]),
            [NR(4,None,6)],
        )
        with self.assertRaisesRegexp(ValueError, "Unknown range type, list"):
            NR(0,None,0).range_intersection([[0]])

        # Test non-overlapping ranges
        self.assertEqual(
            NR(0,4,0).range_intersection([NR(5,10,0)]),
            [],
        )
        self.assertEqual(
            NR(5,10,0).range_intersection([NR(0,4,0)]),
            [],
        )
        self.assertEqual(
            NR(0,4,0).range_intersection([NNR('a')]),
            [],
        )

        # test ranges running in the other direction
        self.assertEqual(
            NR(10,0,-1).range_intersection([NR(7,4,-2)]),
            [NR(5,7,2)],
        )
        self.assertEqual(
            NR(10,0,-1).range_intersection([NR(7,None,-2)]),
            [NR(1,7,2)],
        )
        self.assertEqual(
            NR(0,None,-1).range_intersection([NR(None,-10,0)]),
            [NR(-10,None,-1)],
        )

        # Test continuous ranges
        self.assertEqual(
            NR(0,5,0).range_intersection([NR(5,10,0)]),
            [NR(5,5,0)],
        )
        self.assertEqual(
            NR(0,None,0).range_intersection([NR(5,None,0)]),
            [NR(5,None,0)],
        )

        # Disjoint ranges...
        a = NR(0.25, 10, 1)
        self.assertEqual(a.range_intersection([NR(0.5, 20, 1)]), [])
        self.assertEqual(a.range_intersection([NR(0.5, 20, 2)]), [])
        a = NR(0, 100, 2)
        self.assertEqual(a.range_intersection([NR(1, 100, 4)]), [])
        a = NR(0, None, 2)
        self.assertEqual(a.range_intersection([NR(1, None, 4)]), [])
        a = NR(0.25, None, 1)
        self.assertEqual(a.range_intersection([NR(0.5, None, 1)]), [])

    def test_pickle(self):
        a = NR(0,100,5)
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a,b)
        self.assertEqual(a,b)


class TestAnyRange(unittest.TestCase):
    def test_str(self):
        self.assertEqual(str(AnyRange()), '[*]')

    def test_range_relational(self):
        a = AnyRange()
        b = AnyRange()
        self.assertTrue(a.issubset(b))
        self.assertEqual(a, a)
        self.assertEqual(a, b)

        c = NR(None, None, 0)
        self.assertFalse(a.issubset(c))
        self.assertTrue(c.issubset(b))
        self.assertNotEqual(a, c)
        self.assertNotEqual(c, a)

    def test_contains(self):
        a = AnyRange()
        self.assertIn(None, a)
        self.assertIn(0, a)
        self.assertIn('a', a)

    def test_range_difference(self):
        self.assertEqual(
            AnyRange().range_difference([NR(0,None,1)]),
            [AnyRange()]
        )
        self.assertEqual(
            NR(0,None,1).range_difference([AnyRange()]),
            []
        )
        self.assertEqual(
            AnyRange().range_difference([AnyRange()]),
            []
        )

    def test_range_intersection(self):
        self.assertEqual(
            AnyRange().range_intersection([NR(0,None,1)]),
            [NR(0,None,1)]
        )
        self.assertEqual(
            NR(0,None,1).range_intersection([AnyRange()]),
            [NR(0,None,1)]
        )
        self.assertEqual(
            NR(0,None,-1).range_intersection([AnyRange()]),
            [NR(0,None,-1)]
        )

    def test_info_methods(self):
        a = AnyRange()
        self.assertFalse(a.isdiscrete())
        self.assertFalse(a.isfinite())

    def test_pickle(self):
        a = AnyRange()
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a,b)
        self.assertEqual(a,b)


class TestNonNumericRange(unittest.TestCase):
    def test_str(self):
        a = NNR('a')
        aa = NNR('a')
        b = NNR(None)
        self.assertEqual(str(a), '{a}')
        self.assertEqual(str(aa), '{a}')
        self.assertEqual(str(b), '{None}')

    def test_range_relational(self):
        a = NNR('a')
        aa = NNR('a')
        b = NNR(None)
        self.assertTrue(a.issubset(aa))
        self.assertFalse(a.issubset(b))
        self.assertEqual(a, a)
        self.assertEqual(a, aa)
        self.assertNotEqual(a, b)

        c = NR(None, None, 0)
        self.assertFalse(a.issubset(c))
        self.assertFalse(c.issubset(b))
        self.assertNotEqual(a, c)
        self.assertNotEqual(c, a)

    def test_contains(self):
        a = NNR('a')
        b = NNR(None)
        self.assertIn('a', a)
        self.assertNotIn(0, a)
        self.assertNotIn(None, a)
        self.assertNotIn('a', b)
        self.assertNotIn(0, b)
        self.assertIn(None, b)

    def test_range_difference(self):
        a = NNR('a')
        b = NNR(None)
        self.assertEqual(
            a.range_difference([NNR('a')]),
            []
        )
        self.assertEqual(
            a.range_difference([b]),
            [NNR('a')]
        )

    def test_range_intersection(self):
        a = NNR('a')
        b = NNR(None)
        self.assertEqual(
            a.range_intersection([b]),
            []
        )
        self.assertEqual(
            a.range_intersection([NNR('a')]),
            [NNR('a')]
        )

    def test_info_methods(self):
        a = NNR('a')
        self.assertTrue(a.isdiscrete())
        self.assertTrue(a.isfinite())

    def test_pickle(self):
        a = NNR('a')
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a,b)
        self.assertEqual(a,b)


class TestRangeProduct(unittest.TestCase):
    def test_str(self):
        a = RP([[NR(0,10,1)],[NR(0,10,0),NNR('a')]])
        self.assertEqual(str(a), '<[0:10], ([0..10], {a})>')

    def test_range_relational(self):
        a = RP([[NR(0,10,1)],[NR(0,10,0),NNR('a')]])
        aa = RP([[NR(0,10,1)],[NR(0,10,0),NNR('a')]])
        b = RP([[NR(0,10,1)],[NR(0,10,0),NNR('a'),NNR('b')]])
        c = RP([[NR(0,10,1)],[NR(0,10,0),NNR('b')]])
        d = RP([[NR(0,10,0)],[NR(0,10,0),NNR('a')]])
        d = RP([[NR(0,10,0)],[AnyRange()]])

        self.assertTrue(a.issubset(aa))
        self.assertTrue(a.issubset(b))
        self.assertFalse(a.issubset(c))
        self.assertTrue(a.issubset(d))

        self.assertFalse(a.issubset(NNR('a')))
        self.assertFalse(a.issubset(NR(None,None,0)))
        self.assertTrue(a.issubset(AnyRange()))

    def test_contains(self):
        a = NNR('a')
        b = NR(0,5,0)
        c = NR(5,10,1)
        x = RP([[a],[b,c]])
        self.assertNotIn('a', x)
        self.assertNotIn(0, x)
        self.assertNotIn(None, x)
        self.assertIn(('a',0), x)
        self.assertIn(('a',6), x)
        self.assertNotIn(('a',6.5), x)

    def test_equality(self):
        a = NNR('a')
        b = NR(0,5,0)
        c = NR(5,10,1)
        x = RP([[a],[b,c]])
        y = RP([[a],[c]])
        self.assertEqual(x,x)
        self.assertNotEqual(x,y)

    def test_isdisjoint(self):
        a = NNR('a')
        b = NR(0,5,0)
        c = NR(5,10,1)
        x = RP([[a],[b,c]])
        y = RP([[a],[c]])
        z = RP([[a],[b],[c]])
        w = RP([[AnyRange()], [b]])
        self.assertFalse(x.isdisjoint(x))
        self.assertFalse(x.isdisjoint(y))
        self.assertTrue(x.isdisjoint(z))
        self.assertFalse(x.isdisjoint(w))
        self.assertTrue(x.isdisjoint(a))
        self.assertFalse(y.isdisjoint(w))
        self.assertFalse(x.isdisjoint(AnyRange()))
        v = RP([[AnyRange()],[NR(0,5,0,(False,False))]])
        self.assertTrue(y.isdisjoint(v))

    def test_range_difference(self):
        a = NNR('a')
        b = NR(0,5,0)
        c = NR(5,10,1)
        x = RP([[a],[b,c]])
        y = RP([[a],[c]])
        z = RP([[a],[b],[c]])
        w = RP([list(Any.ranges()), [b]])
        self.assertEqual(x.range_difference([x]), [])
        self.assertEqual(x.range_difference([y]), [RP([[a],[b]])])
        self.assertEqual(x.range_difference([z]), [x])
        self.assertEqual(x.range_difference(Any.ranges()), [])
        self.assertEqual(x.range_difference([w]), [RP([[a],[NR(6,10,1)]])])
        v = RP([[AnyRange()],[NR(0,5,0,(False,False))]])
        self.assertEqual(y.range_difference([v]), [y])

    def test_range_intersection(self):
        a = NNR('a')
        b = NR(0,5,0)
        c = NR(5,10,1)
        x = RP([[a],[b,c]])
        y = RP([[a],[c]])
        z = RP([[a],[b],[c]])
        w = RP([list(Any.ranges()), [b]])
        self.assertEqual(x.range_intersection([x]), [x])
        self.assertEqual(x.range_intersection([y]), [y])
        self.assertEqual(x.range_intersection([z]), [])
        self.assertEqual(x.range_intersection(Any.ranges()), [x])
        self.assertEqual(x.range_intersection([w]), [RP([[a],[b]])])
        self.assertEqual(y.range_intersection([w]), [RP([[a],[NR(5,5,0)]])])
        v = RP([[AnyRange()],[NR(0,5,0,(False,False))]])
        self.assertEqual(y.range_intersection([v]), [])

    def test_info_methods(self):
        a = NNR('a')
        b = NR(0,5,0)
        c = NR(5,10,1)
        x = RP([[a],[b,c]])
        y = RP([[a],[c]])
        self.assertFalse(x.isdiscrete())
        self.assertFalse(x.isfinite())
        self.assertTrue(y.isdiscrete())
        self.assertTrue(y.isfinite())

    def test_pickle(self):
        a = NNR('a')
        b = NR(0,5,0)
        c = NR(5,10,1)
        x = RP([[a],[b,c]])
        y = RP([[a],[c]])

        xx = pickle.loads(pickle.dumps(x))
        self.assertIsNot(x,xx)
        self.assertEqual(x,xx)

        yy = pickle.loads(pickle.dumps(y))
        self.assertIsNot(y,yy)
        self.assertEqual(y,yy)


