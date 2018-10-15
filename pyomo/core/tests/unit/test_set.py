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

from pyomo.core.base.set import _ClosedNumericRange, Any, Reals, Integers

try:
    import numpy as np
    numpy_available = True
except ImportError:
    numpy_available = False

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

        with self.assertRaisesRegexp(
                ValueError,
                '.*start, end ordering incompatible with step direction'):
            CNR(0, 1, -1)

        with self.assertRaisesRegexp(
                ValueError,
                '.*start, end ordering incompatible with step direction'):
            CNR(1, 0, 1)

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

        self.assertEqual(str(CNR(1, 10, 0)), "[1,10]")
        self.assertEqual(str(CNR(1, 10, 1)), "[1:10]")
        self.assertEqual(str(CNR(1, 10, 3)), "[1:10:3]")
        self.assertEqual(str(CNR(1, 1, 1)), "[1,1]")

    def test_eq(self):
        CNR = _ClosedNumericRange

        self.assertEqual(CNR(1, 1, 1), CNR(1, 1, 1))
        self.assertEqual(CNR(1, None, 0), CNR(1, None, 0))
        self.assertEqual(CNR(0, 10, 3), CNR(0, 9, 3))

        self.assertNotEqual(CNR(1, 1, 1), CNR(1, None, 1))
        self.assertNotEqual(CNR(1, None, 0), CNR(1, None, 1))
        self.assertNotEqual(CNR(0, 10, 3), CNR(0, 8, 3))

    def test_contains(self):
        CNR = _ClosedNumericRange

        # Test non-numeric values
        self.assertNotIn(None, CNR(None, None, 0))
        self.assertNotIn(None, CNR(0, 10, 0))
        self.assertNotIn(None, CNR(0, None, 1))
        self.assertNotIn(None, CNR(0, 10, 1))

        self.assertNotIn('1', CNR(None, None, 0))
        self.assertNotIn('1', CNR(0, 10, 0))
        self.assertNotIn('1', CNR(0, None, 1))
        self.assertNotIn('1', CNR(0, 10, 1))

        # Test continuous ranges
        self.assertIn(0, CNR(0, 10, 0))
        self.assertIn(0, CNR(None, 10, 0))
        self.assertIn(0, CNR(0, None, 0))
        self.assertIn(1, CNR(0, 10, 0))
        self.assertIn(1, CNR(None, 10, 0))
        self.assertIn(1, CNR(0, None, 0))
        self.assertIn(10, CNR(0, 10, 0))
        self.assertIn(10, CNR(None, 10, 0))
        self.assertIn(10, CNR(0, None, 0))
        self.assertNotIn(-1, CNR(0, 10, 0))
        self.assertNotIn(-1, CNR(0, None, 0))
        self.assertNotIn(11, CNR(0, 10, 0))
        self.assertNotIn(11, CNR(None, 10, 0))

        # test discrete ranges (both increasing & decreasing)
        self.assertIn(0, CNR(0, 10, 1))
        self.assertIn(0, CNR(10, None, -1))
        self.assertIn(0, CNR(0, None, 1))
        self.assertIn(1, CNR(0, 10, 1))
        self.assertIn(1, CNR(10, None, -1))
        self.assertIn(1, CNR(0, None, 1))
        self.assertIn(10, CNR(0, 10, 1))
        self.assertIn(10, CNR(10, None, -1))
        self.assertIn(10, CNR(0, None, 1))
        self.assertNotIn(-1, CNR(0, 10, 1))
        self.assertNotIn(-1, CNR(0, None, 1))
        self.assertNotIn(11, CNR(0, 10, 1))
        self.assertNotIn(11, CNR(10, None, -1))
        self.assertNotIn(1.1, CNR(0, 10, 1))
        self.assertNotIn(1.1, CNR(10, None, -1))
        self.assertNotIn(1.1, CNR(0, None, 1))

        # test discrete ranges (increasing/decreasing by 2)
        self.assertIn(0, CNR(0, 10, 2))
        self.assertIn(0, CNR(0, -10, -2))
        self.assertIn(0, CNR(10, None, -2))
        self.assertIn(0, CNR(0, None, 2))
        self.assertIn(2, CNR(0, 10, 2))
        self.assertIn(-2, CNR(0, -10, -2))
        self.assertIn(2, CNR(10, None, -2))
        self.assertIn(2, CNR(0, None, 2))
        self.assertIn(10, CNR(0, 10, 2))
        self.assertIn(-10, CNR(0, -10, -2))
        self.assertIn(10, CNR(10, None, -2))
        self.assertIn(10, CNR(0, None, 2))
        self.assertNotIn(1, CNR(0, 10, 2))
        self.assertNotIn(-1, CNR(0, -10, -2))
        self.assertNotIn(1, CNR(10, None, -2))
        self.assertNotIn(1, CNR(0, None, 2))
        self.assertNotIn(-2, CNR(0, 10, 2))
        self.assertNotIn(2, CNR(0, -10, -2))
        self.assertNotIn(-2, CNR(0, None, 2))
        self.assertNotIn(12, CNR(0, 10, 2))
        self.assertNotIn(-12, CNR(0, -10, -2))
        self.assertNotIn(12, CNR(10, None, -2))
        self.assertNotIn(1.1, CNR(0, 10, 2))
        self.assertNotIn(1.1, CNR(0, -10, -2))
        self.assertNotIn(-1.1, CNR(10, None, -2))
        self.assertNotIn(1.1, CNR(0, None, 2))

    def test_isdisjoint(self):
        CNR = _ClosedNumericRange
        def _isdisjoint(expected_result, a, b):
            self.assertIs(expected_result, a.isdisjoint(b))
            self.assertIs(expected_result, b.isdisjoint(a))

        #
        # Simple continuous ranges
        _isdisjoint(True, CNR(0, 1, 0), CNR(2, 3, 0))
        _isdisjoint(True, CNR(2, 3, 0), CNR(0, 1, 0))

        _isdisjoint(False, CNR(0, 1, 0), CNR(1, 2, 0))
        _isdisjoint(False, CNR(0, 1, 0), CNR(0.5, 2, 0))
        _isdisjoint(False, CNR(0, 1, 0), CNR(0, 2, 0))
        _isdisjoint(False, CNR(0, 1, 0), CNR(-1, 2, 0))

        _isdisjoint(False, CNR(0, 1, 0), CNR(-1, 0, 0))
        _isdisjoint(False, CNR(0, 1, 0), CNR(-1, 0.5, 0))
        _isdisjoint(False, CNR(0, 1, 0), CNR(-1, 1, 0))
        _isdisjoint(False, CNR(0, 1, 0), CNR(-1, 2, 0))

        #
        # Continuous to discrete ranges (positive step)
        #
        _isdisjoint(True, CNR(0, 1, 0), CNR(2, 3, 1))
        _isdisjoint(True, CNR(2, 3, 0), CNR(0, 1, 1))

        _isdisjoint(False, CNR(0, 1, 0), CNR(-1, 2, 1))

        _isdisjoint(False, CNR(0.25, 1, 0), CNR(1, 2, 1))
        _isdisjoint(False, CNR(0.25, 1, 0), CNR(0.5, 2, 1))
        _isdisjoint(False, CNR(0.25, 1, 0), CNR(0, 2, 1))
        _isdisjoint(False, CNR(0.25, 1, 0), CNR(-1, 2, 1))

        _isdisjoint(False, CNR(0, 0.75, 0), CNR(-1, 0, 1))
        _isdisjoint(False, CNR(0, 0.75, 0), CNR(-1, 0.5, 1))
        _isdisjoint(False, CNR(0, 0.75, 0), CNR(-1, 1, 1))
        _isdisjoint(False, CNR(0, 0.75, 0), CNR(-1, 2, 1))

        # (additional edge cases)
        _isdisjoint(False, CNR(0, 0.99, 0), CNR(-1, 1, 1))
        _isdisjoint(True, CNR(0.001, 0.99, 0), CNR(-1, 1, 1))

        #
        # Continuous to discrete ranges (negative step)
        #
        _isdisjoint(True, CNR(0, 1, 0), CNR(3, 2, -1))
        _isdisjoint(True, CNR(2, 3, 0), CNR(1, 0, -1))

        _isdisjoint(False, CNR(0, 1, 0), CNR(2, -1, -1))

        _isdisjoint(False, CNR(0.25, 1, 0), CNR(2, 1, -1))
        _isdisjoint(False, CNR(0.25, 1, 0), CNR(2, 0.5, -1))
        _isdisjoint(False, CNR(0.25, 1, 0), CNR(2, 0, -1))
        _isdisjoint(False, CNR(0.25, 1, 0), CNR(2, -1, -1))

        _isdisjoint(False, CNR(0, 0.75, 0), CNR(0, -1, -1))
        _isdisjoint(False, CNR(0, 0.75, 0), CNR(0.5, -1, -1))
        _isdisjoint(False, CNR(0, 0.75, 0), CNR(1, -1, -1))
        _isdisjoint(False, CNR(0, 0.75, 0), CNR(2, -1, -1))

        # (additional edge cases)
        _isdisjoint(False, CNR(0, 0.99, 0), CNR(1, -1, -1))
        _isdisjoint(True, CNR(0.01, 0.99, 0), CNR(1, -1, -1))

        #
        # Discrete to discrete sets
        #
        _isdisjoint(False, CNR(0,10,2), CNR(2,10,2))
        _isdisjoint(True, CNR(0,10,2), CNR(1,10,2))

        _isdisjoint(False, CNR(0,50,5), CNR(0,50,7))
        _isdisjoint(False, CNR(0,34,5), CNR(0,34,7))
        _isdisjoint(False, CNR(5,50,5), CNR(7,50,7))
        _isdisjoint(True, CNR(5,34,5), CNR(7,34,7))
        _isdisjoint(False, CNR(5,50,5), CNR(49,7,-7))
        _isdisjoint(True, CNR(5,34,5), CNR(28,7,-7))

        # 1, 8, 15, 22, 29, 36
        _isdisjoint(False, CNR(0, None, 5), CNR(1, None, 7))
        _isdisjoint(False, CNR(0, None, -5), CNR(1, None, -7))
        # 2, 9, 16, 23, 30, 37
        _isdisjoint(True, CNR(0, None, 5), CNR(23, None, -7))
        # 0, 7, 14, 21, 28, 35
        _isdisjoint(False, CNR(0, None, 5), CNR(28, None, -7))

    def test_issubset(self):
        CNR = _ClosedNumericRange

        # Continuous-continuous
        self.assertTrue(CNR(0, 10, 0).issubset(CNR(0, 10, 0)))
        self.assertTrue(CNR(1, 10, 0).issubset(CNR(0, 10, 0)))
        self.assertTrue(CNR(0, 9, 0).issubset(CNR(0, 10, 0)))
        self.assertTrue(CNR(1, 9, 0).issubset(CNR(0, 10, 0)))
        self.assertFalse(CNR(0, 11, 0).issubset(CNR(0, 10, 0)))
        self.assertFalse(CNR(-1, 10, 0).issubset(CNR(0, 10, 0)))

        self.assertTrue(CNR(0, 10, 0).issubset(CNR(0, None, 0)))
        self.assertTrue(CNR(1, 10, 0).issubset(CNR(0, None, 0)))
        self.assertFalse(CNR(-1, 10, 0).issubset(CNR(0, None, 0)))

        self.assertTrue(CNR(0, 10, 0).issubset(CNR(None, 10, 0)))
        self.assertTrue(CNR(0, 9, 0).issubset(CNR(None, 10, 0)))
        self.assertFalse(CNR(0, 11, 0).issubset(CNR(None, 10, 0)))

        self.assertTrue(CNR(0, None, 0).issubset(CNR(None, None, 0)))
        self.assertTrue(CNR(0, None, 0).issubset(CNR(-1, None, 0)))
        self.assertTrue(CNR(0, None, 0).issubset(CNR(0, None, 0)))
        self.assertFalse(CNR(0, None, 0).issubset(CNR(1, None, 0)))
        self.assertFalse(CNR(0, None, 0).issubset(CNR(None, 1, 0)))

        self.assertTrue(CNR(None, 0, 0).issubset(CNR(None, None, 0)))
        self.assertTrue(CNR(None, 0, 0).issubset(CNR(None, 1, 0)))
        self.assertTrue(CNR(None, 0, 0).issubset(CNR(None, 0, 0)))
        self.assertFalse(CNR(None, 0, 0).issubset(CNR(None, -1, 0)))
        self.assertFalse(CNR(None, 0, 0).issubset(CNR(0, None, 0)))

        # Continuous - discrete
        self.assertTrue(CNR(0, None, 1).issubset(CNR(None, None, 0)))
        self.assertTrue(CNR(0, None, 1).issubset(CNR(0, None, 0)))
        self.assertFalse(CNR(0, None, 1).issubset(CNR(None, 0, 0)))

        self.assertTrue(CNR(0, None, -1).issubset(CNR(None, None, 0)))
        self.assertFalse(CNR(0, None, -1).issubset(CNR(0, None, 0)))
        self.assertTrue(CNR(0, None, -1).issubset(CNR(None, 0, 0)))

        self.assertTrue(CNR(0, 10, 1).issubset(CNR(None, None, 0)))
        self.assertTrue(CNR(0, 10, 1).issubset(CNR(0, None, 0)))
        self.assertTrue(CNR(0, 10, 1).issubset(CNR(0, 10, 0)))

        self.assertFalse(CNR(0, None, 0).issubset(CNR(0, None, 1)))
        self.assertFalse(CNR(0, 10, 0).issubset(CNR(0, 10, 1)))

        # Discrete - discrete
        self.assertTrue(CNR(0, 10, 2).issubset(CNR(0, 10, 2)))
        self.assertTrue(CNR(0, 10, 2).issubset(CNR(-2, 10, 2)))
        self.assertTrue(CNR(0, 10, 2).issubset(CNR(0, 12, 2)))
        self.assertFalse(CNR(0, 10, 3).issubset(CNR(0, 10, 2)))
        self.assertTrue(CNR(0, 11, 2).issubset(CNR(0, 10, 2)))
        self.assertFalse(CNR(1, 10, 2).issubset(CNR(0, 10, 2)))
        self.assertFalse(CNR(0, 10, 2).issubset(CNR(0, 10, 4)))
        self.assertTrue(CNR(0, 10, 2).issubset(CNR(0, 10, 1)))

        self.assertTrue(CNR(10, 0, -2).issubset(CNR(10, 0, -2)))
        self.assertTrue(CNR(10, 0, -2).issubset(CNR(10, -2, -2)))
        self.assertTrue(CNR(10, 0, -2).issubset(CNR(12, 0, -2)))
        self.assertFalse(CNR(10, 0, -3).issubset(CNR(10, 0, -2)))
        self.assertTrue(CNR(10, 1, -2).issubset(CNR(10, 0, -2)))
        self.assertTrue(CNR(8, 0, -2).issubset(CNR(10, 0, -2)))
        self.assertFalse(CNR(10, 0, -2).issubset(CNR(10, 0, -4)))
        self.assertTrue(CNR(10, 0, -2).issubset(CNR(10, 0, -1)))

class InfiniteSetTester(unittest.TestCase):
    def test_Reals(self):
        self.assertIn(0, Reals)
        self.assertIn(1.5, Reals)
        self.assertIn(100, Reals),
        self.assertIn(-100, Reals),
        self.assertNotIn('A', Reals)
        self.assertNotIn(None, Reals)

    def test_Integers(self):
        self.assertIn(0, Integers)
        self.assertNotIn(1.5, Integers)
        self.assertIn(100, Integers),
        self.assertIn(-100, Integers),
        self.assertNotIn('A', Integers)
        self.assertNotIn(None, Integers)

    @unittest.skipIf(not numpy_available, "NumPy required for these tests")
    def test_numpy_compatible(self):
        self.assertIn(np.intc(1), Reals)
        self.assertIn(np.float64(1), Reals)
        self.assertIn(np.float64(1.5), Reals)

        self.assertIn(np.intc(1), Integers)
        self.assertIn(np.float64(1), Integers)
        self.assertNotIn(np.float64(1.5), Integers)
