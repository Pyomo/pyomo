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

from pyomo.core.base.set import (
    _ClosedNumericRange, _AnyRange, _AnySet,
    Any, Reals, NonNegativeReals, Integers, PositiveIntegers,
    InfiniteSimpleSet,
)
CNR = _ClosedNumericRange

try:
    import numpy as np
    numpy_available = True
except ImportError:
    numpy_available = False


class TestNumericRange(unittest.TestCase):
    def test_init(self):
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
        self.assertEqual(str(CNR(1, 10, 0)), "[1,10]")
        self.assertEqual(str(CNR(1, 10, 1)), "[1:10]")
        self.assertEqual(str(CNR(1, 10, 3)), "[1:10:3]")
        self.assertEqual(str(CNR(1, 1, 1)), "[1]")

    def test_eq(self):
        self.assertEqual(CNR(1, 1, 1), CNR(1, 1, 1))
        self.assertEqual(CNR(1, None, 0), CNR(1, None, 0))
        self.assertEqual(CNR(0, 10, 3), CNR(0, 9, 3))

        self.assertNotEqual(CNR(1, 1, 1), CNR(1, None, 1))
        self.assertNotEqual(CNR(1, None, 0), CNR(1, None, 1))
        self.assertNotEqual(CNR(0, 10, 3), CNR(0, 8, 3))

    def test_contains(self):
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

    def test_range_difference(self):
        self.assertEqual(
            CNR(0,None,1).range_difference([CNR(1,None,0)]),
            [CNR(0,0,0)],
        )
        self.assertEqual(
            CNR(0,None,1).range_difference([CNR(0,0,0)]),
            [CNR(1,None,1)],
        )
        self.assertEqual(
            CNR(0,None,2).range_difference([CNR(10,None,3)]),
            [CNR(0,None,6), CNR(2,None,6), CNR(4,4,0)],
        )

        # test ranges running in the other direction
        self.assertEqual(
            CNR(10,0,-1).range_difference([CNR(7,4,-2)]),
            [CNR(10,0,-2), CNR(1,3,2), CNR(9,9,0)],
        )
        self.assertEqual(
            CNR(0,None,-1).range_difference([CNR(-10,10,0)]),
            [CNR(-11,None,-1)],
        )

        # Test non-overlapping ranges
        self.assertEqual(
            CNR(0,4,0).range_difference([CNR(5,10,0)]),
            [CNR(0,4,0)],
        )
        self.assertEqual(
            CNR(5,10,0).range_difference([CNR(0,4,0)]),
            [CNR(5,10,0)],
        )

        # Test continuous ranges

        # FIXME: Subtracting a closed range from a closed range SHOULD
        # result in an open range.
        self.assertEqual(
            CNR(0,None,0).range_difference([CNR(5,None,0)]),
            [CNR(0,5,0)],
        )

    def test_range_intersection(self):
        self.assertEqual(
            CNR(0,None,1).range_intersection([CNR(1,None,0)]),
            [CNR(1,None,1)],
        )
        self.assertEqual(
            CNR(0,None,1).range_intersection([CNR(0,0,0)]),
            [CNR(0,0,0)],
        )
        self.assertEqual(
            CNR(0,None,2).range_intersection([CNR(1,None,3)]),
            [CNR(4,None,6)],
        )

        # Test non-overlapping ranges
        self.assertEqual(
            CNR(0,4,0).range_intersection([CNR(5,10,0)]),
            [],
        )
        self.assertEqual(
            CNR(5,10,0).range_intersection([CNR(0,4,0)]),
            [],
        )

        # test ranges running in the other direction
        self.assertEqual(
            CNR(10,0,-1).range_intersection([CNR(7,4,-2)]),
            [CNR(5,7,2)],
        )
        self.assertEqual(
            CNR(0,None,-1).range_intersection([CNR(None,-10,0)]),
            [CNR(-10,None,-1)],
        )

        # Test continuous ranges
        self.assertEqual(
            CNR(0,5,0).range_intersection([CNR(5,10,0)]),
            [CNR(5,5,0)],
        )
        self.assertEqual(
            CNR(0,None,0).range_intersection([CNR(5,None,0)]),
            [CNR(5,None,0)],
        )

    def test_pickle(self):
        a = CNR(0,100,5)
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a,b)
        self.assertEqual(a,b)

class TestAnyRange(unittest.TestCase):
    def test_str(self):
        self.assertEqual(str(_AnyRange()), '[*]')

    def test_range_relational(self):
        a = _AnyRange()
        b = _AnyRange()
        self.assertTrue(a.issubset(b))
        self.assertEqual(a, b)

        c = CNR(None, None, 0)
        self.assertFalse(a.issubset(c))
        self.assertTrue(c.issubset(b))
        self.assertNotEqual(a, c)

    def test_contains(self):
        a = _AnyRange()
        self.assertIn(None, a)
        self.assertIn(0, a)
        self.assertIn('a', a)

    def test_range_difference(self):
        self.assertEqual(
            _AnyRange().range_difference([CNR(0,None,1)]),
            [_AnyRange()]
        )
        self.assertEqual(
            CNR(0,None,1).range_difference([_AnyRange()]),
            []
        )

    def test_range_intersection(self):
        self.assertEqual(
            _AnyRange().range_intersection([CNR(0,None,1)]),
            [CNR(0,None,1)]
        )
        self.assertEqual(
            CNR(0,None,1).range_intersection([_AnyRange()]),
            [CNR(0,None,1)]
        )
        self.assertEqual(
            CNR(0,None,-1).range_intersection([_AnyRange()]),
            [CNR(0,None,-1)]
        )


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

    def test_Any(self):
        self.assertIn(0, Any)
        self.assertIn(1.5, Any)
        self.assertIn(100, Any),
        self.assertIn(-100, Any),
        self.assertIn('A', Any)
        self.assertIn(None, Any)

    @unittest.skipIf(not numpy_available, "NumPy required for these tests")
    def test_numpy_compatible(self):
        self.assertIn(np.intc(1), Reals)
        self.assertIn(np.float64(1), Reals)
        self.assertIn(np.float64(1.5), Reals)

        self.assertIn(np.intc(1), Integers)
        self.assertIn(np.float64(1), Integers)
        self.assertNotIn(np.float64(1.5), Integers)

    def test_relational_operators(self):
        Any2 = _AnySet()
        self.assertTrue(Any.issubset(Any2))
        self.assertTrue(Any.issuperset(Any2))
        self.assertFalse(Any.isdisjoint(Any2))

        Reals2 = InfiniteSimpleSet(ranges=(CNR(None,None,0),))
        self.assertTrue(Reals.issubset(Reals2))
        self.assertTrue(Reals.issuperset(Reals2))
        self.assertFalse(Reals.isdisjoint(Reals2))

        Integers2 = InfiniteSimpleSet(ranges=(CNR(0,None,-1), CNR(0,None,1)))
        self.assertTrue(Integers.issubset(Integers2))
        self.assertTrue(Integers.issuperset(Integers2))
        self.assertFalse(Integers.isdisjoint(Integers2))

        # Reals / Integers
        self.assertTrue(Integers.issubset(Reals))
        self.assertFalse(Integers.issuperset(Reals))
        self.assertFalse(Integers.isdisjoint(Reals))

        self.assertFalse(Reals.issubset(Integers))
        self.assertTrue(Reals.issuperset(Integers))
        self.assertFalse(Reals.isdisjoint(Integers))

        # Any / Reals
        self.assertTrue(Reals.issubset(Any))
        self.assertFalse(Reals.issuperset(Any))
        self.assertFalse(Reals.isdisjoint(Any))

        self.assertFalse(Any.issubset(Reals))
        self.assertTrue(Any.issuperset(Reals))
        self.assertFalse(Any.isdisjoint(Reals))

        # Integers / Positive Integers
        self.assertFalse(Integers.issubset(PositiveIntegers))
        self.assertTrue(Integers.issuperset(PositiveIntegers))
        self.assertFalse(Integers.isdisjoint(PositiveIntegers))

        self.assertTrue(PositiveIntegers.issubset(Integers))
        self.assertFalse(PositiveIntegers.issuperset(Integers))
        self.assertFalse(PositiveIntegers.isdisjoint(Integers))


    def test_equality(self):
        self.assertEqual(Any, _AnySet())
        self.assertEqual(
            Reals,
            InfiniteSimpleSet(ranges=(CNR(None,None,0),))
        )
        self.assertEqual(
            Integers,
            InfiniteSimpleSet(ranges=(CNR(0,None,-1), CNR(0,None,1)))
        )

        self.assertNotEqual(Integers, Reals)
        self.assertNotEqual(Reals, Integers)
        self.assertNotEqual(Reals, Any)
        self.assertNotEqual(Any, Reals)

        # For equality, ensure that the ranges can be in any order
        self.assertEqual(
            InfiniteSimpleSet(ranges=(CNR(0,None,-1), CNR(0,None,1))),
            InfiniteSimpleSet(ranges=(CNR(0,None,1), CNR(0,None,-1)))
        )

        # And integer ranges can be grounded at different points
        self.assertEqual(
            InfiniteSimpleSet(ranges=(CNR(10,None,-1), CNR(10,None,1))),
            InfiniteSimpleSet(ranges=(CNR(0,None,1), CNR(0,None,-1)))
        )
        self.assertEqual(
            InfiniteSimpleSet(ranges=(CNR(0,None,-1), CNR(0,None,1))),
            InfiniteSimpleSet(ranges=(CNR(10,None,1), CNR(10,None,-1)))
        )

        # Odd positive integers and even positive integers are positive
        # integers
        self.assertEqual(
            PositiveIntegers,
            InfiniteSimpleSet(ranges=(CNR(1,None,2), CNR(2,None,2)))
        )

        # Nututally prime sets of ranges
        self.assertEqual(
            InfiniteSimpleSet(ranges=(CNR(1,None,2), CNR(2,None,2))),
            InfiniteSimpleSet(ranges=(
                CNR(1,None,3), CNR(2,None,3), CNR(3,None,3)
            ))
        )

        # Omitting one of the subranges breaks equality
        # Nututally prime sets of ranges
        self.assertNotEqual(
            InfiniteSimpleSet(ranges=(CNR(1,None,2), CNR(2,None,2))),
            InfiniteSimpleSet(ranges=(
                CNR(1,None,3), CNR(2,None,3)
            ))
        )

        # Changing the reference point breaks equality
        # Nututally prime sets of ranges
        self.assertNotEqual(
            InfiniteSimpleSet(ranges=(CNR(0,None,2), CNR(0,None,2))),
            InfiniteSimpleSet(ranges=(
                CNR(1,None,3), CNR(2,None,3), CNR(3,None,3)
            ))
        )

        # # Concerns:
        #   - union blindly calls the nonexistant CNR.union() method
        #   - need to distinguish between finite and infinite sets?
        #   - need to standardize on where the reference point for
        #     discrete sets is (0?)

