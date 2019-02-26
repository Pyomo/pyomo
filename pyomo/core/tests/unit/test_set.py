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
from six import StringIO

import pyutilib.th as unittest

from pyomo.core.base.set import (
    _ClosedNumericRange as CNR, _NonNumericRange as NNR, _AnyRange, _AnySet,
    Any, Reals, NonNegativeReals, Integers, PositiveIntegers,
    RangeSet, Set, SetOf
)
from pyomo.environ import ConcreteModel

try:
    import numpy as np
    numpy_available = True
except ImportError:
    numpy_available = False


class TestClosedNumericRange(unittest.TestCase):
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
        self.assertFalse(CNR(None, 0, 0).issubset(CNR(0, None, -1)))
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

    def test_lcm(self):
        self.assertEqual(
            CNR(None,None,0)._lcm((CNR(0,1,0),)),
            0
        )
        self.assertEqual(
            CNR(None,None,0)._lcm((CNR(0,0,0),)),
            1
        )
        self.assertEqual(
            CNR(0,None,3)._lcm((CNR(0,None,1),)),
            3
        )
        self.assertEqual(
            CNR(0,None,3)._lcm((CNR(0,None,0),)),
            3
        )
        self.assertEqual(
            CNR(0,None,0)._lcm((CNR(0,None,1),)),
            1
        )
        self.assertEqual(
            CNR(0,None,3)._lcm((CNR(0,None,2),)),
            6
        )
        self.assertEqual(
            CNR(0,None,3)._lcm((CNR(0,None,2),CNR(0,None,5))),
            30
        )
        self.assertEqual(
            CNR(0,None,3)._lcm((CNR(0,None,2),CNR(0,None,10))),
            30
        )

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

        # test relatively prime ranges that don't expand to all offsets
        self.assertEqual(
            CNR(0,7,2).range_difference([CNR(6,None,10)]),
            [CNR(0,0,0), CNR(2,2,0), CNR(4,4,0)],
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
            CNR(10,0,-1).range_intersection([CNR(7,None,-2)]),
            [CNR(1,7,2)],
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

        Reals2 = RangeSet(ranges=(CNR(None,None,0),))
        self.assertTrue(Reals.issubset(Reals2))
        self.assertTrue(Reals.issuperset(Reals2))
        self.assertFalse(Reals.isdisjoint(Reals2))

        Integers2 = RangeSet(ranges=(CNR(0,None,-1), CNR(0,None,1)))
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
        self.assertEqual(Any, Any)
        self.assertEqual(Reals, Reals)
        self.assertEqual(PositiveIntegers, PositiveIntegers)

        self.assertEqual(Any, _AnySet())
        self.assertEqual(
            Reals,
            RangeSet(ranges=(CNR(None,None,0),))
        )
        self.assertEqual(
            Integers,
            RangeSet(ranges=(CNR(0,None,-1), CNR(0,None,1)))
        )

        self.assertNotEqual(Integers, Reals)
        self.assertNotEqual(Reals, Integers)
        self.assertNotEqual(Reals, Any)
        self.assertNotEqual(Any, Reals)

        # For equality, ensure that the ranges can be in any order
        self.assertEqual(
            RangeSet(ranges=(CNR(0,None,-1), CNR(0,None,1))),
            RangeSet(ranges=(CNR(0,None,1), CNR(0,None,-1)))
        )

        # And integer ranges can be grounded at different points
        self.assertEqual(
            RangeSet(ranges=(CNR(10,None,-1), CNR(10,None,1))),
            RangeSet(ranges=(CNR(0,None,1), CNR(0,None,-1)))
        )
        self.assertEqual(
            RangeSet(ranges=(CNR(0,None,-1), CNR(0,None,1))),
            RangeSet(ranges=(CNR(10,None,1), CNR(10,None,-1)))
        )

        # Odd positive integers and even positive integers are positive
        # integers
        self.assertEqual(
            PositiveIntegers,
            RangeSet(ranges=(CNR(1,None,2), CNR(2,None,2)))
        )

        # Nututally prime sets of ranges
        self.assertEqual(
            RangeSet(ranges=(CNR(1,None,2), CNR(2,None,2))),
            RangeSet(ranges=(
                CNR(1,None,3), CNR(2,None,3), CNR(3,None,3)
            ))
        )

        # Nututally prime sets of ranges
        #  ...omitting one of the subranges breaks equality
        self.assertNotEqual(
            RangeSet(ranges=(CNR(1,None,2), CNR(2,None,2))),
            RangeSet(ranges=(
                CNR(1,None,3), CNR(2,None,3)
            ))
        )

        # Mututally prime sets of ranges
        #  ...changing a reference point (so redundant CNR) breaks equality
        self.assertNotEqual(
            RangeSet(ranges=(CNR(0,None,2), CNR(0,None,2))),
            RangeSet(ranges=(
                CNR(1,None,3), CNR(2,None,3), CNR(3,None,3)
            ))
        )


class TestRangeOperations(unittest.TestCase):
    def test_mixed_ranges_isdisjoint(self):
        i = RangeSet(0,10,2)
        j = SetOf([0,1,2,'a'])
        k = Any

        ir = list(i.ranges())
        self.assertEqual(ir, [CNR(0,10,2)])
        self.assertEqual(str(ir), "[[0:10:2]]")
        ir = ir[0]

        jr = list(j.ranges())
        self.assertEqual(jr, [CNR(0,0,0), CNR(1,1,0), CNR(2,2,0), NNR('a')])
        self.assertEqual(str(jr), "[[0], [1], [2], {a}]")
        jr0, jr1, jr2, jr3 = jr

        kr = list(k.ranges())
        self.assertEqual(kr, [_AnyRange()])
        self.assertEqual(str(kr), "[[*]]")
        kr = kr[0]

        self.assertFalse(ir.isdisjoint(ir))
        self.assertFalse(ir.isdisjoint(jr0))
        self.assertTrue(ir.isdisjoint(jr1))
        self.assertTrue(ir.isdisjoint(jr3))
        self.assertFalse(ir.isdisjoint(kr))

        self.assertFalse(jr0.isdisjoint(ir))
        self.assertFalse(jr0.isdisjoint(jr0))
        self.assertTrue(jr0.isdisjoint(jr1))
        self.assertTrue(jr0.isdisjoint(jr3))
        self.assertFalse(jr0.isdisjoint(kr))

        self.assertTrue(jr1.isdisjoint(ir))
        self.assertTrue(jr1.isdisjoint(jr0))
        self.assertFalse(jr1.isdisjoint(jr1))
        self.assertTrue(jr1.isdisjoint(jr3))
        self.assertFalse(jr1.isdisjoint(kr))

        self.assertTrue(jr3.isdisjoint(ir))
        self.assertTrue(jr3.isdisjoint(jr0))
        self.assertTrue(jr3.isdisjoint(jr1))
        self.assertFalse(jr3.isdisjoint(jr3))
        self.assertFalse(jr3.isdisjoint(kr))

        self.assertFalse(kr.isdisjoint(ir))
        self.assertFalse(kr.isdisjoint(jr0))
        self.assertFalse(kr.isdisjoint(jr1))
        self.assertFalse(kr.isdisjoint(jr3))
        self.assertFalse(kr.isdisjoint(kr))

    def test_mixed_ranges_issubset(self):
        i = RangeSet(0, 10, 2)
        j = SetOf([0, 1, 2, 'a'])
        k = Any

        # Note that these ranges are verified in the test above
        (ir,) = list(i.ranges())
        jr0, jr1, jr2, jr3 = list(j.ranges())
        kr, = list(k.ranges())

        self.assertTrue(ir.issubset(ir))
        self.assertFalse(ir.issubset(jr0))
        self.assertFalse(ir.issubset(jr1))
        self.assertFalse(ir.issubset(jr3))
        self.assertTrue(ir.issubset(kr))

        self.assertTrue(jr0.issubset(ir))
        self.assertTrue(jr0.issubset(jr0))
        self.assertFalse(jr0.issubset(jr1))
        self.assertFalse(jr0.issubset(jr3))
        self.assertTrue(jr0.issubset(kr))

        self.assertFalse(jr1.issubset(ir))
        self.assertFalse(jr1.issubset(jr0))
        self.assertTrue(jr1.issubset(jr1))
        self.assertFalse(jr1.issubset(jr3))
        self.assertTrue(jr1.issubset(kr))

        self.assertFalse(jr3.issubset(ir))
        self.assertFalse(jr3.issubset(jr0))
        self.assertFalse(jr3.issubset(jr1))
        self.assertTrue(jr3.issubset(jr3))
        self.assertTrue(jr3.issubset(kr))

        self.assertFalse(kr.issubset(ir))
        self.assertFalse(kr.issubset(jr0))
        self.assertFalse(kr.issubset(jr1))
        self.assertFalse(kr.issubset(jr3))
        self.assertTrue(kr.issubset(kr))

    def test_mixed_ranges_range_difference(self):
        i = RangeSet(0, 10, 2)
        j = SetOf([0, 1, 2, 'a'])
        k = Any

        # Note that these ranges are verified in the test above
        ir, = list(i.ranges())
        jr0, jr1, jr2, jr3 = list(j.ranges())
        kr, = list(k.ranges())

        self.assertEqual(ir.range_difference(i.ranges()), [])
        self.assertEqual(ir.range_difference([jr0]), [CNR(2,10,2)])
        self.assertEqual(ir.range_difference([jr1]), [CNR(0,10,2)])
        self.assertEqual(ir.range_difference([jr2]), [CNR(0,0,0), CNR(4,10,2)])
        self.assertEqual(ir.range_difference([jr3]), [CNR(0,10,2)])
        self.assertEqual(ir.range_difference(j.ranges()), [CNR(4,10,2)])
        self.assertEqual(ir.range_difference(k.ranges()), [])

        self.assertEqual(jr0.range_difference(i.ranges()), [])
        self.assertEqual(jr0.range_difference([jr0]), [])
        self.assertEqual(jr0.range_difference([jr1]), [jr0])
        self.assertEqual(jr0.range_difference([jr2]), [jr0])
        self.assertEqual(jr0.range_difference([jr3]), [jr0])
        self.assertEqual(jr0.range_difference(j.ranges()), [])
        self.assertEqual(jr0.range_difference(k.ranges()), [])

        self.assertEqual(jr1.range_difference(i.ranges()), [jr1])
        self.assertEqual(jr1.range_difference([jr0]), [jr1])
        self.assertEqual(jr1.range_difference([jr1]), [])
        self.assertEqual(jr1.range_difference([jr2]), [jr1])
        self.assertEqual(jr1.range_difference([jr3]), [jr1])
        self.assertEqual(jr1.range_difference(j.ranges()), [])
        self.assertEqual(jr1.range_difference(k.ranges()), [])

        self.assertEqual(jr3.range_difference(i.ranges()), [jr3])
        self.assertEqual(jr3.range_difference([jr0]), [jr3])
        self.assertEqual(jr3.range_difference([jr1]), [jr3])
        self.assertEqual(jr3.range_difference([jr2]), [jr3])
        self.assertEqual(jr3.range_difference([jr3]), [])
        self.assertEqual(jr3.range_difference(j.ranges()), [])
        self.assertEqual(jr3.range_difference(k.ranges()), [])

        self.assertEqual(kr.range_difference(i.ranges()), [kr])
        self.assertEqual(kr.range_difference([jr0]), [kr])
        self.assertEqual(kr.range_difference([jr1]), [kr])
        self.assertEqual(kr.range_difference([jr2]), [kr])
        self.assertEqual(kr.range_difference([jr3]), [kr])
        self.assertEqual(kr.range_difference(j.ranges()), [kr])
        self.assertEqual(kr.range_difference(k.ranges()), [])

    def test_mixed_ranges_range_intersection(self):
        i = RangeSet(0, 10, 2)
        j = SetOf([0, 1, 2, 'a'])
        k = Any

        # Note that these ranges are verified in the test above
        ir, = list(i.ranges())
        jr0, jr1, jr2, jr3 = list(j.ranges())
        kr, = list(k.ranges())

        self.assertEqual(ir.range_intersection(i.ranges()), [ir])
        self.assertEqual(ir.range_intersection([jr0]), [jr0])
        self.assertEqual(ir.range_intersection([jr1]), [])
        self.assertEqual(ir.range_intersection([jr2]), [jr2])
        self.assertEqual(ir.range_intersection([jr3]), [])
        self.assertEqual(ir.range_intersection(j.ranges()), [jr0, jr2])
        self.assertEqual(ir.range_intersection(k.ranges()), [ir])

        self.assertEqual(jr0.range_intersection(i.ranges()), [jr0])
        self.assertEqual(jr0.range_intersection([jr0]), [jr0])
        self.assertEqual(jr0.range_intersection([jr1]), [])
        self.assertEqual(jr0.range_intersection([jr2]), [])
        self.assertEqual(jr0.range_intersection([jr3]), [])
        self.assertEqual(jr0.range_intersection(j.ranges()), [jr0])
        self.assertEqual(jr0.range_intersection(k.ranges()), [jr0])

        self.assertEqual(jr1.range_intersection(i.ranges()), [])
        self.assertEqual(jr1.range_intersection([jr0]), [])
        self.assertEqual(jr1.range_intersection([jr1]), [jr1])
        self.assertEqual(jr1.range_intersection([jr2]), [])
        self.assertEqual(jr1.range_intersection([jr3]), [])
        self.assertEqual(jr1.range_intersection(j.ranges()), [jr1])
        self.assertEqual(jr1.range_intersection(k.ranges()), [jr1])

        self.assertEqual(jr3.range_intersection(i.ranges()), [])
        self.assertEqual(jr3.range_intersection([jr0]), [])
        self.assertEqual(jr3.range_intersection([jr1]), [])
        self.assertEqual(jr3.range_intersection([jr2]), [])
        self.assertEqual(jr3.range_intersection([jr3]), [jr3])
        self.assertEqual(jr3.range_intersection(j.ranges()), [jr3])
        self.assertEqual(jr3.range_intersection(k.ranges()), [jr3])

        self.assertEqual(kr.range_intersection(i.ranges()), [ir])
        self.assertEqual(kr.range_intersection([jr0]), [jr0])
        self.assertEqual(kr.range_intersection([jr1]), [jr1])
        self.assertEqual(kr.range_intersection([jr2]), [jr2])
        self.assertEqual(kr.range_intersection([jr3]), [jr3])
        self.assertEqual(kr.range_intersection(j.ranges()), [jr0,jr1,jr2,jr3])
        self.assertEqual(kr.range_intersection(k.ranges()), [kr])


class Test_SetOf_and_RangeSet(unittest.TestCase):
    def test_RangeSet_constructor(self):
        i = RangeSet(3)
        self.assertEqual(len(i), 3)
        self.assertEqual(len(list(i.ranges())), 1)

        i = RangeSet(1,3)
        self.assertEqual(len(i), 3)
        self.assertEqual(len(list(i.ranges())), 1)

        i = RangeSet(1,3,0)
        with self.assertRaisesRegexp(
                TypeError, ".*'InfiniteSimpleRangeSet' has no len()"):
            len(i)
        self.assertEqual(len(list(i.ranges())), 1)

        with self.assertRaisesRegexp(
                TypeError, ".*'InfiniteSimpleRangeSet' has no len()"):
            len(Integers)
        self.assertEqual(len(list(Integers.ranges())), 2)

    def test_equality(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.NotI = RangeSet(4)

        m.J = SetOf([1,2,3])
        m.NotJ = SetOf([1,2,3,4])

        # Sets are equal to themselves
        self.assertEqual(m.I, m.I)
        self.assertEqual(m.J, m.J)

        # RangeSet to SetOf
        self.assertEqual(m.I, m.J)
        self.assertEqual(m.J, m.I)

        # ordering shouldn't matter
        self.assertEqual(SetOf([1,3,4,2]), SetOf({1,2,3,4}))
        self.assertEqual(SetOf({1,2,3,4}), SetOf([1,3,4,2]))

        # Inequality...
        self.assertNotEqual(m.I, m.NotI)
        self.assertNotEqual(m.NotI, m.I)
        self.assertNotEqual(m.I, m.NotJ)
        self.assertNotEqual(m.NotJ, m.I)
        self.assertNotEqual(m.J, m.NotJ)
        self.assertNotEqual(m.NotJ, m.J)
        self.assertNotEqual(m.I, RangeSet(1,3,0))
        self.assertNotEqual(RangeSet(1,3,0), m.I)

        self.assertNotEqual(SetOf([1,3,5,2]), SetOf({1,2,3,4}))
        self.assertNotEqual(SetOf({1,2,3,4}), SetOf([1,3,5,2]))

        # Sets can be compared against non-set objects
        self.assertEqual(
            RangeSet(0,4,1),
            [0,1,2,3,4]
        )
        self.assertEqual(
            RangeSet(0,4),
            [0,1,2,3,4]
        )
        self.assertEqual(
            RangeSet(4),
            [1,2,3,4]
        )

        # It can even work for non-iterable objects (that can't be cast
        # to set())
        class _NonIterable(object):
            def __init__(self):
                self.data = set({1,3,5})
            def __contains__(self, val):
                return val in self.data
            def __len__(self):
                return len(self.data)
        self.assertEqual(SetOf({1,3,5}), _NonIterable())

    def test_is_functions(self):
        i = SetOf({1,2,3})
        self.assertTrue(i.is_finite())
        self.assertFalse(i.is_ordered())

        i = SetOf([1,2,3])
        self.assertTrue(i.is_finite())
        self.assertTrue(i.is_ordered())

        i = SetOf((1,2,3))
        self.assertTrue(i.is_finite())
        self.assertTrue(i.is_ordered())

        i = RangeSet(3)
        self.assertTrue(i.is_finite())
        self.assertTrue(i.is_ordered())

        i = RangeSet(1,3)
        self.assertTrue(i.is_finite())
        self.assertTrue(i.is_ordered())

        i = RangeSet(1,3,0)
        self.assertFalse(i.is_finite())
        self.assertFalse(i.is_ordered())

    def test_pprint(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.NotI = RangeSet(1,3,0)
        m.J = SetOf([1,2,3])

        buf = StringIO()
        m.pprint(ostream=buf)
        self.assertEqual(buf.getvalue().strip(), """
2 RangeSet Declarations
    I : Dim=0, Dimen=1, Size=3, Bounds=(1, 3)
        Key  : Finite : Members
        None :   True :   [1:3]
    NotI : Dim=0, Dimen=1, Size=Inf, Bounds=(1, 3)
        Key  : Finite : Members
        None :  False :   [1,3]

1 SetOf Declarations
    J : Dim=0, Dimen=1, Size=3, Bounds=(1, 3)
        Key  : Ordered : Members
        None :    True : [1, 2, 3]

3 Declarations: I NotI J""".strip())

    def test_isdisjoint(self):
        i = SetOf({1,2,3})
        self.assertTrue(i.isdisjoint({4,5,6}))
        self.assertFalse(i.isdisjoint({3,4,5,6}))

        self.assertTrue(i.isdisjoint(SetOf({4,5,6})))
        self.assertFalse(i.isdisjoint(SetOf({3,4,5,6})))

        self.assertTrue(i.isdisjoint(RangeSet(4,6,0)))
        self.assertFalse(i.isdisjoint(RangeSet(3,6,0)))

        self.assertTrue(RangeSet(4,6,0).isdisjoint(i))
        self.assertFalse(RangeSet(3,6,0).isdisjoint(i))

        # It can even work for non-iterable objects (that can't be cast
        # to set())
        class _NonIterable(object):
            def __init__(self):
                self.data = set({1,3,5})
            def __contains__(self, val):
                return val in self.data
            def __len__(self):
                return len(self.data)
        self.assertTrue(SetOf({2,4}).isdisjoint(_NonIterable()))
        self.assertFalse(SetOf({2,3,4}).isdisjoint(_NonIterable()))

    def test_issubset(self):
        i = SetOf({1,2,3})
        self.assertTrue(i.issubset({1,2,3,4}))
        self.assertFalse(i.issubset({3,4,5,6}))

        self.assertTrue(i.issubset(SetOf({1,2,3,4})))
        self.assertFalse(i.issubset(SetOf({3,4,5,6})))

        self.assertTrue(i.issubset(RangeSet(1,4,0)))
        self.assertFalse(i.issubset(RangeSet(3,6,0)))

        self.assertTrue(RangeSet(1,3,0).issubset(RangeSet(0,100,0)))
        self.assertFalse(RangeSet(1,3,0).issubset(i))
        self.assertFalse(RangeSet(3,6,0).issubset(i))

        # It can even work for non-iterable objects (that can't be cast
        # to set())
        class _NonIterable(object):
            def __init__(self):
                self.data = set({1,3,5})
            def __contains__(self, val):
                return val in self.data
            def __len__(self):
                return len(self.data)
        self.assertTrue(SetOf({1,5}).issubset(_NonIterable()))
        self.assertFalse(SetOf({1,3,4}).issubset(_NonIterable()))

    def test_issuperset(self):
        i = SetOf({1,2,3})
        self.assertTrue(i.issuperset({1,2}))
        self.assertFalse(i.issuperset({3,4,5,6}))

        self.assertTrue(i.issuperset(SetOf({1,2})))
        self.assertFalse(i.issuperset(SetOf({3,4,5,6})))

        self.assertFalse(i.issuperset(RangeSet(1,3,0)))
        self.assertFalse(i.issuperset(RangeSet(3,6,0)))

        self.assertTrue(RangeSet(1,3,0).issuperset(RangeSet(1,2,0)))
        self.assertTrue(RangeSet(1,3,0).issuperset(i))
        self.assertFalse(RangeSet(3,6,0).issuperset(i))

        # It can even work for non-iterable objects (that can't be cast
        # to set())
        class _NonIterable(object):
            def __init__(self):
                self.data = set({1,3,5})
            def __contains__(self, val):
                return val in self.data
            def __len__(self):
                return len(self.data)
        # self.assertFalse(SetOf({1,5}).issuperset(_NonIterable()))
        # self.assertTrue(SetOf({1,3,4,5}).issuperset(_NonIterable()))

    def test_unordered_setof(self):
        i = SetOf({1,3,2,0})

        self.assertTrue(i.is_finite())
        self.assertFalse(i.is_ordered())

        self.assertEqual(i.ordered(), (0,1,2,3))
        self.assertEqual(i.sorted(), (0,1,2,3))
        self.assertEqual( tuple(reversed(i)),
                          tuple(reversed(list(i))) )

    def test_ordered_setof(self):
        i = SetOf([1,3,2,0])

        self.assertTrue(i.is_finite())
        self.assertTrue(i.is_ordered())

        self.assertEqual(i.ordered(), (1,3,2,0))
        self.assertEqual(i.sorted(), (0,1,2,3))
        self.assertEqual(tuple(reversed(i)), (0,2,3,1))

        self.assertEqual(i[2], 3)
        self.assertEqual(i[-1], 0)
        with self.assertRaisesRegexp(
                IndexError, "Valid index values for sets are 1 .. len\(set\) "
                "or -1 .. -len\(set\)"):
            i[0]
        with self.assertRaisesRegexp(
                IndexError, "Cannot index a Set past the last element"):
            i[5]
        with self.assertRaisesRegexp(
                IndexError, "Cannot index a Set before the first element"):
            i[-5]

        self.assertEqual(i.ord(3), 2)
        with self.assertRaisesRegexp(ValueError, "5 is not in list"):
            i.ord(5)

        self.assertEqual(i.first(), 1)
        self.assertEqual(i.last(), 0)

        self.assertEqual(i.next(3), 2)
        self.assertEqual(i.prev(2), 3)
        self.assertEqual(i.nextw(3), 2)
        self.assertEqual(i.prevw(2), 3)
        self.assertEqual(i.next(3,2), 0)
        self.assertEqual(i.prev(2,2), 1)
        self.assertEqual(i.nextw(3,2), 0)
        self.assertEqual(i.prevw(2,2), 1)

        with self.assertRaisesRegexp(
                IndexError, "Cannot advance past the end of the Set"):
            i.next(0)
        with self.assertRaisesRegexp(
                IndexError, "Cannot advance before the beginning of the Set"):
            i.prev(1)
        self.assertEqual(i.nextw(0), 1)
        self.assertEqual(i.prevw(1), 0)
        with self.assertRaisesRegexp(
                IndexError, "Cannot advance past the end of the Set"):
            i.next(0,2)
        with self.assertRaisesRegexp(
                IndexError, "Cannot advance before the beginning of the Set"):
            i.prev(1,2)
        self.assertEqual(i.nextw(0,2), 3)
        self.assertEqual(i.prevw(1,2), 2)

        i = SetOf([1, None, 'a'])

        self.assertTrue(i.is_finite())
        self.assertTrue(i.is_ordered())

        self.assertEqual(i.ordered(), (1,None,'a'))
        self.assertEqual(i.sorted(), (None,1,'a'))
        self.assertEqual(tuple(reversed(i)), ('a',None,1))


    def test_ranges(self):
        i = SetOf([1,3,2,0])
        r = list(i.ranges())
        self.assertEqual(len(r), 4)
        for idx, x in enumerate(r):
            self.assertIsInstance(x, CNR)
            self.assertTrue(x.is_finite())
            self.assertEqual(x.start, i[idx+1])
            self.assertEqual(x.end, i[idx+1])
            self.assertEqual(x.step, 0)

    def test_bounds(self):
        self.assertEqual(SetOf([1,3,2,0]).bounds(), (0,3))
        self.assertEqual(SetOf([1,3.0,2,0]).bounds(), (0,3.0))
        self.assertEqual(SetOf([None,1,'a']).bounds(), (None,None))
        self.assertEqual(SetOf(['apple','cat','bear']).bounds(),
                         ('apple','cat'))

        self.assertEqual(
            RangeSet(ranges=(CNR(0,10,2),CNR(3,20,2))).bounds(),
            (0,19)
        )
        self.assertEqual(
            RangeSet(ranges=(CNR(None,None,0),CNR(0,10,2))).bounds(),
            (None,None)
        )
        self.assertEqual(
            RangeSet(ranges=(CNR(100,None,-2),CNR(0,10,2))).bounds(),
            (None,100)
        )
        self.assertEqual(
            RangeSet(ranges=(CNR(-10,None,2),CNR(0,10,2))).bounds(),
            (-10,None)
        )
        self.assertEqual(
            RangeSet(ranges=(CNR(0,10,2),CNR(None,None,0))).bounds(),
            (None,None)
        )
        self.assertEqual(
            RangeSet(ranges=(CNR(0,10,2),CNR(100,None,-2))).bounds(),
            (None,100)
        )
        self.assertEqual(
            RangeSet(ranges=(CNR(0,10,2),CNR(-10,None,2))).bounds(),
            (-10,None)
        )

    def test_dimen(self):
        self.assertEqual(SetOf([]).dimen, None)
        self.assertEqual(SetOf([1,2,3]).dimen, 1)
        self.assertEqual(SetOf([(1,2),(2,3),(4,5)]).dimen, 2)
        self.assertEqual(SetOf([1,(2,3)]).dimen, None)

        a = [1,2,3]
        SetOf_a = SetOf(a)
        self.assertEqual(SetOf_a.dimen, 1)
        a.append((1,2))
        self.assertEqual(SetOf_a.dimen, None)


    def test_rangeset_iter(self):
        i = RangeSet(0,10,2)
        self.assertEqual(tuple(i), (0,2,4,6,8,10))

        i = RangeSet(ranges=(CNR(0,5,2),CNR(6,10,2)))
        self.assertEqual(tuple(i), (0,2,4,6,8,10))

        i = RangeSet(ranges=(CNR(0,10,2),CNR(0,10,2)))
        self.assertEqual(tuple(i), (0,2,4,6,8,10))

        i = RangeSet(ranges=(CNR(0,10,2),CNR(10,0,-2)))
        self.assertEqual(tuple(i), (0,2,4,6,8,10))

        i = RangeSet(ranges=(CNR(0,10,2),CNR(9,0,-2)))
        self.assertEqual(tuple(i), (0,1,2,3,4,5,6,7,8,9,10))

        i = RangeSet(ranges=(CNR(0,10,2),CNR(1,10,2)))
        self.assertEqual(tuple(i), tuple(range(11)))

        i = RangeSet(ranges=(CNR(0,30,10),CNR(12,14,1)))
        self.assertEqual(tuple(i), (0,10,12,13,14,20,30))

        i = RangeSet(ranges=(CNR(0,0,0),CNR(3,3,0),CNR(2,2,0)))
        self.assertEqual(tuple(i), (0,2,3))

