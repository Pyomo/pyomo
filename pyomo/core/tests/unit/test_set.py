#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import copy
import pickle
from six import StringIO

import pyutilib.th as unittest

import pyomo.core.base.set as SetModule
from pyomo.core.base.set import (
    _NumericRange as NR, _NonNumericRange as NNR, _AnyRange, _AnySet,
    Any, Reals, NonNegativeReals, Integers, PositiveIntegers,
    NegativeIntegers, PositiveReals,
    RangeSet, Set, SetOf,
    _FiniteRangeSetData, _InfiniteRangeSetData,
    _SetUnion_InfiniteSet, _SetUnion_FiniteSet, _SetUnion_OrderedSet,
    _SetIntersection_InfiniteSet, _SetIntersection_FiniteSet,
    _SetIntersection_OrderedSet,
    _SetDifference_InfiniteSet, _SetDifference_FiniteSet,
    _SetDifference_OrderedSet,
    _SetSymmetricDifference_InfiniteSet, _SetSymmetricDifference_FiniteSet,
    _SetSymmetricDifference_OrderedSet,
    _SetProduct_InfiniteSet, _SetProduct_FiniteSet,
    _SetProduct_OrderedSet,
    Initializer, _ConstantInitializer, _ItemInitializer, _ScalarCallInitializer,
    _IndexedCallInitializer,
    SetInitializer, _SetIntersectInitializer, RangeSetInitializer,
)
from pyomo.environ import ConcreteModel, Var

try:
    import numpy as np
    numpy_available = True
except ImportError:
    numpy_available = False


class TestNumericRange(unittest.TestCase):
    def test_init(self):
        a = NR(None, None, 0)
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

        with self.assertRaisesRegexp(
                ValueError, '.*start, end ordering incompatible with step'):
            NR(0, -1, 1)

        with self.assertRaisesRegexp(
                ValueError, '.*start, end ordering incompatible with step'):
            NR(0, 1, -2)

    def test_str(self):
        self.assertEqual(str(NR(1, 10, 0)), "[1,10]")
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

    def test_lcm(self):
        self.assertEqual(
            NR(None,None,0)._lcm((NR(0,1,0),)),
            0
        )
        self.assertEqual(
            NR(None,None,0)._lcm((NR(0,0,0),)),
            1
        )
        self.assertEqual(
            NR(0,None,3)._lcm((NR(0,None,1),)),
            3
        )
        self.assertEqual(
            NR(0,None,3)._lcm((NR(0,None,0),)),
            3
        )
        self.assertEqual(
            NR(0,None,0)._lcm((NR(0,None,1),)),
            1
        )
        self.assertEqual(
            NR(0,None,3)._lcm((NR(0,None,2),)),
            6
        )
        self.assertEqual(
            NR(0,None,3)._lcm((NR(0,None,2),NR(0,None,5))),
            30
        )
        self.assertEqual(
            NR(0,None,3)._lcm((NR(0,None,2),NR(0,None,10))),
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

    def test_pickle(self):
        a = NR(0,100,5)
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
        self.assertEqual(a, a)
        self.assertEqual(a, b)

        c = NR(None, None, 0)
        self.assertFalse(a.issubset(c))
        self.assertTrue(c.issubset(b))
        self.assertNotEqual(a, c)
        self.assertNotEqual(c, a)

    def test_contains(self):
        a = _AnyRange()
        self.assertIn(None, a)
        self.assertIn(0, a)
        self.assertIn('a', a)

    def test_range_difference(self):
        self.assertEqual(
            _AnyRange().range_difference([NR(0,None,1)]),
            [_AnyRange()]
        )
        self.assertEqual(
            NR(0,None,1).range_difference([_AnyRange()]),
            []
        )

    def test_range_intersection(self):
        self.assertEqual(
            _AnyRange().range_intersection([NR(0,None,1)]),
            [NR(0,None,1)]
        )
        self.assertEqual(
            NR(0,None,1).range_intersection([_AnyRange()]),
            [NR(0,None,1)]
        )
        self.assertEqual(
            NR(0,None,-1).range_intersection([_AnyRange()]),
            [NR(0,None,-1)]
        )

    def test_info_methods(self):
        a = _AnyRange()
        self.assertFalse(a.is_discrete())
        self.assertFalse(a.is_finite())

    def test_pickle(self):
        a = _AnyRange()
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
        self.assertTrue(a.is_discrete())
        self.assertTrue(a.is_finite())

    def test_pickle(self):
        a = NNR('a')
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a,b)
        self.assertEqual(a,b)


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

        Reals2 = RangeSet(ranges=(NR(None,None,0),))
        self.assertTrue(Reals.issubset(Reals2))
        self.assertTrue(Reals.issuperset(Reals2))
        self.assertFalse(Reals.isdisjoint(Reals2))

        Integers2 = RangeSet(ranges=(NR(0,None,-1), NR(0,None,1)))
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
            RangeSet(ranges=(NR(None,None,0),))
        )
        self.assertEqual(
            Integers,
            RangeSet(ranges=(NR(0,None,-1), NR(0,None,1)))
        )

        self.assertNotEqual(Integers, Reals)
        self.assertNotEqual(Reals, Integers)
        self.assertNotEqual(Reals, Any)
        self.assertNotEqual(Any, Reals)

        # For equality, ensure that the ranges can be in any order
        self.assertEqual(
            RangeSet(ranges=(NR(0,None,-1), NR(0,None,1))),
            RangeSet(ranges=(NR(0,None,1), NR(0,None,-1)))
        )

        # And integer ranges can be grounded at different points
        self.assertEqual(
            RangeSet(ranges=(NR(10,None,-1), NR(10,None,1))),
            RangeSet(ranges=(NR(0,None,1), NR(0,None,-1)))
        )
        self.assertEqual(
            RangeSet(ranges=(NR(0,None,-1), NR(0,None,1))),
            RangeSet(ranges=(NR(10,None,1), NR(10,None,-1)))
        )

        # Odd positive integers and even positive integers are positive
        # integers
        self.assertEqual(
            PositiveIntegers,
            RangeSet(ranges=(NR(1,None,2), NR(2,None,2)))
        )

        # Nututally prime sets of ranges
        self.assertEqual(
            RangeSet(ranges=(NR(1,None,2), NR(2,None,2))),
            RangeSet(ranges=(
                NR(1,None,3), NR(2,None,3), NR(3,None,3)
            ))
        )

        # Nututally prime sets of ranges
        #  ...omitting one of the subranges breaks equality
        self.assertNotEqual(
            RangeSet(ranges=(NR(1,None,2), NR(2,None,2))),
            RangeSet(ranges=(
                NR(1,None,3), NR(2,None,3)
            ))
        )

        # Mututally prime sets of ranges
        #  ...changing a reference point (so redundant NR) breaks equality
        self.assertNotEqual(
            RangeSet(ranges=(NR(0,None,2), NR(0,None,2))),
            RangeSet(ranges=(
                NR(1,None,3), NR(2,None,3), NR(3,None,3)
            ))
        )

    def test_bounds(self):
        self.assertEqual(Any.bounds(), (None,None))
        self.assertEqual(Reals.bounds(), (None,None))
        self.assertEqual(PositiveReals.bounds(), (0,None))
        self.assertEqual(NegativeIntegers.bounds(), (None,-1))


class TestRangeOperations(unittest.TestCase):
    def test_mixed_ranges_isdisjoint(self):
        i = RangeSet(0,10,2)
        j = SetOf([0,1,2,'a'])
        k = Any

        ir = list(i.ranges())
        self.assertEqual(ir, [NR(0,10,2)])
        self.assertEqual(str(ir), "[[0:10:2]]")
        ir = ir[0]

        jr = list(j.ranges())
        self.assertEqual(jr, [NR(0,0,0), NR(1,1,0), NR(2,2,0), NNR('a')])
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
        self.assertEqual(ir.range_difference([jr0]), [NR(2,10,2)])
        self.assertEqual(ir.range_difference([jr1]), [NR(0,10,2)])
        self.assertEqual(ir.range_difference([jr2]), [NR(0,0,0), NR(4,10,2)])
        self.assertEqual(ir.range_difference([jr3]), [NR(0,10,2)])
        self.assertEqual(ir.range_difference(j.ranges()), [NR(4,10,2)])
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

        i = RangeSet(ranges=[NR(1,3,1)])
        self.assertEqual(len(i), 3)
        self.assertEqual(list(i.ranges()), [NR(1,3,1)])

        i = RangeSet(1,3,0)
        with self.assertRaisesRegexp(
                TypeError, ".*'InfiniteSimpleRangeSet' has no len()"):
            len(i)
        self.assertEqual(len(list(i.ranges())), 1)

        with self.assertRaisesRegexp(
                TypeError, ".*'GlobalSet' has no len()"):
            len(Integers)
        self.assertEqual(len(list(Integers.ranges())), 2)

        with self.assertRaisesRegexp(
                ValueError, "RangeSet expects 3 or fewer positional "
                "arguments \(received 4\)"):
            RangeSet(1,2,3,4)

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
        self.assertIsInstance(i, _FiniteRangeSetData)

        i = RangeSet(1,3)
        self.assertTrue(i.is_finite())
        self.assertTrue(i.is_ordered())
        self.assertIsInstance(i, _FiniteRangeSetData)

        i = RangeSet(1,3,0)
        self.assertFalse(i.is_finite())
        self.assertFalse(i.is_ordered())
        self.assertIsInstance(i, _InfiniteRangeSetData)

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
                IndexError,"valid index values for Sets are "
                "\[1 .. len\(Set\)\] or \[-1 .. -len\(Set\)\]"):
            i[0]
        with self.assertRaisesRegexp(
                IndexError, "_OrderedSetOf index out of range"):
            i[5]
        with self.assertRaisesRegexp(
                IndexError, "_OrderedSetOf index out of range"):
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

        i = SetOf((1,3,2,0))

        self.assertTrue(i.is_finite())
        self.assertTrue(i.is_ordered())

        self.assertEqual(i.ordered(), (1,3,2,0))
        self.assertEqual(i.sorted(), (0,1,2,3))
        self.assertEqual(tuple(reversed(i)), (0,2,3,1))

        self.assertEqual(i[2], 3)
        self.assertEqual(i[-1], 0)
        with self.assertRaisesRegexp(
                IndexError,"valid index values for Sets are "
                "\[1 .. len\(Set\)\] or \[-1 .. -len\(Set\)\]"):
            i[0]
        with self.assertRaisesRegexp(
                IndexError, "_OrderedSetOf index out of range"):
            i[5]
        with self.assertRaisesRegexp(
                IndexError, "_OrderedSetOf index out of range"):
            i[-5]

        self.assertEqual(i.ord(3), 2)
        with self.assertRaisesRegexp(ValueError, "x not in tuple"):
            i.ord(5)

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
            self.assertIsInstance(x, NR)
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
            RangeSet(ranges=(NR(0,10,2),NR(3,20,2))).bounds(),
            (0,19)
        )
        self.assertEqual(
            RangeSet(ranges=(NR(None,None,0),NR(0,10,2))).bounds(),
            (None,None)
        )
        self.assertEqual(
            RangeSet(ranges=(NR(100,None,-2),NR(0,10,2))).bounds(),
            (None,100)
        )
        self.assertEqual(
            RangeSet(ranges=(NR(-10,None,2),NR(0,10,2))).bounds(),
            (-10,None)
        )
        self.assertEqual(
            RangeSet(ranges=(NR(0,10,2),NR(None,None,0))).bounds(),
            (None,None)
        )
        self.assertEqual(
            RangeSet(ranges=(NR(0,10,2),NR(100,None,-2))).bounds(),
            (None,100)
        )
        self.assertEqual(
            RangeSet(ranges=(NR(0,10,2),NR(-10,None,2))).bounds(),
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

        i = RangeSet(ranges=(NR(0,5,2),NR(6,10,2)))
        self.assertEqual(tuple(i), (0,2,4,6,8,10))

        i = RangeSet(ranges=(NR(0,10,2),NR(0,10,2)))
        self.assertEqual(tuple(i), (0,2,4,6,8,10))

        i = RangeSet(ranges=(NR(0,10,2),NR(10,0,-2)))
        self.assertEqual(tuple(i), (0,2,4,6,8,10))

        i = RangeSet(ranges=(NR(0,10,2),NR(9,0,-2)))
        self.assertEqual(tuple(i), (0,1,2,3,4,5,6,7,8,9,10))

        i = RangeSet(ranges=(NR(0,10,2),NR(1,10,2)))
        self.assertEqual(tuple(i), tuple(range(11)))

        i = RangeSet(ranges=(NR(0,30,10),NR(12,14,1)))
        self.assertEqual(tuple(i), (0,10,12,13,14,20,30))

        i = RangeSet(ranges=(NR(0,0,0),NR(3,3,0),NR(2,2,0)))
        self.assertEqual(tuple(i), (0,2,3))

    def test_finite_ord_index(self):
        r = RangeSet(2,10,2)
        for i,v in enumerate([2,4,6,8,10]):
            self.assertEqual(r.ord(v), i+1)
            self.assertEqual(r[i+1], v)
        with self.assertRaisesRegexp(
                IndexError,"valid index values for Sets are "
                "\[1 .. len\(Set\)\] or \[-1 .. -len\(Set\)\]"):
            r[0]
        with self.assertRaisesRegexp(
                IndexError,"FiniteSimpleRangeSet index out of range"):
            r[10]
        with self.assertRaisesRegexp(
                ValueError,"Cannot identify position of 5 in Set"):
            r.ord(5)

        r = RangeSet(ranges=(NR(2,10,2), NR(6,12,3)))
        for i,v in enumerate([2,4,6,8,9,10,12]):
            self.assertEqual(r.ord(v), i+1)
            self.assertEqual(r[i+1], v)
        with self.assertRaisesRegexp(
                IndexError,"valid index values for Sets are "
                "\[1 .. len\(Set\)\] or \[-1 .. -len\(Set\)\]"):
            r[0]
        with self.assertRaisesRegexp(
                IndexError,"FiniteSimpleRangeSet index out of range"):
            r[10]
        with self.assertRaisesRegexp(
                ValueError,"Cannot identify position of 5 in Set"):
            r.ord(5)


class TestSetUnion(unittest.TestCase):
    def _verify_ordered_union(self, a, b):
        # Note the placement of the second "3" in the middle of the set.
        # This helps catch edge cases where we need to ensure it doesn't
        # count as part of the set membership
        if isinstance(a, SetOf):
            self.assertTrue(a.is_ordered())
            self.assertTrue(a.is_finite())
        else:
            self.assertIs(type(a), list)
        if isinstance(b, SetOf):
            self.assertTrue(b.is_ordered())
            self.assertTrue(b.is_finite())
        else:
            self.assertIs(type(b), list)

        x = a | b
        self.assertIs(type(x), _SetUnion_OrderedSet)
        self.assertTrue(x.is_finite())
        self.assertTrue(x.is_ordered())
        self.assertEqual(len(x), 5)
        self.assertEqual(list(x), [1,3,2,5,4])
        self.assertEqual(x.ordered(), (1,3,2,5,4))
        self.assertEqual(x.sorted(), (1,2,3,4,5))

        self.assertIn(1, x)
        self.assertIn(2, x)
        self.assertIn(3, x)
        self.assertIn(4, x)
        self.assertIn(5, x)
        self.assertNotIn(6, x)

        self.assertEqual(x.ord(1), 1)
        self.assertEqual(x.ord(2), 3)
        self.assertEqual(x.ord(3), 2)
        self.assertEqual(x.ord(4), 5)
        self.assertEqual(x.ord(5), 4)
        with self.assertRaisesRegexp(
                IndexError,
                "Cannot identify position of 6 in Set _SetUnion_OrderedSet"):
            x.ord(6)

        self.assertEqual(x[1], 1)
        self.assertEqual(x[2], 3)
        self.assertEqual(x[3], 2)
        self.assertEqual(x[4], 5)
        self.assertEqual(x[5], 4)
        with self.assertRaisesRegexp(
                IndexError,
                "_SetUnion_OrderedSet index out of range"):
            x[6]

        self.assertEqual(x[-1], 4)
        self.assertEqual(x[-2], 5)
        self.assertEqual(x[-3], 2)
        self.assertEqual(x[-4], 3)
        self.assertEqual(x[-5], 1)
        with self.assertRaisesRegexp(
                IndexError,
                "_SetUnion_OrderedSet index out of range"):
            x[-6]

    def test_ordered_setunion(self):
        self._verify_ordered_union(SetOf([1,3,2]), SetOf([5,3,4]))
        self._verify_ordered_union([1,3,2], SetOf([5,3,4]))
        self._verify_ordered_union(SetOf([1,3,2]), [5,3,4])


    def _verify_finite_union(self, a, b):
        # Note the placement of the second "3" in the middle of the set.
        # This helps catch edge cases where we need to ensure it doesn't
        # count as part of the set membership
        if isinstance(a, SetOf):
            if type(a._ref) is list:
                self.assertTrue(a.is_ordered())
            else:
                self.assertFalse(a.is_ordered())
            self.assertTrue(a.is_finite())
        else:
            self.assertIn(type(a), (list, set))
        if isinstance(b, SetOf):
            if type(b._ref) is list:
                self.assertTrue(b.is_ordered())
            else:
                self.assertFalse(b.is_ordered())
            self.assertTrue(b.is_finite())
        else:
            self.assertIn(type(b), (list, set))

        x = a | b
        self.assertIs(type(x), _SetUnion_FiniteSet)
        self.assertTrue(x.is_finite())
        self.assertFalse(x.is_ordered())
        self.assertEqual(len(x), 5)
        if x._sets[0].is_ordered():
            self.assertEqual(list(x)[:3], [1,3,2])
        if x._sets[1].is_ordered():
            self.assertEqual(list(x)[-2:], [5,4])
        self.assertEqual(sorted(list(x)), [1,2,3,4,5])
        self.assertEqual(x.ordered(), (1,2,3,4,5))
        self.assertEqual(x.sorted(), (1,2,3,4,5))

        self.assertIn(1, x)
        self.assertIn(2, x)
        self.assertIn(3, x)
        self.assertIn(4, x)
        self.assertIn(5, x)
        self.assertNotIn(6, x)

        # THe ranges should at least filter out the duplicates
        self.assertEqual(
            len(list(x._sets[0].ranges()) + list(x._sets[1].ranges())), 6)
        self.assertEqual(len(list(x.ranges())), 5)

    def test_finite_setunion(self):
        self._verify_finite_union(SetOf({1,3,2}), SetOf({5,3,4}))
        self._verify_finite_union([1,3,2], SetOf({5,3,4}))
        self._verify_finite_union(SetOf({1,3,2}), [5,3,4])
        self._verify_finite_union({1,3,2}, SetOf([5,3,4]))
        self._verify_finite_union(SetOf([1,3,2]), {5,3,4})


    def _verify_infinite_union(self, a, b):
        # Note the placement of the second "3" in the middle of the set.
        # This helps catch edge cases where we need to ensure it doesn't
        # count as part of the set membership
        if isinstance(a, RangeSet):
            self.assertFalse(a.is_ordered())
            self.assertFalse(a.is_finite())
        else:
            self.assertIn(type(a), (list, set))
        if isinstance(b, RangeSet):
            self.assertFalse(b.is_ordered())
            self.assertFalse(b.is_finite())
        else:
            self.assertIn(type(b), (list, set))

        x = a | b
        self.assertIs(type(x), _SetUnion_InfiniteSet)
        self.assertFalse(x.is_finite())
        self.assertFalse(x.is_ordered())

        self.assertIn(1, x)
        self.assertIn(2, x)
        self.assertIn(3, x)
        self.assertIn(4, x)
        self.assertIn(5, x)
        self.assertNotIn(6, x)

        self.assertEqual(list(x.ranges()),
                         list(x._sets[0].ranges()) + list(x._sets[1].ranges()))

    def test_infinite_setunion(self):
        self._verify_infinite_union(RangeSet(1,3,0), RangeSet(3,5,0))
        self._verify_infinite_union([1,3,2], RangeSet(3,5,0))
        self._verify_infinite_union(RangeSet(1,3,0), [5,3,4])
        self._verify_infinite_union({1,3,2}, RangeSet(3,5,0))
        self._verify_infinite_union(RangeSet(1,3,0), {5,3,4})


class TestSetIntersection(unittest.TestCase):
    def _verify_ordered_intersection(self, a, b):
        if isinstance(a, (Set, SetOf, RangeSet)):
            a_ordered = a.is_ordered()
        else:
            a_ordered = type(a) is list
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_ordered = b.is_ordered()
        else:
            b_ordered = type(b) is list
        self.assertTrue(a_ordered or b_ordered)

        if a_ordered:
            ref = (3,2,5)
        else:
            ref = (2,3,5)

        x = a & b
        self.assertIs(type(x), _SetIntersection_OrderedSet)
        self.assertTrue(x.is_finite())
        self.assertTrue(x.is_ordered())
        self.assertEqual(len(x), 3)
        self.assertEqual(list(x), list(ref))
        self.assertEqual(x.ordered(), tuple(ref))
        self.assertEqual(x.sorted(), (2,3,5))

        self.assertNotIn(1, x)
        self.assertIn(2, x)
        self.assertIn(3, x)
        self.assertNotIn(4, x)
        self.assertIn(5, x)
        self.assertNotIn(6, x)

        self.assertEqual(x.ord(2), ref.index(2)+1)
        self.assertEqual(x.ord(3), ref.index(3)+1)
        self.assertEqual(x.ord(5), 3)
        with self.assertRaisesRegexp(
                IndexError, "Cannot identify position of 6 in Set "
                "_SetIntersection_OrderedSet"):
            x.ord(6)

        self.assertEqual(x[1], ref[0])
        self.assertEqual(x[2], ref[1])
        self.assertEqual(x[3], 5)
        with self.assertRaisesRegexp(
                IndexError,
                "_SetIntersection_OrderedSet index out of range"):
            x[4]

        self.assertEqual(x[-1], 5)
        self.assertEqual(x[-2], ref[-2])
        self.assertEqual(x[-3], ref[-3])
        with self.assertRaisesRegexp(
                IndexError,
                "_SetIntersection_OrderedSet index out of range"):
            x[-4]

    def test_ordered_setintersection(self):
        self._verify_ordered_intersection(SetOf([1,3,2,5]), SetOf([0,2,3,4,5]))
        self._verify_ordered_intersection(SetOf([1,3,2,5]), SetOf({0,2,3,4,5}))
        self._verify_ordered_intersection(SetOf({1,3,2,5}), SetOf([0,2,3,4,5]))
        self._verify_ordered_intersection(SetOf([1,3,2,5]), [0,2,3,4,5])
        self._verify_ordered_intersection(SetOf([1,3,2,5]), {0,2,3,4,5})
        self._verify_ordered_intersection([1,3,2,5], SetOf([0,2,3,4,5]))
        self._verify_ordered_intersection({1,3,2,5}, SetOf([0,2,3,4,5]))


    def _verify_finite_intersection(self, a, b):
        # Note the placement of the second "3" in the middle of the set.
        # This helps catch edge cases where we need to ensure it doesn't
        # count as part of the set membership
        if isinstance(a, (Set, SetOf, RangeSet)):
            a_finite = a.is_finite()
        else:
            a_finite = True
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_finite = b.is_finite()
        else:
            b_finite = True
        self.assertTrue(a_finite or b_finite)

        x = a & b
        self.assertIs(type(x), _SetIntersection_FiniteSet)
        self.assertTrue(x.is_finite())
        self.assertFalse(x.is_ordered())
        self.assertEqual(len(x), 3)
        if x._sets[0].is_ordered():
            self.assertEqual(list(x)[:3], [3,2,5])
        self.assertEqual(sorted(list(x)), [2,3,5])
        self.assertEqual(x.ordered(), (2,3,5))
        self.assertEqual(x.sorted(), (2,3,5))

        self.assertNotIn(1, x)
        self.assertIn(2, x)
        self.assertIn(3, x)
        self.assertNotIn(4, x)
        self.assertIn(5, x)
        self.assertNotIn(6, x)

        # The ranges should at least filter out the duplicates
        self.assertEqual(
            len(list(x._sets[0].ranges()) + list(x._sets[1].ranges())), 9)
        self.assertEqual(len(list(x.ranges())), 3)


    def test_finite_setintersection(self):
        self._verify_finite_intersection(SetOf({1,3,2,5}), SetOf({0,2,3,4,5}))
        self._verify_finite_intersection({1,3,2,5}, SetOf({0,2,3,4,5}))
        self._verify_finite_intersection(SetOf({1,3,2,5}), {0,2,3,4,5})
        self._verify_finite_intersection(
            RangeSet(ranges=(NR(-5,-1,0), NR(2,3,0), NR(5,5,0), NR(10,20,0))),
            SetOf({0,2,3,4,5}))
        self._verify_finite_intersection(
            SetOf({1,3,2,5}),
            RangeSet(ranges=(NR(2,5,0), NR(2,5,0), NR(6,6,0), NR(6,6,0),
                             NR(6,6,0))))


    def _verify_infinite_intersection(self, a, b):
        if isinstance(a, (Set, SetOf, RangeSet)):
            a_finite = a.is_finite()
        else:
            a_finite = True
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_finite = b.is_finite()
        else:
            b_finite = True
        self.assertEqual([a_finite, b_finite], [False,False])

        x = a & b
        self.assertIs(type(x), _SetIntersection_InfiniteSet)
        self.assertFalse(x.is_finite())
        self.assertFalse(x.is_ordered())

        self.assertNotIn(1, x)
        self.assertIn(2, x)
        self.assertIn(3, x)
        self.assertIn(4, x)
        self.assertNotIn(5, x)
        self.assertNotIn(6, x)

        self.assertEqual(list(x.ranges()),
                         list(RangeSet(2,4,0).ranges()))

    def test_infinite_setintersection(self):
        self._verify_infinite_intersection(RangeSet(0,4,0), RangeSet(2,6,0))

    def test_odd_intersections(self):
        # Test the intersection of an infinite discrete range with a
        # finite continuous one
        a = RangeSet(0, None, 2)
        b = RangeSet(5,10,0)
        x = a & b
        self.assertIs(type(x), _SetIntersection_OrderedSet)
        self.assertEqual(list(x), [6,8,10])

        self.assertEqual(x.ord(6), 1)
        self.assertEqual(x.ord(8), 2)
        self.assertEqual(x.ord(10), 3)

        self.assertEqual(x[1], 6)
        self.assertEqual(x[2], 8)
        self.assertEqual(x[3], 10)
        with self.assertRaisesRegexp(
                IndexError,
                "_SetIntersection_OrderedSet index out of range"):
            x[4]

        self.assertEqual(x[-3], 6)
        self.assertEqual(x[-2], 8)
        self.assertEqual(x[-1], 10)
        with self.assertRaisesRegexp(
                IndexError,
                "_SetIntersection_OrderedSet index out of range"):
            x[-4]



class TestSetDifference(unittest.TestCase):
    def _verify_ordered_difference(self, a, b):
        if isinstance(a, (Set, SetOf, RangeSet)):
            a_ordered = a.is_ordered()
        else:
            a_ordered = type(a) is list
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_ordered = b.is_ordered()
        else:
            b_ordered = type(b) is list
        self.assertTrue(a_ordered)

        x = a - b
        self.assertIs(type(x), _SetDifference_OrderedSet)
        self.assertTrue(x.is_finite())
        self.assertTrue(x.is_ordered())
        self.assertEqual(len(x), 3)
        self.assertEqual(list(x), [3,2,5])
        self.assertEqual(x.ordered(), (3,2,5))
        self.assertEqual(x.sorted(), (2,3,5))

        self.assertNotIn(0, x)
        self.assertNotIn(1, x)
        self.assertIn(2, x)
        self.assertIn(3, x)
        self.assertNotIn(4, x)
        self.assertIn(5, x)
        self.assertNotIn(6, x)

        self.assertEqual(x.ord(2), 2)
        self.assertEqual(x.ord(3), 1)
        self.assertEqual(x.ord(5), 3)
        with self.assertRaisesRegexp(
                IndexError, "Cannot identify position of 6 in Set "
                "_SetDifference_OrderedSet"):
            x.ord(6)

        self.assertEqual(x[1], 3)
        self.assertEqual(x[2], 2)
        self.assertEqual(x[3], 5)
        with self.assertRaisesRegexp(
                IndexError,
                "_SetDifference_OrderedSet index out of range"):
            x[4]

        self.assertEqual(x[-1], 5)
        self.assertEqual(x[-2], 2)
        self.assertEqual(x[-3], 3)
        with self.assertRaisesRegexp(
                IndexError,
                "_SetDifference_OrderedSet index out of range"):
            x[-4]

    def test_ordered_setdifference(self):
        self._verify_ordered_difference(SetOf([0,3,2,1,5,4]), SetOf([0,1,4]))
        self._verify_ordered_difference(SetOf([0,3,2,1,5,4]), SetOf({0,1,4}))
        self._verify_ordered_difference(SetOf([0,3,2,1,5,4]), [0,1,4])
        self._verify_ordered_difference(SetOf([0,3,2,1,5,4]), {0,1,4})
        self._verify_ordered_difference(SetOf([0,3,2,1,5,4]),
                                        RangeSet(ranges=(NR(0,1,0),NR(4,4,0))))
        self._verify_ordered_difference([0,3,2,1,5,4], SetOf([0,1,4]))
        self._verify_ordered_difference([0,3,2,1,5,4], SetOf({0,1,4}))


    def _verify_finite_difference(self, a, b):
        # Note the placement of the second "3" in the middle of the set.
        # This helps catch edge cases where we need to ensure it doesn't
        # count as part of the set membership
        if isinstance(a, (Set, SetOf, RangeSet)):
            a_finite = a.is_finite()
        else:
            a_finite = True
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_finite = b.is_finite()
        else:
            b_finite = True
        self.assertTrue(a_finite or b_finite)

        x = a - b
        self.assertIs(type(x), _SetDifference_FiniteSet)
        self.assertTrue(x.is_finite())
        self.assertFalse(x.is_ordered())
        self.assertEqual(len(x), 3)
        self.assertEqual(sorted(list(x)), [2,3,5])
        self.assertEqual(x.ordered(), (2,3,5))
        self.assertEqual(x.sorted(), (2,3,5))

        self.assertNotIn(0, x)
        self.assertNotIn(1, x)
        self.assertIn(2, x)
        self.assertIn(3, x)
        self.assertNotIn(4, x)
        self.assertIn(5, x)
        self.assertNotIn(6, x)

        # The ranges should at least filter out the duplicates
        self.assertEqual(
            len(list(x._sets[0].ranges()) + list(x._sets[1].ranges())), 9)
        self.assertEqual(len(list(x.ranges())), 3)


    def test_finite_setdifference(self):
        self._verify_finite_difference(SetOf({0,3,2,1,5,4}), SetOf({0,1,4}))
        self._verify_finite_difference(SetOf({0,3,2,1,5,4}), SetOf([0,1,4]))
        self._verify_finite_difference(SetOf({0,3,2,1,5,4}), [0,1,4])
        self._verify_finite_difference(SetOf({0,3,2,1,5,4}), {0,1,4})
        self._verify_finite_difference(
            SetOf({0,3,2,1,5,4}),
            RangeSet(ranges=(NR(0,1,0),NR(4,4,0),NR(6,10,0))))
        self._verify_finite_difference({0,3,2,1,5,4}, SetOf([0,1,4]))
        self._verify_finite_difference({0,3,2,1,5,4}, SetOf({0,1,4}))


    def test_infinite_setdifference(self):
        x = RangeSet(0,4,0) - RangeSet(2,6,0)
        self.assertIs(type(x), _SetDifference_InfiniteSet)
        self.assertFalse(x.is_finite())
        self.assertFalse(x.is_ordered())

        self.assertNotIn(-1, x)
        self.assertIn(0, x)
        self.assertIn(1, x)
        self.assertIn(1.9, x)
        self.assertNotIn(2, x)
        self.assertNotIn(6, x)
        self.assertNotIn(8, x)

        self.assertEqual(
            list(x.ranges()),
            list(RangeSet(ranges=[NR(0,2,0,(True,False))]).ranges()))


class TestSetSymmetricDifference(unittest.TestCase):
    def _verify_ordered_symdifference(self, a, b):
        if isinstance(a, (Set, SetOf, RangeSet)):
            a_ordered = a.is_ordered()
        else:
            a_ordered = type(a) is list
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_ordered = b.is_ordered()
        else:
            b_ordered = type(b) is list
        self.assertTrue(a_ordered)

        x = a ^ b
        self.assertIs(type(x), _SetSymmetricDifference_OrderedSet)
        self.assertTrue(x.is_finite())
        self.assertTrue(x.is_ordered())
        self.assertEqual(len(x), 4)
        self.assertEqual(list(x), [3,2,5,0])
        self.assertEqual(x.ordered(), (3,2,5,0))
        self.assertEqual(x.sorted(), (0,2,3,5))

        self.assertIn(0, x)
        self.assertNotIn(1, x)
        self.assertIn(2, x)
        self.assertIn(3, x)
        self.assertNotIn(4, x)
        self.assertIn(5, x)
        self.assertNotIn(6, x)

        self.assertEqual(x.ord(0), 4)
        self.assertEqual(x.ord(2), 2)
        self.assertEqual(x.ord(3), 1)
        self.assertEqual(x.ord(5), 3)
        with self.assertRaisesRegexp(
                IndexError, "Cannot identify position of 6 in Set "
                "_SetSymmetricDifference_OrderedSet"):
            x.ord(6)

        self.assertEqual(x[1], 3)
        self.assertEqual(x[2], 2)
        self.assertEqual(x[3], 5)
        self.assertEqual(x[4], 0)
        with self.assertRaisesRegexp(
                IndexError,
                "_SetSymmetricDifference_OrderedSet index out of range"):
            x[5]

        self.assertEqual(x[-1], 0)
        self.assertEqual(x[-2], 5)
        self.assertEqual(x[-3], 2)
        self.assertEqual(x[-4], 3)
        with self.assertRaisesRegexp(
                IndexError,
                "_SetSymmetricDifference_OrderedSet index out of range"):
            x[-5]

    def test_ordered_setsymmetricdifference(self):
        self._verify_ordered_symdifference(SetOf([3,2,1,5,4]), SetOf([0,1,4]))
        self._verify_ordered_symdifference(SetOf([3,2,1,5,4]), [0,1,4])
        self._verify_ordered_symdifference([3,2,1,5,4], SetOf([0,1,4]))

    def _verify_finite_symdifference(self, a, b):
        # Note the placement of the second "3" in the middle of the set.
        # This helps catch edge cases where we need to ensure it doesn't
        # count as part of the set membership
        if isinstance(a, (Set, SetOf, RangeSet)):
            a_finite = a.is_finite()
        else:
            a_finite = True
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_finite = b.is_finite()
        else:
            b_finite = True
        self.assertTrue(a_finite or b_finite)

        x = a ^ b
        self.assertIs(type(x), _SetSymmetricDifference_FiniteSet)
        self.assertTrue(x.is_finite())
        self.assertFalse(x.is_ordered())
        self.assertEqual(len(x), 4)
        self.assertEqual(sorted(list(x)), [0,2,3,5])
        self.assertEqual(x.ordered(), (0,2,3,5))
        self.assertEqual(x.sorted(), (0,2,3,5))

        self.assertIn(0, x)
        self.assertNotIn(1, x)
        self.assertIn(2, x)
        self.assertIn(3, x)
        self.assertNotIn(4, x)
        self.assertIn(5, x)
        self.assertNotIn(6, x)

        # The ranges should at least filter out the duplicates
        self.assertEqual(
            len(list(x._sets[0].ranges()) + list(x._sets[1].ranges())), 8)
        self.assertEqual(len(list(x.ranges())), 4)


    def test_finite_setsymmetricdifference(self):
        self._verify_finite_symdifference(SetOf([3,2,1,5,4]), SetOf({0,1,4}))
        self._verify_finite_symdifference(SetOf([3,2,1,5,4]), {0,1,4})
        self._verify_finite_symdifference([3,2,1,5,4], SetOf({0,1,4}))
        self._verify_finite_symdifference(SetOf({3,2,1,5,4}), SetOf({0,1,4}))
        self._verify_finite_symdifference(SetOf({3,2,1,5,4}), SetOf([0,1,4]))
        self._verify_finite_symdifference(SetOf({3,2,1,5,4}), [0,1,4])
        self._verify_finite_symdifference(SetOf({3,2,1,5,4}), {0,1,4})
        self._verify_finite_symdifference({3,2,1,5,4}, SetOf([0,1,4]))
        self._verify_finite_symdifference({3,2,1,5,4}, SetOf({0,1,4}))


    def test_infinite_setdifference(self):
        x = RangeSet(0,4,0) ^ RangeSet(2,6,0)
        self.assertIs(type(x), _SetSymmetricDifference_InfiniteSet)
        self.assertFalse(x.is_finite())
        self.assertFalse(x.is_ordered())

        self.assertNotIn(-1, x)
        self.assertIn(0, x)
        self.assertIn(1, x)
        self.assertIn(1.9, x)
        self.assertNotIn(2, x)
        self.assertNotIn(4, x)
        self.assertIn(4.1, x)
        self.assertIn(6, x)

        self.assertEqual(
            sorted(str(_) for _ in x.ranges()),
            sorted(str(_) for _ in [
                NR(0,2,0,(True,False)), NR(4,6,0,(False, True))
            ]))

        x = SetOf([3,2,1,5,4]) ^ RangeSet(3,6,0)
        self.assertIs(type(x), _SetSymmetricDifference_InfiniteSet)
        self.assertFalse(x.is_finite())
        self.assertFalse(x.is_ordered())

        self.assertNotIn(-1, x)
        self.assertIn(1, x)
        self.assertIn(2, x)
        self.assertNotIn(3, x)
        self.assertNotIn(4, x)
        self.assertNotIn(5, x)
        self.assertIn(4.1, x)
        self.assertIn(5.1, x)
        self.assertIn(6, x)

        self.assertEqual(
            sorted(str(_) for _ in x.ranges()),
            sorted(str(_) for _ in [
                NR(1,1,0),
                NR(2,2,0),
                NR(3,4,0,(False,False)),
                NR(4,5,0,(False,False)),
                NR(5,6,0,(False, True))
            ]))

        x = RangeSet(3,6,0) ^ SetOf([3,2,1,5,4])
        self.assertIs(type(x), _SetSymmetricDifference_InfiniteSet)
        self.assertFalse(x.is_finite())
        self.assertFalse(x.is_ordered())

        self.assertNotIn(-1, x)
        self.assertIn(1, x)
        self.assertIn(2, x)
        self.assertNotIn(3, x)
        self.assertNotIn(4, x)
        self.assertNotIn(5, x)
        self.assertIn(4.1, x)
        self.assertIn(5.1, x)
        self.assertIn(6, x)

        self.assertEqual(
            sorted(str(_) for _ in x.ranges()),
            sorted(str(_) for _ in [
                NR(1,1,0),
                NR(2,2,0),
                NR(3,4,0,(False,False)),
                NR(4,5,0,(False,False)),
                NR(5,6,0,(False, True))
            ]))

class TestSetProduct(unittest.TestCase):
    def test_cutPointGenerator(self):
        CG = _SetProduct_InfiniteSet._cutPointGenerator
        i = Any
        j = SetOf([(1,1),(1,2),(2,1),(2,2)])

        test = list(tuple(_) for _ in CG((i,i), 3))
        ref =  [(0,0,3),(0,1,3),(0,2,3),(0,3,3)]
        self.assertEqual(test, ref)

        test = list(tuple(_) for _ in CG((i,i,i), 3))
        ref =  [
            (0,0,0,3),(0,0,1,3),(0,0,2,3),(0,0,3,3),
            (0,1,1,3),(0,1,2,3),(0,1,3,3),
            (0,2,2,3),(0,2,3,3),
            (0,3,3,3)
        ]
        self.assertEqual(test, ref)

        test = list(tuple(_) for _ in CG((i,j,i), 5))
        ref =  [
            (0,0,2,5),(0,1,3,5),(0,2,4,5),(0,3,5,5),
        ]
        self.assertEqual(test, ref)

    def test_flatten_cross_product(self):
        a = SetOf([1])
        b = SetOf([1])
        c = SetOf([1])
        d = SetOf([1])

        x = a * b
        self.assertEqual(len(x._sets), 2)
        self.assertEqual(list(x.flatten_cross_product()), [a,b])
        x = a * b * c
        self.assertEqual(len(x._sets), 2)
        self.assertEqual(list(x.flatten_cross_product()), [a,b,c])
        x = (a * b) * (c * d)
        self.assertEqual(len(x._sets), 2)
        self.assertEqual(list(x.flatten_cross_product()), [a,b,c,d])

    def test_infinite_setproduct(self):
        x = PositiveIntegers * SetOf([2,3,5,7])
        self.assertFalse(x.is_finite())
        self.assertFalse(x.is_ordered())
        self.assertIn((1,2), x)
        self.assertNotIn((0,2), x)
        self.assertNotIn((1,1), x)
        self.assertNotIn(('a',2), x)
        self.assertNotIn((2,'a'), x)

        x = SetOf([2,3,5,7]) * PositiveIntegers
        self.assertFalse(x.is_finite())
        self.assertFalse(x.is_ordered())
        self.assertIn((3,2), x)
        self.assertNotIn((1,2), x)
        self.assertNotIn((2,0), x)
        self.assertNotIn(('a',2), x)
        self.assertNotIn((2,'a'), x)

        x = PositiveIntegers * PositiveIntegers
        self.assertFalse(x.is_finite())
        self.assertFalse(x.is_ordered())
        self.assertIn((3,2), x)
        self.assertNotIn((0,2), x)
        self.assertNotIn((2,0), x)
        self.assertNotIn(('a',2), x)
        self.assertNotIn((2,'a'), x)

    def _verify_finite_product(self, a, b):
        if isinstance(a, (Set, SetOf, RangeSet)):
            a_ordered = a.is_ordered()
        else:
            a_ordered = type(a) is list
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_ordered = b.is_ordered()
        else:
            b_ordered = type(b) is list
        self.assertFalse(a_ordered and b_ordered)

        x = a * b

        self.assertIs(type(x), _SetProduct_FiniteSet)
        self.assertTrue(x.is_finite())
        self.assertFalse(x.is_ordered())
        self.assertEqual(len(x), 6)
        self.assertEqual(
            sorted(list(x)), [(1,5),(1,6),(2,5),(2,6),(3,5),(3,6)])
        self.assertEqual(x.ordered(), ((1,5),(1,6),(2,5),(2,6),(3,5),(3,6)))
        self.assertEqual(x.sorted(), ((1,5),(1,6),(2,5),(2,6),(3,5),(3,6)))

        self.assertNotIn(1, x)
        self.assertIn((1,5), x)
        self.assertIn(((1,),5), x)
        self.assertNotIn((1,2,3), x)
        self.assertNotIn((2,4), x)

    def test_finite_setproduct(self):
        self._verify_finite_product(SetOf({3,1,2}), SetOf({6,5}))
        self._verify_finite_product(SetOf({3,1,2}), SetOf([6,5]))
        self._verify_finite_product(SetOf([3,1,2]), SetOf({6,5}))
        self._verify_finite_product(SetOf([3,1,2]), {6,5})
        self._verify_finite_product({3,1,2}, SetOf([6,5]))
        self._verify_finite_product(SetOf({3,1,2}), [6,5])
        self._verify_finite_product([3,1,2], SetOf({6,5}))

    def _verify_ordered_product(self, a, b):
        if isinstance(a, (Set, SetOf, RangeSet)):
            a_ordered = a.is_ordered()
        else:
            a_ordered = type(a) is list
        self.assertTrue(a_ordered)
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_ordered = b.is_ordered()
        else:
            b_ordered = type(b) is list
        self.assertTrue(b_ordered)

        x = a * b

        self.assertIs(type(x), _SetProduct_OrderedSet)
        self.assertTrue(x.is_finite())
        self.assertTrue(x.is_ordered())
        self.assertEqual(len(x), 6)
        self.assertEqual(list(x), [(3,6),(3,5),(1,6),(1,5),(2,6),(2,5)])
        self.assertEqual(x.ordered(), ((3,6),(3,5),(1,6),(1,5),(2,6),(2,5)))
        self.assertEqual(x.sorted(), ((1,5),(1,6),(2,5),(2,6),(3,5),(3,6)))

        self.assertNotIn(1, x)
        self.assertIn((1,5), x)
        self.assertIn(((1,),5), x)
        self.assertNotIn((1,2,3), x)
        self.assertNotIn((2,4), x)

        self.assertEqual(x.ord((3,6)), 1)
        self.assertEqual(x.ord((3,5)), 2)
        self.assertEqual(x.ord((1,6)), 3)
        self.assertEqual(x.ord((1,5)), 4)
        self.assertEqual(x.ord((2,6)), 5)
        self.assertEqual(x.ord((2,5)), 6)
        with self.assertRaisesRegexp(
                IndexError, "Cannot identify position of \(3, 4\) in Set "
                "_SetProduct_OrderedSet"):
            x.ord((3,4))

        self.assertEqual(x[1], (3,6))
        self.assertEqual(x[2], (3,5))
        self.assertEqual(x[3], (1,6))
        self.assertEqual(x[4], (1,5))
        self.assertEqual(x[5], (2,6))
        self.assertEqual(x[6], (2,5))
        with self.assertRaisesRegexp(
                IndexError,
                "_SetProduct_OrderedSet index out of range"):
            x[7]

        self.assertEqual(x[-6], (3,6))
        self.assertEqual(x[-5], (3,5))
        self.assertEqual(x[-4], (1,6))
        self.assertEqual(x[-3], (1,5))
        self.assertEqual(x[-2], (2,6))
        self.assertEqual(x[-1], (2,5))
        with self.assertRaisesRegexp(
                IndexError,
                "_SetProduct_OrderedSet index out of range"):
            x[-7]

    def test_ordered_setproduct(self):
        self._verify_ordered_product(SetOf([3,1,2]), SetOf([6,5]))
        self._verify_ordered_product(SetOf([3,1,2]), [6,5])
        self._verify_ordered_product([3,1,2], SetOf([6,5]))

    def test_ordered_multidim_setproduct(self):
        x = SetOf([(1,2),(3,4)]) * SetOf([(5,6),(7,8)])
        self.assertEqual(x.dimen, 4)
        try:
            origFlattenCross = SetModule.FLATTEN_CROSS_PRODUCT

            SetModule.FLATTEN_CROSS_PRODUCT = True
            ref = [(1,2,5,6), (1,2,7,8), (3,4,5,6), (3,4,7,8)]
            self.assertEqual(list(x), ref)

            SetModule.FLATTEN_CROSS_PRODUCT = False
            ref = [((1,2),(5,6)), ((1,2),(7,8)), ((3,4),(5,6)), ((3,4),(7,8))]
            self.assertEqual(list(x), ref)
        finally:
            SetModule.FLATTEN_CROSS_PRODUCT = origFlattenCross

        self.assertIn(((1,2),(5,6)), x)
        self.assertIn((1,(2,5),6), x)
        self.assertIn((1,2,5,6), x)
        self.assertNotIn((5,6,1,2), x)

    def test_ordered_nondim_setproduct(self):
        NonDim = Set(initialize=[2, (2,3)], dimen=None)
        NonDim.construct()

        NonDim2 = Set(initialize=[4, (3,4)], dimen=None)
        NonDim2.construct()

        x = SetOf([1]).cross(NonDim, SetOf([3,4,5]))

        self.assertEqual(len(x), 6)
        try:
            origFlattenCross = SetModule.FLATTEN_CROSS_PRODUCT

            SetModule.FLATTEN_CROSS_PRODUCT = True
            ref = [(1,2,3), (1,2,4), (1,2,5),
                   (1,2,3,3), (1,2,3,4), (1,2,3,5)]
            self.assertEqual(list(x), ref)

            SetModule.FLATTEN_CROSS_PRODUCT = False
            ref = [(1,2,3), (1,2,4), (1,2,5),
                   (1,(2,3),3), (1,(2,3),4), (1,(2,3),5)]
            self.assertEqual(list(x), ref)
        finally:
            SetModule.FLATTEN_CROSS_PRODUCT = origFlattenCross

        self.assertIn((1,2,3), x)
        self.assertNotIn((1,2,6), x)
        self.assertIn((1,(2,3),3), x)
        self.assertIn((1,2,3,3), x)
        self.assertNotIn((1,(2,4),3), x)

        self.assertEqual(x.ord((1, 2, 3)), 1)
        self.assertEqual(x.ord((1, (2, 3), 3)), 4)
        self.assertEqual(x.ord((1, (2, 3), 5)), 6)
        self.assertEqual(x.ord((1, 2, 3, 3)), 4)
        self.assertEqual(x.ord((1, 2, 3, 5)), 6)

        x = SetOf([1]).cross(NonDim, NonDim2, SetOf([0,1]))

        self.assertEqual(len(x), 8)
        try:
            origFlattenCross = SetModule.FLATTEN_CROSS_PRODUCT

            SetModule.FLATTEN_CROSS_PRODUCT = True
            ref = [(1,2,4,0), (1,2,4,1), (1,2,3,4,0), (1,2,3,4,1),
                   (1,2,3,4,0), (1,2,3,4,1), (1,2,3,3,4,0), (1,2,3,3,4,1)]
            self.assertEqual(list(x), ref)
            for i,v in enumerate(ref):
                self.assertEqual(x[i+1], v)

            SetModule.FLATTEN_CROSS_PRODUCT = False
            ref = [(1,2,4,0), (1,2,4,1),
                   (1,2,(3,4),0), (1,2,(3,4),1),
                   (1,(2,3),4,0), (1,(2,3),4,1),
                   (1,(2,3),(3,4),0), (1,(2,3),(3,4),1)]
            self.assertEqual(list(x), ref)
            for i,v in enumerate(ref):
                self.assertEqual(x[i+1], v)
        finally:
            SetModule.FLATTEN_CROSS_PRODUCT = origFlattenCross

        self.assertIn((1,2,4,0), x)
        self.assertNotIn((1,2,6), x)
        self.assertIn((1,(2,3),4,0), x)
        self.assertIn((1,2,(3,4),0), x)
        self.assertIn((1,2,3,4,0), x)
        self.assertNotIn((1,2,5,4,0), x)

        self.assertEqual(x.ord((1, 2, 4, 0)), 1)
        self.assertEqual(x.ord((1, (2, 3), 4, 0)), 5)
        self.assertEqual(x.ord((1, 2, (3, 4), 0)), 3)
        self.assertEqual(x.ord((1, 2, 3, 4, 0)), 3)


def _init_scalar(m):
    return 1

def _init_indexed(m, *args):
    i = 1
    for arg in args:
        i *= (arg+1)
    return i

class Test_Initializer(unittest.TestCase):
    def test_constant(self):
        m = ConcreteModel()
        a = Initializer(m, 5)
        self.assertIs(type(a), _ConstantInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(a(None, 1), 5)

    def test_dict(self):
        m = ConcreteModel()
        a = Initializer(m, {1:5})
        self.assertIs(type(a), _ItemInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(a(None, 1), 5)

    def test_sequence(self):
        m = ConcreteModel()
        a = Initializer(m, [0,5])
        self.assertIs(type(a), _ItemInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(a(None, 1), 5)

        a = Initializer(m, [0,5], treat_sequences_as_mappings=False)
        self.assertIs(type(a), _ConstantInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(a(None, 1), [0,5])

    def test_function(self):
        m = ConcreteModel()
        def a_init(m):
            return 0
        a = Initializer(m, a_init)
        self.assertIs(type(a), _ScalarCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(a(None, 1), 0)

        m.x = Var([1,2,3])
        def x_init(m, i):
            return i+1
        a = Initializer(m.x, x_init)
        self.assertIs(type(a), _IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(a(None, 1), 2)

        m.y = Var([1,2,3], [4,5,6])
        def y_init(m, i, j):
            return j*(i+1)
        a = Initializer(m.y, y_init)
        self.assertIs(type(a), _IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(a(None, (1, 4)), 8)

    def test_generator_fcn(self):
        m = ConcreteModel()
        def a_init(m):
            yield 0
            yield 3
        with self.assertRaisesRegexp(
                ValueError, "Generator functions are not allowed"):
            a = Initializer(m, a_init)

        a = Initializer(m, a_init, allow_generators=True)
        self.assertIs(type(a), _ScalarCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, 1)), [0,3])

        m.x = Var([1,2,3])
        def x_init(m, i):
            yield i
            yield i+1
        a = Initializer(m.x, x_init, allow_generators=True)
        self.assertIs(type(a), _IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, 1)), [1,2])

        m.y = Var([1,2,3], [4,5,6])
        def y_init(m, i, j):
            yield j
            yield i+1
        a = Initializer(m.y, y_init, allow_generators=True)
        self.assertIs(type(a), _IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, (1, 4))), [4,2])

    def test_generators(self):
        m = ConcreteModel()
        with self.assertRaisesRegexp(
                ValueError, "Generators are not allowed"):
            a = Initializer(m, iter([0,3]))

        a = Initializer(m, iter([0,3]), allow_generators=True)
        self.assertIs(type(a), _ConstantInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, 1)), [0,3])

        def x_init():
            yield 0
            yield 3
        with self.assertRaisesRegexp(
                ValueError, "Generators are not allowed"):
            a = Initializer(m, x_init())

        a = Initializer(m, x_init(), allow_generators=True)
        self.assertIs(type(a), _ConstantInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, 1)), [0,3])

    def test_pickle(self):
        m = ConcreteModel()
        a = Initializer(m, 5)
        a.verified = True
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a, b)
        self.assertEqual(a.val, b.val)
        self.assertEqual(a.verified, b.verified)

        a = Initializer(m, {1:5})
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a, b)
        self.assertEqual(a._dict, b._dict)
        self.assertIsNot(a._dict, b._dict)
        self.assertEqual(a.verified, b.verified)

        a = Initializer(m, _init_scalar)
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a, b)
        self.assertIs(a._fcn, b._fcn)
        self.assertEqual(a.verified, b.verified)
        self.assertEqual(a(None, None), 1)
        self.assertEqual(b(None, None), 1)

        m.x = Var([1,2,3])
        a = Initializer(m.x, _init_indexed)
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a, b)
        self.assertIs(a._fcn, b._fcn)
        self.assertEqual(a.verified, b.verified)
        self.assertEqual(a(None, 1), 2)
        self.assertEqual(b(None, 2), 3)


class Test_SetInitializer(unittest.TestCase):
    def test_single_set(self):
        a = SetInitializer(self, None)
        self.assertIs(type(a), SetInitializer)
        self.assertIsNone(a._set)
        self.assertIs(a(None,None), Any)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)

        a = SetInitializer(self, Reals)
        self.assertIs(type(a), SetInitializer)
        self.assertIs(type(a._set), _ConstantInitializer)
        self.assertIs(a(None,None), Reals)
        self.assertIs(a._set.val, Reals)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)

        a = SetInitializer(self, {1:Reals})
        self.assertIs(type(a), SetInitializer)
        self.assertIs(type(a._set), _ItemInitializer)
        self.assertIs(a(None, 1), Reals)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)

    def test_intersect(self):
        a = SetInitializer(self, None)
        a.intersect(SetInitializer(self, None))
        self.assertIs(type(a), SetInitializer)
        self.assertIsNone(a._set)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertIs(a(None,None), Any)

        a = SetInitializer(self, None)
        a.intersect(SetInitializer(self, Reals))
        self.assertIs(type(a), SetInitializer)
        self.assertIs(type(a._set), _ConstantInitializer)
        self.assertIs(a._set.val, Reals)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertIs(a(None,None), Reals)

        a = SetInitializer(self, None)
        a.intersect(RangeSetInitializer(self, 5))
        self.assertIs(type(a), SetInitializer)
        self.assertIs(type(a._set), RangeSetInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(a(None,None), RangeSet(5))

        a = SetInitializer(self, Reals)
        a.intersect(SetInitializer(self, None))
        self.assertIs(type(a), SetInitializer)
        self.assertIs(type(a._set), _ConstantInitializer)
        self.assertIs(a._set.val, Reals)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertIs(a(None,None), Reals)

        a = SetInitializer(self, Reals)
        a.intersect(SetInitializer(self, Integers))
        self.assertIs(type(a), SetInitializer)
        self.assertIs(type(a._set), _SetIntersectInitializer)
        self.assertIs(type(a._set._A), _ConstantInitializer)
        self.assertIs(type(a._set._B), _ConstantInitializer)
        self.assertIs(a._set._A.val, Reals)
        self.assertIs(a._set._B.val, Integers)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertIs(type(s), _SetIntersection_InfiniteSet)
        self.assertIs(s._sets[0], Reals)
        self.assertIs(s._sets[1], Integers)

        a = SetInitializer(self, Reals)
        a.intersect(SetInitializer(self, Integers))
        a.intersect(RangeSetInitializer(self, 3))
        self.assertIs(type(a), SetInitializer)
        self.assertIs(type(a._set), _SetIntersectInitializer)
        self.assertIs(type(a._set._A), _SetIntersectInitializer)
        self.assertIs(type(a._set._B), RangeSetInitializer)
        self.assertIs(a._set._A._A.val, Reals)
        self.assertIs(a._set._A._B.val, Integers)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertIs(type(s), _SetIntersection_OrderedSet)
        self.assertIs(type(s._sets[0]), _SetIntersection_InfiniteSet)
        self.assertIsInstance(s._sets[1], RangeSet)

        a = SetInitializer(self, Reals)
        a.intersect(SetInitializer(self, Integers))
        a.intersect(RangeSetInitializer(self, 3, default_step=0))
        self.assertIs(type(a), SetInitializer)
        self.assertIs(type(a._set), _SetIntersectInitializer)
        self.assertIs(type(a._set._A), _SetIntersectInitializer)
        self.assertIs(type(a._set._B), RangeSetInitializer)
        self.assertIs(a._set._A._A.val, Reals)
        self.assertIs(a._set._A._B.val, Integers)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertIs(type(s), _SetIntersection_OrderedSet)
        self.assertIs(type(s._sets[0]), _SetIntersection_InfiniteSet)
        self.assertIsInstance(s._sets[1], RangeSet)
        self.assertFalse(s._sets[0].is_finite())
        self.assertFalse(s._sets[1].is_finite())
        self.assertTrue(s.is_finite())

        a = SetInitializer(self, Reals)
        a.intersect(SetInitializer(self, {1:Integers}))
        a.intersect(RangeSetInitializer(self, 3, default_step=0))
        self.assertIs(type(a), SetInitializer)
        self.assertIs(type(a._set), _SetIntersectInitializer)
        self.assertIs(type(a._set._A), _SetIntersectInitializer)
        self.assertIs(type(a._set._B), RangeSetInitializer)
        self.assertIs(a._set._A._A.val, Reals)
        self.assertIs(type(a._set._A._B), _ItemInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        with self.assertRaises(KeyError):
            a(None,None)
        s = a(None,1)
        self.assertIs(type(s), _SetIntersection_OrderedSet)
        self.assertIs(type(s._sets[0]), _SetIntersection_InfiniteSet)
        self.assertIsInstance(s._sets[1], RangeSet)
        self.assertFalse(s._sets[0].is_finite())
        self.assertFalse(s._sets[1].is_finite())
        self.assertTrue(s.is_finite())

    def test_rangeset(self):
        a = RangeSetInitializer(None, 5)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertEqual(s, RangeSet(5))

        a = RangeSetInitializer(None, (0,5))
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertEqual(s, RangeSet(0,5))

        a = RangeSetInitializer(None, (0,5,2))
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertEqual(s, RangeSet(0,5,2))

        a = RangeSetInitializer(None, 5, default_step=0)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertEqual(s, RangeSet(1,5,0))

        a = RangeSetInitializer(None, (0,5), default_step=0)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertEqual(s, RangeSet(0,5,0))

        a = RangeSetInitializer(None, (0,5,2), default_step=0)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertEqual(s, RangeSet(0,5,2))

        a = RangeSetInitializer(None, {1:5})
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        s = a(None,1)
        self.assertEqual(s, RangeSet(5))

        a = RangeSetInitializer(None, {1:(0,5)})
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        s = a(None,1)
        self.assertEqual(s, RangeSet(0,5))

    def test_setdefault(self):
        a = SetInitializer(None, None)
        self.assertIs(a(None,None), Any)
        a.setdefault(Reals)
        self.assertIs(a(None,None), Reals)

        a = SetInitializer(None, Integers)
        self.assertIs(a(None,None), Integers)
        a.setdefault(Reals)
        self.assertIs(a(None,None), Integers)

        a = RangeSetInitializer(None, 5)
        self.assertEqual(a(None,None), RangeSet(5))
        a.setdefault(Reals)
        self.assertEqual(a(None,None), RangeSet(5))

        a = SetInitializer(None, Reals)
        a.intersect(SetInitializer(None, Integers))
        self.assertIs(type(a(None,None)), _SetIntersection_InfiniteSet)
        a.setdefault(RangeSet(5))
        self.assertIs(type(a(None,None)), _SetIntersection_InfiniteSet)


class TestGlobalSets(unittest.TestCase):
    def test_globals(self):
        self.assertEqual(Reals.__class__.__name__, 'GlobalSet')
        self.assertIsInstance(Reals, RangeSet)

    def test_pickle(self):
        a = pickle.loads(pickle.dumps(Reals))
        self.assertIs(a, Reals)

    def test_deepcopy(self):
        a = copy.deepcopy(Reals)
        self.assertIs(a, Reals)
