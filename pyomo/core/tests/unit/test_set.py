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
import logging
import pickle
from six import StringIO

import pyutilib.th as unittest

from pyomo.common.log import LoggingIntercept
from pyomo.common import DeveloperError
from pyomo.core.expr import native_numeric_types, native_types
import pyomo.core.base.set as SetModule
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.set import (
    NumericRange as NR, NonNumericRange as NNR, AnyRange, _AnySet,
    Any, Reals, NonNegativeReals, Integers, PositiveIntegers,
    NegativeIntegers, PositiveReals,
    RangeSet, Set, SetOf,
    _FiniteRangeSetData, _InfiniteRangeSetData,
    SetUnion_InfiniteSet, SetUnion_FiniteSet, SetUnion_OrderedSet,
    SetIntersection_InfiniteSet, SetIntersection_FiniteSet,
    SetIntersection_OrderedSet,
    SetDifference_InfiniteSet, SetDifference_FiniteSet,
    SetDifference_OrderedSet,
    SetSymmetricDifference_InfiniteSet, SetSymmetricDifference_FiniteSet,
    SetSymmetricDifference_OrderedSet,
    SetProduct, SetProduct_InfiniteSet, SetProduct_FiniteSet,
    SetProduct_OrderedSet,
    Initializer, _ConstantInitializer, _ItemInitializer, _ScalarCallInitializer,
    _IndexedCallInitializer, _CountedCallInitializer, _CountedCallGenerator,
    SetInitializer, _SetIntersectInitializer, RangeSetInitializer,
    _SetData, _FiniteSetData, _InsertionOrderSetData, _SortedSetData,
    _FiniteSetMixin, _OrderedSetMixin,
    _UnknownSetDimen,
    simple_set_rule, set_options,
)
from pyomo.environ import (
    AbstractModel, ConcreteModel, Var, Param, Suffix, Constraint,
)

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
        self.assertFalse(a.is_discrete())
        self.assertFalse(a.is_finite())

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
        self.assertEqual(kr, [AnyRange()])
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
    def test_constructor(self):
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

        with self.assertRaisesRegexp(
                TypeError, "ranges argument must be an iterable of "
                "NumericRange objects"):
            RangeSet(ranges=(NR(1,5,1), NNR('a')))

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core', logging.DEBUG):
            i = RangeSet(5)
            self.assertEqual(output.getvalue(), "")
            i.construct()
            ref = 'Constructing RangeSet, name=FiniteSimpleRangeSet, '\
                  'from data=None\n'
            self.assertEqual(output.getvalue(), ref)
            # Calling construct() twice bupasses construction the second
            # time around
            i.construct()
            self.assertEqual(output.getvalue(), ref)

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core', logging.DEBUG):
            i = SetOf([1,2,3])
            self.assertEqual(output.getvalue(), "")
            i.construct()
            ref = 'Constructing SetOf, name=OrderedSetOf, '\
                  'from data=None\n'
            self.assertEqual(output.getvalue(), ref)
            # Calling construct() twice bupasses construction the second
            # time around
            i.construct()
            self.assertEqual(output.getvalue(), ref)

    def test_contains(self):
        r = RangeSet(5)
        self.assertIn(1, r)
        self.assertIn((1,), r)
        self.assertNotIn(6, r)
        self.assertNotIn((6,), r)

        r = SetOf([1, (2,)])
        self.assertIn(1, r)
        self.assertIn((1,), r)
        self.assertNotIn(2, r)
        self.assertIn((2,), r)

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

    def test_inequality(self):
        self.assertTrue(SetOf([1,2,3]) <= SetOf({1,2,3}))
        self.assertFalse(SetOf([1,2,3]) < SetOf({1,2,3}))

        self.assertTrue(SetOf([1,2,3]) <= SetOf({1,2,3,4}))
        self.assertTrue(SetOf([1,2,3]) < SetOf({1,2,3,4}))

        self.assertFalse(SetOf([1,2,3]) <= SetOf({1,2}))
        self.assertFalse(SetOf([1,2,3]) < SetOf({1,2}))

        self.assertTrue(SetOf([1,2,3]) >= SetOf({1,2,3}))
        self.assertFalse(SetOf([1,2,3]) > SetOf({1,2,3}))

        self.assertFalse(SetOf([1,2,3]) >= SetOf({1,2,3,4}))
        self.assertFalse(SetOf([1,2,3]) > SetOf({1,2,3,4}))

        self.assertTrue(SetOf([1,2,3]) >= SetOf({1,2}))
        self.assertTrue(SetOf([1,2,3]) > SetOf({1,2}))


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
        m.pprint()
        m.pprint(ostream=buf)
        self.assertEqual(buf.getvalue().strip(), """
2 RangeSet Declarations
    I : Dimen=1, Size=3, Bounds=(1, 3)
        Key  : Finite : Members
        None :   True :   [1:3]
    NotI : Dimen=1, Size=Inf, Bounds=(1, 3)
        Key  : Finite : Members
        None :  False :  [1..3]

1 SetOf Declarations
    J : Dimen=1, Size=3, Bounds=(1, 3)
        Key  : Ordered : Members
        None :    True : [1, 2, 3]

3 Declarations: I NotI J""".strip())

    def test_naming(self):
        m = ConcreteModel()

        i = RangeSet(3)
        self.assertEqual(str(i), "[1:3]")
        m.I = i
        self.assertEqual(str(i), "I")

        j = RangeSet(ranges=(NR(1,3,0), NR(4,7,1)))
        self.assertEqual(str(j), "([1..3] | [4:7])")
        m.J = j
        self.assertEqual(str(j), "J")

        k = SetOf((1,3,5))
        self.assertEqual(str(k), "(1, 3, 5)")
        m.K = k
        self.assertEqual(str(k), "K")

        l = SetOf([1,3,5])
        self.assertEqual(str(l), "[1, 3, 5]")
        m.L = l
        self.assertEqual(str(l), "L")

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

        # It can even work for non-hashable objects (that can't be cast
        # to set())
        m = ConcreteModel()
        m.p = Param(initialize=2)
        m.q = Var(initialize=2)
        _NonHashable = (1,3,5,m.p)
        self.assertFalse(SetOf({2,4}).isdisjoint(_NonHashable))
        self.assertTrue(SetOf({0,4}).isdisjoint(_NonHashable))
        self.assertFalse(SetOf(_NonHashable).isdisjoint(_NonHashable))
        self.assertFalse(SetOf((m.q,1,3,5,m.p)).isdisjoint(_NonHashable))
        # Note: membership in tuples is done through
        # __bool__(a.__eq__(b)), so Params/Vars with equivalent values will
        # match
        self.assertFalse(SetOf((m.q,)).isdisjoint(_NonHashable))

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

        # It can even work for non-hashable objects (that can't be cast
        # to set())
        m = ConcreteModel()
        m.p = Param(initialize=2)
        m.q = Var(initialize=2)
        _NonHashable = (1,3,5,m.p)
        self.assertFalse(SetOf({0,1,3,5}).issubset(_NonHashable))
        self.assertTrue(SetOf({1,3,5}).issubset(_NonHashable))
        self.assertTrue(SetOf(_NonHashable).issubset(_NonHashable))
        self.assertTrue(SetOf((m.q,1,3,5,m.p)).issubset(_NonHashable))
        # Note: membership in tuples is done through
        # __bool__(a.__eq__(b)), so Params/Vars with equivalent values will
        # match
        self.assertTrue(SetOf((m.q,1,3,5)).issubset(_NonHashable))

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

        # It can even work for non-hashable objects (that can't be cast
        # to set())
        m = ConcreteModel()
        m.p = Param(initialize=2)
        m.q = Var(initialize=2)
        _NonHashable = (1,3,5,m.p)
        self.assertFalse(SetOf({1,3,5}).issuperset(_NonHashable))
        self.assertTrue(SetOf(_NonHashable).issuperset(_NonHashable))
        self.assertTrue(SetOf((m.q,1,3,5,m.p)).issuperset(_NonHashable))
        # Note: membership in tuples is done through
        # __bool__(a.__eq__(b)), so Params/Vars with equivalent values will
        # match
        self.assertTrue(SetOf((m.q,1,3,5)).issuperset(_NonHashable))

        # But NOT non-iterable objects: we assume that everything that
        # does not implement is_finite() is a discrete set.
        class _NonIterable(object):
            def __init__(self):
                self.data = set({1,3,5})
            def __contains__(self, val):
                return val in self.data
            def __len__(self):
                return len(self.data)
        with self.assertRaisesRegexp(TypeError, 'not iterable'):
            SetOf({1,5}).issuperset(_NonIterable())
        with self.assertRaisesRegexp(TypeError, 'not iterable'):
            SetOf({1,3,4,5}).issuperset(_NonIterable())

    def test_unordered_setof(self):
        i = SetOf({1,3,2,0})

        self.assertTrue(i.is_finite())
        self.assertFalse(i.is_ordered())

        self.assertEqual(i.ordered_data(), (0,1,2,3))
        self.assertEqual(i.sorted_data(), (0,1,2,3))
        self.assertEqual( tuple(reversed(i)),
                          tuple(reversed(list(i))) )

    def test_ordered_setof(self):
        i = SetOf([1,3,2,0])

        self.assertTrue(i.is_finite())
        self.assertTrue(i.is_ordered())

        self.assertEqual(i.ordered_data(), (1,3,2,0))
        self.assertEqual(i.sorted_data(), (0,1,2,3))
        self.assertEqual(tuple(reversed(i)), (0,2,3,1))

        self.assertEqual(i[2], 3)
        self.assertEqual(i[-1], 0)
        with self.assertRaisesRegexp(
                IndexError,"valid index values for Sets are "
                "\[1 .. len\(Set\)\] or \[-1 .. -len\(Set\)\]"):
            i[0]
        with self.assertRaisesRegexp(
                IndexError, "OrderedSetOf index out of range"):
            i[5]
        with self.assertRaisesRegexp(
                IndexError, "OrderedSetOf index out of range"):
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

        self.assertEqual(i.ordered_data(), (1,3,2,0))
        self.assertEqual(i.sorted_data(), (0,1,2,3))
        self.assertEqual(tuple(reversed(i)), (0,2,3,1))

        self.assertEqual(i[2], 3)
        self.assertEqual(i[-1], 0)
        with self.assertRaisesRegexp(
                IndexError,"valid index values for Sets are "
                "\[1 .. len\(Set\)\] or \[-1 .. -len\(Set\)\]"):
            i[0]
        with self.assertRaisesRegexp(
                IndexError, "OrderedSetOf index out of range"):
            i[5]
        with self.assertRaisesRegexp(
                IndexError, "OrderedSetOf index out of range"):
            i[-5]

        self.assertEqual(i.ord(3), 2)
        with self.assertRaisesRegexp(ValueError, "x not in tuple"):
            i.ord(5)

        i = SetOf([1, None, 'a'])

        self.assertTrue(i.is_finite())
        self.assertTrue(i.is_ordered())

        self.assertEqual(i.ordered_data(), (1,None,'a'))
        self.assertEqual(i.sorted_data(), (None,1,'a'))
        self.assertEqual(tuple(reversed(i)), ('a',None,1))


    def test_ranges(self):
        i_data = [1,3,2,0]
        i = SetOf(i_data)
        r = list(i.ranges())
        self.assertEqual(len(r), 4)
        for idx, x in enumerate(r):
            self.assertIsInstance(x, NR)
            self.assertTrue(x.is_finite())
            self.assertEqual(x.start, i[idx+1])
            self.assertEqual(x.end, i[idx+1])
            self.assertEqual(x.step, 0)

        # Test that apparent numeric types that are not in native_types
        # are handled correctly
        try:
            self.assertIn(int, native_types)
            self.assertIn(int, native_numeric_types)

            native_types.remove(int)
            native_numeric_types.remove(int)

            r = list(i.ranges())
            self.assertEqual(len(r), 4)
            self.assertIn(int, native_types)
            self.assertIn(int, native_numeric_types)
        finally:
            native_types.add(int)
            native_numeric_types.add(int)

        i_data.append('abc')
        try:
            self.assertIn(str, native_types)
            self.assertNotIn(str, native_numeric_types)

            native_types.remove(str)

            r = list(i.ranges())

            self.assertEqual(len(r), 5)
            # Note: as_numeric() will NOT automatically add types to the
            # native_types set
            self.assertNotIn(str, native_types)
            self.assertNotIn(str, native_numeric_types)
            self.assertIs(type(r[-1]), NNR)
        finally:
            native_types.add(str)

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
        self.assertEqual(SetOf([]).dimen, 0)
        self.assertEqual(SetOf([1,2,3]).dimen, 1)
        self.assertEqual(SetOf([(1,2),(2,3),(4,5)]).dimen, 2)
        self.assertEqual(SetOf([1,(2,3)]).dimen, None)

        a = [1,2,3,'abc']
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

    def test_ord_index(self):
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

        so = SetOf([0, (1,), 1])
        self.assertEqual(so.ord((1,)), 2)
        self.assertEqual(so.ord(1), 3)


class TestSetUnion(unittest.TestCase):
    def test_pickle(self):
        a = SetOf([1,3,5]) | SetOf([2,3,4])
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a,b)
        self.assertEqual(a,b)

    def test_naming(self):
        m = ConcreteModel()

        m.I = SetOf([1,2])
        a = m.I | [3,4]
        b = [-1,1] | a
        self.assertEqual(str(a), "I | {3, 4}")
        self.assertEqual(str(b), "{-1, 1} | (I | {3, 4})")
        m.A = a
        self.assertEqual(str(a), "A")
        self.assertEqual(str(b), "{-1, 1} | A")

    def test_domain_and_pprint(self):
        m = ConcreteModel()
        m.I = SetOf([1,2])
        m.A = m.I | [3,4]

        self.assertIs(m.A._domain, m.A)
        # You can always set the domain to "Any" (we will just ignore it)
        m.A._domain = Any
        self.assertIs(m.A._domain, m.A)
        with self.assertRaisesRegexp(
                ValueError,
                "Setting the domain of a Set Operator is not allowed"):
            m.A._domain = None

        output = StringIO()
        m.A.pprint(ostream=output)
        ref="""
A : Size=1, Index=None, Ordered=True
    Key  : Dimen : Domain     : Size : Members
    None :     1 : I | {3, 4} :    4 : {1, 2, 3, 4}
""".strip()
        self.assertEqual(output.getvalue().strip(), ref)


    def test_dimen(self):
        m = ConcreteModel()
        m.I1 = SetOf([1,2,3,4])
        m.I2 = SetOf([(1,2), (3,4)])
        m.IN = SetOf([(1,2), (3,4), 1, 2])
        self.assertEqual((m.I1 | m.I1).dimen, 1)
        self.assertEqual((m.I2 | m.I2).dimen, 2)
        self.assertEqual((m.IN | m.IN).dimen, None)
        self.assertEqual((m.I1 | m.I2).dimen, None)
        self.assertEqual((m.IN | m.I2).dimen, None)
        self.assertEqual((m.I2 | m.IN).dimen, None)
        self.assertEqual((m.IN | m.I1).dimen, None)
        self.assertEqual((m.I1 | m.IN).dimen, None)

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
        self.assertIs(type(x), SetUnion_OrderedSet)
        self.assertTrue(x.is_finite())
        self.assertTrue(x.is_ordered())
        self.assertEqual(len(x), 5)
        self.assertEqual(list(x), [1,3,2,5,4])
        self.assertEqual(x.ordered_data(), (1,3,2,5,4))
        self.assertEqual(x.sorted_data(), (1,2,3,4,5))

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
                "Cannot identify position of 6 in Set SetUnion_OrderedSet"):
            x.ord(6)

        self.assertEqual(x[1], 1)
        self.assertEqual(x[2], 3)
        self.assertEqual(x[3], 2)
        self.assertEqual(x[4], 5)
        self.assertEqual(x[5], 4)
        with self.assertRaisesRegexp(
                IndexError,
                "SetUnion_OrderedSet index out of range"):
            x[6]

        self.assertEqual(x[-1], 4)
        self.assertEqual(x[-2], 5)
        self.assertEqual(x[-3], 2)
        self.assertEqual(x[-4], 3)
        self.assertEqual(x[-5], 1)
        with self.assertRaisesRegexp(
                IndexError,
                "SetUnion_OrderedSet index out of range"):
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
        self.assertIs(type(x), SetUnion_FiniteSet)
        self.assertTrue(x.is_finite())
        self.assertFalse(x.is_ordered())
        self.assertEqual(len(x), 5)
        if x._sets[0].is_ordered():
            self.assertEqual(list(x)[:3], [1,3,2])
        if x._sets[1].is_ordered():
            self.assertEqual(list(x)[-2:], [5,4])
        self.assertEqual(sorted(list(x)), [1,2,3,4,5])
        self.assertEqual(x.ordered_data(), (1,2,3,4,5))
        self.assertEqual(x.sorted_data(), (1,2,3,4,5))

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
        self.assertIs(type(x), SetUnion_InfiniteSet)
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

    def test_invalid_operators(self):
        m = ConcreteModel()
        m.I = RangeSet(5)
        m.J = Set([1,2])
        with self.assertRaisesRegexp(
                TypeError, "Cannot apply a Set operator to an "
                "indexed Set component \(J\)"):
            m.I | m.J
        m.x = Suffix()
        with self.assertRaisesRegexp(
                TypeError, "Cannot apply a Set operator to a "
                "non-Set Suffix component \(x\)"):
            m.I | m.x
        m.y = Var([1,2])
        with self.assertRaisesRegexp(
                TypeError, "Cannot apply a Set operator to an "
                "indexed Var component \(y\)"):
            m.I | m.y
        with self.assertRaisesRegexp(
                TypeError, "Cannot apply a Set operator to a "
                "non-Set component data \(y\[1\]\)"):
            m.I | m.y[1]

class TestSetIntersection(unittest.TestCase):
    def test_pickle(self):
        a = SetOf([1,3,5]) & SetOf([2,3,4])
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a,b)
        self.assertEqual(a,b)

    def test_naming(self):
        m = ConcreteModel()

        m.I = SetOf([1,2])
        a = m.I & [3,4]
        b = [-1,1] & a
        self.assertEqual(str(a), "I & {3, 4}")
        self.assertEqual(str(b), "{-1, 1} & (I & {3, 4})")
        m.A = a
        self.assertEqual(str(a), "A")
        self.assertEqual(str(b), "{-1, 1} & A")

    def test_domain_and_pprint(self):
        m = ConcreteModel()
        m.I = SetOf([1,2])
        m.A = m.I & [3,4]

        self.assertIs(m.A._domain, m.A)
        # You can always set the domain to "Any" (we will just ignore it)
        m.A._domain = Any
        self.assertIs(m.A._domain, m.A)
        with self.assertRaisesRegexp(
                ValueError,
                "Setting the domain of a Set Operator is not allowed"):
            m.A._domain = None

        output = StringIO()
        m.A.pprint(ostream=output)
        ref="""
A : Size=1, Index=None, Ordered=True
    Key  : Dimen : Domain     : Size : Members
    None :     1 : I & {3, 4} :    0 :      {}
""".strip()
        self.assertEqual(output.getvalue().strip(), ref)

    def test_dimen(self):
        m = ConcreteModel()
        m.I1 = SetOf([1,2,3,4])
        m.I2 = SetOf([(1,2), (3,4)])
        m.IN = SetOf([(1,2), (3,4), 1, 2])
        self.assertEqual((m.I1 & m.I1).dimen, 1)
        self.assertEqual((m.I2 & m.I2).dimen, 2)
        self.assertEqual((m.IN & m.IN).dimen, None)
        self.assertEqual((m.I1 & m.I2).dimen, 0)
        self.assertEqual((m.IN & m.I2).dimen, 2)
        self.assertEqual((m.I2 & m.IN).dimen, 2)
        self.assertEqual((m.IN & m.I1).dimen, 1)
        self.assertEqual((m.I1 & m.IN).dimen, 1)

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
        self.assertIs(type(x), SetIntersection_OrderedSet)
        self.assertTrue(x.is_finite())
        self.assertTrue(x.is_ordered())
        self.assertEqual(len(x), 3)
        self.assertEqual(list(x), list(ref))
        self.assertEqual(x.ordered_data(), tuple(ref))
        self.assertEqual(x.sorted_data(), (2,3,5))

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
                "SetIntersection_OrderedSet"):
            x.ord(6)

        self.assertEqual(x[1], ref[0])
        self.assertEqual(x[2], ref[1])
        self.assertEqual(x[3], 5)
        with self.assertRaisesRegexp(
                IndexError,
                "SetIntersection_OrderedSet index out of range"):
            x[4]

        self.assertEqual(x[-1], 5)
        self.assertEqual(x[-2], ref[-2])
        self.assertEqual(x[-3], ref[-3])
        with self.assertRaisesRegexp(
                IndexError,
                "SetIntersection_OrderedSet index out of range"):
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
        self.assertIs(type(x), SetIntersection_FiniteSet)
        self.assertTrue(x.is_finite())
        self.assertFalse(x.is_ordered())
        self.assertEqual(len(x), 3)
        if x._sets[0].is_ordered():
            self.assertEqual(list(x)[:3], [3,2,5])
        self.assertEqual(sorted(list(x)), [2,3,5])
        self.assertEqual(x.ordered_data(), (2,3,5))
        self.assertEqual(x.sorted_data(), (2,3,5))

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
        self.assertIs(type(x), SetIntersection_InfiniteSet)
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
        self.assertIs(type(x), SetIntersection_OrderedSet)
        self.assertEqual(list(x), [6,8,10])

        self.assertEqual(x.ord(6), 1)
        self.assertEqual(x.ord(8), 2)
        self.assertEqual(x.ord(10), 3)

        self.assertEqual(x[1], 6)
        self.assertEqual(x[2], 8)
        self.assertEqual(x[3], 10)
        with self.assertRaisesRegexp(
                IndexError,
                "SetIntersection_OrderedSet index out of range"):
            x[4]

        self.assertEqual(x[-3], 6)
        self.assertEqual(x[-2], 8)
        self.assertEqual(x[-1], 10)
        with self.assertRaisesRegexp(
                IndexError,
                "SetIntersection_OrderedSet index out of range"):
            x[-4]



class TestSetDifference(unittest.TestCase):
    def test_pickle(self):
        a = SetOf([1,3,5]) - SetOf([2,3,4])
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a,b)
        self.assertEqual(a,b)

    def test_naming(self):
        m = ConcreteModel()

        m.I = SetOf([1,2])
        a = m.I - [3,4]
        b = [-1,1] - a
        self.assertEqual(str(a), "I - {3, 4}")
        self.assertEqual(str(b), "{-1, 1} - (I - {3, 4})")
        m.A = a
        self.assertEqual(str(a), "A")
        self.assertEqual(str(b), "{-1, 1} - A")

    def test_domain_and_pprint(self):
        m = ConcreteModel()
        m.I = SetOf([1,2])
        m.A = m.I - [3,4]

        self.assertIs(m.A._domain, m.A)
        # You can always set the domain to "Any" (we will just ignore it)
        m.A._domain = Any
        self.assertIs(m.A._domain, m.A)
        with self.assertRaisesRegexp(
                ValueError,
                "Setting the domain of a Set Operator is not allowed"):
            m.A._domain = None

        output = StringIO()
        m.A.pprint(ostream=output)
        ref="""
A : Size=1, Index=None, Ordered=True
    Key  : Dimen : Domain     : Size : Members
    None :     1 : I - {3, 4} :    2 : {1, 2}
""".strip()
        self.assertEqual(output.getvalue().strip(), ref)

    def test_dimen(self):
        m = ConcreteModel()
        m.I1 = SetOf([1,2,3,4])
        m.I2 = SetOf([(1,2), (3,4)])
        m.IN = SetOf([(1,2), (3,4), 1, 2])
        self.assertEqual((m.I1 - m.I1).dimen, 1)
        self.assertEqual((m.I2 - m.I2).dimen, 2)
        self.assertEqual((m.IN - m.IN).dimen, None)
        self.assertEqual((m.I1 - m.I2).dimen, 1)
        self.assertEqual((m.I2 - m.I1).dimen, 2)
        self.assertEqual((m.IN - m.I2).dimen, None)
        self.assertEqual((m.I2 - m.IN).dimen, 2)
        self.assertEqual((m.IN - m.I1).dimen, None)
        self.assertEqual((m.I1 - m.IN).dimen, 1)

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
        self.assertIs(type(x), SetDifference_OrderedSet)
        self.assertTrue(x.is_finite())
        self.assertTrue(x.is_ordered())
        self.assertEqual(len(x), 3)
        self.assertEqual(list(x), [3,2,5])
        self.assertEqual(x.ordered_data(), (3,2,5))
        self.assertEqual(x.sorted_data(), (2,3,5))

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
                "SetDifference_OrderedSet"):
            x.ord(6)

        self.assertEqual(x[1], 3)
        self.assertEqual(x[2], 2)
        self.assertEqual(x[3], 5)
        with self.assertRaisesRegexp(
                IndexError,
                "SetDifference_OrderedSet index out of range"):
            x[4]

        self.assertEqual(x[-1], 5)
        self.assertEqual(x[-2], 2)
        self.assertEqual(x[-3], 3)
        with self.assertRaisesRegexp(
                IndexError,
                "SetDifference_OrderedSet index out of range"):
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
        self.assertIs(type(x), SetDifference_FiniteSet)
        self.assertTrue(x.is_finite())
        self.assertFalse(x.is_ordered())
        self.assertEqual(len(x), 3)
        self.assertEqual(sorted(list(x)), [2,3,5])
        self.assertEqual(x.ordered_data(), (2,3,5))
        self.assertEqual(x.sorted_data(), (2,3,5))

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
        self.assertIs(type(x), SetDifference_InfiniteSet)
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
    def test_pickle(self):
        a = SetOf([1,3,5]) ^ SetOf([2,3,4])
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a,b)
        self.assertEqual(a,b)

    def test_naming(self):
        m = ConcreteModel()

        m.I = SetOf([1,2])
        a = m.I ^ [3,4]
        b = [-1,1] ^ a
        self.assertEqual(str(a), "I ^ {3, 4}")
        self.assertEqual(str(b), "{-1, 1} ^ (I ^ {3, 4})")
        m.A = a
        self.assertEqual(str(a), "A")
        self.assertEqual(str(b), "{-1, 1} ^ A")

    def test_domain_and_pprint(self):
        m = ConcreteModel()
        m.I = SetOf([1,2])
        m.A = m.I ^ [3,4]

        self.assertIs(m.A._domain, m.A)
        # You can always set the domain to "Any" (we will just ignore it)
        m.A._domain = Any
        self.assertIs(m.A._domain, m.A)
        with self.assertRaisesRegexp(
                ValueError,
                "Setting the domain of a Set Operator is not allowed"):
            m.A._domain = None

        output = StringIO()
        m.A.pprint(ostream=output)
        ref="""
A : Size=1, Index=None, Ordered=True
    Key  : Dimen : Domain     : Size : Members
    None :     1 : I ^ {3, 4} :    4 : {1, 2, 3, 4}
""".strip()
        self.assertEqual(output.getvalue().strip(), ref)

    def test_dimen(self):
        m = ConcreteModel()
        m.I1 = SetOf([1,2,3,4])
        m.I2 = SetOf([(1,2), (3,4)])
        m.IN = SetOf([(1,2), (3,4), 1, 2])
        self.assertEqual((m.I1 ^ m.I1).dimen, 1)
        self.assertEqual((m.I2 ^ m.I2).dimen, 2)
        self.assertEqual((m.IN ^ m.IN).dimen, None)
        self.assertEqual((m.I1 ^ m.I2).dimen, None)
        self.assertEqual((m.I2 ^ m.I1).dimen, None)
        self.assertEqual((m.IN ^ m.I2).dimen, None)
        self.assertEqual((m.I2 ^ m.IN).dimen, None)
        self.assertEqual((m.IN ^ m.I1).dimen, None)
        self.assertEqual((m.I1 ^ m.IN).dimen, None)

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
        self.assertIs(type(x), SetSymmetricDifference_OrderedSet)
        self.assertTrue(x.is_finite())
        self.assertTrue(x.is_ordered())
        self.assertEqual(len(x), 4)
        self.assertEqual(list(x), [3,2,5,0])
        self.assertEqual(x.ordered_data(), (3,2,5,0))
        self.assertEqual(x.sorted_data(), (0,2,3,5))

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
                "SetSymmetricDifference_OrderedSet"):
            x.ord(6)

        self.assertEqual(x[1], 3)
        self.assertEqual(x[2], 2)
        self.assertEqual(x[3], 5)
        self.assertEqual(x[4], 0)
        with self.assertRaisesRegexp(
                IndexError,
                "SetSymmetricDifference_OrderedSet index out of range"):
            x[5]

        self.assertEqual(x[-1], 0)
        self.assertEqual(x[-2], 5)
        self.assertEqual(x[-3], 2)
        self.assertEqual(x[-4], 3)
        with self.assertRaisesRegexp(
                IndexError,
                "SetSymmetricDifference_OrderedSet index out of range"):
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
        self.assertIs(type(x), SetSymmetricDifference_FiniteSet)
        self.assertTrue(x.is_finite())
        self.assertFalse(x.is_ordered())
        self.assertEqual(len(x), 4)
        self.assertEqual(sorted(list(x)), [0,2,3,5])
        self.assertEqual(x.ordered_data(), (0,2,3,5))
        self.assertEqual(x.sorted_data(), (0,2,3,5))

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
        self.assertIs(type(x), SetSymmetricDifference_InfiniteSet)
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
        self.assertIs(type(x), SetSymmetricDifference_InfiniteSet)
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
        self.assertIs(type(x), SetSymmetricDifference_InfiniteSet)
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
    def test_pickle(self):
        a = SetOf([1,3,5]) * SetOf([2,3,4])
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a,b)
        self.assertEqual(a,b)

    def test_naming(self):
        m = ConcreteModel()

        m.I = SetOf([1,2])
        a = m.I * [3,4]
        b = [-1,1] * a
        self.assertEqual(str(a), "I*{3, 4}")
        self.assertEqual(str(b), "{-1, 1}*(I*{3, 4})")
        m.A = a
        self.assertEqual(str(a), "A")
        self.assertEqual(str(b), "{-1, 1}*A")

        c = SetProduct(m.I, [1,2], m.I)
        self.assertEqual(str(c), "I*{1, 2}*I")

    def test_domain_and_pprint(self):
        m = ConcreteModel()
        m.I = SetOf([1,2])
        m.A = m.I * [3,4]

        self.assertIs(m.A._domain, m.A)
        # You can always set the domain to "Any" (we will just ignore it)
        m.A._domain = Any
        self.assertIs(m.A._domain, m.A)
        with self.assertRaisesRegexp(
                ValueError,
                "Setting the domain of a Set Operator is not allowed"):
            m.A._domain = None

        output = StringIO()
        m.A.pprint(ostream=output)
        ref="""
A : Size=1, Index=None, Ordered=True
    Key  : Dimen : Domain   : Size : Members
    None :     2 : I*{3, 4} :    4 : {(1, 3), (1, 4), (2, 3), (2, 4)}
""".strip()
        self.assertEqual(output.getvalue().strip(), ref)

    def test_dimen(self):
        m = ConcreteModel()
        m.I1 = SetOf([1,2,3,4])
        m.I2 = SetOf([(1,2), (3,4)])
        m.IN = SetOf([(1,2), (3,4), 1, 2])
        self.assertEqual((m.I1 * m.I1).dimen, 2)
        self.assertEqual((m.I2 * m.I2).dimen, 4)
        self.assertEqual((m.IN * m.IN).dimen, None)
        self.assertEqual((m.I1 * m.I2).dimen, 3)
        self.assertEqual((m.I2 * m.I1).dimen, 3)
        self.assertEqual((m.IN * m.I2).dimen, None)
        self.assertEqual((m.I2 * m.IN).dimen, None)
        self.assertEqual((m.IN * m.I1).dimen, None)
        self.assertEqual((m.I1 * m.IN).dimen, None)

    def test_cutPointGenerator(self):
        CG = SetProduct_InfiniteSet._cutPointGenerator
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

        self.assertIs(type(x), SetProduct_FiniteSet)
        self.assertTrue(x.is_finite())
        self.assertFalse(x.is_ordered())
        self.assertEqual(len(x), 6)
        self.assertEqual(
            sorted(list(x)), [(1,5),(1,6),(2,5),(2,6),(3,5),(3,6)])
        self.assertEqual(
            x.ordered_data(), ((1,5),(1,6),(2,5),(2,6),(3,5),(3,6)))
        self.assertEqual(
            x.sorted_data(), ((1,5),(1,6),(2,5),(2,6),(3,5),(3,6)))

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

        self.assertIs(type(x), SetProduct_OrderedSet)
        self.assertTrue(x.is_finite())
        self.assertTrue(x.is_ordered())
        self.assertEqual(len(x), 6)
        self.assertEqual(list(x), [(3,6),(3,5),(1,6),(1,5),(2,6),(2,5)])
        self.assertEqual(
            x.ordered_data(), ((3,6),(3,5),(1,6),(1,5),(2,6),(2,5)))
        self.assertEqual(
            x.sorted_data(), ((1,5),(1,6),(2,5),(2,6),(3,5),(3,6)))

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
                "SetProduct_OrderedSet"):
            x.ord((3,4))

        self.assertEqual(x[1], (3,6))
        self.assertEqual(x[2], (3,5))
        self.assertEqual(x[3], (1,6))
        self.assertEqual(x[4], (1,5))
        self.assertEqual(x[5], (2,6))
        self.assertEqual(x[6], (2,5))
        with self.assertRaisesRegexp(
                IndexError,
                "SetProduct_OrderedSet index out of range"):
            x[7]

        self.assertEqual(x[-6], (3,6))
        self.assertEqual(x[-5], (3,5))
        self.assertEqual(x[-4], (1,6))
        self.assertEqual(x[-3], (1,5))
        self.assertEqual(x[-2], (2,6))
        self.assertEqual(x[-1], (2,5))
        with self.assertRaisesRegexp(
                IndexError,
                "SetProduct_OrderedSet index out of range"):
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
        a = Initializer(5)
        self.assertIs(type(a), _ConstantInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(a(None, 1), 5)

    def test_dict(self):
        m = ConcreteModel()
        a = Initializer({1:5})
        self.assertIs(type(a), _ItemInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(a(None, 1), 5)

    def test_sequence(self):
        m = ConcreteModel()
        a = Initializer([0,5])
        self.assertIs(type(a), _ItemInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(a(None, 1), 5)

        a = Initializer([0,5], treat_sequences_as_mappings=False)
        self.assertIs(type(a), _ConstantInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(a(None, 1), [0,5])

    def test_function(self):
        m = ConcreteModel()
        def a_init(m):
            return 0
        a = Initializer(a_init)
        self.assertIs(type(a), _ScalarCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(a(None, 1), 0)

        m.x = Var([1,2,3])
        def x_init(m, i):
            return i+1
        a = Initializer(x_init)
        self.assertIs(type(a), _IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(a(None, 1), 2)

        def x2_init(m):
            return 0
        a = Initializer(x2_init)
        self.assertIs(type(a), _ScalarCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(a(None, 1), 0)

        m.y = Var([1,2,3], [4,5,6])
        def y_init(m, i, j):
            return j*(i+1)
        a = Initializer(y_init)
        self.assertIs(type(a), _IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(a(None, (1, 4)), 8)

        b = _CountedCallInitializer(m.x, a)
        self.assertIs(type(b), _CountedCallInitializer)
        self.assertFalse(b.constant())
        self.assertFalse(b.verified)
        self.assertFalse(b._scalar)
        self.assertIs(a._fcn, b._fcn)
        c = b(None, 1)
        self.assertIs(type(c), _CountedCallGenerator)
        self.assertEqual(next(c), 2)
        self.assertEqual(next(c), 3)
        self.assertEqual(next(c), 4)


    def test_generator_fcn(self):
        m = ConcreteModel()
        def a_init(m):
            yield 0
            yield 3
        with self.assertRaisesRegexp(
                ValueError, "Generator functions are not allowed"):
            a = Initializer(a_init)

        a = Initializer(a_init, allow_generators=True)
        self.assertIs(type(a), _ScalarCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, 1)), [0,3])

        m.x = Var([1,2,3])
        def x_init(m, i):
            yield i
            yield i+1
        a = Initializer(x_init, allow_generators=True)
        self.assertIs(type(a), _IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, 1)), [1,2])

        m.y = Var([1,2,3], [4,5,6])
        def y_init(m, i, j):
            yield j
            yield i+1
        a = Initializer(y_init, allow_generators=True)
        self.assertIs(type(a), _IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, (1, 4))), [4,2])

    def test_generators(self):
        m = ConcreteModel()
        with self.assertRaisesRegexp(
                ValueError, "Generators are not allowed"):
            a = Initializer(iter([0,3]))

        a = Initializer(iter([0,3]), allow_generators=True)
        self.assertIs(type(a), _ConstantInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, 1)), [0,3])

        def x_init():
            yield 0
            yield 3
        with self.assertRaisesRegexp(
                ValueError, "Generators are not allowed"):
            a = Initializer(x_init())

        a = Initializer(x_init(), allow_generators=True)
        self.assertIs(type(a), _ConstantInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, 1)), [0,3])

    def test_pickle(self):
        m = ConcreteModel()
        a = Initializer(5)
        a.verified = True
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a, b)
        self.assertEqual(a.val, b.val)
        self.assertEqual(a.verified, b.verified)

        a = Initializer({1:5})
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a, b)
        self.assertEqual(a._dict, b._dict)
        self.assertIsNot(a._dict, b._dict)
        self.assertEqual(a.verified, b.verified)

        a = Initializer(_init_scalar)
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a, b)
        self.assertIs(a._fcn, b._fcn)
        self.assertEqual(a.verified, b.verified)
        self.assertEqual(a(None, None), 1)
        self.assertEqual(b(None, None), 1)

        m.x = Var([1,2,3])
        a = Initializer(_init_indexed)
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a, b)
        self.assertIs(a._fcn, b._fcn)
        self.assertEqual(a.verified, b.verified)
        self.assertEqual(a(None, 1), 2)
        self.assertEqual(b(None, 2), 3)


class Test_SetInitializer(unittest.TestCase):
    def test_single_set(self):
        a = SetInitializer(None)
        self.assertIs(type(a), SetInitializer)
        self.assertIsNone(a._set)
        self.assertIs(a(None,None), Any)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)

        a = SetInitializer(Reals)
        self.assertIs(type(a), SetInitializer)
        self.assertIs(type(a._set), _ConstantInitializer)
        self.assertIs(a(None,None), Reals)
        self.assertIs(a._set.val, Reals)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)

        a = SetInitializer({1:Reals})
        self.assertIs(type(a), SetInitializer)
        self.assertIs(type(a._set), _ItemInitializer)
        self.assertIs(a(None, 1), Reals)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)

    def test_intersect(self):
        a = SetInitializer(None)
        a.intersect(SetInitializer(None))
        self.assertIs(type(a), SetInitializer)
        self.assertIsNone(a._set)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertIs(a(None,None), Any)

        a = SetInitializer(None)
        a.intersect(SetInitializer(Reals))
        self.assertIs(type(a), SetInitializer)
        self.assertIs(type(a._set), _ConstantInitializer)
        self.assertIs(a._set.val, Reals)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertIs(a(None,None), Reals)

        a = SetInitializer(None)
        a.intersect(RangeSetInitializer(5))
        self.assertIs(type(a), SetInitializer)
        self.assertIs(type(a._set), RangeSetInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(a(None,None), RangeSet(5))

        a = SetInitializer(Reals)
        a.intersect(SetInitializer(None))
        self.assertIs(type(a), SetInitializer)
        self.assertIs(type(a._set), _ConstantInitializer)
        self.assertIs(a._set.val, Reals)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertIs(a(None,None), Reals)

        a = SetInitializer(Reals)
        a.intersect(SetInitializer(Integers))
        self.assertIs(type(a), SetInitializer)
        self.assertIs(type(a._set), _SetIntersectInitializer)
        self.assertIs(type(a._set._A), _ConstantInitializer)
        self.assertIs(type(a._set._B), _ConstantInitializer)
        self.assertIs(a._set._A.val, Reals)
        self.assertIs(a._set._B.val, Integers)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertIs(type(s), SetIntersection_InfiniteSet)
        self.assertIs(s._sets[0], Reals)
        self.assertIs(s._sets[1], Integers)

        a = SetInitializer(Reals)
        a.intersect(SetInitializer(Integers))
        a.intersect(RangeSetInitializer(3))
        self.assertIs(type(a), SetInitializer)
        self.assertIs(type(a._set), _SetIntersectInitializer)
        self.assertIs(type(a._set._A), _SetIntersectInitializer)
        self.assertIs(type(a._set._B), RangeSetInitializer)
        self.assertIs(a._set._A._A.val, Reals)
        self.assertIs(a._set._A._B.val, Integers)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertIs(type(s), SetIntersection_OrderedSet)
        self.assertIs(type(s._sets[0]), SetIntersection_InfiniteSet)
        self.assertIsInstance(s._sets[1], RangeSet)

        a = SetInitializer(Reals)
        a.intersect(SetInitializer(Integers))
        a.intersect(RangeSetInitializer(3, default_step=0))
        self.assertIs(type(a), SetInitializer)
        self.assertIs(type(a._set), _SetIntersectInitializer)
        self.assertIs(type(a._set._A), _SetIntersectInitializer)
        self.assertIs(type(a._set._B), RangeSetInitializer)
        self.assertIs(a._set._A._A.val, Reals)
        self.assertIs(a._set._A._B.val, Integers)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertIs(type(s), SetIntersection_OrderedSet)
        self.assertIs(type(s._sets[0]), SetIntersection_InfiniteSet)
        self.assertIsInstance(s._sets[1], RangeSet)
        self.assertFalse(s._sets[0].is_finite())
        self.assertFalse(s._sets[1].is_finite())
        self.assertTrue(s.is_finite())

        a = SetInitializer(Reals)
        a.intersect(SetInitializer({1:Integers}))
        a.intersect(RangeSetInitializer(3, default_step=0))
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
        self.assertIs(type(s), SetIntersection_OrderedSet)
        self.assertIs(type(s._sets[0]), SetIntersection_InfiniteSet)
        self.assertIsInstance(s._sets[1], RangeSet)
        self.assertFalse(s._sets[0].is_finite())
        self.assertFalse(s._sets[1].is_finite())
        self.assertTrue(s.is_finite())

    def test_rangeset(self):
        a = RangeSetInitializer(5)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertEqual(s, RangeSet(5))

        a = RangeSetInitializer((0,5))
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertEqual(s, RangeSet(0,5))

        a = RangeSetInitializer((0,5,2))
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertEqual(s, RangeSet(0,5,2))

        a = RangeSetInitializer(5, default_step=0)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertEqual(s, RangeSet(1,5,0))

        a = RangeSetInitializer((0,5), default_step=0)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertEqual(s, RangeSet(0,5,0))

        a = RangeSetInitializer((0,5,2), default_step=0)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertEqual(s, RangeSet(0,5,2))

        a = RangeSetInitializer({1:5})
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        s = a(None,1)
        self.assertEqual(s, RangeSet(5))

        a = RangeSetInitializer({1:(0,5)})
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        s = a(None,1)
        self.assertEqual(s, RangeSet(0,5))

    def test_setdefault(self):
        a = SetInitializer(None)
        self.assertIs(a(None,None), Any)
        a.setdefault(Reals)
        self.assertIs(a(None,None), Reals)

        a = SetInitializer(Integers)
        self.assertIs(a(None,None), Integers)
        a.setdefault(Reals)
        self.assertIs(a(None,None), Integers)

        a = RangeSetInitializer(5)
        self.assertEqual(a(None,None), RangeSet(5))
        a.setdefault(Reals)
        self.assertEqual(a(None,None), RangeSet(5))

        a = SetInitializer(Reals)
        a.intersect(SetInitializer(Integers))
        self.assertIs(type(a(None,None)), SetIntersection_InfiniteSet)
        a.setdefault(RangeSet(5))
        self.assertIs(type(a(None,None)), SetIntersection_InfiniteSet)


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

    def test_name(self):
        self.assertEqual(str(Reals), 'Reals')
        self.assertEqual(str(Integers), 'Integers')


def _init_set(m, *args):
    n = 1
    for i in args:
        n *= i
    return xrange(n)


class TestSet(unittest.TestCase):
    def test_deprecated_args(self):
        m = ConcreteModel()
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            m.I = Set(virtual=True)
            self.assertEqual(len(m.I), 0)
        self.assertEqual(
            output.getvalue(),
            "DEPRECATED: Pyomo Sets ignore the 'virtual' keyword argument\n")

    def test_scalar_set_initialize_and_iterate(self):
        m = ConcreteModel()
        m.I = Set()
        self.assertEqual(list(m.I), [])
        self.assertEqual(list(reversed(m.I)), [])
        self.assertEqual(m.I.data(), ())
        self.assertEqual(m.I.dimen, _UnknownSetDimen)

        m = ConcreteModel()
        with self.assertRaisesRegexp(
                KeyError, "Cannot treat the scalar component 'I'"
                "as an indexed component"):
            m.I = Set(initialize={1:(1,3,2,4)})

        m = ConcreteModel()
        m.I = Set(initialize=(1,3,2,4))
        self.assertEqual(list(m.I), [1,3,2,4])
        self.assertEqual(list(reversed(m.I)), [4,2,3,1])
        self.assertEqual(m.I.data(), (1,3,2,4))
        self.assertEqual(m.I.dimen, 1)

        def I_init(m):
            yield 1
            yield 3
            yield 2
            yield 4
        m = ConcreteModel()
        m.I = Set(initialize=I_init)
        self.assertEqual(list(m.I), [1,3,2,4])
        self.assertEqual(list(reversed(m.I)), [4,2,3,1])
        self.assertEqual(m.I.data(), (1,3,2,4))
        self.assertEqual(m.I.dimen, 1)

        m = ConcreteModel()
        m.I = Set(initialize={None: (1,3,2,4)})
        self.assertEqual(list(m.I), [1,3,2,4])
        self.assertEqual(list(reversed(m.I)), [4,2,3,1])
        self.assertEqual(m.I.data(), (1,3,2,4))
        self.assertEqual(m.I.dimen, 1)

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            m = ConcreteModel()
            m.I = Set(initialize={1,3,2,4})
            ref = "Initializing an ordered Set with a " \
                  "fundamentally unordered data source (type: set)."
            self.assertIn(ref, output.getvalue())
        self.assertEqual(m.I.sorted_data(), (1,2,3,4))
        # We can't directly compare the reversed to a reference list
        # (because this is populated from an unordered set!) but we can
        # compare it with the forward list.
        self.assertEqual(list(reversed(list(m.I))), list(reversed(m.I)))
        self.assertEqual(list(reversed(m.I.data())), list(reversed(m.I)))
        self.assertEqual(m.I.dimen, 1)

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            m = ConcreteModel()
            m.I = Set(initialize={1,3,2,4}, ordered=False)
            self.assertEqual(output.getvalue(), "")
        self.assertEqual(list(m.I), [1,2,3,4])
        # We can't directly compare the reversed to a reference list
        # (because ithis is an unordered set!) but we can compare it with
        # the forward list.
        self.assertEqual(list(reversed(list(m.I))), list(reversed(m.I)))
        self.assertEqual(list(reversed(m.I.data())), list(reversed(m.I)))
        self.assertEqual(m.I.dimen, 1)

        m = ConcreteModel()
        m.I = Set(initialize=[1,3,2,4], ordered=Set.SortedOrder)
        self.assertEqual(list(m.I), [1,2,3,4])
        self.assertEqual(list(reversed(m.I)), [4,3,2,1])
        self.assertEqual(m.I.data(), (1,2,3,4))
        self.assertEqual(m.I.dimen, 1)

        with self.assertRaisesRegexp(
                TypeError, "Set 'ordered' argument is not valid \(must "
                "be one of {False, True, <function>, Set.InsertionOrder, "
                "Set.SortedOrder}\)"):
            m = ConcreteModel()
            m.I = Set(initialize=[1,3,2,4], ordered=Set)

        m = ConcreteModel()
        m.I = Set(initialize=[1,3,2,4], ordered=lambda x: reversed(sorted(x)))
        self.assertEqual(list(m.I), [4,3,2,1])
        self.assertEqual(list(reversed(m.I)), [1,2,3,4])
        self.assertEqual(m.I.data(), (4,3,2,1))
        self.assertEqual(m.I.dimen, 1)

    def test_insertion_deletion(self):
        def _verify(_s, _l):
            self.assertTrue(_s.is_ordered())
            self.assertTrue(_s.is_finite())
            for i,v in enumerate(_l):
                self.assertEqual(_s[i+1], v)
            with self.assertRaisesRegexp(IndexError, "I index out of range"):
                _s[len(_l)+1]
            with self.assertRaisesRegexp(IndexError, "I index out of range"):
                _s[len(_l)+2]

            for i,v in enumerate(reversed(_l)):
                self.assertEqual(_s[-(i+1)], v)
            with self.assertRaisesRegexp(IndexError, "I index out of range"):
                _s[-len(_l)-1]
            with self.assertRaisesRegexp(IndexError, "I index out of range"):
                _s[-len(_l)-2]

            for i,v in enumerate(_l):
                self.assertEqual(_s.ord(v), i+1)
                self.assertEqual(_s.ord((v,)), i+1)

            if _l:
                _max = max(_l)
                _min = min(_l)
            else:
                _max = 0
                _min = 0
            with self.assertRaisesRegexp(ValueError, "I.ord\(x\): x not in I"):
                m.I.ord(_max+1)
            with self.assertRaisesRegexp(ValueError, "I.ord\(x\): x not in I"):
                m.I.ord(_min-1)
            with self.assertRaisesRegexp(ValueError, "I.ord\(x\): x not in I"):
                m.I.ord((_max+1,))

        # Testing insertion order sets
        m = ConcreteModel()
        m.I = Set()
        _verify(m.I, [])
        m.I.add(1)
        _verify(m.I, [1])
        m.I.add(3)
        _verify(m.I, [1,3])
        m.I.add(2)
        _verify(m.I, [1,3,2])
        m.I.add(4)
        _verify(m.I, [1,3,2,4])

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            m.I.add(3)
        self.assertEqual(
            output.getvalue(),
            "Element 3 already exists in Set I; no action taken\n")
        _verify(m.I, [1,3,2,4])

        m.I.remove(3)
        _verify(m.I, [1,2,4])

        with self.assertRaisesRegexp(KeyError, "^3$"):
            m.I.remove(3)
        _verify(m.I, [1,2,4])

        m.I.add(3)
        _verify(m.I, [1,2,4,3])

        m.I.discard(3)
        _verify(m.I, [1,2,4])

        m.I.discard(3)
        _verify(m.I, [1,2,4])

        m.I.clear()
        _verify(m.I, [])

        m.I.add(6)
        m.I.add(5)
        _verify(m.I, [6,5])

        tmp = set()
        tmp.add(m.I.pop())
        tmp.add(m.I.pop())
        _verify(m.I, [])
        self.assertEqual(tmp, {5,6})
        with self.assertRaisesRegexp(KeyError, 'pop from an empty set'):
            m.I.pop()

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            m.I.update([6])
            _verify(m.I, [6])
            m.I.update([6,5,6])
            _verify(m.I, [6,5])

            m.I = [0,-1,1]
            _verify(m.I, [0,-1,1])

            self.assertEqual(output.getvalue(), "")

            # Assing unsorted data should generate warnings
            m.I.update({3,4})
            self.assertIn(
                "Calling update() on an insertion order Set with a "
                "fundamentally unordered data source (type: set)",
                output.getvalue()
            )
            self.assertEqual(set(m.I), {0, -1, 1, 3, 4})
            output.truncate(0)

            m.I = {5,6}
            self.assertIn(
                "Calling set_value() on an insertion order Set with a "
                "fundamentally unordered data source (type: set)",
                output.getvalue()
            )
            self.assertEqual(set(m.I), {5,6})

        # Testing sorted sets
        m = ConcreteModel()
        m.I = Set(ordered=Set.SortedOrder)
        _verify(m.I, [])
        m.I.add(1)
        _verify(m.I, [1])
        m.I.add(3)
        _verify(m.I, [1,3])
        m.I.add(2)
        _verify(m.I, [1,2,3])
        m.I.add(4)
        _verify(m.I, [1,2,3,4])

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            m.I.add(3)
        self.assertEqual(
            output.getvalue(),
            "Element 3 already exists in Set I; no action taken\n")
        _verify(m.I, [1,2,3,4])

        m.I.remove(3)
        _verify(m.I, [1,2,4])

        with self.assertRaisesRegexp(KeyError, "^3$"):
            m.I.remove(3)
        _verify(m.I, [1,2,4])

        m.I.add(3)
        _verify(m.I, [1,2,3,4])

        m.I.discard(3)
        _verify(m.I, [1,2,4])

        m.I.discard(3)
        _verify(m.I, [1,2,4])

        m.I.clear()
        _verify(m.I, [])

        m.I.add(6)
        m.I.add(5)
        _verify(m.I, [5,6])

        tmp = set()
        tmp.add(m.I.pop())
        tmp.add(m.I.pop())
        _verify(m.I, [])
        self.assertEqual(tmp, {5,6})
        with self.assertRaisesRegexp(KeyError, 'pop from an empty set'):
            m.I.pop()

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            m.I.update([6])
            _verify(m.I, [6])
            m.I.update([6,5,6])
            _verify(m.I, [5,6])

            m.I = [0,-1,1]
            _verify(m.I, [-1,0,1])

            self.assertEqual(output.getvalue(), "")

            # Assing unsorted data should not generate warnings (since
            # we are sorting the Set!)
            m.I.update({3,4})
            self.assertEqual(output.getvalue(), "")
            _verify(m.I, [-1,0,1,3,4])

            m.I = {5,6}
            self.assertEqual(output.getvalue(), "")
            _verify(m.I, [5,6])

    def test_unordered_insertion_deletion(self):
        def _verify(_s, _l):
            self.assertFalse(_s.is_ordered())
            self.assertTrue(_s.is_finite())

            self.assertEqual(sorted(_s), _l)
            self.assertEqual(list(_s), list(reversed(list(reversed(_s)))))
            for v in _l:
                self.assertIn(v, _s)

            if _l:
                _max = max(_l)
                _min = min(_l)
            else:
                _max = 0
                _min = 0
            self.assertNotIn(_max+1, _s)
            self.assertNotIn(_min-1, _s)

        # Testing unordered sets
        m = ConcreteModel()
        m.I = Set(ordered=False)
        _verify(m.I, [])
        m.I.add(1)
        _verify(m.I, [1])
        m.I.add(3)
        _verify(m.I, [1,3])
        m.I.add(2)
        _verify(m.I, [1,2,3])
        m.I.add(4)
        _verify(m.I, [1,2,3,4])

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            m.I.add(3)
        self.assertEqual(
            output.getvalue(),
            "Element 3 already exists in Set I; no action taken\n")
        _verify(m.I, [1,2,3,4])

        m.I.remove(3)
        _verify(m.I, [1,2,4])

        with self.assertRaisesRegexp(KeyError, "^3$"):
            m.I.remove(3)
        _verify(m.I, [1,2,4])

        m.I.add(3)
        _verify(m.I, [1,2,3,4])

        m.I.discard(3)
        _verify(m.I, [1,2,4])

        m.I.discard(3)
        _verify(m.I, [1,2,4])

        m.I.clear()
        _verify(m.I, [])

        m.I.add(6)
        m.I.add(5)
        _verify(m.I, [5,6])

        tmp = set()
        tmp.add(m.I.pop())
        tmp.add(m.I.pop())
        _verify(m.I, [])
        self.assertEqual(tmp, {5,6})
        with self.assertRaisesRegexp(KeyError, 'pop from an empty set'):
            m.I.pop()

        m.I.update([5])
        _verify(m.I, [5])
        m.I.update([6,5])
        _verify(m.I, [5,6])

        m.I = [0,-1,1]
        _verify(m.I, [-1,0,1])

    def test_indexed_set(self):
        # Implicit construction
        m = ConcreteModel()
        m.I = Set([1,2,3], ordered=False)
        self.assertEqual(len(m.I), 3)
        m.I[1].add(1)
        m.I[2].add(2)
        m.I[3].add(4)
        self.assertEqual(list(m.I[1]), [1])
        self.assertEqual(list(m.I[2]), [2])
        self.assertEqual(list(m.I[3]), [4])
        self.assertIsNot(m.I[1], m.I[2])
        self.assertIsNot(m.I[1], m.I[3])
        self.assertIsNot(m.I[2], m.I[3])
        self.assertFalse(m.I[1].is_ordered())
        self.assertFalse(m.I[2].is_ordered())
        self.assertFalse(m.I[3].is_ordered())
        self.assertIs(type(m.I[1]), _FiniteSetData)
        self.assertIs(type(m.I[2]), _FiniteSetData)
        self.assertIs(type(m.I[3]), _FiniteSetData)

        # Explicit (constant) construction
        m = ConcreteModel()
        m.I = Set([1,2,3], initialize=(4,2,5))
        self.assertEqual(len(m.I), 3)
        self.assertEqual(list(m.I[1]), [4,2,5])
        self.assertEqual(list(m.I[2]), [4,2,5])
        self.assertEqual(list(m.I[3]), [4,2,5])
        self.assertIsNot(m.I[1], m.I[2])
        self.assertIsNot(m.I[1], m.I[3])
        self.assertIsNot(m.I[2], m.I[3])
        self.assertTrue(m.I[1].is_ordered())
        self.assertTrue(m.I[2].is_ordered())
        self.assertTrue(m.I[3].is_ordered())
        self.assertIs(type(m.I[1]), _InsertionOrderSetData)
        self.assertIs(type(m.I[2]), _InsertionOrderSetData)
        self.assertIs(type(m.I[3]), _InsertionOrderSetData)

        # Explicit (constant) construction
        m = ConcreteModel()
        m.I = Set([1,2,3], initialize=(4,2,5), ordered=Set.SortedOrder)
        self.assertEqual(len(m.I), 3)
        self.assertEqual(list(m.I[1]), [2,4,5])
        self.assertEqual(list(m.I[2]), [2,4,5])
        self.assertEqual(list(m.I[3]), [2,4,5])
        self.assertIsNot(m.I[1], m.I[2])
        self.assertIsNot(m.I[1], m.I[3])
        self.assertIsNot(m.I[2], m.I[3])
        self.assertTrue(m.I[1].is_ordered())
        self.assertTrue(m.I[2].is_ordered())
        self.assertTrue(m.I[3].is_ordered())
        self.assertIs(type(m.I[1]), _SortedSetData)
        self.assertIs(type(m.I[2]), _SortedSetData)
        self.assertIs(type(m.I[3]), _SortedSetData)

    def test_naming(self):
        m = ConcreteModel()

        i = Set()
        self.assertEqual(str(i), "OrderedSimpleSet")
        i.construct()
        self.assertEqual(str(i), "{}")
        m.I = i
        self.assertEqual(str(i), "I")

        j = Set(initialize=[1,2,3])
        self.assertEqual(str(j), "OrderedSimpleSet")
        j.construct()
        self.assertEqual(str(j), "{1, 2, 3}")
        m.J = j
        self.assertEqual(str(j), "J")

        k = Set([1,2,3])
        self.assertEqual(str(k), "IndexedSet")
        with self.assertRaisesRegexp(
                ValueError, 'The component has not been constructed.'):
            str(k[1])
        m.K = k
        self.assertEqual(str(k), "K")
        self.assertEqual(str(k[1]), "K[1]")

    def test_indexing(self):
        m = ConcreteModel()
        m.I = Set()
        m.I = [1, 3, 2]
        self.assertEqual(m.I[2], 3)
        with self.assertRaisesRegexp(
                IndexError, "I indices must be integers, not float"):
            m.I[2.5]
        with self.assertRaisesRegexp(
                IndexError, "I indices must be integers, not str"):
            m.I['a']

    def test_add_filter_validate(self):
        m = ConcreteModel()
        m.I = Set(domain=Integers)
        with self.assertRaisesRegexp(
                ValueError,
                "Cannot add value 1.5 to Set I.\n"
                "\tThe value is not in the domain Integers"):
            m.I.add(1.5)

        # Open question: should we cast the added value into the domain
        # (if we do, how?)
        self.assertTrue( m.I.add(1.0) )
        self.assertIn(1, m.I)
        self.assertIn(1., m.I)

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            self.assertFalse( m.I.add(1) )
        self.assertEquals(
            output.getvalue(),
            "Element 1 already exists in Set I; no action taken\n")

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            self.assertFalse( m.I.add((1,)) )
        self.assertEquals(
            output.getvalue(),
            "Element (1,) already exists in Set I; no action taken\n")

        m.J = Set()
        with self.assertRaisesRegexp(
                TypeError,
                "Unable to insert '\[1\]' into Set J:\n"
                "\tTypeError: unhashable type: 'list'"):
            m.J.add([1])

        self.assertTrue( m.J.add((1,)) )
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            self.assertFalse( m.J.add(1) )
        self.assertEquals(
            output.getvalue(),
            "Element 1 already exists in Set J; no action taken\n")


        def _l_tri(m, i, j):
            return i >= j
        m.K = Set(initialize=RangeSet(3)*RangeSet(3), filter=_l_tri)
        self.assertEqual(
            list(m.K), [(1,1), (2,1), (2,2), (3,1), (3,2), (3,3)])

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            self.assertTrue( m.K.add((0,0)) )
            self.assertFalse( m.K.add((0,1)) )
        self.assertEquals(output.getvalue(), "")
        self.assertEqual(
            list(m.K), [(1,1), (2,1), (2,2), (3,1), (3,2), (3,3), (0,0)])

        # This tests a filter that matches the dimentionality of the
        # component.  construct() needs to recognize that the filter is
        # returning a constant in construct() and re-assign it to be the
        # _filter for each _SetData
        def _lt_3(m, i):
            return i < 3
        m.L = Set([1,2,3,4,5], initialize=RangeSet(10), filter=_lt_3)
        self.assertEqual(len(m.L), 5)
        self.assertEqual(list(m.L[1]), [1, 2])
        self.assertEqual(list(m.L[5]), [1, 2])

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            self.assertTrue( m.L[2].add(0) )
            self.assertFalse( m.L[2].add((100)) )
        self.assertEquals(output.getvalue(), "")
        self.assertEqual(list(m.L[2]), [1,2,0])


        def _validate(m,i,j):
            if i + j < 2:
                return True
            if i - j > 2:
                return False
            raise RuntimeError("Bogus value")
        m = ConcreteModel()
        m.I = Set(validate=_validate)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            self.assertTrue( m.I.add((0,1)) )
            self.assertEqual(output.getvalue(), "")
            with self.assertRaisesRegexp(
                    ValueError,
                    "The value=\(4, 1\) violates the validation rule of Set I"):
                m.I.add((4,1))
            self.assertEqual(output.getvalue(), "")
            with self.assertRaisesRegexp(RuntimeError, "Bogus value"):
                m.I.add((2,2))
        self.assertEqual(
            output.getvalue(),
            "Exception raised while validating element '(2, 2)' for Set I\n")

        # Note: one of these indices will trigger the exception in the
        # validot when it is called for the index.
        m.J = Set([(0,0), (2,2)], validate=_validate)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            self.assertTrue( m.J[2,2].add((0,1)) )
            self.assertEqual(output.getvalue(), "")
            with self.assertRaisesRegexp(
                    ValueError,
                    "The value=\(4, 1\) violates the validation rule of "
                    "Set J\[0,0\]"):
                m.J[0,0].add((4,1))
            self.assertEqual(output.getvalue(), "")
            with self.assertRaisesRegexp(RuntimeError, "Bogus value"):
                m.J[2,2].add((2,2))
        self.assertEqual(
            output.getvalue(),
            "Exception raised while validating element '(2, 2)' for "
            "Set J[2,2]\n")

    def test_domain(self):
        m = ConcreteModel()
        m.I = Set(domain=Integers)
        m.I.add(1)
        m.I.add(2.)
        self.assertEqual(list(m.I), [1, 2.])
        with self.assertRaisesRegexp(
                ValueError, 'The value is not in the domain Integers'):
            m.I.add(1.5)

        m = ConcreteModel()
        m.I = Set(within=Integers)
        m.I.add(1)
        m.I.add(2.)
        self.assertEqual(list(m.I), [1, 2.])
        with self.assertRaisesRegexp(
                ValueError, 'The value is not in the domain Integers'):
            m.I.add(1.5)

        m = ConcreteModel()
        m.I = Set(bounds=(1,5))
        m.I.add(1)
        m.I.add(2.)
        self.assertEqual(list(m.I), [1, 2.])
        with self.assertRaisesRegexp(
                ValueError, 'The value is not in the domain \[1..5\]'):
            m.I.add(5.5)

        m = ConcreteModel()
        m.I = Set(domain=Integers, within=RangeSet(0, None, 2), bounds=(0,9))
        m.I = [0,2.,4]
        self.assertEqual(list(m.I), [0,2.,4])
        with self.assertRaisesRegexp(
                ValueError, 'The value is not in the domain '
                '\(Integers & \[0:None:2\]\) & \[0..9\]'):
            m.I.add(1.5)
        with self.assertRaisesRegexp(
                ValueError, 'The value is not in the domain '
                '\(Integers & \[0:None:2\]\) & \[0..9\]'):
            m.I.add(1)
        with self.assertRaisesRegexp(
                ValueError, 'The value is not in the domain '
                '\(Integers & \[0:None:2\]\) & \[0..9\]'):
            m.I.add(10)


    def test_pprint(self):
        def myFcn(x):
            return reversed(sorted(x))

        m = ConcreteModel()
        m.I_index = RangeSet(3)
        m.I = Set(m.I_index, initialize=lambda m,i: xrange(i+1),
                  domain=Integers)
        m.J = Set(ordered=False)
        m.K = Set(initialize=[(1,2), (3,4)], ordered=Set.SortedOrder)
        m.L = Set(initialize=[(1,2), (3,4)], ordered=myFcn)

        buf = StringIO()
        m.pprint()
        m.pprint(ostream=buf)
        self.assertEqual(buf.getvalue().strip(), """
1 RangeSet Declarations
    I_index : Dimen=1, Size=3, Bounds=(1, 3)
        Key  : Finite : Members
        None :   True :   [1:3]

4 Set Declarations
    I : Size=3, Index=I_index, Ordered=Insertion
        Key : Dimen : Domain   : Size : Members
          1 :     1 : Integers :    2 : {0, 1}
          2 :     1 : Integers :    3 : {0, 1, 2}
          3 :     1 : Integers :    4 : {0, 1, 2, 3}
    J : Size=1, Index=None, Ordered=False
        Key  : Dimen : Domain : Size : Members
        None :    -- :    Any :    0 :      {}
    K : Size=1, Index=None, Ordered=Sorted
        Key  : Dimen : Domain : Size : Members
        None :     2 :    Any :    2 : {(1, 2), (3, 4)}
    L : Size=1, Index=None, Ordered={user}
        Key  : Dimen : Domain : Size : Members
        None :     2 :    Any :    2 : {(3, 4), (1, 2)}

5 Declarations: I_index I J K L""".strip())

    def test_pickle(self):
        m = ConcreteModel()
        m.I = Set(initialize={1, 2, 'a'}, ordered=False)
        m.J = Set(initialize=(2,4,1))
        m.K = Set(initialize=(2,4,1), ordered=Set.SortedOrder)
        m.II = Set([1,2,3], m.J, initialize=_init_set)

        buf = StringIO()
        m.pprint(ostream=buf)
        ref = buf.getvalue()

        n = pickle.loads(pickle.dumps(m))

        self.assertIsNot(m, n)
        self.assertIsNot(m.I, n.I)
        self.assertIsNot(m.J, n.J)
        self.assertIsNot(m.K, n.K)
        self.assertIsNot(m.II, n.II)

        self.assertEqual(m.I, n.I)
        self.assertEqual(m.J, n.J)
        self.assertEqual(m.K, n.K)
        for i in m.II:
            self.assertEqual(m.II[i], n.II[i])

        buf = StringIO()
        n.pprint(ostream=buf)
        self.assertEqual(ref, buf.getvalue())

    def test_dimen(self):
        m = ConcreteModel()
        m.I = Set()
        self.assertEqual(m.I.dimen, _UnknownSetDimen)
        m.I.add((1,2))
        self.assertEqual(m.I.dimen, 2)

        m.J = Set(initialize=[1,2,3])
        self.assertEqual(m.J.dimen, 1)

        m.K = Set(initialize=[(1,2,3)])
        self.assertEqual(m.K.dimen, 3)

        with self.assertRaisesRegexp(
                ValueError,
                "The value=1 has dimension 1 and is not valid for Set K "
                "which has dimen=3"):
            m.K.add(1)

        m.L = Set(dimen=None)
        self.assertIsNone(m.L.dimen)
        m.L.add(1)
        self.assertIsNone(m.L.dimen)
        m.L.add((2,3))
        self.assertIsNone(m.L.dimen)
        self.assertEqual(list(m.L), [1, (2,3)])

    def test_construction(self):
        m = AbstractModel()
        m.I = Set(initialize=[1,2,3])
        m.J = Set(initialize=[4,5,6])
        m.II = Set([1,2,3], initialize={1:[0], 2:[1,2], 3: xrange(3)})
        m.JJ = Set([1,2,3], initialize={1:[0], 2:[1,2], 3: xrange(3)})

        output = StringIO()
        m.pprint()
        m.I.pprint(ostream=output)
        m.II.pprint(ostream=output)
        m.J.pprint(ostream=output)
        m.JJ.pprint(ostream=output)
        ref = """
I : Size=0, Index=None, Ordered=Insertion
    Not constructed
II : Size=0, Index=II_index, Ordered=Insertion
    Not constructed
J : Size=0, Index=None, Ordered=Insertion
    Not constructed
JJ : Size=0, Index=JJ_index, Ordered=Insertion
    Not constructed""".strip()
        self.assertEqual(output.getvalue().strip(), ref)

        i = m.create_instance(data={
            None: {'I': [-1,0], 'II': {1: [10,11], 3:[30]}}
        })

        self.assertEqual(list(i.I), [-1,0])
        self.assertEqual(list(i.J), [4,5,6])
        self.assertEqual(list(i.II[1]), [10,11])
        self.assertEqual(list(i.II[3]), [30])
        self.assertEqual(list(i.JJ[1]), [0])
        self.assertEqual(list(i.JJ[2]), [1,2])
        self.assertEqual(list(i.JJ[3]), [0,1,2])

        # Implicitly-constructed set should fall back on initialize!
        self.assertEqual(list(i.II[2]), [1,2])

        ref = """
Constructing OrderedSimpleSet 'I' on [Model] from data=None
Constructing Set, name=I, from data=None
Constructed component ''[Model].I'':
I : Size=1, Index=None, Ordered=Insertion
    Key  : Dimen : Domain : Size : Members
    None :    -- :    Any :    0 :      {}
""".strip()
        m = ConcreteModel()
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core', logging.DEBUG):
            m.I = Set()
            print output.getvalue()
            self.assertEqual(output.getvalue().strip(), ref)
            # but re-constructing the set doesn't re-log the message
            m.I.construct()
            self.assertEqual(output.getvalue().strip(), ref)

        # Test generators
        m = ConcreteModel()
        def _i_init(m):
            yield 1
            yield 3
            yield 2
        m.I = Set(initialize=_i_init)
        self.assertEqual(list(m.I), [1,3,2])

        m = ConcreteModel()
        def _i_init(m):
            yield 1
            yield 3
            yield Set.End
            yield 2
        m.I = Set(initialize=_i_init)
        self.assertEqual(list(m.I), [1,3])

        m = ConcreteModel()
        m.I = Set(initialize=[1,3,Set.End,2])
        self.assertEqual(list(m.I), [1,3])

    def test_set_end(self):
        # Tested counted initialization
        m = ConcreteModel()
        def _i_init(m, i):
            if i < 5:
                return 2*i
            return Set.End
        m.I = Set(initialize=_i_init)
        self.assertEqual(list(m.I), [2,4,6,8])

        m = ConcreteModel()
        def _i_init(m, i, j):
            if i < j:
                return 2*i
            return Set.End
        m.I = Set([1,2,3], initialize=_i_init)
        self.assertEqual(list(m.I[1]), [])
        self.assertEqual(list(m.I[2]), [2])
        self.assertEqual(list(m.I[3]), [2,4])

        m = ConcreteModel()
        def _i_init(m, i, j, k):
            if i < j+k:
                return 2*i
            return Set.End
        m.I = Set([1,2], [2,3], initialize=_i_init)
        self.assertEqual(list(m.I[1,2]), [2,4])
        self.assertEqual(list(m.I[1,3]), [2,4,6])
        self.assertEqual(list(m.I[2,2]), [2,4,6])
        self.assertEqual(list(m.I[2,3]), [2,4,6,8])

        m = ConcreteModel()
        def _i_init(m, i):
            if i > 3:
                return None
            return i
        with self.assertRaisesRegexp(
                ValueError, "Set rule returned None instead of Set.End"):
            m.I1 = Set(initialize=_i_init)
        @simple_set_rule
        def _j_init(m, i):
            if i > 3:
                return None
            return i
        m.J = Set(initialize=_j_init)
        self.assertEqual(list(m.J), [1,2,3])

    def test_set_skip(self):
        # Test Set.Skip
        m = ConcreteModel()
        def _i_init(m,i):
            if i % 2:
                return Set.Skip
            return range(i)
        m.I = Set([1,2,3,4,5], initialize=_i_init)
        self.assertEqual(len(m.I), 2)
        self.assertIn(2, m.I)
        self.assertEqual(list(m.I[2]), [0,1])
        self.assertIn(4, m.I)
        self.assertEqual(list(m.I[4]), [0,1,2,3])
        self.assertNotIn(1, m.I)
        self.assertNotIn(3, m.I)
        self.assertNotIn(5, m.I)
        output = StringIO()
        m.I.pprint(ostream=output)
        print output.getvalue()
        ref = """
I : Size=2, Index=I_index, Ordered=Insertion
    Key : Dimen : Domain : Size : Members
      2 :     1 :    Any :    2 : {0, 1}
      4 :     1 :    Any :    4 : {0, 1, 2, 3}
""".strip()
        self.assertEqual(output.getvalue().strip(), ref.strip())

        m = ConcreteModel()
        def _i_init(m,i):
            if i % 2:
                return None
            return range(i)
        with self.assertRaisesRegexp(
                ValueError,
                "Set rule or initializer returned None instead of Set.Skip"):
            m.I = Set([1,2,3,4,5], initialize=_i_init)

        def _j_init(m):
            return None
        with self.assertRaisesRegexp(
                ValueError,
                "Set rule or initializer returned None instead of Set.Skip"):
            m.J = Set(initialize=_j_init)

    def test_sorted_operations(self):
        I = Set(ordered=Set.SortedOrder, initialize=[0])
        I.construct()
        i = 0
        self.assertTrue(i in I)
        self.assertFalse(I._is_sorted)
        I._sort()
        self.assertTrue(I._is_sorted)

        # adding a value already in the set does not affect _is_sorted
        self.assertTrue(I._is_sorted)
        self.assertFalse(I.add(i))
        self.assertTrue(I._is_sorted)

        # adding a new value clears _is_sorted
        self.assertTrue(I._is_sorted)
        self.assertTrue(I.add(1))
        self.assertFalse(I._is_sorted)

        # __str__
        i += 1
        I.update((i, -i))
        self.assertFalse(I._is_sorted)
        self.assertEqual(
            str(I), "{%s}" % ', '.join(str(_) for _ in range(-i,i+1)))
        self.assertTrue(I._is_sorted)

        # ranges()
        i += 1
        I.update((i, -i))
        self.assertFalse(I._is_sorted)
        self.assertEqual(
            ','.join(str(_) for _ in I.ranges()),
            ','.join('[%s]' % _ for _ in range(-i,i+1))
        )
        self.assertTrue(I._is_sorted)

        # __iter__
        i += 1
        I.update((i, -i))
        self.assertFalse(I._is_sorted)
        self.assertEqual(
            ','.join(str(_) for _ in I),
            ','.join(str(_) for _ in range(-i,i+1))
        )
        self.assertTrue(I._is_sorted)

        # __reversed__
        i += 1
        I.update((i, -i))
        self.assertFalse(I._is_sorted)
        self.assertEqual(
            ','.join(str(_) for _ in reversed(I)),
            ','.join(str(_) for _ in reversed(range(-i,i+1)))
        )
        self.assertTrue(I._is_sorted)

        # data()
        i += 1
        I.update((i, -i))
        self.assertFalse(I._is_sorted)
        self.assertEqual(
            ','.join(str(_) for _ in I.data()),
            ','.join(str(_) for _ in range(-i,i+1))
        )
        self.assertTrue(I._is_sorted)

        # ordered_data()
        i += 1
        I.update((i, -i))
        self.assertFalse(I._is_sorted)
        self.assertEqual(
            ','.join(str(_) for _ in I.ordered_data()),
            ','.join(str(_) for _ in range(-i,i+1))
        )
        self.assertTrue(I._is_sorted)

        # sorted_data()
        i += 1
        I.update((i, -i))
        self.assertFalse(I._is_sorted)
        self.assertEqual(
            ','.join(str(_) for _ in I.sorted_data()),
            ','.join(str(_) for _ in range(-i,i+1))
        )
        self.assertTrue(I._is_sorted)

        # bounds()
        i += 1
        I.update((i, -i))
        self.assertFalse(I._is_sorted)
        self.assertEqual(I.bounds(), (-i,i))
        self.assertTrue(I._is_sorted)

        # remove()
        I.remove(0)
        self.assertTrue(I._is_sorted)
        self.assertEqual(
            ','.join(str(_) for _ in I),
            ','.join(str(_) for _ in range(-i,i+1) if _ != 0)
        )
        self.assertTrue(I._is_sorted)

        # add()
        I.add(0)
        self.assertFalse(I._is_sorted)
        self.assertEqual(
            ','.join(str(_) for _ in I),
            ','.join(str(_) for _ in range(-i,i+1))
        )
        self.assertTrue(I._is_sorted)

        # discard()
        I.discard(0)
        self.assertTrue(I._is_sorted)
        self.assertEqual(
            ','.join(str(_) for _ in I),
            ','.join(str(_) for _ in range(-i,i+1) if _ != 0)
        )
        self.assertTrue(I._is_sorted)

        # clear()
        I.clear()
        self.assertTrue(I._is_sorted)
        self.assertEqual(','.join(str(_) for _ in I), '')
        self.assertTrue(I._is_sorted)

        # set_value()
        i = 1
        I.set_value({-i,0,i})
        self.assertFalse(I._is_sorted)
        self.assertEqual(
            ','.join(str(_) for _ in I),
            ','.join(str(_) for _ in range(-i,i+1))
        )
        self.assertTrue(I._is_sorted)

        # pop()
        i += 1
        I.update((i, -i))
        self.assertFalse(I._is_sorted)
        self.assertEqual(I.pop(), i)
        self.assertTrue(I._is_sorted)

        # first()
        i += 1
        I.update((i, -i))
        self.assertFalse(I._is_sorted)
        self.assertEqual(I.first(), -i)
        self.assertTrue(I._is_sorted)

        # last()
        i += 1
        I.update((i, -i))
        self.assertFalse(I._is_sorted)
        self.assertEqual(I.last(), i)
        self.assertTrue(I._is_sorted)

        # next()
        i += 1
        I.update((i, -i))
        self.assertFalse(I._is_sorted)
        self.assertEqual(I.next(-i), -i+1)
        self.assertTrue(I._is_sorted)

        # nextw()
        i += 1
        I.update((i, -i))
        self.assertFalse(I._is_sorted)
        self.assertEqual(I.nextw(i), -i)
        self.assertTrue(I._is_sorted)

        # prev()
        i += 1
        I.update((i, -i))
        self.assertFalse(I._is_sorted)
        self.assertEqual(I.prev(i), i-1)
        self.assertTrue(I._is_sorted)

        # prevw()
        i += 1
        I.update((i, -i))
        self.assertFalse(I._is_sorted)
        self.assertEqual(I.prevw(-i), i)
        self.assertTrue(I._is_sorted)

        # getitem()
        i += 1
        I.update((i, -i))
        self.assertFalse(I._is_sorted)
        self.assertEqual(I[i+1], 0)
        self.assertTrue(I._is_sorted)

        # ord()
        i += 1
        I.update((i, -i))
        self.assertFalse(I._is_sorted)
        self.assertEqual(I.ord(0), i+1)
        self.assertTrue(I._is_sorted)

    def test_set_options(self):
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            @set_options(domain=Integers)
            def Bindex(m):
                return range(5)
        self.assertIn(
            "DEPRECATED: The set_options decorator seems nonessential",
            output.getvalue())

        m = ConcreteModel()
        m.I = Set(initialize=[8,9])
        m.J = m.I.cross(Bindex)
        self.assertIs(m.J._sets[1]._domain, Integers)

        # TODO: Once this is merged into IndexedContainer, the following
        # should work
        #
        #m.K = Set(Bindex)
        #self.assertIs(m.K.index_set()._domain, Integers)

    def test_no_normalize_index(self):
        try:
            _oldFlatten = normalize_index.flatten

            normalize_index.flatten = False
            m = ConcreteModel()
            m.I = Set()
            self.assertIs(m.I._dimen, _UnknownSetDimen)
            m.I.add((1,(2,3)))
            self.assertIs(m.I._dimen, None)
            self.assertIn((1,(2,3)), m.I)
            self.assertNotIn((1,2,3), m.I)

            normalize_index.flatten = True
            m = ConcreteModel()
            m.I = Set()
            self.assertIs(m.I._dimen, _UnknownSetDimen)
            m.I.add((1,(2,3)))
            self.assertIs(m.I._dimen, 3)
            self.assertNotIn((1,(2,3)), m.I)
            self.assertIn((1,2,3), m.I)
        finally:
            normalize_index.flatten = _oldFlatten

class TestAbstractSetAPI(unittest.TestCase):
    def test_SetData(self):
        # This tests an anstract non-finite set

        m = ConcreteModel()
        m.I = Set(initialize=[1])
        s = _SetData(m.I)

        #
        # _SetData API
        #

        with self.assertRaises(DeveloperError):
            # __contains__
            None in s

        self.assertFalse(s == m.I)
        self.assertFalse(m.I == s)
        self.assertTrue(s != m.I)
        self.assertTrue(m.I != s)

        with self.assertRaises(DeveloperError):
            str(s)
        with self.assertRaises(DeveloperError):
            s.dimen

        self.assertFalse(s.is_finite())
        self.assertFalse(s.is_ordered())

        with self.assertRaises(DeveloperError):
            s.ranges()

        with self.assertRaises(DeveloperError):
            s.isdisjoint(m.I)
        with self.assertRaises(DeveloperError):
            m.I.isdisjoint(s)

        with self.assertRaises(DeveloperError):
            s.issuperset(m.I)
        self.assertFalse(m.I.issuperset(s))

        self.assertFalse(s.issubset(m.I))
        with self.assertRaises(DeveloperError):
            m.I.issubset(s)

        self.assertIs(type(s.union(m.I)), SetUnion_InfiniteSet)
        self.assertIs(type(m.I.union(s)), SetUnion_InfiniteSet)

        self.assertIs(type(s.intersection(m.I)), SetIntersection_OrderedSet)
        self.assertIs(type(m.I.intersection(s)), SetIntersection_OrderedSet)

        self.assertIs(type(s.difference(m.I)), SetDifference_InfiniteSet)
        self.assertIs(type(m.I.difference(s)), SetDifference_OrderedSet)

        self.assertIs(type(s.symmetric_difference(m.I)),
                      SetSymmetricDifference_InfiniteSet)
        self.assertIs(type(m.I.symmetric_difference(s)),
                      SetSymmetricDifference_InfiniteSet)

        self.assertIs(type(s.cross(m.I)), SetProduct_InfiniteSet)
        self.assertIs(type(m.I.cross(s)), SetProduct_InfiniteSet)

        self.assertIs(type(s | m.I), SetUnion_InfiniteSet)
        self.assertIs(type(m.I | s), SetUnion_InfiniteSet)

        self.assertIs(type(s & m.I), SetIntersection_OrderedSet)
        self.assertIs(type(m.I & s), SetIntersection_OrderedSet)

        self.assertIs(type(s - m.I), SetDifference_InfiniteSet)
        self.assertIs(type(m.I - s), SetDifference_OrderedSet)

        self.assertIs(type(s ^ m.I), SetSymmetricDifference_InfiniteSet)
        self.assertIs(type(m.I ^ s), SetSymmetricDifference_InfiniteSet)

        self.assertIs(type(s * m.I), SetProduct_InfiniteSet)
        self.assertIs(type(m.I * s), SetProduct_InfiniteSet)

        self.assertFalse(s < m.I)
        with self.assertRaises(DeveloperError):
            self.assertFalse(m.I < s)

        with self.assertRaises(DeveloperError):
            self.assertFalse(s > m.I)
        self.assertFalse(m.I > s)

    def test_FiniteMixin(self):
        # This tests an anstract finite set
        class FiniteMixin(_FiniteSetMixin, _SetData):
            pass

        m = ConcreteModel()
        m.I = Set(initialize=[1])
        s = FiniteMixin(m.I)

        #
        # _SetData API
        #

        with self.assertRaises(DeveloperError):
            # __contains__
            None in s

        with self.assertRaises(DeveloperError):
            self.assertFalse(s == m.I)
        with self.assertRaises(DeveloperError):
            self.assertFalse(m.I == s)
        with self.assertRaises(DeveloperError):
            self.assertTrue(s != m.I)
        with self.assertRaises(DeveloperError):
            self.assertTrue(m.I != s)

        with self.assertRaises(DeveloperError):
            str(s)
        with self.assertRaises(DeveloperError):
            s.dimen

        self.assertTrue(s.is_finite())
        self.assertFalse(s.is_ordered())

        range_iter = s.ranges()
        with self.assertRaises(DeveloperError):
            list(range_iter)

        with self.assertRaises(DeveloperError):
            s.isdisjoint(m.I)
        with self.assertRaises(DeveloperError):
            m.I.isdisjoint(s)

        with self.assertRaises(DeveloperError):
            s.issuperset(m.I)
        with self.assertRaises(DeveloperError):
            self.assertFalse(m.I.issuperset(s))

        with self.assertRaises(DeveloperError):
            self.assertFalse(s.issubset(m.I))
        with self.assertRaises(DeveloperError):
            m.I.issubset(s)

        self.assertIs(type(s.union(m.I)), SetUnion_FiniteSet)
        self.assertIs(type(m.I.union(s)), SetUnion_FiniteSet)

        self.assertIs(type(s.intersection(m.I)), SetIntersection_OrderedSet)
        self.assertIs(type(m.I.intersection(s)), SetIntersection_OrderedSet)

        self.assertIs(type(s.difference(m.I)), SetDifference_FiniteSet)
        self.assertIs(type(m.I.difference(s)), SetDifference_OrderedSet)

        self.assertIs(type(s.symmetric_difference(m.I)),
                      SetSymmetricDifference_FiniteSet)
        self.assertIs(type(m.I.symmetric_difference(s)),
                      SetSymmetricDifference_FiniteSet)

        self.assertIs(type(s.cross(m.I)), SetProduct_FiniteSet)
        self.assertIs(type(m.I.cross(s)), SetProduct_FiniteSet)

        self.assertIs(type(s | m.I), SetUnion_FiniteSet)
        self.assertIs(type(m.I | s), SetUnion_FiniteSet)

        self.assertIs(type(s & m.I), SetIntersection_OrderedSet)
        self.assertIs(type(m.I & s), SetIntersection_OrderedSet)

        self.assertIs(type(s - m.I), SetDifference_FiniteSet)
        self.assertIs(type(m.I - s), SetDifference_OrderedSet)

        self.assertIs(type(s ^ m.I), SetSymmetricDifference_FiniteSet)
        self.assertIs(type(m.I ^ s), SetSymmetricDifference_FiniteSet)

        self.assertIs(type(s * m.I), SetProduct_FiniteSet)
        self.assertIs(type(m.I * s), SetProduct_FiniteSet)


        with self.assertRaises(DeveloperError):
            self.assertFalse(s < m.I)
        with self.assertRaises(DeveloperError):
            self.assertFalse(m.I < s)

        with self.assertRaises(DeveloperError):
            self.assertFalse(s > m.I)
        with self.assertRaises(DeveloperError):
            self.assertFalse(m.I > s)

        #
        # _FiniteSetMixin API
        #

        with self.assertRaises(DeveloperError):
            len(s)

        with self.assertRaises(DeveloperError):
            # __iter__
            iter(s)

        with self.assertRaises(DeveloperError):
            # __reversed__
            reversed(s)

        with self.assertRaises(DeveloperError):
            s.data()

        with self.assertRaises(DeveloperError):
            s.ordered_data()

        with self.assertRaises(DeveloperError):
            s.sorted_data()

        self.assertEqual(s.bounds(), (None,None))

    def test_OrderedMixin(self):
        # This tests an anstract finite set
        class OrderedMixin(_OrderedSetMixin, _FiniteSetMixin, _SetData):
            pass

        m = ConcreteModel()
        m.I = Set(initialize=[1])
        s = OrderedMixin(m.I)

        #
        # _SetData API
        #

        with self.assertRaises(DeveloperError):
            # __contains__
            None in s

        with self.assertRaises(DeveloperError):
            self.assertFalse(s == m.I)
        with self.assertRaises(DeveloperError):
            self.assertFalse(m.I == s)
        with self.assertRaises(DeveloperError):
            self.assertTrue(s != m.I)
        with self.assertRaises(DeveloperError):
            self.assertTrue(m.I != s)

        with self.assertRaises(DeveloperError):
            str(s)
        with self.assertRaises(DeveloperError):
            s.dimen

        self.assertTrue(s.is_finite())
        self.assertTrue(s.is_ordered())

        range_iter = s.ranges()
        with self.assertRaises(DeveloperError):
            list(range_iter)

        with self.assertRaises(DeveloperError):
            s.isdisjoint(m.I)
        with self.assertRaises(DeveloperError):
            m.I.isdisjoint(s)

        with self.assertRaises(DeveloperError):
            s.issuperset(m.I)
        with self.assertRaises(DeveloperError):
            self.assertFalse(m.I.issuperset(s))

        with self.assertRaises(DeveloperError):
            self.assertFalse(s.issubset(m.I))
        with self.assertRaises(DeveloperError):
            m.I.issubset(s)

        self.assertIs(type(s.union(m.I)), SetUnion_OrderedSet)
        self.assertIs(type(m.I.union(s)), SetUnion_OrderedSet)

        self.assertIs(type(s.intersection(m.I)), SetIntersection_OrderedSet)
        self.assertIs(type(m.I.intersection(s)), SetIntersection_OrderedSet)

        self.assertIs(type(s.difference(m.I)), SetDifference_OrderedSet)
        self.assertIs(type(m.I.difference(s)), SetDifference_OrderedSet)

        self.assertIs(type(s.symmetric_difference(m.I)),
                      SetSymmetricDifference_OrderedSet)
        self.assertIs(type(m.I.symmetric_difference(s)),
                      SetSymmetricDifference_OrderedSet)

        self.assertIs(type(s.cross(m.I)), SetProduct_OrderedSet)
        self.assertIs(type(m.I.cross(s)), SetProduct_OrderedSet)

        self.assertIs(type(s | m.I), SetUnion_OrderedSet)
        self.assertIs(type(m.I | s), SetUnion_OrderedSet)

        self.assertIs(type(s & m.I), SetIntersection_OrderedSet)
        self.assertIs(type(m.I & s), SetIntersection_OrderedSet)

        self.assertIs(type(s - m.I), SetDifference_OrderedSet)
        self.assertIs(type(m.I - s), SetDifference_OrderedSet)

        self.assertIs(type(s ^ m.I), SetSymmetricDifference_OrderedSet)
        self.assertIs(type(m.I ^ s), SetSymmetricDifference_OrderedSet)

        self.assertIs(type(s * m.I), SetProduct_OrderedSet)
        self.assertIs(type(m.I * s), SetProduct_OrderedSet)


        with self.assertRaises(DeveloperError):
            self.assertFalse(s < m.I)
        with self.assertRaises(DeveloperError):
            self.assertFalse(m.I < s)

        with self.assertRaises(DeveloperError):
            self.assertFalse(s > m.I)
        with self.assertRaises(DeveloperError):
            self.assertFalse(m.I > s)

        #
        # _FiniteSetMixin API
        #

        with self.assertRaises(DeveloperError):
            len(s)

        with self.assertRaises(DeveloperError):
            # __iter__
            iter(s)

        with self.assertRaises(DeveloperError):
            # __reversed__
            reversed(s)

        with self.assertRaises(DeveloperError):
            s.data()

        with self.assertRaises(DeveloperError):
            s.ordered_data()

        with self.assertRaises(DeveloperError):
            s.sorted_data()

        self.assertEqual(s.bounds(), (None,None))

        #
        # _OrderedSetMixin API
        #

        with self.assertRaises(DeveloperError):
            s.first()

        with self.assertRaises(DeveloperError):
            s.last()

        with self.assertRaises(DeveloperError):
            s.next(0)

        with self.assertRaises(DeveloperError):
            s.nextw(0)

        with self.assertRaises(DeveloperError):
            s.prev(0)

        with self.assertRaises(DeveloperError):
            s.prevw(0)

        with self.assertRaises(DeveloperError):
            s[0]

        with self.assertRaises(DeveloperError):
            s.ord(0)

class TestIssues(unittest.TestCase):
    def test_issue_43(self):
        model = ConcreteModel()
        model.Jobs = Set(initialize=[0,1,2,3])
        model.Dummy = Set(model.Jobs, within=model.Jobs,
                          initialize=lambda m,i: range(i))
        model.Cars = Set(initialize=['a','b'])

        a = model.Cars * model.Dummy[1]
        self.assertEqual(len(a), 2)
        self.assertIn(('a', 0), a)
        self.assertIn(('b', 0), a)

        b = model.Dummy[2] * model.Cars
        self.assertEqual(len(b), 4)
        self.assertIn((0, 'a'), b)
        self.assertIn((0, 'b'), b)
        self.assertIn((1, 'a'), b)
        self.assertIn((1, 'b'), b)

    def test_issue_116(self):
        m = ConcreteModel()
        m.s = Set(initialize=['one'])
        m.t = Set([1], initialize=['one'])
        m.x = Var(m.s)
        self.assertFalse(m.s in m.s)
        with self.assertRaisesRegexp(KeyError, "Index 's' is not valid"):
            m.x[m.s].display()
        self.assertEqual(list(m.x), ['one'])

    def test_issue_121(self):
        model = ConcreteModel()
        model.s = Set(initialize=[1,2,3])
        self.assertEqual(list(model.s), [1,2,3])
        model.s = [3,9]
        self.assertEqual(list(model.s), [3,9])

    def test_issue_134(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1,2])
        m.J = Set(initialize=[4,5])
        m.IJ = m.I * m.J
        self.assertEqual(len(m.IJ), 4)
        self.assertEqual(m.IJ.dimen, 2)
        m.IJ2 = m.IJ * m.IJ
        self.assertEqual(len(m.IJ2), 16)
        self.assertEqual(m.IJ2.dimen, 4)
        self.assertEqual(len(m.IJ), 4)
        self.assertEqual(m.IJ.dimen, 2)

    def test_issue_142(self):
        CHOICES = [((1,2,3), 4,3), ((1,2,2), 4,3), ((1,3,3), 4,3)]

        try:
            _oldFlatten = normalize_index.flatten

            normalize_index.flatten = False
            m = ConcreteModel()
            output = StringIO()
            with LoggingIntercept(output, 'pyomo.core'):
                m.CHOICES = Set(initialize=CHOICES, dimen=3)
                self.assertIn('Ignoring non-None dimen (3) for set CHOICES',
                              output.getvalue())

            self.assertEqual(m.CHOICES.dimen, None)
            m.x = Var(m.CHOICES)
            def c_rule(m, a, b, c):
                return m.x[a,b,c] == 0
            m.c = Constraint(m.CHOICES, rule=c_rule)
            output = StringIO()
            m.CHOICES.pprint(ostream=output)
            m.x.pprint(ostream=output)
            m.c.pprint(ostream=output)
            print output.getvalue()
            ref="""
CHOICES : Size=1, Index=None, Ordered=Insertion
    Key  : Dimen : Domain : Size : Members
    None :  None :    Any :    3 : {((1, 2, 3), 4, 3), ((1, 2, 2), 4, 3), ((1, 3, 3), 4, 3)}
x : Size=3, Index=CHOICES
    Key               : Lower : Value : Upper : Fixed : Stale : Domain
    ((1, 2, 2), 4, 3) :  None :  None :  None : False :  True :  Reals
    ((1, 2, 3), 4, 3) :  None :  None :  None : False :  True :  Reals
    ((1, 3, 3), 4, 3) :  None :  None :  None : False :  True :  Reals
c : Size=3, Index=CHOICES, Active=True
    Key               : Lower : Body           : Upper : Active
    ((1, 2, 2), 4, 3) :   0.0 : x[(1,2,2),4,3] :   0.0 :   True
    ((1, 2, 3), 4, 3) :   0.0 : x[(1,2,3),4,3] :   0.0 :   True
    ((1, 3, 3), 4, 3) :   0.0 : x[(1,3,3),4,3] :   0.0 :   True
""".strip()
            self.assertEqual(output.getvalue().strip(), ref)

            normalize_index.flatten = True
            m = ConcreteModel()
            output = StringIO()
            with LoggingIntercept(output, 'pyomo.core'):
                m.CHOICES = Set(initialize=CHOICES)
                self.assertEqual('',output.getvalue())
            self.assertEqual(m.CHOICES.dimen, 5)
            m.x = Var(m.CHOICES)
            def c_rule(m, a1, a2, a3, b, c):
                return m.x[a1,a2,a3,b,c] == 0
            m.c = Constraint(m.CHOICES, rule=c_rule)

            output = StringIO()
            m.CHOICES.pprint(ostream=output)
            m.x.pprint(ostream=output)
            m.c.pprint(ostream=output)
            print output.getvalue()
            ref="""
CHOICES : Size=1, Index=None, Ordered=Insertion
    Key  : Dimen : Domain : Size : Members
    None :     5 :    Any :    3 : {(1, 2, 3, 4, 3), (1, 2, 2, 4, 3), (1, 3, 3, 4, 3)}
x : Size=3, Index=CHOICES
    Key             : Lower : Value : Upper : Fixed : Stale : Domain
    (1, 2, 2, 4, 3) :  None :  None :  None : False :  True :  Reals
    (1, 2, 3, 4, 3) :  None :  None :  None : False :  True :  Reals
    (1, 3, 3, 4, 3) :  None :  None :  None : False :  True :  Reals
c : Size=3, Index=CHOICES, Active=True
    Key             : Lower : Body         : Upper : Active
    (1, 2, 2, 4, 3) :   0.0 : x[1,2,2,4,3] :   0.0 :   True
    (1, 2, 3, 4, 3) :   0.0 : x[1,2,3,4,3] :   0.0 :   True
    (1, 3, 3, 4, 3) :   0.0 : x[1,3,3,4,3] :   0.0 :   True
""".strip()
            self.assertEqual(output.getvalue().strip(), ref)
        finally:
            normalize_index.flatten = _oldFlatten

    def test_issue_148(self):
        legal = set(['a','b','c'])
        m = ConcreteModel()
        m.s = Set(initialize=['a','b'], within=legal)
        self.assertEqual(set(m.s), {'a','b'})
        with self.assertRaisesRegexp(ValueError, 'Cannot add value d to Set s'):
            m.s.add('d')
