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
import itertools
import logging
import pickle
from six import StringIO, PY2
from six.moves import xrange
from collections import namedtuple as NamedTuple

try:
    from typing import NamedTuple
except ImportError:
    NamedTuple = None

import pyutilib.th as unittest

from pyomo.common import DeveloperError
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import pandas as pd, pandas_available
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import native_numeric_types, native_types
import pyomo.core.base.set as SetModule
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.util import (
    ConstantInitializer, ItemInitializer, IndexedCallInitializer,
)
from pyomo.core.base.set import (
    NumericRange as NR, NonNumericRange as NNR,
    AnyRange, _AnySet, Any, AnyWithNone, _EmptySet, EmptySet, Binary,
    Reals, NonNegativeReals, PositiveReals, NonPositiveReals, NegativeReals,
    Integers, PositiveIntegers, NegativeIntegers,
    NonNegativeIntegers,
    Set,
    SetOf, OrderedSetOf, UnorderedSetOf,
    RangeSet, _FiniteRangeSetData, _InfiniteRangeSetData,
    FiniteSimpleRangeSet, InfiniteSimpleRangeSet,
    AbstractFiniteSimpleRangeSet, 
    SetUnion_InfiniteSet, SetUnion_FiniteSet, SetUnion_OrderedSet,
    SetIntersection_InfiniteSet, SetIntersection_FiniteSet,
    SetIntersection_OrderedSet,
    SetDifference_InfiniteSet, SetDifference_FiniteSet,
    SetDifference_OrderedSet,
    SetSymmetricDifference_InfiniteSet, SetSymmetricDifference_FiniteSet,
    SetSymmetricDifference_OrderedSet,
    SetProduct, SetProduct_InfiniteSet, SetProduct_FiniteSet,
    SetProduct_OrderedSet,
    _SetData, _FiniteSetData, _InsertionOrderSetData, _SortedSetData,
    _FiniteSetMixin, _OrderedSetMixin,
    SetInitializer, SetIntersectInitializer, BoundsInitializer,
    UnknownSetDimen, UnindexedComponent_set,
    DeclareGlobalSet, IntegerSet, RealSet,
    simple_set_rule, set_options,
 )
from pyomo.environ import (
    AbstractModel, ConcreteModel, Block, Var, Param, Suffix, Constraint,
    Objective,
)


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
        self.assertIs(type(a._set), ConstantInitializer)
        self.assertIs(a(None,None), Reals)
        self.assertIs(a._set.val, Reals)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)

        a = SetInitializer({1:Reals})
        self.assertIs(type(a), SetInitializer)
        self.assertIs(type(a._set), ItemInitializer)
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
        self.assertIs(type(a._set), ConstantInitializer)
        self.assertIs(a._set.val, Reals)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertIs(a(None,None), Reals)

        a = SetInitializer(None)
        a.intersect(BoundsInitializer(5, default_step=1))
        self.assertIs(type(a), SetInitializer)
        self.assertIs(type(a._set), BoundsInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(a(None,None), RangeSet(5))

        a = SetInitializer(Reals)
        a.intersect(SetInitializer(None))
        self.assertIs(type(a), SetInitializer)
        self.assertIs(type(a._set), ConstantInitializer)
        self.assertIs(a._set.val, Reals)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertIs(a(None,None), Reals)

        a = SetInitializer(Reals)
        a.intersect(SetInitializer(Integers))
        self.assertIs(type(a), SetInitializer)
        self.assertIs(type(a._set), SetIntersectInitializer)
        self.assertIs(type(a._set._A), ConstantInitializer)
        self.assertIs(type(a._set._B), ConstantInitializer)
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
        a.intersect(BoundsInitializer(3, default_step=1))
        self.assertIs(type(a), SetInitializer)
        self.assertIs(type(a._set), SetIntersectInitializer)
        self.assertIs(type(a._set._A), SetIntersectInitializer)
        self.assertIs(type(a._set._B), BoundsInitializer)
        self.assertIs(a._set._A._A.val, Reals)
        self.assertIs(a._set._A._B.val, Integers)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertIs(type(s), SetIntersection_OrderedSet)
        self.assertIs(type(s._sets[0]), SetIntersection_InfiniteSet)
        self.assertIsInstance(s._sets[1], RangeSet)

        p = Param(initialize=3)
        a = SetInitializer(Reals)
        a.intersect(SetInitializer(Integers))
        a.intersect(BoundsInitializer(p, default_step=0))
        self.assertIs(type(a), SetInitializer)
        self.assertIs(type(a._set), SetIntersectInitializer)
        self.assertIs(type(a._set._A), SetIntersectInitializer)
        self.assertIs(type(a._set._B), BoundsInitializer)
        self.assertIs(a._set._A._A.val, Reals)
        self.assertIs(a._set._A._B.val, Integers)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertIs(type(s), SetIntersection_InfiniteSet)
        p.construct()
        s.construct()
        self.assertIs(type(s), SetIntersection_OrderedSet)
        self.assertIs(type(s._sets[0]), SetIntersection_InfiniteSet)
        self.assertIsInstance(s._sets[1], RangeSet)
        self.assertFalse(s._sets[0].isfinite())
        self.assertFalse(s._sets[1].isfinite())
        self.assertTrue(s.isfinite())

        p = Param(initialize=3)
        a = SetInitializer(Reals)
        a.intersect(SetInitializer({1:Integers}))
        a.intersect(BoundsInitializer(p, default_step=0))
        self.assertIs(type(a), SetInitializer)
        self.assertIs(type(a._set), SetIntersectInitializer)
        self.assertIs(type(a._set._A), SetIntersectInitializer)
        self.assertIs(type(a._set._B), BoundsInitializer)
        self.assertIs(a._set._A._A.val, Reals)
        self.assertIs(type(a._set._A._B), ItemInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        with self.assertRaises(KeyError):
            a(None,None)
        s = a(None,1)
        self.assertIs(type(s), SetIntersection_InfiniteSet)
        p.construct()
        s.construct()
        self.assertIs(type(s), SetIntersection_OrderedSet)
        self.assertIs(type(s._sets[0]), SetIntersection_InfiniteSet)
        self.assertIsInstance(s._sets[1], RangeSet)
        self.assertFalse(s._sets[0].isfinite())
        self.assertFalse(s._sets[1].isfinite())
        self.assertTrue(s.isfinite())

    def test_boundsinit(self):
        a = BoundsInitializer(5, default_step=1)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertEqual(s, RangeSet(5))

        a = BoundsInitializer((0,5), default_step=1)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertEqual(s, RangeSet(0,5))

        a = BoundsInitializer((0,5,2))
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertEqual(s, RangeSet(0,5,2))

        a = BoundsInitializer(())
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertEqual(s, RangeSet(None,None,0))

        a = BoundsInitializer(5)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertEqual(s, RangeSet(1,5,0))

        a = BoundsInitializer((0,5))
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertEqual(s, RangeSet(0,5,0))

        a = BoundsInitializer((0,5,2))
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        s = a(None,None)
        self.assertEqual(s, RangeSet(0,5,2))

        a = BoundsInitializer({1:5}, default_step=1)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        s = a(None,1)
        self.assertEqual(s, RangeSet(5))

        a = BoundsInitializer({1:(0,5)}, default_step=1)
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

        a = BoundsInitializer(5, default_step=1)
        self.assertEqual(a(None,None), RangeSet(5))
        a.setdefault(Reals)
        self.assertEqual(a(None,None), RangeSet(5))

        a = SetInitializer(Reals)
        a.intersect(SetInitializer(Integers))
        self.assertIs(type(a(None,None)), SetIntersection_InfiniteSet)
        a.setdefault(RangeSet(5))
        self.assertIs(type(a(None,None)), SetIntersection_InfiniteSet)

    def test_indices(self):
        a = SetInitializer(None)
        self.assertFalse(a.contains_indices())
        with self.assertRaisesRegex(
                RuntimeError, 'does not contain embedded indices'):
            a.indices()

        a = SetInitializer([1,2,3])
        self.assertFalse(a.contains_indices())
        with self.assertRaisesRegex(
                RuntimeError, 'does not contain embedded indices'):
            a.indices()

        # intersection initializers
        a = SetInitializer({1: [1,2,3], 2: [4]})
        self.assertTrue(a.contains_indices())
        self.assertEqual(list(a.indices()), [1,2])

        a.intersect(SetInitializer({1: [4], 2: [1,2]}))
        self.assertTrue(a.contains_indices())
        self.assertEqual(list(a.indices()), [1,2])

        # intersection initializer mismatch
        a = SetInitializer({1: [1,2,3], 2: [4]})
        self.assertTrue(a.contains_indices())
        self.assertEqual(list(a.indices()), [1,2])

        a.intersect(SetInitializer({1: [4], 3: [1,2]}))
        self.assertTrue(a.contains_indices())
        with self.assertRaisesRegex(
                ValueError, 'contains two sub-initializers with inconsistent'):
            a.indices()

        # intersection initializer mismatch (unindexed)
        a = SetInitializer([1,2])
        self.assertFalse(a.contains_indices())
        a.intersect(SetInitializer([1,2]))
        self.assertFalse(a.contains_indices())
        with self.assertRaisesRegex(
                RuntimeError, 'does not contain embedded indices'):
            a.indices()


class InfiniteSetTester(unittest.TestCase):
    def test_Reals(self):
        self.assertIn(0, Reals)
        self.assertIn(1.5, Reals)
        self.assertIn(100, Reals),
        self.assertIn(-100, Reals),
        self.assertNotIn('A', Reals)
        self.assertNotIn(None, Reals)

        self.assertFalse(Reals.isdiscrete())
        self.assertFalse(Reals.isfinite())

        self.assertEqual(Reals.dim(), 0)
        self.assertIs(Reals.index_set(), UnindexedComponent_set)
        with self.assertRaisesRegex(
                TypeError, ".*'GlobalSet' has no len"):
            len(Reals)
        with self.assertRaisesRegex(
                TypeError, "'GlobalSet' object is not iterable "
                "\(non-finite Set 'Reals' is not iterable\)"):
            list(Reals)
        self.assertEqual(list(Reals.ranges()), [NR(None,None,0)])
        self.assertEqual(Reals.bounds(), (None,None))
        self.assertEqual(Reals.dimen, 1)

        tmp = RealSet()
        self.assertFalse(tmp.isdiscrete())
        self.assertFalse(tmp.isfinite())
        self.assertEqual(Reals, tmp)
        self.assertEqual(tmp, Reals)
        tmp.clear()
        self.assertEqual(EmptySet, tmp)
        self.assertEqual(tmp, EmptySet)

        self.assertEqual(tmp.domain, Reals)
        self.assertEqual(str(Reals), 'Reals')
        self.assertEqual(str(tmp), 'Reals')
        b = ConcreteModel()
        b.tmp = tmp
        self.assertEqual(str(tmp), 'tmp')

    def test_Integers(self):
        self.assertIn(0, Integers)
        self.assertNotIn(1.5, Integers)
        self.assertIn(100, Integers),
        self.assertIn(-100, Integers),
        self.assertNotIn('A', Integers)
        self.assertNotIn(None, Integers)

        self.assertTrue(Integers.isdiscrete())
        self.assertFalse(Integers.isfinite())

        self.assertEqual(Integers.dim(), 0)
        self.assertIs(Integers.index_set(), UnindexedComponent_set)
        with self.assertRaisesRegex(
                TypeError, ".*'GlobalSet' has no len"):
            len(Integers)
        with self.assertRaisesRegex(
                TypeError, "'GlobalSet' object is not iterable "
                "\(non-finite Set 'Integers' is not iterable\)"):
            list(Integers)
        self.assertEqual(list(Integers.ranges()), [NR(0,None,1),NR(0,None,-1)])
        self.assertEqual(Integers.bounds(), (None,None))
        self.assertEqual(Integers.dimen, 1)

        tmp = IntegerSet()
        self.assertTrue(tmp.isdiscrete())
        self.assertFalse(tmp.isfinite())
        self.assertEqual(Integers, tmp)
        self.assertEqual(tmp, Integers)
        tmp.clear()
        self.assertEqual(EmptySet, tmp)
        self.assertEqual(tmp, EmptySet)

        self.assertEqual(tmp.domain, Reals)
        self.assertEqual(str(Integers), 'Integers')
        self.assertEqual(str(tmp), 'Integers')
        b = ConcreteModel()
        b.tmp = tmp
        self.assertEqual(str(tmp), 'tmp')

    def test_Any(self):
        self.assertIn(0, Any)
        self.assertIn(1.5, Any)
        self.assertIn(100, Any),
        self.assertIn(-100, Any),
        self.assertIn('A', Any)
        self.assertIn(None, Any)

        self.assertFalse(Any.isdiscrete())
        self.assertFalse(Any.isfinite())

        self.assertEqual(Any.dim(), 0)
        self.assertIs(Any.index_set(), UnindexedComponent_set)
        with self.assertRaisesRegex(
                TypeError, ".*'Any' has no len"):
            len(Any)
        with self.assertRaisesRegex(
                TypeError, "'GlobalSet' object is not iterable "
                "\(non-finite Set 'Any' is not iterable\)"):
            list(Any)
        self.assertEqual(list(Any.ranges()), [AnyRange()])
        self.assertEqual(Any.bounds(), (None,None))
        self.assertEqual(Any.dimen, None)

        tmp = _AnySet()
        self.assertFalse(tmp.isdiscrete())
        self.assertFalse(tmp.isfinite())
        self.assertEqual(Any, tmp)
        tmp.clear()
        self.assertEqual(Any, tmp)

        self.assertEqual(tmp.domain, Any)
        self.assertEqual(str(Any), 'Any')
        self.assertEqual(str(tmp), '_AnySet')
        b = ConcreteModel()
        b.tmp = tmp
        self.assertEqual(str(tmp), 'tmp')

    def test_AnyWithNone(self):
        os = StringIO()
        with LoggingIntercept(os, 'pyomo'):
            self.assertIn(None, AnyWithNone)
            self.assertIn(1, AnyWithNone)
        self.assertRegexpMatches(
            os.getvalue(),
            "^DEPRECATED: The AnyWithNone set is deprecated")

        self.assertEqual(Any, AnyWithNone)
        self.assertEqual(AnyWithNone, Any)

    def test_EmptySet(self):
        self.assertNotIn(0, EmptySet)
        self.assertNotIn(1.5, EmptySet)
        self.assertNotIn(100, EmptySet),
        self.assertNotIn(-100, EmptySet),
        self.assertNotIn('A', EmptySet)
        self.assertNotIn(None, EmptySet)

        self.assertTrue(EmptySet.isdiscrete())
        self.assertTrue(EmptySet.isfinite())

        self.assertEqual(EmptySet.dim(), 0)
        self.assertIs(EmptySet.index_set(), UnindexedComponent_set)
        self.assertEqual(len(EmptySet), 0)
        self.assertEqual(list(EmptySet), [])
        self.assertEqual(list(EmptySet.ranges()), [])
        self.assertEqual(EmptySet.bounds(), (None,None))
        self.assertEqual(EmptySet.dimen, 0)

        tmp = _EmptySet()
        self.assertTrue(tmp.isdiscrete())
        self.assertTrue(tmp.isfinite())
        self.assertEqual(EmptySet, tmp)
        tmp.clear()
        self.assertEqual(EmptySet, tmp)

        self.assertEqual(tmp.domain, EmptySet)
        self.assertEqual(str(EmptySet), 'EmptySet')
        self.assertEqual(str(tmp), '_EmptySet')
        b = ConcreteModel()
        b.tmp = tmp
        self.assertEqual(str(tmp), 'tmp')

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

        # Special case: cleared non-finite rangesets
        tmp = IntegerSet()
        tmp.clear()
        self.assertTrue(tmp.issubset(EmptySet))
        self.assertTrue(tmp.issuperset(EmptySet))
        self.assertTrue(tmp.isdisjoint(EmptySet))

        self.assertTrue(EmptySet.issubset(tmp))
        self.assertTrue(EmptySet.issuperset(tmp))
        self.assertTrue(EmptySet.isdisjoint(tmp))


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
        i = SetOf([1,2,3])
        self.assertIs(type(i), OrderedSetOf)
        j = OrderedSetOf([1,2,3])
        self.assertIs(type(i), OrderedSetOf)
        self.assertEqual(i, j)

        i = SetOf({1,2,3})
        self.assertIs(type(i), UnorderedSetOf)
        j = UnorderedSetOf([1,2,3])
        self.assertIs(type(i), UnorderedSetOf)
        self.assertEqual(i, j)

        i = RangeSet(3)
        self.assertTrue(i.is_constructed())
        self.assertEqual(len(i), 3)
        self.assertEqual(len(list(i.ranges())), 1)

        i = RangeSet(1,3)
        self.assertTrue(i.is_constructed())
        self.assertEqual(len(i), 3)
        self.assertEqual(len(list(i.ranges())), 1)

        i = RangeSet(ranges=[NR(1,3,1)])
        self.assertTrue(i.is_constructed())
        self.assertEqual(len(i), 3)
        self.assertEqual(list(i.ranges()), [NR(1,3,1)])

        i = RangeSet(1,3,0)
        with self.assertRaisesRegexp(
                TypeError, ".*'InfiniteSimpleRangeSet' has no len"):
            len(i)
        self.assertEqual(len(list(i.ranges())), 1)

        with self.assertRaisesRegexp(
                TypeError, ".*'GlobalSet' has no len"):
            len(Integers)
        self.assertEqual(len(list(Integers.ranges())), 2)

        with self.assertRaisesRegexp(
                ValueError, "RangeSet expects 3 or fewer positional "
                "arguments \(received 4\)"):
            RangeSet(1,2,3,4)

        with self.assertRaisesRegexp(
                TypeError, "'ranges' argument must be an iterable of "
                "NumericRange objects"):
            RangeSet(ranges=(NR(1,5,1), NNR('a')))

        with self.assertRaisesRegexp(
                ValueError, "Constructing a finite RangeSet over a "
                "non-finite range "):
            RangeSet(finite=True, ranges=(NR(1,5,0),))

        with self.assertRaisesRegexp(
                ValueError, "RangeSet does not support unbounded ranges "
                "with a non-integer step"):
            RangeSet(0,None,0.5)

        class _AlmostNumeric(object):
            def __init__(self, val):
                self.val = val
            def __float__(self):
                return self.val
            def __add__(self, other):
                return self.val+other
            def __sub__(self, other):
                return self.val-other

        i = RangeSet(_AlmostNumeric(1))
        self.assertFalse(i.is_constructed())
        i.construct()
        self.assertEqual(list(i), [1])

        output = StringIO()
        p = Param(initialize=5)
        i = RangeSet(p)
        self.assertFalse(i.is_constructed())
        self.assertIs(type(i), AbstractFiniteSimpleRangeSet)
        p.construct()
        with LoggingIntercept(output, 'pyomo.core', logging.DEBUG):
            self.assertEqual(output.getvalue(), "")
            i.construct()
            ref = 'Constructing RangeSet, '\
                  'name=FiniteSimpleRangeSet, from data=None\n'
            self.assertEqual(output.getvalue(), ref)
            self.assertTrue(i.is_constructed())
            self.assertIs(type(i), FiniteSimpleRangeSet)
            # Calling construct() twice bypasses construction the second
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
            # Calling construct() twice bypasses construction the second
            # time around
            i.construct()
            self.assertEqual(output.getvalue(), ref)

        i = RangeSet(0)
        self.assertEqual(len(i), 0)
        self.assertEqual(len(list(i.ranges())), 0)

        # Special case: we do not error when the constructing a 0-length
        # RangeSetwith bounds (i, i-1)
        i = RangeSet(0,-1)
        self.assertEqual(len(i), 0)
        self.assertEqual(len(list(i.ranges())), 0)

        # Test non-finite RangeSets
        i = RangeSet(1,10)
        self.assertIs(type(i), FiniteSimpleRangeSet)
        i = RangeSet(1,10,0)
        self.assertIs(type(i), InfiniteSimpleRangeSet)
        i = RangeSet(1,1,0)
        self.assertIs(type(i), FiniteSimpleRangeSet)
        j = RangeSet(1, float('inf'))
        self.assertIs(type(j), InfiniteSimpleRangeSet)
        i = RangeSet(1,None)
        self.assertIs(type(i), InfiniteSimpleRangeSet)
        self.assertEqual(i,j)
        self.assertIn(1, i)
        self.assertIn(100, i)
        self.assertNotIn(0, i)
        self.assertNotIn(1.5, i)
        i = RangeSet(None,1)
        self.assertIs(type(i), InfiniteSimpleRangeSet)
        self.assertIn(1, i)
        self.assertNotIn(100, i)
        self.assertIn(0, i)
        self.assertNotIn(0.5, i)
        i = RangeSet(None,None)
        self.assertIs(type(i), InfiniteSimpleRangeSet)
        self.assertIn(1, i)
        self.assertIn(100, i)
        self.assertIn(0, i)
        self.assertNotIn(0.5, i)

        i = RangeSet(None,None,bounds=(-5,10))
        self.assertIs(type(i), InfiniteSimpleRangeSet)
        self.assertIn(10, i)
        self.assertNotIn(11, i)
        self.assertIn(-5, i)
        self.assertNotIn(-6, i)
        self.assertNotIn(0.5, i)

        p = Param(initialize=float('inf'))
        i = RangeSet(1, p, 1)
        self.assertIs(type(i), AbstractFiniteSimpleRangeSet)
        p.construct()
        i = RangeSet(1, p, 1)
        self.assertIs(type(i), InfiniteSimpleRangeSet)


        # Test abstract RangeSets
        m = AbstractModel()
        m.p = Param()
        m.q = Param()
        m.s = Param()
        m.i = RangeSet(m.p, m.q, m.s, finite=True)
        self.assertIs(type(m.i), AbstractFiniteSimpleRangeSet)
        i = m.create_instance(
            data={None: {'p': {None: 1}, 'q': {None: 5}, 's': {None: 2}}})
        self.assertIs(type(i.i), FiniteSimpleRangeSet)
        self.assertEqual(list(i.i), [1,3,5])

        with self.assertRaisesRegexp(
                ValueError,
                "finite RangeSet over a non-finite range \(\[1..5\]\)"):
            i = m.create_instance(
                data={None: {'p': {None: 1}, 'q': {None: 5}, 's': {None: 0}}})

        with self.assertRaisesRegexp(
                ValueError,
                "RangeSet.construct\(\) does not support the data= argument."):
            i = m.create_instance(
                data={None: {'p': {None: 1}, 'q': {None: 5}, 's': {None: 1},
                             'i': {None: [1,2,3]} }})

    def test_filter(self):
        def rFilter(m, i):
            return i % 2
        # Simple filter (beginning with the *first* element)
        r = RangeSet(10, filter=rFilter)
        self.assertEqual(r, [1,3,5,7,9])

        # Nothing to remove
        r = RangeSet(1, filter=rFilter)
        self.assertEqual(r, [1])

        # Remove the only element in the range
        r = RangeSet(2,2, filter=rFilter)
        self.assertEqual(r, [])

        # remove the *second* element in the range
        r = RangeSet(2,3, filter=rFilter)
        self.assertEqual(r, [3])

        # Test a filter that doesn't raise an exception for "None"
        def rFilter(m, i):
            return i is None or i % 2
        r = RangeSet(10, filter=rFilter)
        self.assertEqual(r, [1,3,5,7,9])

        with self.assertRaisesRegexp(
                ValueError, "The 'filter' keyword argument is not "
                "valid for non-finite RangeSet component"):
            r = RangeSet(1,10,0, filter=rFilter)

    def test_validate(self):
        def rFilter(m, i):
            self.assertIs(m, None)
            return i % 2
        # Simple validation
        r = RangeSet(1,10,2, validate=rFilter)
        self.assertEqual(r, [1,3,5,7,9])

        # Failed validation
        with self.assertRaisesRegexp(
                ValueError, "The value=2 violates the validation rule"):
            r = RangeSet(10, validate=rFilter)

        # Test a validation that doesn't raise an exception for "None"
        def rFilter(m, i):
            return i is None or i % 2
        r = RangeSet(1,10,2, validate=rFilter)
        self.assertEqual(r, [1,3,5,7,9])

        with self.assertRaisesRegexp(
                ValueError, "The 'validate' keyword argument is not "
                "valid for non-finite RangeSet component"):
            r = RangeSet(1,10,0, validate=rFilter)

        def badRule(m, i):
            raise RuntimeError("ERROR: %s" % i)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            with self.assertRaisesRegexp(
                    RuntimeError, "ERROR: 1"):
                r = RangeSet(10, validate=badRule)
        self.assertEqual(
            output.getvalue(),
            "Exception raised while validating element "
            "'1' for Set FiniteSimpleRangeSet\n")

    def test_bounds(self):
        r = RangeSet(100, bounds=(2.5, 5.5))
        self.assertEqual(r, [3,4,5])

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

        # Test types that cannot be case to set
        self.assertNotEqual(SetOf({3,}), 3)

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
        self.assertTrue(i.isdiscrete())
        self.assertTrue(i.isfinite())
        self.assertFalse(i.isordered())

        i = SetOf([1,2,3])
        self.assertTrue(i.isdiscrete())
        self.assertTrue(i.isfinite())
        self.assertTrue(i.isordered())

        i = SetOf((1,2,3))
        self.assertTrue(i.isdiscrete())
        self.assertTrue(i.isfinite())
        self.assertTrue(i.isordered())

        i = RangeSet(3)
        self.assertTrue(i.isdiscrete())
        self.assertTrue(i.isfinite())
        self.assertTrue(i.isordered())
        self.assertIsInstance(i, _FiniteRangeSetData)

        i = RangeSet(1,3)
        self.assertTrue(i.isdiscrete())
        self.assertTrue(i.isfinite())
        self.assertTrue(i.isordered())
        self.assertIsInstance(i, _FiniteRangeSetData)

        i = RangeSet(1,3,0)
        self.assertFalse(i.isdiscrete())
        self.assertFalse(i.isfinite())
        self.assertFalse(i.isordered())
        self.assertIsInstance(i, _InfiniteRangeSetData)

    def test_pprint(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.K1 = RangeSet(0)
        m.K2 = RangeSet(10, 9)
        m.NotI = RangeSet(1,3,0)
        m.J = SetOf([1,2,3])

        buf = StringIO()
        m.pprint(ostream=buf)
        self.assertEqual(buf.getvalue().strip(), """
4 RangeSet Declarations
    I : Dimen=1, Size=3, Bounds=(1, 3)
        Key  : Finite : Members
        None :   True :   [1:3]
    K1 : Dimen=1, Size=0, Bounds=(None, None)
        Key  : Finite : Members
        None :   True :      []
    K2 : Dimen=1, Size=0, Bounds=(None, None)
        Key  : Finite : Members
        None :   True :      []
    NotI : Dimen=1, Size=Inf, Bounds=(1, 3)
        Key  : Finite : Members
        None :  False :  [1..3]

1 SetOf Declarations
    J : Dimen=1, Size=3, Bounds=(1, 3)
        Key  : Ordered : Members
        None :    True : [1, 2, 3]

5 Declarations: I K1 K2 NotI J""".strip())

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

        n = RangeSet(0)
        self.assertEqual(str(n), "[]")
        m.N = n
        self.assertEqual(str(n), "N")

        m.a = Param(initialize=3)
        o = RangeSet(m.a)
        self.assertEqual(str(o), "[1:3]")
        m.O = o
        self.assertEqual(str(o), "O")

        p = RangeSet(m.a, finite=False)
        self.assertEqual(str(p), "[1:3]")
        m.P = p
        self.assertEqual(str(p), "P")

        b = Param(initialize=3)
        oo = RangeSet(b)
        self.assertEqual(str(oo), "AbstractFiniteSimpleRangeSet")
        pp = RangeSet(b, finite=False)
        self.assertEqual(str(pp), "AbstractInfiniteSimpleRangeSet")

        b.construct()
        m.OO = oo
        self.assertEqual(str(oo), "OO")
        m.PP = pp
        self.assertEqual(str(pp), "PP")

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

        # test bad type
        with self.assertRaisesRegexp(
                TypeError, "'int' object is not iterable"):
            i.isdisjoint(1)

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

        # test bad type
        with self.assertRaisesRegexp(
                TypeError, "'int' object is not iterable"):
            i.issubset(1)

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
        # does not implement isfinite() is a discrete set.
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

        # test bad type
        with self.assertRaisesRegexp(
                TypeError, "'int' object is not iterable"):
            i.issuperset(1)

    def test_unordered_setof(self):
        i = SetOf({1,3,2,0})

        self.assertTrue(i.isfinite())
        self.assertFalse(i.isordered())

        self.assertEqual(i.ordered_data(), (0,1,2,3))
        self.assertEqual(i.sorted_data(), (0,1,2,3))
        self.assertEqual( tuple(reversed(i)),
                          tuple(reversed(list(i))) )

    def test_ordered_setof(self):
        i = SetOf([1,3,2,0])

        self.assertTrue(i.isfinite())
        self.assertTrue(i.isordered())

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

        self.assertTrue(i.isfinite())
        self.assertTrue(i.isordered())

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

        self.assertTrue(i.isfinite())
        self.assertTrue(i.isordered())

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
            self.assertTrue(x.isfinite())
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
            for idx, x in enumerate(r):
                self.assertIsInstance(x, NR)
                self.assertTrue(x.isfinite())
                self.assertEqual(x.start, i[idx+1])
                self.assertEqual(x.end, i[idx+1])
                self.assertEqual(x.step, 0)

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
            for idx, x in enumerate(r[:-1]):
                self.assertIsInstance(x, NR)
                self.assertTrue(x.isfinite())
                self.assertEqual(x.start, i[idx+1])
                self.assertEqual(x.end, i[idx+1])
                self.assertEqual(x.step, 0)
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

    def test_float_steps(self):
        a = RangeSet(0, 4, .5)
        self.assertEqual(len(a), 9)
        self.assertEqual(list(a - RangeSet(0,4,1)), [0.5, 1.5, 2.5, 3.5])

        with self.assertRaisesRegexp(
                ValueError, "RangeSet: start, end ordering incompatible with "
                "step direction \(got \[0:4:-0.5\]\)"):
            RangeSet(0,4,-.5)

    def test_check_values(self):
        m = ConcreteModel()
        m.I = RangeSet(5)

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core', logging.DEBUG):
            self.assertTrue(m.I.check_values())
        self.assertRegexpMatches(
            output.getvalue(),
            "^DEPRECATED: check_values\(\) is deprecated:")


class Test_SetOperator(unittest.TestCase):
    def test_construct(self):
        p = Param(initialize=3)
        a = RangeSet(p)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core', logging.DEBUG):
            i = a * a
            self.assertEqual(output.getvalue(), "")
        p.construct()
        with LoggingIntercept(output, 'pyomo.core', logging.DEBUG):
            i.construct()
            ref = 'Constructing SetOperator, name=SetProduct_OrderedSet, '\
                  'from data=None\n' \
                  'Constructing RangeSet, name=FiniteSimpleRangeSet, '\
                  'from data=None\n'\
                  'Constructing Set, name=SetProduct_OrderedSet, '\
                  'from data=None\n'
            self.assertEqual(output.getvalue(), ref)
            # Calling construct() twice bypasses construction the second
            # time around
            i.construct()
            self.assertEqual(output.getvalue(), ref)

    def test_deepcopy(self):
        # This tests the example in Set.__deepcopy__()
        # This also tests that returning Set.Skip from a rule works...
        a = AbstractModel()
        a.A = Set(initialize=[1,2])
        a.B = Set(initialize=[3,4])
        def x_init(m,i):
            if i == 2:
                return Set.Skip
            else:
                return []
        a.x = Set( [1,2],
                   domain={1: a.A*a.B, 2: a.A*a.A},
                   initialize=x_init )

        i = a.create_instance()
        self.assertEqual(len(i.x), 1)
        self.assertIn(1, i.x)
        self.assertNotIn(2, i.x)
        self.assertEqual(i.x[1].dimen, 2)
        self.assertEqual(i.x[1].domain, i.A*i.B)
        self.assertEqual(i.x[1], [])

    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_pandas_multiindex_set_init(self):
        # Test that TuplizeValuesInitializer does not assume truthiness
        # If it does, pandas will complain with the following error:
        # ValueError: The truth value of a MultiIndex is ambiguous. 
        # Use a.empty, a.bool(), a.item(), a.any() or a.all().
        iterables = [['bar', 'baz', 'foo', 'qux'], ['one', 'two']]
        pandas_index = pd.MultiIndex.from_product(
            iterables, 
            names=['first', 'second']
        )

        model = ConcreteModel()
        model.a = Set(initialize=pandas_index,
                      dimen=pandas_index.nlevels)

        # we will confirm that dimension is inferred correctly
        model.b = Set(initialize=pandas_index)

        self.assertIsInstance(model.a, Set)
        self.assertEquals(list(model.a), list(pandas_index))
        self.assertEquals(model.a.dimen, pandas_index.nlevels)

        self.assertIsInstance(model.b, Set)
        self.assertEquals(list(model.b), list(pandas_index))
        self.assertEquals(model.b.dimen, pandas_index.nlevels)


class TestSetUnion(unittest.TestCase):
    def test_pickle(self):
        a = SetOf([1,3,5]) | SetOf([2,3,4])
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a,b)
        self.assertEqual(a,b)

    def test_len(self):
        a = SetOf([1,2,3])
        self.assertEqual(len(a), 3)
        b = a | Reals
        with self.assertRaisesRegexp(
                OverflowError, 'The length of a non-finite Set is Inf'):
            len(b)

    def test_bounds(self):
        a = SetOf([-2,-1,0,1])
        b = a | NonNegativeReals
        self.assertEqual(b.bounds(), (-2, None))
        b = NonNegativeReals | a
        self.assertEqual(b.bounds(), (-2, None))
        b = a | RangeSet(3)
        self.assertEqual(b.bounds(), (-2, 3))
        b = NegativeReals | NonNegativeReals
        self.assertEqual(b.bounds(), (None, None))

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
    Key  : Dimen : Domain        : Size : Members
    None :     1 : I | A_index_0 :    4 : {1, 2, 3, 4}
""".strip()
        self.assertEqual(output.getvalue().strip(), ref)


    def test_dimen(self):
        m = ConcreteModel()
        m.I1 = SetOf([1,2,3,4])
        m.I2 = SetOf([(1,2), (3,4)])
        m.IN = SetOf([(1,2), (3,4), 1, 2])
        m.J = Set()
        self.assertEqual((m.I1 | m.I1).dimen, 1)
        self.assertEqual((m.I2 | m.I2).dimen, 2)
        self.assertEqual((m.IN | m.IN).dimen, None)
        self.assertEqual((m.I1 | m.I2).dimen, None)
        self.assertEqual((m.IN | m.I2).dimen, None)
        self.assertEqual((m.I2 | m.IN).dimen, None)
        self.assertEqual((m.IN | m.I1).dimen, None)
        self.assertEqual((m.I1 | m.IN).dimen, None)
        self.assertEqual((m.I1 | m.J).dimen, UnknownSetDimen)
        self.assertEqual((m.I2 | m.J).dimen, UnknownSetDimen)
        self.assertEqual((m.IN | m.J).dimen, None)
        self.assertEqual((m.J | m.I1).dimen, UnknownSetDimen)
        self.assertEqual((m.J | m.I2).dimen, UnknownSetDimen)
        self.assertEqual((m.J | m.IN).dimen, None)

    def _verify_ordered_union(self, a, b):
        # Note the placement of the second "3" in the middle of the set.
        # This helps catch edge cases where we need to ensure it doesn't
        # count as part of the set membership
        if isinstance(a, SetOf):
            self.assertTrue(a.isordered())
            self.assertTrue(a.isfinite())
        else:
            self.assertIs(type(a), list)
        if isinstance(b, SetOf):
            self.assertTrue(b.isordered())
            self.assertTrue(b.isfinite())
        else:
            self.assertIs(type(b), list)

        x = a | b
        self.assertIs(type(x), SetUnion_OrderedSet)
        self.assertTrue(x.isfinite())
        self.assertTrue(x.isordered())
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
                self.assertTrue(a.isordered())
            else:
                self.assertFalse(a.isordered())
            self.assertTrue(a.isfinite())
        else:
            self.assertIn(type(a), (list, set))
        if isinstance(b, SetOf):
            if type(b._ref) is list:
                self.assertTrue(b.isordered())
            else:
                self.assertFalse(b.isordered())
            self.assertTrue(b.isfinite())
        else:
            self.assertIn(type(b), (list, set))

        x = a | b
        self.assertIs(type(x), SetUnion_FiniteSet)
        self.assertTrue(x.isfinite())
        self.assertFalse(x.isordered())
        self.assertEqual(len(x), 5)
        if x._sets[0].isordered():
            self.assertEqual(list(x)[:3], [1,3,2])
        if x._sets[1].isordered():
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
            self.assertFalse(a.isordered())
            self.assertFalse(a.isfinite())
        else:
            self.assertIn(type(a), (list, set))
        if isinstance(b, RangeSet):
            self.assertFalse(b.isordered())
            self.assertFalse(b.isfinite())
        else:
            self.assertIn(type(b), (list, set))

        x = a | b
        self.assertIs(type(x), SetUnion_InfiniteSet)
        self.assertFalse(x.isfinite())
        self.assertFalse(x.isordered())

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

    def test_bounds(self):
        a = SetOf([-2,-1,0,1])
        b = a & NonNegativeReals
        self.assertEqual(b.bounds(), (0, 1))
        b = NonNegativeReals & a
        self.assertEqual(b.bounds(), (0, 1))
        b = a & RangeSet(3)
        self.assertEqual(b.bounds(), (1, 1))

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
    Key  : Dimen : Domain        : Size : Members
    None :     1 : I & A_index_0 :    0 :      {}
""".strip()
        self.assertEqual(output.getvalue().strip(), ref)

    def test_dimen(self):
        m = ConcreteModel()
        m.I1 = SetOf([1,2,3,4])
        m.I2 = SetOf([(1,2), (3,4)])
        m.IN = SetOf([(1,2), (3,4), 1, 2])
        m.J = Set()
        self.assertEqual((m.I1 & m.I1).dimen, 1)
        self.assertEqual((m.I2 & m.I2).dimen, 2)
        self.assertEqual((m.IN & m.IN).dimen, None)
        self.assertEqual((m.I1 & m.I2).dimen, 0)
        self.assertEqual((m.IN & m.I2).dimen, 2)
        self.assertEqual((m.I2 & m.IN).dimen, 2)
        self.assertEqual((m.IN & m.I1).dimen, 1)
        self.assertEqual((m.I1 & m.IN).dimen, 1)
        self.assertEqual((m.I1 & m.J).dimen, UnknownSetDimen)
        self.assertEqual((m.I2 & m.J).dimen, UnknownSetDimen)
        self.assertEqual((m.IN & m.J).dimen, UnknownSetDimen)
        self.assertEqual((m.J & m.I1).dimen, UnknownSetDimen)
        self.assertEqual((m.J & m.I2).dimen, UnknownSetDimen)
        self.assertEqual((m.J & m.IN).dimen, UnknownSetDimen)

    def _verify_ordered_intersection(self, a, b):
        if isinstance(a, (Set, SetOf, RangeSet)):
            a_ordered = a.isordered()
        else:
            a_ordered = type(a) is list
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_ordered = b.isordered()
        else:
            b_ordered = type(b) is list
        self.assertTrue(a_ordered or b_ordered)

        if a_ordered:
            ref = (3,2,5)
        else:
            ref = (2,3,5)

        x = a & b
        self.assertIs(type(x), SetIntersection_OrderedSet)
        self.assertTrue(x.isfinite())
        self.assertTrue(x.isordered())
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
            a_finite = a.isfinite()
        else:
            a_finite = True
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_finite = b.isfinite()
        else:
            b_finite = True
        self.assertTrue(a_finite or b_finite)

        x = a & b
        self.assertIs(type(x), SetIntersection_FiniteSet)
        self.assertTrue(x.isfinite())
        self.assertFalse(x.isordered())
        self.assertEqual(len(x), 3)
        if x._sets[0].isordered():
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
            a_finite = a.isfinite()
        else:
            a_finite = True
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_finite = b.isfinite()
        else:
            b_finite = True
        self.assertEqual([a_finite, b_finite], [False,False])

        x = a & b
        self.assertIs(type(x), SetIntersection_InfiniteSet)
        self.assertFalse(x.isfinite())
        self.assertFalse(x.isordered())

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
        m = AbstractModel()
        m.p = Param(initialize=0)
        m.a = RangeSet(0, None, 2)
        m.b = RangeSet(5,10,m.p, finite=False)
        m.x = m.a & m.b
        self.assertTrue(m.a._constructed)
        self.assertFalse(m.b._constructed)
        self.assertFalse(m.x._constructed)
        self.assertIs(type(m.x), SetIntersection_InfiniteSet)
        i = m.create_instance()
        self.assertIs(type(i.x), SetIntersection_OrderedSet)
        self.assertEqual(list(i.x), [6,8,10])

        self.assertEqual(i.x.ord(6), 1)
        self.assertEqual(i.x.ord(8), 2)
        self.assertEqual(i.x.ord(10), 3)

        self.assertEqual(i.x[1], 6)
        self.assertEqual(i.x[2], 8)
        self.assertEqual(i.x[3], 10)
        with self.assertRaisesRegexp(
                IndexError,
                "x index out of range"):
            i.x[4]

        self.assertEqual(i.x[-3], 6)
        self.assertEqual(i.x[-2], 8)
        self.assertEqual(i.x[-1], 10)
        with self.assertRaisesRegexp(
                IndexError,
                "x index out of range"):
            i.x[-4]

    def test_subsets(self):
        a = SetOf([1])
        b = SetOf([1])
        c = SetOf([1])
        d = SetOf([1])

        x = a & b
        self.assertEqual(len(x._sets), 2)
        self.assertEqual(list(x.subsets()), [x])
        self.assertEqual(list(x.subsets(False)), [x])
        self.assertEqual(list(x.subsets(True)), [a,b])
        x = a & b & c
        self.assertEqual(len(x._sets), 2)
        self.assertEqual(list(x.subsets()), [x])
        self.assertEqual(list(x.subsets(False)), [x])
        self.assertEqual(list(x.subsets(True)), [a,b,c])
        x = (a & b) & (c & d)
        self.assertEqual(len(x._sets), 2)
        self.assertEqual(list(x.subsets()), [x])
        self.assertEqual(list(x.subsets(False)), [x])
        self.assertEqual(list(x.subsets(True)), [a,b,c,d])

        x = (a & b) * (c & d)
        self.assertEqual(len(x._sets), 2)
        self.assertEqual(len(list(x.subsets())), 2)
        self.assertEqual(list(x.subsets()), [a&b, c&d])
        self.assertEqual(list(x.subsets(False)), [a&b, c&d])
        self.assertEqual(len(list(x.subsets(True))), 4)
        self.assertEqual(list(x.subsets(True)), [a,b,c,d])


class TestSetDifference(unittest.TestCase):
    def test_pickle(self):
        a = SetOf([1,3,5]) - SetOf([2,3,4])
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a,b)
        self.assertEqual(a,b)

    def test_bounds(self):
        a = SetOf([-2,-1,0,1])
        b = a - NonNegativeReals
        self.assertEqual(b.bounds(), (-2, -1))
        b = a - RangeSet(3)
        self.assertEqual(b.bounds(), (-2, 0))

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
    Key  : Dimen : Domain        : Size : Members
    None :     1 : I - A_index_0 :    2 : {1, 2}
""".strip()
        self.assertEqual(output.getvalue().strip(), ref)

    def test_dimen(self):
        m = ConcreteModel()
        m.I1 = SetOf([1,2,3,4])
        m.I2 = SetOf([(1,2), (3,4)])
        m.IN = SetOf([(1,2), (3,4), 1, 2])
        m.J = Set()
        self.assertEqual((m.I1 - m.I1).dimen, 1)
        self.assertEqual((m.I2 - m.I2).dimen, 2)
        self.assertEqual((m.IN - m.IN).dimen, None)
        self.assertEqual((m.I1 - m.I2).dimen, 1)
        self.assertEqual((m.I2 - m.I1).dimen, 2)
        self.assertEqual((m.IN - m.I2).dimen, None)
        self.assertEqual((m.I2 - m.IN).dimen, 2)
        self.assertEqual((m.IN - m.I1).dimen, None)
        self.assertEqual((m.I1 - m.IN).dimen, 1)
        self.assertEqual((m.I1 - m.J).dimen, 1)
        self.assertEqual((m.I2 - m.J).dimen, 2)
        self.assertEqual((m.IN - m.J).dimen, None)
        self.assertEqual((m.J - m.I1).dimen, UnknownSetDimen)
        self.assertEqual((m.J - m.I2).dimen, UnknownSetDimen)
        self.assertEqual((m.J - m.IN).dimen, UnknownSetDimen)

    def _verify_ordered_difference(self, a, b):
        if isinstance(a, (Set, SetOf, RangeSet)):
            a_ordered = a.isordered()
        else:
            a_ordered = type(a) is list
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_ordered = b.isordered()
        else:
            b_ordered = type(b) is list
        self.assertTrue(a_ordered)

        x = a - b
        self.assertIs(type(x), SetDifference_OrderedSet)
        self.assertTrue(x.isfinite())
        self.assertTrue(x.isordered())
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
            a_finite = a.isfinite()
        else:
            a_finite = True
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_finite = b.isfinite()
        else:
            b_finite = True
        self.assertTrue(a_finite or b_finite)

        x = a - b
        self.assertIs(type(x), SetDifference_FiniteSet)
        self.assertTrue(x.isfinite())
        self.assertFalse(x.isordered())
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
        self.assertFalse(x.isfinite())
        self.assertFalse(x.isordered())

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

    def test_bounds(self):
        a = SetOf([-2,-1,0,1])
        b = a ^ NonNegativeReals
        self.assertEqual(b.bounds(), (-2, None))
        c = a ^ RangeSet(3)
        self.assertEqual(c.bounds(), (-2, 3))

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
    Key  : Dimen : Domain        : Size : Members
    None :     1 : I ^ A_index_0 :    4 : {1, 2, 3, 4}
""".strip()
        self.assertEqual(output.getvalue().strip(), ref)

    def test_dimen(self):
        m = ConcreteModel()
        m.I1 = SetOf([1,2,3,4])
        m.I2 = SetOf([(1,2), (3,4)])
        m.IN = SetOf([(1,2), (3,4), 1, 2])
        m.J = Set()
        self.assertEqual((m.I1 ^ m.I1).dimen, 1)
        self.assertEqual((m.I2 ^ m.I2).dimen, 2)
        self.assertEqual((m.IN ^ m.IN).dimen, None)
        self.assertEqual((m.I1 ^ m.I2).dimen, None)
        self.assertEqual((m.I2 ^ m.I1).dimen, None)
        self.assertEqual((m.IN ^ m.I2).dimen, None)
        self.assertEqual((m.I2 ^ m.IN).dimen, None)
        self.assertEqual((m.IN ^ m.I1).dimen, None)
        self.assertEqual((m.I1 ^ m.IN).dimen, None)
        self.assertEqual((m.I1 ^ m.J).dimen, UnknownSetDimen)
        self.assertEqual((m.I2 ^ m.J).dimen, UnknownSetDimen)
        self.assertEqual((m.IN ^ m.J).dimen, None)
        self.assertEqual((m.J ^ m.I1).dimen, UnknownSetDimen)
        self.assertEqual((m.J ^ m.I2).dimen, UnknownSetDimen)
        self.assertEqual((m.J ^ m.IN).dimen, None)

    def _verify_ordered_symdifference(self, a, b):
        if isinstance(a, (Set, SetOf, RangeSet)):
            a_ordered = a.isordered()
        else:
            a_ordered = type(a) is list
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_ordered = b.isordered()
        else:
            b_ordered = type(b) is list
        self.assertTrue(a_ordered)

        x = a ^ b
        self.assertIs(type(x), SetSymmetricDifference_OrderedSet)
        self.assertTrue(x.isfinite())
        self.assertTrue(x.isordered())
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
            a_finite = a.isfinite()
        else:
            a_finite = True
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_finite = b.isfinite()
        else:
            b_finite = True
        self.assertTrue(a_finite or b_finite)

        x = a ^ b
        self.assertIs(type(x), SetSymmetricDifference_FiniteSet)
        self.assertTrue(x.isfinite())
        self.assertFalse(x.isordered())
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
        self.assertFalse(x.isfinite())
        self.assertFalse(x.isordered())

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
        self.assertFalse(x.isfinite())
        self.assertFalse(x.isordered())

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
        self.assertFalse(x.isfinite())
        self.assertFalse(x.isordered())

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

    def test_bounds(self):
        a = SetOf([-2,-1,0,1])
        b = a * NonNegativeReals
        self.assertEqual(b.bounds(), ((-2, 0), (1, None)))
        c = a * RangeSet(3)
        self.assertEqual(c.bounds(), ((-2, 1), (1, 3)))

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
    Key  : Dimen : Domain      : Size : Members
    None :     2 : I*A_index_0 :    4 : {(1, 3), (1, 4), (2, 3), (2, 4)}
""".strip()
        self.assertEqual(output.getvalue().strip(), ref)

        m = ConcreteModel()
        m.I = Set(initialize=[1,2,3])
        m.J = Reals * m.I
        output = StringIO()
        m.J.pprint(ostream=output)
        ref="""
J : Size=1, Index=None, Ordered=False
    Key  : Dimen : Domain  : Size : Members
    None :     2 : Reals*I :  Inf : <[None..None], ([1], [2], [3])>
""".strip()
        self.assertEqual(output.getvalue().strip(), ref)

    def test_dimen(self):
        m = ConcreteModel()
        m.I1 = SetOf([1,2,3,4])
        m.I2 = SetOf([(1,2), (3,4)])
        m.IN = SetOf([(1,2), (3,4), 1, 2])
        m.J = Set()
        self.assertEqual((m.I1 * m.I1).dimen, 2)
        self.assertEqual((m.I2 * m.I2).dimen, 4)
        self.assertEqual((m.IN * m.IN).dimen, None)
        self.assertEqual((m.I1 * m.I2).dimen, 3)
        self.assertEqual((m.I2 * m.I1).dimen, 3)
        self.assertEqual((m.IN * m.I2).dimen, None)
        self.assertEqual((m.I2 * m.IN).dimen, None)
        self.assertEqual((m.IN * m.I1).dimen, None)
        self.assertEqual((m.I1 * m.IN).dimen, None)
        self.assertIs((m.J * m.I1).dimen, UnknownSetDimen)
        self.assertIs((m.J * m.I2).dimen, UnknownSetDimen)
        self.assertIs((m.J * m.IN).dimen, None)
        self.assertIs((m.I1 * m.J).dimen, UnknownSetDimen)
        self.assertIs((m.I2 * m.J).dimen, UnknownSetDimen)
        self.assertIs((m.IN * m.J).dimen, None)

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

    def test_subsets(self):
        a = SetOf([1])
        b = SetOf([1])
        c = SetOf([1])
        d = SetOf([1])

        x = a * b
        self.assertEqual(len(x._sets), 2)
        self.assertEqual(list(x.subsets()), [a,b])
        self.assertEqual(list(x.subsets(True)), [a,b])
        self.assertEqual(list(x.subsets(False)), [a,b])
        x = a * b * c
        self.assertEqual(len(x._sets), 2)
        self.assertEqual(list(x.subsets()), [a,b,c])
        self.assertEqual(list(x.subsets(True)), [a,b,c])
        self.assertEqual(list(x.subsets(False)), [a,b,c])
        x = (a * b) * (c * d)
        self.assertEqual(len(x._sets), 2)
        self.assertEqual(list(x.subsets()), [a,b,c,d])
        self.assertEqual(list(x.subsets(True)), [a,b,c,d])
        self.assertEqual(list(x.subsets(False)), [a,b,c,d])

        x = (a - b) * (c * d)
        self.assertEqual(len(x._sets), 2)
        self.assertEqual(len(list(x.subsets())), 3)
        self.assertEqual(len(list(x.subsets(False))), 3)
        self.assertEqual(list(x.subsets()), [(a-b),c,d])
        self.assertEqual(len(list(x.subsets(True))), 4)
        self.assertEqual(list(x.subsets(True)), [a,b,c,d])

    def test_set_tuple(self):
        a = SetOf([1])
        b = SetOf([1])
        x = a * b
        os = StringIO()
        with LoggingIntercept(os, 'pyomo'):
            self.assertEqual(x.set_tuple, [a,b])
        self.assertRegexpMatches(
            os.getvalue(),
            '^DEPRECATED: SetProduct.set_tuple is deprecated.')

    def test_no_normalize_index(self):
        try:
            _oldFlatten = normalize_index.flatten
            I = SetOf([1, (1,2)])
            J = SetOf([3, (2,3)])
            x = I * J

            normalize_index.flatten = False
            self.assertIs(x.dimen, None)
            self.assertIn(((1,2),3), x)
            self.assertIn((1,(2,3)), x)
            # if we are not flattening, then lookup must match the
            # subsets exactly.
            self.assertNotIn((1,2,3), x)

            normalize_index.flatten = True
            self.assertIs(x.dimen, None)
            self.assertIn(((1,2),3), x)
            self.assertIn((1,(2,3)), x)
            self.assertIn((1,2,3), x)
        finally:
            normalize_index.flatten = _oldFlatten

    def test_infinite_setproduct(self):
        x = PositiveIntegers * SetOf([2,3,5,7])
        self.assertFalse(x.isfinite())
        self.assertFalse(x.isordered())
        self.assertIn((1,2), x)
        self.assertNotIn((0,2), x)
        self.assertNotIn((1,1), x)
        self.assertNotIn(('a',2), x)
        self.assertNotIn((2,'a'), x)

        x = SetOf([2,3,5,7]) * PositiveIntegers
        self.assertFalse(x.isfinite())
        self.assertFalse(x.isordered())
        self.assertIn((3,2), x)
        self.assertNotIn((1,2), x)
        self.assertNotIn((2,0), x)
        self.assertNotIn(('a',2), x)
        self.assertNotIn((2,'a'), x)

        x = PositiveIntegers * PositiveIntegers
        self.assertFalse(x.isfinite())
        self.assertFalse(x.isordered())
        self.assertIn((3,2), x)
        self.assertNotIn((0,2), x)
        self.assertNotIn((2,0), x)
        self.assertNotIn(('a',2), x)
        self.assertNotIn((2,'a'), x)

    def _verify_finite_product(self, a, b):
        if isinstance(a, (Set, SetOf, RangeSet)):
            a_ordered = a.isordered()
        else:
            a_ordered = type(a) is list
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_ordered = b.isordered()
        else:
            b_ordered = type(b) is list
        self.assertFalse(a_ordered and b_ordered)

        x = a * b

        self.assertIs(type(x), SetProduct_FiniteSet)
        self.assertTrue(x.isfinite())
        self.assertFalse(x.isordered())
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
            a_ordered = a.isordered()
        else:
            a_ordered = type(a) is list
        self.assertTrue(a_ordered)
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_ordered = b.isordered()
        else:
            b_ordered = type(b) is list
        self.assertTrue(b_ordered)

        x = a * b

        self.assertIs(type(x), SetProduct_OrderedSet)
        self.assertTrue(x.isfinite())
        self.assertTrue(x.isordered())
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
            self.assertEqual(x.dimen, 4)

            SetModule.FLATTEN_CROSS_PRODUCT = False
            ref = [((1,2),(5,6)), ((1,2),(7,8)), ((3,4),(5,6)), ((3,4),(7,8))]
            self.assertEqual(list(x), ref)
            self.assertEqual(x.dimen, None)
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
            self.assertEqual(x.dimen, None)

            SetModule.FLATTEN_CROSS_PRODUCT = False
            ref = [(1,2,3), (1,2,4), (1,2,5),
                   (1,(2,3),3), (1,(2,3),4), (1,(2,3),5)]
            self.assertEqual(list(x), ref)
            self.assertEqual(x.dimen, None)
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
            self.assertEqual(x.dimen, None)

            SetModule.FLATTEN_CROSS_PRODUCT = False
            ref = [(1,2,4,0), (1,2,4,1),
                   (1,2,(3,4),0), (1,2,(3,4),1),
                   (1,(2,3),4,0), (1,(2,3),4,1),
                   (1,(2,3),(3,4),0), (1,(2,3),(3,4),1)]
            self.assertEqual(list(x), ref)
            for i,v in enumerate(ref):
                self.assertEqual(x[i+1], v)
            self.assertEqual(x.dimen, None)
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

    def test_setproduct_construct_data(self):
        m = AbstractModel()
        m.I = Set(initialize=[1,2])
        m.J = m.I * m.I
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            m.create_instance(
                data={None:{'J': {None: [(1,1),(1,2),(2,1),(2,2)]}}})
        self.assertRegexpMatches(
            output.getvalue().replace('\n',' '),
            "^DEPRECATED: Providing construction data to SetOperator objects "
            "is deprecated")

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            with self.assertRaisesRegexp(
                    ValueError, "Constructing SetOperator J with "
                    "incompatible data \(data=\{None: \[\(1, 1\), \(1, 2\), "
                    "\(2, 1\)\]\}"):
                m.create_instance(
                    data={None:{'J': {None: [(1,1),(1,2),(2,1)]}}})
        self.assertRegexpMatches(
            output.getvalue().replace('\n',' '),
            "^DEPRECATED: Providing construction data to SetOperator objects "
            "is deprecated")

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            with self.assertRaisesRegexp(
                    ValueError, "Constructing SetOperator J with "
                    "incompatible data \(data=\{None: \[\(1, 3\), \(1, 2\), "
                    "\(2, 1\), \(2, 2\)\]\}"):
                m.create_instance(
                    data={None:{'J': {None: [(1,3),(1,2),(2,1),(2,2)]}}})
        self.assertRegexpMatches(
            output.getvalue().replace('\n',' '),
            "^DEPRECATED: Providing construction data to SetOperator objects "
            "is deprecated")

    def test_setproduct_nondim_set(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1,2,3])
        m.J = Set()
        m.K = Set(initialize=[4,5,6])
        m.Z = m.I * m.J * m.K
        self.assertEqual(len(m.Z), 0)
        self.assertNotIn((2,5), m.Z)

        m.J.add(0)
        self.assertEqual(len(m.Z), 9)
        self.assertIn((2,0,5), m.Z)

    def test_setproduct_toolong_val(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1,2,3])
        m.J = Set(initialize=[4,5,6])
        m.Z = m.I * m.J
        self.assertIn((2,5), m.Z)
        self.assertNotIn((2,5,3), m.Z)

        m = ConcreteModel()
        m.I = Set(initialize=[1,2,3])
        m.J = Set(initialize=[4,5,6], dimen=None)
        m.Z = m.I * m.J
        self.assertIn((2,5), m.Z)
        self.assertNotIn((2,5,3), m.Z)


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

    def test_iteration(self):
        with self.assertRaisesRegexp(
                TypeError, "'GlobalSet' object is not iterable "
                "\(non-finite Set 'Reals' is not iterable\)"):
            iter(Reals)

        with self.assertRaisesRegexp(
                TypeError, "'GlobalSet' object is not iterable "
                "\(non-finite Set 'Integers' is not iterable\)"):
            iter(Integers)

        self.assertEqual(list(iter(Binary)), [0,1])

    def test_declare(self):
        NS = {}
        DeclareGlobalSet(RangeSet( name='TrinarySet',
                                   ranges=(NR(0,2,1),) ),
                         NS)
        self.assertEqual(list(NS['TrinarySet']), [0,1,2])
        a = pickle.loads(pickle.dumps(NS['TrinarySet']))
        self.assertIs(a, NS['TrinarySet'])
        with self.assertRaisesRegex(
                NameError, "name 'TrinarySet' is not defined"):
            TrinarySet
        del SetModule.GlobalSets['TrinarySet']
        del NS['TrinarySet']

        # Now test the automatic identification of the globals() scope
        DeclareGlobalSet(RangeSet( name='TrinarySet',
                                   ranges=(NR(0,2,1),) ))
        self.assertEqual(list(TrinarySet), [0,1,2])
        a = pickle.loads(pickle.dumps(TrinarySet))
        self.assertIs(a, TrinarySet)
        del SetModule.GlobalSets['TrinarySet']
        del globals()['TrinarySet']
        with self.assertRaisesRegex(
                NameError, "name 'TrinarySet' is not defined"):
            TrinarySet

    def test_exceptions(self):
        with self.assertRaisesRegex(
                RuntimeError, "Duplicate Global Set declaration, Reals"):
            DeclareGlobalSet(RangeSet( name='Reals', ranges=(NR(0,2,1),) ))

        # But repeat declarations are OK
        a = Reals
        DeclareGlobalSet(Reals)
        self.assertIs(a, Reals)
        self.assertIs(a, globals()['Reals'])
        self.assertIs(a, SetModule.GlobalSets['Reals'])

        NS = {}
        ts = DeclareGlobalSet(
            RangeSet(name='TrinarySet', ranges=(NR(0,2,1),)), NS)
        self.assertIs(NS['TrinarySet'], ts)

        # Repeat declaration is OK
        DeclareGlobalSet(ts, NS)
        self.assertIs(NS['TrinarySet'], ts)

        # but conflicting one raises exception
        NS['foo'] = None
        with self.assertRaisesRegex(
                RuntimeError, "Refusing to overwrite global object, foo"):
            DeclareGlobalSet(
                RangeSet( name='foo', ranges=(NR(0,2,1),) ), NS)

    def test_RealSet_IntegerSet(self):
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.RealSet()
        self.assertIn('DEPRECATED: The use of RealSet,', output.getvalue())
        self.assertEqual(a, Reals)
        self.assertIsNot(a, Reals)

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.RealSet(bounds=(1,3))
        self.assertIn('DEPRECATED: The use of RealSet,', output.getvalue())
        self.assertEqual(a.bounds(), (1,3))

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.IntegerSet()
        self.assertIn('DEPRECATED: The use of RealSet,', output.getvalue())
        self.assertEqual(a, Integers)
        self.assertIsNot(a, Integers)

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.IntegerSet(bounds=(1,3))
        self.assertIn('DEPRECATED: The use of RealSet,', output.getvalue())
        self.assertEqual(a.bounds(), (1,3))
        self.assertEqual(list(a), [1,2,3])

        m = ConcreteModel()

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            m.x = Var(within=SetModule.RealSet)
        self.assertIn('DEPRECATED: The use of RealSet,', output.getvalue())

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            m.y = Var(within=SetModule.RealSet())
        self.assertIn('DEPRECATED: The use of RealSet,', output.getvalue())

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            m.z = Var(within=SetModule.RealSet(bounds=(0,None)))
        self.assertIn('DEPRECATED: The use of RealSet,', output.getvalue())

        with self.assertRaisesRegex(
                RuntimeError, "Unexpected keyword arguments: \{'foo': 5\}"):
            IntegerSet(foo=5)

    def test_intervals(self):
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.RealInterval()
        self.assertIn("RealInterval has been deprecated.", output.getvalue())
        self.assertEqual(a, Reals)

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.RealInterval(bounds=(0,None))
        self.assertIn("RealInterval has been deprecated.", output.getvalue())
        self.assertEqual(a, NonNegativeReals)

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.RealInterval(bounds=5)
        self.assertIn("RealInterval has been deprecated.", output.getvalue())
        self.assertEqual(a, RangeSet(1,5,0))


        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.RealInterval(bounds=(5,))
        self.assertIn("RealInterval has been deprecated.", output.getvalue())
        self.assertEqual(a, RangeSet(1,5,0))

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.IntegerInterval()
        self.assertIn("IntegerInterval has been deprecated.", output.getvalue())
        self.assertEqual(a, Integers)

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.IntegerInterval(bounds=(0,None))
        self.assertIn("IntegerInterval has been deprecated.", output.getvalue())
        self.assertEqual(a, NonNegativeIntegers)
        self.assertFalse(a.isfinite())

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.IntegerInterval(bounds=(None,-1))
        self.assertIn("IntegerInterval has been deprecated.", output.getvalue())
        self.assertEqual(a, NegativeIntegers)
        self.assertFalse(a.isfinite())

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.IntegerInterval(bounds=(-float('inf'),-1))
        self.assertIn("IntegerInterval has been deprecated.", output.getvalue())
        self.assertEqual(a, NegativeIntegers)
        self.assertFalse(a.isfinite())

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.IntegerInterval(bounds=(0,3))
        self.assertIn("IntegerInterval has been deprecated.", output.getvalue())
        self.assertEqual(list(a), [0,1,2,3])
        self.assertTrue(a.isfinite())

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.IntegerInterval(bounds=5)
        self.assertIn("IntegerInterval has been deprecated.", output.getvalue())
        self.assertEqual(list(a), [1,2,3,4,5])

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.IntegerInterval(bounds=(5,))
        self.assertIn("IntegerInterval has been deprecated.", output.getvalue())
        self.assertEqual(list(a), [1,2,3,4,5])


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
        self.assertRegexpMatches(
            output.getvalue(),
            "^DEPRECATED: Pyomo Sets ignore the 'virtual' keyword argument")

    def test_scalar_set_initialize_and_iterate(self):
        m = ConcreteModel()
        m.I = Set()
        self.assertEqual(list(m.I), [])
        self.assertEqual(list(reversed(m.I)), [])
        self.assertEqual(m.I.data(), ())
        self.assertEqual(m.I.dimen, UnknownSetDimen)

        m = ConcreteModel()
        with self.assertRaisesRegexp(
                KeyError, "Cannot treat the scalar component 'I' "
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
            ref = "Initializing ordered Set I with a " \
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
        self.assertEqual(sorted(list(m.I)), [1,2,3,4])
        # We can't directly compare the reversed to a reference list
        # (because this is an unordered set!) but we can compare it with
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

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            with self.assertRaisesRegexp(
                    TypeError, "'int' object is not iterable"):
                m = ConcreteModel()
                m.I = Set(initialize=5)
            ref = "Initializer for Set I returned non-iterable object " \
                  "of type int."
            self.assertIn(ref, output.getvalue())

    def test_insertion_deletion(self):
        def _verify(_s, _l):
            self.assertTrue(_s.isordered())
            self.assertTrue(_s.isfinite())
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
            self.assertFalse(_s.isordered())
            self.assertTrue(_s.isfinite())

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

    def test_multiple_insertion(self):
        m = ConcreteModel()
        m.I = Set(ordered=True, initialize=[1])

        self.assertEqual(m.I.add(3,2,4), 3)
        self.assertEqual(tuple(m.I.data()), (1,3,2,4))

        self.assertEqual(m.I.add(1,5,4), 1)
        self.assertEqual(tuple(m.I.data()), (1,3,2,4,5))


    def test_indexed_set(self):
        # Implicit construction
        m = ConcreteModel()
        m.I = Set([1,2,3], ordered=False)
        self.assertEqual(len(m.I), 0)
        self.assertEqual(m.I.data(), {})
        m.I[1]
        self.assertEqual(len(m.I), 1)
        self.assertEqual(m.I[1], [])
        self.assertEqual(m.I.data(), {1:()})

        self.assertEqual(m.I[2], [])
        self.assertEqual(len(m.I), 2)
        self.assertEqual(m.I.data(), {1:(), 2:()})

        m.I[1].add(1)
        m.I[2].add(2)
        m.I[3].add(4)
        self.assertEqual(len(m.I), 3)
        self.assertEqual(list(m.I[1]), [1])
        self.assertEqual(list(m.I[2]), [2])
        self.assertEqual(list(m.I[3]), [4])
        self.assertIsNot(m.I[1], m.I[2])
        self.assertIsNot(m.I[1], m.I[3])
        self.assertIsNot(m.I[2], m.I[3])
        self.assertFalse(m.I[1].isordered())
        self.assertFalse(m.I[2].isordered())
        self.assertFalse(m.I[3].isordered())
        self.assertIs(type(m.I[1]), _FiniteSetData)
        self.assertIs(type(m.I[2]), _FiniteSetData)
        self.assertIs(type(m.I[3]), _FiniteSetData)
        self.assertEqual(m.I.data(), {1:(1,), 2:(2,), 3:(4,)})

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
        self.assertTrue(m.I[1].isordered())
        self.assertTrue(m.I[2].isordered())
        self.assertTrue(m.I[3].isordered())
        self.assertIs(type(m.I[1]), _InsertionOrderSetData)
        self.assertIs(type(m.I[2]), _InsertionOrderSetData)
        self.assertIs(type(m.I[3]), _InsertionOrderSetData)
        self.assertEqual(m.I.data(), {1:(4,2,5), 2:(4,2,5), 3:(4,2,5)})

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
        self.assertTrue(m.I[1].isordered())
        self.assertTrue(m.I[2].isordered())
        self.assertTrue(m.I[3].isordered())
        self.assertIs(type(m.I[1]), _SortedSetData)
        self.assertIs(type(m.I[2]), _SortedSetData)
        self.assertIs(type(m.I[3]), _SortedSetData)
        self.assertEqual(m.I.data(), {1:(2,4,5), 2:(2,4,5), 3:(2,4,5)})

        # Explicit (procedural) construction
        m = ConcreteModel()
        m.I = Set([1,2,3], ordered=True)
        self.assertEqual(len(m.I), 0)
        m.I[1] = [1,2,3]
        m.I[(2,)] = [4,5,6]
        # test index mapping
        self.assertEqual(sorted(m.I._data.keys()), [1,2])
        self.assertEqual(list(m.I[1]), [1,2,3])
        self.assertEqual(list(m.I[2]), [4,5,6])
        self.assertEqual(m.I.data(), {1:(1,2,3), 2:(4,5,6)})


    def test_naming(self):
        m = ConcreteModel()

        i = Set()
        self.assertEqual(str(i), "AbstractOrderedSimpleSet")
        i.construct()
        self.assertEqual(str(i), "{}")
        m.I = i
        self.assertEqual(str(i), "I")

        j = Set(initialize=[1,2,3])
        self.assertEqual(str(j), "AbstractOrderedSimpleSet")
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
        self.assertIs(m.I.filter, None)
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
        # Note that pypy raises a different exception from cpython
        err = "Unable to insert '{}' into Set J:\n\tTypeError: "\
            "((unhashable type: 'dict')|('dict' objects are unhashable))"
        with self.assertRaisesRegexp(TypeError, err):
            m.J.add({})

        self.assertTrue( m.J.add((1,)) )
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            self.assertFalse( m.J.add(1) )
        self.assertEquals(
            output.getvalue(),
            "Element 1 already exists in Set J; no action taken\n")


        def _l_tri(model, i, j):
            self.assertIs(model, m)
            return i >= j
        m.K = Set(initialize=RangeSet(3)*RangeSet(3), filter=_l_tri)
        self.assertIsInstance(m.K.filter, IndexedCallInitializer)
        self.assertIs(m.K.filter._fcn, _l_tri)
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
        def _lt_3(model, i):
            self.assertIs(model, m)
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


        m = ConcreteModel()
        def _validate(model,i,j):
            self.assertIs(model, m)
            if i + j < 2:
                return True
            if i - j > 2:
                return False
            raise RuntimeError("Bogus value")
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
        m.I = Set()
        self.assertIs(m.I.domain, Any)

        m = ConcreteModel()
        m.I = Set(domain=Integers)
        self.assertIs(m.I.domain, Integers)
        m.I.add(1)
        m.I.add(2.)
        self.assertEqual(list(m.I), [1, 2.])
        with self.assertRaisesRegexp(
                ValueError, 'The value is not in the domain Integers'):
            m.I.add(1.5)

        m = ConcreteModel()
        m.I = Set(within=Integers)
        self.assertIs(m.I.domain, Integers)
        m.I.add(1)
        m.I.add(2.)
        self.assertEqual(list(m.I), [1, 2.])
        with self.assertRaisesRegexp(
                ValueError, 'The value is not in the domain Integers'):
            m.I.add(1.5)

        m = ConcreteModel()
        m.I = Set(bounds=(1,5))
        self.assertEqual(m.I.domain, RangeSet(1,5,0))
        m.I.add(1)
        m.I.add(2.)
        self.assertEqual(list(m.I), [1, 2.])
        with self.assertRaisesRegexp(
                ValueError, 'The value is not in the domain \[1..5\]'):
            m.I.add(5.5)

        m = ConcreteModel()
        m.I = Set(domain=Integers, within=RangeSet(0, None, 2), bounds=(0,9))
        self.assertEqual(m.I.domain, RangeSet(0,9,2))
        m.I = [0,2.,4]
        self.assertEqual(list(m.I), [0,2.,4])
        with self.assertRaisesRegexp(
                ValueError, 'The value is not in the domain '
                '\(Integers & I_domain_index_0_index_1'):
            m.I.add(1.5)
        with self.assertRaisesRegexp(
                ValueError, 'The value is not in the domain '
                '\(Integers & I_domain_index_0_index_1'):
            m.I.add(1)
        with self.assertRaisesRegexp(
                ValueError, 'The value is not in the domain '
                '\(Integers & I_domain_index_0_index_1'):
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
        m.M = Reals - SetOf([0])
        m.N = Integers - Reals

        buf = StringIO()
        m.pprint(ostream=buf)
        self.assertEqual(buf.getvalue().strip(), """
6 Set Declarations
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
    M : Size=1, Index=None, Ordered=False
        Key  : Dimen : Domain            : Size : Members
        None :     1 : Reals - M_index_1 :  Inf : ([None..0) | (0..None])
    N : Size=1, Index=None, Ordered=False
        Key  : Dimen : Domain           : Size : Members
        None :     1 : Integers - Reals :  Inf :      []

1 RangeSet Declarations
    I_index : Dimen=1, Size=3, Bounds=(1, 3)
        Key  : Finite : Members
        None :   True :   [1:3]

1 SetOf Declarations
    M_index_1 : Dimen=1, Size=1, Bounds=(0, 0)
        Key  : Ordered : Members
        None :    True :     [0]

8 Declarations: I_index I J K L M_index_1 M N""".strip())

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
        self.assertEqual(m.I.dimen, UnknownSetDimen)
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

        a = AbstractModel()
        a.I = Set(initialize=[1,2,3])
        self.assertEqual(a.I.dimen, UnknownSetDimen)
        a.J = Set(initialize=[1,2,3], dimen=1)
        self.assertEqual(a.J.dimen, 1)
        m = a.create_instance(data={None:{'I': {None:[(1,2), (3,4)]}}})
        self.assertEqual(m.I.dimen, 2)
        self.assertEqual(m.J.dimen, 1)

    def test_construction(self):
        m = AbstractModel()
        m.I = Set(initialize=[1,2,3])
        m.J = Set(initialize=[4,5,6])
        m.K = Set(initialize=[(1,4),(2,6),(3,5)], within=m.I*m.J)
        m.II = Set([1,2,3], initialize={1:[0], 2:[1,2], 3: xrange(3)})
        m.JJ = Set([1,2,3], initialize={1:[0], 2:[1,2], 3: xrange(3)})
        m.KK = Set([1,2], initialize=[], dimen=lambda m,i: i)

        output = StringIO()
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
            None: {'I': [-1,0], 'II': {1: [10,11], 3:[30]},
                   'K': [-1, 4, -1, 6, 0, 5]}
        })

        self.assertEqual(list(i.I), [-1,0])
        self.assertEqual(list(i.J), [4,5,6])
        self.assertEqual(list(i.K), [(-1,4),(-1,6),(0,5)])
        self.assertEqual(list(i.II[1]), [10,11])
        self.assertEqual(list(i.II[3]), [30])
        self.assertEqual(list(i.JJ[1]), [0])
        self.assertEqual(list(i.JJ[2]), [1,2])
        self.assertEqual(list(i.JJ[3]), [0,1,2])
        self.assertEqual(list(i.KK[1]), [])
        self.assertEqual(list(i.KK[2]), [])

        # Implicitly-constructed set should fall back on initialize!
        self.assertEqual(list(i.II[2]), [1,2])

        # Additional tests for tuplize:
        i = m.create_instance(data={
            None: {'K': [(1,4),(2,6)],
                   'KK': [1,4,2,6]}
        })
        self.assertEqual(list(i.K), [(1,4),(2,6)])
        self.assertEqual(list(i.KK), [1,2])
        self.assertEqual(list(i.KK[1]), [1,4,2,6])
        self.assertEqual(list(i.KK[2]), [(1,4),(2,6)])
        i = m.create_instance(data={
            None: {'K': []}
        })
        self.assertEqual(list(i.K), [])
        with self.assertRaisesRegexp(
                ValueError, "Cannot tuplize list data for set K because "
                "its length 3 is not a multiple of dimen=2"):
            i = m.create_instance(data={
                None: {'K': [1,2,3]}
            })
        with self.assertRaisesRegexp(
                ValueError, "Cannot tuplize list data for set KK\[2\] because "
                "its length 3 is not a multiple of dimen=2"):
            i = m.create_instance(data={
                None: {'KK': {2: [1,2,3]}}
            })

        ref = """
Constructing AbstractOrderedSimpleSet 'I' on [Model] from data=None
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

    def test_unconstructed_api(self):
        m = AbstractModel()
        m.I = Set(ordered=False)
        m.J = Set(ordered=Set.InsertionOrder)
        m.K = Set(ordered=Set.SortedOrder)

        with self.assertRaisesRegexp(
                RuntimeError,
                "Cannot iterate over AbstractFiniteSimpleSet 'I'"
                " before it has been constructed \(initialized\)."):
            for i in m.I:
                pass

        with self.assertRaisesRegexp(
                RuntimeError,
                "Cannot iterate over AbstractOrderedSimpleSet 'J'"
                " before it has been constructed \(initialized\)."):
            for i in m.J:
                pass

        with self.assertRaisesRegexp(
                RuntimeError,
                "Cannot iterate over AbstractSortedSimpleSet 'K'"
                " before it has been constructed \(initialized\)."):
            for i in m.K:
                pass

        with self.assertRaisesRegexp(
                RuntimeError,
                "Cannot test membership in AbstractFiniteSimpleSet 'I'"
                " before it has been constructed \(initialized\)."):
            1 in m.I

        with self.assertRaisesRegexp(
                RuntimeError,
                "Cannot test membership in AbstractOrderedSimpleSet 'J'"
                " before it has been constructed \(initialized\)."):
            1 in m.J

        with self.assertRaisesRegexp(
                RuntimeError,
                "Cannot test membership in AbstractSortedSimpleSet 'K'"
                " before it has been constructed \(initialized\)."):
            1 in m.K

        with self.assertRaisesRegexp(
                RuntimeError,
                "Cannot access __len__ on AbstractFiniteSimpleSet 'I'"
                " before it has been constructed \(initialized\)."):
            len(m.I)

        with self.assertRaisesRegexp(
                RuntimeError,
                "Cannot access __len__ on AbstractOrderedSimpleSet 'J'"
                " before it has been constructed \(initialized\)."):
            len(m.J)

        with self.assertRaisesRegexp(
                RuntimeError,
                "Cannot access __len__ on AbstractSortedSimpleSet 'K'"
                " before it has been constructed \(initialized\)."):
            len(m.K)

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

        # Backwards compatability: Test rule for indexed component that
        # does not take the index
        @simple_set_rule
        def _k_init(m):
            return [1,2,3]
        m.K = Set([1], initialize=_k_init)
        self.assertEqual(list(m.K[1]), [1,2,3])


        @simple_set_rule
        def _l_init(m, l):
            if l > 3:
                return None
            return tuple(range(l))
        m.L = Set(initialize=_l_init, dimen=None)
        self.assertEqual(list(m.L), [0, (0,1), (0,1,2)])

        m.M = Set([1,2,3], initialize=_l_init)
        self.assertEqual(list(m.M), [1,2,3])
        self.assertEqual(list(m.M[1]), [0])
        self.assertEqual(list(m.M[2]), [0,1])
        self.assertEqual(list(m.M[3]), [0,1,2])


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

    def test_process_setarg(self):
        m = AbstractModel()
        m.I = Set([1,2,3])
        self.assertTrue(m.I.index_set().is_constructed())
        self.assertTrue(m.I.index_set().isordered())
        i = m.create_instance()
        self.assertEqual(i.I.index_set(), [1,2,3])

        m = AbstractModel()
        m.I = Set({1,2,3})
        self.assertTrue(m.I.index_set().is_constructed())
        self.assertFalse(m.I.index_set().isordered())
        i = m.create_instance()
        self.assertEqual(i.I.index_set(), [1,2,3])

        m = AbstractModel()
        m.I = Set(RangeSet(3))
        self.assertTrue(m.I.index_set().is_constructed())
        self.assertTrue(m.I.index_set().isordered())
        i = m.create_instance()
        self.assertEqual(i.I.index_set(), [1,2,3])

        m = AbstractModel()
        m.p = Param(initialize=3)
        m.I = Set(RangeSet(m.p))
        self.assertFalse(m.I.index_set().is_constructed())
        self.assertTrue(m.I.index_set().isordered())
        i = m.create_instance()
        self.assertEqual(i.I.index_set(), [1,2,3])

        m = AbstractModel()
        m.I = Set(lambda m: [1,2,3])
        self.assertFalse(m.I.index_set().is_constructed())
        self.assertTrue(m.I.index_set().isordered())
        i = m.create_instance()
        self.assertEqual(i.I.index_set(), [1,2,3])

        def _i_idx(m):
            return [1,2,3]
        m = AbstractModel()
        m.I = Set(_i_idx)
        self.assertFalse(m.I.index_set().is_constructed())
        self.assertTrue(m.I.index_set().isordered())
        i = m.create_instance()
        self.assertEqual(i.I.index_set(), [1,2,3])

        # Note: generators are uncopyable, so we will mock up the same
        # behavior as above using an unconstructed block
        def _i_idx():
            yield 1
            yield 2
            yield 3
        m = Block()
        m.I = Set(_i_idx())
        self.assertFalse(m.I.index_set().is_constructed())
        self.assertTrue(m.I.index_set().isordered())
        i = ConcreteModel()
        i.m = m
        self.assertEqual(i.m.I.index_set(), [1,2,3])

    def test_set_options(self):
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            @set_options(domain=Integers)
            def Bindex(m):
                return range(5)
        self.assertIn(
            "The set_options decorator is deprecated",
            output.getvalue())

        m = ConcreteModel()
        m.I = Set(initialize=[8,9])
        m.J = m.I.cross(Bindex)
        self.assertIs(m.J._sets[1]._domain, Integers)

        m.K = Set(Bindex)
        self.assertIs(m.K.index_set()._domain, Integers)
        self.assertEqual(m.K.index_set(), [0,1,2,3,4])

    def test_no_normalize_index(self):
        try:
            _oldFlatten = normalize_index.flatten

            normalize_index.flatten = False
            m = ConcreteModel()
            m.I = Set()
            self.assertIs(m.I._dimen, UnknownSetDimen)
            self.assertTrue(m.I.add((1,(2,3))))
            self.assertIs(m.I._dimen, None)
            self.assertNotIn(((1,2),3), m.I)
            self.assertIn((1,(2,3)), m.I)
            self.assertNotIn((1,2,3), m.I)

            m.J = Set()
            self.assertTrue(m.J.add(1))
            self.assertIn(1, m.J)
            self.assertNotIn((1,), m.J)
            self.assertTrue(m.J.add((1,)))
            self.assertIn(1, m.J)
            self.assertIn((1,), m.J)
            self.assertTrue(m.J.add((2,)))
            self.assertNotIn(2, m.J)
            self.assertIn((2,), m.J)

            normalize_index.flatten = True
            m = ConcreteModel()
            m.I = Set()
            self.assertIs(m.I._dimen, UnknownSetDimen)
            m.I.add((1,(2,3)))
            self.assertIs(m.I._dimen, 3)
            self.assertIn(((1,2),3), m.I)
            self.assertIn((1,(2,3)), m.I)
            self.assertIn((1,2,3), m.I)

            m.J = Set()
            self.assertTrue(m.J.add(1))
            self.assertIn(1, m.J)
            self.assertIn((1,), m.J)
            self.assertFalse(m.J.add((1,))) # Not added!
            self.assertIn(1, m.J)
            self.assertIn((1,), m.J)
            self.assertTrue(m.J.add((2,)))
            self.assertIn(2, m.J)
            self.assertIn((2,), m.J)
        finally:
            normalize_index.flatten = _oldFlatten


class TestAbstractSetAPI(unittest.TestCase):
    def test_SetData(self):
        # This tests an anstract non-finite set API

        m = ConcreteModel()
        m.I = Set(initialize=[1])
        s = _SetData(m.I)

        #
        # _SetData API
        #

        with self.assertRaises(DeveloperError):
            # __contains__
            None in s

        with self.assertRaises(DeveloperError):
            s == m.I
        with self.assertRaises(DeveloperError):
            m.I == s
        with self.assertRaises(DeveloperError):
            s != m.I
        with self.assertRaises(DeveloperError):
            m.I != s

        with self.assertRaises(DeveloperError):
            str(s)
        with self.assertRaises(DeveloperError):
            s.dimen
        with self.assertRaises(DeveloperError):
            s.domain

        self.assertFalse(s.isfinite())
        self.assertFalse(s.isordered())

        with self.assertRaises(DeveloperError):
            s.ranges()

        with self.assertRaises(DeveloperError):
            s.isdisjoint(m.I)
        with self.assertRaises(DeveloperError):
            m.I.isdisjoint(s)

        with self.assertRaises(DeveloperError):
            s.issuperset(m.I)
        with self.assertRaises(DeveloperError):
            m.I.issuperset(s)

        with self.assertRaises(DeveloperError):
            s.issubset(m.I)
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

        with self.assertRaises(DeveloperError):
            s < m.I
        with self.assertRaises(DeveloperError):
            m.I < s

        with self.assertRaises(DeveloperError):
            s > m.I
        with self.assertRaises(DeveloperError):
            m.I > s

    def test_FiniteMixin(self):
        # This tests an anstract finite set API
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
        with self.assertRaises(DeveloperError):
            s.domain

        self.assertTrue(s.isfinite())
        self.assertFalse(s.isordered())

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
        # This tests an anstract ordered set API
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
        with self.assertRaises(DeveloperError):
            s.domain

        self.assertTrue(s.isfinite())
        self.assertTrue(s.isordered())

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


class TestSetUtils(unittest.TestCase):
    def test_get_continuous_interval(self):
        self.assertEqual(Reals.get_interval(), (None,None,0))
        self.assertEqual(PositiveReals.get_interval(), (0,None,0))
        self.assertEqual(NonNegativeReals.get_interval(), (0,None,0))
        self.assertEqual(NonPositiveReals.get_interval(), (None,0,0))
        self.assertEqual(NegativeReals.get_interval(), (None,0,0))

        a = NonNegativeReals | NonPositiveReals
        self.assertEqual(a.get_interval(), (None, None, 0))
        a = NonPositiveReals | NonNegativeReals
        self.assertEqual(a.get_interval(), (None, None, 0))

        a = NegativeReals | PositiveReals
        self.assertEqual(a.get_interval(), (None, None, None))
        a = NegativeReals | PositiveReals | [0]
        self.assertEqual(a.get_interval(), (None, None, 0))
        a = NegativeReals | PositiveReals | RangeSet(0,5)
        self.assertEqual(a.get_interval(), (None, None, 0))

        a = NegativeReals | RangeSet(-3, 3)
        self.assertEqual(a.get_interval(), (None, 3, None))
        a = NegativeReals | Binary
        self.assertEqual(a.get_interval(), (None, 1, None))
        a = PositiveReals | Binary
        self.assertEqual(a.get_interval(), (0, None, 0))

        a = RangeSet(1,10,0) | RangeSet(5,15,0)
        self.assertEqual(a.get_interval(), (1,15,0))
        a = RangeSet(5,15,0) | RangeSet(1,10,0)
        self.assertEqual(a.get_interval(), (1,15,0))

        a = RangeSet(5,15,0) | RangeSet(1,4,0)
        self.assertEqual(a.get_interval(), (1, 15, None))
        a = RangeSet(1,4,0) | RangeSet(5,15,0)
        self.assertEqual(a.get_interval(), (1, 15, None))

        a = NegativeReals | Any
        self.assertEqual(a.get_interval(), (None, None, None))
        a = Any | NegativeReals
        self.assertEqual(a.get_interval(), (None, None, None))
        a = SetOf('abc') | NegativeReals
        self.assertEqual(a.get_interval(), (None, None, None))
        a = NegativeReals | SetOf('abc')
        self.assertEqual(a.get_interval(), (None, None, None))

    def test_get_discrete_interval(self):
        self.assertEqual(Integers.get_interval(), (None,None,1))
        self.assertEqual(PositiveIntegers.get_interval(), (1,None,1))
        self.assertEqual(NegativeIntegers.get_interval(), (None,-1,1))
        self.assertEqual(Binary.get_interval(), (0,1,1))

        a = PositiveIntegers | NegativeIntegers
        self.assertEqual(a.get_interval(), (None, None, None))
        a = NegativeIntegers | NonNegativeIntegers
        self.assertEqual(a.get_interval(), (None, None, 1))

        a = SetOf([1,3,5,6,4,2])
        self.assertEqual(a.get_interval(), (1, 6, 1))
        a = SetOf([1,3,5,6,2])
        self.assertEqual(a.get_interval(), (1, 6, None))
        a = SetOf([1,3,5,6,4,2,'a'])
        self.assertEqual(a.get_interval(), (None, None, None))
        a = SetOf([3])
        self.assertEqual(a.get_interval(), (3,3,0))

        a = RangeSet(ranges=(NR(0,5,1), NR(5,10,1)))
        self.assertEqual(a.get_interval(), (0, 10, 1))
        a = RangeSet(ranges=(NR(5,10,1), NR(0,5,1)))
        self.assertEqual(a.get_interval(), (0, 10, 1))

        a = RangeSet(ranges=(NR(0,4,1), NR(5,10,1)))
        self.assertEqual(a.get_interval(), (0, 10, 1))
        a = RangeSet(ranges=(NR(5,10,1), NR(0,4,1)))
        self.assertEqual(a.get_interval(), (0, 10, 1))

        a = RangeSet(ranges=(NR(0,3,1), NR(5,10,1)))
        self.assertEqual(a.get_interval(), (0, 10, None))
        a = RangeSet(ranges=(NR(5,10,1), NR(0,3,1)))
        self.assertEqual(a.get_interval(), (0, 10, None))

        a = RangeSet(ranges=(NR(0,4,2), NR(6,10,2)))
        self.assertEqual(a.get_interval(), (0, 10, 2))
        a = RangeSet(ranges=(NR(6,10,2), NR(0,4,2)))
        self.assertEqual(a.get_interval(), (0, 10, 2))

        a = RangeSet(ranges=(NR(0,4,2), NR(5,10,2)))
        self.assertEqual(a.get_interval(), (0, 9, None))
        a = RangeSet(ranges=(NR(5,10,2), NR(0,4,2)))
        self.assertEqual(a.get_interval(), (0, 9, None))

        a = RangeSet(ranges=(NR(0,10,2), NR(0,10,3)))
        self.assertEqual(a.get_interval(), (0, 10, None))
        a = RangeSet(ranges=(NR(0,10,3), NR(0,10,2)))
        self.assertEqual(a.get_interval(), (0, 10, None))

        a = RangeSet(ranges=(NR(2,10,2), NR(0,12,4)))
        self.assertEqual(a.get_interval(), (0,12,2))
        a = RangeSet(ranges=(NR(0,12,4), NR(2,10,2)))
        self.assertEqual(a.get_interval(), (0,12,2))

        # Even though the following are reasonable intervals, we
        # currently don't support resolving it:
        a = RangeSet(ranges=(NR(0,10,2), NR(1,10,2)))
        self.assertEqual(a.get_interval(), (0, 10, None))
        a = RangeSet(ranges=(NR(0,10,3), NR(1,10,3), NR(2,10,3)))
        self.assertEqual(a.get_interval(), (0, 10, None))


class TestDeprecation(unittest.TestCase):
    def test_filter(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1,2,3])
        m.J = m.I*m.I
        m.K = Set(initialize=[1,2,3], filter=lambda m,i: i%2)

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core', logging.DEBUG):
            self.assertIsNone(m.I.filter)
        self.assertRegexpMatches(
            output.getvalue(),
            "^DEPRECATED: 'filter' is no longer a public attribute")

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core', logging.DEBUG):
            self.assertIsNone(m.J.filter)
        self.assertRegexpMatches(
            output.getvalue(),
            "^DEPRECATED: 'filter' is no longer a public attribute")

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core', logging.DEBUG):
            self.assertIsInstance(m.K.filter, IndexedCallInitializer)
        self.assertRegexpMatches(
            output.getvalue(),
            "^DEPRECATED: 'filter' is no longer a public attribute")

    def test_virtual(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1,2,3])
        m.J = m.I*m.I

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core', logging.DEBUG):
            self.assertFalse(m.I.virtual)
        self.assertRegexpMatches(
            output.getvalue(),
            "^DEPRECATED: The 'virtual' attribute is no longer supported")

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core', logging.DEBUG):
            self.assertTrue(m.J.virtual)
        self.assertRegexpMatches(
            output.getvalue(),
            "^DEPRECATED: The 'virtual' attribute is no longer supported")

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            m.J.virtual = True
        self.assertRegexpMatches(
            output.getvalue(),
            "^DEPRECATED: The 'virtual' attribute is no longer supported")
        with self.assertRaisesRegexp(
                ValueError,
                "Attempting to set the \(deprecated\) 'virtual' attribute on J "
                "to an invalid value \(False\)"):
            m.J.virtual = False

    def test_concrete(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1,2,3])
        m.J = m.I*m.I

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core', logging.DEBUG):
            self.assertTrue(m.I.concrete)
        self.assertRegexpMatches(
            output.getvalue(),
            "^DEPRECATED: The 'concrete' attribute is no longer supported")

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core', logging.DEBUG):
            self.assertTrue(m.J.concrete)
        self.assertRegexpMatches(
            output.getvalue(),
            "^DEPRECATED: The 'concrete' attribute is no longer supported")

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core', logging.DEBUG):
            self.assertFalse(Reals.concrete)
        self.assertRegexpMatches(
            output.getvalue(),
            "^DEPRECATED: The 'concrete' attribute is no longer supported")

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            m.J.concrete = True
        self.assertRegexpMatches(
            output.getvalue(),
            "^DEPRECATED: The 'concrete' attribute is no longer supported.")
        with self.assertRaisesRegexp(
                ValueError,
                "Attempting to set the \(deprecated\) 'concrete' attribute on "
                "J to an invalid value \(False\)"):
            m.J.concrete = False

    def test_ordered_attr(self):
        m = ConcreteModel()
        m.J = Set(ordered=True)
        m.K = Set(ordered=False)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            self.assertTrue(m.J.ordered)
        self.assertRegexpMatches(
            output.getvalue(),
            "^DEPRECATED: The 'ordered' attribute is no longer supported.")
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            self.assertFalse(m.K.ordered)
        self.assertRegexpMatches(
            output.getvalue(),
            "^DEPRECATED: The 'ordered' attribute is no longer supported.")

    def test_value_attr(self):
        m = ConcreteModel()
        m.J = Set(ordered=True, initialize=[1,3,2])
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            tmp = m.J.value
        self.assertIs(type(tmp), set)
        self.assertEqual(tmp, set([1,3,2]))
        self.assertRegexpMatches(
            output.getvalue(),
            "^DEPRECATED: The 'value' attribute is deprecated.  Use .data\(\)")

    def test_value_list_attr(self):
        m = ConcreteModel()
        m.J = Set(ordered=True, initialize=[1,3,2])
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            tmp = m.J.value_list
        self.assertIs(type(tmp), list)
        self.assertEqual(tmp, list([1,3,2]))
        self.assertRegexpMatches(
            output.getvalue().replace('\n',' '),
            "^DEPRECATED: The 'value_list' attribute is deprecated.  "
            "Use .ordered_data\(\)")

    def test_check_values(self):
        m = ConcreteModel()
        m.I = Set(ordered=True, initialize=[1,3,2])
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            self.assertTrue(m.I.check_values())
        self.assertRegexpMatches(
            output.getvalue(),
            "^DEPRECATED: check_values\(\) is deprecated: Sets only "
            "contain valid")

        m.J = m.I*m.I
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core', logging.DEBUG):
            self.assertTrue(m.J.check_values())
        self.assertRegexpMatches(
            output.getvalue(),
            "^DEPRECATED: check_values\(\) is deprecated:")

        # We historically supported check_values on indexed sets
        m.K = Set([1,2], ordered=True, initialize=[1,3,2])
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            self.assertTrue(m.K.check_values())
        self.assertRegexpMatches(
            output.getvalue(),
            "^DEPRECATED: check_values\(\) is deprecated: Sets only "
            "contain valid")


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

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            self.assertTrue(m.s in m.s)
        self.assertIn(
            "Testing for set subsets with 'a in b' is deprecated.",
            output.getvalue()
        )
        if PY2:
            self.assertFalse(m.s in m.t)
            with self.assertRaisesRegexp(KeyError, "Index 's' is not valid"):
                m.x[m.s].display()
        else:
            # Note that pypy raises a different exception from cpython
            err = "((unhashable type: 'OrderedSimpleSet')" \
                "|('OrderedSimpleSet' objects are unhashable))"
            with self.assertRaisesRegexp(TypeError, err):
                self.assertFalse(m.s in m.t)
            with self.assertRaisesRegexp(TypeError, err):
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

    def test_issue_165(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var(domain=Binary)
        m_binaries = [
            v for v in m.component_data_objects(
                ctype=Var, descend_into=True)
            if v.domain is Binary and not v.fixed
        ]
        self.assertEqual(len(m_binaries), 1)
        self.assertIs(m_binaries[0], m.y)

        m2 = m.clone()
        m2_binaries = [
            v for v in m2.component_data_objects(
                ctype=Var, descend_into=True)
            if v.domain is Binary and not v.fixed
        ]
        self.assertEqual(len(m2_binaries), 1)
        self.assertIs(m2_binaries[0], m2.y)

    def test_issue_191(self):
        m = ConcreteModel()
        m.s = Set(['s1','s2'], initialize=[1,2,3])
        m.s2 = Set(initialize=['a','b','c'])

        m.p = Param(m.s['s1'], initialize=10)
        temp = m.s['s1'] * m.s2
        m.v = Var(temp, initialize=5)
        self.assertEqual(len(m.v), 9)

        m.v_1 = Var(m.s['s1'], m.s2, initialize=10)
        self.assertEqual(len(m.v_1), 9)

    def test_issue_325(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2, 'a', 3], ordered=False)
        self.assertEqual(set(m.I.data()), set([1, 2, 'a', 3]))
        self.assertEqual(list(m.I.ordered_data()), [1, 2, 3, 'a'])
        self.assertEqual(list(m.I.sorted_data()), [1, 2, 3, 'a'])

        # Default sets are ordered by insertion order
        m.I = Set(initialize=[1, 2, 'a', 3])
        self.assertEqual(set(m.I.data()), set([1, 2, 'a', 3]))
        self.assertEqual(list(m.I.data()), [1, 2, 'a', 3])
        self.assertEqual(list(m.I.ordered_data()), [1, 2, 'a', 3])
        self.assertEqual(list(m.I.sorted_data()), [1, 2, 3, 'a'])

    def test_issue_358(self):
        m = ConcreteModel()
        m.s = RangeSet(1)
        m.s2 = RangeSet(1)
        m.set_mult = m.s * m.s2
        m.s3 = RangeSet(1)

        def _test(b, x, y, z):
            print(x, y, z)
        m.test = Block(m.set_mult, m.s3, rule=_test)
        self.assertEqual(len(m.test), 1)
        m.test2 = Block(m.set_mult, m.s3, rule=_test)
        self.assertEqual(len(m.test2), 1)

    def test_issue_637(self):
        constraints = {
            c for c in itertools.product(['constrA', 'constrB'], range(5))
        }
        vars = {
            v for v in itertools.product(['var1', 'var2', 'var3'], range(5))
        }
        matrix_coefficients = {m for m in itertools.product(constraints, vars)}
        m = ConcreteModel()
        m.IDX = Set(initialize=matrix_coefficients)
        m.Matrix = Param(m.IDX, default=0)
        self.assertEqual(len(m.Matrix), 2*5*3*5)

    def test_issue_758(self):
        m = ConcreteModel()
        m.I = RangeSet(5)

        self.assertEqual(m.I.next(1), 2)
        self.assertEqual(m.I.next(4), 5)
        with self.assertRaisesRegexp(
                IndexError, "Cannot advance past the end of the Set"):
            m.I.next(5)

        self.assertEqual(m.I.prev(2), 1)
        self.assertEqual(m.I.prev(5), 4)
        with self.assertRaisesRegexp(
                IndexError, "Cannot advance before the beginning of the Set"):
            m.I.prev(1)

        self.assertEqual(m.I.nextw(1), 2)
        self.assertEqual(m.I.nextw(4), 5)
        self.assertEqual(m.I.nextw(5), 1)

        self.assertEqual(m.I.prevw(2), 1)
        self.assertEqual(m.I.prevw(5), 4)
        self.assertEqual(m.I.prevw(1), 5)

    def test_issue_835(self):
        a = ["a", "x", "c", "b"]

        model = ConcreteModel()
        model.S = Set(initialize=a)
        model.OS = Set(initialize=a, ordered=True)

        self.assertEqual(list(model.S), a)
        self.assertEqual(list(model.OS), a)

        out1 = StringIO()
        model.S.pprint(ostream=out1)
        out2 = StringIO()
        model.OS.pprint(ostream=out2)

        self.assertEqual(
            out1.getvalue().strip(),
            out2.getvalue().strip()[1:],
        )

    @unittest.skipIf(NamedTuple is None, "typing module not available")
    def test_issue_938(self):
        NodeKey = NamedTuple('NodeKey', [('id', int)])
        ArcKey = NamedTuple('ArcKey',
                            [('node_from', NodeKey), ('node_to', NodeKey)])
        def build_model():
            model = ConcreteModel()
            model.node_keys = Set(doc='Set of nodes',
                                  initialize=[NodeKey(0), NodeKey(1)])
            model.arc_keys = Set(doc='Set of arcs',
                                 within=model.node_keys * model.node_keys,
                                 initialize=[
                                     ArcKey(NodeKey(0), NodeKey(0)),
                                     ArcKey(NodeKey(0), NodeKey(1)),
                                 ])
            model.arc_variables = Var(model.arc_keys,
                                      within=Binary)

            def objective_rule(model_arg):
                return sum(var for var in model_arg.arc_variables.values())
            model.obj = Objective(rule=objective_rule)
            return model

        try:
            _oldFlatten = normalize_index.flatten

            normalize_index.flatten = True
            m = build_model()
            output = StringIO()
            m.pprint(ostream=output)
            ref = """
3 Set Declarations
    arc_keys : Set of arcs
        Size=1, Index=None, Ordered=Insertion
        Key  : Dimen : Domain          : Size : Members
        None :     2 : arc_keys_domain :    2 : {(0, 0), (0, 1)}
    arc_keys_domain : Size=1, Index=None, Ordered=True
        Key  : Dimen : Domain              : Size : Members
        None :     2 : node_keys*node_keys :    4 : {(0, 0), (0, 1), (1, 0), (1, 1)}
    node_keys : Set of nodes
        Size=1, Index=None, Ordered=Insertion
        Key  : Dimen : Domain : Size : Members
        None :     1 :    Any :    2 : {0, 1}

1 Var Declarations
    arc_variables : Size=2, Index=arc_keys
        Key    : Lower : Value : Upper : Fixed : Stale : Domain
        (0, 0) :     0 :  None :     1 : False :  True : Binary
        (0, 1) :     0 :  None :     1 : False :  True : Binary

1 Objective Declarations
    obj : Size=1, Index=None, Active=True
        Key  : Active : Sense    : Expression
        None :   True : minimize : arc_variables[0,0] + arc_variables[0,1]

5 Declarations: node_keys arc_keys_domain arc_keys arc_variables obj
""".strip()
            self.assertEqual(output.getvalue().strip(), ref)

            normalize_index.flatten = False
            m = build_model()
            output = StringIO()
            m.pprint(ostream=output)
            ref = """
3 Set Declarations
    arc_keys : Set of arcs
        Size=1, Index=None, Ordered=Insertion
        Key  : Dimen : Domain          : Size : Members
        None :  None : arc_keys_domain :    2 : {ArcKey(node_from=NodeKey(id=0), node_to=NodeKey(id=0)), ArcKey(node_from=NodeKey(id=0), node_to=NodeKey(id=1))}
    arc_keys_domain : Size=1, Index=None, Ordered=True
        Key  : Dimen : Domain              : Size : Members
        None :  None : node_keys*node_keys :    4 : {(NodeKey(id=0), NodeKey(id=0)), (NodeKey(id=0), NodeKey(id=1)), (NodeKey(id=1), NodeKey(id=0)), (NodeKey(id=1), NodeKey(id=1))}
    node_keys : Set of nodes
        Size=1, Index=None, Ordered=Insertion
        Key  : Dimen : Domain : Size : Members
        None :  None :    Any :    2 : {NodeKey(id=0), NodeKey(id=1)}

1 Var Declarations
    arc_variables : Size=2, Index=arc_keys
        Key                                                    : Lower : Value : Upper : Fixed : Stale : Domain
        ArcKey(node_from=NodeKey(id=0), node_to=NodeKey(id=0)) :     0 :  None :     1 : False :  True : Binary
        ArcKey(node_from=NodeKey(id=0), node_to=NodeKey(id=1)) :     0 :  None :     1 : False :  True : Binary

1 Objective Declarations
    obj : Size=1, Index=None, Active=True
        Key  : Active : Sense    : Expression
        None :   True : minimize : arc_variables[ArcKey(node_from=NodeKey(id=0), node_to=NodeKey(id=0))] + arc_variables[ArcKey(node_from=NodeKey(id=0), node_to=NodeKey(id=1))]

5 Declarations: node_keys arc_keys_domain arc_keys arc_variables obj
""".strip()
            self.assertEqual(output.getvalue().strip(), ref)

        finally:
            normalize_index.flatten = _oldFlatten

    def test_issue_1375(self):
        def a_rule(m):
            for i in range(0):
                yield i

        def b_rule(m):
            for i in range(3):
                for j in range(0):
                    yield i, j

        m = ConcreteModel()
        m.a = Set(initialize=a_rule, dimen=1)
        self.assertEqual(len(m.a), 0)
        m.b = Set(initialize=b_rule, dimen=2)
        self.assertEqual(len(m.b), 0)
