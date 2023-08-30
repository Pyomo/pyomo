#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
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

currdir = dirname(abspath(__file__)) + os.sep

import pyomo.common.unittest as unittest
from io import StringIO

from pyomo.environ import (
    ConcreteModel,
    Block,
    Var,
    Set,
    RangeSet,
    Param,
    value,
    NonNegativeIntegers,
)
from pyomo.common.collections import ComponentSet
from pyomo.common.log import LoggingIntercept
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.set import (
    SetProduct,
    FiniteSetOf,
    OrderedSetOf,
    UnknownSetDimen,
    normalize_index,
)
from pyomo.core.base.indexed_component import UnindexedComponent_set, IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.reference import (
    _ReferenceDict,
    _ReferenceSet,
    Reference,
    UnindexedComponent_ReferenceSet,
)


class TestReferenceDict(unittest.TestCase):
    def setUp(self):
        self.m = m = ConcreteModel()

        @m.Block([1, 2], [4, 5])
        def b(b, i, j):
            b.x = Var([7, 8], [10, 11], initialize=0)
            b.y = Var([7, 8], initialize=0)
            b.z = Var()

        @m.Block([1, 2])
        def c(b, i):
            b.x = Var([7, 8], [10, 11], initialize=0)
            b.y = Var([7, 8], initialize=0)
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
            _ = tuple([0] * i)
            self.assertNotIn(_, rd)
            with self.assertRaises(KeyError):
                rd[_]

    def test_simple_lookup(self):
        m = self.m

        self._lookupTester(m.b[:, :].x[:, :], (1, 5, 7, 10), m.b[1, 5].x[7, 10])
        self._lookupTester(m.b[:, 4].x[8, :], (1, 10), m.b[1, 4].x[8, 10])
        self._lookupTester(m.b[:, 4].x[8, 10], (1,), m.b[1, 4].x[8, 10])
        self._lookupTester(m.b[1, 4].x[8, :], (10,), m.b[1, 4].x[8, 10])

        self._lookupTester(m.b[:, :].y[:], (1, 5, 7), m.b[1, 5].y[7])
        self._lookupTester(m.b[:, 4].y[:], (1, 7), m.b[1, 4].y[7])
        self._lookupTester(m.b[:, 4].y[8], (1,), m.b[1, 4].y[8])

        self._lookupTester(m.b[:, :].z, (1, 5), m.b[1, 5].z)
        self._lookupTester(m.b[:, 4].z, (1,), m.b[1, 4].z)

        self._lookupTester(m.c[:].x[:, :], (1, 7, 10), m.c[1].x[7, 10])
        self._lookupTester(m.c[:].x[8, :], (1, 10), m.c[1].x[8, 10])
        self._lookupTester(m.c[:].x[8, 10], (1,), m.c[1].x[8, 10])
        self._lookupTester(m.c[1].x[:, :], (8, 10), m.c[1].x[8, 10])
        self._lookupTester(m.c[1].x[8, :], (10,), m.c[1].x[8, 10])

        self._lookupTester(m.c[:].y[:], (1, 7), m.c[1].y[7])
        self._lookupTester(m.c[:].y[8], (1,), m.c[1].y[8])
        self._lookupTester(m.c[1].y[:], (8,), m.c[1].y[8])

        self._lookupTester(m.c[:].z, (1,), m.c[1].z)

        m.jagged_set = Set(initialize=[1, (2, 3)], dimen=None)
        m.jb = Block(m.jagged_set)
        m.jb[1].x = Var([1, 2, 3])
        m.jb[2, 3].x = Var([1, 2, 3])
        self._lookupTester(m.jb[...], (1,), m.jb[1])
        self._lookupTester(m.jb[...].x[:], (1, 2), m.jb[1].x[2])
        self._lookupTester(m.jb[...].x[:], (2, 3, 2), m.jb[2, 3].x[2])

        rd = _ReferenceDict(m.jb[:, :, :].x[:])
        with self.assertRaises(KeyError):
            rd[2, 3, 4, 2]
        rd = _ReferenceDict(m.b[:, 4].x[:])
        with self.assertRaises(KeyError):
            rd[1, 0]

    def test_len(self):
        m = self.m

        rd = _ReferenceDict(m.b[:, :].x[:, :])
        self.assertEqual(len(rd), 2 * 2 * 2 * 2)

        rd = _ReferenceDict(m.b[:, 4].x[8, :])
        self.assertEqual(len(rd), 2 * 2)

    def test_iterators(self):
        m = self.m
        rd = _ReferenceDict(m.b[:, 4].x[8, :])

        self.assertEqual(list(rd), [(1, 10), (1, 11), (2, 10), (2, 11)])
        self.assertEqual(list(rd.keys()), [(1, 10), (1, 11), (2, 10), (2, 11)])
        self.assertEqual(
            list(rd.values()),
            [
                m.b[1, 4].x[8, 10],
                m.b[1, 4].x[8, 11],
                m.b[2, 4].x[8, 10],
                m.b[2, 4].x[8, 11],
            ],
        )
        self.assertEqual(
            list(rd.items()),
            [
                ((1, 10), m.b[1, 4].x[8, 10]),
                ((1, 11), m.b[1, 4].x[8, 11]),
                ((2, 10), m.b[2, 4].x[8, 10]),
                ((2, 11), m.b[2, 4].x[8, 11]),
            ],
        )

    def test_ordered_iterators(self):
        # Test slice; common indexing set
        m = ConcreteModel()
        m.I = Set(initialize=[3, 2])
        m.b = Block([1, 0])
        m.b[1].x = Var(m.I)
        m.b[0].x = Var(m.I)
        m.y = Reference(m.b[:].x[:])
        self.assertEqual(list(m.y.index_set().subsets()), [m.b.index_set(), m.I])
        self.assertEqual(list(m.y), [(1, 3), (1, 2), (0, 3), (0, 2)])
        self.assertEqual(list(m.y.keys()), [(1, 3), (1, 2), (0, 3), (0, 2)])
        self.assertEqual(
            list(m.y.values()), [m.b[1].x[3], m.b[1].x[2], m.b[0].x[3], m.b[0].x[2]]
        )
        self.assertEqual(
            list(m.y.items()),
            [
                ((1, 3), m.b[1].x[3]),
                ((1, 2), m.b[1].x[2]),
                ((0, 3), m.b[0].x[3]),
                ((0, 2), m.b[0].x[2]),
            ],
        )
        self.assertEqual(list(m.y.keys(True)), [(0, 2), (0, 3), (1, 2), (1, 3)])
        self.assertEqual(
            list(m.y.values(True)), [m.b[0].x[2], m.b[0].x[3], m.b[1].x[2], m.b[1].x[3]]
        )
        self.assertEqual(
            list(m.y.items(True)),
            [
                ((0, 2), m.b[0].x[2]),
                ((0, 3), m.b[0].x[3]),
                ((1, 2), m.b[1].x[2]),
                ((1, 3), m.b[1].x[3]),
            ],
        )

        # Test slice; ReferenceSet indexing set
        m = ConcreteModel()
        m.b = Block([1, 0])
        m.b[1].x = Var([3, 2])
        m.b[0].x = Var([5, 4])
        m.y = Reference(m.b[:].x[:])
        self.assertIs(type(m.y.index_set()), FiniteSetOf)
        self.assertEqual(list(m.y), [(1, 3), (1, 2), (0, 5), (0, 4)])
        self.assertEqual(list(m.y.keys()), [(1, 3), (1, 2), (0, 5), (0, 4)])
        self.assertEqual(
            list(m.y.values()), [m.b[1].x[3], m.b[1].x[2], m.b[0].x[5], m.b[0].x[4]]
        )
        self.assertEqual(
            list(m.y.items()),
            [
                ((1, 3), m.b[1].x[3]),
                ((1, 2), m.b[1].x[2]),
                ((0, 5), m.b[0].x[5]),
                ((0, 4), m.b[0].x[4]),
            ],
        )
        self.assertEqual(list(m.y.keys(True)), [(0, 4), (0, 5), (1, 2), (1, 3)])
        self.assertEqual(
            list(m.y.values(True)), [m.b[0].x[4], m.b[0].x[5], m.b[1].x[2], m.b[1].x[3]]
        )
        self.assertEqual(
            list(m.y.items(True)),
            [
                ((0, 4), m.b[0].x[4]),
                ((0, 5), m.b[0].x[5]),
                ((1, 2), m.b[1].x[2]),
                ((1, 3), m.b[1].x[3]),
            ],
        )

        # Test dict, ReferenceSet indexing set
        m = ConcreteModel()
        m.b = Block([1, 0])
        m.b[1].x = Var([3, 2])
        m.b[0].x = Var([5, 4])
        m.y = Reference(
            {
                (1, 3): m.b[1].x[3],
                (0, 5): m.b[0].x[5],
                (1, 2): m.b[1].x[2],
                (0, 4): m.b[0].x[4],
            }
        )
        self.assertIs(type(m.y.index_set()), FiniteSetOf)
        self.assertEqual(list(m.y), [(1, 3), (0, 5), (1, 2), (0, 4)])
        self.assertEqual(list(m.y.keys()), [(1, 3), (0, 5), (1, 2), (0, 4)])
        self.assertEqual(
            list(m.y.values()), [m.b[1].x[3], m.b[0].x[5], m.b[1].x[2], m.b[0].x[4]]
        )
        self.assertEqual(
            list(m.y.items()),
            [
                ((1, 3), m.b[1].x[3]),
                ((0, 5), m.b[0].x[5]),
                ((1, 2), m.b[1].x[2]),
                ((0, 4), m.b[0].x[4]),
            ],
        )
        self.assertEqual(list(m.y.keys(True)), [(0, 4), (0, 5), (1, 2), (1, 3)])
        self.assertEqual(
            list(m.y.values(True)), [m.b[0].x[4], m.b[0].x[5], m.b[1].x[2], m.b[1].x[3]]
        )
        self.assertEqual(
            list(m.y.items(True)),
            [
                ((0, 4), m.b[0].x[4]),
                ((0, 5), m.b[0].x[5]),
                ((1, 2), m.b[1].x[2]),
                ((1, 3), m.b[1].x[3]),
            ],
        )

    def test_nested_assignment(self):
        m = self.m

        rd = _ReferenceDict(m.b[:, :].x[:, :])
        self.assertEqual(sum(x.value for x in rd.values()), 0)
        rd[1, 5, 7, 10] = 10
        self.assertEqual(m.b[1, 5].x[7, 10].value, 10)
        self.assertEqual(sum(x.value for x in rd.values()), 10)

        rd = _ReferenceDict(m.b[:, 4].x[8, :])
        self.assertEqual(sum(x.value for x in rd.values()), 0)
        rd[1, 10] = 20
        self.assertEqual(m.b[1, 4].x[8, 10].value, 20)
        self.assertEqual(sum(x.value for x in rd.values()), 20)

    def test_attribute_assignment(self):
        m = self.m

        rd = _ReferenceDict(m.b[:, :].x[:, :].value)
        self.assertEqual(sum(x for x in rd.values()), 0)
        rd[1, 5, 7, 10] = 10
        self.assertEqual(m.b[1, 5].x[7, 10].value, 10)
        self.assertEqual(sum(x for x in rd.values()), 10)

        rd = _ReferenceDict(m.b[:, 4].x[8, :].value)
        self.assertEqual(sum(x for x in rd.values()), 0)
        rd[1, 10] = 20
        self.assertEqual(m.b[1, 4].x[8, 10].value, 20)
        self.assertEqual(sum(x for x in rd.values()), 20)

        m.x = Var([1, 2], initialize=0)
        rd = _ReferenceDict(m.x[:])
        self.assertEqual(sum(x.value for x in rd.values()), 0)
        rd[2] = 10
        self.assertEqual(m.x[1].value, 0)
        self.assertEqual(m.x[2].value, 10)
        self.assertEqual(sum(x.value for x in rd.values()), 10)

    def test_single_attribute_assignment(self):
        m = self.m

        rd = _ReferenceDict(m.b[1, 5].x[:, :])
        self.assertEqual(sum(x.value for x in rd.values()), 0)
        rd[7, 10].value = 10
        self.assertEqual(m.b[1, 5].x[7, 10].value, 10)
        self.assertEqual(sum(x.value for x in rd.values()), 10)

        rd = _ReferenceDict(m.b[1, 4].x[8, :])
        self.assertEqual(sum(x.value for x in rd.values()), 0)
        rd[10].value = 20
        self.assertEqual(m.b[1, 4].x[8, 10].value, 20)
        self.assertEqual(sum(x.value for x in rd.values()), 20)

    def test_nested_attribute_assignment(self):
        m = self.m

        rd = _ReferenceDict(m.b[:, :].x[:, :])
        self.assertEqual(sum(x.value for x in rd.values()), 0)
        rd[1, 5, 7, 10].value = 10
        self.assertEqual(m.b[1, 5].x[7, 10].value, 10)
        self.assertEqual(sum(x.value for x in rd.values()), 10)

        rd = _ReferenceDict(m.b[:, 4].x[8, :])
        self.assertEqual(sum(x.value for x in rd.values()), 0)
        rd[1, 10].value = 20
        self.assertEqual(m.b[1, 4].x[8, 10].value, 20)
        self.assertEqual(sum(x.value for x in rd.values()), 20)

    def test_single_deletion(self):
        m = self.m

        rd = _ReferenceDict(m.b[1, 5].x[:, :])
        self.assertEqual(len(list(x.value for x in rd.values())), 2 * 2)
        self.assertTrue((7, 10) in rd)
        del rd[7, 10]
        self.assertFalse((7, 10) in rd)
        self.assertEqual(len(list(x.value for x in rd.values())), 3)

        rd = _ReferenceDict(m.b[1, 4].x[8, :])
        self.assertEqual(len(list(x.value for x in rd.values())), 2)
        self.assertTrue((10) in rd)
        del rd[10]
        self.assertFalse(10 in rd)
        self.assertEqual(len(list(x.value for x in rd.values())), 2 - 1)

        with self.assertRaisesRegex(
            KeyError, r"\(8, 10\) is not valid for indexed component 'b\[1,4\].x'"
        ):
            del rd[10]

        rd = _ReferenceDict(m.b[1, :].x[8, 0])
        with self.assertRaisesRegex(
            KeyError, r"'\(8, 0\)' is not valid for indexed component 'b\[1,4\].x'"
        ):
            del rd[4]

    def test_nested_deletion(self):
        m = self.m

        rd = _ReferenceDict(m.b[:, :].x[:, :])
        self.assertEqual(len(list(x.value for x in rd.values())), 2 * 2 * 2 * 2)
        self.assertTrue((1, 5, 7, 10) in rd)
        del rd[1, 5, 7, 10]
        self.assertFalse((1, 5, 7, 10) in rd)
        self.assertEqual(len(list(x.value for x in rd.values())), 2 * 2 * 2 * 2 - 1)

        rd = _ReferenceDict(m.b[:, 4].x[8, :])
        self.assertEqual(len(list(x.value for x in rd.values())), 2 * 2)
        self.assertTrue((1, 10) in rd)
        del rd[1, 10]
        self.assertFalse((1, 10) in rd)
        self.assertEqual(len(list(x.value for x in rd.values())), 2 * 2 - 1)

    def test_attribute_deletion(self):
        m = self.m

        rd = _ReferenceDict(m.b[:, :].z)
        rd._slice.attribute_errors_generate_exceptions = False
        self.assertEqual(len(list(x.value for x in rd.values())), 2 * 2)
        self.assertTrue((1, 5) in rd)
        self.assertTrue(hasattr(m.b[1, 5], 'z'))
        self.assertTrue(hasattr(m.b[2, 5], 'z'))
        del rd[1, 5]
        self.assertFalse((1, 5) in rd)
        self.assertFalse(hasattr(m.b[1, 5], 'z'))
        self.assertTrue(hasattr(m.b[2, 5], 'z'))
        self.assertEqual(len(list(x.value for x in rd.values())), 3)

        rd = _ReferenceDict(m.b[2, :].z)
        rd._slice.attribute_errors_generate_exceptions = False
        self.assertEqual(len(list(x.value for x in rd.values())), 2)
        self.assertTrue(5 in rd)
        self.assertTrue(hasattr(m.b[2, 4], 'z'))
        self.assertTrue(hasattr(m.b[2, 5], 'z'))
        del rd[5]
        self.assertFalse(5 in rd)
        self.assertTrue(hasattr(m.b[2, 4], 'z'))
        self.assertFalse(hasattr(m.b[2, 5], 'z'))
        self.assertEqual(len(list(x.value for x in rd.values())), 2 - 1)

    def test_deprecations(self):
        m = self.m
        rd = _ReferenceDict(m.b[:, :].z)

        items = rd.items()
        with LoggingIntercept() as LOG:
            iteritems = rd.iteritems()
        self.assertIs(type(items), type(iteritems))
        self.assertEqual(list(items), list(iteritems))
        self.assertIn(
            "DEPRECATED: The iteritems method is deprecated. Use dict.items",
            LOG.getvalue(),
        )

        values = rd.values()
        with LoggingIntercept() as LOG:
            itervalues = rd.itervalues()
        self.assertIs(type(values), type(itervalues))
        self.assertEqual(list(values), list(itervalues))
        self.assertIn(
            "DEPRECATED: The itervalues method is deprecated. Use dict.values",
            LOG.getvalue(),
        )


class TestReferenceSet(unittest.TestCase):
    def test_str(self):
        m = ConcreteModel()

        @m.Block([1, 2], [4, 5])
        def b(b, i, j):
            b.x = Var([7, 8], [10, 11], initialize=0)
            b.y = Var([7, 8], initialize=0)
            b.z = Var()

        rs = _ReferenceSet(m.b[:, 5].z)
        self.assertEqual(str(rs), 'ReferenceSet(b[:, 5].z)')

    def test_lookup_and_iter_dense_data(self):
        m = ConcreteModel()

        @m.Block([1, 2], [4, 5])
        def b(b, i, j):
            b.x = Var([7, 8], [10, 11], initialize=0)
            b.y = Var([7, 8], initialize=0)
            b.z = Var()

        rs = _ReferenceSet(m.b[:, 5].z)
        self.assertNotIn((0,), rs)
        self.assertIn(1, rs)
        self.assertIn((1,), rs)
        self.assertEqual(len(rs), 2)
        self.assertEqual(list(rs), [1, 2])

        rs = _ReferenceSet(m.b[:, 5].bad)
        self.assertNotIn((0,), rs)
        self.assertNotIn((1,), rs)
        self.assertEqual(len(rs), 0)
        self.assertEqual(list(rs), [])

        @m.Block([1, 2, 3])
        def d(b, i):
            if i % 2:
                b.x = Var(range(i))

        rs = _ReferenceSet(m.d[:].x[:])
        self.assertIn((1, 0), rs)
        self.assertIn((3, 0), rs)
        self.assertNotIn((2, 0), rs)
        self.assertEqual(len(rs), 4)
        self.assertEqual(list(rs), [(1, 0), (3, 0), (3, 1), (3, 2)])

        rs = _ReferenceSet(m.d[...].x[...])
        self.assertIn((1, 0), rs)
        self.assertIn((3, 0), rs)
        self.assertNotIn((2, 0), rs)
        self.assertEqual(len(rs), 4)
        self.assertEqual(list(rs), [(1, 0), (3, 0), (3, 1), (3, 2)])

        # Test the SliceEllipsisError case (lookup into a jagged set
        # with an ellipsis)

        m.e_index = Set(initialize=[2, (2, 3)], dimen=None)

        @m.Block(m.e_index)
        def e(b, *args):
            b.x_index = Set(initialize=[1, (3, 4)], dimen=None)
            b.x = Var(b.x_index)

        rs = _ReferenceSet(m.e[...].x[...])
        self.assertIn((2, 1), rs)
        self.assertIn((2, 3, 1), rs)
        self.assertIn((2, 3, 4), rs)
        self.assertNotIn((2, 3, 5), rs)
        self.assertEqual(len(rs), 4)
        self.assertEqual(list(rs), [(2, 1), (2, 3, 4), (2, 3, 1), (2, 3, 3, 4)])

        # Make sure scalars and tuples work for jagged sets
        rs = _ReferenceSet(m.e[...])
        self.assertIn(2, rs)
        self.assertIn((2,), rs)
        self.assertNotIn(0, rs)
        self.assertNotIn((0,), rs)

    def test_lookup_and_iter_sparse_data(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.x = Var(m.I, m.I, dense=False)

        rd = _ReferenceDict(m.x[...])
        rs = _ReferenceSet(m.x[...])
        self.assertEqual(len(rd), 0)
        # Note: we will periodically re-check the dict to ensure
        # iteration doesn't accidentally declare data
        self.assertEqual(len(rd), 0)

        self.assertEqual(len(rs), 9)
        self.assertEqual(len(rd), 0)

        self.assertIn((1, 1), rs)
        self.assertEqual(len(rd), 0)
        self.assertEqual(len(rs), 9)

    def test_otdered_sorted_iter(self):
        # Test ordered reference
        m = ConcreteModel()

        @m.Block([2, 1], [4, 5])
        def b(b, i, j):
            b.x = Var([8, 7], initialize=0)

        rs = _ReferenceSet(m.b[...].x[:])
        self.assertEqual(
            list(rs),
            [
                (2, 4, 8),
                (2, 4, 7),
                (2, 5, 8),
                (2, 5, 7),
                (1, 4, 8),
                (1, 4, 7),
                (1, 5, 8),
                (1, 5, 7),
            ],
        )

        rs = _ReferenceSet(m.b[...].x[:])
        self.assertEqual(
            list(rs.ordered_iter()),
            [
                (2, 4, 8),
                (2, 4, 7),
                (2, 5, 8),
                (2, 5, 7),
                (1, 4, 8),
                (1, 4, 7),
                (1, 5, 8),
                (1, 5, 7),
            ],
        )

        rs = _ReferenceSet(m.b[...].x[:])
        self.assertEqual(
            list(rs.sorted_iter()),
            [
                (1, 4, 7),
                (1, 4, 8),
                (1, 5, 7),
                (1, 5, 8),
                (2, 4, 7),
                (2, 4, 8),
                (2, 5, 7),
                (2, 5, 8),
            ],
        )

        # Test unordered reference
        m = ConcreteModel()
        m.I = FiniteSetOf([2, 1])
        m.J = FiniteSetOf([4, 5])
        m.K = FiniteSetOf([8, 7])

        @m.Block(m.I, m.J)
        def b(b, i, j):
            b.x = Var(m.K, initialize=0)

        rs = _ReferenceSet(m.b[...].x[:])
        self.assertEqual(
            list(rs),
            [
                (2, 4, 8),
                (2, 4, 7),
                (2, 5, 8),
                (2, 5, 7),
                (1, 4, 8),
                (1, 4, 7),
                (1, 5, 8),
                (1, 5, 7),
            ],
        )

        rs = _ReferenceSet(m.b[...].x[:])
        self.assertEqual(
            list(rs.ordered_iter()),
            [
                (1, 4, 7),
                (1, 4, 8),
                (1, 5, 7),
                (1, 5, 8),
                (2, 4, 7),
                (2, 4, 8),
                (2, 5, 7),
                (2, 5, 8),
            ],
        )

        rs = _ReferenceSet(m.b[...].x[:])
        self.assertEqual(
            list(rs.sorted_iter()),
            [
                (1, 4, 7),
                (1, 4, 8),
                (1, 5, 7),
                (1, 5, 8),
                (2, 4, 7),
                (2, 4, 8),
                (2, 5, 7),
                (2, 5, 8),
            ],
        )


class TestReference(unittest.TestCase):
    def test_constructor_error(self):
        m = ConcreteModel()
        m.x = Var([1, 2])

        class Foo(object):
            pass

        self.assertRaisesRegex(
            TypeError,
            "First argument to Reference constructors must be a "
            r"component, component slice, Sequence, or Mapping \(received Foo",
            Reference,
            Foo(),
        )
        self.assertRaisesRegex(
            TypeError,
            "First argument to Reference constructors must be a "
            r"component, component slice, Sequence, or Mapping \(received int",
            Reference,
            5,
        )
        self.assertRaisesRegex(
            TypeError,
            "First argument to Reference constructors must be a "
            r"component, component slice, Sequence, or Mapping \(received None",
            Reference,
            None,
        )

    def test_component_reference(self):
        m = ConcreteModel()
        m.x = Var()
        m.r = Reference(m.x)

        self.assertIs(m.r.ctype, Var)
        self.assertIsNot(m.r.index_set(), m.x.index_set())
        self.assertIs(m.x.index_set(), UnindexedComponent_set)
        self.assertIs(m.r.index_set(), UnindexedComponent_ReferenceSet)
        self.assertEqual(len(m.r), 1)
        self.assertTrue(m.r.is_indexed())
        self.assertIn(None, m.r)
        self.assertNotIn(1, m.r)
        self.assertIs(m.r[None], m.x)
        with self.assertRaises(KeyError):
            m.r[1]

        m.s = Reference(m.x[:])

        self.assertIs(m.s.ctype, Var)
        self.assertIsNot(m.s.index_set(), m.x.index_set())
        self.assertIs(m.x.index_set(), UnindexedComponent_set)
        self.assertIs(type(m.s.index_set()), OrderedSetOf)
        self.assertEqual(len(m.s), 1)
        self.assertTrue(m.s.is_indexed())
        self.assertIn(None, m.s)
        self.assertNotIn(1, m.s)
        self.assertIs(m.s[None], m.x)
        with self.assertRaises(KeyError):
            m.s[1]

        m.y = Var([1, 2])
        m.t = Reference(m.y)

        self.assertIs(m.t.ctype, Var)
        self.assertIs(m.t.index_set(), m.y.index_set())
        self.assertEqual(len(m.t), 2)
        self.assertTrue(m.t.is_indexed())
        self.assertNotIn(None, m.t)
        self.assertIn(1, m.t)
        self.assertIn(2, m.t)
        self.assertIs(m.t[1], m.y[1])
        with self.assertRaises(KeyError):
            m.t[3]

    def test_component_data_reference(self):
        m = ConcreteModel()
        m.y = Var([1, 2])
        m.r = Reference(m.y[2])

        self.assertIs(m.r.ctype, Var)
        self.assertIsNot(m.r.index_set(), m.y.index_set())
        self.assertIs(m.y.index_set(), m.y_index)
        self.assertIs(m.r.index_set(), UnindexedComponent_ReferenceSet)
        self.assertEqual(len(m.r), 1)
        self.assertTrue(m.r.is_reference())
        self.assertTrue(m.r.is_indexed())
        self.assertIn(None, m.r)
        self.assertNotIn(1, m.r)
        self.assertIs(m.r[None], m.y[2])
        with self.assertRaises(KeyError):
            m.r[2]

    def test_component_data_reference_clone(self):
        m = ConcreteModel()
        m.b = Block()
        m.b.x = Var([1, 2])
        m.c = Block()
        m.c.r1 = Reference(m.b.x[2])
        m.c.r2 = Reference(m.b.x)

        self.assertIs(m.c.r1[None], m.b.x[2])
        m.d = m.c.clone()
        self.assertIs(m.d.r1[None], m.b.x[2])
        self.assertIs(m.d.r2[1], m.b.x[1])
        self.assertIs(m.d.r2[2], m.b.x[2])

        i = m.clone()
        self.assertIs(i.c.r1[None], i.b.x[2])
        self.assertIs(i.c.r2[1], i.b.x[1])
        self.assertIs(i.c.r2[2], i.b.x[2])
        self.assertIsNot(i.c.r1[None], m.b.x[2])
        self.assertIsNot(i.c.r2[1], m.b.x[1])
        self.assertIsNot(i.c.r2[2], m.b.x[2])
        self.assertIs(i.d.r1[None], i.b.x[2])
        self.assertIs(i.d.r2[1], i.b.x[1])
        self.assertIs(i.d.r2[2], i.b.x[2])

    def test_reference_var_pprint(self):
        m = ConcreteModel()
        m.x = Var([1, 2], initialize={1: 4, 2: 8})
        m.r = Reference(m.x)
        buf = StringIO()
        m.r.pprint(ostream=buf)
        self.assertEqual(
            buf.getvalue(),
            """r : Size=2, Index=x_index, ReferenceTo=x
    Key : Lower : Value : Upper : Fixed : Stale : Domain
      1 :  None :     4 :  None : False : False :  Reals
      2 :  None :     8 :  None : False : False :  Reals
""",
        )
        m.s = Reference(m.x[:, ...])
        buf = StringIO()
        m.s.pprint(ostream=buf)
        self.assertEqual(
            buf.getvalue(),
            """s : Size=2, Index=x_index, ReferenceTo=x[:, ...]
    Key : Lower : Value : Upper : Fixed : Stale : Domain
      1 :  None :     4 :  None : False : False :  Reals
      2 :  None :     8 :  None : False : False :  Reals
""",
        )

    def test_reference_indexedcomponent_pprint(self):
        m = ConcreteModel()
        m.x = Var([1, 2], initialize={1: 4, 2: 8})
        m.r = Reference(m.x, ctype=IndexedComponent)
        buf = StringIO()
        m.r.pprint(ostream=buf)
        self.assertEqual(
            buf.getvalue(),
            """r : Size=2, Index=x_index, ReferenceTo=x
    Key : Object
      1 : <class 'pyomo.core.base.var._GeneralVarData'>
      2 : <class 'pyomo.core.base.var._GeneralVarData'>
""",
        )
        m.s = Reference(m.x[:, ...], ctype=IndexedComponent)
        buf = StringIO()
        m.s.pprint(ostream=buf)
        self.assertEqual(
            buf.getvalue(),
            """s : Size=2, Index=x_index, ReferenceTo=x[:, ...]
    Key : Object
      1 : <class 'pyomo.core.base.var._GeneralVarData'>
      2 : <class 'pyomo.core.base.var._GeneralVarData'>
""",
        )

    def test_single_reference(self):
        m = ConcreteModel()
        m.b = Block([1, 2])
        m.b[1].x = Var(bounds=(1, None))
        m.b[2].x = Var(bounds=(2, None))
        m.r = Reference(m.b[:].x)

        self.assertIs(m.r.ctype, Var)
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
        m.I = Set(initialize=[1, 2])
        m.J = Set(initialize=[3, 4])

        @m.Block(m.I)
        def b(b, i):
            b.x = Var(b.model().J, bounds=(i, None))

        m.r = Reference(m.b[:].x[:])

        self.assertIs(m.r.ctype, Var)
        self.assertIsInstance(m.r.index_set(), SetProduct)
        self.assertIs(m.r.index_set().set_tuple[0], m.I)
        self.assertIs(m.r.index_set().set_tuple[1], m.J)
        self.assertEqual(len(m.r), 2 * 2)
        self.assertEqual(m.r[1, 3].lb, 1)
        self.assertEqual(m.r[2, 4].lb, 2)
        self.assertIn((1, 3), m.r)
        self.assertIn((2, 4), m.r)
        self.assertNotIn(0, m.r)
        self.assertNotIn((1, 0), m.r)
        self.assertNotIn((1, 3, 0), m.r)
        with self.assertRaises(KeyError):
            m.r[0]

    def test_nested_reference_multidim_set(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2])
        m.J = Set(initialize=[(3, 3), (4, 4)])

        @m.Block(m.I)
        def b(b, i):
            b.x = Var(b.model().J, bounds=(i, None))

        m.r = Reference(m.b[:].x[:, :])

        self.assertIs(m.r.ctype, Var)
        self.assertIsInstance(m.r.index_set(), SetProduct)
        self.assertIs(m.r.index_set().set_tuple[0], m.I)
        self.assertIs(m.r.index_set().set_tuple[1], m.J)
        self.assertEqual(len(m.r), 2 * 2)
        self.assertEqual(m.r[1, 3, 3].lb, 1)
        self.assertEqual(m.r[2, 4, 4].lb, 2)
        self.assertIn((1, 3, 3), m.r)
        self.assertIn((2, 4, 4), m.r)
        self.assertNotIn(0, m.r)
        self.assertNotIn((1, 0), m.r)
        self.assertNotIn((1, 3, 0), m.r)
        self.assertNotIn((1, 3, 3, 0), m.r)
        with self.assertRaises(KeyError):
            m.r[0]

    def test_nested_reference_partial_multidim_set(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2])
        m.J = Set(initialize=[(3, 3), (4, 4)])

        @m.Block(m.I)
        def b(b, i):
            b.x = Var(b.model().J, bounds=(i, None))

        m.r = Reference(m.b[:].x[3, :])

        self.assertIs(m.r.ctype, Var)
        self.assertIs(type(m.r.index_set()), FiniteSetOf)
        self.assertEqual(len(m.r), 2 * 1)
        self.assertEqual(m.r[1, 3].lb, 1)
        self.assertEqual(m.r[2, 3].lb, 2)
        self.assertIn((1, 3), m.r)
        self.assertNotIn((2, 4), m.r)
        self.assertNotIn(0, m.r)
        self.assertNotIn((1, 0), m.r)
        self.assertNotIn((1, 3, 0), m.r)
        with self.assertRaises(KeyError):
            m.r[0]

    def test_nested_reference_nonuniform_indexes(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2])
        m.J = Set(initialize=[3, 4])

        @m.Block(m.I)
        def b(b, i):
            b.x = Var([3, 4], bounds=(i, None))

        m.r = Reference(m.b[:].x[:])

        self.assertIs(m.r.ctype, Var)
        self.assertIs(type(m.r.index_set()), FiniteSetOf)
        self.assertEqual(len(m.r), 2 * 2)
        self.assertEqual(m.r[1, 3].lb, 1)
        self.assertEqual(m.r[2, 4].lb, 2)
        self.assertIn((1, 3), m.r)
        self.assertIn((2, 4), m.r)
        self.assertNotIn(0, m.r)
        self.assertNotIn((1, 0), m.r)
        self.assertNotIn((1, 3, 0), m.r)
        with self.assertRaises(KeyError):
            m.r[0]

    def test_nested_reference_nondimen_set(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2])
        m.J = Set(initialize=[3, 4], dimen=None)

        @m.Block(m.I)
        def b(b, i):
            b.x = Var(b.model().J, bounds=(i, None))

        m.r = Reference(m.b[:].x[:])

        self.assertIs(m.r.ctype, Var)
        self.assertIs(type(m.r.index_set()), FiniteSetOf)
        self.assertEqual(len(m.r), 2 * 2)
        self.assertEqual(m.r[1, 3].lb, 1)
        self.assertEqual(m.r[2, 4].lb, 2)
        self.assertIn((1, 3), m.r)
        self.assertIn((2, 4), m.r)
        self.assertNotIn(0, m.r)
        self.assertNotIn((1, 0), m.r)
        self.assertNotIn((1, 3, 0), m.r)
        with self.assertRaises(KeyError):
            m.r[0]

    def test_nested_reference_nonuniform_index_size(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2])
        m.J = Set(initialize=[3, 4])
        m.b = Block(m.I)
        m.b[1].x = Var([(3, 3), (3, 4), (4, 3), (4, 4)], bounds=(1, None))
        m.b[2].x = Var(m.J, m.J, bounds=(2, None))

        m.r = Reference(m.b[:].x[:, :])

        self.assertIs(m.r.ctype, Var)
        self.assertIs(type(m.r.index_set()), FiniteSetOf)
        self.assertEqual(len(m.r), 2 * 2 * 2)
        self.assertEqual(m.r[1, 3, 3].lb, 1)
        self.assertEqual(m.r[2, 4, 3].lb, 2)
        self.assertIn((1, 3, 3), m.r)
        self.assertIn((2, 4, 4), m.r)
        self.assertNotIn(0, m.r)
        self.assertNotIn((1, 0), m.r)
        self.assertNotIn((1, 3, 0), m.r)
        self.assertNotIn((1, 3, 3, 0), m.r)
        with self.assertRaises(KeyError):
            m.r[0]

    def test_nested_scalars(self):
        m = ConcreteModel()
        m.b = Block()
        m.b.x = Var()
        m.r = Reference(m.b[:].x[:])
        self.assertEqual(len(m.r), 1)
        self.assertEqual(m.r.index_set().dimen, 2)
        base_sets = list(m.r.index_set().subsets())
        self.assertEqual(len(base_sets), 2)
        self.assertIs(type(base_sets[0]), OrderedSetOf)
        self.assertIs(type(base_sets[1]), OrderedSetOf)

    def test_ctype_detection(self):
        m = ConcreteModel()
        m.js = Set(initialize=[1, (2, 3)], dimen=None)
        m.b = Block([1, 2])
        m.b[1].x = Var(m.js)
        m.b[1].y = Var()
        m.b[1].z = Var([1, 2])
        m.b[2].x = Param(initialize=0)
        m.b[2].y = Var()
        m.b[2].z = Var([1, 2])

        m.x = Reference(m.b[:].x[...])
        self.assertIs(type(m.x), IndexedComponent)

        m.y = Reference(m.b[:].y[...])
        self.assertIs(type(m.y), IndexedVar)
        self.assertIs(m.y.ctype, Var)
        m.y1 = Reference(m.b[:].y[...], ctype=None)
        self.assertIs(type(m.y1), IndexedComponent)
        self.assertIs(m.y1.ctype, IndexedComponent)

        m.z = Reference(m.b[:].z)
        self.assertIs(type(m.z), IndexedComponent)
        self.assertIs(m.z.ctype, IndexedComponent)

    def test_reference_to_sparse(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.x = Var(m.I, m.I, dense=False)
        m.xx = Reference(m.x[...], ctype=Var)

        self.assertEqual(len(m.x), 0)
        self.assertNotIn((1, 1), m.x)
        self.assertNotIn((1, 1), m.xx)
        self.assertIn((1, 1), m.x.index_set())
        self.assertIn((1, 1), m.xx.index_set())
        self.assertEqual(len(m.x), 0)

        m.xx[1, 2]
        self.assertEqual(len(m.x), 1)
        self.assertIs(m.xx[1, 2], m.x[1, 2])
        self.assertEqual(len(m.x), 1)

        m.xx[1, 3] = 5
        self.assertEqual(len(m.x), 2)
        self.assertIs(m.xx[1, 3], m.x[1, 3])
        self.assertEqual(len(m.x), 2)
        self.assertEqual(value(m.x[1, 3]), 5)

        m.xx.add((1, 1))
        self.assertEqual(len(m.x), 3)
        self.assertIs(m.xx[1, 1], m.x[1, 1])
        self.assertEqual(len(m.x), 3)

    def test_nested_reference_to_sparse(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1])

        @m.Block(m.I)
        def b(b, i):
            b.x = Var(b.model().I, dense=False)

        m.xx = Reference(m.b[:].x[:], ctype=Var)
        m.I.add(2)
        m.I.add(3)

        self.assertEqual(len(m.b), 1)
        self.assertEqual(len(m.b[1].x), 0)
        self.assertIn(1, m.b)
        self.assertNotIn((1, 1), m.xx)
        self.assertIn(1, m.b[1].x.index_set())
        self.assertIn((1, 1), m.xx.index_set())
        self.assertEqual(len(m.b), 1)
        self.assertEqual(len(m.b[1].x), 0)

        m.xx[1, 2]
        self.assertEqual(len(m.b), 1)
        self.assertEqual(len(m.b[1].x), 1)
        self.assertIs(m.xx[1, 2], m.b[1].x[2])
        self.assertEqual(len(m.b), 1)
        self.assertEqual(len(m.b[1].x), 1)

        m.xx[1, 3] = 5
        self.assertEqual(len(m.b), 1)
        self.assertEqual(len(m.b[1].x), 2)
        self.assertIs(m.xx[1, 3], m.b[1].x[3])
        self.assertEqual(value(m.b[1].x[3]), 5)
        self.assertEqual(len(m.b), 1)
        self.assertEqual(len(m.b[1].x), 2)

        m.xx.add((1, 1))
        self.assertEqual(len(m.b), 1)
        self.assertEqual(len(m.b[1].x), 3)
        self.assertIs(m.xx[1, 1], m.b[1].x[1])
        self.assertEqual(len(m.b), 1)
        self.assertEqual(len(m.b[1].x), 3)

        # While (2,2) appears to be a valid member of the slice, because
        # 2 was not in the Set when the Block rule fired, there is no
        # m.b[2] block data.  Accessing m.xx[2,1] will construct the
        # b[2] block data, fire the rule, and then add the new value to
        # the Var x.
        self.assertEqual(len(m.xx), 3)
        m.xx[2, 2] = 10
        self.assertEqual(len(m.b), 2)
        self.assertEqual(len(list(m.b[2].component_objects())), 1)
        self.assertEqual(len(m.xx), 4)
        self.assertIs(m.xx[2, 2], m.b[2].x[2])
        self.assertEqual(value(m.b[2].x[2]), 10)

    def test_insert_var(self):
        m = ConcreteModel()
        m.T = Set(initialize=[1, 5])
        m.x = Var(m.T, initialize=lambda m, i: i)

        @m.Block(m.T)
        def b(b, i):
            b.y = Var(initialize=lambda b: 10 * b.index())

        ref_x = Reference(m.x[:])
        ref_y = Reference(m.b[:].y)

        self.assertEqual(len(m.x), 2)
        self.assertEqual(len(ref_x), 2)
        self.assertEqual(len(m.b), 2)
        self.assertEqual(len(ref_y), 2)
        self.assertEqual(value(ref_x[1]), 1)
        self.assertEqual(value(ref_x[5]), 5)
        self.assertEqual(value(ref_y[1]), 10)
        self.assertEqual(value(ref_y[5]), 50)

        m.T.add(2)
        _x = ref_x[2]
        self.assertEqual(len(m.x), 3)
        self.assertIs(_x, m.x[2])
        self.assertEqual(value(_x), 2)
        self.assertEqual(value(m.x[2]), 2)
        self.assertEqual(value(ref_x[2]), 2)

        _y = ref_y[2]
        self.assertEqual(len(m.b), 3)
        self.assertIs(_y, m.b[2].y)
        self.assertEqual(value(_y), 20)
        self.assertEqual(value(ref_y[2]), 20)
        self.assertEqual(value(m.b[2].y), 20)

    def test_reference_to_dict(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var([1, 2, 3])
        m.r = Reference({1: m.x, 'a': m.y[2], 3: m.y[1]})
        self.assertFalse(m.r.index_set().isordered())
        self.assertEqual(len(m.r), 3)
        self.assertEqual(set(m.r.keys()), {1, 3, 'a'})
        self.assertEqual(
            ComponentSet(m.r.values()), ComponentSet([m.x, m.y[2], m.y[1]])
        )
        # You can delete something from the reference
        del m.r[1]
        self.assertEqual(len(m.r), 2)
        self.assertEqual(set(m.r.keys()), {3, 'a'})
        self.assertEqual(ComponentSet(m.r.values()), ComponentSet([m.y[2], m.y[1]]))
        # But not add it back
        with self.assertRaisesRegex(
            KeyError, "Index '1' is not valid for indexed component 'r'"
        ):
            m.r[1] = m.x

    def test_reference_to_list(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var([1, 2, 3])
        m.r = Reference([m.x, m.y[2], m.y[1]])
        self.assertTrue(m.r.index_set().isordered())
        self.assertEqual(len(m.r), 3)
        self.assertEqual(list(m.r.keys()), [0, 1, 2])
        self.assertEqual(list(m.r.values()), [m.x, m.y[2], m.y[1]])
        # You can delete something from the reference
        del m.r[1]
        self.assertEqual(len(m.r), 2)
        self.assertEqual(list(m.r.keys()), [0, 2])
        self.assertEqual(list(m.r.values()), [m.x, m.y[1]])
        # But not add it back
        with self.assertRaisesRegex(
            KeyError, "Index '1' is not valid for indexed component 'r'"
        ):
            m.r[1] = m.x

    def test_reference_to_set(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 3, 5])
        m.r = Reference(m.I)
        self.assertEqual(len(m.r), 1)
        self.assertEqual(list(m.r.keys()), [None])
        self.assertEqual(list(m.r.values()), [m.I])
        self.assertIs(m.r[None], m.I)

        # Test that a referent Set containing None doesn't break the
        # None index
        m = ConcreteModel()
        m.I = Set(initialize=[1, 3, None, 5])
        m.r = Reference(m.I)
        self.assertEqual(len(m.r), 1)
        self.assertEqual(list(m.r.keys()), [None])
        self.assertEqual(list(m.r.values()), [m.I])
        self.assertIs(m.r[None], m.I)

    def test_is_reference(self):
        m = ConcreteModel()
        m.v0 = Var()
        m.v1 = Var([1, 2, 3])

        m.ref0 = Reference(m.v0)
        m.ref1 = Reference(m.v1)
        m.ref2 = Reference(m.v1[2])

        self.assertFalse(m.v0.is_reference())
        self.assertFalse(m.v1.is_reference())
        self.assertFalse(m.v1[2].is_reference())

        self.assertTrue(m.ref0.is_reference())
        self.assertTrue(m.ref1.is_reference())
        self.assertTrue(m.ref2.is_reference())

        unique_vars = list(v for v in m.component_objects(Var) if not v.is_reference())
        self.assertEqual(len(unique_vars), 2)

    def test_referent(self):
        m = ConcreteModel()
        m.v0 = Var()
        m.v2 = Var([1, 2, 3], ['a', 'b'])

        varlist = [m.v2[1, 'a'], m.v2[1, 'b']]

        vardict = {0: m.v0, 1: m.v2[1, 'a'], 2: m.v2[2, 'a'], 3: m.v2[3, 'a']}

        scalar_ref = Reference(m.v0)
        self.assertIs(scalar_ref.referent, m.v0)

        sliced_ref = Reference(m.v2[:, 'a'])
        referent = sliced_ref.referent
        self.assertIs(type(referent), IndexedComponent_slice)
        self.assertEqual(len(referent._call_stack), 1)
        call, info = referent._call_stack[0]
        self.assertEqual(call, IndexedComponent_slice.slice_info)
        self.assertIs(info[0], m.v2)
        self.assertEqual(info[1], {1: 'a'})  # Fixed
        self.assertEqual(info[2], {0: slice(None)})  # Sliced
        self.assertIs(info[3], None)  # Ellipsis

        list_ref = Reference(varlist)
        self.assertIs(list_ref.referent, varlist)

        dict_ref = Reference(vardict)
        self.assertIs(dict_ref.referent, vardict)

    def test_UnknownSetDimen(self):
        # Replicate the bug reported in #1928
        m = ConcreteModel()
        m.thinga = Set(initialize=['e1', 'e2', 'e3'])
        m.thingb = Set(initialize=[])
        m.v = Var(m.thinga | m.thingb)
        self.assertIs(m.v.dim(), UnknownSetDimen)
        with self.assertRaisesRegex(
            IndexError,
            'Slicing components relies on knowing the underlying set dimensionality',
        ):
            Reference(m.v)

    def test_contains_with_nonflattened(self):
        # test issue #1800
        _old_flatten = normalize_index.flatten
        try:
            normalize_index.flatten = False
            m = ConcreteModel()
            m.d1 = Set(initialize=[1, 2])
            m.d2 = Set(initialize=[('a', 1), ('b', 2)])
            m.v = Var(m.d2, m.d1)
            m.ref = Reference(m.v[:, 1])
            self.assertIn(('a', 1), m.ref)
            self.assertNotIn(('a', 10), m.ref)
        finally:
            normalize_index.flatten = _old_flatten

    def test_pprint_nonfinite_sets(self):
        self.maxDiff = None
        m = ConcreteModel()
        m.v = Var(NonNegativeIntegers, dense=False)
        m.ref = Reference(m.v)
        buf = StringIO()
        m.pprint(ostream=buf)
        self.assertEqual(
            buf.getvalue().strip(),
            """
2 Var Declarations
    ref : Size=0, Index=NonNegativeIntegers, ReferenceTo=v
        Key : Lower : Value : Upper : Fixed : Stale : Domain
    v : Size=0, Index=NonNegativeIntegers
        Key : Lower : Value : Upper : Fixed : Stale : Domain

2 Declarations: v ref
""".strip(),
        )

        m.v[3]
        m.ref[5]
        buf = StringIO()
        m.pprint(ostream=buf)
        self.assertEqual(
            buf.getvalue().strip(),
            """
2 Var Declarations
    ref : Size=2, Index=NonNegativeIntegers, ReferenceTo=v
        Key : Lower : Value : Upper : Fixed : Stale : Domain
          3 :  None :  None :  None : False :  True :  Reals
          5 :  None :  None :  None : False :  True :  Reals
    v : Size=2, Index=NonNegativeIntegers
        Key : Lower : Value : Upper : Fixed : Stale : Domain
          3 :  None :  None :  None : False :  True :  Reals
          5 :  None :  None :  None : False :  True :  Reals

2 Declarations: v ref
""".strip(),
        )

    def test_pprint_nonfinite_sets_ctypeNone(self):
        # test issue #2039
        self.maxDiff = None
        m = ConcreteModel()
        m.v = Var(NonNegativeIntegers, dense=False)
        m.ref = Reference(m.v, ctype=None)
        buf = StringIO()
        m.pprint(ostream=buf)
        self.assertEqual(
            buf.getvalue().strip(),
            """
1 Var Declarations
    v : Size=0, Index=NonNegativeIntegers
        Key : Lower : Value : Upper : Fixed : Stale : Domain

1 IndexedComponent Declarations
    ref : Size=0, Index=NonNegativeIntegers, ReferenceTo=v
        Key : Object

2 Declarations: v ref
""".strip(),
        )

        m.v[3]
        m.ref[5]
        buf = StringIO()
        m.pprint(ostream=buf)
        self.assertEqual(
            buf.getvalue().strip(),
            """
1 Var Declarations
    v : Size=2, Index=NonNegativeIntegers
        Key : Lower : Value : Upper : Fixed : Stale : Domain
          3 :  None :  None :  None : False :  True :  Reals
          5 :  None :  None :  None : False :  True :  Reals

1 IndexedComponent Declarations
    ref : Size=2, Index=NonNegativeIntegers, ReferenceTo=v
        Key : Object
          3 : <class 'pyomo.core.base.var._GeneralVarData'>
          5 : <class 'pyomo.core.base.var._GeneralVarData'>

2 Declarations: v ref
""".strip(),
        )

    def test_pprint_nested(self):
        m = ConcreteModel()

        @m.Block([1, 2])
        def b(b, i):
            b.x = Var([3, 4], bounds=(i, None))

        m.r = Reference(m.b[:].x[:])
        buf = StringIO()
        m.r.pprint(ostream=buf)
        self.assertEqual(
            buf.getvalue().strip(),
            """
r : Size=4, Index=r_index, ReferenceTo=b[:].x[:]
    Key    : Lower : Value : Upper : Fixed : Stale : Domain
    (1, 3) :     1 :  None :  None : False :  True :  Reals
    (1, 4) :     1 :  None :  None : False :  True :  Reals
    (2, 3) :     2 :  None :  None : False :  True :  Reals
    (2, 4) :     2 :  None :  None : False :  True :  Reals
""".strip(),
        )


if __name__ == "__main__":
    unittest.main()
