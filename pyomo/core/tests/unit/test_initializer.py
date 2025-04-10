#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import functools
import pickle
import platform
import sys
import types

import pyomo.common.unittest as unittest
from pyomo.common.config import ConfigValue, ConfigList, ConfigDict
from pyomo.common.dependencies import (
    pandas as pd,
    pandas_available,
    numpy as np,
    numpy_available,
)

from pyomo.core.base.util import flatten_tuple
from pyomo.core.base.initializer import (
    Initializer,
    BoundInitializer,
    ConstantInitializer,
    ItemInitializer,
    ScalarCallInitializer,
    IndexedCallInitializer,
    CountedCallInitializer,
    CountedCallGenerator,
    DataFrameInitializer,
    DefaultInitializer,
    ParameterizedInitializer,
    ParameterizedIndexedCallInitializer,
    ParameterizedScalarCallInitializer,
    function_types,
)
from pyomo.environ import ConcreteModel, Var


is_pypy = platform.python_implementation().lower().startswith("pypy")


def _init_scalar(m):
    return 1


def _init_indexed(m, *args):
    i = 1
    for arg in args:
        i *= arg + 1
    return i


class Test_Initializer(unittest.TestCase):
    def test_flattener(self):
        tup = (1, 0, (0, 1), (2, 3))
        self.assertEqual((1, 0, 0, 1, 2, 3), flatten_tuple(tup))
        li = [0]
        self.assertEqual((0,), flatten_tuple(li))
        ex = [(1, 0), [2, 3]]
        self.assertEqual((1, 0, 2, 3), flatten_tuple(ex))

    def test_constant(self):
        a = Initializer(5)
        self.assertIs(type(a), ConstantInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        with self.assertRaisesRegex(
            RuntimeError,
            "Initializer ConstantInitializer does not contain embedded indices",
        ):
            a.indices()
        self.assertEqual(a(None, 1), 5)

    def test_dict(self):
        a = Initializer({1: 5})
        self.assertIs(type(a), ItemInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertTrue(a.contains_indices())
        self.assertEqual(list(a.indices()), [1])
        self.assertEqual(a(None, 1), 5)

    def test_sequence(self):
        a = Initializer([0, 5])
        self.assertIs(type(a), ItemInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertTrue(a.contains_indices())
        self.assertEqual(list(a.indices()), [0, 1])
        self.assertEqual(a(None, 1), 5)

        a = Initializer([0, 5], treat_sequences_as_mappings=False)
        self.assertIs(type(a), ConstantInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, 1), [0, 5])

    def test_function(self):
        def a_init(m):
            return 0

        a = Initializer(a_init)
        self.assertIs(type(a), ScalarCallInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, 1), 0)

        def x_init(m, i):
            return i + 1

        a = Initializer(x_init)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, 1), 2)

        def x2_init(m):
            return 0

        a = Initializer(x2_init)
        self.assertIs(type(a), ScalarCallInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, 1), 0)

        def y_init(m, i, j):
            return j * (i + 1)

        a = Initializer(y_init)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, (1, 4)), 8)

    def test_counted_call(self):
        def x_init(m, i):
            return i + 1

        def y_init(m, i, j):
            return j * (i + 1)

        def z_init(m, i, j, k):
            return i * 100 + j * 10 + k

        def bogus(m, i, j):
            return None

        m = ConcreteModel()
        m.x = Var([1, 2, 3])
        a = Initializer(x_init)
        b = CountedCallInitializer(m.x, a)
        self.assertIs(type(b), CountedCallInitializer)
        self.assertFalse(b.constant())
        self.assertFalse(b.verified)
        self.assertFalse(b.contains_indices())
        self.assertFalse(b._scalar)
        self.assertIs(a._fcn, b._fcn)
        c = b(None, 1)
        self.assertIs(type(c), int)
        self.assertEqual(c, 2)

        a = Initializer(bogus)
        b = CountedCallInitializer(m.x, a)
        self.assertIs(type(b), CountedCallInitializer)
        self.assertFalse(b.constant())
        self.assertFalse(b.verified)
        self.assertFalse(b.contains_indices())
        self.assertFalse(b._scalar)
        self.assertIs(a._fcn, b._fcn)
        c = b(None, 1)
        self.assertIs(type(c), CountedCallGenerator)
        with self.assertRaisesRegex(ValueError, 'Counted Var rule returned None'):
            next(c)

        a = Initializer(y_init)
        b = CountedCallInitializer(m.x, a)
        self.assertIs(type(b), CountedCallInitializer)
        self.assertFalse(b.constant())
        self.assertFalse(b.verified)
        self.assertFalse(b.contains_indices())
        self.assertFalse(b._scalar)
        self.assertIs(a._fcn, b._fcn)
        c = b(None, 1)
        self.assertIs(type(c), CountedCallGenerator)
        self.assertEqual(next(c), 2)
        self.assertEqual(next(c), 3)
        self.assertEqual(next(c), 4)

        m.y = Var([(1, 2), (3, 5)])
        a = Initializer(y_init)
        b = CountedCallInitializer(m.y, a)
        self.assertIs(type(b), CountedCallInitializer)
        self.assertFalse(b.constant())
        self.assertFalse(b.verified)
        self.assertFalse(b.contains_indices())
        self.assertFalse(b._scalar)
        self.assertIs(a._fcn, b._fcn)
        c = b(None, (3, 5))
        self.assertIs(type(c), int)
        self.assertEqual(c, 20)

        a = Initializer(z_init)
        b = CountedCallInitializer(m.y, a)
        self.assertIs(type(b), CountedCallInitializer)
        self.assertFalse(b.constant())
        self.assertFalse(b.verified)
        self.assertFalse(b.contains_indices())
        self.assertFalse(b._scalar)
        self.assertIs(a._fcn, b._fcn)
        c = b(None, (3, 5))
        self.assertIs(type(c), CountedCallGenerator)
        self.assertEqual(next(c), 135)
        self.assertEqual(next(c), 235)
        self.assertEqual(next(c), 335)

    def test_method(self):
        class Init(object):
            def a_init(self, m):
                return 0

            def x_init(self, m, i):
                return i + 1

            def x2_init(self, m):
                return 0

            def y_init(self, m, i, j):
                return j * (i + 1)

        init = Init()

        a = Initializer(init.a_init)
        self.assertIs(type(a), ScalarCallInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, 1), 0)

        a = Initializer(init.x_init)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, 1), 2)

        a = Initializer(init.x2_init)
        self.assertIs(type(a), ScalarCallInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, 1), 0)

        a = Initializer(init.y_init)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, (1, 4)), 8)

        m = ConcreteModel()
        m.x = Var([1, 2, 3])
        a = Initializer(init.y_init)
        b = CountedCallInitializer(m.x, a)
        self.assertIs(type(b), CountedCallInitializer)
        self.assertFalse(b.constant())
        self.assertFalse(b.verified)
        self.assertFalse(a.contains_indices())
        self.assertFalse(b._scalar)
        self.assertIs(a._fcn, b._fcn)
        c = b(None, 10)
        self.assertIs(type(c), CountedCallGenerator)
        self.assertEqual(next(c), 20)
        self.assertEqual(next(c), 30)
        self.assertEqual(next(c), 40)

    def test_classmethod(self):
        class Init(object):
            @classmethod
            def a_init(cls, m):
                return 0

            @classmethod
            def x_init(cls, m, i):
                return i + 1

            @classmethod
            def x2_init(cls, m):
                return 0

            @classmethod
            def y_init(cls, m, i, j):
                return j * (i + 1)

        a = Initializer(Init.a_init)
        self.assertIs(type(a), ScalarCallInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, 1), 0)

        a = Initializer(Init.x_init)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, 1), 2)

        a = Initializer(Init.x2_init)
        self.assertIs(type(a), ScalarCallInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, 1), 0)

        a = Initializer(Init.y_init)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, (1, 4)), 8)

        m = ConcreteModel()
        m.x = Var([1, 2, 3])
        a = Initializer(Init.y_init)
        b = CountedCallInitializer(m.x, a)
        self.assertIs(type(b), CountedCallInitializer)
        self.assertFalse(b.constant())
        self.assertFalse(b.verified)
        self.assertFalse(a.contains_indices())
        self.assertFalse(b._scalar)
        self.assertIs(a._fcn, b._fcn)
        c = b(None, 10)
        self.assertIs(type(c), CountedCallGenerator)
        self.assertEqual(next(c), 20)
        self.assertEqual(next(c), 30)
        self.assertEqual(next(c), 40)

    def test_staticmethod(self):
        class Init(object):
            @staticmethod
            def a_init(m):
                return 0

            @staticmethod
            def x_init(m, i):
                return i + 1

            @staticmethod
            def x2_init(m):
                return 0

            @staticmethod
            def y_init(m, i, j):
                return j * (i + 1)

        a = Initializer(Init.a_init)
        self.assertIs(type(a), ScalarCallInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, 1), 0)

        a = Initializer(Init.x_init)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, 1), 2)

        a = Initializer(Init.x2_init)
        self.assertIs(type(a), ScalarCallInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, 1), 0)

        a = Initializer(Init.y_init)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, (1, 4)), 8)

        m = ConcreteModel()
        m.x = Var([1, 2, 3])
        a = Initializer(Init.y_init)
        b = CountedCallInitializer(m.x, a)
        self.assertIs(type(b), CountedCallInitializer)
        self.assertFalse(b.constant())
        self.assertFalse(b.verified)
        self.assertFalse(b.contains_indices())
        self.assertFalse(b._scalar)
        self.assertIs(a._fcn, b._fcn)
        c = b(None, 10)
        self.assertIs(type(c), CountedCallGenerator)
        self.assertEqual(next(c), 20)
        self.assertEqual(next(c), 30)
        self.assertEqual(next(c), 40)

    def test_generator_fcn(self):
        def a_init(m):
            yield 0
            yield 3

        with self.assertRaisesRegex(ValueError, "Generator functions are not allowed"):
            a = Initializer(a_init)

        a = Initializer(a_init, allow_generators=True)
        self.assertIs(type(a), ScalarCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, 1)), [0, 3])

        def x_init(m, i):
            yield i
            yield i + 1

        a = Initializer(x_init, allow_generators=True)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, 1)), [1, 2])

        def y_init(m, i, j):
            yield j
            yield i + 1

        a = Initializer(y_init, allow_generators=True)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, (1, 4))), [4, 2])

    def test_generator_method(self):
        class Init(object):
            def a_init(self, m):
                yield 0
                yield 3

            def x_init(self, m, i):
                yield i
                yield i + 1

            def y_init(self, m, i, j):
                yield j
                yield i + 1

        init = Init()

        with self.assertRaisesRegex(ValueError, "Generator functions are not allowed"):
            a = Initializer(init.a_init)

        a = Initializer(init.a_init, allow_generators=True)
        self.assertIs(type(a), ScalarCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, 1)), [0, 3])

        a = Initializer(init.x_init, allow_generators=True)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, 1)), [1, 2])

        a = Initializer(init.y_init, allow_generators=True)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, (1, 4))), [4, 2])

    def test_generators(self):
        with self.assertRaisesRegex(ValueError, "Generators are not allowed"):
            a = Initializer(iter([0, 3]))

        a = Initializer(iter([0, 3]), allow_generators=True)
        self.assertIs(type(a), ConstantInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, 1)), [0, 3])

        def x_init():
            yield 0
            yield 3

        with self.assertRaisesRegex(ValueError, "Generators are not allowed"):
            a = Initializer(x_init())

        a = Initializer(x_init(), allow_generators=True)
        self.assertIs(type(a), ConstantInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, 1)), [0, 3])

    def test_functor(self):
        class InitScalar(object):
            def __init__(self, val):
                self.val = val

            def __call__(self, m):
                return self.val

        a = Initializer(InitScalar(10))
        self.assertIs(type(a), ScalarCallInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(a(None, None), 10)

        class InitIndexed(object):
            def __init__(self, val):
                self.val = val

            def __call__(self, m, i):
                return self.val + i

        a = Initializer(InitIndexed(10))
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(a(None, 5), 15)

    def test_derived_function(self):
        def _scalar(m):
            return 10

        dynf = types.FunctionType(_scalar.__code__, {})

        a = Initializer(dynf)
        self.assertIs(type(a), ScalarCallInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(a(None, None), 10)

        def _indexed(m, i):
            return 10 + i

        dynf = types.FunctionType(_indexed.__code__, {})

        a = Initializer(dynf)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(a(None, 5), 15)

    def test_function(self):
        def _scalar(m):
            return 10

        a = Initializer(_scalar)
        self.assertIs(type(a), ScalarCallInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(a(None, None), 10)

        def _indexed(m, i):
            return 10 + i

        a = Initializer(_indexed)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(a(None, 5), 15)

        try:
            original_fcn_types = set(function_types)
            function_types.clear()
            self.assertEqual(len(function_types), 0)

            a = Initializer(_scalar)
            self.assertIs(type(a), ScalarCallInitializer)
            self.assertTrue(a.constant())
            self.assertFalse(a.verified)
            self.assertEqual(a(None, None), 10)
            self.assertEqual(len(function_types), 1)
        finally:
            function_types.clear()
            function_types.update(original_fcn_types)

        try:
            original_fcn_types = set(function_types)
            function_types.clear()
            self.assertEqual(len(function_types), 0)

            a = Initializer(_indexed)
            self.assertIs(type(a), IndexedCallInitializer)
            self.assertFalse(a.constant())
            self.assertFalse(a.verified)
            self.assertEqual(a(None, 5), 15)
        finally:
            function_types.clear()
            function_types.update(original_fcn_types)

    def test_no_argspec(self):
        a = Initializer(getattr)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, '__class__'), type(None))

        basetwo = functools.partial(int)
        a = Initializer(basetwo)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a('111', 2), 7)

        # Special case: getfullargspec fails for int under CPython and
        # PyPy<7.3.14, so we assume it is an IndexedCallInitializer.
        basetwo = functools.partial(int, '101', base=2)
        a = Initializer(basetwo)
        if is_pypy and sys.pypy_version_info[:3] >= (7, 3, 14):
            # PyPy behavior diverged from CPython in 7.3.14.  Arguably
            # this is "more correct", so we will allow the difference to
            # persist through Pyomo's Initializer handling (and not
            # special case it there)
            self.assertIs(type(a), ScalarCallInitializer)
            self.assertTrue(a.constant())
        else:
            self.assertIs(type(a), IndexedCallInitializer)
            self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        # but this is not callable, as int won't accept the 'model'
        # self.assertEqual(a(None, None), 5)

    def test_partial(self):
        def fcn(k, m, i, j):
            return i * 100 + j * 10 + k

        part = functools.partial(fcn, 2)
        a = Initializer(part)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, (5, 7)), 572)

        def fcn(k, i, j, m):
            return i * 100 + j * 10 + k

        part = functools.partial(fcn, 2, 5, 7)
        a = Initializer(part)
        self.assertIs(type(a), ScalarCallInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, None), 572)

        def fcn(m, k, i, j):
            return i * 100 + j * 10 + k

        part = functools.partial(fcn, i=2, j=5, k=7)
        a = Initializer(part)
        self.assertIs(type(a), ScalarCallInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, None), 257)

    @unittest.skipUnless(pandas_available, "Pandas is not installed")
    def test_dataframe(self):
        d = {'col1': [1, 2, 4]}
        df = pd.DataFrame(data=d)
        a = Initializer(df)
        self.assertIs(type(a), DataFrameInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertTrue(a.contains_indices())
        self.assertEqual(list(a.indices()), [0, 1, 2])
        self.assertEqual(a(None, 0), 1)
        self.assertEqual(a(None, 1), 2)
        self.assertEqual(a(None, 2), 4)

        d = {'col1': [1, 2, 4], 'col2': [10, 20, 40]}
        df = pd.DataFrame(data=d)
        a = Initializer(df)
        self.assertIs(type(a), DataFrameInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertTrue(a.contains_indices())
        self.assertEqual(
            list(a.indices()),
            [
                (0, 'col1'),
                (0, 'col2'),
                (1, 'col1'),
                (1, 'col2'),
                (2, 'col1'),
                (2, 'col2'),
            ],
        )
        self.assertEqual(a(None, (0, 'col1')), 1)
        self.assertEqual(a(None, (1, 'col2')), 20)
        self.assertEqual(a(None, (2, 'col2')), 40)

        a = DataFrameInitializer(df, 'col2')
        self.assertIs(type(a), DataFrameInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertTrue(a.contains_indices())
        self.assertEqual(list(a.indices()), [0, 1, 2])
        self.assertEqual(a(None, 0), 10)
        self.assertEqual(a(None, 1), 20)
        self.assertEqual(a(None, 2), 40)

        df = pd.DataFrame([10, 20, 30, 40], index=[[0, 0, 1, 1], [0, 1, 0, 1]])
        a = Initializer(df)
        self.assertIs(type(a), DataFrameInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertTrue(a.contains_indices())
        self.assertEqual(list(a.indices()), [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertEqual(a(None, (0, 0)), 10)
        self.assertEqual(a(None, (0, 1)), 20)
        self.assertEqual(a(None, (1, 0)), 30)
        self.assertEqual(a(None, (1, 1)), 40)

    @unittest.skipUnless(pandas_available, "Pandas is not installed")
    def test_series(self):
        d = pd.Series({0: 1, 1: 2, 2: 4})
        a = Initializer(d)
        self.assertIs(type(a), ItemInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertTrue(a.contains_indices())
        self.assertEqual(list(a.indices()), [0, 1, 2])
        self.assertEqual(a(None, 0), 1)
        self.assertEqual(a(None, 1), 2)
        self.assertEqual(a(None, 2), 4)

    @unittest.skipUnless(numpy_available, "Numpy is not installed")
    def test_ndarray(self):
        d = np.array([1, 2, 4])
        a = Initializer(d)
        self.assertIs(type(a), ItemInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertTrue(a.contains_indices())
        self.assertEqual(list(a.indices()), [0, 1, 2])
        self.assertEqual(a(None, 0), 1)
        self.assertEqual(a(None, 1), 2)
        self.assertEqual(a(None, 2), 4)

        # TODO: How should we handle ndarray matrices?
        # d = np.array([[1,2],[4,6]])
        # a = Initializer(d)
        # self.assertIs(type(a), ItemInitializer)
        # self.assertFalse(a.constant())
        # self.assertFalse(a.verified)
        # self.assertTrue(a.contains_indices())
        # self.assertEqual(list(a.indices()), [0,1,2])
        # self.assertEqual(a(None, 0), 1)
        # self.assertEqual(a(None, 1), 2)
        # self.assertEqual(a(None, 2), 4)

    def test_str(self):
        a = Initializer("a string")
        self.assertIs(type(a), ConstantInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, None), "a string")

    @unittest.skipUnless(pandas_available, "Pandas is not installed")
    def test_initializer_initializer(self):
        d = {'col1': [1, 2, 4], 'col2': [10, 20, 40]}
        df = pd.DataFrame(data=d)
        a = Initializer(DataFrameInitializer(df, 'col2'))
        self.assertIs(type(a), DataFrameInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertTrue(a.contains_indices())
        self.assertEqual(list(a.indices()), [0, 1, 2])
        self.assertEqual(a(None, 0), 10)
        self.assertEqual(a(None, 1), 20)
        self.assertEqual(a(None, 2), 40)

    def test_pickle(self):
        a = Initializer(5)
        a.verified = True
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a, b)
        self.assertEqual(a.val, b.val)
        self.assertEqual(a.verified, b.verified)

        a = Initializer({1: 5})
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

        a = Initializer(_init_indexed)
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a, b)
        self.assertIs(a._fcn, b._fcn)
        self.assertEqual(a.verified, b.verified)
        self.assertEqual(a(None, 1), 2)
        self.assertEqual(b(None, 2), 3)

    def test_default_initializer(self):
        a = Initializer({1: 5})
        d = DefaultInitializer(a, None, KeyError)
        self.assertFalse(d.constant())
        self.assertTrue(d.contains_indices())
        self.assertEqual(list(d.indices()), [1])
        self.assertEqual(d(None, 1), 5)
        self.assertIsNone(d(None, 2))

        def rule(m, i):
            if i == 0:
                return 10
            elif i == 1:
                raise KeyError("key")
            elif i == 2:
                raise TypeError("type")
            else:
                raise RuntimeError("runtime")

        a = Initializer(rule)
        d = DefaultInitializer(a, 100, (KeyError, RuntimeError))
        self.assertFalse(d.constant())
        self.assertFalse(d.contains_indices())
        self.assertEqual(d(None, 0), 10)
        self.assertEqual(d(None, 1), 100)
        with self.assertRaisesRegex(TypeError, 'type'):
            d(None, 2)
        self.assertEqual(d(None, 3), 100)

    def test_config_integration(self):
        c = ConfigList()
        c.add(1)
        c.add(3)
        c.add(5)
        a = Initializer(c)
        self.assertIs(type(a), ItemInitializer)
        self.assertTrue(a.contains_indices())
        self.assertEqual(list(a.indices()), [0, 1, 2])
        self.assertEqual(a(None, 0), 1)
        self.assertEqual(a(None, 1), 3)
        self.assertEqual(a(None, 2), 5)

        c = ConfigDict()
        c.declare('opt_1', ConfigValue(default=1))
        c.declare('opt_3', ConfigValue(default=3))
        c.declare('opt_5', ConfigValue(default=5))
        a = Initializer(c)
        self.assertIs(type(a), ItemInitializer)
        self.assertTrue(a.contains_indices())
        self.assertEqual(list(a.indices()), ['opt_1', 'opt_3', 'opt_5'])
        self.assertEqual(a(None, 'opt_1'), 1)
        self.assertEqual(a(None, 'opt_3'), 3)
        self.assertEqual(a(None, 'opt_5'), 5)

    def _bound_function1(self, m, i):
        return m, i

    def _bound_function2(self, m, i, j):
        return m, i, j

    def test_additional_args(self):
        def a_init(m):
            yield 0
            yield 3

        with self.assertRaisesRegex(
            ValueError,
            "Generator functions are not allowed when passing additional args",
        ):
            a = Initializer(a_init, additional_args=1)

        a = Initializer(self._bound_function1, additional_args=1)
        self.assertIs(type(a), ParameterizedScalarCallInitializer)
        self.assertEqual(a('m', None, 5), ('m', 5))

        a = Initializer(self._bound_function2, additional_args=1)
        self.assertIs(type(a), ParameterizedIndexedCallInitializer)
        self.assertEqual(a('m', 1, 5), ('m', 5, 1))

        class Functor(object):
            def __init__(self, i):
                self.i = i

            def __call__(self, m, i):
                return m, i * self.i

        a = Initializer(Functor(10), additional_args=1)
        self.assertIs(type(a), ParameterizedScalarCallInitializer)
        self.assertEqual(a('m', None, 5), ('m', 50))

        a_init = {1: lambda m, i: ('m', i), 2: lambda m, i: ('m', 2 * i)}
        a = Initializer(a_init, additional_args=1)
        self.assertIs(type(a), ParameterizedInitializer)
        self.assertFalse(a.constant())
        self.assertTrue(a.contains_indices())
        self.assertEqual(list(a.indices()), [1, 2])
        self.assertEqual(a('m', 1, 5), ('m', 5))
        self.assertEqual(a('m', 2, 5), ('m', 10))

    def test_bound_initializer(self):
        m = ConcreteModel()
        m.x = Var([0, 1, 2])
        m.y = Var()

        b = BoundInitializer(None, m.x)
        self.assertIsNone(b)

        b = BoundInitializer((0, 1), m.x)
        self.assertIs(type(b), BoundInitializer)
        self.assertTrue(b.constant())
        self.assertFalse(b.verified)
        self.assertFalse(b.contains_indices())
        self.assertEqual(b(None, 1), (0, 1))

        b = BoundInitializer([(0, 1)], m.x)
        self.assertIs(type(b), BoundInitializer)
        self.assertFalse(b.constant())
        self.assertFalse(b.verified)
        self.assertTrue(b.contains_indices())
        self.assertTrue(list(b.indices()), [0])
        self.assertEqual(b(None, 0), (0, 1))

        init = {1: (2, 3), 4: (5, 6)}
        b = BoundInitializer(init, m.x)
        self.assertIs(type(b), BoundInitializer)
        self.assertFalse(b.constant())
        self.assertFalse(b.verified)
        self.assertTrue(b.contains_indices())
        self.assertEqual(list(b.indices()), [1, 4])
        self.assertEqual(b(None, 1), (2, 3))
        self.assertEqual(b(None, 4), (5, 6))

        b = BoundInitializer((0, 1), m.y)
        self.assertEqual(b(None, None), (0, 1))

        b = BoundInitializer(5, m.y)
        self.assertEqual(b(None, None), (5, 5))
