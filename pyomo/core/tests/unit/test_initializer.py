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

import pyomo.common.unittest as unittest
from pyomo.common.config import ConfigValue, ConfigList, ConfigDict
from pyomo.common.dependencies import pandas as pd, pandas_available

from pyomo.core.base.util import flatten_tuple
from pyomo.core.base.initializer import (
    Initializer, ConstantInitializer, ItemInitializer, ScalarCallInitializer,
    IndexedCallInitializer, CountedCallInitializer, CountedCallGenerator,
    DataFrameInitializer, DefaultInitializer,
)
from pyomo.environ import (
    ConcreteModel, Var,
)

def _init_scalar(m):
    return 1

def _init_indexed(m, *args):
    i = 1
    for arg in args:
        i *= (arg+1)
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
        m = ConcreteModel()
        a = Initializer(5)
        self.assertIs(type(a), ConstantInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        with self.assertRaisesRegex(
                RuntimeError, "Initializer ConstantInitializer does "
                "not contain embedded indices"):
            a.indices()
        self.assertEqual(a(None, 1), 5)

    def test_dict(self):
        m = ConcreteModel()
        a = Initializer({1:5})
        self.assertIs(type(a), ItemInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertTrue(a.contains_indices())
        self.assertEqual(list(a.indices()), [1])
        self.assertEqual(a(None, 1), 5)

    def test_sequence(self):
        m = ConcreteModel()
        a = Initializer([0,5])
        self.assertIs(type(a), ItemInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertTrue(a.contains_indices())
        self.assertEqual(list(a.indices()), [0,1])
        self.assertEqual(a(None, 1), 5)

        a = Initializer([0,5], treat_sequences_as_mappings=False)
        self.assertIs(type(a), ConstantInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, 1), [0,5])

    def test_function(self):
        m = ConcreteModel()
        def a_init(m):
            return 0
        a = Initializer(a_init)
        self.assertIs(type(a), ScalarCallInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, 1), 0)

        m.x = Var([1,2,3])
        def x_init(m, i):
            return i+1
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

        m.y = Var([1,2,3], [4,5,6])
        def y_init(m, i, j):
            return j*(i+1)
        a = Initializer(y_init)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, (1, 4)), 8)

        b = CountedCallInitializer(m.x, a)
        self.assertIs(type(b), CountedCallInitializer)
        self.assertFalse(b.constant())
        self.assertFalse(b.verified)
        self.assertFalse(a.contains_indices())
        self.assertFalse(b._scalar)
        self.assertIs(a._fcn, b._fcn)
        c = b(None, 1)
        self.assertIs(type(c), CountedCallGenerator)
        self.assertEqual(next(c), 2)
        self.assertEqual(next(c), 3)
        self.assertEqual(next(c), 4)

    def test_method(self):
        class Init(object):
            def a_init(self, m):
                return 0

            def x_init(self, m, i):
                return i+1

            def x2_init(self, m):
                return 0

            def y_init(self, m, i, j):
                return j*(i+1)

        init = Init()

        m = ConcreteModel()
        a = Initializer(init.a_init)
        self.assertIs(type(a), ScalarCallInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, 1), 0)

        m.x = Var([1,2,3])
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

        m.y = Var([1,2,3], [4,5,6])
        a = Initializer(init.y_init)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, (1, 4)), 8)

        b = CountedCallInitializer(m.x, a)
        self.assertIs(type(b), CountedCallInitializer)
        self.assertFalse(b.constant())
        self.assertFalse(b.verified)
        self.assertFalse(a.contains_indices())
        self.assertFalse(b._scalar)
        self.assertIs(a._fcn, b._fcn)
        c = b(None, 1)
        self.assertIs(type(c), CountedCallGenerator)
        self.assertEqual(next(c), 2)
        self.assertEqual(next(c), 3)
        self.assertEqual(next(c), 4)

    def test_classmethod(self):
        class Init(object):
            @classmethod
            def a_init(cls, m):
                return 0

            @classmethod
            def x_init(cls, m, i):
                return i+1

            @classmethod
            def x2_init(cls, m):
                return 0

            @classmethod
            def y_init(cls, m, i, j):
                return j*(i+1)

        m = ConcreteModel()
        a = Initializer(Init.a_init)
        self.assertIs(type(a), ScalarCallInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, 1), 0)

        m.x = Var([1,2,3])
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

        m.y = Var([1,2,3], [4,5,6])
        a = Initializer(Init.y_init)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, (1, 4)), 8)

        b = CountedCallInitializer(m.x, a)
        self.assertIs(type(b), CountedCallInitializer)
        self.assertFalse(b.constant())
        self.assertFalse(b.verified)
        self.assertFalse(a.contains_indices())
        self.assertFalse(b._scalar)
        self.assertIs(a._fcn, b._fcn)
        c = b(None, 1)
        self.assertIs(type(c), CountedCallGenerator)
        self.assertEqual(next(c), 2)
        self.assertEqual(next(c), 3)
        self.assertEqual(next(c), 4)

    def test_staticmethod(self):
        class Init(object):
            @staticmethod
            def a_init(m):
                return 0

            @staticmethod
            def x_init(m, i):
                return i+1

            @staticmethod
            def x2_init(m):
                return 0

            @staticmethod
            def y_init(m, i, j):
                return j*(i+1)

        m = ConcreteModel()
        a = Initializer(Init.a_init)
        self.assertIs(type(a), ScalarCallInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, 1), 0)

        m.x = Var([1,2,3])
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

        m.y = Var([1,2,3], [4,5,6])
        a = Initializer(Init.y_init)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertFalse(a.contains_indices())
        self.assertEqual(a(None, (1, 4)), 8)

        b = CountedCallInitializer(m.x, a)
        self.assertIs(type(b), CountedCallInitializer)
        self.assertFalse(b.constant())
        self.assertFalse(b.verified)
        self.assertFalse(a.contains_indices())
        self.assertFalse(b._scalar)
        self.assertIs(a._fcn, b._fcn)
        c = b(None, 1)
        self.assertIs(type(c), CountedCallGenerator)
        self.assertEqual(next(c), 2)
        self.assertEqual(next(c), 3)
        self.assertEqual(next(c), 4)

    def test_generator_fcn(self):
        m = ConcreteModel()
        def a_init(m):
            yield 0
            yield 3
        with self.assertRaisesRegex(
                ValueError, "Generator functions are not allowed"):
            a = Initializer(a_init)

        a = Initializer(a_init, allow_generators=True)
        self.assertIs(type(a), ScalarCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, 1)), [0,3])

        m.x = Var([1,2,3])
        def x_init(m, i):
            yield i
            yield i+1
        a = Initializer(x_init, allow_generators=True)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, 1)), [1,2])

        m.y = Var([1,2,3], [4,5,6])
        def y_init(m, i, j):
            yield j
            yield i+1
        a = Initializer(y_init, allow_generators=True)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, (1, 4))), [4,2])

    def test_generator_method(self):
        class Init(object):
            def a_init(self, m):
                yield 0
                yield 3

            def x_init(self, m, i):
                yield i
                yield i+1

            def y_init(self, m, i, j):
                yield j
                yield i+1
        init = Init()

        m = ConcreteModel()
        with self.assertRaisesRegex(
                ValueError, "Generator functions are not allowed"):
            a = Initializer(init.a_init)

        a = Initializer(init.a_init, allow_generators=True)
        self.assertIs(type(a), ScalarCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, 1)), [0,3])

        m.x = Var([1,2,3])
        a = Initializer(init.x_init, allow_generators=True)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, 1)), [1,2])

        m.y = Var([1,2,3], [4,5,6])
        a = Initializer(init.y_init, allow_generators=True)
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, (1, 4))), [4,2])

    def test_generators(self):
        m = ConcreteModel()
        with self.assertRaisesRegex(
                ValueError, "Generators are not allowed"):
            a = Initializer(iter([0,3]))

        a = Initializer(iter([0,3]), allow_generators=True)
        self.assertIs(type(a), ConstantInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, 1)), [0,3])

        def x_init():
            yield 0
            yield 3
        with self.assertRaisesRegex(
                ValueError, "Generators are not allowed"):
            a = Initializer(x_init())

        a = Initializer(x_init(), allow_generators=True)
        self.assertIs(type(a), ConstantInitializer)
        self.assertTrue(a.constant())
        self.assertFalse(a.verified)
        self.assertEqual(list(a(None, 1)), [0,3])

    @unittest.skipUnless(pandas_available, "Pandas is not installed")
    def test_dataframe(self):
        d = {'col1': [1, 2, 4]}
        df = pd.DataFrame(data=d)
        a = Initializer(df)
        self.assertIs(type(a), DataFrameInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertTrue(a.contains_indices())
        self.assertEqual(list(a.indices()), [0,1,2])
        self.assertEqual(a(None, 0), 1)
        self.assertEqual(a(None, 1), 2)
        self.assertEqual(a(None, 2), 4)

        d = {'col1': [1, 2, 4], 'col2': [10, 20, 40]}
        df = pd.DataFrame(data=d)
        with self.assertRaisesRegex(
                ValueError,
                'DataFrameInitializer for DataFrame with multiple columns'):
            a = Initializer(df)
        a = DataFrameInitializer(df, 'col2')
        self.assertIs(type(a), DataFrameInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertTrue(a.contains_indices())
        self.assertEqual(list(a.indices()), [0,1,2])
        self.assertEqual(a(None, 0), 10)
        self.assertEqual(a(None, 1), 20)
        self.assertEqual(a(None, 2), 40)

        df = pd.DataFrame([10, 20, 30, 40], index=[[0,0,1,1],[0,1,0,1]])
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
    def test_initializer_initializer(self):
        d = {'col1': [1, 2, 4], 'col2': [10, 20, 40]}
        df = pd.DataFrame(data=d)
        a = Initializer(DataFrameInitializer(df, 'col2'))
        self.assertIs(type(a), DataFrameInitializer)
        self.assertFalse(a.constant())
        self.assertFalse(a.verified)
        self.assertTrue(a.contains_indices())
        self.assertEqual(list(a.indices()), [0,1,2])
        self.assertEqual(a(None, 0), 10)
        self.assertEqual(a(None, 1), 20)
        self.assertEqual(a(None, 2), 40)

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

    def test_default_initializer(self):
        a = Initializer({1:5})
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
