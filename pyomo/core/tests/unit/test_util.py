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

from pyomo.common import DeveloperError
from pyomo.core.base.util import (
    Initializer, ConstantInitializer, ItemInitializer, ScalarCallInitializer,
    IndexedCallInitializer, CountedCallInitializer, CountedCallGenerator,
    disable_methods,
)
from pyomo.environ import (
    ConcreteModel, Var,
)

class _simple(object):
    def __init__(self, name):
        self.name = name

    def construct(self, data=None):
        return 'construct'

    def a(self):
        return 'a'

    def b(self):
        return 'b'

    def c(self):
        return 'c'

    @property
    def d(self):
        return 'd'

    @property
    def e(self):
        return 'e'

    def f(self, arg1, arg2=1):
        return "f%s%s" % (arg1, arg2)

@disable_methods(('a',('b', 'custom_msg'),'d',('e', 'custom_pmsg'),'f'))
class _abstract_simple(_simple):
    pass

class TestDisableMethods(unittest.TestCase):
    def test_disable(self):
        x = _abstract_simple('foo')
        self.assertIs(type(x), _abstract_simple)
        self.assertIsInstance(x, _simple)
        with self.assertRaisesRegexp(
                RuntimeError, "Cannot access a on _abstract_simple "
                "'foo' before it has been constructed"):
            x.a()
        with self.assertRaisesRegexp(
                RuntimeError, "Cannot custom_msg _abstract_simple "
                "'foo' before it has been constructed"):
            x.b()
        self.assertEqual(x.c(), 'c')
        with self.assertRaisesRegexp(
                RuntimeError, "Cannot access property d on _abstract_simple "
                "'foo' before it has been constructed"):
            x.d
        with self.assertRaisesRegexp(
                RuntimeError, "Cannot set property d on _abstract_simple "
                "'foo' before it has been constructed"):
            x.d = 1
        with self.assertRaisesRegexp(
                RuntimeError, "Cannot custom_pmsg _abstract_simple "
                "'foo' before it has been constructed"):
            x.e
        with self.assertRaisesRegexp(
                RuntimeError, "Cannot custom_pmsg _abstract_simple "
                "'foo' before it has been constructed"):
            x.e = 1

        # Verify that the wrapper function enforces the same API as the
        # wrapped function
        with self.assertRaisesRegexp(
                TypeError, "f\(\) takes "):
            x.f(1,2,3)
        with self.assertRaisesRegexp(
                RuntimeError, "Cannot access f on _abstract_simple "
                "'foo' before it has been constructed"):
            x.f(1,2)


        self.assertEqual(x.construct(), 'construct')
        self.assertIs(type(x), _simple)
        self.assertIsInstance(x, _simple)
        self.assertEqual(x.a(), 'a')
        self.assertEqual(x.b(), 'b')
        self.assertEqual(x.c(), 'c')
        self.assertEqual(x.d, 'd')
        self.assertEqual(x.e, 'e')
        self.assertEqual(x.f(1,2), 'f12')

    def test_bad_api(self):
        with self.assertRaisesRegexp(
                DeveloperError, "Cannot disable method not_there on "
                "<class '.*\.foo'>"):

            @disable_methods(('a','not_there'))
            class foo(_simple):
                pass


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
        self.assertFalse(a.constant())
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
        self.assertFalse(a.constant())
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
        self.assertFalse(a.constant())
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
        self.assertFalse(a.constant())
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
        self.assertFalse(a.constant())
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
        self.assertFalse(a.constant())
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
        self.assertFalse(a.constant())
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
        self.assertFalse(a.constant())
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
        with self.assertRaisesRegexp(
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
        with self.assertRaisesRegexp(
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
        with self.assertRaisesRegexp(
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
        with self.assertRaisesRegexp(
                ValueError, "Generators are not allowed"):
            a = Initializer(x_init())

        a = Initializer(x_init(), allow_generators=True)
        self.assertIs(type(a), ConstantInitializer)
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
