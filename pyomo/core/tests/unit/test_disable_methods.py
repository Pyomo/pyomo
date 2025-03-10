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

import inspect
import pyomo.common.unittest as unittest

from pyomo.common import DeveloperError
from pyomo.core.base.disable_methods import disable_methods
from pyomo.common.modeling import NOTSET


class LocalClass(object):
    def __init__(self, name):
        self._name = name

    def __str__(self):
        return self._name


local_instance = LocalClass('local')


class _simple(object):
    def __init__(self, name):
        self.name = name
        self._d = 'd'
        self._e = 'e'

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
        return self._d

    @d.setter
    def d(self, value='d'):
        self._d = value

    @property
    def e(self):
        return self._e

    @e.setter
    def e(self, value='e'):
        self._e = value

    def f(self, arg1, arg2=1, arg3=NOTSET, arg4=local_instance):
        return "f:%s,%s,%s,%s" % (arg1, arg2, arg3, arg4)

    @property
    def g(self):
        return 'g'

    @property
    def h(self):
        return 'h'


@disable_methods(
    (
        'a',
        ('b', 'custom_msg'),
        'd',
        ('e', 'custom_pmsg'),
        'f',
        'g',
        ('h', 'custom_pmsg'),
    )
)
class _abstract_simple(_simple):
    pass


class TestDisableMethods(unittest.TestCase):
    def test_signature(self):
        # check that signatures are properly preserved
        self.assertEqual(
            inspect.signature(_simple.construct),
            inspect.signature(_abstract_simple.construct),
        )
        self.assertEqual(
            inspect.signature(_simple.a), inspect.signature(_abstract_simple.a)
        )
        self.assertEqual(
            inspect.signature(_simple.b), inspect.signature(_abstract_simple.b)
        )
        self.assertEqual(
            inspect.signature(_simple.c), inspect.signature(_abstract_simple.c)
        )
        self.assertEqual(
            inspect.signature(_simple.d.fget),
            inspect.signature(_abstract_simple.d.fget),
        )
        self.assertEqual(
            inspect.signature(_simple.d.fset),
            inspect.signature(_abstract_simple.d.fset),
        )
        self.assertEqual(
            inspect.signature(_simple.e.fget),
            inspect.signature(_abstract_simple.e.fget),
        )
        self.assertEqual(
            inspect.signature(_simple.e.fset),
            inspect.signature(_abstract_simple.e.fset),
        )

        self.assertEqual(
            inspect.signature(_simple.f), inspect.signature(_abstract_simple.f)
        )

        self.assertEqual(
            inspect.signature(_simple.g.fget),
            inspect.signature(_abstract_simple.g.fget),
        )
        self.assertIsNone(_simple.g.fset)
        self.assertIsNone(_abstract_simple.g.fset)
        self.assertEqual(
            inspect.signature(_simple.h.fget),
            inspect.signature(_abstract_simple.h.fget),
        )
        self.assertIsNone(_simple.h.fset)
        self.assertIsNone(_abstract_simple.h.fset)

    def test_disable(self):
        x = _abstract_simple('foo')
        self.assertIs(type(x), _abstract_simple)
        self.assertIsInstance(x, _simple)
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot access 'a' on _abstract_simple "
            "'foo' before it has been constructed",
        ):
            x.a()
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot custom_msg _abstract_simple "
            "'foo' before it has been constructed",
        ):
            x.b()
        self.assertEqual(x.c(), 'c')
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot access property 'd' on _abstract_simple "
            "'foo' before it has been constructed",
        ):
            x.d
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot set property 'd' on _abstract_simple "
            "'foo' before it has been constructed",
        ):
            x.d = 1
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot custom_pmsg _abstract_simple "
            "'foo' before it has been constructed",
        ):
            x.e
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot custom_pmsg _abstract_simple "
            "'foo' before it has been constructed",
        ):
            x.e = 1

        # Verify that the wrapper function enforces the same API as the
        # wrapped function
        with self.assertRaisesRegex(TypeError, r"f\(\) takes "):
            x.f(1, 2, 3, 4, 5)
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot access 'f' on _abstract_simple "
            "'foo' before it has been constructed",
        ):
            x.f(1, 2)

        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot access property 'g' on _abstract_simple "
            "'foo' before it has been constructed",
        ):
            x.g
        with self.assertRaisesRegex(
            AttributeError, "(can't set attribute)|(object has no setter)"
        ):
            x.g = 1
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot custom_pmsg _abstract_simple "
            "'foo' before it has been constructed",
        ):
            x.h
        with self.assertRaisesRegex(
            AttributeError, "(can't set attribute)|(object has no setter)"
        ):
            x.h = 1

        self.assertEqual(x.construct(), 'construct')
        self.assertIs(type(x), _simple)
        self.assertIsInstance(x, _simple)
        self.assertEqual(x.a(), 'a')
        self.assertEqual(x.b(), 'b')
        self.assertEqual(x.c(), 'c')
        self.assertEqual(x.d, 'd')
        x.d = 1
        self.assertEqual(x.d, 1)
        self.assertEqual(x.e, 'e')
        x.e = 2
        self.assertEqual(x.e, 2)
        self.assertEqual(x.f(1, 2), 'f:1,2,NOTSET,local')
        self.assertEqual(x.g, 'g')
        self.assertEqual(x.h, 'h')

    def test_bad_api(self):
        with self.assertRaisesRegex(
            DeveloperError,
            r"Cannot disable method not_there on <class '.*\.foo'>",
            normalize_whitespace=True,
        ):

            @disable_methods(('a', 'not_there'))
            class foo(_simple):
                pass
