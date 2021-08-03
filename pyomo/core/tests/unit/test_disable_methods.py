#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest

from pyomo.common import DeveloperError
from pyomo.core.base.disable_methods import disable_methods

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
        with self.assertRaisesRegex(
                RuntimeError, "Cannot access 'a' on _abstract_simple "
                "'foo' before it has been constructed"):
            x.a()
        with self.assertRaisesRegex(
                RuntimeError, "Cannot custom_msg _abstract_simple "
                "'foo' before it has been constructed"):
            x.b()
        self.assertEqual(x.c(), 'c')
        with self.assertRaisesRegex(
                RuntimeError, "Cannot access property 'd' on _abstract_simple "
                "'foo' before it has been constructed"):
            x.d
        with self.assertRaisesRegex(
                RuntimeError, "Cannot set property 'd' on _abstract_simple "
                "'foo' before it has been constructed"):
            x.d = 1
        with self.assertRaisesRegex(
                RuntimeError, "Cannot custom_pmsg _abstract_simple "
                "'foo' before it has been constructed"):
            x.e
        with self.assertRaisesRegex(
                RuntimeError, "Cannot custom_pmsg _abstract_simple "
                "'foo' before it has been constructed"):
            x.e = 1

        # Verify that the wrapper function enforces the same API as the
        # wrapped function
        with self.assertRaisesRegex(
                TypeError, r"f\(\) takes "):
            x.f(1,2,3)
        with self.assertRaisesRegex(
                RuntimeError, "Cannot access 'f' on _abstract_simple "
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
        with self.assertRaisesRegex(
                DeveloperError, r"Cannot disable method not_there on "
                r"<class '.*\.foo'>"):

            @disable_methods(('a','not_there'))
            class foo(_simple):
                pass
