#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Unit Tests for Python numeric values
#

import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.math
import pyutilib.th as unittest

from pyomo.environ import *

try:
    unicode
except:
    long = int
try:
    import numpy
    numpy_available=True
except:
    numpy_available=False

class MyBogusType(object):
    def __init__(self, val=0):
        self.val = float(val)

class MyBogusNumericType(MyBogusType):
    def __add__(self, other):
        return MyBogusNumericType(self.val + float(other))

class Test_value(unittest.TestCase):

    def test_none(self):
        val = None
        self.assertEqual(val, value(val))

    def test_bool(self):
        val = False
        self.assertEqual(val, value(val))

    def test_float(self):
        val = 1.1
        self.assertEqual(val, value(val))

    def test_int(self):
        val = 1
        self.assertEqual(val, value(val))

    def test_long(self):
        val = long(1e10)
        self.assertEqual(val, value(val))

    def test_nan(self):
        val = pyutilib.math.nan
        self.assertEqual(id(val), id(value(val)))

    def test_inf(self):
        val = pyutilib.math.infinity
        self.assertEqual(id(val), id(value(val)))

    def test_string(self):
        val = 'foo'
        self.assertEqual(val, value(val))

    def test_const1(self):
        val = NumericConstant(1.0)
        self.assertEqual(1.0, value(val))

    def test_const2(self):
        val = NumericConstant('foo')
        self.assertEqual('foo', value(val))

    def test_const3(self):
        val = NumericConstant(pyutilib.math.nan)
        self.assertEqual(id(pyutilib.math.nan), id(value(val)))

    def test_const4(self):
        val = NumericConstant(pyutilib.math.infinity)
        self.assertEqual(id(pyutilib.math.infinity), id(value(val)))

    def test_error1(self):
        class A(object): pass
        val = A()
        try:
            value(val)
            self.fail("Expected TypeError")
        except TypeError:
            pass

    def test_error2(self):
        val = NumericConstant(None)
        try:
            value(val)
            self.fail("Expected ValueError")
        except ValueError:
            pass


class Test_is_constant(unittest.TestCase):

    def test_none(self):
        self.assertTrue(is_constant(None))

    def test_bool(self):
        self.assertTrue(is_constant(True))

    def test_float(self):
        self.assertTrue(is_constant(1.1))

    def test_int(self):
        self.assertTrue(is_constant(1))

    def test_long(self):
        val = long(1e10)
        self.assertTrue(is_constant(val))

    def test_string(self):
        self.assertTrue(is_constant('foo'))

    def test_const1(self):
        val = NumericConstant(1.0)
        self.assertTrue(is_constant(val))

    def test_const2(self):
        val = NumericConstant('foo')
        self.assertTrue(is_constant(val))

    def test_error(self):
        class A(object): pass
        val = A()
        try:
            is_constant(val)
            self.fail("Expected TypeError")
        except TypeError:
            pass


class Test_as_numeric(unittest.TestCase):

    def test_none(self):
        val = None
        try:
            as_numeric(val)
            self.fail("Expected ValueError")
        except:
            pass

    def test_bool(self):
        val = False
        try:
            as_numeric(val)
            self.fail("Expected ValueError")
        except:
            pass
        val = True
        try:
            as_numeric(val)
            self.fail("Expected ValueError")
        except:
            pass

    def test_float(self):
        val = 1.1
        self.assertEqual(val, as_numeric(val))

    def test_int(self):
        val = 1
        self.assertEqual(val, as_numeric(val))

    def test_long(self):
        val = long(1e10)
        self.assertEqual(val, as_numeric(val))

    def test_string(self):
        val = 'foo'
        try:
            as_numeric(val)
            self.fail("Expected ValueError")
        except:
            pass

    def test_const1(self):
        val = NumericConstant(1.0)
        self.assertEqual(1.0, as_numeric(val))

    def test_const2(self):
        val = NumericConstant('foo')
        try:
            as_numeric(val)
            self.fail("Expected ValueError")
        except:
            pass

    def test_error1(self):
        class A(object): pass
        val = A()
        try:
            as_numeric(val)
            self.fail("Expected TypeError")
        except TypeError:
            pass

    def test_error2(self):
        val = NumericConstant(None)
        num = as_numeric(val)
        try:
            value(num)
            self.fail("Expected ValueError")
        except ValueError:
            pass

    def test_unknownType(self):
        ref = MyBogusType(42)
        try:
            val = as_numeric(ref)
            self.fail("Expected TypeError")
        except TypeError:
            pass

    def test_unknownNumericType(self):
        ref = MyBogusNumericType(42)
        val = as_numeric(ref)
        self.assertEqual(val().val, 42)
        from pyomo.core.base.numvalue import native_numeric_types, native_types
        self.assertIn(MyBogusNumericType, native_numeric_types)
        self.assertIn(MyBogusNumericType, native_types)
        native_numeric_types.remove(MyBogusNumericType)
        native_types.remove(MyBogusNumericType)

    def test_numpy_basic_float_registration(self):
        if not numpy_available:
            self.skipTest("This test requires NumPy")
        from pyomo.core.base.numvalue import native_numeric_types, native_integer_types, native_boolean_types, native_types
        self.assertIn(numpy.float_, native_numeric_types)
        self.assertNotIn(numpy.float_, native_integer_types)
        self.assertIn(numpy.float_, native_boolean_types)
        self.assertIn(numpy.float_, native_types)

    def test_numpy_basic_int_registration(self):
        if not numpy_available:
            self.skipTest("This test requires NumPy")
        from pyomo.core.base.numvalue import native_numeric_types, native_integer_types, native_boolean_types, native_types
        self.assertIn(numpy.int_, native_numeric_types)
        self.assertIn(numpy.int_, native_integer_types)
        self.assertIn(numpy.int_, native_boolean_types)
        self.assertIn(numpy.int_, native_types)

    def test_numpy_basic_bool_registration(self):
        if not numpy_available:
            self.skipTest("This test requires NumPy")
        from pyomo.core.base.numvalue import native_numeric_types, native_integer_types, native_boolean_types, native_types
        self.assertNotIn(numpy.bool_, native_numeric_types)
        self.assertNotIn(numpy.bool_, native_integer_types)
        self.assertIn(numpy.bool_, native_boolean_types)
        self.assertIn(numpy.bool_, native_types)


if __name__ == "__main__":
    unittest.main()

