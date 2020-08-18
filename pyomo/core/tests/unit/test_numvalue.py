#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Unit Tests for Python numeric values
#

import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

from pyutilib.math import nan, infinity
import pyutilib.th as unittest

from pyomo.environ import (value, ConcreteModel, Param, Var, 
                           polynomial_degree, is_constant, is_fixed,
                           is_potentially_variable, is_variable_type)
from pyomo.core.expr.numvalue import (NumericConstant,
                                      as_numeric,
                                      is_numeric_data)

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
        try:
            value(val)
            self.fail("Expected ValueError")
        except ValueError:
            pass

    def test_bool(self):
        val = False
        self.assertEqual(val, value(val))
        val = True
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

    def test_string(self):
        val = 'foo'
        try:
            value(val)
            self.fail("Expected ValueError")
        except ValueError:
            pass

    def test_const1(self):
        val = NumericConstant(1.0)
        self.assertEqual(1.0, value(val))

    def test_error1(self):
        class A(object): pass
        val = A()
        try:
            value(val)
            self.fail("Expected TypeError")
        except TypeError:
            pass

    def test_unknownType(self):
        ref = MyBogusType(42)
        try:
            val = value(ref)
            self.fail("Expected TypeError")
        except TypeError:
            pass

    def test_unknownNumericType(self):
        ref = MyBogusNumericType(42)
        val = value(ref)
        self.assertEqual(val().val, 42)
        from pyomo.core.base.numvalue import native_numeric_types, native_types
        self.assertIn(MyBogusNumericType, native_numeric_types)
        self.assertIn(MyBogusNumericType, native_types)
        native_numeric_types.remove(MyBogusNumericType)
        native_types.remove(MyBogusNumericType)


class Test_is_numeric_data(unittest.TestCase):

    def test_string(self):
        self.assertEqual(is_numeric_data("a"), False)
        self.assertEqual(is_numeric_data(b"a"), False)

    def test_float(self):
        self.assertEqual(is_numeric_data(0.0), True)

    def test_int(self):
        self.assertEqual(is_numeric_data(0), True)

    def test_NumericValue(self):
        self.assertEqual(is_numeric_data(NumericConstant(1.0)), True)

    def test_error(self):
        class A(object): pass
        val = A()
        self.assertEqual(False, is_numeric_data(val))

    def test_unknownNumericType(self):
        ref = MyBogusNumericType(42)
        self.assertTrue(is_numeric_data(ref))
        from pyomo.core.base.numvalue import native_numeric_types, native_types
        self.assertIn(MyBogusNumericType, native_numeric_types)
        self.assertIn(MyBogusNumericType, native_types)
        native_numeric_types.remove(MyBogusNumericType)
        native_types.remove(MyBogusNumericType)


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
        val = nan
        self.assertEqual(id(val), id(value(val)))

    def test_inf(self):
        val = infinity
        self.assertEqual(id(val), id(value(val)))

    def test_string(self):
        val = 'foo'
        self.assertEqual(val, value(val))

    def test_const1(self):
        val = NumericConstant(1.0)
        self.assertEqual(1.0, value(val))

    def test_const3(self):
        val = NumericConstant(nan)
        self.assertEqual(id(nan), id(value(val)))

    def test_const4(self):
        val = NumericConstant(infinity)
        self.assertEqual(id(infinity), id(value(val)))

    def test_param1(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=2)
        self.assertEqual(2, value(m.p))

    def test_param2(self):
        m = ConcreteModel()
        m.p = Param(mutable=True)
        self.assertRaises(ValueError, value, m.p, exception=True)

    def test_param3(self):
        m = ConcreteModel()
        m.p = Param(mutable=True)
        self.assertEqual(None, value(m.p, exception=False))

    def test_var1(self):
        m = ConcreteModel()
        m.x = Var()
        self.assertRaises(ValueError, value, m.x, exception=True)

    def test_var2(self):
        m = ConcreteModel()
        m.x = Var()
        self.assertEqual(None, value(m.x, exception=False))

    def test_error1(self):
        class A(object): pass
        val = A()
        try:
            value(val)
            self.fail("Expected TypeError")
        except TypeError:
            pass

    def test_unknownNumericType(self):
        ref = MyBogusNumericType(42)
        val = value(ref)
        self.assertEqual(val.val, 42.0)
        #self.assertEqual(val().val, 42)
        from pyomo.core.base.numvalue import native_numeric_types, native_types
        self.assertIn(MyBogusNumericType, native_numeric_types)
        self.assertIn(MyBogusNumericType, native_types)
        native_numeric_types.remove(MyBogusNumericType)
        native_types.remove(MyBogusNumericType)


class Test_polydegree(unittest.TestCase):

    def test_none(self):
        val = None
        self.assertRaises(TypeError, polynomial_degree, val)

    def test_bool(self):
        val = False
        self.assertEqual(0, polynomial_degree(val))

    def test_float(self):
        val = 1.1
        self.assertEqual(0, polynomial_degree(val))

    def test_int(self):
        val = 1
        self.assertEqual(0, polynomial_degree(val))

    def test_long(self):
        val = long(1e10)
        self.assertEqual(0, polynomial_degree(val))

    def test_nan(self):
        val = nan
        self.assertEqual(0, polynomial_degree(val))

    def test_inf(self):
        val = infinity
        self.assertEqual(0, polynomial_degree(val))

    def test_string(self):
        val = 'foo'
        self.assertRaises(TypeError, polynomial_degree, val)

    def test_const1(self):
        val = NumericConstant(1.0)
        self.assertEqual(0, polynomial_degree(val))

    def test_const3(self):
        val = NumericConstant(nan)
        self.assertEqual(0, polynomial_degree(val))

    def test_const4(self):
        val = NumericConstant(infinity)
        self.assertEqual(0, polynomial_degree(val))

    def test_param1(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=2)
        self.assertEqual(0, polynomial_degree(m.p))

    def test_param2(self):
        m = ConcreteModel()
        m.p = Param(mutable=True)
        self.assertEqual(0, polynomial_degree(m.p))

    def test_var1(self):
        m = ConcreteModel()
        m.x = Var()
        self.assertTrue(1, polynomial_degree(m.x))

    def test_error1(self):
        class A(object): pass
        val = A()
        try:
            polynomial_degree(val)
            self.fail("Expected TypeError")
        except TypeError:
            pass

    def test_unknownNumericType(self):
        ref = MyBogusNumericType(42)
        val = polynomial_degree(ref)
        self.assertEqual(val, 0)
        #self.assertEqual(val().val, 42)
        from pyomo.core.base.numvalue import native_numeric_types, native_types
        self.assertIn(MyBogusNumericType, native_numeric_types)
        self.assertIn(MyBogusNumericType, native_types)
        native_numeric_types.remove(MyBogusNumericType)
        native_types.remove(MyBogusNumericType)


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

    def test_unknownNumericType(self):
        ref = MyBogusNumericType(42)
        self.assertTrue(is_constant(ref))
        from pyomo.core.base.numvalue import native_numeric_types, native_types
        self.assertIn(MyBogusNumericType, native_numeric_types)
        self.assertIn(MyBogusNumericType, native_types)
        native_numeric_types.remove(MyBogusNumericType)
        native_types.remove(MyBogusNumericType)


class Test_is_fixed(unittest.TestCase):

    def test_none(self):
        self.assertTrue(is_fixed(None))

    def test_bool(self):
        self.assertTrue(is_fixed(True))

    def test_float(self):
        self.assertTrue(is_fixed(1.1))

    def test_int(self):
        self.assertTrue(is_fixed(1))

    def test_long(self):
        val = long(1e10)
        self.assertTrue(is_fixed(val))

    def test_string(self):
        self.assertTrue(is_fixed('foo'))

    def test_const1(self):
        val = NumericConstant(1.0)
        self.assertTrue(is_fixed(val))

    def test_error(self):
        class A(object): pass
        val = A()
        try:
            is_fixed(val)
            self.fail("Expected TypeError")
        except TypeError:
            pass

    def test_unknownNumericType(self):
        ref = MyBogusNumericType(42)
        self.assertTrue(is_fixed(ref))
        from pyomo.core.base.numvalue import native_numeric_types, native_types
        self.assertIn(MyBogusNumericType, native_numeric_types)
        self.assertIn(MyBogusNumericType, native_types)
        native_numeric_types.remove(MyBogusNumericType)
        native_types.remove(MyBogusNumericType)


class Test_is_variable_type(unittest.TestCase):

    def test_none(self):
        self.assertFalse(is_variable_type(None))

    def test_bool(self):
        self.assertFalse(is_variable_type(True))

    def test_float(self):
        self.assertFalse(is_variable_type(1.1))

    def test_int(self):
        self.assertFalse(is_variable_type(1))

    def test_long(self):
        val = long(1e10)
        self.assertFalse(is_variable_type(val))

    def test_string(self):
        self.assertFalse(is_variable_type('foo'))

    def test_const1(self):
        val = NumericConstant(1.0)
        self.assertFalse(is_variable_type(val))

    def test_error(self):
        class A(object): pass
        val = A()
        self.assertFalse(is_variable_type(val))

    def test_unknownNumericType(self):
        ref = MyBogusNumericType(42)
        self.assertFalse(is_variable_type(ref))


class Test_is_potentially_variable(unittest.TestCase):

    def test_none(self):
        self.assertFalse(is_potentially_variable(None))

    def test_bool(self):
        self.assertFalse(is_potentially_variable(True))

    def test_float(self):
        self.assertFalse(is_potentially_variable(1.1))

    def test_int(self):
        self.assertFalse(is_potentially_variable(1))

    def test_long(self):
        val = long(1e10)
        self.assertFalse(is_potentially_variable(val))

    def test_string(self):
        self.assertFalse(is_potentially_variable('foo'))

    def test_const1(self):
        val = NumericConstant(1.0)
        self.assertFalse(is_potentially_variable(val))

    def test_error(self):
        class A(object): pass
        val = A()
        self.assertFalse(is_potentially_variable(val))

    def test_unknownNumericType(self):
        ref = MyBogusNumericType(42)
        self.assertFalse(is_potentially_variable(ref))


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
        nval = as_numeric(val)
        self.assertEqual(val, nval)
        self.assertEqual(nval/2, 0.55)

    def test_int(self):
        val = 1
        nval = as_numeric(val)
        self.assertEqual(1.0, nval)
        #self.assertEqual(val, nval)
        self.assertEqual(nval/2, 0.5)

    def test_long(self):
        val = long(1e10)
        nval = as_numeric(val)
        self.assertEqual(1.0e10, nval)
        #self.assertEqual(val, as_numeric(val))
        self.assertEqual(nval/2, 5.0e9)

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

    def test_error1(self):
        class A(object): pass
        val = A()
        try:
            as_numeric(val)
            self.fail("Expected TypeError")
        except TypeError:
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
        self.assertEqual(val().val, 42.0)
        #self.assertEqual(val().val, 42)
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

