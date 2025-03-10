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
#
# Unit Tests for Python numeric values
#

import subprocess
import sys
from math import nan, inf

import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.core.base.units_container import pint_available

from pyomo.environ import (
    value,
    ConcreteModel,
    Param,
    Var,
    polynomial_degree,
    is_constant,
    is_fixed,
    is_potentially_variable,
    is_variable_type,
)

from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.numvalue import (
    NumericConstant,
    as_numeric,
    is_numeric_data,
    native_types,
    native_numeric_types,
    native_integer_types,
)
from pyomo.common.numeric_types import _native_boolean_types


class MyBogusType(object):
    def __init__(self, val=0):
        self.val = float(val)


class MyBogusNumericType(MyBogusType):
    def __add__(self, other):
        if other.__class__ in native_numeric_types:
            return MyBogusNumericType(self.val + float(other))
        else:
            return NotImplemented

    def __le__(self, other):
        if other.__class__ in native_numeric_types:
            return self.val <= float(other)
        else:
            return NotImplemented

    def __lt__(self, other):
        return self.val < float(other)

    def __ge__(self, other):
        return self.val >= float(other)


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
        class A(object):
            pass

        val = A()
        self.assertEqual(False, is_numeric_data(val))

    def test_unknownNumericType(self):
        ref = MyBogusNumericType(42)
        self.assertTrue(is_numeric_data(ref))
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
        val = int(1e10)
        self.assertEqual(val, value(val))

    def test_nan(self):
        val = nan
        self.assertEqual(id(val), id(value(val)))

    def test_inf(self):
        val = inf
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
        val = NumericConstant(inf)
        self.assertEqual(id(inf), id(value(val)))

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
        class A(object):
            pass

        val = A()
        with self.assertRaisesRegex(
            TypeError, "Cannot evaluate object with unknown type: A"
        ):
            value(val)

    def test_unknownType(self):
        ref = MyBogusType(42)
        with self.assertRaisesRegex(
            TypeError, "Cannot evaluate object with unknown type: MyBogusType"
        ):
            value(ref)

    def test_unknownNumericType(self):
        ref = MyBogusNumericType(42)
        val = value(ref)
        self.assertEqual(val.val, 42.0)
        # self.assertEqual(val().val, 42)
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
        with self.assertRaisesRegex(
            TypeError,
            "Cannot evaluate the polynomial degree of a non-numeric type: bool",
        ):
            polynomial_degree(val)

    def test_float(self):
        val = 1.1
        self.assertEqual(0, polynomial_degree(val))

    def test_int(self):
        val = 1
        self.assertEqual(0, polynomial_degree(val))

    def test_long(self):
        val = int(1e10)
        self.assertEqual(0, polynomial_degree(val))

    def test_nan(self):
        val = nan
        self.assertEqual(0, polynomial_degree(val))

    def test_inf(self):
        val = inf
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
        val = NumericConstant(inf)
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
        class A(object):
            pass

        val = A()
        with self.assertRaisesRegex(
            TypeError, "Cannot assess properties of object with unknown type: A"
        ):
            polynomial_degree(val)

    def test_unknownNumericType(self):
        ref = MyBogusNumericType(42)
        val = polynomial_degree(ref)
        self.assertEqual(val, 0)
        # self.assertEqual(val().val, 42)
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
        val = int(1e10)
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
        class A(object):
            pass

        val = A()
        with self.assertRaisesRegex(
            TypeError, "Cannot assess properties of object with unknown type: A"
        ):
            is_constant(val)

    def test_unknownNumericType(self):
        ref = MyBogusNumericType(42)
        self.assertTrue(is_constant(ref))
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
        val = int(1e10)
        self.assertTrue(is_fixed(val))

    def test_string(self):
        self.assertTrue(is_fixed('foo'))

    def test_const1(self):
        val = NumericConstant(1.0)
        self.assertTrue(is_fixed(val))

    def test_error(self):
        class A(object):
            pass

        val = A()
        with self.assertRaisesRegex(
            TypeError, "Cannot assess properties of object with unknown type: A"
        ):
            is_fixed(val)

    def test_unknownNumericType(self):
        ref = MyBogusNumericType(42)
        self.assertTrue(is_fixed(ref))
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
        val = int(1e10)
        self.assertFalse(is_variable_type(val))

    def test_string(self):
        self.assertFalse(is_variable_type('foo'))

    def test_const1(self):
        val = NumericConstant(1.0)
        self.assertFalse(is_variable_type(val))

    def test_error(self):
        class A(object):
            pass

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
        val = int(1e10)
        self.assertFalse(is_potentially_variable(val))

    def test_string(self):
        self.assertFalse(is_potentially_variable('foo'))

    def test_const1(self):
        val = NumericConstant(1.0)
        self.assertFalse(is_potentially_variable(val))

    def test_error(self):
        class A(object):
            pass

        val = A()
        self.assertFalse(is_potentially_variable(val))

    def test_unknownNumericType(self):
        ref = MyBogusNumericType(42)
        self.assertFalse(is_potentially_variable(ref))


class Test_as_numeric(unittest.TestCase):
    def test_none(self):
        val = None
        with self.assertRaisesRegex(
            TypeError,
            r"NoneType values \('None'\) are not allowed "
            "in Pyomo numeric expressions",
        ):
            as_numeric(val)

    def test_bool(self):
        with self.assertRaisesRegex(
            TypeError,
            r"bool values \('False'\) are not allowed in Pyomo numeric expressions",
        ):
            as_numeric(False)
        with self.assertRaisesRegex(
            TypeError,
            r"bool values \('True'\) are not allowed in Pyomo numeric expressions",
        ):
            as_numeric(True)

    def test_float(self):
        val = 1.1
        nval = as_numeric(val)
        self.assertEqual(val, nval)
        self.assertEqual(nval / 2, 0.55)

    def test_int(self):
        val = 1
        nval = as_numeric(val)
        self.assertEqual(1.0, nval)
        # self.assertEqual(val, nval)
        self.assertEqual(nval / 2, 0.5)

    def test_long(self):
        val = int(1e10)
        nval = as_numeric(val)
        self.assertEqual(1.0e10, nval)
        # self.assertEqual(val, as_numeric(val))
        self.assertEqual(nval / 2, 5.0e9)

    def test_string(self):
        val = 'foo'
        with self.assertRaisesRegex(
            TypeError,
            r"str values \('foo'\) are not allowed in Pyomo numeric expressions",
        ):
            as_numeric(val)

    def test_const1(self):
        val = NumericConstant(1.0)
        self.assertEqual(1.0, as_numeric(val))

    def test_error1(self):
        class A(object):
            pass

        val = A()
        with self.assertRaisesRegex(
            TypeError,
            r"Cannot treat the value '.*' as a "
            "numeric value because it has unknown type 'A'",
        ):
            as_numeric(val)

    def test_unknownType(self):
        ref = MyBogusType(42)
        with self.assertRaisesRegex(
            TypeError,
            r"Cannot treat the value '.*' as a "
            "numeric value because it has unknown type 'MyBogusType'",
        ):
            as_numeric(ref)

    def test_non_numeric_component(self):
        m = ConcreteModel()
        m.v = Var([1, 2])
        with self.assertRaisesRegex(
            TypeError,
            "The 'IndexedVar' object 'v' is not a valid "
            "type for Pyomo numeric expressions",
        ):
            as_numeric(m.v)

        obj = PyomoObject()
        with self.assertRaisesRegex(
            TypeError,
            "The 'PyomoObject' object '.*' is not a valid "
            "type for Pyomo numeric expressions",
        ):
            as_numeric(obj)

    def test_unknownNumericType(self):
        ref = MyBogusNumericType(42)
        self.assertNotIn(MyBogusNumericType, native_numeric_types)
        self.assertNotIn(MyBogusNumericType, native_types)
        try:
            val = as_numeric(ref)
            self.assertEqual(val().val, 42.0)
            self.assertIn(MyBogusNumericType, native_numeric_types)
            self.assertIn(MyBogusNumericType, native_types)
        finally:
            native_numeric_types.remove(MyBogusNumericType)
            native_types.remove(MyBogusNumericType)

    @unittest.skipUnless(numpy_available, "This test requires NumPy")
    def test_numpy_basic_float_registration(self):
        self.assertIn(numpy.float64, native_numeric_types)
        self.assertNotIn(numpy.float64, native_integer_types)
        self.assertIn(numpy.float64, _native_boolean_types)
        self.assertIn(numpy.float64, native_types)

    @unittest.skipUnless(numpy_available, "This test requires NumPy")
    def test_numpy_basic_int_registration(self):
        self.assertIn(numpy.int_, native_numeric_types)
        self.assertIn(numpy.int_, native_integer_types)
        self.assertIn(numpy.int_, _native_boolean_types)
        self.assertIn(numpy.int_, native_types)

    @unittest.skipUnless(numpy_available, "This test requires NumPy")
    def test_numpy_basic_bool_registration(self):
        self.assertNotIn(numpy.bool_, native_numeric_types)
        self.assertNotIn(numpy.bool_, native_integer_types)
        self.assertIn(numpy.bool_, _native_boolean_types)
        self.assertIn(numpy.bool_, native_types)

    @unittest.skipUnless(numpy_available, "This test requires NumPy")
    def test_automatic_numpy_registration(self):
        cmd = (
            'from pyomo.common.numeric_types import native_numeric_types as nnt; '
            'print("float64" in [_.__name__ for _ in nnt]); '
            'import numpy; '
            'print("float64" in [_.__name__ for _ in nnt])'
        )

        rc = subprocess.run(
            [sys.executable, '-c', cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.assertEqual((rc.returncode, rc.stdout), (0, "False\nTrue\n"))

        cmd = (
            'import numpy; '
            'from pyomo.common.numeric_types import native_numeric_types as nnt; '
            'print("float64" in [_.__name__ for _ in nnt])'
        )

        rc = subprocess.run(
            [sys.executable, '-c', cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.assertEqual((rc.returncode, rc.stdout), (0, "True\n"))

    def test_unknownNumericType_expr_registration(self):
        cmd = (
            'import pyomo; '
            'from pyomo.core.base import Var, Param; '
            'from pyomo.core.base.units_container import units; '
            'from pyomo.common.numeric_types import native_numeric_types as nnt; '
            f'from {__name__} import MyBogusNumericType; '
            'ref = MyBogusNumericType(42); '
            'print(MyBogusNumericType in nnt); %s; print(MyBogusNumericType in nnt); '
        )

        def _tester(expr):
            rc = subprocess.run(
                [sys.executable, '-c', cmd % expr],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(
                (rc.returncode, rc.stdout),
                (
                    0,
                    '''False
WARNING: Dynamically registering the following numeric type:
        pyomo.core.tests.unit.test_numvalue.MyBogusNumericType
    Dynamic registration is supported for convenience, but there are known
    limitations to this approach.  We recommend explicitly registering numeric
    types using RegisterNumericType() or RegisterIntegerType().
True
''',
                ),
            )

        _tester('Var() <= ref')
        _tester('ref <= Var()')
        _tester('ref + Var()')
        _tester('Var() + ref')
        _tester('v = Var(); v.construct(); v.value = ref')
        _tester('p = Param(mutable=True); p.construct(); p.value = ref')
        if pint_available:
            _tester('v = Var(units=units.m); v.construct(); v.value = ref')
            _tester(
                'p = Param(mutable=True, units=units.m); p.construct(); p.value = ref'
            )


if __name__ == "__main__":
    unittest.main()
