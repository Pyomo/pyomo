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

import pyomo.common.numeric_types as nt
import pyomo.common.unittest as unittest

from pyomo.common.dependencies import numpy, numpy_available
from pyomo.core.expr import LinearExpression
from pyomo.environ import Var

_type_sets = (
    'native_types',
    'native_numeric_types',
    'native_logical_types',
    'native_integer_types',
    'native_complex_types',
)


class TestNativeTypes(unittest.TestCase):
    def setUp(self):
        bool(numpy_available)
        for s in _type_sets:
            setattr(self, s, set(getattr(nt, s)))
            getattr(nt, s).clear()

    def tearDown(self):
        for s in _type_sets:
            getattr(nt, s).clear()
            getattr(nt, s).update(getattr(self, s))

    def test_check_if_native_type(self):
        self.assertEqual(nt.native_types, set())
        self.assertEqual(nt.native_logical_types, set())
        self.assertEqual(nt.native_numeric_types, set())
        self.assertEqual(nt.native_integer_types, set())
        self.assertEqual(nt.native_complex_types, set())

        self.assertTrue(nt.check_if_native_type("a"))
        self.assertIn(str, nt.native_types)
        self.assertNotIn(str, nt.native_logical_types)
        self.assertNotIn(str, nt.native_numeric_types)
        self.assertNotIn(str, nt.native_integer_types)
        self.assertNotIn(str, nt.native_complex_types)

        self.assertTrue(nt.check_if_native_type(1))
        self.assertIn(int, nt.native_types)
        self.assertNotIn(int, nt.native_logical_types)
        self.assertIn(int, nt.native_numeric_types)
        self.assertIn(int, nt.native_integer_types)
        self.assertNotIn(int, nt.native_complex_types)

        self.assertTrue(nt.check_if_native_type(1.5))
        self.assertIn(float, nt.native_types)
        self.assertNotIn(float, nt.native_logical_types)
        self.assertIn(float, nt.native_numeric_types)
        self.assertNotIn(float, nt.native_integer_types)
        self.assertNotIn(float, nt.native_complex_types)

        self.assertTrue(nt.check_if_native_type(True))
        self.assertIn(bool, nt.native_types)
        self.assertIn(bool, nt.native_logical_types)
        self.assertNotIn(bool, nt.native_numeric_types)
        self.assertNotIn(bool, nt.native_integer_types)
        self.assertNotIn(bool, nt.native_complex_types)

        self.assertFalse(nt.check_if_native_type(slice(None, None, None)))
        self.assertNotIn(slice, nt.native_types)
        self.assertNotIn(slice, nt.native_logical_types)
        self.assertNotIn(slice, nt.native_numeric_types)
        self.assertNotIn(slice, nt.native_integer_types)
        self.assertNotIn(slice, nt.native_complex_types)

    def test_check_if_logical_type(self):
        self.assertEqual(nt.native_types, set())
        self.assertEqual(nt.native_logical_types, set())
        self.assertEqual(nt.native_numeric_types, set())
        self.assertEqual(nt.native_integer_types, set())
        self.assertEqual(nt.native_complex_types, set())

        self.assertFalse(nt.check_if_logical_type("a"))
        self.assertNotIn(str, nt.native_types)
        self.assertNotIn(str, nt.native_logical_types)
        self.assertNotIn(str, nt.native_numeric_types)
        self.assertNotIn(str, nt.native_integer_types)
        self.assertNotIn(str, nt.native_complex_types)

        self.assertFalse(nt.check_if_logical_type("a"))

        self.assertTrue(nt.check_if_logical_type(True))
        self.assertIn(bool, nt.native_types)
        self.assertIn(bool, nt.native_logical_types)
        self.assertNotIn(bool, nt.native_numeric_types)
        self.assertNotIn(bool, nt.native_integer_types)
        self.assertNotIn(bool, nt.native_complex_types)

        self.assertTrue(nt.check_if_logical_type(True))

        self.assertFalse(nt.check_if_logical_type(1))
        self.assertNotIn(int, nt.native_types)
        self.assertNotIn(int, nt.native_logical_types)
        self.assertNotIn(int, nt.native_numeric_types)
        self.assertNotIn(int, nt.native_integer_types)
        self.assertNotIn(int, nt.native_complex_types)

        if numpy_available:
            self.assertTrue(nt.check_if_logical_type(numpy.bool_(1)))
            self.assertIn(numpy.bool_, nt.native_types)
            self.assertIn(numpy.bool_, nt.native_logical_types)
            self.assertNotIn(numpy.bool_, nt.native_numeric_types)
            self.assertNotIn(numpy.bool_, nt.native_integer_types)
            self.assertNotIn(numpy.bool_, nt.native_complex_types)

    def test_check_if_numeric_type(self):
        self.assertEqual(nt.native_types, set())
        self.assertEqual(nt.native_logical_types, set())
        self.assertEqual(nt.native_numeric_types, set())
        self.assertEqual(nt.native_integer_types, set())
        self.assertEqual(nt.native_complex_types, set())

        self.assertFalse(nt.check_if_numeric_type("a"))
        self.assertFalse(nt.check_if_numeric_type("a"))
        self.assertNotIn(str, nt.native_types)
        self.assertNotIn(str, nt.native_logical_types)
        self.assertNotIn(str, nt.native_numeric_types)
        self.assertNotIn(str, nt.native_integer_types)
        self.assertNotIn(str, nt.native_complex_types)

        self.assertFalse(nt.check_if_numeric_type(True))
        self.assertFalse(nt.check_if_numeric_type(True))
        self.assertNotIn(bool, nt.native_types)
        self.assertNotIn(bool, nt.native_logical_types)
        self.assertNotIn(bool, nt.native_numeric_types)
        self.assertNotIn(bool, nt.native_integer_types)
        self.assertNotIn(bool, nt.native_complex_types)

        self.assertTrue(nt.check_if_numeric_type(1))
        self.assertTrue(nt.check_if_numeric_type(1))
        self.assertIn(int, nt.native_types)
        self.assertNotIn(int, nt.native_logical_types)
        self.assertIn(int, nt.native_numeric_types)
        self.assertIn(int, nt.native_integer_types)
        self.assertNotIn(int, nt.native_complex_types)

        self.assertTrue(nt.check_if_numeric_type(1.5))
        self.assertTrue(nt.check_if_numeric_type(1.5))
        self.assertIn(float, nt.native_types)
        self.assertNotIn(float, nt.native_logical_types)
        self.assertIn(float, nt.native_numeric_types)
        self.assertNotIn(float, nt.native_integer_types)
        self.assertNotIn(float, nt.native_complex_types)

        self.assertFalse(nt.check_if_numeric_type(1j))
        self.assertIn(complex, nt.native_types)
        self.assertNotIn(complex, nt.native_logical_types)
        self.assertNotIn(complex, nt.native_numeric_types)
        self.assertNotIn(complex, nt.native_integer_types)
        self.assertIn(complex, nt.native_complex_types)

        v = Var()
        v.construct()
        self.assertFalse(nt.check_if_numeric_type(v))
        self.assertNotIn(type(v), nt.native_types)
        self.assertNotIn(type(v), nt.native_logical_types)
        self.assertNotIn(type(v), nt.native_numeric_types)
        self.assertNotIn(type(v), nt.native_integer_types)
        self.assertNotIn(type(v), nt.native_complex_types)

        e = LinearExpression([1])
        self.assertFalse(nt.check_if_numeric_type(e))
        self.assertNotIn(type(e), nt.native_types)
        self.assertNotIn(type(e), nt.native_logical_types)
        self.assertNotIn(type(e), nt.native_numeric_types)
        self.assertNotIn(type(e), nt.native_integer_types)
        self.assertNotIn(type(e), nt.native_complex_types)

        if numpy_available:
            self.assertFalse(nt.check_if_numeric_type(numpy.bool_(1)))
            self.assertNotIn(numpy.bool_, nt.native_types)
            self.assertNotIn(numpy.bool_, nt.native_logical_types)
            self.assertNotIn(numpy.bool_, nt.native_numeric_types)
            self.assertNotIn(numpy.bool_, nt.native_integer_types)
            self.assertNotIn(numpy.bool_, nt.native_complex_types)

            self.assertFalse(nt.check_if_numeric_type(numpy.array([1])))
            self.assertNotIn(numpy.ndarray, nt.native_types)
            self.assertNotIn(numpy.ndarray, nt.native_logical_types)
            self.assertNotIn(numpy.ndarray, nt.native_numeric_types)
            self.assertNotIn(numpy.ndarray, nt.native_integer_types)
            self.assertNotIn(numpy.ndarray, nt.native_complex_types)

            self.assertTrue(nt.check_if_numeric_type(numpy.float64(1)))
            self.assertIn(numpy.float64, nt.native_types)
            self.assertNotIn(numpy.float64, nt.native_logical_types)
            self.assertIn(numpy.float64, nt.native_numeric_types)
            self.assertNotIn(numpy.float64, nt.native_integer_types)
            self.assertNotIn(numpy.float64, nt.native_complex_types)

            self.assertTrue(nt.check_if_numeric_type(numpy.int64(1)))
            self.assertIn(numpy.int64, nt.native_types)
            self.assertNotIn(numpy.int64, nt.native_logical_types)
            self.assertIn(numpy.int64, nt.native_numeric_types)
            self.assertIn(numpy.int64, nt.native_integer_types)
            self.assertNotIn(numpy.int64, nt.native_complex_types)

            self.assertFalse(nt.check_if_numeric_type(numpy.complex128(1)))
            self.assertIn(numpy.complex128, nt.native_types)
            self.assertNotIn(numpy.complex128, nt.native_logical_types)
            self.assertNotIn(numpy.complex128, nt.native_numeric_types)
            self.assertNotIn(numpy.complex128, nt.native_integer_types)
            self.assertIn(numpy.complex128, nt.native_complex_types)
