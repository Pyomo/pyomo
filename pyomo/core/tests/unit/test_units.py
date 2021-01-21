# -*- coding: utf-8 -*-
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
#
import pickle

import pyutilib.th as unittest
from pyomo.environ import (
    ConcreteModel, Var, Param, Set, Constraint, Objective, Expression,
    ExternalFunction, value, sum_product, maximize, units,
    log, log10, exp, sqrt, cos, sin, tan, asin, acos, atan, cosh, sinh,
    tanh, asinh, acosh, atanh, ceil, floor,
)
from pyomo.common.log import LoggingIntercept
from pyomo.util.check_units import assert_units_consistent
from pyomo.core.expr import inequality
import pyomo.core.expr.current as EXPR
from pyomo.core.base.units_container import (
    pint_available, InconsistentUnitsError, UnitsError, PyomoUnitsContainer,
)
from six import StringIO

def python_callback_function(arg1, arg2):
    return 42.0

@unittest.skipIf(not pint_available, 'Testing units requires pint')
class TestPyomoUnit(unittest.TestCase):

    def test_PyomoUnit_NumericValueMethods(self):
        m = ConcreteModel()
        uc = units
        kg = uc.kg

        self.assertEqual(kg.getname(), 'kg')
        self.assertEqual(kg.name, 'kg')
        self.assertEqual(kg.local_name, 'kg')

        m.kg = uc.kg

        self.assertEqual(m.kg.name, 'kg')
        self.assertEqual(m.kg.local_name, 'kg')

        self.assertEqual(kg.is_constant(), False)
        self.assertEqual(kg.is_fixed(), True)
        self.assertEqual(kg.is_parameter_type(), False)
        self.assertEqual(kg.is_variable_type(), False)
        self.assertEqual(kg.is_potentially_variable(), False)
        self.assertEqual(kg.is_named_expression_type(), False)
        self.assertEqual(kg.is_expression_type(), False)
        self.assertEqual(kg.is_component_type(), False)
        self.assertEqual(kg.is_relational(), False)
        self.assertEqual(kg.is_indexed(), False)
        self.assertEqual(kg._compute_polynomial_degree(None), 0)

        with self.assertRaises(TypeError):
            x = float(kg)
        with self.assertRaises(TypeError):
            x = int(kg)

        assert_units_consistent(kg < m.kg)
        assert_units_consistent(kg > m.kg)
        assert_units_consistent(kg <= m.kg)
        assert_units_consistent(kg >= m.kg)
        assert_units_consistent(kg == m.kg)
        assert_units_consistent(kg + m.kg)
        assert_units_consistent(kg - m.kg)

        with self.assertRaises(InconsistentUnitsError):
            assert_units_consistent(kg + 3)

        with self.assertRaises(InconsistentUnitsError):
            assert_units_consistent(kg - 3)

        with self.assertRaises(InconsistentUnitsError):
            assert_units_consistent(3 + kg)

        with self.assertRaises(InconsistentUnitsError):
            assert_units_consistent(3 - kg)

        # should not assert
        # check __mul__
        self.assertEqual(str(uc.get_units(kg*3)), 'kg')
        # check __rmul__
        self.assertEqual(str(uc.get_units(3*kg)), 'kg')
        # check div / truediv
        self.assertEqual(str(uc.get_units(kg/3.0)), 'kg')
        # check rdiv / rtruediv
        self.assertEqual(str(uc.get_units(3.0/kg)), '1/kg')
        # check pow
        self.assertEqual(str(uc.get_units(kg**2)), 'kg**2')

        # check rpow
        x = 2 ** kg  # creation is allowed, only fails when units are "checked"
        with self.assertRaises(UnitsError):
            assert_units_consistent(x)

        x = kg
        x += kg
        self.assertEqual(str(uc.get_units(x)), 'kg')

        x = kg
        x -= 2.0*kg
        self.assertEqual(str(uc.get_units(x)), 'kg')

        x = kg
        x *= 3
        self.assertEqual(str(uc.get_units(x)), 'kg')

        x = kg
        x **= 3
        self.assertEqual(str(uc.get_units(x)), 'kg**3')

        self.assertEqual(str(uc.get_units(-kg)), 'kg')
        self.assertEqual(str(uc.get_units(+kg)), 'kg')
        self.assertEqual(str(uc.get_units(abs(kg))), 'kg')

        self.assertEqual(str(kg), 'kg')
        self.assertEqual(kg.to_string(), 'kg')
        # ToDo: is this really the correct behavior for verbose?
        self.assertEqual(kg.to_string(verbose=True), 'kg')
        self.assertEqual(kg.to_string(), 'kg')
        self.assertEqual(kg.to_string(), 'kg')

        # check __nonzero__ / __bool__
        self.assertEqual(bool(kg), True)

        # __call__ returns 1.0
        self.assertEqual(kg(), 1.0)
        self.assertEqual(value(kg), 1.0)

        # test pprint
        buf = StringIO()
        kg.pprint(ostream=buf)
        self.assertEqual('kg', buf.getvalue())

        # test str representations for dimensionless
        dless = uc.dimensionless
        self.assertEqual('dimensionless', str(dless))


    def _get_check_units_ok(self, x, pyomo_units_container, str_check=None, expected_type=None):
        if expected_type is not None:
            self.assertEqual(expected_type, type(x))

        assert_units_consistent(x)
        if str_check is not None:
            self.assertEqual(str_check, str(pyomo_units_container.get_units(x)))
        else:
            # if str_check is None, then we expect the units to be None
            self.assertIsNone(pyomo_units_container.get_units(x))

    def _get_check_units_fail(self, x, pyomo_units_container, expected_type=None, expected_error=InconsistentUnitsError):
        if expected_type is not None:
            self.assertEqual(expected_type, type(x))

        with self.assertRaises(expected_error):
            assert_units_consistent(x)

        # we also expect get_units to fail
        with self.assertRaises(expected_error):
            pyomo_units_container.get_units(x)

    def test_get_check_units_on_all_expressions(self):
        # this method is going to test all the expression types that should work
        # to be defensive, we will also test that we actually have the expected expression type
        # therefore, if the expression system changes and we get a different expression type,
        # we will know we need to change these tests

        uc = units
        kg = uc.kg
        m = uc.m

        model = ConcreteModel()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        model.p = Param(initialize=42.0, mutable=True)
        model.xkg = Var(units=kg)
        model.ym = Var(units=m)

        # test equality
        self._get_check_units_ok(3.0*kg == 1.0*kg, uc, 'kg', EXPR.EqualityExpression)
        self._get_check_units_fail(3.0*kg == 2.0*m, uc, EXPR.EqualityExpression)

        # test inequality
        self._get_check_units_ok(3.0*kg <= 1.0*kg, uc, 'kg', EXPR.InequalityExpression)
        self._get_check_units_fail(3.0*kg <= 2.0*m, uc, EXPR.InequalityExpression)
        self._get_check_units_ok(3.0*kg >= 1.0*kg, uc, 'kg', EXPR.InequalityExpression)
        self._get_check_units_fail(3.0*kg >= 2.0*m, uc, EXPR.InequalityExpression)

        # test RangedExpression
        self._get_check_units_ok(inequality(3.0*kg, 4.0*kg, 5.0*kg), uc, 'kg', EXPR.RangedExpression)
        self._get_check_units_fail(inequality(3.0*m, 4.0*kg, 5.0*kg), uc, EXPR.RangedExpression)
        self._get_check_units_fail(inequality(3.0*kg, 4.0*m, 5.0*kg), uc, EXPR.RangedExpression)
        self._get_check_units_fail(inequality(3.0*kg, 4.0*kg, 5.0*m), uc, EXPR.RangedExpression)

        # test SumExpression, NPV_SumExpression
        self._get_check_units_ok(3.0*model.x*kg + 1.0*model.y*kg + 3.65*model.z*kg, uc, 'kg', EXPR.SumExpression)
        self._get_check_units_fail(3.0*model.x*kg + 1.0*model.y*m + 3.65*model.z*kg, uc, EXPR.SumExpression)

        self._get_check_units_ok(3.0*kg + 1.0*kg + 2.0*kg, uc, 'kg', EXPR.NPV_SumExpression)
        self._get_check_units_fail(3.0*kg + 1.0*kg + 2.0*m, uc, EXPR.NPV_SumExpression)

        # test ProductExpression, NPV_ProductExpression
        self._get_check_units_ok(model.x*kg * model.y*m, uc, 'kg*m', EXPR.ProductExpression)
        self._get_check_units_ok(3.0*kg * 1.0*m, uc, 'kg*m', EXPR.NPV_ProductExpression)
        self._get_check_units_ok(3.0*kg*m, uc, 'kg*m', EXPR.NPV_ProductExpression)
        # I don't think that there are combinations that can "fail" for products

        # test MonomialTermExpression
        self._get_check_units_ok(model.x*kg, uc, 'kg', EXPR.MonomialTermExpression)

        # test DivisionExpression, NPV_DivisionExpression
        self._get_check_units_ok(1.0/(model.x*kg), uc, '1/kg', EXPR.DivisionExpression)
        self._get_check_units_ok(2.0/kg, uc, '1/kg', EXPR.NPV_DivisionExpression)
        self._get_check_units_ok((model.x*kg)/1.0, uc, 'kg', EXPR.MonomialTermExpression)
        self._get_check_units_ok(kg/2.0, uc, 'kg', EXPR.NPV_DivisionExpression)
        self._get_check_units_ok(model.y*m/(model.x*kg), uc, 'm/kg', EXPR.DivisionExpression)
        self._get_check_units_ok(m/kg, uc, 'm/kg', EXPR.NPV_DivisionExpression)
        # I don't think that there are combinations that can "fail" for products

        # test PowExpression, NPV_PowExpression
        # ToDo: fix the str representation to combine the powers or the expression system
        self._get_check_units_ok((model.x*kg**2)**3, uc, 'kg**6', EXPR.PowExpression) # would want this to be kg**6
        self._get_check_units_fail(kg**model.x, uc, EXPR.PowExpression, UnitsError)
        self._get_check_units_fail(model.x**kg, uc, EXPR.PowExpression, UnitsError)
        self._get_check_units_ok(kg**2, uc, 'kg**2', EXPR.NPV_PowExpression)
        self._get_check_units_fail(3.0**kg, uc, EXPR.NPV_PowExpression, UnitsError)

        # test NegationExpression, NPV_NegationExpression
        self._get_check_units_ok(-(kg*model.x*model.y), uc, 'kg', EXPR.NegationExpression)
        self._get_check_units_ok(-kg, uc, 'kg', EXPR.NPV_NegationExpression)
        # don't think there are combinations that fan "fail" for negation

        # test AbsExpression, NPV_AbsExpression
        self._get_check_units_ok(abs(kg*model.x), uc, 'kg', EXPR.AbsExpression)
        self._get_check_units_ok(abs(kg), uc, 'kg', EXPR.NPV_AbsExpression)
        # don't think there are combinations that fan "fail" for abs

        # test the different UnaryFunctionExpression / NPV_UnaryFunctionExpression types
        # log
        self._get_check_units_ok(log(3.0*model.x), uc, None, EXPR.UnaryFunctionExpression)
        self._get_check_units_fail(log(3.0*kg*model.x), uc, EXPR.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(log(3.0*model.p), uc, None, EXPR.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(log(3.0*kg), uc, EXPR.NPV_UnaryFunctionExpression, UnitsError)
        # log10
        self._get_check_units_ok(log10(3.0*model.x), uc, None, EXPR.UnaryFunctionExpression)
        self._get_check_units_fail(log10(3.0*kg*model.x), uc, EXPR.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(log10(3.0*model.p), uc, None, EXPR.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(log10(3.0*kg), uc, EXPR.NPV_UnaryFunctionExpression, UnitsError)
        # sin
        self._get_check_units_ok(sin(3.0*model.x*uc.radians), uc, None, EXPR.UnaryFunctionExpression)
        self._get_check_units_fail(sin(3.0*kg*model.x), uc, EXPR.UnaryFunctionExpression, UnitsError)
        self._get_check_units_fail(sin(3.0*kg*model.x*uc.kg), uc, EXPR.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(sin(3.0*model.p*uc.radians), uc, None, EXPR.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(sin(3.0*kg), uc, EXPR.NPV_UnaryFunctionExpression, UnitsError)
        # cos
        self._get_check_units_ok(cos(3.0*model.x*uc.radians), uc, None, EXPR.UnaryFunctionExpression)
        self._get_check_units_fail(cos(3.0*kg*model.x), uc, EXPR.UnaryFunctionExpression, UnitsError)
        self._get_check_units_fail(cos(3.0*kg*model.x*uc.kg), uc, EXPR.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(cos(3.0*model.p*uc.radians), uc, None, EXPR.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(cos(3.0*kg), uc, EXPR.NPV_UnaryFunctionExpression, UnitsError)
        # tan
        self._get_check_units_ok(tan(3.0*model.x*uc.radians), uc, None, EXPR.UnaryFunctionExpression)
        self._get_check_units_fail(tan(3.0*kg*model.x), uc, EXPR.UnaryFunctionExpression, UnitsError)
        self._get_check_units_fail(tan(3.0*kg*model.x*uc.kg), uc, EXPR.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(tan(3.0*model.p*uc.radians), uc, None, EXPR.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(tan(3.0*kg), uc, EXPR.NPV_UnaryFunctionExpression, UnitsError)
        # sin
        self._get_check_units_ok(sinh(3.0*model.x*uc.radians), uc, None, EXPR.UnaryFunctionExpression)
        self._get_check_units_fail(sinh(3.0*kg*model.x), uc, EXPR.UnaryFunctionExpression, UnitsError)
        self._get_check_units_fail(sinh(3.0*kg*model.x*uc.kg), uc, EXPR.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(sinh(3.0*model.p*uc.radians), uc, None, EXPR.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(sinh(3.0*kg), uc, EXPR.NPV_UnaryFunctionExpression, UnitsError)
        # cos
        self._get_check_units_ok(cosh(3.0*model.x*uc.radians), uc, None, EXPR.UnaryFunctionExpression)
        self._get_check_units_fail(cosh(3.0*kg*model.x), uc, EXPR.UnaryFunctionExpression, UnitsError)
        self._get_check_units_fail(cosh(3.0*kg*model.x*uc.kg), uc, EXPR.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(cosh(3.0*model.p*uc.radians), uc, None, EXPR.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(cosh(3.0*kg), uc, EXPR.NPV_UnaryFunctionExpression, UnitsError)
        # tan
        self._get_check_units_ok(tanh(3.0*model.x*uc.radians), uc, None, EXPR.UnaryFunctionExpression)
        self._get_check_units_fail(tanh(3.0*kg*model.x), uc, EXPR.UnaryFunctionExpression, UnitsError)
        self._get_check_units_fail(tanh(3.0*kg*model.x*uc.kg), uc, EXPR.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(tanh(3.0*model.p*uc.radians), uc, None, EXPR.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(tanh(3.0*kg), uc, EXPR.NPV_UnaryFunctionExpression, UnitsError)
        # asin
        self._get_check_units_ok(asin(3.0*model.x), uc, 'rad', EXPR.UnaryFunctionExpression)
        self._get_check_units_fail(asin(3.0*kg*model.x), uc, EXPR.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(asin(3.0*model.p), uc, 'rad', EXPR.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(asin(3.0*model.p*kg), uc, EXPR.NPV_UnaryFunctionExpression, UnitsError)
        # acos
        self._get_check_units_ok(acos(3.0*model.x), uc, 'rad', EXPR.UnaryFunctionExpression)
        self._get_check_units_fail(acos(3.0*kg*model.x), uc, EXPR.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(acos(3.0*model.p), uc, 'rad', EXPR.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(acos(3.0*model.p*kg), uc, EXPR.NPV_UnaryFunctionExpression, UnitsError)
        # atan
        self._get_check_units_ok(atan(3.0*model.x), uc, 'rad', EXPR.UnaryFunctionExpression)
        self._get_check_units_fail(atan(3.0*kg*model.x), uc, EXPR.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(atan(3.0*model.p), uc, 'rad', EXPR.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(atan(3.0*model.p*kg), uc, EXPR.NPV_UnaryFunctionExpression, UnitsError)
        # exp
        self._get_check_units_ok(exp(3.0*model.x), uc, None, EXPR.UnaryFunctionExpression)
        self._get_check_units_fail(exp(3.0*kg*model.x), uc, EXPR.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(exp(3.0*model.p), uc, None, EXPR.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(exp(3.0*kg), uc, EXPR.NPV_UnaryFunctionExpression, UnitsError)
        # sqrt
        self._get_check_units_ok(sqrt(3.0*model.x), uc, None, EXPR.UnaryFunctionExpression)
        self._get_check_units_ok(sqrt(3.0*model.x*kg**2), uc, 'kg', EXPR.UnaryFunctionExpression)
        self._get_check_units_ok(sqrt(3.0*model.x*kg), uc, 'kg**0.5', EXPR.UnaryFunctionExpression)
        self._get_check_units_ok(sqrt(3.0*model.p), uc, None, EXPR.NPV_UnaryFunctionExpression)
        self._get_check_units_ok(sqrt(3.0*model.p*kg**2), uc, 'kg', EXPR.NPV_UnaryFunctionExpression)
        self._get_check_units_ok(sqrt(3.0*model.p*kg), uc, 'kg**0.5', EXPR.NPV_UnaryFunctionExpression)
        # asinh
        self._get_check_units_ok(asinh(3.0*model.x), uc, 'rad', EXPR.UnaryFunctionExpression)
        self._get_check_units_fail(asinh(3.0*kg*model.x), uc, EXPR.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(asinh(3.0*model.p), uc, 'rad', EXPR.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(asinh(3.0*model.p*kg), uc, EXPR.NPV_UnaryFunctionExpression, UnitsError)
        # acosh
        self._get_check_units_ok(acosh(3.0*model.x), uc, 'rad', EXPR.UnaryFunctionExpression)
        self._get_check_units_fail(acosh(3.0*kg*model.x), uc, EXPR.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(acosh(3.0*model.p), uc, 'rad', EXPR.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(acosh(3.0*model.p*kg), uc, EXPR.NPV_UnaryFunctionExpression, UnitsError)
        # atanh
        self._get_check_units_ok(atanh(3.0*model.x), uc, 'rad', EXPR.UnaryFunctionExpression)
        self._get_check_units_fail(atanh(3.0*kg*model.x), uc, EXPR.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(atanh(3.0*model.p), uc, 'rad', EXPR.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(atanh(3.0*model.p*kg), uc, EXPR.NPV_UnaryFunctionExpression, UnitsError)
        # ceil
        self._get_check_units_ok(ceil(kg*model.x), uc, 'kg', EXPR.UnaryFunctionExpression)
        self._get_check_units_ok(ceil(kg), uc, 'kg', EXPR.NPV_UnaryFunctionExpression)
        # don't think there are combinations that fan "fail" for ceil
        # floor
        self._get_check_units_ok(floor(kg*model.x), uc, 'kg', EXPR.UnaryFunctionExpression)
        self._get_check_units_ok(floor(kg), uc, 'kg', EXPR.NPV_UnaryFunctionExpression)
        # don't think there are combinations that fan "fail" for floor

        # test Expr_ifExpression
        # consistent if, consistent then/else
        self._get_check_units_ok(EXPR.Expr_if(IF=model.x*kg + kg >= 2.0*kg, THEN=model.x*kg, ELSE=model.y*kg),
                                 uc, 'kg', EXPR.Expr_ifExpression)
        # unitless if, consistent then/else
        self._get_check_units_ok(EXPR.Expr_if(IF=model.x >= 2.0, THEN=model.x*kg, ELSE=model.y*kg),
                                 uc, 'kg', EXPR.Expr_ifExpression)
        # consistent if, unitless then/else
        self._get_check_units_ok(EXPR.Expr_if(IF=model.x*kg + kg >= 2.0*kg, THEN=model.x, ELSE=model.x),
                                 uc, None, EXPR.Expr_ifExpression)
        # inconsistent then/else
        self._get_check_units_fail(EXPR.Expr_if(IF=model.x >= 2.0, THEN=model.x*m, ELSE=model.y*kg),
                                 uc, EXPR.Expr_ifExpression)
        # inconsistent then/else NPV
        self._get_check_units_fail(EXPR.Expr_if(IF=model.x >= 2.0, THEN=model.p*m, ELSE=model.p*kg),
                                 uc, EXPR.Expr_ifExpression)
        # inconsistent then/else NPV units only
        self._get_check_units_fail(EXPR.Expr_if(IF=model.x >= 2.0, THEN=m, ELSE=kg),
                                 uc, EXPR.Expr_ifExpression)

        # test EXPR.IndexTemplate and GetItemExpression
        model.S = Set()
        i = EXPR.IndexTemplate(model.S)
        j = EXPR.IndexTemplate(model.S)
        self._get_check_units_ok(i, uc, None, EXPR.IndexTemplate)

        model.mat = Var(model.S, model.S)
        self._get_check_units_ok(model.mat[i,j+1], uc, None, EXPR.GetItemExpression)

        # test ExternalFunctionExpression, NPV_ExternalFunctionExpression
        model.ef = ExternalFunction(python_callback_function)
        self._get_check_units_ok(model.ef(model.x, model.y), uc, None, EXPR.ExternalFunctionExpression)
        self._get_check_units_ok(model.ef(1.0, 2.0), uc, None, EXPR.NPV_ExternalFunctionExpression)
        self._get_check_units_fail(model.ef(model.x*kg, model.y), uc, EXPR.ExternalFunctionExpression, UnitsError)
        self._get_check_units_fail(model.ef(2.0*kg, 1.0), uc, EXPR.NPV_ExternalFunctionExpression, UnitsError)

        # test ExternalFunctionExpression, NPV_ExternalFunctionExpression
        model.ef2 = ExternalFunction(python_callback_function, units=uc.kg)
        self._get_check_units_ok(model.ef2(model.x, model.y), uc, 'kg', EXPR.ExternalFunctionExpression)
        self._get_check_units_ok(model.ef2(1.0, 2.0), uc, 'kg', EXPR.NPV_ExternalFunctionExpression)
        self._get_check_units_fail(model.ef2(model.x*kg, model.y), uc, EXPR.ExternalFunctionExpression, UnitsError)
        self._get_check_units_fail(model.ef2(2.0*kg, 1.0), uc, EXPR.NPV_ExternalFunctionExpression, UnitsError)

        # test ExternalFunctionExpression, NPV_ExternalFunctionExpression
        model.ef3 = ExternalFunction(python_callback_function, units=uc.kg, arg_units=[uc.kg, uc.m])
        self._get_check_units_fail(model.ef3(model.x, model.y), uc, EXPR.ExternalFunctionExpression)
        self._get_check_units_fail(model.ef3(1.0, 2.0), uc, EXPR.NPV_ExternalFunctionExpression)
        self._get_check_units_fail(model.ef3(model.x*kg, model.y), uc, EXPR.ExternalFunctionExpression, UnitsError)
        self._get_check_units_fail(model.ef3(2.0*kg, 1.0), uc, EXPR.NPV_ExternalFunctionExpression, UnitsError)
        self._get_check_units_ok(model.ef3(2.0*kg, 1.0*uc.m), uc, 'kg', EXPR.NPV_ExternalFunctionExpression)
        self._get_check_units_ok(model.ef3(model.x*kg, model.y*m), uc, 'kg', EXPR.ExternalFunctionExpression)
        self._get_check_units_ok(model.ef3(model.xkg, model.ym), uc, 'kg', EXPR.ExternalFunctionExpression)
        self._get_check_units_fail(model.ef3(model.ym, model.xkg), uc, EXPR.ExternalFunctionExpression, InconsistentUnitsError)

    # @unittest.skip('Skipped testing LinearExpression since StreamBasedExpressionVisitor does not handle LinearExpressions')
    def test_linear_expression(self):
        uc = units
        model = ConcreteModel()
        kg = uc.kg
        m = uc.m

        # test LinearExpression
        # ToDo: Once this test is working correctly, this code should be moved to the test above
        model.vv = Var(['A', 'B', 'C'])
        self._get_check_units_ok(sum_product(model.vv), uc, None, EXPR.LinearExpression)

        linex1 = sum_product(model.vv, {'A': kg, 'B': kg, 'C':kg}, index=['A', 'B', 'C'])
        self._get_check_units_ok(linex1, uc, 'kg', EXPR.LinearExpression)

        linex2 = sum_product(model.vv, {'A': kg, 'B': m, 'C':kg}, index=['A', 'B', 'C'])
        self._get_check_units_fail(linex2, uc, EXPR.LinearExpression)

    def test_named_expression(self):
        uc = units
        m = ConcreteModel()
        m.x = Var(units=uc.kg)
        m.y = Var(units=uc.m)
        m.e = Expression(expr=m.x/m.y)
        self.assertEqual(str(uc.get_units(m.e)), 'kg/m')

    def test_dimensionless(self):
        uc = units
        kg = uc.kg
        dless = uc.dimensionless
        self._get_check_units_ok(2.0 == 2.0*dless, uc, None, EXPR.EqualityExpression)
        self.assertEqual(uc.get_units(2.0*dless), uc.get_units(2.0))
        self.assertIsNone(uc.get_units(2.0*dless))
        self.assertIsNone(uc.get_units(kg/kg))

    def test_temperatures(self):
        uc = units

        # Pyomo units framework disallows "offset" units
        with self.assertRaises(UnitsError):
            degC = uc.celsius
        with self.assertRaises(UnitsError):
            degF = uc.degF

        # although we test delta versions here, users should not use these and
        # use absolute units instead
        delta_degC = uc.delta_degC
        K = uc.kelvin
        delta_degF = uc.delta_degF
        R = uc.rankine

        # In some recent versions of pint, rankine can be either
        # 'rankine' or '°R' (note UTF-8 encoding, which requires the
        # "coding: utf-8" comment flag at the top of this file).
        R_str = R.getname()
        #self.assertIn(R_str, ['rankine', '°R'])

        self._get_check_units_ok(2.0*R + 3.0*R, uc, R_str, EXPR.NPV_SumExpression)
        self._get_check_units_ok(2.0*K + 3.0*K, uc, 'K', EXPR.NPV_SumExpression)

        ex = 2.0*delta_degC + 3.0*delta_degC + 1.0*delta_degC
        self.assertEqual(type(ex), EXPR.NPV_SumExpression)
        assert_units_consistent(ex)

        ex = 2.0*delta_degF + 3.0*delta_degF
        self.assertEqual(type(ex), EXPR.NPV_SumExpression)
        assert_units_consistent(ex)

        self._get_check_units_fail(2.0*K + 3.0*R, uc, EXPR.NPV_SumExpression)
        self._get_check_units_fail(2.0*delta_degC + 3.0*delta_degF, uc, EXPR.NPV_SumExpression)

        self.assertAlmostEqual(uc.convert_temp_K_to_C(323.15), 50.0, places=5)
        self.assertAlmostEqual(uc.convert_temp_C_to_K(50.0), 323.15, places=5)
        self.assertAlmostEqual(uc.convert_temp_R_to_F(509.67), 50.0, places=5)
        self.assertAlmostEqual(uc.convert_temp_F_to_R(50.0), 509.67, places=5)

        with self.assertRaises(UnitsError):
            uc.convert_temp_K_to_C(ex)

    def test_module_example(self):
        from pyomo.environ import ConcreteModel, Var, Objective, units
        model = ConcreteModel()
        model.acc = Var()
        model.obj = Objective(expr=(model.acc*units.m/units.s**2 - 9.81*units.m/units.s**2)**2)
        self.assertEqual('m**2/s**4', str(units.get_units(model.obj.expr)))

    def test_convert_value(self):
        u = units
        x = 0.4535923
        expected_lb_value = 1.0
        actual_lb_value = u.convert_value(num_value=x, from_units=u.kg, to_units=u.lb)
        self.assertAlmostEqual(expected_lb_value, actual_lb_value, places=5)
        actual_lb_value = u.convert_value(num_value=value(x*u.kg), from_units=u.kg, to_units=u.lb)
        self.assertAlmostEqual(expected_lb_value, actual_lb_value, places=5)

        with self.assertRaises(UnitsError):
            # cannot convert from meters to pounds
            actual_lb_value = u.convert_value(num_value=x, from_units=u.meters, to_units=u.lb)

        with self.assertRaises(UnitsError):
            # num_value must be a native numerical type
            actual_lb_value = u.convert_value(num_value=x*u.kg, from_units=u.kg, to_units=u.lb)

    def test_convert(self):
        u = units
        m = ConcreteModel()
        m.dx = Var(units=u.m, initialize=0.10188943773836046)
        m.dy = Var(units=u.m, initialize=0.0)
        m.vx = Var(units=u.m/u.s, initialize=0.7071067769802851)
        m.vy = Var(units=u.m/u.s, initialize=0.7071067769802851)
        m.t = Var(units=u.min, bounds=(1e-5,10.0), initialize=0.0024015570927624456)
        m.theta = Var(bounds=(0, 0.49*3.14), initialize=0.7853981693583533, units=u.radians)
        m.a = Param(initialize=-32.2, units=u.ft/u.s**2)

        m.obj = Objective(expr = m.dx, sense=maximize)
        m.vx_con = Constraint(expr = m.vx == 1.0*u.m/u.s*cos(m.theta))
        m.vy_con = Constraint(expr = m.vy == 1.0*u.m/u.s*sin(m.theta))
        m.dx_con = Constraint(expr = m.dx == m.vx*u.convert(m.t, to_units=u.s))
        m.dy_con = Constraint(expr = m.dy == m.vy*u.convert(m.t, to_units=u.s)
                              + 0.5*(u.convert(m.a, to_units=u.m/u.s**2))*(u.convert(m.t, to_units=u.s))**2)
        m.ground = Constraint(expr = m.dy == 0)

        with self.assertRaises(UnitsError):
            u.convert(m.a, to_units=u.kg)

        self.assertAlmostEqual(value(m.obj), 0.10188943773836046, places=5)
        self.assertAlmostEqual(value(m.vx_con.body), 0.0, places=5)
        self.assertAlmostEqual(value(m.vy_con.body), 0.0, places=5)
        self.assertAlmostEqual(value(m.dx_con.body), 0.0, places=5)
        self.assertAlmostEqual(value(m.dy_con.body), 0.0, places=5)
        self.assertAlmostEqual(value(m.ground.body), 0.0, places=5)

    def test_convert_dimensionless(self):
        u = units
        m = ConcreteModel()
        m.x = Var()
        foo = u.convert(m.x, to_units=u.dimensionless)
        foo = u.convert(m.x, to_units=None)
        foo = u.convert(m.x, to_units=1.0)
        with self.assertRaises(InconsistentUnitsError):
            foo = u.convert(m.x, to_units=u.kg)
        m.y = Var(units=u.kg)
        with self.assertRaises(InconsistentUnitsError):
            foo = u.convert(m.y, to_units=u.dimensionless)
        with self.assertRaises(InconsistentUnitsError):
            foo = u.convert(m.y, to_units=None)
        with self.assertRaises(InconsistentUnitsError):
            foo = u.convert(m.y, to_units=1.0)

    def test_usd(self):
        u = units
        u.load_definitions_from_strings(["USD = [currency]"])
        expr = 3.0*u.USD
        self._get_check_units_ok(expr, u, 'USD')

    def test_clone(self):
        m = ConcreteModel()
        m.x = Var(units=units.kg)
        m.c = Constraint(expr=m.x**2 <= 10*units.kg**2)
        i = m.clone()
        self.assertIs(m.x._units, i.x._units)
        self.assertEqual(str(m.c.upper), str(i.c.upper))
        base = StringIO()
        m.pprint(base)
        test = StringIO()
        i.pprint(test)
        self.assertEqual(base.getvalue(), test.getvalue())

    def test_pickle(self):
        m = ConcreteModel()
        m.x = Var(units=units.kg)
        m.c = Constraint(expr=m.x**2 <= 10*units.kg**2)
        log = StringIO()
        with LoggingIntercept(log, 'pyomo.core.base'):
            i = pickle.loads(pickle.dumps(m))
        self.assertEqual("", log.getvalue())
        self.assertIsNot(m.x, i.x)
        self.assertIsNot(m.x._units, i.x._units)
        self.assertEqual(m.x._units, i.x._units)
        self.assertEqual(str(m.c.upper), str(i.c.upper))
        base = StringIO()
        m.pprint(base)
        test = StringIO()
        i.pprint(test)
        self.assertEqual(base.getvalue(), test.getvalue())

        # Test pickling a custom units manager
        um = PyomoUnitsContainer()
        m = ConcreteModel()
        m.x = Var(units=um.kg)
        m.c = Constraint(expr=m.x**2 <= 10*um.kg**2)
        log = StringIO()
        with LoggingIntercept(log, 'pyomo.core.base'):
            i = pickle.loads(pickle.dumps(m))
        self.assertIn(
            "pickling a _PyomoUnit associated with a PyomoUnitsContainer "
            "that is not the default singleton "
            "(pyomo.core.base.units_container.units)", log.getvalue())
        self.assertIsNot(m.x, i.x)
        self.assertIsNot(m.x._units, i.x._units)
        # Note that pint is inconsistent when comparing standard units
        # across different UnitRegistry instances: older versions of
        # pint would have them compare "not equal" while newer versions
        # compare equal
        #
        # self.assertNotEqual(m.x._units, i.x._units)
        self.assertEqual(str(m.c.upper), str(i.c.upper))
        base = StringIO()
        m.pprint(base)
        test = StringIO()
        i.pprint(test)
        self.assertEqual(base.getvalue(), test.getvalue())

if __name__ == "__main__":
    unittest.main()
