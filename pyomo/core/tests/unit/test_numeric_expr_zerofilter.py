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
# Unit Tests for expression generation
#
import math
import operator

import pyomo.common.unittest as unittest

from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
    DivisionExpression,
    NPV_DivisionExpression,
    SumExpression,
    NPV_SumExpression,
    LinearExpression,
    MonomialTermExpression,
    NegationExpression,
    NPV_NegationExpression,
    ProductExpression,
    NPV_ProductExpression,
    PowExpression,
    NPV_PowExpression,
    AbsExpression,
    NPV_AbsExpression,
    UnaryFunctionExpression,
    NPV_UnaryFunctionExpression,
)
from pyomo.core.expr.numeric_expr import (
    ARG_TYPE,
    _categorize_arg_type,
    _known_arg_types,
    _MutableSumExpression,
    _MutableLinearExpression,
    _MutableNPVSumExpression,
    enable_expression_optimizations,
)
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types

from .test_numeric_expr_dispatcher import BaseNumeric


class TestExpressionGeneration_ZeroFilter(BaseNumeric, unittest.TestCase):
    def setUp(self):
        super().setUp()
        enable_expression_optimizations(zero=True, one=True)

    #
    #
    # ADDITION
    #
    #

    def test_add_invalid(self):
        tests = [
            # "invalid(str) + invalid(str)" is a legitimate Python
            # operation and should never hit the Pyomo expression
            # system
            (self.invalid, self.invalid, self.SKIP),
            (self.invalid, self.asbinary, NotImplemented),
            (self.invalid, self.zero, NotImplemented),
            (self.invalid, self.one, NotImplemented),
            # 4:
            (self.invalid, self.native, NotImplemented),
            (self.invalid, self.npv, NotImplemented),
            (self.invalid, self.param, NotImplemented),
            (self.invalid, self.param_mut, NotImplemented),
            # 8:
            (self.invalid, self.var, NotImplemented),
            (self.invalid, self.mon_native, NotImplemented),
            (self.invalid, self.mon_param, NotImplemented),
            (self.invalid, self.mon_npv, NotImplemented),
            # 12:
            (self.invalid, self.linear, NotImplemented),
            (self.invalid, self.sum, NotImplemented),
            (self.invalid, self.other, NotImplemented),
            (self.invalid, self.mutable_l0, NotImplemented),
            # 16:
            (self.invalid, self.mutable_l1, NotImplemented),
            (self.invalid, self.mutable_l2, NotImplemented),
            (self.invalid, self.param0, NotImplemented),
            (self.invalid, self.param1, NotImplemented),
            # 20:
            (self.invalid, self.mutable_l3, NotImplemented),
        ]
        self._run_cases(tests, operator.add)
        self._run_cases(tests, operator.iadd)

    def test_add_asbinary(self):
        tests = [
            (self.asbinary, self.invalid, NotImplemented),
            # BooleanVar objects do not support addition
            (self.asbinary, self.asbinary, NotImplemented),
            (self.asbinary, self.zero, self.bin),
            (self.asbinary, self.one, LinearExpression([self.bin, 1])),
            # 4:
            (self.asbinary, self.native, LinearExpression([self.bin, 5])),
            (self.asbinary, self.npv, LinearExpression([self.bin, self.npv])),
            (self.asbinary, self.param, LinearExpression([self.bin, 6])),
            (
                self.asbinary,
                self.param_mut,
                LinearExpression([self.bin, self.param_mut]),
            ),
            # 8:
            (self.asbinary, self.var, LinearExpression([self.bin, self.var])),
            (
                self.asbinary,
                self.mon_native,
                LinearExpression([self.bin, self.mon_native]),
            ),
            (
                self.asbinary,
                self.mon_param,
                LinearExpression([self.bin, self.mon_param]),
            ),
            (self.asbinary, self.mon_npv, LinearExpression([self.bin, self.mon_npv])),
            # 12:
            (
                self.asbinary,
                self.linear,
                LinearExpression(self.linear.args + [self.bin]),
            ),
            (self.asbinary, self.sum, SumExpression(self.sum.args + [self.bin])),
            (self.asbinary, self.other, SumExpression([self.bin, self.other])),
            (self.asbinary, self.mutable_l0, self.bin),
            # 16:
            (
                self.asbinary,
                self.mutable_l1,
                LinearExpression([self.bin, self.mon_npv]),
            ),
            (self.asbinary, self.mutable_l2, SumExpression(self.l2.args + [self.bin])),
            (self.asbinary, self.param0, self.bin),
            (self.asbinary, self.param1, LinearExpression([self.bin, 1])),
            # 20:
            (self.asbinary, self.mutable_l3, LinearExpression([self.bin, self.npv])),
        ]
        self._run_cases(tests, operator.add)
        self._run_cases(tests, operator.iadd)

    def test_add_zero(self):
        tests = [
            (self.zero, self.invalid, NotImplemented),
            (self.zero, self.asbinary, self.bin),
            (self.zero, self.zero, 0),
            (self.zero, self.one, 1),
            # 4:
            (self.zero, self.native, 5),
            (self.zero, self.npv, self.npv),
            (self.zero, self.param, 6),
            (self.zero, self.param_mut, self.param_mut),
            # 8:
            (self.zero, self.var, self.var),
            (self.zero, self.mon_native, self.mon_native),
            (self.zero, self.mon_param, self.mon_param),
            (self.zero, self.mon_npv, self.mon_npv),
            # 12:
            (self.zero, self.linear, self.linear),
            (self.zero, self.sum, self.sum),
            (self.zero, self.other, self.other),
            (self.zero, self.mutable_l0, 0),
            # 16:
            (self.zero, self.mutable_l1, self.mon_npv),
            (self.zero, self.mutable_l2, self.l2),
            (self.zero, self.param0, 0),
            (self.zero, self.param1, 1),
            # 20:
            (self.zero, self.mutable_l3, self.npv),
        ]
        self._run_cases(tests, operator.add)
        self._run_cases(tests, operator.iadd)

    def test_add_one(self):
        tests = [
            (self.one, self.invalid, NotImplemented),
            (self.one, self.asbinary, LinearExpression([1, self.bin])),
            (self.one, self.zero, 1),
            (self.one, self.one, 2),
            # 4:
            (self.one, self.native, 6),
            (self.one, self.npv, NPV_SumExpression([1, self.npv])),
            (self.one, self.param, 7),
            (self.one, self.param_mut, NPV_SumExpression([1, self.param_mut])),
            # 8:
            (self.one, self.var, LinearExpression([1, self.var])),
            (self.one, self.mon_native, LinearExpression([1, self.mon_native])),
            (self.one, self.mon_param, LinearExpression([1, self.mon_param])),
            (self.one, self.mon_npv, LinearExpression([1, self.mon_npv])),
            # 12:
            (self.one, self.linear, LinearExpression(self.linear.args + [1])),
            (self.one, self.sum, SumExpression(self.sum.args + [1])),
            (self.one, self.other, SumExpression([1, self.other])),
            (self.one, self.mutable_l0, 1),
            # 16:
            (self.one, self.mutable_l1, LinearExpression([1, self.l1])),
            (self.one, self.mutable_l2, SumExpression(self.l2.args + [1])),
            (self.one, self.param0, 1),
            (self.one, self.param1, 2),
            # 20:
            (self.one, self.mutable_l3, NPV_SumExpression([1, self.npv])),
        ]
        self._run_cases(tests, operator.add)
        self._run_cases(tests, operator.iadd)

    def test_add_native(self):
        tests = [
            (self.native, self.invalid, NotImplemented),
            (self.native, self.asbinary, LinearExpression([5, self.bin])),
            (self.native, self.zero, 5),
            (self.native, self.one, 6),
            # 4:
            (self.native, self.native, 10),
            (self.native, self.npv, NPV_SumExpression([5, self.npv])),
            (self.native, self.param, 11),
            (self.native, self.param_mut, NPV_SumExpression([5, self.param_mut])),
            # 8:
            (self.native, self.var, LinearExpression([5, self.var])),
            (self.native, self.mon_native, LinearExpression([5, self.mon_native])),
            (self.native, self.mon_param, LinearExpression([5, self.mon_param])),
            (self.native, self.mon_npv, LinearExpression([5, self.mon_npv])),
            # 12:
            (self.native, self.linear, LinearExpression(self.linear.args + [5])),
            (self.native, self.sum, SumExpression(self.sum.args + [5])),
            (self.native, self.other, SumExpression([5, self.other])),
            (self.native, self.mutable_l0, 5),
            # 16:
            (self.native, self.mutable_l1, LinearExpression([5, self.l1])),
            (self.native, self.mutable_l2, SumExpression(self.l2.args + [5])),
            (self.native, self.param0, 5),
            (self.native, self.param1, 6),
            # 20:
            (self.native, self.mutable_l3, NPV_SumExpression([5, self.npv])),
        ]
        self._run_cases(tests, operator.add)
        self._run_cases(tests, operator.iadd)

    def test_add_npv(self):
        tests = [
            (self.npv, self.invalid, NotImplemented),
            (self.npv, self.asbinary, LinearExpression([self.npv, self.bin])),
            (self.npv, self.zero, self.npv),
            (self.npv, self.one, NPV_SumExpression([self.npv, 1])),
            # 4:
            (self.npv, self.native, NPV_SumExpression([self.npv, 5])),
            (self.npv, self.npv, NPV_SumExpression([self.npv, self.npv])),
            (self.npv, self.param, NPV_SumExpression([self.npv, 6])),
            (self.npv, self.param_mut, NPV_SumExpression([self.npv, self.param_mut])),
            # 8:
            (self.npv, self.var, LinearExpression([self.npv, self.var])),
            (self.npv, self.mon_native, LinearExpression([self.npv, self.mon_native])),
            (self.npv, self.mon_param, LinearExpression([self.npv, self.mon_param])),
            (self.npv, self.mon_npv, LinearExpression([self.npv, self.mon_npv])),
            # 12:
            (self.npv, self.linear, LinearExpression(self.linear.args + [self.npv])),
            (self.npv, self.sum, SumExpression(self.sum.args + [self.npv])),
            (self.npv, self.other, SumExpression([self.npv, self.other])),
            (self.npv, self.mutable_l0, self.npv),
            # 16:
            (self.npv, self.mutable_l1, LinearExpression([self.npv, self.l1])),
            (self.npv, self.mutable_l2, SumExpression(self.l2.args + [self.npv])),
            (self.npv, self.param0, self.npv),
            (self.npv, self.param1, NPV_SumExpression([self.npv, 1])),
            # 20:
            (self.npv, self.mutable_l3, NPV_SumExpression([self.npv, self.npv])),
        ]
        self._run_cases(tests, operator.add)
        self._run_cases(tests, operator.iadd)

    def test_add_param(self):
        tests = [
            (self.param, self.invalid, NotImplemented),
            (self.param, self.asbinary, LinearExpression([6, self.bin])),
            (self.param, self.zero, 6),
            (self.param, self.one, 7),
            # 4:
            (self.param, self.native, 11),
            (self.param, self.npv, NPV_SumExpression([6, self.npv])),
            (self.param, self.param, 12),
            (self.param, self.param_mut, NPV_SumExpression([6, self.param_mut])),
            # 8:
            (self.param, self.var, LinearExpression([6, self.var])),
            (self.param, self.mon_native, LinearExpression([6, self.mon_native])),
            (self.param, self.mon_param, LinearExpression([6, self.mon_param])),
            (self.param, self.mon_npv, LinearExpression([6, self.mon_npv])),
            # 12:
            (self.param, self.linear, LinearExpression(self.linear.args + [6])),
            (self.param, self.sum, SumExpression(self.sum.args + [6])),
            (self.param, self.other, SumExpression([6, self.other])),
            (self.param, self.mutable_l0, 6),
            # 16:
            (self.param, self.mutable_l1, LinearExpression([6, self.l1])),
            (self.param, self.mutable_l2, SumExpression(self.l2.args + [6])),
            (self.param, self.param0, 6),
            (self.param, self.param1, 7),
            # 20:
            (self.param, self.mutable_l3, NPV_SumExpression([6, self.npv])),
        ]
        self._run_cases(tests, operator.add)
        self._run_cases(tests, operator.iadd)

    def test_add_param_mut(self):
        tests = [
            (self.param_mut, self.invalid, NotImplemented),
            (
                self.param_mut,
                self.asbinary,
                LinearExpression([self.param_mut, self.bin]),
            ),
            (self.param_mut, self.zero, self.param_mut),
            (self.param_mut, self.one, NPV_SumExpression([self.param_mut, 1])),
            # 4:
            (self.param_mut, self.native, NPV_SumExpression([self.param_mut, 5])),
            (self.param_mut, self.npv, NPV_SumExpression([self.param_mut, self.npv])),
            (self.param_mut, self.param, NPV_SumExpression([self.param_mut, 6])),
            (
                self.param_mut,
                self.param_mut,
                NPV_SumExpression([self.param_mut, self.param_mut]),
            ),
            # 8:
            (self.param_mut, self.var, LinearExpression([self.param_mut, self.var])),
            (
                self.param_mut,
                self.mon_native,
                LinearExpression([self.param_mut, self.mon_native]),
            ),
            (
                self.param_mut,
                self.mon_param,
                LinearExpression([self.param_mut, self.mon_param]),
            ),
            (
                self.param_mut,
                self.mon_npv,
                LinearExpression([self.param_mut, self.mon_npv]),
            ),
            # 12:
            (
                self.param_mut,
                self.linear,
                LinearExpression(self.linear.args + [self.param_mut]),
            ),
            (self.param_mut, self.sum, SumExpression(self.sum.args + [self.param_mut])),
            (self.param_mut, self.other, SumExpression([self.param_mut, self.other])),
            (self.param_mut, self.mutable_l0, self.param_mut),
            # 16:
            (
                self.param_mut,
                self.mutable_l1,
                LinearExpression([self.param_mut, self.l1]),
            ),
            (
                self.param_mut,
                self.mutable_l2,
                SumExpression(self.l2.args + [self.param_mut]),
            ),
            (self.param_mut, self.param0, self.param_mut),
            (self.param_mut, self.param1, NPV_SumExpression([self.param_mut, 1])),
            # 20:
            (
                self.param_mut,
                self.mutable_l3,
                NPV_SumExpression([self.param_mut, self.npv]),
            ),
        ]
        self._run_cases(tests, operator.add)
        self._run_cases(tests, operator.iadd)

    def test_add_var(self):
        tests = [
            (self.var, self.invalid, NotImplemented),
            (self.var, self.asbinary, LinearExpression([self.var, self.bin])),
            (self.var, self.zero, self.var),
            (self.var, self.one, LinearExpression([self.var, 1])),
            # 4:
            (self.var, self.native, LinearExpression([self.var, 5])),
            (self.var, self.npv, LinearExpression([self.var, self.npv])),
            (self.var, self.param, LinearExpression([self.var, 6])),
            (self.var, self.param_mut, LinearExpression([self.var, self.param_mut])),
            # 8:
            (self.var, self.var, LinearExpression([self.var, self.var])),
            (self.var, self.mon_native, LinearExpression([self.var, self.mon_native])),
            (self.var, self.mon_param, LinearExpression([self.var, self.mon_param])),
            (self.var, self.mon_npv, LinearExpression([self.var, self.mon_npv])),
            # 12:
            (self.var, self.linear, LinearExpression(self.linear.args + [self.var])),
            (self.var, self.sum, SumExpression(self.sum.args + [self.var])),
            (self.var, self.other, SumExpression([self.var, self.other])),
            (self.var, self.mutable_l0, self.var),
            # 16:
            (self.var, self.mutable_l1, LinearExpression([self.var, self.l1])),
            (self.var, self.mutable_l2, SumExpression(self.l2.args + [self.var])),
            (self.var, self.param0, self.var),
            (self.var, self.param1, LinearExpression([self.var, 1])),
            # 20:
            (self.var, self.mutable_l3, LinearExpression([self.var, self.npv])),
        ]
        self._run_cases(tests, operator.add)
        self._run_cases(tests, operator.iadd)

    def test_add_mon_native(self):
        tests = [
            (self.mon_native, self.invalid, NotImplemented),
            (
                self.mon_native,
                self.asbinary,
                LinearExpression([self.mon_native, self.bin]),
            ),
            (self.mon_native, self.zero, self.mon_native),
            (self.mon_native, self.one, LinearExpression([self.mon_native, 1])),
            # 4:
            (self.mon_native, self.native, LinearExpression([self.mon_native, 5])),
            (self.mon_native, self.npv, LinearExpression([self.mon_native, self.npv])),
            (self.mon_native, self.param, LinearExpression([self.mon_native, 6])),
            (
                self.mon_native,
                self.param_mut,
                LinearExpression([self.mon_native, self.param_mut]),
            ),
            # 8:
            (self.mon_native, self.var, LinearExpression([self.mon_native, self.var])),
            (
                self.mon_native,
                self.mon_native,
                LinearExpression([self.mon_native, self.mon_native]),
            ),
            (
                self.mon_native,
                self.mon_param,
                LinearExpression([self.mon_native, self.mon_param]),
            ),
            (
                self.mon_native,
                self.mon_npv,
                LinearExpression([self.mon_native, self.mon_npv]),
            ),
            # 12:
            (
                self.mon_native,
                self.linear,
                LinearExpression(self.linear.args + [self.mon_native]),
            ),
            (
                self.mon_native,
                self.sum,
                SumExpression(self.sum.args + [self.mon_native]),
            ),
            (self.mon_native, self.other, SumExpression([self.mon_native, self.other])),
            (self.mon_native, self.mutable_l0, self.mon_native),
            # 16:
            (
                self.mon_native,
                self.mutable_l1,
                LinearExpression([self.mon_native, self.l1]),
            ),
            (
                self.mon_native,
                self.mutable_l2,
                SumExpression(self.l2.args + [self.mon_native]),
            ),
            (self.mon_native, self.param0, self.mon_native),
            (self.mon_native, self.param1, LinearExpression([self.mon_native, 1])),
            # 20:
            (
                self.mon_native,
                self.mutable_l3,
                LinearExpression([self.mon_native, self.npv]),
            ),
        ]
        self._run_cases(tests, operator.add)
        self._run_cases(tests, operator.iadd)

    def test_add_mon_param(self):
        tests = [
            (self.mon_param, self.invalid, NotImplemented),
            (
                self.mon_param,
                self.asbinary,
                LinearExpression([self.mon_param, self.bin]),
            ),
            (self.mon_param, self.zero, self.mon_param),
            (self.mon_param, self.one, LinearExpression([self.mon_param, 1])),
            # 4:
            (self.mon_param, self.native, LinearExpression([self.mon_param, 5])),
            (self.mon_param, self.npv, LinearExpression([self.mon_param, self.npv])),
            (self.mon_param, self.param, LinearExpression([self.mon_param, 6])),
            (
                self.mon_param,
                self.param_mut,
                LinearExpression([self.mon_param, self.param_mut]),
            ),
            # 8:
            (self.mon_param, self.var, LinearExpression([self.mon_param, self.var])),
            (
                self.mon_param,
                self.mon_native,
                LinearExpression([self.mon_param, self.mon_native]),
            ),
            (
                self.mon_param,
                self.mon_param,
                LinearExpression([self.mon_param, self.mon_param]),
            ),
            (
                self.mon_param,
                self.mon_npv,
                LinearExpression([self.mon_param, self.mon_npv]),
            ),
            # 12:
            (
                self.mon_param,
                self.linear,
                LinearExpression(self.linear.args + [self.mon_param]),
            ),
            (self.mon_param, self.sum, SumExpression(self.sum.args + [self.mon_param])),
            (self.mon_param, self.other, SumExpression([self.mon_param, self.other])),
            (self.mon_param, self.mutable_l0, self.mon_param),
            # 16:
            (
                self.mon_param,
                self.mutable_l1,
                LinearExpression([self.mon_param, self.l1]),
            ),
            (
                self.mon_param,
                self.mutable_l2,
                SumExpression(self.l2.args + [self.mon_param]),
            ),
            (self.mon_param, self.param0, self.mon_param),
            (self.mon_param, self.param1, LinearExpression([self.mon_param, 1])),
            # 20:
            (
                self.mon_param,
                self.mutable_l3,
                LinearExpression([self.mon_param, self.npv]),
            ),
        ]
        self._run_cases(tests, operator.add)
        self._run_cases(tests, operator.iadd)

    def test_add_mon_npv(self):
        tests = [
            (self.mon_npv, self.invalid, NotImplemented),
            (self.mon_npv, self.asbinary, LinearExpression([self.mon_npv, self.bin])),
            (self.mon_npv, self.zero, self.mon_npv),
            (self.mon_npv, self.one, LinearExpression([self.mon_npv, 1])),
            # 4:
            (self.mon_npv, self.native, LinearExpression([self.mon_npv, 5])),
            (self.mon_npv, self.npv, LinearExpression([self.mon_npv, self.npv])),
            (self.mon_npv, self.param, LinearExpression([self.mon_npv, 6])),
            (
                self.mon_npv,
                self.param_mut,
                LinearExpression([self.mon_npv, self.param_mut]),
            ),
            # 8:
            (self.mon_npv, self.var, LinearExpression([self.mon_npv, self.var])),
            (
                self.mon_npv,
                self.mon_native,
                LinearExpression([self.mon_npv, self.mon_native]),
            ),
            (
                self.mon_npv,
                self.mon_param,
                LinearExpression([self.mon_npv, self.mon_param]),
            ),
            (
                self.mon_npv,
                self.mon_npv,
                LinearExpression([self.mon_npv, self.mon_npv]),
            ),
            # 12:
            (
                self.mon_npv,
                self.linear,
                LinearExpression(self.linear.args + [self.mon_npv]),
            ),
            (self.mon_npv, self.sum, SumExpression(self.sum.args + [self.mon_npv])),
            (self.mon_npv, self.other, SumExpression([self.mon_npv, self.other])),
            (self.mon_npv, self.mutable_l0, self.mon_npv),
            # 16:
            (self.mon_npv, self.mutable_l1, LinearExpression([self.mon_npv, self.l1])),
            (
                self.mon_npv,
                self.mutable_l2,
                SumExpression(self.l2.args + [self.mon_npv]),
            ),
            (self.mon_npv, self.param0, self.mon_npv),
            (self.mon_npv, self.param1, LinearExpression([self.mon_npv, 1])),
            # 20:
            (self.mon_npv, self.mutable_l3, LinearExpression([self.mon_npv, self.npv])),
        ]
        self._run_cases(tests, operator.add)
        self._run_cases(tests, operator.iadd)

    def test_add_linear(self):
        tests = [
            (self.linear, self.invalid, NotImplemented),
            (
                self.linear,
                self.asbinary,
                LinearExpression(self.linear.args + [self.bin]),
            ),
            (self.linear, self.zero, self.linear),
            (self.linear, self.one, LinearExpression(self.linear.args + [1])),
            # 4:
            (self.linear, self.native, LinearExpression(self.linear.args + [5])),
            (self.linear, self.npv, LinearExpression(self.linear.args + [self.npv])),
            (self.linear, self.param, LinearExpression(self.linear.args + [6])),
            (
                self.linear,
                self.param_mut,
                LinearExpression(self.linear.args + [self.param_mut]),
            ),
            # 8:
            (self.linear, self.var, LinearExpression(self.linear.args + [self.var])),
            (
                self.linear,
                self.mon_native,
                LinearExpression(self.linear.args + [self.mon_native]),
            ),
            (
                self.linear,
                self.mon_param,
                LinearExpression(self.linear.args + [self.mon_param]),
            ),
            (
                self.linear,
                self.mon_npv,
                LinearExpression(self.linear.args + [self.mon_npv]),
            ),
            # 12:
            (
                self.linear,
                self.linear,
                LinearExpression(self.linear.args + self.linear.args),
            ),
            (self.linear, self.sum, SumExpression(self.sum.args + [self.linear])),
            (self.linear, self.other, SumExpression([self.linear, self.other])),
            (self.linear, self.mutable_l0, self.linear),
            # 16:
            (
                self.linear,
                self.mutable_l1,
                LinearExpression(self.linear.args + [self.l1]),
            ),
            (self.linear, self.mutable_l2, SumExpression(self.l2.args + [self.linear])),
            (self.linear, self.param0, self.linear),
            (self.linear, self.param1, LinearExpression(self.linear.args + [1])),
            # 20:
            (
                self.linear,
                self.mutable_l3,
                LinearExpression(self.linear.args + [self.npv]),
            ),
        ]
        self._run_cases(tests, operator.add)
        self._run_cases(tests, operator.iadd)

    def test_add_sum(self):
        tests = [
            (self.sum, self.invalid, NotImplemented),
            (self.sum, self.asbinary, SumExpression(self.sum.args + [self.bin])),
            (self.sum, self.zero, self.sum),
            (self.sum, self.one, SumExpression(self.sum.args + [1])),
            # 4:
            (self.sum, self.native, SumExpression(self.sum.args + [5])),
            (self.sum, self.npv, SumExpression(self.sum.args + [self.npv])),
            (self.sum, self.param, SumExpression(self.sum.args + [6])),
            (self.sum, self.param_mut, SumExpression(self.sum.args + [self.param_mut])),
            # 8:
            (self.sum, self.var, SumExpression(self.sum.args + [self.var])),
            (
                self.sum,
                self.mon_native,
                SumExpression(self.sum.args + [self.mon_native]),
            ),
            (self.sum, self.mon_param, SumExpression(self.sum.args + [self.mon_param])),
            (self.sum, self.mon_npv, SumExpression(self.sum.args + [self.mon_npv])),
            # 12:
            (self.sum, self.linear, SumExpression(self.sum.args + [self.linear])),
            (self.sum, self.sum, SumExpression(self.sum.args + self.sum.args)),
            (self.sum, self.other, SumExpression(self.sum.args + [self.other])),
            (self.sum, self.mutable_l0, self.sum),
            # 16:
            (self.sum, self.mutable_l1, SumExpression(self.sum.args + [self.l1])),
            (self.sum, self.mutable_l2, SumExpression(self.sum.args + self.l2.args)),
            (self.sum, self.param0, self.sum),
            (self.sum, self.param1, SumExpression(self.sum.args + [1])),
            # 20:
            (self.sum, self.mutable_l3, SumExpression(self.sum.args + [self.npv])),
        ]
        self._run_cases(tests, operator.add)
        self._run_cases(tests, operator.iadd)

    def test_add_other(self):
        tests = [
            (self.other, self.invalid, NotImplemented),
            (self.other, self.asbinary, SumExpression([self.other, self.bin])),
            (self.other, self.zero, self.other),
            (self.other, self.one, SumExpression([self.other, 1])),
            # 4:
            (self.other, self.native, SumExpression([self.other, 5])),
            (self.other, self.npv, SumExpression([self.other, self.npv])),
            (self.other, self.param, SumExpression([self.other, 6])),
            (self.other, self.param_mut, SumExpression([self.other, self.param_mut])),
            # 8:
            (self.other, self.var, SumExpression([self.other, self.var])),
            (self.other, self.mon_native, SumExpression([self.other, self.mon_native])),
            (self.other, self.mon_param, SumExpression([self.other, self.mon_param])),
            (self.other, self.mon_npv, SumExpression([self.other, self.mon_npv])),
            # 12:
            (self.other, self.linear, SumExpression([self.other, self.linear])),
            (self.other, self.sum, SumExpression(self.sum.args + [self.other])),
            (self.other, self.other, SumExpression([self.other, self.other])),
            (self.other, self.mutable_l0, self.other),
            # 16:
            (self.other, self.mutable_l1, SumExpression([self.other, self.mon_npv])),
            (self.other, self.mutable_l2, SumExpression(self.l2.args + [self.other])),
            (self.other, self.param0, self.other),
            (self.other, self.param1, SumExpression([self.other, 1])),
            # 20:
            (self.other, self.mutable_l3, SumExpression([self.other, self.npv])),
        ]
        self._run_cases(tests, operator.add)
        self._run_cases(tests, operator.iadd)

    def test_add_mutable_l0(self):
        tests = [
            (self.mutable_l0, self.invalid, NotImplemented),
            (self.mutable_l0, self.asbinary, self.bin),
            (self.mutable_l0, self.zero, 0),
            (self.mutable_l0, self.one, 1),
            # 4:
            (self.mutable_l0, self.native, 5),
            (self.mutable_l0, self.npv, self.npv),
            (self.mutable_l0, self.param, 6),
            (self.mutable_l0, self.param_mut, self.param_mut),
            # 8:
            (self.mutable_l0, self.var, self.var),
            (self.mutable_l0, self.mon_native, self.mon_native),
            (self.mutable_l0, self.mon_param, self.mon_param),
            (self.mutable_l0, self.mon_npv, self.mon_npv),
            # 12:
            (self.mutable_l0, self.linear, self.linear),
            (self.mutable_l0, self.sum, self.sum),
            (self.mutable_l0, self.other, self.other),
            (self.mutable_l0, self.mutable_l0, 0),
            # 16:
            (self.mutable_l0, self.mutable_l1, self.mon_npv),
            (self.mutable_l0, self.mutable_l2, self.l2),
            (self.mutable_l0, self.param0, 0),
            (self.mutable_l0, self.param1, 1),
            # 20:
            (self.mutable_l0, self.mutable_l3, self.npv),
        ]
        self._run_cases(tests, operator.add)
        # Mutable iadd handled by separate tests
        # self._run_cases(tests, operator.iadd)

    def test_add_mutable_l1(self):
        tests = [
            (self.mutable_l1, self.invalid, NotImplemented),
            (self.mutable_l1, self.asbinary, LinearExpression([self.l1, self.bin])),
            (self.mutable_l1, self.zero, self.mon_npv),
            (self.mutable_l1, self.one, LinearExpression([self.l1, 1])),
            # 4:
            (self.mutable_l1, self.native, LinearExpression([self.l1, 5])),
            (self.mutable_l1, self.npv, LinearExpression([self.l1, self.npv])),
            (self.mutable_l1, self.param, LinearExpression([self.l1, 6])),
            (
                self.mutable_l1,
                self.param_mut,
                LinearExpression([self.l1, self.param_mut]),
            ),
            # 8:
            (self.mutable_l1, self.var, LinearExpression([self.l1, self.var])),
            (
                self.mutable_l1,
                self.mon_native,
                LinearExpression([self.l1, self.mon_native]),
            ),
            (
                self.mutable_l1,
                self.mon_param,
                LinearExpression([self.l1, self.mon_param]),
            ),
            (self.mutable_l1, self.mon_npv, LinearExpression([self.l1, self.mon_npv])),
            # 12:
            (
                self.mutable_l1,
                self.linear,
                LinearExpression(self.linear.args + [self.l1]),
            ),
            (self.mutable_l1, self.sum, SumExpression(self.sum.args + [self.l1])),
            (self.mutable_l1, self.other, SumExpression([self.l1, self.other])),
            (self.mutable_l1, self.mutable_l0, self.mon_npv),
            # 16:
            (self.mutable_l1, self.mutable_l1, LinearExpression([self.l1, self.l1])),
            (self.mutable_l1, self.mutable_l2, SumExpression(self.l2.args + [self.l1])),
            (self.mutable_l1, self.param0, self.mon_npv),
            (self.mutable_l1, self.param1, LinearExpression([self.l1, 1])),
            # 20:
            (self.mutable_l1, self.mutable_l3, LinearExpression([self.l1, self.npv])),
        ]
        self._run_cases(tests, operator.add)
        # Mutable iadd handled by separate tests
        # self._run_cases(tests, operator.iadd)

    def test_add_mutable_l2(self):
        tests = [
            (self.mutable_l2, self.invalid, NotImplemented),
            (self.mutable_l2, self.asbinary, SumExpression(self.l2.args + [self.bin])),
            (self.mutable_l2, self.zero, self.l2),
            (self.mutable_l2, self.one, SumExpression(self.l2.args + [1])),
            # 4:
            (self.mutable_l2, self.native, SumExpression(self.l2.args + [5])),
            (self.mutable_l2, self.npv, SumExpression(self.l2.args + [self.npv])),
            (self.mutable_l2, self.param, SumExpression(self.l2.args + [6])),
            (
                self.mutable_l2,
                self.param_mut,
                SumExpression(self.l2.args + [self.param_mut]),
            ),
            # 8:
            (self.mutable_l2, self.var, SumExpression(self.l2.args + [self.var])),
            (
                self.mutable_l2,
                self.mon_native,
                SumExpression(self.l2.args + [self.mon_native]),
            ),
            (
                self.mutable_l2,
                self.mon_param,
                SumExpression(self.l2.args + [self.mon_param]),
            ),
            (
                self.mutable_l2,
                self.mon_npv,
                SumExpression(self.l2.args + [self.mon_npv]),
            ),
            # 12:
            (self.mutable_l2, self.linear, SumExpression(self.l2.args + [self.linear])),
            (self.mutable_l2, self.sum, SumExpression(self.l2.args + self.sum.args)),
            (self.mutable_l2, self.other, SumExpression(self.l2.args + [self.other])),
            (self.mutable_l2, self.mutable_l0, self.l2),
            # 16:
            (self.mutable_l2, self.mutable_l1, SumExpression(self.l2.args + [self.l1])),
            (
                self.mutable_l2,
                self.mutable_l2,
                SumExpression(self.l2.args + self.l2.args),
            ),
            (self.mutable_l2, self.param0, self.l2),
            (self.mutable_l2, self.param1, SumExpression(self.l2.args + [1])),
            # 20:
            (
                self.mutable_l2,
                self.mutable_l3,
                SumExpression(self.l2.args + [self.npv]),
            ),
        ]
        self._run_cases(tests, operator.add)
        # Mutable iadd handled by separate tests
        # self._run_cases(tests, operator.iadd)

    def test_add_param0(self):
        tests = [
            (self.param0, self.invalid, NotImplemented),
            (self.param0, self.asbinary, self.bin),
            (self.param0, self.zero, 0),
            (self.param0, self.one, 1),
            # 4:
            (self.param0, self.native, 5),
            (self.param0, self.npv, self.npv),
            (self.param0, self.param, 6),
            (self.param0, self.param_mut, self.param_mut),
            # 8:
            (self.param0, self.var, self.var),
            (self.param0, self.mon_native, self.mon_native),
            (self.param0, self.mon_param, self.mon_param),
            (self.param0, self.mon_npv, self.mon_npv),
            # 12:
            (self.param0, self.linear, self.linear),
            (self.param0, self.sum, self.sum),
            (self.param0, self.other, self.other),
            (self.param0, self.mutable_l0, 0),
            # 16:
            (self.param0, self.mutable_l1, self.mon_npv),
            (self.param0, self.mutable_l2, self.l2),
            (self.param0, self.param0, 0),
            (self.param0, self.param1, 1),
            # 20:
            (self.param0, self.mutable_l3, self.npv),
        ]
        self._run_cases(tests, operator.add)
        self._run_cases(tests, operator.iadd)

    def test_add_param1(self):
        tests = [
            (self.param1, self.invalid, NotImplemented),
            (self.param1, self.asbinary, LinearExpression([1, self.bin])),
            (self.param1, self.zero, 1),
            (self.param1, self.one, 2),
            # 4:
            (self.param1, self.native, 6),
            (self.param1, self.npv, NPV_SumExpression([1, self.npv])),
            (self.param1, self.param, 7),
            (self.param1, self.param_mut, NPV_SumExpression([1, self.param_mut])),
            # 8:
            (self.param1, self.var, LinearExpression([1, self.var])),
            (self.param1, self.mon_native, LinearExpression([1, self.mon_native])),
            (self.param1, self.mon_param, LinearExpression([1, self.mon_param])),
            (self.param1, self.mon_npv, LinearExpression([1, self.mon_npv])),
            # 12:
            (self.param1, self.linear, LinearExpression(self.linear.args + [1])),
            (self.param1, self.sum, SumExpression(self.sum.args + [1])),
            (self.param1, self.other, SumExpression([1, self.other])),
            (self.param1, self.mutable_l0, 1),
            # 16:
            (self.param1, self.mutable_l1, LinearExpression([1, self.l1])),
            (self.param1, self.mutable_l2, SumExpression(self.l2.args + [1])),
            (self.param1, self.param0, 1),
            (self.param1, self.param1, 2),
            # 20:
            (self.param1, self.mutable_l3, NPV_SumExpression([1, self.npv])),
        ]
        self._run_cases(tests, operator.add)
        self._run_cases(tests, operator.iadd)

    def test_add_mutable_l3(self):
        tests = [
            (self.mutable_l3, self.invalid, NotImplemented),
            (self.mutable_l3, self.asbinary, LinearExpression([self.l3, self.bin])),
            (self.mutable_l3, self.zero, self.npv),
            (self.mutable_l3, self.one, NPV_SumExpression([self.l3, 1])),
            # 4:
            (self.mutable_l3, self.native, NPV_SumExpression([self.l3, 5])),
            (self.mutable_l3, self.npv, NPV_SumExpression([self.l3, self.npv])),
            (self.mutable_l3, self.param, NPV_SumExpression([self.l3, 6])),
            (
                self.mutable_l3,
                self.param_mut,
                NPV_SumExpression([self.l3, self.param_mut]),
            ),
            # 8:
            (self.mutable_l3, self.var, LinearExpression([self.l3, self.var])),
            (
                self.mutable_l3,
                self.mon_native,
                LinearExpression([self.l3, self.mon_native]),
            ),
            (
                self.mutable_l3,
                self.mon_param,
                LinearExpression([self.l3, self.mon_param]),
            ),
            (self.mutable_l3, self.mon_npv, LinearExpression([self.l3, self.mon_npv])),
            # 12:
            (
                self.mutable_l3,
                self.linear,
                LinearExpression(self.linear.args + [self.l3]),
            ),
            (self.mutable_l3, self.sum, SumExpression(self.sum.args + [self.l3])),
            (self.mutable_l3, self.other, SumExpression([self.l3, self.other])),
            (self.mutable_l3, self.mutable_l0, self.npv),
            # 16:
            (self.mutable_l3, self.mutable_l1, LinearExpression([self.l3, self.l1])),
            (self.mutable_l3, self.mutable_l2, SumExpression(self.l2.args + [self.l3])),
            (self.mutable_l3, self.param0, self.npv),
            (self.mutable_l3, self.param1, NPV_SumExpression([self.l3, 1])),
            # 20:
            (self.mutable_l3, self.mutable_l3, NPV_SumExpression([self.l3, self.npv])),
        ]
        self._run_cases(tests, operator.add)
        # Mutable iadd handled by separate tests
        # self._run_cases(tests, operator.iadd)

    #
    #
    # SUBTRACTION
    #
    #

    def test_sub_invalid(self):
        tests = [
            (self.invalid, self.invalid, NotImplemented),
            (self.invalid, self.asbinary, NotImplemented),
            (self.invalid, self.zero, NotImplemented),
            (self.invalid, self.one, NotImplemented),
            # 4:
            (self.invalid, self.native, NotImplemented),
            (self.invalid, self.npv, NotImplemented),
            (self.invalid, self.param, NotImplemented),
            (self.invalid, self.param_mut, NotImplemented),
            # 8:
            (self.invalid, self.var, NotImplemented),
            (self.invalid, self.mon_native, NotImplemented),
            (self.invalid, self.mon_param, NotImplemented),
            (self.invalid, self.mon_npv, NotImplemented),
            # 12:
            (self.invalid, self.linear, NotImplemented),
            (self.invalid, self.sum, NotImplemented),
            (self.invalid, self.other, NotImplemented),
            (self.invalid, self.mutable_l0, NotImplemented),
            # 16:
            (self.invalid, self.mutable_l1, NotImplemented),
            (self.invalid, self.mutable_l2, NotImplemented),
            (self.invalid, self.param0, NotImplemented),
            (self.invalid, self.param1, NotImplemented),
            # 20:
            (self.invalid, self.mutable_l3, NotImplemented),
        ]
        self._run_cases(tests, operator.sub)
        self._run_cases(tests, operator.isub)

    def test_sub_asbinary(self):
        tests = [
            (self.asbinary, self.invalid, NotImplemented),
            # BooleanVar objects do not support addition
            (self.asbinary, self.asbinary, NotImplemented),
            (self.asbinary, self.zero, self.bin),
            (self.asbinary, self.one, LinearExpression([self.bin, -1])),
            # 4:
            (self.asbinary, self.native, LinearExpression([self.bin, -5])),
            (self.asbinary, self.npv, LinearExpression([self.bin, self.minus_npv])),
            (self.asbinary, self.param, LinearExpression([self.bin, -6])),
            (
                self.asbinary,
                self.param_mut,
                LinearExpression([self.bin, self.minus_param_mut]),
            ),
            # 8:
            (self.asbinary, self.var, LinearExpression([self.bin, self.minus_var])),
            (
                self.asbinary,
                self.mon_native,
                LinearExpression([self.bin, self.minus_mon_native]),
            ),
            (
                self.asbinary,
                self.mon_param,
                LinearExpression([self.bin, self.minus_mon_param]),
            ),
            (
                self.asbinary,
                self.mon_npv,
                LinearExpression([self.bin, self.minus_mon_npv]),
            ),
            # 12:
            (self.asbinary, self.linear, SumExpression([self.bin, self.minus_linear])),
            (self.asbinary, self.sum, SumExpression([self.bin, self.minus_sum])),
            (self.asbinary, self.other, SumExpression([self.bin, self.minus_other])),
            (self.asbinary, self.mutable_l0, self.bin),
            # 16:
            (
                self.asbinary,
                self.mutable_l1,
                LinearExpression([self.bin, self.minus_mon_npv]),
            ),
            (self.asbinary, self.mutable_l2, SumExpression([self.bin, self.minus_l2])),
            (self.asbinary, self.param0, self.bin),
            (self.asbinary, self.param1, LinearExpression([self.bin, -1])),
            # 20:
            (
                self.asbinary,
                self.mutable_l3,
                LinearExpression([self.bin, self.minus_npv]),
            ),
        ]
        self._run_cases(tests, operator.sub)
        self._run_cases(tests, operator.isub)

    def test_sub_zero(self):
        tests = [
            (self.zero, self.invalid, NotImplemented),
            (self.zero, self.asbinary, self.minus_bin),
            (self.zero, self.zero, 0),
            (self.zero, self.one, -1),
            # 4:
            (self.zero, self.native, -5),
            (self.zero, self.npv, self.minus_npv),
            (self.zero, self.param, -6),
            (self.zero, self.param_mut, self.minus_param_mut),
            # 8:
            (self.zero, self.var, self.minus_var),
            (self.zero, self.mon_native, self.minus_mon_native),
            (self.zero, self.mon_param, self.minus_mon_param),
            (self.zero, self.mon_npv, self.minus_mon_npv),
            # 12:
            (self.zero, self.linear, self.minus_linear),
            (self.zero, self.sum, self.minus_sum),
            (self.zero, self.other, self.minus_other),
            (self.zero, self.mutable_l0, 0),
            # 16:
            (self.zero, self.mutable_l1, self.minus_mon_npv),
            (self.zero, self.mutable_l2, self.minus_l2),
            (self.zero, self.param0, 0),
            (self.zero, self.param1, -1),
            # 20:
            (self.zero, self.mutable_l3, self.minus_npv),
        ]
        self._run_cases(tests, operator.sub)
        self._run_cases(tests, operator.isub)

    def test_sub_one(self):
        tests = [
            (self.one, self.invalid, NotImplemented),
            (self.one, self.asbinary, LinearExpression([1, self.minus_bin])),
            (self.one, self.zero, 1),
            (self.one, self.one, 0),
            # 4:
            (self.one, self.native, -4),
            (self.one, self.npv, NPV_SumExpression([1, self.minus_npv])),
            (self.one, self.param, -5),
            (self.one, self.param_mut, NPV_SumExpression([1, self.minus_param_mut])),
            # 8:
            (self.one, self.var, LinearExpression([1, self.minus_var])),
            (self.one, self.mon_native, LinearExpression([1, self.minus_mon_native])),
            (self.one, self.mon_param, LinearExpression([1, self.minus_mon_param])),
            (self.one, self.mon_npv, LinearExpression([1, self.minus_mon_npv])),
            # 12:
            (self.one, self.linear, SumExpression([1, self.minus_linear])),
            (self.one, self.sum, SumExpression([1, self.minus_sum])),
            (self.one, self.other, SumExpression([1, self.minus_other])),
            (self.one, self.mutable_l0, 1),
            # 16:
            (self.one, self.mutable_l1, LinearExpression([1, self.minus_mon_npv])),
            (self.one, self.mutable_l2, SumExpression([1, self.minus_l2])),
            (self.one, self.param0, 1),
            (self.one, self.param1, 0),
            # 20:
            (self.one, self.mutable_l3, NPV_SumExpression([1, self.minus_npv])),
        ]
        self._run_cases(tests, operator.sub)
        self._run_cases(tests, operator.isub)

    def test_sub_native(self):
        tests = [
            (self.native, self.invalid, NotImplemented),
            (self.native, self.asbinary, LinearExpression([5, self.minus_bin])),
            (self.native, self.zero, 5),
            (self.native, self.one, 4),
            # 4:
            (self.native, self.native, 0),
            (self.native, self.npv, NPV_SumExpression([5, self.minus_npv])),
            (self.native, self.param, -1),
            (self.native, self.param_mut, NPV_SumExpression([5, self.minus_param_mut])),
            # 8:
            (self.native, self.var, LinearExpression([5, self.minus_var])),
            (
                self.native,
                self.mon_native,
                LinearExpression([5, self.minus_mon_native]),
            ),
            (self.native, self.mon_param, LinearExpression([5, self.minus_mon_param])),
            (self.native, self.mon_npv, LinearExpression([5, self.minus_mon_npv])),
            # 12:
            (self.native, self.linear, SumExpression([5, self.minus_linear])),
            (self.native, self.sum, SumExpression([5, self.minus_sum])),
            (self.native, self.other, SumExpression([5, self.minus_other])),
            (self.native, self.mutable_l0, 5),
            # 16:
            (self.native, self.mutable_l1, LinearExpression([5, self.minus_mon_npv])),
            (self.native, self.mutable_l2, SumExpression([5, self.minus_l2])),
            (self.native, self.param0, 5),
            (self.native, self.param1, 4),
            # 20:
            (self.native, self.mutable_l3, NPV_SumExpression([5, self.minus_npv])),
        ]
        self._run_cases(tests, operator.sub)
        self._run_cases(tests, operator.isub)

    def test_sub_npv(self):
        tests = [
            (self.npv, self.invalid, NotImplemented),
            (self.npv, self.asbinary, LinearExpression([self.npv, self.minus_bin])),
            (self.npv, self.zero, self.npv),
            (self.npv, self.one, NPV_SumExpression([self.npv, -1])),
            # 4:
            (self.npv, self.native, NPV_SumExpression([self.npv, -5])),
            (self.npv, self.npv, NPV_SumExpression([self.npv, self.minus_npv])),
            (self.npv, self.param, NPV_SumExpression([self.npv, -6])),
            (
                self.npv,
                self.param_mut,
                NPV_SumExpression([self.npv, self.minus_param_mut]),
            ),
            # 8:
            (self.npv, self.var, LinearExpression([self.npv, self.minus_var])),
            (
                self.npv,
                self.mon_native,
                LinearExpression([self.npv, self.minus_mon_native]),
            ),
            (
                self.npv,
                self.mon_param,
                LinearExpression([self.npv, self.minus_mon_param]),
            ),
            (self.npv, self.mon_npv, LinearExpression([self.npv, self.minus_mon_npv])),
            # 12:
            (self.npv, self.linear, SumExpression([self.npv, self.minus_linear])),
            (self.npv, self.sum, SumExpression([self.npv, self.minus_sum])),
            (self.npv, self.other, SumExpression([self.npv, self.minus_other])),
            (self.npv, self.mutable_l0, self.npv),
            # 16:
            (
                self.npv,
                self.mutable_l1,
                LinearExpression([self.npv, self.minus_mon_npv]),
            ),
            (self.npv, self.mutable_l2, SumExpression([self.npv, self.minus_l2])),
            (self.npv, self.param0, self.npv),
            (self.npv, self.param1, NPV_SumExpression([self.npv, -1])),
            # 20:
            (self.npv, self.mutable_l3, NPV_SumExpression([self.npv, self.minus_npv])),
        ]
        self._run_cases(tests, operator.sub)
        self._run_cases(tests, operator.isub)

    def test_sub_param(self):
        tests = [
            (self.param, self.invalid, NotImplemented),
            (self.param, self.asbinary, LinearExpression([6, self.minus_bin])),
            (self.param, self.zero, 6),
            (self.param, self.one, 5),
            # 4:
            (self.param, self.native, 1),
            (self.param, self.npv, NPV_SumExpression([6, self.minus_npv])),
            (self.param, self.param, 0),
            (self.param, self.param_mut, NPV_SumExpression([6, self.minus_param_mut])),
            # 8:
            (self.param, self.var, LinearExpression([6, self.minus_var])),
            (self.param, self.mon_native, LinearExpression([6, self.minus_mon_native])),
            (self.param, self.mon_param, LinearExpression([6, self.minus_mon_param])),
            (self.param, self.mon_npv, LinearExpression([6, self.minus_mon_npv])),
            # 12:
            (self.param, self.linear, SumExpression([6, self.minus_linear])),
            (self.param, self.sum, SumExpression([6, self.minus_sum])),
            (self.param, self.other, SumExpression([6, self.minus_other])),
            (self.param, self.mutable_l0, 6),
            # 16:
            (self.param, self.mutable_l1, LinearExpression([6, self.minus_mon_npv])),
            (self.param, self.mutable_l2, SumExpression([6, self.minus_l2])),
            (self.param, self.param0, 6),
            (self.param, self.param1, 5),
            # 20:
            (self.param, self.mutable_l3, NPV_SumExpression([6, self.minus_npv])),
        ]
        self._run_cases(tests, operator.sub)
        self._run_cases(tests, operator.isub)

    def test_sub_param_mut(self):
        tests = [
            (self.param_mut, self.invalid, NotImplemented),
            (
                self.param_mut,
                self.asbinary,
                LinearExpression([self.param_mut, self.minus_bin]),
            ),
            (self.param_mut, self.zero, self.param_mut),
            (self.param_mut, self.one, NPV_SumExpression([self.param_mut, -1])),
            # 4:
            (self.param_mut, self.native, NPV_SumExpression([self.param_mut, -5])),
            (
                self.param_mut,
                self.npv,
                NPV_SumExpression([self.param_mut, self.minus_npv]),
            ),
            (self.param_mut, self.param, NPV_SumExpression([self.param_mut, -6])),
            (
                self.param_mut,
                self.param_mut,
                NPV_SumExpression([self.param_mut, self.minus_param_mut]),
            ),
            # 8:
            (
                self.param_mut,
                self.var,
                LinearExpression([self.param_mut, self.minus_var]),
            ),
            (
                self.param_mut,
                self.mon_native,
                LinearExpression([self.param_mut, self.minus_mon_native]),
            ),
            (
                self.param_mut,
                self.mon_param,
                LinearExpression([self.param_mut, self.minus_mon_param]),
            ),
            (
                self.param_mut,
                self.mon_npv,
                LinearExpression([self.param_mut, self.minus_mon_npv]),
            ),
            # 12:
            (
                self.param_mut,
                self.linear,
                SumExpression([self.param_mut, self.minus_linear]),
            ),
            (self.param_mut, self.sum, SumExpression([self.param_mut, self.minus_sum])),
            (
                self.param_mut,
                self.other,
                SumExpression([self.param_mut, self.minus_other]),
            ),
            (self.param_mut, self.mutable_l0, self.param_mut),
            # 16:
            (
                self.param_mut,
                self.mutable_l1,
                LinearExpression([self.param_mut, self.minus_mon_npv]),
            ),
            (
                self.param_mut,
                self.mutable_l2,
                SumExpression([self.param_mut, self.minus_l2]),
            ),
            (self.param_mut, self.param0, self.param_mut),
            (self.param_mut, self.param1, NPV_SumExpression([self.param_mut, -1])),
            # 20:
            (
                self.param_mut,
                self.mutable_l3,
                NPV_SumExpression([self.param_mut, self.minus_npv]),
            ),
        ]
        self._run_cases(tests, operator.sub)
        self._run_cases(tests, operator.isub)

    def test_sub_var(self):
        tests = [
            (self.var, self.invalid, NotImplemented),
            (self.var, self.asbinary, LinearExpression([self.var, self.minus_bin])),
            (self.var, self.zero, self.var),
            (self.var, self.one, LinearExpression([self.var, -1])),
            # 4:
            (self.var, self.native, LinearExpression([self.var, -5])),
            (self.var, self.npv, LinearExpression([self.var, self.minus_npv])),
            (self.var, self.param, LinearExpression([self.var, -6])),
            (
                self.var,
                self.param_mut,
                LinearExpression([self.var, self.minus_param_mut]),
            ),
            # 8:
            (self.var, self.var, LinearExpression([self.var, self.minus_var])),
            (
                self.var,
                self.mon_native,
                LinearExpression([self.var, self.minus_mon_native]),
            ),
            (
                self.var,
                self.mon_param,
                LinearExpression([self.var, self.minus_mon_param]),
            ),
            (self.var, self.mon_npv, LinearExpression([self.var, self.minus_mon_npv])),
            # 12:
            (
                self.var,
                self.linear,
                SumExpression([self.var, NegationExpression((self.linear,))]),
            ),
            (self.var, self.sum, SumExpression([self.var, self.minus_sum])),
            (self.var, self.other, SumExpression([self.var, self.minus_other])),
            (self.var, self.mutable_l0, self.var),
            # 16:
            (
                self.var,
                self.mutable_l1,
                LinearExpression([self.var, self.minus_mon_npv]),
            ),
            (self.var, self.mutable_l2, SumExpression([self.var, self.minus_l2])),
            (self.var, self.param0, self.var),
            (self.var, self.param1, LinearExpression([self.var, -1])),
            # 20:
            (self.var, self.mutable_l3, LinearExpression([self.var, self.minus_npv])),
        ]
        self._run_cases(tests, operator.sub)
        self._run_cases(tests, operator.isub)

    def test_sub_mon_native(self):
        tests = [
            (self.mon_native, self.invalid, NotImplemented),
            (
                self.mon_native,
                self.asbinary,
                LinearExpression([self.mon_native, self.minus_bin]),
            ),
            (self.mon_native, self.zero, self.mon_native),
            (self.mon_native, self.one, LinearExpression([self.mon_native, -1])),
            # 4:
            (self.mon_native, self.native, LinearExpression([self.mon_native, -5])),
            (
                self.mon_native,
                self.npv,
                LinearExpression([self.mon_native, self.minus_npv]),
            ),
            (self.mon_native, self.param, LinearExpression([self.mon_native, -6])),
            (
                self.mon_native,
                self.param_mut,
                LinearExpression([self.mon_native, self.minus_param_mut]),
            ),
            # 8:
            (
                self.mon_native,
                self.var,
                LinearExpression([self.mon_native, self.minus_var]),
            ),
            (
                self.mon_native,
                self.mon_native,
                LinearExpression([self.mon_native, self.minus_mon_native]),
            ),
            (
                self.mon_native,
                self.mon_param,
                LinearExpression([self.mon_native, self.minus_mon_param]),
            ),
            (
                self.mon_native,
                self.mon_npv,
                LinearExpression([self.mon_native, self.minus_mon_npv]),
            ),
            # 12:
            (
                self.mon_native,
                self.linear,
                SumExpression([self.mon_native, self.minus_linear]),
            ),
            (
                self.mon_native,
                self.sum,
                SumExpression([self.mon_native, self.minus_sum]),
            ),
            (
                self.mon_native,
                self.other,
                SumExpression([self.mon_native, self.minus_other]),
            ),
            (self.mon_native, self.mutable_l0, self.mon_native),
            # 16:
            (
                self.mon_native,
                self.mutable_l1,
                LinearExpression([self.mon_native, self.minus_mon_npv]),
            ),
            (
                self.mon_native,
                self.mutable_l2,
                SumExpression([self.mon_native, self.minus_l2]),
            ),
            (self.mon_native, self.param0, self.mon_native),
            (self.mon_native, self.param1, LinearExpression([self.mon_native, -1])),
            # 20:
            (
                self.mon_native,
                self.mutable_l3,
                LinearExpression([self.mon_native, self.minus_npv]),
            ),
        ]
        self._run_cases(tests, operator.sub)
        self._run_cases(tests, operator.isub)

    def test_sub_mon_param(self):
        tests = [
            (self.mon_param, self.invalid, NotImplemented),
            (
                self.mon_param,
                self.asbinary,
                LinearExpression([self.mon_param, self.minus_bin]),
            ),
            (self.mon_param, self.zero, self.mon_param),
            (self.mon_param, self.one, LinearExpression([self.mon_param, -1])),
            # 4:
            (self.mon_param, self.native, LinearExpression([self.mon_param, -5])),
            (
                self.mon_param,
                self.npv,
                LinearExpression([self.mon_param, self.minus_npv]),
            ),
            (self.mon_param, self.param, LinearExpression([self.mon_param, -6])),
            (
                self.mon_param,
                self.param_mut,
                LinearExpression([self.mon_param, self.minus_param_mut]),
            ),
            # 8:
            (
                self.mon_param,
                self.var,
                LinearExpression([self.mon_param, self.minus_var]),
            ),
            (
                self.mon_param,
                self.mon_native,
                LinearExpression([self.mon_param, self.minus_mon_native]),
            ),
            (
                self.mon_param,
                self.mon_param,
                LinearExpression([self.mon_param, self.minus_mon_param]),
            ),
            (
                self.mon_param,
                self.mon_npv,
                LinearExpression([self.mon_param, self.minus_mon_npv]),
            ),
            # 12:
            (
                self.mon_param,
                self.linear,
                SumExpression([self.mon_param, self.minus_linear]),
            ),
            (self.mon_param, self.sum, SumExpression([self.mon_param, self.minus_sum])),
            (
                self.mon_param,
                self.other,
                SumExpression([self.mon_param, self.minus_other]),
            ),
            (self.mon_param, self.mutable_l0, self.mon_param),
            # 16:
            (
                self.mon_param,
                self.mutable_l1,
                LinearExpression([self.mon_param, self.minus_mon_npv]),
            ),
            (
                self.mon_param,
                self.mutable_l2,
                SumExpression([self.mon_param, self.minus_l2]),
            ),
            (self.mon_param, self.param0, self.mon_param),
            (self.mon_param, self.param1, LinearExpression([self.mon_param, -1])),
            # 20:
            (
                self.mon_param,
                self.mutable_l3,
                LinearExpression([self.mon_param, self.minus_npv]),
            ),
        ]
        self._run_cases(tests, operator.sub)
        self._run_cases(tests, operator.isub)

    def test_sub_mon_npv(self):
        tests = [
            (self.mon_npv, self.invalid, NotImplemented),
            (
                self.mon_npv,
                self.asbinary,
                LinearExpression([self.mon_npv, self.minus_bin]),
            ),
            (self.mon_npv, self.zero, self.mon_npv),
            (self.mon_npv, self.one, LinearExpression([self.mon_npv, -1])),
            # 4:
            (self.mon_npv, self.native, LinearExpression([self.mon_npv, -5])),
            (self.mon_npv, self.npv, LinearExpression([self.mon_npv, self.minus_npv])),
            (self.mon_npv, self.param, LinearExpression([self.mon_npv, -6])),
            (
                self.mon_npv,
                self.param_mut,
                LinearExpression([self.mon_npv, self.minus_param_mut]),
            ),
            # 8:
            (self.mon_npv, self.var, LinearExpression([self.mon_npv, self.minus_var])),
            (
                self.mon_npv,
                self.mon_native,
                LinearExpression([self.mon_npv, self.minus_mon_native]),
            ),
            (
                self.mon_npv,
                self.mon_param,
                LinearExpression([self.mon_npv, self.minus_mon_param]),
            ),
            (
                self.mon_npv,
                self.mon_npv,
                LinearExpression([self.mon_npv, self.minus_mon_npv]),
            ),
            # 12:
            (
                self.mon_npv,
                self.linear,
                SumExpression([self.mon_npv, self.minus_linear]),
            ),
            (self.mon_npv, self.sum, SumExpression([self.mon_npv, self.minus_sum])),
            (self.mon_npv, self.other, SumExpression([self.mon_npv, self.minus_other])),
            (self.mon_npv, self.mutable_l0, self.mon_npv),
            # 16:
            (
                self.mon_npv,
                self.mutable_l1,
                LinearExpression([self.mon_npv, self.minus_mon_npv]),
            ),
            (
                self.mon_npv,
                self.mutable_l2,
                SumExpression([self.mon_npv, self.minus_l2]),
            ),
            (self.mon_npv, self.param0, self.mon_npv),
            (self.mon_npv, self.param1, LinearExpression([self.mon_npv, -1])),
            # 20:
            (
                self.mon_npv,
                self.mutable_l3,
                LinearExpression([self.mon_npv, self.minus_npv]),
            ),
        ]
        self._run_cases(tests, operator.sub)
        self._run_cases(tests, operator.isub)

    def test_sub_linear(self):
        tests = [
            (self.linear, self.invalid, NotImplemented),
            (
                self.linear,
                self.asbinary,
                LinearExpression(self.linear.args + [self.minus_bin]),
            ),
            (self.linear, self.zero, self.linear),
            (self.linear, self.one, LinearExpression(self.linear.args + [-1])),
            # 4:
            (self.linear, self.native, LinearExpression(self.linear.args + [-5])),
            (
                self.linear,
                self.npv,
                LinearExpression(self.linear.args + [self.minus_npv]),
            ),
            (self.linear, self.param, LinearExpression(self.linear.args + [-6])),
            (
                self.linear,
                self.param_mut,
                LinearExpression(self.linear.args + [self.minus_param_mut]),
            ),
            # 8:
            (
                self.linear,
                self.var,
                LinearExpression(self.linear.args + [self.minus_var]),
            ),
            (
                self.linear,
                self.mon_native,
                LinearExpression(self.linear.args + [self.minus_mon_native]),
            ),
            (
                self.linear,
                self.mon_param,
                LinearExpression(self.linear.args + [self.minus_mon_param]),
            ),
            (
                self.linear,
                self.mon_npv,
                LinearExpression(self.linear.args + [self.minus_mon_npv]),
            ),
            # 12:
            (self.linear, self.linear, SumExpression([self.linear, self.minus_linear])),
            (self.linear, self.sum, SumExpression([self.linear, self.minus_sum])),
            (self.linear, self.other, SumExpression([self.linear, self.minus_other])),
            (self.linear, self.mutable_l0, self.linear),
            # 16:
            (
                self.linear,
                self.mutable_l1,
                LinearExpression(self.linear.args + [self.minus_mon_npv]),
            ),
            (self.linear, self.mutable_l2, SumExpression([self.linear, self.minus_l2])),
            (self.linear, self.param0, self.linear),
            (self.linear, self.param1, LinearExpression(self.linear.args + [-1])),
            # 20:
            (
                self.linear,
                self.mutable_l3,
                LinearExpression(self.linear.args + [self.minus_npv]),
            ),
        ]
        self._run_cases(tests, operator.sub)
        self._run_cases(tests, operator.isub)

    def test_sub_sum(self):
        tests = [
            (self.sum, self.invalid, NotImplemented),
            (self.sum, self.asbinary, SumExpression(self.sum.args + [self.minus_bin])),
            (self.sum, self.zero, self.sum),
            (self.sum, self.one, SumExpression(self.sum.args + [-1])),
            # 4:
            (self.sum, self.native, SumExpression(self.sum.args + [-5])),
            (self.sum, self.npv, SumExpression(self.sum.args + [self.minus_npv])),
            (self.sum, self.param, SumExpression(self.sum.args + [-6])),
            (
                self.sum,
                self.param_mut,
                SumExpression(self.sum.args + [self.minus_param_mut]),
            ),
            # 8:
            (self.sum, self.var, SumExpression(self.sum.args + [self.minus_var])),
            (
                self.sum,
                self.mon_native,
                SumExpression(self.sum.args + [self.minus_mon_native]),
            ),
            (
                self.sum,
                self.mon_param,
                SumExpression(self.sum.args + [self.minus_mon_param]),
            ),
            (
                self.sum,
                self.mon_npv,
                SumExpression(self.sum.args + [self.minus_mon_npv]),
            ),
            # 12:
            (self.sum, self.linear, SumExpression(self.sum.args + [self.minus_linear])),
            (self.sum, self.sum, SumExpression(self.sum.args + [self.minus_sum])),
            (self.sum, self.other, SumExpression(self.sum.args + [self.minus_other])),
            (self.sum, self.mutable_l0, self.sum),
            # 16:
            (
                self.sum,
                self.mutable_l1,
                SumExpression(self.sum.args + [self.minus_mon_npv]),
            ),
            (self.sum, self.mutable_l2, SumExpression(self.sum.args + [self.minus_l2])),
            (self.sum, self.param0, self.sum),
            (self.sum, self.param1, SumExpression(self.sum.args + [-1])),
            # 20:
            (
                self.sum,
                self.mutable_l3,
                SumExpression(self.sum.args + [self.minus_npv]),
            ),
        ]
        self._run_cases(tests, operator.sub)
        self._run_cases(tests, operator.isub)

    def test_sub_other(self):
        tests = [
            (self.other, self.invalid, NotImplemented),
            (self.other, self.asbinary, SumExpression([self.other, self.minus_bin])),
            (self.other, self.zero, self.other),
            (self.other, self.one, SumExpression([self.other, -1])),
            # 4:
            (self.other, self.native, SumExpression([self.other, -5])),
            (self.other, self.npv, SumExpression([self.other, self.minus_npv])),
            (self.other, self.param, SumExpression([self.other, -6])),
            (
                self.other,
                self.param_mut,
                SumExpression([self.other, self.minus_param_mut]),
            ),
            # 8:
            (self.other, self.var, SumExpression([self.other, self.minus_var])),
            (
                self.other,
                self.mon_native,
                SumExpression([self.other, self.minus_mon_native]),
            ),
            (
                self.other,
                self.mon_param,
                SumExpression([self.other, self.minus_mon_param]),
            ),
            (self.other, self.mon_npv, SumExpression([self.other, self.minus_mon_npv])),
            # 12:
            (self.other, self.linear, SumExpression([self.other, self.minus_linear])),
            (self.other, self.sum, SumExpression([self.other, self.minus_sum])),
            (self.other, self.other, SumExpression([self.other, self.minus_other])),
            (self.other, self.mutable_l0, self.other),
            # 16:
            (
                self.other,
                self.mutable_l1,
                SumExpression([self.other, self.minus_mon_npv]),
            ),
            (self.other, self.mutable_l2, SumExpression([self.other, self.minus_l2])),
            (self.other, self.param0, self.other),
            (self.other, self.param1, SumExpression([self.other, -1])),
            # 20:
            (self.other, self.mutable_l3, SumExpression([self.other, self.minus_npv])),
        ]
        self._run_cases(tests, operator.sub)
        self._run_cases(tests, operator.isub)

    def test_sub_mutable_l0(self):
        tests = [
            (self.mutable_l0, self.invalid, NotImplemented),
            (self.mutable_l0, self.asbinary, self.minus_bin),
            (self.mutable_l0, self.zero, 0),
            (self.mutable_l0, self.one, -1),
            # 4:
            (self.mutable_l0, self.native, -5),
            (self.mutable_l0, self.npv, self.minus_npv),
            (self.mutable_l0, self.param, -6),
            (self.mutable_l0, self.param_mut, self.minus_param_mut),
            # 8:
            (self.mutable_l0, self.var, self.minus_var),
            (self.mutable_l0, self.mon_native, self.minus_mon_native),
            (self.mutable_l0, self.mon_param, self.minus_mon_param),
            (self.mutable_l0, self.mon_npv, self.minus_mon_npv),
            # 12:
            (self.mutable_l0, self.linear, self.minus_linear),
            (self.mutable_l0, self.sum, self.minus_sum),
            (self.mutable_l0, self.other, self.minus_other),
            (self.mutable_l0, self.mutable_l0, self.l0),
            # 16:
            (self.mutable_l0, self.mutable_l1, self.minus_mon_npv),
            (self.mutable_l0, self.mutable_l2, self.minus_l2),
            (self.mutable_l0, self.param0, 0),
            (self.mutable_l0, self.param1, -1),
            # 20:
            (self.mutable_l0, self.mutable_l3, self.minus_npv),
        ]
        self._run_cases(tests, operator.sub)
        # Mutable isub handled by separate tests
        # self._run_cases(tests, operator.isub)

    def test_sub_mutable_l1(self):
        tests = [
            (self.mutable_l1, self.invalid, NotImplemented),
            (
                self.mutable_l1,
                self.asbinary,
                LinearExpression([self.l1, self.minus_bin]),
            ),
            (self.mutable_l1, self.zero, self.mon_npv),
            (self.mutable_l1, self.one, LinearExpression([self.l1, -1])),
            # 4:
            (self.mutable_l1, self.native, LinearExpression([self.l1, -5])),
            (self.mutable_l1, self.npv, LinearExpression([self.l1, self.minus_npv])),
            (self.mutable_l1, self.param, LinearExpression([self.l1, -6])),
            (
                self.mutable_l1,
                self.param_mut,
                LinearExpression([self.l1, self.minus_param_mut]),
            ),
            # 8:
            (self.mutable_l1, self.var, LinearExpression([self.l1, self.minus_var])),
            (
                self.mutable_l1,
                self.mon_native,
                LinearExpression([self.l1, self.minus_mon_native]),
            ),
            (
                self.mutable_l1,
                self.mon_param,
                LinearExpression([self.l1, self.minus_mon_param]),
            ),
            (
                self.mutable_l1,
                self.mon_npv,
                LinearExpression([self.l1, self.minus_mon_npv]),
            ),
            # 12:
            (self.mutable_l1, self.linear, SumExpression([self.l1, self.minus_linear])),
            (self.mutable_l1, self.sum, SumExpression([self.l1, self.minus_sum])),
            (self.mutable_l1, self.other, SumExpression([self.l1, self.minus_other])),
            (self.mutable_l1, self.mutable_l0, self.mon_npv),
            # 16:
            (
                self.mutable_l1,
                self.mutable_l1,
                LinearExpression([self.l1, self.minus_mon_npv]),
            ),
            (self.mutable_l1, self.mutable_l2, SumExpression([self.l1, self.minus_l2])),
            (self.mutable_l1, self.param0, self.mon_npv),
            (self.mutable_l1, self.param1, LinearExpression([self.l1, -1])),
            # 20:
            (
                self.mutable_l1,
                self.mutable_l3,
                LinearExpression([self.l1, self.minus_npv]),
            ),
        ]
        self._run_cases(tests, operator.sub)
        # Mutable isub handled by separate tests
        # self._run_cases(tests, operator.isub)

    def test_sub_mutable_l2(self):
        tests = [
            (self.mutable_l2, self.invalid, NotImplemented),
            (
                self.mutable_l2,
                self.asbinary,
                SumExpression(self.l2.args + [self.minus_bin]),
            ),
            (self.mutable_l2, self.zero, self.l2),
            (self.mutable_l2, self.one, SumExpression(self.l2.args + [-1])),
            # 4:
            (self.mutable_l2, self.native, SumExpression(self.l2.args + [-5])),
            (self.mutable_l2, self.npv, SumExpression(self.l2.args + [self.minus_npv])),
            (self.mutable_l2, self.param, SumExpression(self.l2.args + [-6])),
            (
                self.mutable_l2,
                self.param_mut,
                SumExpression(self.l2.args + [self.minus_param_mut]),
            ),
            # 8:
            (self.mutable_l2, self.var, SumExpression(self.l2.args + [self.minus_var])),
            (
                self.mutable_l2,
                self.mon_native,
                SumExpression(self.l2.args + [self.minus_mon_native]),
            ),
            (
                self.mutable_l2,
                self.mon_param,
                SumExpression(self.l2.args + [self.minus_mon_param]),
            ),
            (
                self.mutable_l2,
                self.mon_npv,
                SumExpression(self.l2.args + [self.minus_mon_npv]),
            ),
            # 12:
            (
                self.mutable_l2,
                self.linear,
                SumExpression(self.l2.args + [self.minus_linear]),
            ),
            (self.mutable_l2, self.sum, SumExpression(self.l2.args + [self.minus_sum])),
            (
                self.mutable_l2,
                self.other,
                SumExpression(self.l2.args + [self.minus_other]),
            ),
            (self.mutable_l2, self.mutable_l0, self.l2),
            # 16:
            (
                self.mutable_l2,
                self.mutable_l1,
                SumExpression(self.l2.args + [self.minus_mon_npv]),
            ),
            (
                self.mutable_l2,
                self.mutable_l2,
                SumExpression(self.l2.args + [self.minus_l2]),
            ),
            (self.mutable_l2, self.param0, self.l2),
            (self.mutable_l2, self.param1, SumExpression(self.l2.args + [-1])),
            # 20:
            (
                self.mutable_l2,
                self.mutable_l3,
                SumExpression(self.l2.args + [self.minus_npv]),
            ),
        ]
        self._run_cases(tests, operator.sub)
        # Mutable isub handled by separate tests
        # self._run_cases(tests, operator.isub)

    def test_sub_param0(self):
        tests = [
            (self.param0, self.invalid, NotImplemented),
            (self.param0, self.asbinary, self.minus_bin),
            (self.param0, self.zero, 0),
            (self.param0, self.one, -1),
            # 4:
            (self.param0, self.native, -5),
            (self.param0, self.npv, self.minus_npv),
            (self.param0, self.param, -6),
            (self.param0, self.param_mut, self.minus_param_mut),
            # 8:
            (self.param0, self.var, self.minus_var),
            (self.param0, self.mon_native, self.minus_mon_native),
            (self.param0, self.mon_param, self.minus_mon_param),
            (self.param0, self.mon_npv, self.minus_mon_npv),
            # 12:
            (self.param0, self.linear, self.minus_linear),
            (self.param0, self.sum, self.minus_sum),
            (self.param0, self.other, self.minus_other),
            (self.param0, self.mutable_l0, 0),
            # 16:
            (self.param0, self.mutable_l1, self.minus_mon_npv),
            (self.param0, self.mutable_l2, self.minus_l2),
            (self.param0, self.param0, 0),
            (self.param0, self.param1, -1),
            # 20:
            (self.param0, self.mutable_l3, self.minus_npv),
        ]
        self._run_cases(tests, operator.sub)
        self._run_cases(tests, operator.isub)

    def test_sub_param1(self):
        tests = [
            (self.param1, self.invalid, NotImplemented),
            (self.param1, self.asbinary, LinearExpression([1, self.minus_bin])),
            (self.param1, self.zero, 1),
            (self.param1, self.one, 0),
            # 4:
            (self.param1, self.native, -4),
            (self.param1, self.npv, NPV_SumExpression([1, self.minus_npv])),
            (self.param1, self.param, -5),
            (self.param1, self.param_mut, NPV_SumExpression([1, self.minus_param_mut])),
            # 8:
            (self.param1, self.var, LinearExpression([1, self.minus_var])),
            (
                self.param1,
                self.mon_native,
                LinearExpression([1, self.minus_mon_native]),
            ),
            (self.param1, self.mon_param, LinearExpression([1, self.minus_mon_param])),
            (self.param1, self.mon_npv, LinearExpression([1, self.minus_mon_npv])),
            # 12:
            (self.param1, self.linear, SumExpression([1, self.minus_linear])),
            (self.param1, self.sum, SumExpression([1, self.minus_sum])),
            (self.param1, self.other, SumExpression([1, self.minus_other])),
            (self.param1, self.mutable_l0, 1),
            # 16:
            (self.param1, self.mutable_l1, LinearExpression([1, self.minus_mon_npv])),
            (self.param1, self.mutable_l2, SumExpression([1, self.minus_l2])),
            (self.param1, self.param0, 1),
            (self.param1, self.param1, 0),
            # 20:
            (self.param1, self.mutable_l3, NPV_SumExpression([1, self.minus_npv])),
        ]
        self._run_cases(tests, operator.sub)
        self._run_cases(tests, operator.isub)

    def test_sub_mutable_l3(self):
        tests = [
            (self.mutable_l3, self.invalid, NotImplemented),
            (
                self.mutable_l3,
                self.asbinary,
                LinearExpression([self.l3, self.minus_bin]),
            ),
            (self.mutable_l3, self.zero, self.npv),
            (self.mutable_l3, self.one, NPV_SumExpression([self.l3, -1])),
            # 4:
            (self.mutable_l3, self.native, NPV_SumExpression([self.l3, -5])),
            (self.mutable_l3, self.npv, NPV_SumExpression([self.l3, self.minus_npv])),
            (self.mutable_l3, self.param, NPV_SumExpression([self.l3, -6])),
            (
                self.mutable_l3,
                self.param_mut,
                NPV_SumExpression([self.l3, self.minus_param_mut]),
            ),
            # 8:
            (self.mutable_l3, self.var, LinearExpression([self.l3, self.minus_var])),
            (
                self.mutable_l3,
                self.mon_native,
                LinearExpression([self.l3, self.minus_mon_native]),
            ),
            (
                self.mutable_l3,
                self.mon_param,
                LinearExpression([self.l3, self.minus_mon_param]),
            ),
            (
                self.mutable_l3,
                self.mon_npv,
                LinearExpression([self.l3, self.minus_mon_npv]),
            ),
            # 12:
            (self.mutable_l3, self.linear, SumExpression([self.l3, self.minus_linear])),
            (self.mutable_l3, self.sum, SumExpression([self.l3, self.minus_sum])),
            (self.mutable_l3, self.other, SumExpression([self.l3, self.minus_other])),
            (self.mutable_l3, self.mutable_l0, self.npv),
            # 16:
            (
                self.mutable_l3,
                self.mutable_l1,
                LinearExpression([self.l3, self.minus_mon_npv]),
            ),
            (self.mutable_l3, self.mutable_l2, SumExpression([self.l3, self.minus_l2])),
            (self.mutable_l3, self.param0, self.npv),
            (self.mutable_l3, self.param1, NPV_SumExpression([self.l3, -1])),
            # 20:
            # Note that because the mutable is resolved to a NPV_Sum in
            # the negation, the 1-term summation for the first arg is
            # not resolved to a bare term
            (
                self.mutable_l3,
                self.mutable_l3,
                NPV_SumExpression([self.l3, self.minus_npv]),
            ),
        ]
        self._run_cases(tests, operator.sub)
        # Mutable isub handled by separate tests
        # self._run_cases(tests, operator.isub)

    #
    #
    # MULTIPLICATION
    #
    #

    def test_mul_invalid(self):
        tests = [
            (self.invalid, self.invalid, NotImplemented),
            (self.invalid, self.asbinary, NotImplemented),
            # "invalid(str) * {0, 1, native}" are legitimate Python
            # operations and should never hit the Pyomo expression
            # system
            (self.invalid, self.zero, self.SKIP),
            (self.invalid, self.one, self.SKIP),
            # 4:
            (self.invalid, self.native, self.SKIP),
            (self.invalid, self.npv, NotImplemented),
            (self.invalid, self.param, NotImplemented),
            (self.invalid, self.param_mut, NotImplemented),
            # 8:
            (self.invalid, self.var, NotImplemented),
            (self.invalid, self.mon_native, NotImplemented),
            (self.invalid, self.mon_param, NotImplemented),
            (self.invalid, self.mon_npv, NotImplemented),
            # 12:
            (self.invalid, self.linear, NotImplemented),
            (self.invalid, self.sum, NotImplemented),
            (self.invalid, self.other, NotImplemented),
            (self.invalid, self.mutable_l0, NotImplemented),
            # 16:
            (self.invalid, self.mutable_l1, NotImplemented),
            (self.invalid, self.mutable_l2, NotImplemented),
            (self.invalid, self.param0, NotImplemented),
            (self.invalid, self.param1, NotImplemented),
            # 20:
            (self.invalid, self.mutable_l3, NotImplemented),
        ]
        self._run_cases(tests, operator.mul)
        self._run_cases(tests, operator.imul)

    def test_mul_asbinary(self):
        tests = [
            (self.asbinary, self.invalid, NotImplemented),
            # BooleanVar objects do not support multiplication
            (self.asbinary, self.asbinary, NotImplemented),
            (self.asbinary, self.zero, 0),
            (self.asbinary, self.one, self.bin),
            # 4:
            (self.asbinary, self.native, MonomialTermExpression((5, self.bin))),
            (self.asbinary, self.npv, MonomialTermExpression((self.npv, self.bin))),
            (self.asbinary, self.param, MonomialTermExpression((6, self.bin))),
            (
                self.asbinary,
                self.param_mut,
                MonomialTermExpression((self.param_mut, self.bin)),
            ),
            # 8:
            (self.asbinary, self.var, ProductExpression((self.bin, self.var))),
            (
                self.asbinary,
                self.mon_native,
                ProductExpression((self.bin, self.mon_native)),
            ),
            (
                self.asbinary,
                self.mon_param,
                ProductExpression((self.bin, self.mon_param)),
            ),
            (self.asbinary, self.mon_npv, ProductExpression((self.bin, self.mon_npv))),
            # 12:
            (self.asbinary, self.linear, ProductExpression((self.bin, self.linear))),
            (self.asbinary, self.sum, ProductExpression((self.bin, self.sum))),
            (self.asbinary, self.other, ProductExpression((self.bin, self.other))),
            (self.asbinary, self.mutable_l0, 0),
            # 16:
            (
                self.asbinary,
                self.mutable_l1,
                ProductExpression((self.bin, self.mon_npv)),
            ),
            (self.asbinary, self.mutable_l2, ProductExpression((self.bin, self.l2))),
            (self.asbinary, self.param0, 0),
            (self.asbinary, self.param1, self.bin),
            # 20:
            (
                self.asbinary,
                self.mutable_l3,
                MonomialTermExpression((self.npv, self.bin)),
            ),
        ]
        self._run_cases(tests, operator.mul)
        self._run_cases(tests, operator.imul)

    def test_mul_zero(self):
        tests = [
            # "Zero * invalid(str)" is a legitimate Python operation and
            # should never hit the Pyomo expression system
            (self.zero, self.invalid, self.SKIP),
            (self.zero, self.asbinary, 0),
            (self.zero, self.zero, 0),
            (self.zero, self.one, 0),
            # 4:
            (self.zero, self.native, 0),
            (self.zero, self.npv, 0),
            (self.zero, self.param, 0),
            (self.zero, self.param_mut, 0),
            # 8:
            (self.zero, self.var, 0),
            (self.zero, self.mon_native, 0),
            (self.zero, self.mon_param, 0),
            (self.zero, self.mon_npv, 0),
            # 12:
            (self.zero, self.linear, 0),
            (self.zero, self.sum, 0),
            (self.zero, self.other, 0),
            (self.zero, self.mutable_l0, 0),
            # 16:
            (self.zero, self.mutable_l1, 0),
            (self.zero, self.mutable_l2, 0),
            (self.zero, self.param0, 0),
            (self.zero, self.param1, 0),
            # 20:
            (self.zero, self.mutable_l3, 0),
        ]
        self._run_cases(tests, operator.mul)
        self._run_cases(tests, operator.imul)

    def test_mul_one(self):
        tests = [
            # "One * invalid(str)" is a legitimate Python operation and
            # should never hit the Pyomo expression system
            (self.one, self.invalid, self.SKIP),
            (self.one, self.asbinary, self.bin),
            (self.one, self.zero, 0),
            (self.one, self.one, 1),
            # 4:
            (self.one, self.native, 5),
            (self.one, self.npv, self.npv),
            (self.one, self.param, self.param),
            (self.one, self.param_mut, self.param_mut),
            # 8:
            (self.one, self.var, self.var),
            (self.one, self.mon_native, self.mon_native),
            (self.one, self.mon_param, self.mon_param),
            (self.one, self.mon_npv, self.mon_npv),
            # 12:
            (self.one, self.linear, self.linear),
            (self.one, self.sum, self.sum),
            (self.one, self.other, self.other),
            (self.one, self.mutable_l0, 0),
            # 16:
            (self.one, self.mutable_l1, self.mon_npv),
            (self.one, self.mutable_l2, self.l2),
            (self.one, self.param0, self.param0),
            (self.one, self.param1, self.param1),
            # 20:
            (self.one, self.mutable_l3, self.npv),
        ]
        self._run_cases(tests, operator.mul)
        self._run_cases(tests, operator.imul)

    def test_mul_native(self):
        tests = [
            # "Native * invalid(str) is a legitimate Python operation and
            # should never hit the Pyomo expression system
            (self.native, self.invalid, self.SKIP),
            (self.native, self.asbinary, MonomialTermExpression((5, self.bin))),
            (self.native, self.zero, 0),
            (self.native, self.one, 5),
            # 4:
            (self.native, self.native, 25),
            (self.native, self.npv, NPV_ProductExpression((5, self.npv))),
            (self.native, self.param, 30),
            (self.native, self.param_mut, NPV_ProductExpression((5, self.param_mut))),
            # 8:
            (self.native, self.var, MonomialTermExpression((5, self.var))),
            (
                self.native,
                self.mon_native,
                MonomialTermExpression((15, self.mon_native.arg(1))),
            ),
            (
                self.native,
                self.mon_param,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((5, self.mon_param.arg(0))),
                        self.mon_param.arg(1),
                    )
                ),
            ),
            (
                self.native,
                self.mon_npv,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((5, self.mon_npv.arg(0))),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            # 12:
            (self.native, self.linear, ProductExpression((5, self.linear))),
            (self.native, self.sum, ProductExpression((5, self.sum))),
            (self.native, self.other, ProductExpression((5, self.other))),
            (self.native, self.mutable_l0, 0),
            # 16:
            (
                self.native,
                self.mutable_l1,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((5, self.mon_npv.arg(0))),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            (self.native, self.mutable_l2, ProductExpression((5, self.l2))),
            (self.native, self.param0, 0),
            (self.native, self.param1, 5),
            # 20:
            (self.native, self.mutable_l3, NPV_ProductExpression((5, self.npv))),
        ]
        self._run_cases(tests, operator.mul)
        self._run_cases(tests, operator.imul)

    def test_mul_npv(self):
        tests = [
            (self.npv, self.invalid, NotImplemented),
            (self.npv, self.asbinary, MonomialTermExpression((self.npv, self.bin))),
            (self.npv, self.zero, 0),
            (self.npv, self.one, self.npv),
            # 4:
            (self.npv, self.native, NPV_ProductExpression((self.npv, 5))),
            (self.npv, self.npv, NPV_ProductExpression((self.npv, self.npv))),
            (self.npv, self.param, NPV_ProductExpression((self.npv, 6))),
            (
                self.npv,
                self.param_mut,
                NPV_ProductExpression((self.npv, self.param_mut)),
            ),
            # 8:
            (self.npv, self.var, MonomialTermExpression((self.npv, self.var))),
            (
                self.npv,
                self.mon_native,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.npv, self.mon_native.arg(0))),
                        self.mon_native.arg(1),
                    )
                ),
            ),
            (
                self.npv,
                self.mon_param,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.npv, self.mon_param.arg(0))),
                        self.mon_param.arg(1),
                    )
                ),
            ),
            (
                self.npv,
                self.mon_npv,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.npv, self.mon_npv.arg(0))),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            # 12:
            (self.npv, self.linear, ProductExpression((self.npv, self.linear))),
            (self.npv, self.sum, ProductExpression((self.npv, self.sum))),
            (self.npv, self.other, ProductExpression((self.npv, self.other))),
            (self.npv, self.mutable_l0, 0),
            # 16:
            (
                self.npv,
                self.mutable_l1,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.npv, self.mon_npv.arg(0))),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            (self.npv, self.mutable_l2, ProductExpression((self.npv, self.l2))),
            (self.npv, self.param0, 0),
            (self.npv, self.param1, self.npv),
            # 20:
            (self.npv, self.mutable_l3, NPV_ProductExpression((self.npv, self.npv))),
        ]
        self._run_cases(tests, operator.mul)
        self._run_cases(tests, operator.imul)

    def test_mul_param(self):
        tests = [
            (self.param, self.invalid, NotImplemented),
            (self.param, self.asbinary, MonomialTermExpression((6, self.bin))),
            (self.param, self.zero, 0),
            (self.param, self.one, 6),
            # 4:
            (self.param, self.native, 30),
            (self.param, self.npv, NPV_ProductExpression((6, self.npv))),
            (self.param, self.param, 36),
            (self.param, self.param_mut, NPV_ProductExpression((6, self.param_mut))),
            # 8:
            (self.param, self.var, MonomialTermExpression((6, self.var))),
            (
                self.param,
                self.mon_native,
                MonomialTermExpression((18, self.mon_native.arg(1))),
            ),
            (
                self.param,
                self.mon_param,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((6, self.mon_param.arg(0))),
                        self.mon_param.arg(1),
                    )
                ),
            ),
            (
                self.param,
                self.mon_npv,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((6, self.mon_npv.arg(0))),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            # 12:
            (self.param, self.linear, ProductExpression((6, self.linear))),
            (self.param, self.sum, ProductExpression((6, self.sum))),
            (self.param, self.other, ProductExpression((6, self.other))),
            (self.param, self.mutable_l0, 0),
            # 16:
            (
                self.param,
                self.mutable_l1,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((6, self.mon_npv.arg(0))),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            (self.param, self.mutable_l2, ProductExpression((6, self.l2))),
            (self.param, self.param0, 0),
            (self.param, self.param1, 6),
            # 20:
            (self.param, self.mutable_l3, NPV_ProductExpression((6, self.npv))),
        ]
        self._run_cases(tests, operator.mul)
        self._run_cases(tests, operator.imul)

    def test_mul_param_mut(self):
        tests = [
            (self.param_mut, self.invalid, NotImplemented),
            (
                self.param_mut,
                self.asbinary,
                MonomialTermExpression((self.param_mut, self.bin)),
            ),
            (self.param_mut, self.zero, 0),
            (self.param_mut, self.one, self.param_mut),
            # 4:
            (self.param_mut, self.native, NPV_ProductExpression((self.param_mut, 5))),
            (
                self.param_mut,
                self.npv,
                NPV_ProductExpression((self.param_mut, self.npv)),
            ),
            (self.param_mut, self.param, NPV_ProductExpression((self.param_mut, 6))),
            (
                self.param_mut,
                self.param_mut,
                NPV_ProductExpression((self.param_mut, self.param_mut)),
            ),
            # 8:
            (
                self.param_mut,
                self.var,
                MonomialTermExpression((self.param_mut, self.var)),
            ),
            (
                self.param_mut,
                self.mon_native,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.param_mut, self.mon_native.arg(0))),
                        self.mon_native.arg(1),
                    )
                ),
            ),
            (
                self.param_mut,
                self.mon_param,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.param_mut, self.mon_param.arg(0))),
                        self.mon_param.arg(1),
                    )
                ),
            ),
            (
                self.param_mut,
                self.mon_npv,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.param_mut, self.mon_npv.arg(0))),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            # 12:
            (
                self.param_mut,
                self.linear,
                ProductExpression((self.param_mut, self.linear)),
            ),
            (self.param_mut, self.sum, ProductExpression((self.param_mut, self.sum))),
            (
                self.param_mut,
                self.other,
                ProductExpression((self.param_mut, self.other)),
            ),
            (self.param_mut, self.mutable_l0, 0),
            # 16:
            (
                self.param_mut,
                self.mutable_l1,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.param_mut, self.mon_npv.arg(0))),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            (
                self.param_mut,
                self.mutable_l2,
                ProductExpression((self.param_mut, self.l2)),
            ),
            (self.param_mut, self.param0, 0),
            (self.param_mut, self.param1, self.param_mut),
            # 20:
            (
                self.param_mut,
                self.mutable_l3,
                NPV_ProductExpression((self.param_mut, self.npv)),
            ),
        ]
        self._run_cases(tests, operator.mul)
        self._run_cases(tests, operator.imul)

    def test_mul_var(self):
        tests = [
            (self.var, self.invalid, NotImplemented),
            (self.var, self.asbinary, ProductExpression((self.var, self.bin))),
            (self.var, self.zero, 0),
            (self.var, self.one, self.var),
            # 4:
            (self.var, self.native, MonomialTermExpression((5, self.var))),
            (self.var, self.npv, MonomialTermExpression((self.npv, self.var))),
            (self.var, self.param, MonomialTermExpression((6, self.var))),
            (
                self.var,
                self.param_mut,
                MonomialTermExpression((self.param_mut, self.var)),
            ),
            # 8:
            (self.var, self.var, ProductExpression((self.var, self.var))),
            (self.var, self.mon_native, ProductExpression((self.var, self.mon_native))),
            (self.var, self.mon_param, ProductExpression((self.var, self.mon_param))),
            (self.var, self.mon_npv, ProductExpression((self.var, self.mon_npv))),
            # 12:
            (self.var, self.linear, ProductExpression((self.var, self.linear))),
            (self.var, self.sum, ProductExpression((self.var, self.sum))),
            (self.var, self.other, ProductExpression((self.var, self.other))),
            (self.var, self.mutable_l0, 0),
            # 16:
            (self.var, self.mutable_l1, ProductExpression((self.var, self.mon_npv))),
            (self.var, self.mutable_l2, ProductExpression((self.var, self.l2))),
            (self.var, self.param0, 0),
            (self.var, self.param1, self.var),
            # 20:
            (self.var, self.mutable_l3, MonomialTermExpression((self.npv, self.var))),
        ]
        self._run_cases(tests, operator.mul)
        self._run_cases(tests, operator.imul)

    def test_mul_mon_native(self):
        tests = [
            (self.mon_native, self.invalid, NotImplemented),
            (
                self.mon_native,
                self.asbinary,
                ProductExpression((self.mon_native, self.bin)),
            ),
            (self.mon_native, self.zero, 0),
            (self.mon_native, self.one, self.mon_native),
            # 4:
            (
                self.mon_native,
                self.native,
                MonomialTermExpression((15, self.mon_native.arg(1))),
            ),
            (
                self.mon_native,
                self.npv,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_native.arg(0), self.npv)),
                        self.mon_native.arg(1),
                    )
                ),
            ),
            (
                self.mon_native,
                self.param,
                MonomialTermExpression((18, self.mon_native.arg(1))),
            ),
            (
                self.mon_native,
                self.param_mut,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_native.arg(0), self.param_mut)),
                        self.mon_native.arg(1),
                    )
                ),
            ),
            # 8:
            (self.mon_native, self.var, ProductExpression((self.mon_native, self.var))),
            (
                self.mon_native,
                self.mon_native,
                ProductExpression((self.mon_native, self.mon_native)),
            ),
            (
                self.mon_native,
                self.mon_param,
                ProductExpression((self.mon_native, self.mon_param)),
            ),
            (
                self.mon_native,
                self.mon_npv,
                ProductExpression((self.mon_native, self.mon_npv)),
            ),
            # 12:
            (
                self.mon_native,
                self.linear,
                ProductExpression((self.mon_native, self.linear)),
            ),
            (self.mon_native, self.sum, ProductExpression((self.mon_native, self.sum))),
            (
                self.mon_native,
                self.other,
                ProductExpression((self.mon_native, self.other)),
            ),
            (self.mon_native, self.mutable_l0, 0),
            # 16:
            (
                self.mon_native,
                self.mutable_l1,
                ProductExpression((self.mon_native, self.mon_npv)),
            ),
            (
                self.mon_native,
                self.mutable_l2,
                ProductExpression((self.mon_native, self.l2)),
            ),
            (self.mon_native, self.param0, 0),
            (self.mon_native, self.param1, self.mon_native),
            # 20:
            (
                self.mon_native,
                self.mutable_l3,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_native.arg(0), self.npv)),
                        self.mon_native.arg(1),
                    )
                ),
            ),
        ]
        self._run_cases(tests, operator.mul)
        self._run_cases(tests, operator.imul)

    def test_mul_mon_param(self):
        tests = [
            (self.mon_param, self.invalid, NotImplemented),
            (
                self.mon_param,
                self.asbinary,
                ProductExpression((self.mon_param, self.bin)),
            ),
            (self.mon_param, self.zero, 0),
            (self.mon_param, self.one, self.mon_param),
            # 4:
            (
                self.mon_param,
                self.native,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_param.arg(0), 5)),
                        self.mon_param.arg(1),
                    )
                ),
            ),
            (
                self.mon_param,
                self.npv,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_param.arg(0), self.npv)),
                        self.mon_param.arg(1),
                    )
                ),
            ),
            (
                self.mon_param,
                self.param,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_param.arg(0), 6)),
                        self.mon_param.arg(1),
                    )
                ),
            ),
            (
                self.mon_param,
                self.param_mut,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_param.arg(0), self.param_mut)),
                        self.mon_param.arg(1),
                    )
                ),
            ),
            # 8:
            (self.mon_param, self.var, ProductExpression((self.mon_param, self.var))),
            (
                self.mon_param,
                self.mon_native,
                ProductExpression((self.mon_param, self.mon_native)),
            ),
            (
                self.mon_param,
                self.mon_param,
                ProductExpression((self.mon_param, self.mon_param)),
            ),
            (
                self.mon_param,
                self.mon_npv,
                ProductExpression((self.mon_param, self.mon_npv)),
            ),
            # 12:
            (
                self.mon_param,
                self.linear,
                ProductExpression((self.mon_param, self.linear)),
            ),
            (self.mon_param, self.sum, ProductExpression((self.mon_param, self.sum))),
            (
                self.mon_param,
                self.other,
                ProductExpression((self.mon_param, self.other)),
            ),
            (self.mon_param, self.mutable_l0, 0),
            # 16:
            (
                self.mon_param,
                self.mutable_l1,
                ProductExpression((self.mon_param, self.mon_npv)),
            ),
            (
                self.mon_param,
                self.mutable_l2,
                ProductExpression((self.mon_param, self.l2)),
            ),
            (self.mon_param, self.param0, 0),
            (self.mon_param, self.param1, self.mon_param),
            # 20:
            (
                self.mon_param,
                self.mutable_l3,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_param.arg(0), self.npv)),
                        self.mon_param.arg(1),
                    )
                ),
            ),
        ]
        self._run_cases(tests, operator.mul)
        self._run_cases(tests, operator.imul)

    def test_mul_mon_npv(self):
        tests = [
            (self.mon_npv, self.invalid, NotImplemented),
            (self.mon_npv, self.asbinary, ProductExpression((self.mon_npv, self.bin))),
            (self.mon_npv, self.zero, 0),
            (self.mon_npv, self.one, self.mon_npv),
            # 4:
            (
                self.mon_npv,
                self.native,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_npv.arg(0), 5)),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            (
                self.mon_npv,
                self.npv,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_npv.arg(0), self.npv)),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            (
                self.mon_npv,
                self.param,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_npv.arg(0), 6)),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            (
                self.mon_npv,
                self.param_mut,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_npv.arg(0), self.param_mut)),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            # 8:
            (self.mon_npv, self.var, ProductExpression((self.mon_npv, self.var))),
            (
                self.mon_npv,
                self.mon_native,
                ProductExpression((self.mon_npv, self.mon_native)),
            ),
            (
                self.mon_npv,
                self.mon_param,
                ProductExpression((self.mon_npv, self.mon_param)),
            ),
            (
                self.mon_npv,
                self.mon_npv,
                ProductExpression((self.mon_npv, self.mon_npv)),
            ),
            # 12:
            (self.mon_npv, self.linear, ProductExpression((self.mon_npv, self.linear))),
            (self.mon_npv, self.sum, ProductExpression((self.mon_npv, self.sum))),
            (self.mon_npv, self.other, ProductExpression((self.mon_npv, self.other))),
            (self.mon_npv, self.mutable_l0, 0),
            # 16:
            (
                self.mon_npv,
                self.mutable_l1,
                ProductExpression((self.mon_npv, self.mon_npv)),
            ),
            (self.mon_npv, self.mutable_l2, ProductExpression((self.mon_npv, self.l2))),
            (self.mon_npv, self.param0, 0),
            (self.mon_npv, self.param1, self.mon_npv),
            # 20:
            (
                self.mon_npv,
                self.mutable_l3,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_npv.arg(0), self.npv)),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
        ]
        self._run_cases(tests, operator.mul)
        self._run_cases(tests, operator.imul)

    def test_mul_linear(self):
        tests = [
            (self.linear, self.invalid, NotImplemented),
            (self.linear, self.asbinary, ProductExpression((self.linear, self.bin))),
            (self.linear, self.zero, 0),
            (self.linear, self.one, self.linear),
            # 4:
            (self.linear, self.native, ProductExpression((self.linear, 5))),
            (self.linear, self.npv, ProductExpression((self.linear, self.npv))),
            (self.linear, self.param, ProductExpression((self.linear, 6))),
            (
                self.linear,
                self.param_mut,
                ProductExpression((self.linear, self.param_mut)),
            ),
            # 8:
            (self.linear, self.var, ProductExpression((self.linear, self.var))),
            (
                self.linear,
                self.mon_native,
                ProductExpression((self.linear, self.mon_native)),
            ),
            (
                self.linear,
                self.mon_param,
                ProductExpression((self.linear, self.mon_param)),
            ),
            (self.linear, self.mon_npv, ProductExpression((self.linear, self.mon_npv))),
            # 12:
            (self.linear, self.linear, ProductExpression((self.linear, self.linear))),
            (self.linear, self.sum, ProductExpression((self.linear, self.sum))),
            (self.linear, self.other, ProductExpression((self.linear, self.other))),
            (self.linear, self.mutable_l0, 0),
            # 16:
            (
                self.linear,
                self.mutable_l1,
                ProductExpression((self.linear, self.mon_npv)),
            ),
            (self.linear, self.mutable_l2, ProductExpression((self.linear, self.l2))),
            (self.linear, self.param0, 0),
            (self.linear, self.param1, self.linear),
            # 20:
            (self.linear, self.mutable_l3, ProductExpression((self.linear, self.npv))),
        ]
        self._run_cases(tests, operator.mul)
        self._run_cases(tests, operator.imul)

    def test_mul_sum(self):
        tests = [
            (self.sum, self.invalid, NotImplemented),
            (self.sum, self.asbinary, ProductExpression((self.sum, self.bin))),
            (self.sum, self.zero, 0),
            (self.sum, self.one, self.sum),
            # 4:
            (self.sum, self.native, ProductExpression((self.sum, 5))),
            (self.sum, self.npv, ProductExpression((self.sum, self.npv))),
            (self.sum, self.param, ProductExpression((self.sum, 6))),
            (self.sum, self.param_mut, ProductExpression((self.sum, self.param_mut))),
            # 8:
            (self.sum, self.var, ProductExpression((self.sum, self.var))),
            (self.sum, self.mon_native, ProductExpression((self.sum, self.mon_native))),
            (self.sum, self.mon_param, ProductExpression((self.sum, self.mon_param))),
            (self.sum, self.mon_npv, ProductExpression((self.sum, self.mon_npv))),
            # 12:
            (self.sum, self.linear, ProductExpression((self.sum, self.linear))),
            (self.sum, self.sum, ProductExpression((self.sum, self.sum))),
            (self.sum, self.other, ProductExpression((self.sum, self.other))),
            (self.sum, self.mutable_l0, 0),
            # 16:
            (self.sum, self.mutable_l1, ProductExpression((self.sum, self.mon_npv))),
            (self.sum, self.mutable_l2, ProductExpression((self.sum, self.l2))),
            (self.sum, self.param0, 0),
            (self.sum, self.param1, self.sum),
            # 20:
            (self.sum, self.mutable_l3, ProductExpression((self.sum, self.npv))),
        ]
        self._run_cases(tests, operator.mul)
        self._run_cases(tests, operator.imul)

    def test_mul_other(self):
        tests = [
            (self.other, self.invalid, NotImplemented),
            (self.other, self.asbinary, ProductExpression((self.other, self.bin))),
            (self.other, self.zero, 0),
            (self.other, self.one, self.other),
            # 4:
            (self.other, self.native, ProductExpression((self.other, 5))),
            (self.other, self.npv, ProductExpression((self.other, self.npv))),
            (self.other, self.param, ProductExpression((self.other, 6))),
            (
                self.other,
                self.param_mut,
                ProductExpression((self.other, self.param_mut)),
            ),
            # 8:
            (self.other, self.var, ProductExpression((self.other, self.var))),
            (
                self.other,
                self.mon_native,
                ProductExpression((self.other, self.mon_native)),
            ),
            (
                self.other,
                self.mon_param,
                ProductExpression((self.other, self.mon_param)),
            ),
            (self.other, self.mon_npv, ProductExpression((self.other, self.mon_npv))),
            # 12:
            (self.other, self.linear, ProductExpression((self.other, self.linear))),
            (self.other, self.sum, ProductExpression((self.other, self.sum))),
            (self.other, self.other, ProductExpression((self.other, self.other))),
            (self.other, self.mutable_l0, 0),
            # 16:
            (
                self.other,
                self.mutable_l1,
                ProductExpression((self.other, self.mon_npv)),
            ),
            (self.other, self.mutable_l2, ProductExpression((self.other, self.l2))),
            (self.other, self.param0, 0),
            (self.other, self.param1, self.other),
            # 20:
            (self.other, self.mutable_l3, ProductExpression((self.other, self.npv))),
        ]
        self._run_cases(tests, operator.mul)
        self._run_cases(tests, operator.imul)

    def test_mul_mutable_l0(self):
        tests = [
            (self.mutable_l0, self.invalid, NotImplemented),
            (self.mutable_l0, self.asbinary, 0),
            (self.mutable_l0, self.zero, 0),
            (self.mutable_l0, self.one, 0),
            # 4:
            (self.mutable_l0, self.native, 0),
            (self.mutable_l0, self.npv, 0),
            (self.mutable_l0, self.param, 0),
            (self.mutable_l0, self.param_mut, 0),
            # 8:
            (self.mutable_l0, self.var, 0),
            (self.mutable_l0, self.mon_native, 0),
            (self.mutable_l0, self.mon_param, 0),
            (self.mutable_l0, self.mon_npv, 0),
            # 12:
            (self.mutable_l0, self.linear, 0),
            (self.mutable_l0, self.sum, 0),
            (self.mutable_l0, self.other, 0),
            (self.mutable_l0, self.mutable_l0, 0),
            # 16:
            (self.mutable_l0, self.mutable_l1, 0),
            (self.mutable_l0, self.mutable_l2, 0),
            (self.mutable_l0, self.param0, 0),
            (self.mutable_l0, self.param1, 0),
            # 20:
            (self.mutable_l0, self.mutable_l3, 0),
        ]
        self._run_cases(tests, operator.mul)
        self._run_cases(tests, operator.imul)

    def test_mul_mutable_l1(self):
        tests = [
            (self.mutable_l1, self.invalid, NotImplemented),
            (
                self.mutable_l1,
                self.asbinary,
                ProductExpression((self.mon_npv, self.bin)),
            ),
            (self.mutable_l1, self.zero, 0),
            (self.mutable_l1, self.one, self.mon_npv),
            # 4:
            (
                self.mutable_l1,
                self.native,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_npv.arg(0), 5)),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            (
                self.mutable_l1,
                self.npv,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_npv.arg(0), self.npv)),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            (
                self.mutable_l1,
                self.param,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_npv.arg(0), 6)),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            (
                self.mutable_l1,
                self.param_mut,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_npv.arg(0), self.param_mut)),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            # 8:
            (self.mutable_l1, self.var, ProductExpression((self.mon_npv, self.var))),
            (
                self.mutable_l1,
                self.mon_native,
                ProductExpression((self.mon_npv, self.mon_native)),
            ),
            (
                self.mutable_l1,
                self.mon_param,
                ProductExpression((self.mon_npv, self.mon_param)),
            ),
            (
                self.mutable_l1,
                self.mon_npv,
                ProductExpression((self.mon_npv, self.mon_npv)),
            ),
            # 12:
            (
                self.mutable_l1,
                self.linear,
                ProductExpression((self.mon_npv, self.linear)),
            ),
            (self.mutable_l1, self.sum, ProductExpression((self.mon_npv, self.sum))),
            (
                self.mutable_l1,
                self.other,
                ProductExpression((self.mon_npv, self.other)),
            ),
            (self.mutable_l1, self.mutable_l0, 0),
            # 16:
            (
                self.mutable_l1,
                self.mutable_l1,
                ProductExpression((self.mon_npv, self.mon_npv)),
            ),
            (
                self.mutable_l1,
                self.mutable_l2,
                ProductExpression((self.mon_npv, self.l2)),
            ),
            (self.mutable_l1, self.param0, 0),
            (self.mutable_l1, self.param1, self.mon_npv),
            # 20:
            (
                self.mutable_l1,
                self.mutable_l3,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_npv.arg(0), self.npv)),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
        ]
        self._run_cases(tests, operator.mul)
        self._run_cases(tests, operator.imul)

    def test_mul_mutable_l2(self):
        tests = [
            (self.mutable_l2, self.invalid, NotImplemented),
            (self.mutable_l2, self.asbinary, ProductExpression((self.l2, self.bin))),
            (self.mutable_l2, self.zero, 0),
            (self.mutable_l2, self.one, self.l2),
            # 4:
            (self.mutable_l2, self.native, ProductExpression((self.l2, 5))),
            (self.mutable_l2, self.npv, ProductExpression((self.l2, self.npv))),
            (self.mutable_l2, self.param, ProductExpression((self.l2, 6))),
            (
                self.mutable_l2,
                self.param_mut,
                ProductExpression((self.l2, self.param_mut)),
            ),
            # 8:
            (self.mutable_l2, self.var, ProductExpression((self.l2, self.var))),
            (
                self.mutable_l2,
                self.mon_native,
                ProductExpression((self.l2, self.mon_native)),
            ),
            (
                self.mutable_l2,
                self.mon_param,
                ProductExpression((self.l2, self.mon_param)),
            ),
            (self.mutable_l2, self.mon_npv, ProductExpression((self.l2, self.mon_npv))),
            # 12:
            (self.mutable_l2, self.linear, ProductExpression((self.l2, self.linear))),
            (self.mutable_l2, self.sum, ProductExpression((self.l2, self.sum))),
            (self.mutable_l2, self.other, ProductExpression((self.l2, self.other))),
            (self.mutable_l2, self.mutable_l0, 0),
            # 16:
            (
                self.mutable_l2,
                self.mutable_l1,
                ProductExpression((self.l2, self.mon_npv)),
            ),
            (self.mutable_l2, self.mutable_l2, ProductExpression((self.l2, self.l2))),
            (self.mutable_l2, self.param0, 0),
            (self.mutable_l2, self.param1, self.l2),
            # 20:
            (self.mutable_l2, self.mutable_l3, ProductExpression((self.l2, self.npv))),
        ]
        self._run_cases(tests, operator.mul)
        self._run_cases(tests, operator.imul)

    def test_mul_param0(self):
        tests = [
            # "Param0 * invalid(str)" is a legitimate Python operation and
            # should never hit the Pyomo expression system
            (self.param0, self.invalid, self.SKIP),
            (self.param0, self.asbinary, 0),
            (self.param0, self.zero, 0),
            (self.param0, self.one, 0),
            # 4:
            (self.param0, self.native, 0),
            (self.param0, self.npv, 0),
            (self.param0, self.param, 0),
            (self.param0, self.param_mut, 0),
            # 8:
            (self.param0, self.var, 0),
            (self.param0, self.mon_native, 0),
            (self.param0, self.mon_param, 0),
            (self.param0, self.mon_npv, 0),
            # 12:
            (self.param0, self.linear, 0),
            (self.param0, self.sum, 0),
            (self.param0, self.other, 0),
            (self.param0, self.mutable_l0, 0),
            # 16:
            (self.param0, self.mutable_l1, 0),
            (self.param0, self.mutable_l2, 0),
            (self.param0, self.param0, 0),
            (self.param0, self.param1, 0),
            # 20:
            (self.param0, self.mutable_l3, 0),
        ]
        self._run_cases(tests, operator.mul)
        self._run_cases(tests, operator.imul)

    def test_mul_param1(self):
        tests = [
            # "One * invalid(str)" is a legitimate Python operation and
            # should never hit the Pyomo expression system
            (self.param1, self.invalid, self.SKIP),
            (self.param1, self.asbinary, self.bin),
            (self.param1, self.zero, 0),
            (self.param1, self.one, 1),
            # 4:
            (self.param1, self.native, 5),
            (self.param1, self.npv, self.npv),
            (self.param1, self.param, self.param),
            (self.param1, self.param_mut, self.param_mut),
            # 8:
            (self.param1, self.var, self.var),
            (self.param1, self.mon_native, self.mon_native),
            (self.param1, self.mon_param, self.mon_param),
            (self.param1, self.mon_npv, self.mon_npv),
            # 12:
            (self.param1, self.linear, self.linear),
            (self.param1, self.sum, self.sum),
            (self.param1, self.other, self.other),
            (self.param1, self.mutable_l0, 0),
            # 16:
            (self.param1, self.mutable_l1, self.mon_npv),
            (self.param1, self.mutable_l2, self.l2),
            (self.param1, self.param0, self.param0),
            (self.param1, self.param1, self.param1),
            # 20:
            (self.param1, self.mutable_l3, self.npv),
        ]
        self._run_cases(tests, operator.mul)
        self._run_cases(tests, operator.imul)

    def test_mul_mutable_l3(self):
        tests = [
            (self.mutable_l3, self.invalid, NotImplemented),
            (
                self.mutable_l3,
                self.asbinary,
                MonomialTermExpression((self.npv, self.bin)),
            ),
            (self.mutable_l3, self.zero, 0),
            (self.mutable_l3, self.one, self.npv),
            # 4:
            (self.mutable_l3, self.native, NPV_ProductExpression((self.npv, 5))),
            (self.mutable_l3, self.npv, NPV_ProductExpression((self.npv, self.npv))),
            (self.mutable_l3, self.param, NPV_ProductExpression((self.npv, 6))),
            (
                self.mutable_l3,
                self.param_mut,
                NPV_ProductExpression((self.npv, self.param_mut)),
            ),
            # 8:
            (self.mutable_l3, self.var, MonomialTermExpression((self.npv, self.var))),
            (
                self.mutable_l3,
                self.mon_native,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.npv, self.mon_native.arg(0))),
                        self.mon_native.arg(1),
                    )
                ),
            ),
            (
                self.mutable_l3,
                self.mon_param,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.npv, self.mon_param.arg(0))),
                        self.mon_param.arg(1),
                    )
                ),
            ),
            (
                self.mutable_l3,
                self.mon_npv,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.npv, self.mon_npv.arg(0))),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            # 12:
            (self.mutable_l3, self.linear, ProductExpression((self.npv, self.linear))),
            (self.mutable_l3, self.sum, ProductExpression((self.npv, self.sum))),
            (self.mutable_l3, self.other, ProductExpression((self.npv, self.other))),
            (self.mutable_l3, self.mutable_l0, 0),
            # 16:
            (
                self.mutable_l3,
                self.mutable_l1,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.npv, self.mon_npv.arg(0))),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            (self.mutable_l3, self.mutable_l2, ProductExpression((self.npv, self.l2))),
            (self.mutable_l3, self.param0, 0),
            (self.mutable_l3, self.param1, self.npv),
            # 20:
            (
                self.mutable_l3,
                self.mutable_l3,
                NPV_ProductExpression((self.npv, self.npv)),
            ),
        ]
        self._run_cases(tests, operator.mul)
        self._run_cases(tests, operator.imul)

    #
    #
    # DIVISION
    #
    #

    def test_div_invalid(self):
        tests = [
            (self.invalid, self.invalid, NotImplemented),
            (self.invalid, self.asbinary, NotImplemented),
            (self.invalid, self.zero, NotImplemented),
            (self.invalid, self.one, NotImplemented),
            # 4:
            (self.invalid, self.native, NotImplemented),
            (self.invalid, self.npv, NotImplemented),
            (self.invalid, self.param, NotImplemented),
            (self.invalid, self.param_mut, NotImplemented),
            # 8:
            (self.invalid, self.var, NotImplemented),
            (self.invalid, self.mon_native, NotImplemented),
            (self.invalid, self.mon_param, NotImplemented),
            (self.invalid, self.mon_npv, NotImplemented),
            # 12:
            (self.invalid, self.linear, NotImplemented),
            (self.invalid, self.sum, NotImplemented),
            (self.invalid, self.other, NotImplemented),
            (self.invalid, self.mutable_l0, NotImplemented),
            # 16:
            (self.invalid, self.mutable_l1, NotImplemented),
            (self.invalid, self.mutable_l2, NotImplemented),
            (self.invalid, self.param0, NotImplemented),
            (self.invalid, self.param1, NotImplemented),
            # 20:
            (self.invalid, self.mutable_l3, NotImplemented),
        ]
        self._run_cases(tests, operator.truediv)
        self._run_cases(tests, operator.itruediv)

    def test_div_asbinary(self):
        tests = [
            (self.asbinary, self.invalid, NotImplemented),
            # BooleanVar objects do not support division
            (self.asbinary, self.asbinary, NotImplemented),
            (self.asbinary, self.zero, ZeroDivisionError),
            (self.asbinary, self.one, self.bin),
            # 4:
            (self.asbinary, self.native, MonomialTermExpression((0.2, self.bin))),
            (
                self.asbinary,
                self.npv,
                MonomialTermExpression(
                    (NPV_DivisionExpression((1, self.npv)), self.bin)
                ),
            ),
            (self.asbinary, self.param, MonomialTermExpression((1 / 6, self.bin))),
            (
                self.asbinary,
                self.param_mut,
                MonomialTermExpression(
                    (NPV_DivisionExpression((1, self.param_mut)), self.bin)
                ),
            ),
            # 8:
            (self.asbinary, self.var, DivisionExpression((self.bin, self.var))),
            (
                self.asbinary,
                self.mon_native,
                DivisionExpression((self.bin, self.mon_native)),
            ),
            (
                self.asbinary,
                self.mon_param,
                DivisionExpression((self.bin, self.mon_param)),
            ),
            (self.asbinary, self.mon_npv, DivisionExpression((self.bin, self.mon_npv))),
            # 12:
            (self.asbinary, self.linear, DivisionExpression((self.bin, self.linear))),
            (self.asbinary, self.sum, DivisionExpression((self.bin, self.sum))),
            (self.asbinary, self.other, DivisionExpression((self.bin, self.other))),
            (self.asbinary, self.mutable_l0, ZeroDivisionError),
            # 16:
            (
                self.asbinary,
                self.mutable_l1,
                DivisionExpression((self.bin, self.mon_npv)),
            ),
            (self.asbinary, self.mutable_l2, DivisionExpression((self.bin, self.l2))),
            (self.asbinary, self.param0, ZeroDivisionError),
            (self.asbinary, self.param1, self.bin),
            # 20:
            (
                self.asbinary,
                self.mutable_l3,
                MonomialTermExpression(
                    (NPV_DivisionExpression((1, self.npv)), self.bin)
                ),
            ),
        ]
        self._run_cases(tests, operator.truediv)
        self._run_cases(tests, operator.itruediv)

    def test_div_zero(self):
        tests = [
            (self.zero, self.invalid, NotImplemented),
            (self.zero, self.asbinary, 0),
            (self.zero, self.zero, ZeroDivisionError),
            (self.zero, self.one, 0.0),
            # 4:
            (self.zero, self.native, 0.0),
            (self.zero, self.npv, 0),
            (self.zero, self.param, 0.0),
            (self.zero, self.param_mut, 0),
            # 8:
            (self.zero, self.var, 0),
            (self.zero, self.mon_native, 0),
            (self.zero, self.mon_param, 0),
            (self.zero, self.mon_npv, 0),
            # 12:
            (self.zero, self.linear, 0),
            (self.zero, self.sum, 0),
            (self.zero, self.other, 0),
            (self.zero, self.mutable_l0, ZeroDivisionError),
            # 16:
            (self.zero, self.mutable_l1, 0),
            (self.zero, self.mutable_l2, 0),
            (self.zero, self.param0, ZeroDivisionError),
            (self.zero, self.param1, 0.0),
            # 20:
            (self.zero, self.mutable_l3, 0),
        ]
        self._run_cases(tests, operator.truediv)
        self._run_cases(tests, operator.itruediv)

    def test_div_one(self):
        tests = [
            (self.one, self.invalid, NotImplemented),
            (self.one, self.asbinary, DivisionExpression((1, self.bin))),
            (self.one, self.zero, ZeroDivisionError),
            (self.one, self.one, 1.0),
            # 4:
            (self.one, self.native, 0.2),
            (self.one, self.npv, NPV_DivisionExpression((1, self.npv))),
            (self.one, self.param, 1 / 6),
            (self.one, self.param_mut, NPV_DivisionExpression((1, self.param_mut))),
            # 8:
            (self.one, self.var, DivisionExpression((1, self.var))),
            (self.one, self.mon_native, DivisionExpression((1, self.mon_native))),
            (self.one, self.mon_param, DivisionExpression((1, self.mon_param))),
            (self.one, self.mon_npv, DivisionExpression((1, self.mon_npv))),
            # 12:
            (self.one, self.linear, DivisionExpression((1, self.linear))),
            (self.one, self.sum, DivisionExpression((1, self.sum))),
            (self.one, self.other, DivisionExpression((1, self.other))),
            (self.one, self.mutable_l0, ZeroDivisionError),
            # 16:
            (self.one, self.mutable_l1, DivisionExpression((1, self.mon_npv))),
            (self.one, self.mutable_l2, DivisionExpression((1, self.l2))),
            (self.one, self.param0, ZeroDivisionError),
            (self.one, self.param1, 1.0),
            # 20:
            (self.one, self.mutable_l3, NPV_DivisionExpression((1, self.npv))),
        ]
        self._run_cases(tests, operator.truediv)
        self._run_cases(tests, operator.itruediv)

    def test_div_native(self):
        tests = [
            (self.native, self.invalid, NotImplemented),
            (self.native, self.asbinary, DivisionExpression((5, self.bin))),
            (self.native, self.zero, ZeroDivisionError),
            (self.native, self.one, 5.0),
            # 4:
            (self.native, self.native, 1.0),
            (self.native, self.npv, NPV_DivisionExpression((5, self.npv))),
            (self.native, self.param, 5 / 6),
            (self.native, self.param_mut, NPV_DivisionExpression((5, self.param_mut))),
            # 8:
            (self.native, self.var, DivisionExpression((5, self.var))),
            (self.native, self.mon_native, DivisionExpression((5, self.mon_native))),
            (self.native, self.mon_param, DivisionExpression((5, self.mon_param))),
            (self.native, self.mon_npv, DivisionExpression((5, self.mon_npv))),
            # 12:
            (self.native, self.linear, DivisionExpression((5, self.linear))),
            (self.native, self.sum, DivisionExpression((5, self.sum))),
            (self.native, self.other, DivisionExpression((5, self.other))),
            (self.native, self.mutable_l0, ZeroDivisionError),
            # 16:
            (self.native, self.mutable_l1, DivisionExpression((5, self.mon_npv))),
            (self.native, self.mutable_l2, DivisionExpression((5, self.l2))),
            (self.native, self.param0, ZeroDivisionError),
            (self.native, self.param1, 5.0),
            # 20:
            (self.native, self.mutable_l3, NPV_DivisionExpression((5, self.npv))),
        ]
        self._run_cases(tests, operator.truediv)
        self._run_cases(tests, operator.itruediv)

    def test_div_npv(self):
        tests = [
            (self.npv, self.invalid, NotImplemented),
            (self.npv, self.asbinary, DivisionExpression((self.npv, self.bin))),
            (self.npv, self.zero, ZeroDivisionError),
            (self.npv, self.one, self.npv),
            # 4:
            (self.npv, self.native, NPV_DivisionExpression((self.npv, 5))),
            (self.npv, self.npv, NPV_DivisionExpression((self.npv, self.npv))),
            (self.npv, self.param, NPV_DivisionExpression((self.npv, 6))),
            (
                self.npv,
                self.param_mut,
                NPV_DivisionExpression((self.npv, self.param_mut)),
            ),
            # 8:
            (self.npv, self.var, DivisionExpression((self.npv, self.var))),
            (
                self.npv,
                self.mon_native,
                DivisionExpression((self.npv, self.mon_native)),
            ),
            (self.npv, self.mon_param, DivisionExpression((self.npv, self.mon_param))),
            (self.npv, self.mon_npv, DivisionExpression((self.npv, self.mon_npv))),
            # 12:
            (self.npv, self.linear, DivisionExpression((self.npv, self.linear))),
            (self.npv, self.sum, DivisionExpression((self.npv, self.sum))),
            (self.npv, self.other, DivisionExpression((self.npv, self.other))),
            (self.npv, self.mutable_l0, ZeroDivisionError),
            # 16:
            (self.npv, self.mutable_l1, DivisionExpression((self.npv, self.mon_npv))),
            (self.npv, self.mutable_l2, DivisionExpression((self.npv, self.l2))),
            (self.npv, self.param0, ZeroDivisionError),
            (self.npv, self.param1, self.npv),
            # 20:
            (self.npv, self.mutable_l3, NPV_DivisionExpression((self.npv, self.npv))),
        ]
        self._run_cases(tests, operator.truediv)
        self._run_cases(tests, operator.itruediv)

    def test_div_param(self):
        tests = [
            (self.param, self.invalid, NotImplemented),
            (self.param, self.asbinary, DivisionExpression((6, self.bin))),
            (self.param, self.zero, ZeroDivisionError),
            (self.param, self.one, 6.0),
            # 4:
            (self.param, self.native, 1.2),
            (self.param, self.npv, NPV_DivisionExpression((6, self.npv))),
            (self.param, self.param, 1.0),
            (self.param, self.param_mut, NPV_DivisionExpression((6, self.param_mut))),
            # 8:
            (self.param, self.var, DivisionExpression((6, self.var))),
            (self.param, self.mon_native, DivisionExpression((6, self.mon_native))),
            (self.param, self.mon_param, DivisionExpression((6, self.mon_param))),
            (self.param, self.mon_npv, DivisionExpression((6, self.mon_npv))),
            # 12:
            (self.param, self.linear, DivisionExpression((6, self.linear))),
            (self.param, self.sum, DivisionExpression((6, self.sum))),
            (self.param, self.other, DivisionExpression((6, self.other))),
            (self.param, self.mutable_l0, ZeroDivisionError),
            # 16:
            (self.param, self.mutable_l1, DivisionExpression((6, self.mon_npv))),
            (self.param, self.mutable_l2, DivisionExpression((6, self.l2))),
            (self.param, self.param0, ZeroDivisionError),
            (self.param, self.param1, 6.0),
            # 20:
            (self.param, self.mutable_l3, NPV_DivisionExpression((6, self.npv))),
        ]
        self._run_cases(tests, operator.truediv)
        self._run_cases(tests, operator.itruediv)

    def test_div_param_mut(self):
        tests = [
            (self.param_mut, self.invalid, NotImplemented),
            (
                self.param_mut,
                self.asbinary,
                DivisionExpression((self.param_mut, self.bin)),
            ),
            (self.param_mut, self.zero, ZeroDivisionError),
            (self.param_mut, self.one, self.param_mut),
            # 4:
            (self.param_mut, self.native, NPV_DivisionExpression((self.param_mut, 5))),
            (
                self.param_mut,
                self.npv,
                NPV_DivisionExpression((self.param_mut, self.npv)),
            ),
            (self.param_mut, self.param, NPV_DivisionExpression((self.param_mut, 6))),
            (
                self.param_mut,
                self.param_mut,
                NPV_DivisionExpression((self.param_mut, self.param_mut)),
            ),
            # 8:
            (self.param_mut, self.var, DivisionExpression((self.param_mut, self.var))),
            (
                self.param_mut,
                self.mon_native,
                DivisionExpression((self.param_mut, self.mon_native)),
            ),
            (
                self.param_mut,
                self.mon_param,
                DivisionExpression((self.param_mut, self.mon_param)),
            ),
            (
                self.param_mut,
                self.mon_npv,
                DivisionExpression((self.param_mut, self.mon_npv)),
            ),
            # 12:
            (
                self.param_mut,
                self.linear,
                DivisionExpression((self.param_mut, self.linear)),
            ),
            (self.param_mut, self.sum, DivisionExpression((self.param_mut, self.sum))),
            (
                self.param_mut,
                self.other,
                DivisionExpression((self.param_mut, self.other)),
            ),
            (self.param_mut, self.mutable_l0, ZeroDivisionError),
            # 16:
            (
                self.param_mut,
                self.mutable_l1,
                DivisionExpression((self.param_mut, self.mon_npv)),
            ),
            (
                self.param_mut,
                self.mutable_l2,
                DivisionExpression((self.param_mut, self.l2)),
            ),
            (self.param_mut, self.param0, ZeroDivisionError),
            (self.param_mut, self.param1, self.param_mut),
            # 20:
            (
                self.param_mut,
                self.mutable_l3,
                NPV_DivisionExpression((self.param_mut, self.npv)),
            ),
        ]
        self._run_cases(tests, operator.truediv)
        self._run_cases(tests, operator.itruediv)

    def test_div_var(self):
        tests = [
            (self.var, self.invalid, NotImplemented),
            (self.var, self.asbinary, DivisionExpression((self.var, self.bin))),
            (self.var, self.zero, ZeroDivisionError),
            (self.var, self.one, self.var),
            # 4:
            (self.var, self.native, MonomialTermExpression((0.2, self.var))),
            (
                self.var,
                self.npv,
                MonomialTermExpression(
                    (NPV_DivisionExpression((1, self.npv)), self.var)
                ),
            ),
            (self.var, self.param, MonomialTermExpression((1 / 6.0, self.var))),
            (
                self.var,
                self.param_mut,
                MonomialTermExpression(
                    (NPV_DivisionExpression((1, self.param_mut)), self.var)
                ),
            ),
            # 8:
            (self.var, self.var, DivisionExpression((self.var, self.var))),
            (
                self.var,
                self.mon_native,
                DivisionExpression((self.var, self.mon_native)),
            ),
            (self.var, self.mon_param, DivisionExpression((self.var, self.mon_param))),
            (self.var, self.mon_npv, DivisionExpression((self.var, self.mon_npv))),
            # 12:
            (self.var, self.linear, DivisionExpression((self.var, self.linear))),
            (self.var, self.sum, DivisionExpression((self.var, self.sum))),
            (self.var, self.other, DivisionExpression((self.var, self.other))),
            (self.var, self.mutable_l0, ZeroDivisionError),
            # 16:
            (self.var, self.mutable_l1, DivisionExpression((self.var, self.mon_npv))),
            (self.var, self.mutable_l2, DivisionExpression((self.var, self.l2))),
            (self.var, self.param0, ZeroDivisionError),
            (self.var, self.param1, self.var),
            # 20:
            (
                self.var,
                self.mutable_l3,
                MonomialTermExpression(
                    (NPV_DivisionExpression((1, self.npv)), self.var)
                ),
            ),
        ]
        self._run_cases(tests, operator.truediv)
        self._run_cases(tests, operator.itruediv)

    def test_div_mon_native(self):
        tests = [
            (self.mon_native, self.invalid, NotImplemented),
            (
                self.mon_native,
                self.asbinary,
                DivisionExpression((self.mon_native, self.bin)),
            ),
            (self.mon_native, self.zero, ZeroDivisionError),
            (self.mon_native, self.one, self.mon_native),
            # 4:
            (
                self.mon_native,
                self.native,
                MonomialTermExpression((0.6, self.mon_native.arg(1))),
            ),
            (
                self.mon_native,
                self.npv,
                MonomialTermExpression(
                    (
                        NPV_DivisionExpression((self.mon_native.arg(0), self.npv)),
                        self.mon_native.arg(1),
                    )
                ),
            ),
            (
                self.mon_native,
                self.param,
                MonomialTermExpression((0.5, self.mon_native.arg(1))),
            ),
            (
                self.mon_native,
                self.param_mut,
                MonomialTermExpression(
                    (
                        NPV_DivisionExpression(
                            (self.mon_native.arg(0), self.param_mut)
                        ),
                        self.mon_native.arg(1),
                    )
                ),
            ),
            # 8:
            (
                self.mon_native,
                self.var,
                DivisionExpression((self.mon_native, self.var)),
            ),
            (
                self.mon_native,
                self.mon_native,
                DivisionExpression((self.mon_native, self.mon_native)),
            ),
            (
                self.mon_native,
                self.mon_param,
                DivisionExpression((self.mon_native, self.mon_param)),
            ),
            (
                self.mon_native,
                self.mon_npv,
                DivisionExpression((self.mon_native, self.mon_npv)),
            ),
            # 12:
            (
                self.mon_native,
                self.linear,
                DivisionExpression((self.mon_native, self.linear)),
            ),
            (
                self.mon_native,
                self.sum,
                DivisionExpression((self.mon_native, self.sum)),
            ),
            (
                self.mon_native,
                self.other,
                DivisionExpression((self.mon_native, self.other)),
            ),
            (self.mon_native, self.mutable_l0, ZeroDivisionError),
            # 16:
            (
                self.mon_native,
                self.mutable_l1,
                DivisionExpression((self.mon_native, self.mon_npv)),
            ),
            (
                self.mon_native,
                self.mutable_l2,
                DivisionExpression((self.mon_native, self.l2)),
            ),
            (self.mon_native, self.param0, ZeroDivisionError),
            (self.mon_native, self.param1, self.mon_native),
            # 20:
            (
                self.mon_native,
                self.mutable_l3,
                MonomialTermExpression(
                    (
                        NPV_DivisionExpression((self.mon_native.arg(0), self.npv)),
                        self.mon_native.arg(1),
                    )
                ),
            ),
        ]
        self._run_cases(tests, operator.truediv)
        self._run_cases(tests, operator.itruediv)

    def test_div_mon_param(self):
        tests = [
            (self.mon_param, self.invalid, NotImplemented),
            (
                self.mon_param,
                self.asbinary,
                DivisionExpression((self.mon_param, self.bin)),
            ),
            (self.mon_param, self.zero, ZeroDivisionError),
            (self.mon_param, self.one, self.mon_param),
            # 4:
            (
                self.mon_param,
                self.native,
                MonomialTermExpression(
                    (
                        NPV_DivisionExpression((self.mon_param.arg(0), 5)),
                        self.mon_param.arg(1),
                    )
                ),
            ),
            (
                self.mon_param,
                self.npv,
                MonomialTermExpression(
                    (
                        NPV_DivisionExpression((self.mon_param.arg(0), self.npv)),
                        self.mon_param.arg(1),
                    )
                ),
            ),
            (
                self.mon_param,
                self.param,
                MonomialTermExpression(
                    (
                        NPV_DivisionExpression((self.mon_param.arg(0), 6)),
                        self.mon_param.arg(1),
                    )
                ),
            ),
            (
                self.mon_param,
                self.param_mut,
                MonomialTermExpression(
                    (
                        NPV_DivisionExpression((self.mon_param.arg(0), self.param_mut)),
                        self.mon_param.arg(1),
                    )
                ),
            ),
            # 8:
            (self.mon_param, self.var, DivisionExpression((self.mon_param, self.var))),
            (
                self.mon_param,
                self.mon_native,
                DivisionExpression((self.mon_param, self.mon_native)),
            ),
            (
                self.mon_param,
                self.mon_param,
                DivisionExpression((self.mon_param, self.mon_param)),
            ),
            (
                self.mon_param,
                self.mon_npv,
                DivisionExpression((self.mon_param, self.mon_npv)),
            ),
            # 12:
            (
                self.mon_param,
                self.linear,
                DivisionExpression((self.mon_param, self.linear)),
            ),
            (self.mon_param, self.sum, DivisionExpression((self.mon_param, self.sum))),
            (
                self.mon_param,
                self.other,
                DivisionExpression((self.mon_param, self.other)),
            ),
            (self.mon_param, self.mutable_l0, ZeroDivisionError),
            # 16:
            (
                self.mon_param,
                self.mutable_l1,
                DivisionExpression((self.mon_param, self.mon_npv)),
            ),
            (
                self.mon_param,
                self.mutable_l2,
                DivisionExpression((self.mon_param, self.l2)),
            ),
            (self.mon_param, self.param0, ZeroDivisionError),
            (self.mon_param, self.param1, self.mon_param),
            # 20:
            (
                self.mon_param,
                self.mutable_l3,
                MonomialTermExpression(
                    (
                        NPV_DivisionExpression((self.mon_param.arg(0), self.npv)),
                        self.mon_param.arg(1),
                    )
                ),
            ),
        ]
        self._run_cases(tests, operator.truediv)
        self._run_cases(tests, operator.itruediv)

    def test_div_mon_npv(self):
        tests = [
            (self.mon_npv, self.invalid, NotImplemented),
            (self.mon_npv, self.asbinary, DivisionExpression((self.mon_npv, self.bin))),
            (self.mon_npv, self.zero, ZeroDivisionError),
            (self.mon_npv, self.one, self.mon_npv),
            # 4:
            (
                self.mon_npv,
                self.native,
                MonomialTermExpression(
                    (
                        NPV_DivisionExpression((self.mon_npv.arg(0), 5)),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            (
                self.mon_npv,
                self.npv,
                MonomialTermExpression(
                    (
                        NPV_DivisionExpression((self.mon_npv.arg(0), self.npv)),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            (
                self.mon_npv,
                self.param,
                MonomialTermExpression(
                    (
                        NPV_DivisionExpression((self.mon_npv.arg(0), 6)),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            (
                self.mon_npv,
                self.param_mut,
                MonomialTermExpression(
                    (
                        NPV_DivisionExpression((self.mon_npv.arg(0), self.param_mut)),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            # 8:
            (self.mon_npv, self.var, DivisionExpression((self.mon_npv, self.var))),
            (
                self.mon_npv,
                self.mon_native,
                DivisionExpression((self.mon_npv, self.mon_native)),
            ),
            (
                self.mon_npv,
                self.mon_param,
                DivisionExpression((self.mon_npv, self.mon_param)),
            ),
            (
                self.mon_npv,
                self.mon_npv,
                DivisionExpression((self.mon_npv, self.mon_npv)),
            ),
            # 12:
            (
                self.mon_npv,
                self.linear,
                DivisionExpression((self.mon_npv, self.linear)),
            ),
            (self.mon_npv, self.sum, DivisionExpression((self.mon_npv, self.sum))),
            (self.mon_npv, self.other, DivisionExpression((self.mon_npv, self.other))),
            (self.mon_npv, self.mutable_l0, ZeroDivisionError),
            # 16:
            (
                self.mon_npv,
                self.mutable_l1,
                DivisionExpression((self.mon_npv, self.mon_npv)),
            ),
            (
                self.mon_npv,
                self.mutable_l2,
                DivisionExpression((self.mon_npv, self.l2)),
            ),
            (self.mon_npv, self.param0, ZeroDivisionError),
            (self.mon_npv, self.param1, self.mon_npv),
            # 20:
            (
                self.mon_npv,
                self.mutable_l3,
                MonomialTermExpression(
                    (
                        NPV_DivisionExpression((self.mon_npv.arg(0), self.npv)),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
        ]
        self._run_cases(tests, operator.truediv)
        self._run_cases(tests, operator.itruediv)

    def test_div_linear(self):
        tests = [
            (self.linear, self.invalid, NotImplemented),
            (self.linear, self.asbinary, DivisionExpression((self.linear, self.bin))),
            (self.linear, self.zero, ZeroDivisionError),
            (self.linear, self.one, self.linear),
            # 4:
            (self.linear, self.native, DivisionExpression((self.linear, 5))),
            (self.linear, self.npv, DivisionExpression((self.linear, self.npv))),
            (self.linear, self.param, DivisionExpression((self.linear, 6))),
            (
                self.linear,
                self.param_mut,
                DivisionExpression((self.linear, self.param_mut)),
            ),
            # 8:
            (self.linear, self.var, DivisionExpression((self.linear, self.var))),
            (
                self.linear,
                self.mon_native,
                DivisionExpression((self.linear, self.mon_native)),
            ),
            (
                self.linear,
                self.mon_param,
                DivisionExpression((self.linear, self.mon_param)),
            ),
            (
                self.linear,
                self.mon_npv,
                DivisionExpression((self.linear, self.mon_npv)),
            ),
            # 12:
            (self.linear, self.linear, DivisionExpression((self.linear, self.linear))),
            (self.linear, self.sum, DivisionExpression((self.linear, self.sum))),
            (self.linear, self.other, DivisionExpression((self.linear, self.other))),
            (self.linear, self.mutable_l0, ZeroDivisionError),
            # 16:
            (
                self.linear,
                self.mutable_l1,
                DivisionExpression((self.linear, self.mon_npv)),
            ),
            (self.linear, self.mutable_l2, DivisionExpression((self.linear, self.l2))),
            (self.linear, self.param0, ZeroDivisionError),
            (self.linear, self.param1, self.linear),
            # 20:
            (self.linear, self.mutable_l3, DivisionExpression((self.linear, self.npv))),
        ]
        self._run_cases(tests, operator.truediv)
        self._run_cases(tests, operator.itruediv)

    def test_div_sum(self):
        tests = [
            (self.sum, self.invalid, NotImplemented),
            (self.sum, self.asbinary, DivisionExpression((self.sum, self.bin))),
            (self.sum, self.zero, ZeroDivisionError),
            (self.sum, self.one, self.sum),
            # 4:
            (self.sum, self.native, DivisionExpression((self.sum, 5))),
            (self.sum, self.npv, DivisionExpression((self.sum, self.npv))),
            (self.sum, self.param, DivisionExpression((self.sum, 6))),
            (self.sum, self.param_mut, DivisionExpression((self.sum, self.param_mut))),
            # 8:
            (self.sum, self.var, DivisionExpression((self.sum, self.var))),
            (
                self.sum,
                self.mon_native,
                DivisionExpression((self.sum, self.mon_native)),
            ),
            (self.sum, self.mon_param, DivisionExpression((self.sum, self.mon_param))),
            (self.sum, self.mon_npv, DivisionExpression((self.sum, self.mon_npv))),
            # 12:
            (self.sum, self.linear, DivisionExpression((self.sum, self.linear))),
            (self.sum, self.sum, DivisionExpression((self.sum, self.sum))),
            (self.sum, self.other, DivisionExpression((self.sum, self.other))),
            (self.sum, self.mutable_l0, ZeroDivisionError),
            # 16:
            (self.sum, self.mutable_l1, DivisionExpression((self.sum, self.mon_npv))),
            (self.sum, self.mutable_l2, DivisionExpression((self.sum, self.l2))),
            (self.sum, self.param0, ZeroDivisionError),
            (self.sum, self.param1, self.sum),
            # 20:
            (self.sum, self.mutable_l3, DivisionExpression((self.sum, self.npv))),
        ]
        self._run_cases(tests, operator.truediv)
        self._run_cases(tests, operator.itruediv)

    def test_div_other(self):
        tests = [
            (self.other, self.invalid, NotImplemented),
            (self.other, self.asbinary, DivisionExpression((self.other, self.bin))),
            (self.other, self.zero, ZeroDivisionError),
            (self.other, self.one, self.other),
            # 4:
            (self.other, self.native, DivisionExpression((self.other, 5))),
            (self.other, self.npv, DivisionExpression((self.other, self.npv))),
            (self.other, self.param, DivisionExpression((self.other, 6))),
            (
                self.other,
                self.param_mut,
                DivisionExpression((self.other, self.param_mut)),
            ),
            # 8:
            (self.other, self.var, DivisionExpression((self.other, self.var))),
            (
                self.other,
                self.mon_native,
                DivisionExpression((self.other, self.mon_native)),
            ),
            (
                self.other,
                self.mon_param,
                DivisionExpression((self.other, self.mon_param)),
            ),
            (self.other, self.mon_npv, DivisionExpression((self.other, self.mon_npv))),
            # 12:
            (self.other, self.linear, DivisionExpression((self.other, self.linear))),
            (self.other, self.sum, DivisionExpression((self.other, self.sum))),
            (self.other, self.other, DivisionExpression((self.other, self.other))),
            (self.other, self.mutable_l0, ZeroDivisionError),
            # 16:
            (
                self.other,
                self.mutable_l1,
                DivisionExpression((self.other, self.mon_npv)),
            ),
            (self.other, self.mutable_l2, DivisionExpression((self.other, self.l2))),
            (self.other, self.param0, ZeroDivisionError),
            (self.other, self.param1, self.other),
            # 20:
            (self.other, self.mutable_l3, DivisionExpression((self.other, self.npv))),
        ]
        self._run_cases(tests, operator.truediv)
        self._run_cases(tests, operator.itruediv)

    def test_div_mutable_l0(self):
        tests = [
            (self.mutable_l0, self.invalid, NotImplemented),
            (self.mutable_l0, self.asbinary, 0),
            (self.mutable_l0, self.zero, ZeroDivisionError),
            (self.mutable_l0, self.one, 0.0),
            # 4:
            (self.mutable_l0, self.native, 0.0),
            (self.mutable_l0, self.npv, 0),
            (self.mutable_l0, self.param, 0.0),
            (self.mutable_l0, self.param_mut, 0),
            # 8:
            (self.mutable_l0, self.var, 0),
            (self.mutable_l0, self.mon_native, 0),
            (self.mutable_l0, self.mon_param, 0),
            (self.mutable_l0, self.mon_npv, 0),
            # 12:
            (self.mutable_l0, self.linear, 0),
            (self.mutable_l0, self.sum, 0),
            (self.mutable_l0, self.other, 0),
            (self.mutable_l0, self.mutable_l0, ZeroDivisionError),
            # 16:
            (self.mutable_l0, self.mutable_l1, 0),
            (self.mutable_l0, self.mutable_l2, 0),
            (self.mutable_l0, self.param0, ZeroDivisionError),
            (self.mutable_l0, self.param1, 0.0),
            # 20:
            (self.mutable_l0, self.mutable_l3, 0),
        ]
        self._run_cases(tests, operator.truediv)
        self._run_cases(tests, operator.itruediv)

    def test_div_mutable_l1(self):
        tests = [
            (self.mutable_l1, self.invalid, NotImplemented),
            (
                self.mutable_l1,
                self.asbinary,
                DivisionExpression((self.mon_npv, self.bin)),
            ),
            (self.mutable_l1, self.zero, ZeroDivisionError),
            (self.mutable_l1, self.one, self.mon_npv),
            # 4:
            (
                self.mutable_l1,
                self.native,
                MonomialTermExpression(
                    (
                        NPV_DivisionExpression((self.mon_npv.arg(0), self.native)),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            (
                self.mutable_l1,
                self.npv,
                MonomialTermExpression(
                    (
                        NPV_DivisionExpression((self.mon_npv.arg(0), self.npv)),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            (
                self.mutable_l1,
                self.param,
                MonomialTermExpression(
                    (
                        NPV_DivisionExpression((self.mon_npv.arg(0), 6)),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            (
                self.mutable_l1,
                self.param_mut,
                MonomialTermExpression(
                    (
                        NPV_DivisionExpression((self.mon_npv.arg(0), self.param_mut)),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            # 8:
            (self.mutable_l1, self.var, DivisionExpression((self.mon_npv, self.var))),
            (
                self.mutable_l1,
                self.mon_native,
                DivisionExpression((self.mon_npv, self.mon_native)),
            ),
            (
                self.mutable_l1,
                self.mon_param,
                DivisionExpression((self.mon_npv, self.mon_param)),
            ),
            (
                self.mutable_l1,
                self.mon_npv,
                DivisionExpression((self.mon_npv, self.mon_npv)),
            ),
            # 12:
            (
                self.mutable_l1,
                self.linear,
                DivisionExpression((self.mon_npv, self.linear)),
            ),
            (self.mutable_l1, self.sum, DivisionExpression((self.mon_npv, self.sum))),
            (
                self.mutable_l1,
                self.other,
                DivisionExpression((self.mon_npv, self.other)),
            ),
            (self.mutable_l1, self.mutable_l0, ZeroDivisionError),
            # 16:
            (
                self.mutable_l1,
                self.mutable_l1,
                DivisionExpression((self.mon_npv, self.mon_npv)),
            ),
            (
                self.mutable_l1,
                self.mutable_l2,
                DivisionExpression((self.mon_npv, self.l2)),
            ),
            (self.mutable_l1, self.param0, ZeroDivisionError),
            (self.mutable_l1, self.param1, self.mon_npv),
            # 20:
            (
                self.mutable_l1,
                self.mutable_l3,
                MonomialTermExpression(
                    (
                        NPV_DivisionExpression((self.mon_npv.arg(0), self.npv)),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
        ]
        self._run_cases(tests, operator.truediv)
        self._run_cases(tests, operator.itruediv)

    def test_div_mutable_l2(self):
        tests = [
            (self.mutable_l2, self.invalid, NotImplemented),
            (self.mutable_l2, self.asbinary, DivisionExpression((self.l2, self.bin))),
            (self.mutable_l2, self.zero, ZeroDivisionError),
            (self.mutable_l2, self.one, self.l2),
            # 4:
            (self.mutable_l2, self.native, DivisionExpression((self.l2, 5))),
            (self.mutable_l2, self.npv, DivisionExpression((self.l2, self.npv))),
            (self.mutable_l2, self.param, DivisionExpression((self.l2, 6))),
            (
                self.mutable_l2,
                self.param_mut,
                DivisionExpression((self.l2, self.param_mut)),
            ),
            # 8:
            (self.mutable_l2, self.var, DivisionExpression((self.l2, self.var))),
            (
                self.mutable_l2,
                self.mon_native,
                DivisionExpression((self.l2, self.mon_native)),
            ),
            (
                self.mutable_l2,
                self.mon_param,
                DivisionExpression((self.l2, self.mon_param)),
            ),
            (
                self.mutable_l2,
                self.mon_npv,
                DivisionExpression((self.l2, self.mon_npv)),
            ),
            # 12:
            (self.mutable_l2, self.linear, DivisionExpression((self.l2, self.linear))),
            (self.mutable_l2, self.sum, DivisionExpression((self.l2, self.sum))),
            (self.mutable_l2, self.other, DivisionExpression((self.l2, self.other))),
            (self.mutable_l2, self.mutable_l0, ZeroDivisionError),
            # 16:
            (
                self.mutable_l2,
                self.mutable_l1,
                DivisionExpression((self.l2, self.mon_npv)),
            ),
            (self.mutable_l2, self.mutable_l2, DivisionExpression((self.l2, self.l2))),
            (self.mutable_l2, self.param0, ZeroDivisionError),
            (self.mutable_l2, self.param1, self.l2),
            # 20:
            (self.mutable_l2, self.mutable_l3, DivisionExpression((self.l2, self.npv))),
        ]
        self._run_cases(tests, operator.truediv)
        self._run_cases(tests, operator.itruediv)

    def test_div_param0(self):
        tests = [
            (self.param0, self.invalid, NotImplemented),
            (self.param0, self.asbinary, 0),
            (self.param0, self.zero, ZeroDivisionError),
            (self.param0, self.one, 0.0),
            # 4:
            (self.param0, self.native, 0.0),
            (self.param0, self.npv, 0),
            (self.param0, self.param, 0.0),
            (self.param0, self.param_mut, 0),
            # 8:
            (self.param0, self.var, 0),
            (self.param0, self.mon_native, 0),
            (self.param0, self.mon_param, 0),
            (self.param0, self.mon_npv, 0),
            # 12:
            (self.param0, self.linear, 0),
            (self.param0, self.sum, 0),
            (self.param0, self.other, 0),
            (self.param0, self.mutable_l0, ZeroDivisionError),
            # 16:
            (self.param0, self.mutable_l1, 0),
            (self.param0, self.mutable_l2, 0),
            (self.param0, self.param0, ZeroDivisionError),
            (self.param0, self.param1, 0.0),
            # 20:
            (self.param0, self.mutable_l3, 0),
        ]
        self._run_cases(tests, operator.truediv)
        self._run_cases(tests, operator.itruediv)

    def test_div_param1(self):
        tests = [
            (self.param1, self.invalid, NotImplemented),
            (self.param1, self.asbinary, DivisionExpression((1, self.bin))),
            (self.param1, self.zero, ZeroDivisionError),
            (self.param1, self.one, 1.0),
            # 4:
            (self.param1, self.native, 0.2),
            (self.param1, self.npv, NPV_DivisionExpression((1, self.npv))),
            (self.param1, self.param, 1 / 6),
            (self.param1, self.param_mut, NPV_DivisionExpression((1, self.param_mut))),
            # 8:
            (self.param1, self.var, DivisionExpression((1, self.var))),
            (self.param1, self.mon_native, DivisionExpression((1, self.mon_native))),
            (self.param1, self.mon_param, DivisionExpression((1, self.mon_param))),
            (self.param1, self.mon_npv, DivisionExpression((1, self.mon_npv))),
            # 12:
            (self.param1, self.linear, DivisionExpression((1, self.linear))),
            (self.param1, self.sum, DivisionExpression((1, self.sum))),
            (self.param1, self.other, DivisionExpression((1, self.other))),
            (self.param1, self.mutable_l0, ZeroDivisionError),
            # 16:
            (self.param1, self.mutable_l1, DivisionExpression((1, self.mon_npv))),
            (self.param1, self.mutable_l2, DivisionExpression((1, self.l2))),
            (self.param1, self.param0, ZeroDivisionError),
            (self.param1, self.param1, 1.0),
            # 20:
            (self.param1, self.mutable_l3, NPV_DivisionExpression((1, self.npv))),
        ]
        self._run_cases(tests, operator.truediv)
        self._run_cases(tests, operator.itruediv)

    def test_div_mutable_l3(self):
        tests = [
            (self.mutable_l3, self.invalid, NotImplemented),
            (self.mutable_l3, self.asbinary, DivisionExpression((self.npv, self.bin))),
            (self.mutable_l3, self.zero, ZeroDivisionError),
            (self.mutable_l3, self.one, self.npv),
            # 4:
            (self.mutable_l3, self.native, NPV_DivisionExpression((self.npv, 5))),
            (self.mutable_l3, self.npv, NPV_DivisionExpression((self.npv, self.npv))),
            (self.mutable_l3, self.param, NPV_DivisionExpression((self.npv, 6))),
            (
                self.mutable_l3,
                self.param_mut,
                NPV_DivisionExpression((self.npv, self.param_mut)),
            ),
            # 8:
            (self.mutable_l3, self.var, DivisionExpression((self.npv, self.var))),
            (
                self.mutable_l3,
                self.mon_native,
                DivisionExpression((self.npv, self.mon_native)),
            ),
            (
                self.mutable_l3,
                self.mon_param,
                DivisionExpression((self.npv, self.mon_param)),
            ),
            (
                self.mutable_l3,
                self.mon_npv,
                DivisionExpression((self.npv, self.mon_npv)),
            ),
            # 12:
            (self.mutable_l3, self.linear, DivisionExpression((self.npv, self.linear))),
            (self.mutable_l3, self.sum, DivisionExpression((self.npv, self.sum))),
            (self.mutable_l3, self.other, DivisionExpression((self.npv, self.other))),
            (self.mutable_l3, self.mutable_l0, ZeroDivisionError),
            # 16:
            (
                self.mutable_l3,
                self.mutable_l1,
                DivisionExpression((self.npv, self.mon_npv)),
            ),
            (self.mutable_l3, self.mutable_l2, DivisionExpression((self.npv, self.l2))),
            (self.mutable_l3, self.param0, ZeroDivisionError),
            (self.mutable_l3, self.param1, self.npv),
            # 20:
            (
                self.mutable_l3,
                self.mutable_l3,
                NPV_DivisionExpression((self.npv, self.npv)),
            ),
        ]
        self._run_cases(tests, operator.truediv)
        self._run_cases(tests, operator.itruediv)

    #
    #
    # EXPONENTIATION
    #
    #

    def test_pow_invalid(self):
        tests = [
            (self.invalid, self.invalid, NotImplemented),
            (self.invalid, self.asbinary, NotImplemented),
            (self.invalid, self.zero, NotImplemented),
            (self.invalid, self.one, NotImplemented),
            # 4:
            (self.invalid, self.native, NotImplemented),
            (self.invalid, self.npv, NotImplemented),
            (self.invalid, self.param, NotImplemented),
            (self.invalid, self.param_mut, NotImplemented),
            # 8:
            (self.invalid, self.var, NotImplemented),
            (self.invalid, self.mon_native, NotImplemented),
            (self.invalid, self.mon_param, NotImplemented),
            (self.invalid, self.mon_npv, NotImplemented),
            # 12:
            (self.invalid, self.linear, NotImplemented),
            (self.invalid, self.sum, NotImplemented),
            (self.invalid, self.other, NotImplemented),
            (self.invalid, self.mutable_l0, NotImplemented),
            # 16:
            (self.invalid, self.mutable_l1, NotImplemented),
            (self.invalid, self.mutable_l2, NotImplemented),
            (self.invalid, self.param0, NotImplemented),
            (self.invalid, self.param1, NotImplemented),
            # 20:
            (self.invalid, self.mutable_l3, NotImplemented),
        ]
        self._run_cases(tests, operator.pow)
        self._run_cases(tests, operator.ipow)

    def test_pow_asbinary(self):
        tests = [
            (self.asbinary, self.invalid, NotImplemented),
            # BooleanVar objects do not support exponentiation
            (self.asbinary, self.asbinary, NotImplemented),
            (self.asbinary, self.zero, 1),
            (self.asbinary, self.one, self.bin),
            # 4:
            (self.asbinary, self.native, PowExpression((self.bin, 5))),
            (self.asbinary, self.npv, PowExpression((self.bin, self.npv))),
            (self.asbinary, self.param, PowExpression((self.bin, 6))),
            (self.asbinary, self.param_mut, PowExpression((self.bin, self.param_mut))),
            # 8:
            (self.asbinary, self.var, PowExpression((self.bin, self.var))),
            (
                self.asbinary,
                self.mon_native,
                PowExpression((self.bin, self.mon_native)),
            ),
            (self.asbinary, self.mon_param, PowExpression((self.bin, self.mon_param))),
            (self.asbinary, self.mon_npv, PowExpression((self.bin, self.mon_npv))),
            # 12:
            (self.asbinary, self.linear, PowExpression((self.bin, self.linear))),
            (self.asbinary, self.sum, PowExpression((self.bin, self.sum))),
            (self.asbinary, self.other, PowExpression((self.bin, self.other))),
            (self.asbinary, self.mutable_l0, 1),
            # 16:
            (self.asbinary, self.mutable_l1, PowExpression((self.bin, self.mon_npv))),
            (self.asbinary, self.mutable_l2, PowExpression((self.bin, self.l2))),
            (self.asbinary, self.param0, 1),
            (self.asbinary, self.param1, self.bin),
            # 20:
            (self.asbinary, self.mutable_l3, PowExpression((self.bin, self.npv))),
        ]
        self._run_cases(tests, operator.pow)
        self._run_cases(tests, operator.ipow)

    def test_pow_zero(self):
        tests = [
            (self.zero, self.invalid, NotImplemented),
            (self.zero, self.asbinary, PowExpression((0, self.bin))),
            (self.zero, self.zero, 1),
            (self.zero, self.one, 0),
            # 4:
            (self.zero, self.native, 0),
            (self.zero, self.npv, NPV_PowExpression((0, self.npv))),
            (self.zero, self.param, 0),
            (self.zero, self.param_mut, NPV_PowExpression((0, self.param_mut))),
            # 8:
            (self.zero, self.var, PowExpression((0, self.var))),
            (self.zero, self.mon_native, PowExpression((0, self.mon_native))),
            (self.zero, self.mon_param, PowExpression((0, self.mon_param))),
            (self.zero, self.mon_npv, PowExpression((0, self.mon_npv))),
            # 12:
            (self.zero, self.linear, PowExpression((0, self.linear))),
            (self.zero, self.sum, PowExpression((0, self.sum))),
            (self.zero, self.other, PowExpression((0, self.other))),
            (self.zero, self.mutable_l0, 1),
            # 16:
            (self.zero, self.mutable_l1, PowExpression((0, self.mon_npv))),
            (self.zero, self.mutable_l2, PowExpression((0, self.l2))),
            (self.zero, self.param0, 1),
            (self.zero, self.param1, 0),
            # 20:
            (self.zero, self.mutable_l3, NPV_PowExpression((0, self.npv))),
        ]
        self._run_cases(tests, operator.pow)
        self._run_cases(tests, operator.ipow)

    def test_pow_one(self):
        tests = [
            (self.one, self.invalid, NotImplemented),
            (self.one, self.asbinary, PowExpression((1, self.bin))),
            (self.one, self.zero, 1),
            (self.one, self.one, 1),
            # 4:
            (self.one, self.native, 1),
            (self.one, self.npv, NPV_PowExpression((1, self.npv))),
            (self.one, self.param, 1),
            (self.one, self.param_mut, NPV_PowExpression((1, self.param_mut))),
            # 8:
            (self.one, self.var, PowExpression((1, self.var))),
            (self.one, self.mon_native, PowExpression((1, self.mon_native))),
            (self.one, self.mon_param, PowExpression((1, self.mon_param))),
            (self.one, self.mon_npv, PowExpression((1, self.mon_npv))),
            # 12:
            (self.one, self.linear, PowExpression((1, self.linear))),
            (self.one, self.sum, PowExpression((1, self.sum))),
            (self.one, self.other, PowExpression((1, self.other))),
            (self.one, self.mutable_l0, 1),
            # 16:
            (self.one, self.mutable_l1, PowExpression((1, self.mon_npv))),
            (self.one, self.mutable_l2, PowExpression((1, self.l2))),
            (self.one, self.param0, 1),
            (self.one, self.param1, 1),
            # 20:
            (self.one, self.mutable_l3, NPV_PowExpression((1, self.npv))),
        ]
        self._run_cases(tests, operator.pow)
        self._run_cases(tests, operator.ipow)

    def test_pow_native(self):
        tests = [
            (self.native, self.invalid, NotImplemented),
            (self.native, self.asbinary, PowExpression((5, self.bin))),
            (self.native, self.zero, 1),
            (self.native, self.one, 5),
            # 4:
            (self.native, self.native, 3125),
            (self.native, self.npv, NPV_PowExpression((5, self.npv))),
            (self.native, self.param, 15625),
            (self.native, self.param_mut, NPV_PowExpression((5, self.param_mut))),
            # 8:
            (self.native, self.var, PowExpression((5, self.var))),
            (self.native, self.mon_native, PowExpression((5, self.mon_native))),
            (self.native, self.mon_param, PowExpression((5, self.mon_param))),
            (self.native, self.mon_npv, PowExpression((5, self.mon_npv))),
            # 12:
            (self.native, self.linear, PowExpression((5, self.linear))),
            (self.native, self.sum, PowExpression((5, self.sum))),
            (self.native, self.other, PowExpression((5, self.other))),
            (self.native, self.mutable_l0, 1),
            # 16:
            (self.native, self.mutable_l1, PowExpression((5, self.mon_npv))),
            (self.native, self.mutable_l2, PowExpression((5, self.l2))),
            (self.native, self.param0, 1),
            (self.native, self.param1, 5),
            # 20:
            (self.native, self.mutable_l3, NPV_PowExpression((5, self.npv))),
        ]
        self._run_cases(tests, operator.pow)
        self._run_cases(tests, operator.ipow)

    def test_pow_npv(self):
        tests = [
            (self.npv, self.invalid, NotImplemented),
            (self.npv, self.asbinary, PowExpression((self.npv, self.bin))),
            (self.npv, self.zero, 1),
            (self.npv, self.one, self.npv),
            # 4:
            (self.npv, self.native, NPV_PowExpression((self.npv, 5))),
            (self.npv, self.npv, NPV_PowExpression((self.npv, self.npv))),
            (self.npv, self.param, NPV_PowExpression((self.npv, 6))),
            (self.npv, self.param_mut, NPV_PowExpression((self.npv, self.param_mut))),
            # 8:
            (self.npv, self.var, PowExpression((self.npv, self.var))),
            (self.npv, self.mon_native, PowExpression((self.npv, self.mon_native))),
            (self.npv, self.mon_param, PowExpression((self.npv, self.mon_param))),
            (self.npv, self.mon_npv, PowExpression((self.npv, self.mon_npv))),
            # 12:
            (self.npv, self.linear, PowExpression((self.npv, self.linear))),
            (self.npv, self.sum, PowExpression((self.npv, self.sum))),
            (self.npv, self.other, PowExpression((self.npv, self.other))),
            (self.npv, self.mutable_l0, 1),
            # 16:
            (self.npv, self.mutable_l1, PowExpression((self.npv, self.mon_npv))),
            (self.npv, self.mutable_l2, PowExpression((self.npv, self.l2))),
            (self.npv, self.param0, 1),
            (self.npv, self.param1, self.npv),
            # 20:
            (self.npv, self.mutable_l3, NPV_PowExpression((self.npv, self.npv))),
        ]
        self._run_cases(tests, operator.pow)
        self._run_cases(tests, operator.ipow)

    def test_pow_param(self):
        tests = [
            (self.param, self.invalid, NotImplemented),
            (self.param, self.asbinary, PowExpression((6, self.bin))),
            (self.param, self.zero, 1),
            (self.param, self.one, 6),
            # 4:
            (self.param, self.native, 7776),
            (self.param, self.npv, NPV_PowExpression((6, self.npv))),
            (self.param, self.param, 46656),
            (self.param, self.param_mut, NPV_PowExpression((6, self.param_mut))),
            # 8:
            (self.param, self.var, PowExpression((6, self.var))),
            (self.param, self.mon_native, PowExpression((6, self.mon_native))),
            (self.param, self.mon_param, PowExpression((6, self.mon_param))),
            (self.param, self.mon_npv, PowExpression((6, self.mon_npv))),
            # 12:
            (self.param, self.linear, PowExpression((6, self.linear))),
            (self.param, self.sum, PowExpression((6, self.sum))),
            (self.param, self.other, PowExpression((6, self.other))),
            (self.param, self.mutable_l0, 1),
            # 16:
            (self.param, self.mutable_l1, PowExpression((6, self.mon_npv))),
            (self.param, self.mutable_l2, PowExpression((6, self.l2))),
            (self.param, self.param0, 1),
            (self.param, self.param1, 6),
            # 20:
            (self.param, self.mutable_l3, NPV_PowExpression((6, self.npv))),
        ]
        self._run_cases(tests, operator.pow)
        self._run_cases(tests, operator.ipow)

    def test_pow_param_mut(self):
        tests = [
            (self.param_mut, self.invalid, NotImplemented),
            (self.param_mut, self.asbinary, PowExpression((self.param_mut, self.bin))),
            (self.param_mut, self.zero, 1),
            (self.param_mut, self.one, self.param_mut),
            # 4:
            (self.param_mut, self.native, NPV_PowExpression((self.param_mut, 5))),
            (self.param_mut, self.npv, NPV_PowExpression((self.param_mut, self.npv))),
            (self.param_mut, self.param, NPV_PowExpression((self.param_mut, 6))),
            (
                self.param_mut,
                self.param_mut,
                NPV_PowExpression((self.param_mut, self.param_mut)),
            ),
            # 8:
            (self.param_mut, self.var, PowExpression((self.param_mut, self.var))),
            (
                self.param_mut,
                self.mon_native,
                PowExpression((self.param_mut, self.mon_native)),
            ),
            (
                self.param_mut,
                self.mon_param,
                PowExpression((self.param_mut, self.mon_param)),
            ),
            (
                self.param_mut,
                self.mon_npv,
                PowExpression((self.param_mut, self.mon_npv)),
            ),
            # 12:
            (self.param_mut, self.linear, PowExpression((self.param_mut, self.linear))),
            (self.param_mut, self.sum, PowExpression((self.param_mut, self.sum))),
            (self.param_mut, self.other, PowExpression((self.param_mut, self.other))),
            (self.param_mut, self.mutable_l0, 1),
            # 16:
            (
                self.param_mut,
                self.mutable_l1,
                PowExpression((self.param_mut, self.mon_npv)),
            ),
            (self.param_mut, self.mutable_l2, PowExpression((self.param_mut, self.l2))),
            (self.param_mut, self.param0, 1),
            (self.param_mut, self.param1, self.param_mut),
            # 20:
            (
                self.param_mut,
                self.mutable_l3,
                NPV_PowExpression((self.param_mut, self.npv)),
            ),
        ]
        self._run_cases(tests, operator.pow)
        self._run_cases(tests, operator.ipow)

    def test_pow_var(self):
        tests = [
            (self.var, self.invalid, NotImplemented),
            (self.var, self.asbinary, PowExpression((self.var, self.bin))),
            (self.var, self.zero, 1),
            (self.var, self.one, self.var),
            # 4:
            (self.var, self.native, PowExpression((self.var, 5))),
            (self.var, self.npv, PowExpression((self.var, self.npv))),
            (self.var, self.param, PowExpression((self.var, 6))),
            (self.var, self.param_mut, PowExpression((self.var, self.param_mut))),
            # 8:
            (self.var, self.var, PowExpression((self.var, self.var))),
            (self.var, self.mon_native, PowExpression((self.var, self.mon_native))),
            (self.var, self.mon_param, PowExpression((self.var, self.mon_param))),
            (self.var, self.mon_npv, PowExpression((self.var, self.mon_npv))),
            # 12:
            (self.var, self.linear, PowExpression((self.var, self.linear))),
            (self.var, self.sum, PowExpression((self.var, self.sum))),
            (self.var, self.other, PowExpression((self.var, self.other))),
            (self.var, self.mutable_l0, 1),
            # 16:
            (self.var, self.mutable_l1, PowExpression((self.var, self.mon_npv))),
            (self.var, self.mutable_l2, PowExpression((self.var, self.l2))),
            (self.var, self.param0, 1),
            (self.var, self.param1, self.var),
            # 20:
            (self.var, self.mutable_l3, PowExpression((self.var, self.npv))),
        ]
        self._run_cases(tests, operator.pow)
        self._run_cases(tests, operator.ipow)

    def test_pow_mon_native(self):
        tests = [
            (self.mon_native, self.invalid, NotImplemented),
            (
                self.mon_native,
                self.asbinary,
                PowExpression((self.mon_native, self.bin)),
            ),
            (self.mon_native, self.zero, 1),
            (self.mon_native, self.one, self.mon_native),
            # 4:
            (self.mon_native, self.native, PowExpression((self.mon_native, 5))),
            (self.mon_native, self.npv, PowExpression((self.mon_native, self.npv))),
            (self.mon_native, self.param, PowExpression((self.mon_native, 6))),
            (
                self.mon_native,
                self.param_mut,
                PowExpression((self.mon_native, self.param_mut)),
            ),
            # 8:
            (self.mon_native, self.var, PowExpression((self.mon_native, self.var))),
            (
                self.mon_native,
                self.mon_native,
                PowExpression((self.mon_native, self.mon_native)),
            ),
            (
                self.mon_native,
                self.mon_param,
                PowExpression((self.mon_native, self.mon_param)),
            ),
            (
                self.mon_native,
                self.mon_npv,
                PowExpression((self.mon_native, self.mon_npv)),
            ),
            # 12:
            (
                self.mon_native,
                self.linear,
                PowExpression((self.mon_native, self.linear)),
            ),
            (self.mon_native, self.sum, PowExpression((self.mon_native, self.sum))),
            (self.mon_native, self.other, PowExpression((self.mon_native, self.other))),
            (self.mon_native, self.mutable_l0, 1),
            # 16:
            (
                self.mon_native,
                self.mutable_l1,
                PowExpression((self.mon_native, self.mon_npv)),
            ),
            (
                self.mon_native,
                self.mutable_l2,
                PowExpression((self.mon_native, self.l2)),
            ),
            (self.mon_native, self.param0, 1),
            (self.mon_native, self.param1, self.mon_native),
            # 20:
            (
                self.mon_native,
                self.mutable_l3,
                PowExpression((self.mon_native, self.npv)),
            ),
        ]
        self._run_cases(tests, operator.pow)
        self._run_cases(tests, operator.ipow)

    def test_pow_mon_param(self):
        tests = [
            (self.mon_param, self.invalid, NotImplemented),
            (self.mon_param, self.asbinary, PowExpression((self.mon_param, self.bin))),
            (self.mon_param, self.zero, 1),
            (self.mon_param, self.one, self.mon_param),
            # 4:
            (self.mon_param, self.native, PowExpression((self.mon_param, 5))),
            (self.mon_param, self.npv, PowExpression((self.mon_param, self.npv))),
            (self.mon_param, self.param, PowExpression((self.mon_param, 6))),
            (
                self.mon_param,
                self.param_mut,
                PowExpression((self.mon_param, self.param_mut)),
            ),
            # 8:
            (self.mon_param, self.var, PowExpression((self.mon_param, self.var))),
            (
                self.mon_param,
                self.mon_native,
                PowExpression((self.mon_param, self.mon_native)),
            ),
            (
                self.mon_param,
                self.mon_param,
                PowExpression((self.mon_param, self.mon_param)),
            ),
            (
                self.mon_param,
                self.mon_npv,
                PowExpression((self.mon_param, self.mon_npv)),
            ),
            # 12:
            (self.mon_param, self.linear, PowExpression((self.mon_param, self.linear))),
            (self.mon_param, self.sum, PowExpression((self.mon_param, self.sum))),
            (self.mon_param, self.other, PowExpression((self.mon_param, self.other))),
            (self.mon_param, self.mutable_l0, 1),
            # 16:
            (
                self.mon_param,
                self.mutable_l1,
                PowExpression((self.mon_param, self.mon_npv)),
            ),
            (self.mon_param, self.mutable_l2, PowExpression((self.mon_param, self.l2))),
            (self.mon_param, self.param0, 1),
            (self.mon_param, self.param1, self.mon_param),
            # 20:
            (
                self.mon_param,
                self.mutable_l3,
                PowExpression((self.mon_param, self.npv)),
            ),
        ]
        self._run_cases(tests, operator.pow)
        self._run_cases(tests, operator.ipow)

    def test_pow_mon_npv(self):
        tests = [
            (self.mon_npv, self.invalid, NotImplemented),
            (self.mon_npv, self.asbinary, PowExpression((self.mon_npv, self.bin))),
            (self.mon_npv, self.zero, 1),
            (self.mon_npv, self.one, self.mon_npv),
            # 4:
            (self.mon_npv, self.native, PowExpression((self.mon_npv, 5))),
            (self.mon_npv, self.npv, PowExpression((self.mon_npv, self.npv))),
            (self.mon_npv, self.param, PowExpression((self.mon_npv, 6))),
            (
                self.mon_npv,
                self.param_mut,
                PowExpression((self.mon_npv, self.param_mut)),
            ),
            # 8:
            (self.mon_npv, self.var, PowExpression((self.mon_npv, self.var))),
            (
                self.mon_npv,
                self.mon_native,
                PowExpression((self.mon_npv, self.mon_native)),
            ),
            (
                self.mon_npv,
                self.mon_param,
                PowExpression((self.mon_npv, self.mon_param)),
            ),
            (self.mon_npv, self.mon_npv, PowExpression((self.mon_npv, self.mon_npv))),
            # 12:
            (self.mon_npv, self.linear, PowExpression((self.mon_npv, self.linear))),
            (self.mon_npv, self.sum, PowExpression((self.mon_npv, self.sum))),
            (self.mon_npv, self.other, PowExpression((self.mon_npv, self.other))),
            (self.mon_npv, self.mutable_l0, 1),
            # 16:
            (
                self.mon_npv,
                self.mutable_l1,
                PowExpression((self.mon_npv, self.mon_npv)),
            ),
            (self.mon_npv, self.mutable_l2, PowExpression((self.mon_npv, self.l2))),
            (self.mon_npv, self.param0, 1),
            (self.mon_npv, self.param1, self.mon_npv),
            # 20:
            (self.mon_npv, self.mutable_l3, PowExpression((self.mon_npv, self.npv))),
        ]
        self._run_cases(tests, operator.pow)
        self._run_cases(tests, operator.ipow)

    def test_pow_linear(self):
        tests = [
            (self.linear, self.invalid, NotImplemented),
            (self.linear, self.asbinary, PowExpression((self.linear, self.bin))),
            (self.linear, self.zero, 1),
            (self.linear, self.one, self.linear),
            # 4:
            (self.linear, self.native, PowExpression((self.linear, 5))),
            (self.linear, self.npv, PowExpression((self.linear, self.npv))),
            (self.linear, self.param, PowExpression((self.linear, 6))),
            (self.linear, self.param_mut, PowExpression((self.linear, self.param_mut))),
            # 8:
            (self.linear, self.var, PowExpression((self.linear, self.var))),
            (
                self.linear,
                self.mon_native,
                PowExpression((self.linear, self.mon_native)),
            ),
            (self.linear, self.mon_param, PowExpression((self.linear, self.mon_param))),
            (self.linear, self.mon_npv, PowExpression((self.linear, self.mon_npv))),
            # 12:
            (self.linear, self.linear, PowExpression((self.linear, self.linear))),
            (self.linear, self.sum, PowExpression((self.linear, self.sum))),
            (self.linear, self.other, PowExpression((self.linear, self.other))),
            (self.linear, self.mutable_l0, 1),
            # 16:
            (self.linear, self.mutable_l1, PowExpression((self.linear, self.mon_npv))),
            (self.linear, self.mutable_l2, PowExpression((self.linear, self.l2))),
            (self.linear, self.param0, 1),
            (self.linear, self.param1, self.linear),
            # 20:
            (self.linear, self.mutable_l3, PowExpression((self.linear, self.npv))),
        ]
        self._run_cases(tests, operator.pow)
        self._run_cases(tests, operator.ipow)

    def test_pow_sum(self):
        tests = [
            (self.sum, self.invalid, NotImplemented),
            (self.sum, self.asbinary, PowExpression((self.sum, self.bin))),
            (self.sum, self.zero, 1),
            (self.sum, self.one, self.sum),
            # 4:
            (self.sum, self.native, PowExpression((self.sum, 5))),
            (self.sum, self.npv, PowExpression((self.sum, self.npv))),
            (self.sum, self.param, PowExpression((self.sum, 6))),
            (self.sum, self.param_mut, PowExpression((self.sum, self.param_mut))),
            # 8:
            (self.sum, self.var, PowExpression((self.sum, self.var))),
            (self.sum, self.mon_native, PowExpression((self.sum, self.mon_native))),
            (self.sum, self.mon_param, PowExpression((self.sum, self.mon_param))),
            (self.sum, self.mon_npv, PowExpression((self.sum, self.mon_npv))),
            # 12:
            (self.sum, self.linear, PowExpression((self.sum, self.linear))),
            (self.sum, self.sum, PowExpression((self.sum, self.sum))),
            (self.sum, self.other, PowExpression((self.sum, self.other))),
            (self.sum, self.mutable_l0, 1),
            # 16:
            (self.sum, self.mutable_l1, PowExpression((self.sum, self.mon_npv))),
            (self.sum, self.mutable_l2, PowExpression((self.sum, self.l2))),
            (self.sum, self.param0, 1),
            (self.sum, self.param1, self.sum),
            # 20:
            (self.sum, self.mutable_l3, PowExpression((self.sum, self.npv))),
        ]
        self._run_cases(tests, operator.pow)
        self._run_cases(tests, operator.ipow)

    def test_pow_other(self):
        tests = [
            (self.other, self.invalid, NotImplemented),
            (self.other, self.asbinary, PowExpression((self.other, self.bin))),
            (self.other, self.zero, 1),
            (self.other, self.one, self.other),
            # 4:
            (self.other, self.native, PowExpression((self.other, 5))),
            (self.other, self.npv, PowExpression((self.other, self.npv))),
            (self.other, self.param, PowExpression((self.other, 6))),
            (self.other, self.param_mut, PowExpression((self.other, self.param_mut))),
            # 8:
            (self.other, self.var, PowExpression((self.other, self.var))),
            (self.other, self.mon_native, PowExpression((self.other, self.mon_native))),
            (self.other, self.mon_param, PowExpression((self.other, self.mon_param))),
            (self.other, self.mon_npv, PowExpression((self.other, self.mon_npv))),
            # 12:
            (self.other, self.linear, PowExpression((self.other, self.linear))),
            (self.other, self.sum, PowExpression((self.other, self.sum))),
            (self.other, self.other, PowExpression((self.other, self.other))),
            (self.other, self.mutable_l0, 1),
            # 16:
            (self.other, self.mutable_l1, PowExpression((self.other, self.mon_npv))),
            (self.other, self.mutable_l2, PowExpression((self.other, self.l2))),
            (self.other, self.param0, 1),
            (self.other, self.param1, self.other),
            # 20:
            (self.other, self.mutable_l3, PowExpression((self.other, self.npv))),
        ]
        self._run_cases(tests, operator.pow)
        self._run_cases(tests, operator.ipow)

    def test_pow_mutable_l0(self):
        tests = [
            (self.mutable_l0, self.invalid, NotImplemented),
            (self.mutable_l0, self.asbinary, PowExpression((0, self.bin))),
            (self.mutable_l0, self.zero, 1),
            (self.mutable_l0, self.one, 0),
            # 4:
            (self.mutable_l0, self.native, 0),
            (self.mutable_l0, self.npv, NPV_PowExpression((0, self.npv))),
            (self.mutable_l0, self.param, 0),
            (self.mutable_l0, self.param_mut, NPV_PowExpression((0, self.param_mut))),
            # 8:
            (self.mutable_l0, self.var, PowExpression((0, self.var))),
            (self.mutable_l0, self.mon_native, PowExpression((0, self.mon_native))),
            (self.mutable_l0, self.mon_param, PowExpression((0, self.mon_param))),
            (self.mutable_l0, self.mon_npv, PowExpression((0, self.mon_npv))),
            # 12:
            (self.mutable_l0, self.linear, PowExpression((0, self.linear))),
            (self.mutable_l0, self.sum, PowExpression((0, self.sum))),
            (self.mutable_l0, self.other, PowExpression((0, self.other))),
            (self.mutable_l0, self.mutable_l0, 1),
            # 16:
            (self.mutable_l0, self.mutable_l1, PowExpression((0, self.mon_npv))),
            (self.mutable_l0, self.mutable_l2, PowExpression((0, self.l2))),
            (self.mutable_l0, self.param0, 1),
            (self.mutable_l0, self.param1, 0),
            # 20:
            (self.mutable_l0, self.mutable_l3, NPV_PowExpression((0, self.npv))),
        ]
        self._run_cases(tests, operator.pow)
        self._run_cases(tests, operator.ipow)

    def test_pow_mutable_l1(self):
        tests = [
            (self.mutable_l1, self.invalid, NotImplemented),
            (self.mutable_l1, self.asbinary, PowExpression((self.mon_npv, self.bin))),
            (self.mutable_l1, self.zero, 1),
            (self.mutable_l1, self.one, self.mon_npv),
            # 4:
            (self.mutable_l1, self.native, PowExpression((self.mon_npv, 5))),
            (self.mutable_l1, self.npv, PowExpression((self.mon_npv, self.npv))),
            (self.mutable_l1, self.param, PowExpression((self.mon_npv, 6))),
            (
                self.mutable_l1,
                self.param_mut,
                PowExpression((self.mon_npv, self.param_mut)),
            ),
            # 8:
            (self.mutable_l1, self.var, PowExpression((self.mon_npv, self.var))),
            (
                self.mutable_l1,
                self.mon_native,
                PowExpression((self.mon_npv, self.mon_native)),
            ),
            (
                self.mutable_l1,
                self.mon_param,
                PowExpression((self.mon_npv, self.mon_param)),
            ),
            (
                self.mutable_l1,
                self.mon_npv,
                PowExpression((self.mon_npv, self.mon_npv)),
            ),
            # 12:
            (self.mutable_l1, self.linear, PowExpression((self.mon_npv, self.linear))),
            (self.mutable_l1, self.sum, PowExpression((self.mon_npv, self.sum))),
            (self.mutable_l1, self.other, PowExpression((self.mon_npv, self.other))),
            (self.mutable_l1, self.mutable_l0, 1),
            # 16:
            (
                self.mutable_l1,
                self.mutable_l1,
                PowExpression((self.mon_npv, self.mon_npv)),
            ),
            (self.mutable_l1, self.mutable_l2, PowExpression((self.mon_npv, self.l2))),
            (self.mutable_l1, self.param0, 1),
            (self.mutable_l1, self.param1, self.mon_npv),
            # 20:
            (self.mutable_l1, self.mutable_l3, PowExpression((self.mon_npv, self.npv))),
        ]
        self._run_cases(tests, operator.pow)
        self._run_cases(tests, operator.ipow)

    def test_pow_mutable_l2(self):
        tests = [
            (self.mutable_l2, self.invalid, NotImplemented),
            (self.mutable_l2, self.asbinary, PowExpression((self.l2, self.bin))),
            (self.mutable_l2, self.zero, 1),
            (self.mutable_l2, self.one, self.l2),
            # 4:
            (self.mutable_l2, self.native, PowExpression((self.l2, 5))),
            (self.mutable_l2, self.npv, PowExpression((self.l2, self.npv))),
            (self.mutable_l2, self.param, PowExpression((self.l2, 6))),
            (self.mutable_l2, self.param_mut, PowExpression((self.l2, self.param_mut))),
            # 8:
            (self.mutable_l2, self.var, PowExpression((self.l2, self.var))),
            (
                self.mutable_l2,
                self.mon_native,
                PowExpression((self.l2, self.mon_native)),
            ),
            (self.mutable_l2, self.mon_param, PowExpression((self.l2, self.mon_param))),
            (self.mutable_l2, self.mon_npv, PowExpression((self.l2, self.mon_npv))),
            # 12:
            (self.mutable_l2, self.linear, PowExpression((self.l2, self.linear))),
            (self.mutable_l2, self.sum, PowExpression((self.l2, self.sum))),
            (self.mutable_l2, self.other, PowExpression((self.l2, self.other))),
            (self.mutable_l2, self.mutable_l0, 1),
            # 16:
            (self.mutable_l2, self.mutable_l1, PowExpression((self.l2, self.mon_npv))),
            (self.mutable_l2, self.mutable_l2, PowExpression((self.l2, self.l2))),
            (self.mutable_l2, self.param0, 1),
            (self.mutable_l2, self.param1, self.l2),
            # 20:
            (self.mutable_l2, self.mutable_l3, PowExpression((self.l2, self.npv))),
        ]
        self._run_cases(tests, operator.pow)
        self._run_cases(tests, operator.ipow)

    def test_pow_param0(self):
        tests = [
            (self.param0, self.invalid, NotImplemented),
            (self.param0, self.asbinary, PowExpression((0, self.bin))),
            (self.param0, self.zero, 1),
            (self.param0, self.one, 0),
            # 4:
            (self.param0, self.native, 0),
            (self.param0, self.npv, NPV_PowExpression((0, self.npv))),
            (self.param0, self.param, 0),
            (self.param0, self.param_mut, NPV_PowExpression((0, self.param_mut))),
            # 8:
            (self.param0, self.var, PowExpression((0, self.var))),
            (self.param0, self.mon_native, PowExpression((0, self.mon_native))),
            (self.param0, self.mon_param, PowExpression((0, self.mon_param))),
            (self.param0, self.mon_npv, PowExpression((0, self.mon_npv))),
            # 12:
            (self.param0, self.linear, PowExpression((0, self.linear))),
            (self.param0, self.sum, PowExpression((0, self.sum))),
            (self.param0, self.other, PowExpression((0, self.other))),
            (self.param0, self.mutable_l0, 1),
            # 16:
            (self.param0, self.mutable_l1, PowExpression((0, self.mon_npv))),
            (self.param0, self.mutable_l2, PowExpression((0, self.l2))),
            (self.param0, self.param0, 1),
            (self.param0, self.param1, 0),
            # 20:
            (self.param0, self.mutable_l3, NPV_PowExpression((0, self.npv))),
        ]
        self._run_cases(tests, operator.pow)
        self._run_cases(tests, operator.ipow)

    def test_pow_param1(self):
        tests = [
            (self.param1, self.invalid, NotImplemented),
            (self.param1, self.asbinary, PowExpression((1, self.bin))),
            (self.param1, self.zero, 1),
            (self.param1, self.one, 1),
            # 4:
            (self.param1, self.native, 1),
            (self.param1, self.npv, NPV_PowExpression((1, self.npv))),
            (self.param1, self.param, 1),
            (self.param1, self.param_mut, NPV_PowExpression((1, self.param_mut))),
            # 8:
            (self.param1, self.var, PowExpression((1, self.var))),
            (self.param1, self.mon_native, PowExpression((1, self.mon_native))),
            (self.param1, self.mon_param, PowExpression((1, self.mon_param))),
            (self.param1, self.mon_npv, PowExpression((1, self.mon_npv))),
            # 12:
            (self.param1, self.linear, PowExpression((1, self.linear))),
            (self.param1, self.sum, PowExpression((1, self.sum))),
            (self.param1, self.other, PowExpression((1, self.other))),
            (self.param1, self.mutable_l0, 1),
            # 16:
            (self.param1, self.mutable_l1, PowExpression((1, self.mon_npv))),
            (self.param1, self.mutable_l2, PowExpression((1, self.l2))),
            (self.param1, self.param0, 1),
            (self.param1, self.param1, 1),
            # 20:
            (self.param1, self.mutable_l3, NPV_PowExpression((1, self.npv))),
        ]
        self._run_cases(tests, operator.pow)
        self._run_cases(tests, operator.ipow)

    def test_pow_mutable_l3(self):
        tests = [
            (self.mutable_l3, self.invalid, NotImplemented),
            (self.mutable_l3, self.asbinary, PowExpression((self.npv, self.bin))),
            (self.mutable_l3, self.zero, 1),
            (self.mutable_l3, self.one, self.npv),
            # 4:
            (self.mutable_l3, self.native, NPV_PowExpression((self.npv, 5))),
            (self.mutable_l3, self.npv, NPV_PowExpression((self.npv, self.npv))),
            (self.mutable_l3, self.param, NPV_PowExpression((self.npv, 6))),
            (
                self.mutable_l3,
                self.param_mut,
                NPV_PowExpression((self.npv, self.param_mut)),
            ),
            # 8:
            (self.mutable_l3, self.var, PowExpression((self.npv, self.var))),
            (
                self.mutable_l3,
                self.mon_native,
                PowExpression((self.npv, self.mon_native)),
            ),
            (
                self.mutable_l3,
                self.mon_param,
                PowExpression((self.npv, self.mon_param)),
            ),
            (self.mutable_l3, self.mon_npv, PowExpression((self.npv, self.mon_npv))),
            # 12:
            (self.mutable_l3, self.linear, PowExpression((self.npv, self.linear))),
            (self.mutable_l3, self.sum, PowExpression((self.npv, self.sum))),
            (self.mutable_l3, self.other, PowExpression((self.npv, self.other))),
            (self.mutable_l3, self.mutable_l0, 1),
            # 16:
            (self.mutable_l3, self.mutable_l1, PowExpression((self.npv, self.mon_npv))),
            (self.mutable_l3, self.mutable_l2, PowExpression((self.npv, self.l2))),
            (self.mutable_l3, self.param0, 1),
            (self.mutable_l3, self.param1, self.npv),
            # 20:
            (self.mutable_l3, self.mutable_l3, NPV_PowExpression((self.npv, self.npv))),
        ]
        self._run_cases(tests, operator.pow)
        self._run_cases(tests, operator.ipow)

    #
    #
    # NEGATION
    #
    #

    def test_neg(self):
        tests = [
            (self.invalid, NotImplemented),
            (self.asbinary, MonomialTermExpression((-1, self.bin))),
            (self.zero, 0),
            (self.one, -1),
            # 4:
            (self.native, -5),
            (self.npv, NPV_NegationExpression((self.npv,))),
            (self.param, -6),
            (self.param_mut, NPV_NegationExpression((self.param_mut,))),
            # 8:
            (self.var, MonomialTermExpression((-1, self.var))),
            (self.mon_native, self.minus_mon_native),
            (self.mon_param, self.minus_mon_param),
            (self.mon_npv, self.minus_mon_npv),
            # 12:
            (self.linear, NegationExpression((self.linear,))),
            (self.sum, NegationExpression((self.sum,))),
            (self.other, NegationExpression((self.other,))),
            (self.mutable_l0, 0),
            # 16:
            (self.mutable_l1, self.minus_mon_npv),
            (self.mutable_l2, NegationExpression((self.l2,))),
            (self.param0, 0),
            (self.param1, -1),
            # 20:
            (self.mutable_l3, self.minus_npv),
        ]
        self._run_cases(tests, operator.neg)

    def test_neg_neg(self):
        def _neg_neg(x):
            return operator.neg(operator.neg(x))

        tests = [
            (self.invalid, NotImplemented),
            (self.asbinary, MonomialTermExpression((1, self.bin))),
            (self.zero, 0),
            (self.one, 1),
            # 4:
            (self.native, 5),
            (self.npv, self.npv),
            (self.param, 6),
            (self.param_mut, self.param_mut),
            # 8:
            (self.var, MonomialTermExpression((1, self.var))),
            (self.mon_native, self.mon_native),
            (self.mon_param, self.mon_param),
            (self.mon_npv, self.mon_npv),
            # 12:
            (self.linear, self.linear),
            (self.sum, self.sum),
            (self.other, self.other),
            (self.mutable_l0, 0),
            # 16:
            (self.mutable_l1, self.mon_npv),
            (self.mutable_l2, self.l2),
            (self.param0, 0),
            (self.param1, 1),
            # 20:
            (self.mutable_l3, self.npv),
        ]
        self._run_cases(tests, _neg_neg)

    #
    #
    # ABSOLUTE VALUE
    #
    #

    def test_abs(self):
        tests = [
            (self.invalid, NotImplemented),
            (self.asbinary, AbsExpression((self.bin,))),
            (self.zero, 0),
            (self.one, 1),
            # 4:
            (self.native, 5),
            (self.npv, NPV_AbsExpression((self.npv,))),
            (self.param, 6),
            (self.param_mut, NPV_AbsExpression((self.param_mut,))),
            # 8:
            (self.var, AbsExpression((self.var,))),
            (self.mon_native, AbsExpression((self.mon_native,))),
            (self.mon_param, AbsExpression((self.mon_param,))),
            (self.mon_npv, AbsExpression((self.mon_npv,))),
            # 12:
            (self.linear, AbsExpression((self.linear,))),
            (self.sum, AbsExpression((self.sum,))),
            (self.other, AbsExpression((self.other,))),
            (self.mutable_l0, 0),
            # 16:
            (self.mutable_l1, AbsExpression((self.mon_npv,))),
            (self.mutable_l2, AbsExpression((self.l2,))),
            (self.param0, 0),
            (self.param1, 1),
            # 20:
            (self.mutable_l3, NPV_AbsExpression((self.npv,))),
        ]
        self._run_cases(tests, operator.abs)

    #
    #
    # UNARY FUNCTION
    #
    #

    def test_unary(self):
        SKIP_0 = {'log', 'log10', 'acosh'}
        SKIP_1 = {'atanh'}
        SKIP_5 = {'asin', 'acos', 'atanh'}
        SKIP_6 = SKIP_5
        for op, name, fcn in [
            (EXPR.ceil, 'ceil', math.ceil),
            (EXPR.floor, 'floor', math.floor),
            (EXPR.exp, 'exp', math.exp),
            (EXPR.log, 'log', math.log),
            (EXPR.log10, 'log10', math.log10),
            (EXPR.sqrt, 'sqrt', math.sqrt),
            (EXPR.sin, 'sin', math.sin),
            (EXPR.cos, 'cos', math.cos),
            (EXPR.tan, 'tan', math.tan),
            (EXPR.asin, 'asin', math.asin),
            (EXPR.acos, 'acos', math.acos),
            (EXPR.atan, 'atan', math.atan),
            (EXPR.sinh, 'sinh', math.sinh),
            (EXPR.cosh, 'cosh', math.cosh),
            (EXPR.tanh, 'tanh', math.tanh),
            (EXPR.asinh, 'asinh', math.asinh),
            (EXPR.acosh, 'acosh', math.acosh),
            (EXPR.atanh, 'atanh', math.atanh),
        ]:
            tests = [
                (self.invalid, NotImplemented),
                (self.asbinary, UnaryFunctionExpression((self.bin,), name, fcn)),
                (self.zero, ValueError if name in SKIP_0 else fcn(0)),
                (self.one, ValueError if name in SKIP_1 else fcn(1)),
                # 4:
                (self.native, ValueError if name in SKIP_5 else fcn(5)),
                (self.npv, NPV_UnaryFunctionExpression((self.npv,), name, fcn)),
                (self.param, ValueError if name in SKIP_6 else fcn(6)),
                (
                    self.param_mut,
                    NPV_UnaryFunctionExpression((self.param_mut,), name, fcn),
                ),
                # 8:
                (self.var, UnaryFunctionExpression((self.var,), name, fcn)),
                (
                    self.mon_native,
                    UnaryFunctionExpression((self.mon_native,), name, fcn),
                ),
                (self.mon_param, UnaryFunctionExpression((self.mon_param,), name, fcn)),
                (self.mon_npv, UnaryFunctionExpression((self.mon_npv,), name, fcn)),
                # 12:
                (self.linear, UnaryFunctionExpression((self.linear,), name, fcn)),
                (self.sum, UnaryFunctionExpression((self.sum,), name, fcn)),
                (self.other, UnaryFunctionExpression((self.other,), name, fcn)),
                (self.mutable_l0, ValueError if name in SKIP_0 else fcn(0)),
                # 16:
                (self.mutable_l1, UnaryFunctionExpression((self.mon_npv,), name, fcn)),
                (self.mutable_l2, UnaryFunctionExpression((self.l2,), name, fcn)),
                (self.param0, ValueError if name in SKIP_0 else fcn(0)),
                (self.param1, ValueError if name in SKIP_1 else fcn(1)),
                # 20:
                (self.mutable_l3, NPV_UnaryFunctionExpression((self.npv,), name, fcn)),
            ]
            self._run_cases(tests, op)

    #
    #
    # MUTABLE SUM IADD EXPRESSIONS
    #
    #

    def test_mutable_nvp_iadd(self):
        mutable_npv = _MutableNPVSumExpression([])
        tests = [
            (mutable_npv, self.invalid, NotImplemented),
            (mutable_npv, self.asbinary, _MutableLinearExpression([self.bin])),
            (mutable_npv, self.zero, _MutableNPVSumExpression([])),
            (mutable_npv, self.one, _MutableNPVSumExpression([1])),
            # 4:
            (mutable_npv, self.native, _MutableNPVSumExpression([5])),
            (mutable_npv, self.npv, _MutableNPVSumExpression([self.npv])),
            (mutable_npv, self.param, _MutableNPVSumExpression([6])),
            (mutable_npv, self.param_mut, _MutableNPVSumExpression([self.param_mut])),
            # 8:
            (mutable_npv, self.var, _MutableLinearExpression([self.var])),
            (mutable_npv, self.mon_native, _MutableLinearExpression([self.mon_native])),
            (mutable_npv, self.mon_param, _MutableLinearExpression([self.mon_param])),
            (mutable_npv, self.mon_npv, _MutableLinearExpression([self.mon_npv])),
            # 12:
            (mutable_npv, self.linear, _MutableLinearExpression(self.linear.args)),
            (mutable_npv, self.sum, _MutableSumExpression(self.sum.args)),
            (mutable_npv, self.other, _MutableSumExpression([self.other])),
            (mutable_npv, self.mutable_l0, _MutableNPVSumExpression([])),
            # 16:
            (
                mutable_npv,
                self.mutable_l1,
                _MutableLinearExpression(self.mutable_l1.args),
            ),
            (mutable_npv, self.mutable_l2, _MutableSumExpression(self.l2.args)),
            (mutable_npv, self.param0, _MutableNPVSumExpression([])),
            (mutable_npv, self.param1, _MutableNPVSumExpression([1])),
            # 20:
            (mutable_npv, self.mutable_l3, _MutableNPVSumExpression([self.npv])),
        ]
        self._run_iadd_cases(tests, operator.iadd)

        mutable_npv = _MutableNPVSumExpression([10])
        tests = [
            (mutable_npv, self.invalid, NotImplemented),
            (mutable_npv, self.asbinary, _MutableLinearExpression([10, self.bin])),
            (mutable_npv, self.zero, _MutableNPVSumExpression([10])),
            (mutable_npv, self.one, _MutableNPVSumExpression([11])),
            # 4:
            (mutable_npv, self.native, _MutableNPVSumExpression([15])),
            (mutable_npv, self.npv, _MutableNPVSumExpression([10, self.npv])),
            (mutable_npv, self.param, _MutableNPVSumExpression([16])),
            (
                mutable_npv,
                self.param_mut,
                _MutableNPVSumExpression([10, self.param_mut]),
            ),
            # 8:
            (mutable_npv, self.var, _MutableLinearExpression([10, self.var])),
            (
                mutable_npv,
                self.mon_native,
                _MutableLinearExpression([10, self.mon_native]),
            ),
            (
                mutable_npv,
                self.mon_param,
                _MutableLinearExpression([10, self.mon_param]),
            ),
            (mutable_npv, self.mon_npv, _MutableLinearExpression([10, self.mon_npv])),
            # 12:
            (
                mutable_npv,
                self.linear,
                _MutableLinearExpression([10] + self.linear.args),
            ),
            (mutable_npv, self.sum, _MutableSumExpression([10] + self.sum.args)),
            (mutable_npv, self.other, _MutableSumExpression([10, self.other])),
            (mutable_npv, self.mutable_l0, _MutableNPVSumExpression([10])),
            # 16:
            (mutable_npv, self.mutable_l1, _MutableLinearExpression([10, self.l1])),
            (mutable_npv, self.mutable_l2, _MutableSumExpression([10] + self.l2.args)),
            (mutable_npv, self.param0, _MutableNPVSumExpression([10])),
            (mutable_npv, self.param1, _MutableNPVSumExpression([11])),
            # 20:
            (mutable_npv, self.mutable_l3, _MutableNPVSumExpression([10, self.npv])),
        ]
        self._run_iadd_cases(tests, operator.iadd)

    def test_mutable_lin_iadd(self):
        mutable_lin = _MutableLinearExpression([])
        tests = [
            (mutable_lin, self.invalid, NotImplemented),
            (mutable_lin, self.asbinary, _MutableLinearExpression([self.bin])),
            (mutable_lin, self.zero, _MutableLinearExpression([])),
            (mutable_lin, self.one, _MutableLinearExpression([1])),
            # 4:
            (mutable_lin, self.native, _MutableLinearExpression([5])),
            (mutable_lin, self.npv, _MutableLinearExpression([self.npv])),
            (mutable_lin, self.param, _MutableLinearExpression([6])),
            (mutable_lin, self.param_mut, _MutableLinearExpression([self.param_mut])),
            # 8:
            (mutable_lin, self.var, _MutableLinearExpression([self.var])),
            (mutable_lin, self.mon_native, _MutableLinearExpression([self.mon_native])),
            (mutable_lin, self.mon_param, _MutableLinearExpression([self.mon_param])),
            (mutable_lin, self.mon_npv, _MutableLinearExpression([self.mon_npv])),
            # 12:
            (mutable_lin, self.linear, _MutableLinearExpression(self.linear.args)),
            (mutable_lin, self.sum, _MutableSumExpression(self.sum.args)),
            (mutable_lin, self.other, _MutableSumExpression([self.other])),
            (mutable_lin, self.mutable_l0, _MutableLinearExpression([])),
            # 16:
            (
                mutable_lin,
                self.mutable_l1,
                _MutableLinearExpression(self.mutable_l1.args),
            ),
            (mutable_lin, self.mutable_l2, _MutableSumExpression(self.l2.args)),
            (mutable_lin, self.param0, _MutableLinearExpression([])),
            (mutable_lin, self.param1, _MutableLinearExpression([1])),
            # 20:
            (mutable_lin, self.mutable_l3, _MutableLinearExpression([self.npv])),
        ]
        self._run_iadd_cases(tests, operator.iadd)

        mutable_lin = _MutableLinearExpression([self.bin])
        tests = [
            (mutable_lin, self.invalid, NotImplemented),
            (
                mutable_lin,
                self.asbinary,
                _MutableLinearExpression([self.bin, self.bin]),
            ),
            (mutable_lin, self.zero, _MutableLinearExpression([self.bin])),
            (mutable_lin, self.one, _MutableLinearExpression([self.bin, 1])),
            # 4:
            (mutable_lin, self.native, _MutableLinearExpression([self.bin, 5])),
            (mutable_lin, self.npv, _MutableLinearExpression([self.bin, self.npv])),
            (mutable_lin, self.param, _MutableLinearExpression([self.bin, 6])),
            (
                mutable_lin,
                self.param_mut,
                _MutableLinearExpression([self.bin, self.param_mut]),
            ),
            # 8:
            (mutable_lin, self.var, _MutableLinearExpression([self.bin, self.var])),
            (
                mutable_lin,
                self.mon_native,
                _MutableLinearExpression([self.bin, self.mon_native]),
            ),
            (
                mutable_lin,
                self.mon_param,
                _MutableLinearExpression([self.bin, self.mon_param]),
            ),
            (
                mutable_lin,
                self.mon_npv,
                _MutableLinearExpression([self.bin, self.mon_npv]),
            ),
            # 12:
            (
                mutable_lin,
                self.linear,
                _MutableLinearExpression([self.bin] + self.linear.args),
            ),
            (mutable_lin, self.sum, _MutableSumExpression([self.bin] + self.sum.args)),
            (mutable_lin, self.other, _MutableSumExpression([self.bin, self.other])),
            (mutable_lin, self.mutable_l0, _MutableLinearExpression([self.bin])),
            # 16:
            (
                mutable_lin,
                self.mutable_l1,
                _MutableLinearExpression([self.bin, self.l1]),
            ),
            (
                mutable_lin,
                self.mutable_l2,
                _MutableSumExpression([self.bin] + self.l2.args),
            ),
            (mutable_lin, self.param0, _MutableLinearExpression([self.bin])),
            (mutable_lin, self.param1, _MutableLinearExpression([self.bin, 1])),
            # 20:
            (
                mutable_lin,
                self.mutable_l3,
                _MutableLinearExpression([self.bin, self.npv]),
            ),
        ]
        self._run_iadd_cases(tests, operator.iadd)

    def test_mutable_sum_iadd(self):
        mutable_sum = _MutableSumExpression([])
        tests = [
            (mutable_sum, self.invalid, NotImplemented),
            (mutable_sum, self.asbinary, _MutableSumExpression([self.bin])),
            (mutable_sum, self.zero, _MutableSumExpression([])),
            (mutable_sum, self.one, _MutableSumExpression([1])),
            # 4:
            (mutable_sum, self.native, _MutableSumExpression([5])),
            (mutable_sum, self.npv, _MutableSumExpression([self.npv])),
            (mutable_sum, self.param, _MutableSumExpression([6])),
            (mutable_sum, self.param_mut, _MutableSumExpression([self.param_mut])),
            # 8:
            (mutable_sum, self.var, _MutableSumExpression([self.var])),
            (mutable_sum, self.mon_native, _MutableSumExpression([self.mon_native])),
            (mutable_sum, self.mon_param, _MutableSumExpression([self.mon_param])),
            (mutable_sum, self.mon_npv, _MutableSumExpression([self.mon_npv])),
            # 12:
            (mutable_sum, self.linear, _MutableSumExpression([self.linear])),
            (mutable_sum, self.sum, _MutableSumExpression(self.sum.args)),
            (mutable_sum, self.other, _MutableSumExpression([self.other])),
            (mutable_sum, self.mutable_l0, _MutableSumExpression([])),
            # 16:
            (mutable_sum, self.mutable_l1, _MutableSumExpression(self.mutable_l1.args)),
            (mutable_sum, self.mutable_l2, _MutableSumExpression(self.l2.args)),
            (mutable_sum, self.param0, _MutableSumExpression([])),
            (mutable_sum, self.param1, _MutableSumExpression([1])),
            # 20:
            (mutable_sum, self.mutable_l3, _MutableSumExpression([self.npv])),
        ]
        self._run_iadd_cases(tests, operator.iadd)

        mutable_sum = _MutableSumExpression([self.other])
        tests = [
            (mutable_sum, self.invalid, NotImplemented),
            (mutable_sum, self.asbinary, _MutableSumExpression([self.other, self.bin])),
            (mutable_sum, self.zero, _MutableSumExpression([self.other])),
            (mutable_sum, self.one, _MutableSumExpression([self.other, 1])),
            # 4:
            (mutable_sum, self.native, _MutableSumExpression([self.other, 5])),
            (mutable_sum, self.npv, _MutableSumExpression([self.other, self.npv])),
            (mutable_sum, self.param, _MutableSumExpression([self.other, 6])),
            (
                mutable_sum,
                self.param_mut,
                _MutableSumExpression([self.other, self.param_mut]),
            ),
            # 8:
            (mutable_sum, self.var, _MutableSumExpression([self.other, self.var])),
            (
                mutable_sum,
                self.mon_native,
                _MutableSumExpression([self.other, self.mon_native]),
            ),
            (
                mutable_sum,
                self.mon_param,
                _MutableSumExpression([self.other, self.mon_param]),
            ),
            (
                mutable_sum,
                self.mon_npv,
                _MutableSumExpression([self.other, self.mon_npv]),
            ),
            # 12:
            (
                mutable_sum,
                self.linear,
                _MutableSumExpression([self.other, self.linear]),
            ),
            (
                mutable_sum,
                self.sum,
                _MutableSumExpression([self.other] + self.sum.args),
            ),
            (mutable_sum, self.other, _MutableSumExpression([self.other, self.other])),
            (mutable_sum, self.mutable_l0, _MutableSumExpression([self.other])),
            # 16:
            (
                mutable_sum,
                self.mutable_l1,
                _MutableSumExpression([self.other, self.l1]),
            ),
            (
                mutable_sum,
                self.mutable_l2,
                _MutableSumExpression([self.other] + self.l2.args),
            ),
            (mutable_sum, self.param0, _MutableSumExpression([self.other])),
            (mutable_sum, self.param1, _MutableSumExpression([self.other, 1])),
            # 20:
            (
                mutable_sum,
                self.mutable_l3,
                _MutableSumExpression([self.other, self.npv]),
            ),
        ]
        self._run_iadd_cases(tests, operator.iadd)
