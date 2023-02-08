#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Unit Tests for expression generation
#
import logging
import operator

import pyomo.common.unittest as unittest

from pyomo.core.expr.compare import (
    assertExpressionsEqual,
    assertExpressionsStructurallyEqual,
)
from pyomo.core.expr.current import (
    LinearExpression,
    SumExpression,
    MonomialTermExpression,
    ProductExpression,
    NPV_ProductExpression,
    PowExpression,
    NPV_PowExpression,
    _MutableSumExpression,
)
from pyomo.core.expr.visitor import clone_expression
from pyomo.environ import ConcreteModel, Param, Var, BooleanVar
from pyomo.gdp import Disjunct

logger = logging.getLogger(__name__)


class TestExprGen(unittest.TestCase):
    def setUp(self):
        # Note there are 11 basic argument "types" that determine how
        # expressions are generated (defined by the _EXPR_TYPE enum):
        #
        # class _EXPR_TYPE(enum.Enum):
        #     MUTABLE = -2
        #     ASBINARY = -1
        #     INVALID = 0
        #     NATIVE = 1
        #     NPV = 2
        #     PARAM = 3
        #     VAR = 4
        #     MONOMIAL = 5
        #     LINEAR = 6
        #     SUM = 7
        #     OTHER = 8
        self.m = ConcreteModel()
        self.m.p = Param(initialize=6, mutable=False)
        self.m.q = Param(initialize=7, mutable=True)
        self.m.x = Var()
        self.m.d = Disjunct()
        self.bin = self.m.d.indicator_var.as_binary()

        self.invalid = 'str'
        self.asbinary = self.m.d.indicator_var
        self.zero = 0
        self.one = 1
        self.native = 5
        self.npv = NPV_PowExpression((self.m.q, 2))
        self.param = self.m.p
        self.param_mut = self.m.q
        self.var = self.m.x
        self.mon_native = MonomialTermExpression((3, self.m.x))
        self.mon_param = MonomialTermExpression((self.m.q, self.m.x))
        self.mon_npv = MonomialTermExpression((self.npv, self.m.x))
        self.linear = LinearExpression([4, self.mon_native])
        self.sum = SumExpression([4, self.mon_native, self.m.x ** 2])
        self.other = PowExpression((self.m.x, 2))

        self.mutable_l0 = _MutableSumExpression([])
        self.mutable_l1 = _MutableSumExpression([self.mon_npv])
        self.mutable_l2 = _MutableSumExpression([self.mon_npv, self.other])

        # tests = [
        #     (self.xxx, self.invalid, NotImplemented),
        #     (self.xxx, self.asbinary, ),
        #     (self.xxx, self.zero, ),
        #     (self.xxx, self.one, ),
        #     # 4:
        #     (self.xxx, self.native, ),
        #     (self.xxx, self.npv, ),
        #     (self.xxx, self.param, ),
        #     (self.xxx, self.param_mut, ),
        #     # 8:
        #     (self.xxx, self.var, ),
        #     (self.xxx, self.mon_native, ),
        #     (self.xxx, self.mon_param, ),
        #     (self.xxx, self.mon_npv, ),
        #     # 12:
        #     (self.xxx, self.linear, ),
        #     (self.xxx, self.sum, ),
        #     (self.xxx, self.other, ),
        #     (self.xxx, self.mutable_l0, ),
        #     # 16:
        #     (self.xxx, self.mutable_l1, ),
        #     (self.xxx, self.mutable_l2, ),
        # ]
        # self._run_cases(tests, operator.mul)

    def _run_cases(self, tests, op):
        ans = None
        try:
            for test_num, test in enumerate(tests):
                args = test[:-1]
                result = test[-1]
                orig_args = [clone_expression(arg) for arg in args]
                try:
                    mutable = [isinstance(arg, _MutableSumExpression) for arg in args]
                    classes = [arg.__class__ for arg in args]
                    ans = op(*args)
                    assertExpressionsEqual(self, result, ans)
                    for i, arg in enumerate(args):
                        self.assertFalse(isinstance(arg, _MutableSumExpression))
                        if mutable[i]:
                            self.assertIsNot(arg.__class__, classes[i])
                        else:
                            assertExpressionsEqual(self, orig_args[i], arg)
                            self.assertIs(arg.__class__, classes[i])
                except TypeError:
                    if result is not NotImplemented and result is not TypeError:
                        raise
                finally:
                    for i, arg in enumerate(args):
                        if mutable[i]:
                            arg.__class__ = classes[i]
        except:
            logger.error(
                f"Failed test {test_num}:\n\t"
                + '\n\t'.join(f'{arg}  ({arg.__class__.__name__})' for arg in test)
                + f'\n\t{ans} (result: {ans.__class__.__name__})'
            )
            raise

    def test_product_invalid(self):
        tests = [
            # "invalid(str) * {invalid(str), 0, 1, native}" should never
            # hit the Pyomo expression system
            #
            # (self.invalid, self.invalid, NotImplemented),
            (self.invalid, self.asbinary, NotImplemented),
            # (self.invalid, self.zero, NotImplemented),
            # (self.invalid, self.one, NotImplemented),
            # 4:
            # (self.invalid, self.native, NotImplemented),
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
        ]
        self._run_cases(tests, operator.mul)

    def test_product_asbinary(self):
        tests = [
            (self.asbinary, self.invalid, NotImplemented),
            # BooleanVar objects do not support multiplication
            (self.asbinary, self.asbinary, TypeError),
            (self.asbinary, self.zero, MonomialTermExpression((0, self.bin))),
            (self.asbinary, self.one, self.bin),
            # 4:
            (
                self.asbinary,
                self.native,
                MonomialTermExpression((self.native, self.bin)),
            ),
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
            (self.asbinary, self.mutable_l0, MonomialTermExpression((0, self.bin))),
            # 16:
            (
                self.asbinary,
                self.mutable_l1,
                ProductExpression((self.bin, self.mutable_l1.arg(0))),
            ),
            (
                self.asbinary,
                self.mutable_l2,
                ProductExpression((self.bin, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.mul)

    def test_product_zero(self):
        tests = [
            # "Zero * invalid(str)" (i.e., Pyomo doesn't support it)
            # should never hit the Pyomo expression system:
            #
            # (self.zero, self.invalid, NotImplemented),
            (self.zero, self.asbinary, MonomialTermExpression((0, self.bin))),
            (self.zero, self.zero, 0),
            (self.zero, self.one, 0),
            # 4:
            (self.zero, self.native, 0),
            (self.zero, self.npv, NPV_ProductExpression((0, self.npv))),
            (self.zero, self.param, 0),
            (self.zero, self.param_mut, NPV_ProductExpression((0, self.param_mut))),
            # 8:
            (self.zero, self.var, MonomialTermExpression((0, self.var))),
            (
                self.zero,
                self.mon_native,
                MonomialTermExpression((0, self.mon_native.arg(1))),
            ),
            (
                self.zero,
                self.mon_param,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((0, self.mon_param.arg(0))),
                        self.mon_param.arg(1),
                    )
                ),
            ),
            (
                self.zero,
                self.mon_npv,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((0, self.mon_npv.arg(0))),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            # 12:
            (self.zero, self.linear, ProductExpression((0, self.linear))),
            (self.zero, self.sum, ProductExpression((0, self.sum))),
            (self.zero, self.other, ProductExpression((0, self.other))),
            (self.zero, self.mutable_l0, 0),
            # 16:
            (
                self.zero,
                self.mutable_l1,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((0, self.mutable_l1.arg(0).arg(0))),
                        self.mutable_l1.arg(0).arg(1),
                    )
                ),
            ),
            (self.zero, self.mutable_l2, ProductExpression((0, self.mutable_l2))),
        ]
        self._run_cases(tests, operator.mul)

    def test_product_one(self):
        tests = [
            # "One * invalid(str)" (i.e., Pyomo doesn't support it)
            # should never hit the Pyomo expression system:
            #
            # (self.one, self.invalid, NotImplemented),
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
            (self.one, self.mutable_l1, self.mutable_l1.arg(0)),
            (self.one, self.mutable_l2, self.mutable_l2),
        ]
        self._run_cases(tests, operator.mul)

    def test_product_native(self):
        tests = [
            # "Native * invalid(str) (i.e., Pyomo doesn't support it)
            # should never hit the Pyomo expression system:
            #
            # (self.native, self.invalid, NotImplemented),
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
                        NPV_ProductExpression((5, self.mutable_l1.arg(0).arg(0))),
                        self.mutable_l1.arg(0).arg(1),
                    )
                ),
            ),
            (self.native, self.mutable_l2, ProductExpression((5, self.mutable_l2))),
        ]
        self._run_cases(tests, operator.mul)

    def test_product_npv(self):
        tests = [
            (self.npv, self.invalid, NotImplemented),
            (self.npv, self.asbinary, MonomialTermExpression((self.npv, self.bin))),
            (self.npv, self.zero, NPV_ProductExpression((self.npv, 0))),
            (self.npv, self.one, self.npv),
            # 4:
            (self.npv, self.native, NPV_ProductExpression((self.npv, self.native))),
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
            (self.npv, self.mutable_l0, NPV_ProductExpression((self.npv, 0))),
            # 16:
            (
                self.npv,
                self.mutable_l1,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression(
                            (self.npv, self.mutable_l1.arg(0).arg(0))
                        ),
                        self.mutable_l1.arg(0).arg(1),
                    )
                ),
            ),
            (self.npv, self.mutable_l2, ProductExpression((self.npv, self.mutable_l2))),
        ]
        self._run_cases(tests, operator.mul)

    def test_product_param(self):
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
                        NPV_ProductExpression((6, self.mutable_l1.arg(0).arg(0))),
                        self.mutable_l1.arg(0).arg(1),
                    )
                ),
            ),
            (self.param, self.mutable_l2, ProductExpression((6, self.mutable_l2))),
        ]
        self._run_cases(tests, operator.mul)

    def test_product_param_mut(self):
        tests = [
            (self.param_mut, self.invalid, NotImplemented),
            (
                self.param_mut,
                self.asbinary,
                MonomialTermExpression((self.param_mut, self.bin)),
            ),
            (self.param_mut, self.zero, NPV_ProductExpression((self.param_mut, 0))),
            (self.param_mut, self.one, self.param_mut),
            # 4:
            (
                self.param_mut,
                self.native,
                NPV_ProductExpression((self.param_mut, self.native)),
            ),
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
            (
                self.param_mut,
                self.mutable_l0,
                NPV_ProductExpression((self.param_mut, 0)),
            ),
            # 16:
            (
                self.param_mut,
                self.mutable_l1,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression(
                            (self.param_mut, self.mutable_l1.arg(0).arg(0))
                        ),
                        self.mutable_l1.arg(0).arg(1),
                    )
                ),
            ),
            (
                self.param_mut,
                self.mutable_l2,
                ProductExpression((self.param_mut, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.mul)

    def test_product_var(self):
        tests = [
            (self.var, self.invalid, NotImplemented),
            (self.var, self.asbinary, ProductExpression((self.var, self.bin))),
            (self.var, self.zero, MonomialTermExpression((0, self.var))),
            (self.var, self.one, self.var),
            # 4:
            (self.var, self.native, MonomialTermExpression((self.native, self.var))),
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
            (self.var, self.mutable_l0, MonomialTermExpression((0, self.var))),
            # 16:
            (
                self.var,
                self.mutable_l1,
                ProductExpression((self.var, self.mutable_l1.arg(0))),
            ),
            (self.var, self.mutable_l2, ProductExpression((self.var, self.mutable_l2))),
        ]
        self._run_cases(tests, operator.mul)

    def test_product_mon_native(self):
        tests = [
            (self.mon_native, self.invalid, NotImplemented),
            (
                self.mon_native,
                self.asbinary,
                ProductExpression((self.mon_native, self.bin)),
            ),
            (
                self.mon_native,
                self.zero,
                MonomialTermExpression((0, self.mon_native.arg(1))),
            ),
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
            (
                self.mon_native,
                self.mutable_l0,
                MonomialTermExpression((0, self.mon_native.arg(1))),
            ),
            # 16:
            (
                self.mon_native,
                self.mutable_l1,
                ProductExpression((self.mon_native, self.mutable_l1.arg(0))),
            ),
            (
                self.mon_native,
                self.mutable_l2,
                ProductExpression((self.mon_native, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.mul)

    def test_product_mon_param(self):
        tests = [
            (self.mon_param, self.invalid, NotImplemented),
            (
                self.mon_param,
                self.asbinary,
                ProductExpression((self.mon_param, self.bin)),
            ),
            (
                self.mon_param,
                self.zero,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_param.arg(0), 0)),
                        self.mon_param.arg(1),
                    )
                ),
            ),
            (self.mon_param, self.one, self.mon_param),
            # 4:
            (
                self.mon_param,
                self.native,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_param.arg(0), self.native)),
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
            (
                self.mon_param,
                self.mutable_l0,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_param.arg(0), 0)),
                        self.mon_param.arg(1),
                    )
                ),
            ),
            # 16:
            (
                self.mon_param,
                self.mutable_l1,
                ProductExpression((self.mon_param, self.mutable_l1.arg(0))),
            ),
            (
                self.mon_param,
                self.mutable_l2,
                ProductExpression((self.mon_param, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.mul)

    def test_product_mon_npv(self):
        tests = [
            (self.mon_npv, self.invalid, NotImplemented),
            (self.mon_npv, self.asbinary, ProductExpression((self.mon_npv, self.bin))),
            (
                self.mon_npv,
                self.zero,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_npv.arg(0), 0)),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            (self.mon_npv, self.one, self.mon_npv),
            # 4:
            (
                self.mon_npv,
                self.native,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_npv.arg(0), self.native)),
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
            (
                self.mon_npv,
                self.mutable_l0,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_npv.arg(0), 0)),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            # 16:
            (
                self.mon_npv,
                self.mutable_l1,
                ProductExpression((self.mon_npv, self.mutable_l1.arg(0))),
            ),
            (
                self.mon_npv,
                self.mutable_l2,
                ProductExpression((self.mon_npv, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.mul)

    def test_product_linear(self):
        tests = [
            (self.linear, self.invalid, NotImplemented),
            (self.linear, self.asbinary, ProductExpression((self.linear, self.bin))),
            (self.linear, self.zero, ProductExpression((self.linear, self.zero))),
            (self.linear, self.one, self.linear),
            # 4:
            (self.linear, self.native, ProductExpression((self.linear, self.native))),
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
            (self.linear, self.mutable_l0, ProductExpression((self.linear, 0))),
            # 16:
            (
                self.linear,
                self.mutable_l1,
                ProductExpression((self.linear, self.mutable_l1.arg(0))),
            ),
            (
                self.linear,
                self.mutable_l2,
                ProductExpression((self.linear, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.mul)

    def test_product_sum(self):
        tests = [
            (self.sum, self.invalid, NotImplemented),
            (self.sum, self.asbinary, ProductExpression((self.sum, self.bin))),
            (self.sum, self.zero, ProductExpression((self.sum, self.zero))),
            (self.sum, self.one, self.sum),
            # 4:
            (self.sum, self.native, ProductExpression((self.sum, self.native))),
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
            (self.sum, self.mutable_l0, ProductExpression((self.sum, 0))),
            # 16:
            (
                self.sum,
                self.mutable_l1,
                ProductExpression((self.sum, self.mutable_l1.arg(0))),
            ),
            (self.sum, self.mutable_l2, ProductExpression((self.sum, self.mutable_l2))),
        ]
        self._run_cases(tests, operator.mul)

    def test_product_other(self):
        tests = [
            (self.other, self.invalid, NotImplemented),
            (self.other, self.asbinary, ProductExpression((self.other, self.bin))),
            (self.other, self.zero, ProductExpression((self.other, self.zero))),
            (self.other, self.one, self.other),
            # 4:
            (self.other, self.native, ProductExpression((self.other, self.native))),
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
            (self.other, self.mutable_l0, ProductExpression((self.other, 0))),
            # 16:
            (
                self.other,
                self.mutable_l1,
                ProductExpression((self.other, self.mutable_l1.arg(0))),
            ),
            (
                self.other,
                self.mutable_l2,
                ProductExpression((self.other, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.mul)

    def test_product_mutable_l0(self):
        tests = [
            (self.mutable_l0, self.invalid, NotImplemented),
            (self.mutable_l0, self.asbinary, MonomialTermExpression((0, self.bin))),
            (self.mutable_l0, self.zero, 0),
            (self.mutable_l0, self.one, 0),
            # 4:
            (self.mutable_l0, self.native, 0),
            (self.mutable_l0, self.npv, NPV_ProductExpression((0, self.npv))),
            (self.mutable_l0, self.param, 0),
            (
                self.mutable_l0,
                self.param_mut,
                NPV_ProductExpression((0, self.param_mut)),
            ),
            # 8:
            (self.mutable_l0, self.var, MonomialTermExpression((0, self.var))),
            (
                self.mutable_l0,
                self.mon_native,
                MonomialTermExpression((0, self.mon_native.arg(1))),
            ),
            (
                self.mutable_l0,
                self.mon_param,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((0, self.mon_param.arg(0))),
                        self.mon_param.arg(1),
                    )
                ),
            ),
            (
                self.mutable_l0,
                self.mon_npv,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((0, self.mon_npv.arg(0))),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            # 12:
            (self.mutable_l0, self.linear, ProductExpression((0, self.linear))),
            (self.mutable_l0, self.sum, ProductExpression((0, self.sum))),
            (self.mutable_l0, self.other, ProductExpression((0, self.other))),
            (self.mutable_l0, self.mutable_l0, 0),
            # 16:
            (
                self.mutable_l0,
                self.mutable_l1,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((0, self.mon_npv.arg(0))),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            (self.mutable_l0, self.mutable_l2, ProductExpression((0, self.mutable_l2))),
        ]
        self._run_cases(tests, operator.mul)

    def test_product_mutable_l1(self):
        tests = [
            (self.mutable_l1, self.invalid, NotImplemented),
            (
                self.mutable_l1,
                self.asbinary,
                ProductExpression((self.mon_npv, self.bin)),
            ),
            (
                self.mutable_l1,
                self.zero,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_npv.arg(0), 0)),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            (self.mutable_l1, self.one, self.mon_npv),
            # 4:
            (
                self.mutable_l1,
                self.native,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_npv.arg(0), self.native)),
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
            (
                self.mutable_l1,
                self.var,
                ProductExpression((self.mutable_l1.arg(0), self.var)),
            ),
            (
                self.mutable_l1,
                self.mon_native,
                ProductExpression((self.mutable_l1.arg(0), self.mon_native)),
            ),
            (
                self.mutable_l1,
                self.mon_param,
                ProductExpression((self.mutable_l1.arg(0), self.mon_param)),
            ),
            (
                self.mutable_l1,
                self.mon_npv,
                ProductExpression((self.mutable_l1.arg(0), self.mon_npv)),
            ),
            # 12:
            (
                self.mutable_l1,
                self.linear,
                ProductExpression((self.mutable_l1.arg(0), self.linear)),
            ),
            (
                self.mutable_l1,
                self.sum,
                ProductExpression((self.mutable_l1.arg(0), self.sum)),
            ),
            (
                self.mutable_l1,
                self.other,
                ProductExpression((self.mutable_l1.arg(0), self.other)),
            ),
            (
                self.mutable_l1,
                self.mutable_l0,
                MonomialTermExpression(
                    (
                        NPV_ProductExpression((self.mon_npv.arg(0), 0)),
                        self.mon_npv.arg(1),
                    )
                ),
            ),
            # 16:
            (
                self.mutable_l1,
                self.mutable_l1,
                ProductExpression((self.mon_npv, self.mon_npv)),
            ),
            (
                self.mutable_l1,
                self.mutable_l2,
                ProductExpression((self.mon_npv, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.mul)

    def test_product_mutable_l2(self):
        tests = [
            (self.mutable_l2, self.invalid, NotImplemented),
            (
                self.mutable_l2,
                self.asbinary,
                ProductExpression((self.mutable_l2, self.bin)),
            ),
            (self.mutable_l2, self.zero, ProductExpression((self.mutable_l2, 0))),
            (self.mutable_l2, self.one, self.mutable_l2),
            # 4:
            (
                self.mutable_l2,
                self.native,
                ProductExpression((self.mutable_l2, self.native)),
            ),
            (self.mutable_l2, self.npv, ProductExpression((self.mutable_l2, self.npv))),
            (self.mutable_l2, self.param, ProductExpression((self.mutable_l2, 6))),
            (
                self.mutable_l2,
                self.param_mut,
                ProductExpression((self.mutable_l2, self.param_mut)),
            ),
            # 8:
            (self.mutable_l2, self.var, ProductExpression((self.mutable_l2, self.var))),
            (
                self.mutable_l2,
                self.mon_native,
                ProductExpression((self.mutable_l2, self.mon_native)),
            ),
            (
                self.mutable_l2,
                self.mon_param,
                ProductExpression((self.mutable_l2, self.mon_param)),
            ),
            (
                self.mutable_l2,
                self.mon_npv,
                ProductExpression((self.mutable_l2, self.mon_npv)),
            ),
            # 12:
            (
                self.mutable_l2,
                self.linear,
                ProductExpression((self.mutable_l2, self.linear)),
            ),
            (self.mutable_l2, self.sum, ProductExpression((self.mutable_l2, self.sum))),
            (
                self.mutable_l2,
                self.other,
                ProductExpression((self.mutable_l2, self.other)),
            ),
            (self.mutable_l2, self.mutable_l0, ProductExpression((self.mutable_l2, 0))),
            # 16:
            (
                self.mutable_l2,
                self.mutable_l1,
                ProductExpression((self.mutable_l2, self.mon_npv)),
            ),
            (
                self.mutable_l2,
                self.mutable_l2,
                ProductExpression((self.mutable_l2, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.mul)
