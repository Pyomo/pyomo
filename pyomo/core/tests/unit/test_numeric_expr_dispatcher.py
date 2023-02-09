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

from pyomo.common.log import LoggingIntercept
from pyomo.core.expr.compare import (
    assertExpressionsEqual,
    assertExpressionsStructurallyEqual,
)
from pyomo.core.expr.current import (
    DivisionExpression,
    NPV_DivisionExpression,
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
        try:
            for test_num, test in enumerate(tests):
                ans = None
                args = test[:-1]
                result = test[-1]
                orig_args = [clone_expression(arg) for arg in args]
                try:
                    mutable = [isinstance(arg, _MutableSumExpression) for arg in args]
                    classes = [arg.__class__ for arg in args]
                    with LoggingIntercept() as LOG:
                        ans = op(*args)
                    if not any(arg is self.asbinary for arg in args):
                        self.assertEqual(LOG.getvalue(), "")
                    assertExpressionsEqual(self, result, ans)
                    for i, arg in enumerate(args):
                        self.assertFalse(isinstance(arg, _MutableSumExpression))
                        if mutable[i]:
                            self.assertIsNot(arg.__class__, classes[i])
                        else:
                            assertExpressionsEqual(self, orig_args[i], arg)
                            self.assertIs(arg.__class__, classes[i])
                except TypeError:
                    if result is not NotImplemented:
                        raise
                except ZeroDivisionError:
                    if result is not ZeroDivisionError:
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

    def test_mul_invalid(self):
        tests = [
            (self.invalid, self.invalid, NotImplemented),
            (self.invalid, self.asbinary, NotImplemented),
            # "invalid(str) * {0, 1, native}" are legitimate Python
            # operations and should never hit the Pyomo expression
            # system
            #
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

    #
    #
    # MULTIPLICATION
    #
    #

    def test_mul_asbinary(self):
        tests = [
            (self.asbinary, self.invalid, NotImplemented),
            # BooleanVar objects do not support multiplication
            (self.asbinary, self.asbinary, NotImplemented),
            (self.asbinary, self.zero, MonomialTermExpression((0, self.bin))),
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

    def test_mul_zero(self):
        tests = [
            # "Zero * invalid(str)" is a legitimate Python operation and
            # should never hit the Pyomo expression system
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

    def test_mul_one(self):
        tests = [
            # "One * invalid(str)" is a legitimate Python operation and
            # should never hit the Pyomo expression system
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

    def test_mul_native(self):
        tests = [
            # "Native * invalid(str) is a legitimate Python operation and
            # should never hit the Pyomo expression system
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

    def test_mul_npv(self):
        tests = [
            (self.npv, self.invalid, NotImplemented),
            (self.npv, self.asbinary, MonomialTermExpression((self.npv, self.bin))),
            (self.npv, self.zero, NPV_ProductExpression((self.npv, 0))),
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
                        NPV_ProductExpression((6, self.mutable_l1.arg(0).arg(0))),
                        self.mutable_l1.arg(0).arg(1),
                    )
                ),
            ),
            (self.param, self.mutable_l2, ProductExpression((6, self.mutable_l2))),
        ]
        self._run_cases(tests, operator.mul)

    def test_mul_param_mut(self):
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

    def test_mul_var(self):
        tests = [
            (self.var, self.invalid, NotImplemented),
            (self.var, self.asbinary, ProductExpression((self.var, self.bin))),
            (self.var, self.zero, MonomialTermExpression((0, self.var))),
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

    def test_mul_mon_native(self):
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

    def test_mul_mon_param(self):
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

    def test_mul_mon_npv(self):
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

    def test_mul_linear(self):
        tests = [
            (self.linear, self.invalid, NotImplemented),
            (self.linear, self.asbinary, ProductExpression((self.linear, self.bin))),
            (self.linear, self.zero, ProductExpression((self.linear, self.zero))),
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

    def test_mul_sum(self):
        tests = [
            (self.sum, self.invalid, NotImplemented),
            (self.sum, self.asbinary, ProductExpression((self.sum, self.bin))),
            (self.sum, self.zero, ProductExpression((self.sum, self.zero))),
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

    def test_mul_other(self):
        tests = [
            (self.other, self.invalid, NotImplemented),
            (self.other, self.asbinary, ProductExpression((self.other, self.bin))),
            (self.other, self.zero, ProductExpression((self.other, self.zero))),
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

    def test_mul_mutable_l0(self):
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

    def test_mul_mutable_l1(self):
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

    def test_mul_mutable_l2(self):
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
            (self.mutable_l2, self.native, ProductExpression((self.mutable_l2, 5))),
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
        ]
        self._run_cases(tests, operator.truediv)

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
                DivisionExpression((self.bin, self.mutable_l1.arg(0))),
            ),
            (
                self.asbinary,
                self.mutable_l2,
                DivisionExpression((self.bin, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.truediv)

    def test_div_zero(self):
        tests = [
            (self.zero, self.invalid, NotImplemented),
            (self.zero, self.asbinary, DivisionExpression((0, self.bin))),
            (self.zero, self.zero, ZeroDivisionError),
            (self.zero, self.one, 0.0),
            # 4:
            (self.zero, self.native, 0.0),
            (self.zero, self.npv, NPV_DivisionExpression((0, self.npv))),
            (self.zero, self.param, 0.0),
            (self.zero, self.param_mut, NPV_DivisionExpression((0, self.param_mut))),
            # 8:
            (self.zero, self.var, DivisionExpression((0, self.var))),
            (self.zero, self.mon_native, DivisionExpression((0, self.mon_native))),
            (self.zero, self.mon_param, DivisionExpression((0, self.mon_param))),
            (self.zero, self.mon_npv, DivisionExpression((0, self.mon_npv))),
            # 12:
            (self.zero, self.linear, DivisionExpression((0, self.linear))),
            (self.zero, self.sum, DivisionExpression((0, self.sum))),
            (self.zero, self.other, DivisionExpression((0, self.other))),
            (self.zero, self.mutable_l0, ZeroDivisionError),
            # 16:
            (
                self.zero,
                self.mutable_l1,
                DivisionExpression((0, self.mutable_l1.arg(0))),
            ),
            (self.zero, self.mutable_l2, DivisionExpression((0, self.mutable_l2))),
        ]
        self._run_cases(tests, operator.truediv)

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
            (
                self.one,
                self.mutable_l1,
                DivisionExpression((1, self.mutable_l1.arg(0))),
            ),
            (self.one, self.mutable_l2, DivisionExpression((1, self.mutable_l2))),
        ]
        self._run_cases(tests, operator.truediv)

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
            (
                self.native,
                self.mutable_l1,
                DivisionExpression((5, self.mutable_l1.arg(0))),
            ),
            (self.native, self.mutable_l2, DivisionExpression((5, self.mutable_l2))),
        ]
        self._run_cases(tests, operator.truediv)

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
            (
                self.npv,
                self.mutable_l1,
                DivisionExpression((self.npv, self.mutable_l1.arg(0))),
            ),
            (
                self.npv,
                self.mutable_l2,
                DivisionExpression((self.npv, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.truediv)

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
            (
                self.param,
                self.mutable_l1,
                DivisionExpression((6, self.mutable_l1.arg(0))),
            ),
            (self.param, self.mutable_l2, DivisionExpression((6, self.mutable_l2))),
        ]
        self._run_cases(tests, operator.truediv)

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
            (
                self.param_mut,
                self.native,
                NPV_DivisionExpression((self.param_mut, self.native)),
            ),
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
                DivisionExpression((self.param_mut, self.mutable_l1.arg(0))),
            ),
            (
                self.param_mut,
                self.mutable_l2,
                DivisionExpression((self.param_mut, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.truediv)

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
            (
                self.var,
                self.mutable_l1,
                DivisionExpression((self.var, self.mutable_l1.arg(0))),
            ),
            (
                self.var,
                self.mutable_l2,
                DivisionExpression((self.var, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.truediv)

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
                DivisionExpression((self.mon_native, self.mutable_l1.arg(0))),
            ),
            (
                self.mon_native,
                self.mutable_l2,
                DivisionExpression((self.mon_native, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.truediv)

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
                        NPV_DivisionExpression((self.mon_param.arg(0), self.native)),
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
                DivisionExpression((self.mon_param, self.mutable_l1.arg(0))),
            ),
            (
                self.mon_param,
                self.mutable_l2,
                DivisionExpression((self.mon_param, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.truediv)

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
                        NPV_DivisionExpression((self.mon_npv.arg(0), self.native)),
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
                DivisionExpression((self.mon_npv, self.mutable_l1.arg(0))),
            ),
            (
                self.mon_npv,
                self.mutable_l2,
                DivisionExpression((self.mon_npv, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.truediv)

    def test_div_linear(self):
        tests = [
            (self.linear, self.invalid, NotImplemented),
            (self.linear, self.asbinary, DivisionExpression((self.linear, self.bin))),
            (self.linear, self.zero, ZeroDivisionError),
            (self.linear, self.one, self.linear),
            # 4:
            (self.linear, self.native, DivisionExpression((self.linear, self.native))),
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
                DivisionExpression((self.linear, self.mutable_l1.arg(0))),
            ),
            (
                self.linear,
                self.mutable_l2,
                DivisionExpression((self.linear, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.truediv)

    def test_div_sum(self):
        tests = [
            (self.sum, self.invalid, NotImplemented),
            (self.sum, self.asbinary, DivisionExpression((self.sum, self.bin))),
            (self.sum, self.zero, ZeroDivisionError),
            (self.sum, self.one, self.sum),
            # 4:
            (self.sum, self.native, DivisionExpression((self.sum, self.native))),
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
            (
                self.sum,
                self.mutable_l1,
                DivisionExpression((self.sum, self.mutable_l1.arg(0))),
            ),
            (
                self.sum,
                self.mutable_l2,
                DivisionExpression((self.sum, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.truediv)

    def test_div_other(self):
        tests = [
            (self.other, self.invalid, NotImplemented),
            (self.other, self.asbinary, DivisionExpression((self.other, self.bin))),
            (self.other, self.zero, ZeroDivisionError),
            (self.other, self.one, self.other),
            # 4:
            (self.other, self.native, DivisionExpression((self.other, self.native))),
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
                DivisionExpression((self.other, self.mutable_l1.arg(0))),
            ),
            (
                self.other,
                self.mutable_l2,
                DivisionExpression((self.other, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.truediv)

    def test_div_mutable_l0(self):
        tests = [
            (self.mutable_l0, self.invalid, NotImplemented),
            (self.mutable_l0, self.asbinary, DivisionExpression((0, self.bin))),
            (self.mutable_l0, self.zero, ZeroDivisionError),
            (self.mutable_l0, self.one, 0.0),
            # 4:
            (self.mutable_l0, self.native, 0.0),
            (self.mutable_l0, self.npv, NPV_DivisionExpression((0, self.npv))),
            (self.mutable_l0, self.param, 0.0),
            (
                self.mutable_l0,
                self.param_mut,
                NPV_DivisionExpression((0, self.param_mut)),
            ),
            # 8:
            (self.mutable_l0, self.var, DivisionExpression((0, self.var))),
            (
                self.mutable_l0,
                self.mon_native,
                DivisionExpression((0, self.mon_native)),
            ),
            (self.mutable_l0, self.mon_param, DivisionExpression((0, self.mon_param))),
            (self.mutable_l0, self.mon_npv, DivisionExpression((0, self.mon_npv))),
            # 12:
            (self.mutable_l0, self.linear, DivisionExpression((0, self.linear))),
            (self.mutable_l0, self.sum, DivisionExpression((0, self.sum))),
            (self.mutable_l0, self.other, DivisionExpression((0, self.other))),
            (self.mutable_l0, self.mutable_l0, ZeroDivisionError),
            # 16:
            (
                self.mutable_l0,
                self.mutable_l1,
                DivisionExpression((0, self.mutable_l1.arg(0))),
            ),
            (
                self.mutable_l0,
                self.mutable_l2,
                DivisionExpression((0, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.truediv)

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
            (
                self.mutable_l1,
                self.var,
                DivisionExpression((self.mutable_l1.arg(0), self.var)),
            ),
            (
                self.mutable_l1,
                self.mon_native,
                DivisionExpression((self.mutable_l1.arg(0), self.mon_native)),
            ),
            (
                self.mutable_l1,
                self.mon_param,
                DivisionExpression((self.mutable_l1.arg(0), self.mon_param)),
            ),
            (
                self.mutable_l1,
                self.mon_npv,
                DivisionExpression((self.mutable_l1.arg(0), self.mon_npv)),
            ),
            # 12:
            (
                self.mutable_l1,
                self.linear,
                DivisionExpression((self.mutable_l1.arg(0), self.linear)),
            ),
            (
                self.mutable_l1,
                self.sum,
                DivisionExpression((self.mutable_l1.arg(0), self.sum)),
            ),
            (
                self.mutable_l1,
                self.other,
                DivisionExpression((self.mutable_l1.arg(0), self.other)),
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
                DivisionExpression((self.mon_npv, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.truediv)

    def test_div_mutable_l2(self):
        tests = [
            (self.mutable_l2, self.invalid, NotImplemented),
            (
                self.mutable_l2,
                self.asbinary,
                DivisionExpression((self.mutable_l2, self.bin)),
            ),
            (self.mutable_l2, self.zero, ZeroDivisionError),
            (self.mutable_l2, self.one, self.mutable_l2),
            # 4:
            (
                self.mutable_l2,
                self.native,
                DivisionExpression((self.mutable_l2, self.native)),
            ),
            (
                self.mutable_l2,
                self.npv,
                DivisionExpression((self.mutable_l2, self.npv)),
            ),
            (self.mutable_l2, self.param, DivisionExpression((self.mutable_l2, 6))),
            (
                self.mutable_l2,
                self.param_mut,
                DivisionExpression((self.mutable_l2, self.param_mut)),
            ),
            # 8:
            (
                self.mutable_l2,
                self.var,
                DivisionExpression((self.mutable_l2, self.var)),
            ),
            (
                self.mutable_l2,
                self.mon_native,
                DivisionExpression((self.mutable_l2, self.mon_native)),
            ),
            (
                self.mutable_l2,
                self.mon_param,
                DivisionExpression((self.mutable_l2, self.mon_param)),
            ),
            (
                self.mutable_l2,
                self.mon_npv,
                DivisionExpression((self.mutable_l2, self.mon_npv)),
            ),
            # 12:
            (
                self.mutable_l2,
                self.linear,
                DivisionExpression((self.mutable_l2, self.linear)),
            ),
            (
                self.mutable_l2,
                self.sum,
                DivisionExpression((self.mutable_l2, self.sum)),
            ),
            (
                self.mutable_l2,
                self.other,
                DivisionExpression((self.mutable_l2, self.other)),
            ),
            (self.mutable_l2, self.mutable_l0, ZeroDivisionError),
            # 16:
            (
                self.mutable_l2,
                self.mutable_l1,
                DivisionExpression((self.mutable_l2, self.mon_npv)),
            ),
            (
                self.mutable_l2,
                self.mutable_l2,
                DivisionExpression((self.mutable_l2, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.truediv)

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
        ]
        self._run_cases(tests, operator.pow)

    def test_pow_asbinary(self):
        tests = [
            (self.asbinary, self.invalid, NotImplemented),
            # BooleanVar objects do not support division
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
            (
                self.asbinary,
                self.mutable_l1,
                PowExpression((self.bin, self.mutable_l1.arg(0))),
            ),
            (
                self.asbinary,
                self.mutable_l2,
                PowExpression((self.bin, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.pow)

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
            (self.zero, self.mutable_l1, PowExpression((0, self.mutable_l1.arg(0)))),
            (self.zero, self.mutable_l2, PowExpression((0, self.mutable_l2))),
        ]
        self._run_cases(tests, operator.pow)

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
            (self.one, self.mutable_l1, PowExpression((1, self.mutable_l1.arg(0)))),
            (self.one, self.mutable_l2, PowExpression((1, self.mutable_l2))),
        ]
        self._run_cases(tests, operator.pow)

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
            (self.native, self.mutable_l1, PowExpression((5, self.mutable_l1.arg(0)))),
            (self.native, self.mutable_l2, PowExpression((5, self.mutable_l2))),
        ]
        self._run_cases(tests, operator.pow)

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
            (
                self.npv,
                self.mutable_l1,
                PowExpression((self.npv, self.mutable_l1.arg(0))),
            ),
            (self.npv, self.mutable_l2, PowExpression((self.npv, self.mutable_l2))),
        ]
        self._run_cases(tests, operator.pow)

    def test_pow_param(self):
        tests = [
            (self.param, self.invalid, NotImplemented),
            (self.param, self.asbinary, PowExpression((6, self.bin))),
            (self.param, self.zero, 1),
            (self.param, self.one, self.param),
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
            (self.param, self.mutable_l1, PowExpression((6, self.mutable_l1.arg(0)))),
            (self.param, self.mutable_l2, PowExpression((6, self.mutable_l2))),
        ]
        self._run_cases(tests, operator.pow)

    def test_pow_param_mut(self):
        tests = [
            (self.param_mut, self.invalid, NotImplemented),
            (self.param_mut, self.asbinary, PowExpression((self.param_mut, self.bin))),
            (self.param_mut, self.zero, 1),
            (self.param_mut, self.one, self.param_mut),
            # 4:
            (
                self.param_mut,
                self.native,
                NPV_PowExpression((self.param_mut, self.native)),
            ),
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
                PowExpression((self.param_mut, self.mutable_l1.arg(0))),
            ),
            (
                self.param_mut,
                self.mutable_l2,
                PowExpression((self.param_mut, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.pow)

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
            (
                self.var,
                self.mutable_l1,
                PowExpression((self.var, self.mutable_l1.arg(0))),
            ),
            (self.var, self.mutable_l2, PowExpression((self.var, self.mutable_l2))),
        ]
        self._run_cases(tests, operator.pow)

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
                PowExpression((self.mon_native, self.mutable_l1.arg(0))),
            ),
            (
                self.mon_native,
                self.mutable_l2,
                PowExpression((self.mon_native, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.pow)

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
                PowExpression((self.mon_param, self.mutable_l1.arg(0))),
            ),
            (
                self.mon_param,
                self.mutable_l2,
                PowExpression((self.mon_param, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.pow)

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
                PowExpression((self.mon_npv, self.mutable_l1.arg(0))),
            ),
            (
                self.mon_npv,
                self.mutable_l2,
                PowExpression((self.mon_npv, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.pow)

    def test_pow_linear(self):
        tests = [
            (self.linear, self.invalid, NotImplemented),
            (self.linear, self.asbinary, PowExpression((self.linear, self.bin))),
            (self.linear, self.zero, 1),
            (self.linear, self.one, self.linear),
            # 4:
            (self.linear, self.native, PowExpression((self.linear, self.native))),
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
            (
                self.linear,
                self.mutable_l1,
                PowExpression((self.linear, self.mutable_l1.arg(0))),
            ),
            (
                self.linear,
                self.mutable_l2,
                PowExpression((self.linear, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.pow)

    def test_pow_sum(self):
        tests = [
            (self.sum, self.invalid, NotImplemented),
            (self.sum, self.asbinary, PowExpression((self.sum, self.bin))),
            (self.sum, self.zero, 1),
            (self.sum, self.one, self.sum),
            # 4:
            (self.sum, self.native, PowExpression((self.sum, self.native))),
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
            (
                self.sum,
                self.mutable_l1,
                PowExpression((self.sum, self.mutable_l1.arg(0))),
            ),
            (self.sum, self.mutable_l2, PowExpression((self.sum, self.mutable_l2))),
        ]
        self._run_cases(tests, operator.pow)

    def test_pow_other(self):
        tests = [
            (self.other, self.invalid, NotImplemented),
            (self.other, self.asbinary, PowExpression((self.other, self.bin))),
            (self.other, self.zero, 1),
            (self.other, self.one, self.other),
            # 4:
            (self.other, self.native, PowExpression((self.other, self.native))),
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
            (
                self.other,
                self.mutable_l1,
                PowExpression((self.other, self.mutable_l1.arg(0))),
            ),
            (self.other, self.mutable_l2, PowExpression((self.other, self.mutable_l2))),
        ]
        self._run_cases(tests, operator.pow)

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
            (
                self.mutable_l0,
                self.mutable_l1,
                PowExpression((0, self.mutable_l1.arg(0))),
            ),
            (self.mutable_l0, self.mutable_l2, PowExpression((0, self.mutable_l2))),
        ]
        self._run_cases(tests, operator.pow)

    def test_pow_mutable_l1(self):
        tests = [
            (self.mutable_l1, self.invalid, NotImplemented),
            (self.mutable_l1, self.asbinary, PowExpression((self.mon_npv, self.bin))),
            (self.mutable_l1, self.zero, 1),
            (self.mutable_l1, self.one, self.mon_npv),
            # 4:
            (self.mutable_l1, self.native, PowExpression((self.mutable_l1.arg(0), 5))),
            (
                self.mutable_l1,
                self.npv,
                PowExpression((self.mutable_l1.arg(0), self.npv)),
            ),
            (self.mutable_l1, self.param, PowExpression((self.mutable_l1.arg(0), 6))),
            (
                self.mutable_l1,
                self.param_mut,
                PowExpression((self.mutable_l1.arg(0), self.param_mut)),
            ),
            # 8:
            (
                self.mutable_l1,
                self.var,
                PowExpression((self.mutable_l1.arg(0), self.var)),
            ),
            (
                self.mutable_l1,
                self.mon_native,
                PowExpression((self.mutable_l1.arg(0), self.mon_native)),
            ),
            (
                self.mutable_l1,
                self.mon_param,
                PowExpression((self.mutable_l1.arg(0), self.mon_param)),
            ),
            (
                self.mutable_l1,
                self.mon_npv,
                PowExpression((self.mutable_l1.arg(0), self.mon_npv)),
            ),
            # 12:
            (
                self.mutable_l1,
                self.linear,
                PowExpression((self.mutable_l1.arg(0), self.linear)),
            ),
            (
                self.mutable_l1,
                self.sum,
                PowExpression((self.mutable_l1.arg(0), self.sum)),
            ),
            (
                self.mutable_l1,
                self.other,
                PowExpression((self.mutable_l1.arg(0), self.other)),
            ),
            (self.mutable_l1, self.mutable_l0, 1),
            # 16:
            (
                self.mutable_l1,
                self.mutable_l1,
                PowExpression((self.mon_npv, self.mon_npv)),
            ),
            (
                self.mutable_l1,
                self.mutable_l2,
                PowExpression((self.mon_npv, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.pow)

    def test_pow_mutable_l2(self):
        tests = [
            (self.mutable_l2, self.invalid, NotImplemented),
            (
                self.mutable_l2,
                self.asbinary,
                PowExpression((self.mutable_l2, self.bin)),
            ),
            (self.mutable_l2, self.zero, 1),
            (self.mutable_l2, self.one, self.mutable_l2),
            # 4:
            (
                self.mutable_l2,
                self.native,
                PowExpression((self.mutable_l2, self.native)),
            ),
            (self.mutable_l2, self.npv, PowExpression((self.mutable_l2, self.npv))),
            (self.mutable_l2, self.param, PowExpression((self.mutable_l2, 6))),
            (
                self.mutable_l2,
                self.param_mut,
                PowExpression((self.mutable_l2, self.param_mut)),
            ),
            # 8:
            (self.mutable_l2, self.var, PowExpression((self.mutable_l2, self.var))),
            (
                self.mutable_l2,
                self.mon_native,
                PowExpression((self.mutable_l2, self.mon_native)),
            ),
            (
                self.mutable_l2,
                self.mon_param,
                PowExpression((self.mutable_l2, self.mon_param)),
            ),
            (
                self.mutable_l2,
                self.mon_npv,
                PowExpression((self.mutable_l2, self.mon_npv)),
            ),
            # 12:
            (
                self.mutable_l2,
                self.linear,
                PowExpression((self.mutable_l2, self.linear)),
            ),
            (self.mutable_l2, self.sum, PowExpression((self.mutable_l2, self.sum))),
            (self.mutable_l2, self.other, PowExpression((self.mutable_l2, self.other))),
            (self.mutable_l2, self.mutable_l0, 1),
            # 16:
            (
                self.mutable_l2,
                self.mutable_l1,
                PowExpression((self.mutable_l2, self.mon_npv)),
            ),
            (
                self.mutable_l2,
                self.mutable_l2,
                PowExpression((self.mutable_l2, self.mutable_l2)),
            ),
        ]
        self._run_cases(tests, operator.pow)
