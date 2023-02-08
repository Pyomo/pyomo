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
from pyomo.environ import ConcreteModel, Param, Var, BooleanVar, value
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

        # (self.xxx, self.mutable_l0, ),
        # (self.xxx, self.mutable_l1, ),
        # (self.xxx, self.mutable_l2, ),
        # (self.xxx, self.asbinary, ),
        # (self.xxx, self.invalid, ),
        # (self.xxx, self.zero, ),
        # (self.xxx, self.one, ),
        # (self.xxx, self.native, ),
        # (self.xxx, self.npv, ),
        # (self.xxx, self.param, ),
        # (self.xxx, self.param_mut, ),
        # (self.xxx, self.var, ),
        # (self.xxx, self.mon_native, ),
        # (self.xxx, self.mon_param, ),
        # (self.xxx, self.mon_npv, ),
        # (self.xxx, self.linear, ),
        # (self.xxx, self.sum, ),
        # (self.xxx, self.other, ),

    def _run_cases(self, tests, op):
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
                    if result is not NotImplemented:
                        raise
                finally:
                    for i, arg in enumerate(args):
                        if mutable[i]:
                            arg.__class__ = classes[i]
        except:
            logger.error(
                f"Failed test {test_num}:\n\t"
                + '\n\t'.join(f'{arg}  ({arg.__class__.__name__})' for arg in test)
                + f'\n\t{ans} (result)'
            )
            raise

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
                        NPV_ProductExpression((self.mon_npv.arg(0), value(self.param))),
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
