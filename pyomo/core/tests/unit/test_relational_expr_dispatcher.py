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
import logging
import operator

import pyomo.common.unittest as unittest

from pyomo.core.expr import (
    inequality,
    EqualityExpression,
    InequalityExpression,
    RangedExpression,
)

from pyomo.core.tests.unit.test_numeric_expr_dispatcher import Base

logger = logging.getLogger(__name__)


class BaseRelational(Base):
    NUM_TESTS = 25

    def tearDown(self):
        pass

    def setUp(self):
        super().setUp()

        # Note there are 13 basic argument "types" that determine how
        # expressions are generated (defined by the _EXPR_TYPE enum):
        #
        # class RELATIONAL_ARG_TYPE(enum.Enum):
        #     MUTABLE = -2
        #     ASNUMERIC = -1
        #     INVALID = 0
        #     NATIVE = 1
        #     NPV = 2
        #     PARAM = 3
        #     VAR = 4
        #     MONOMIAL = 5
        #     LINEAR = 6
        #     SUM = 7
        #     OTHER = 8
        #     INEQUALITY = 100
        #     INVALID_RELATIONAL = 101

        # self.m = ConcreteModel()
        # self.m.p0 = Param(initialize=0, mutable=False)
        # self.m.p1 = Param(initialize=1, mutable=False)
        # self.m.p = Param(initialize=6, mutable=False)
        # self.m.q = Param(initialize=7, mutable=True)
        # self.m.x = Var()
        # self.m.d = Disjunct()
        # self.bin = self.m.d.indicator_var.as_numeric()

        self.eq = self.m.x == self.m.q
        self.le = self.m.x <= self.m.q
        self.lt = self.m.x < self.m.p
        self.ranged = inequality(self.m.p, self.m.x, self.m.q)

        # self.TEMPLATE = [
        #     self.invalid,
        #     self.asbinary,
        #     self.zero,
        #     self.one,
        #     # 4:
        #     self.native,
        #     self.npv,
        #     self.param,
        #     self.param_mut,
        #     # 8:
        #     self.var,
        #     self.mon_native,
        #     self.mon_param,
        #     self.mon_npv,
        #     # 12:
        #     self.linear,
        #     self.sum,
        #     self.other,
        #     self.mutable_l0,
        #     # 16:
        #     self.mutable_l1,
        #     self.mutable_l2,
        #     self.param0,
        #     self.param1,
        #     # 20:
        #     self.mutable_l3,
        # ]
        self.TEMPLATE.extend(
            [
                # 21:
                self.eq,
                self.le,
                self.lt,
                self.ranged,
            ]
        )


#
#
# EQUALITY
#
#


class TestEquality(BaseRelational, unittest.TestCase):

    def test_eq_invalid(self):
        tests = [
            # "invalid(str) == invalid(str)" is a legitimate Python
            # operation and should never hit the Pyomo expression
            # system
            (self.invalid, self.invalid, True),
            (self.invalid, self.asbinary, False),
            (self.invalid, self.zero, False),
            (self.invalid, self.one, False),
            # 4:
            (self.invalid, self.native, False),
            (self.invalid, self.npv, False),
            (self.invalid, self.param, False),
            (self.invalid, self.param_mut, False),
            # 8:
            (self.invalid, self.var, False),
            (self.invalid, self.mon_native, False),
            (self.invalid, self.mon_param, False),
            (self.invalid, self.mon_npv, False),
            # 12:
            (self.invalid, self.linear, False),
            (self.invalid, self.sum, False),
            (self.invalid, self.other, False),
            (self.invalid, self.mutable_l0, False),
            # 16:
            (self.invalid, self.mutable_l1, False),
            (self.invalid, self.mutable_l2, False),
            (self.invalid, self.param0, False),
            (self.invalid, self.param1, False),
            # 20:
            (self.invalid, self.mutable_l3, False),
            (self.invalid, self.eq, False),
            (self.invalid, self.le, False),
            (self.invalid, self.lt, False),
            # 24:
            (self.invalid, self.ranged, False),
        ]
        self._run_cases(tests, operator.eq)

    def test_eq_asbinary(self):
        tests = [
            (self.asbinary, self.invalid, False),
            (self.asbinary, self.asbinary, True),
            (self.asbinary, self.zero, EqualityExpression((self.bin, 0))),
            (self.asbinary, self.one, EqualityExpression((self.bin, 1))),
            # 4:
            (self.asbinary, self.native, EqualityExpression((self.bin, 5))),
            (self.asbinary, self.npv, EqualityExpression((self.bin, self.npv))),
            (self.asbinary, self.param, EqualityExpression((self.bin, 6))),
            (
                self.asbinary,
                self.param_mut,
                EqualityExpression((self.bin, self.param_mut)),
            ),
            # 8:
            (self.asbinary, self.var, EqualityExpression((self.bin, self.var))),
            (
                self.asbinary,
                self.mon_native,
                EqualityExpression((self.bin, self.mon_native)),
            ),
            (
                self.asbinary,
                self.mon_param,
                EqualityExpression((self.bin, self.mon_param)),
            ),
            (self.asbinary, self.mon_npv, EqualityExpression((self.bin, self.mon_npv))),
            # 12:
            (self.asbinary, self.linear, EqualityExpression((self.bin, self.linear))),
            (self.asbinary, self.sum, EqualityExpression((self.bin, self.sum))),
            (self.asbinary, self.other, EqualityExpression((self.bin, self.other))),
            (self.asbinary, self.mutable_l0, EqualityExpression((self.bin, self.l0))),
            # 16:
            (self.asbinary, self.mutable_l1, EqualityExpression((self.bin, self.l1))),
            (self.asbinary, self.mutable_l2, EqualityExpression((self.bin, self.l2))),
            (self.asbinary, self.param0, EqualityExpression((self.bin, 0))),
            (self.asbinary, self.param1, EqualityExpression((self.bin, 1))),
            # 20:
            (self.asbinary, self.mutable_l3, EqualityExpression((self.bin, self.l3))),
            (
                self.asbinary,
                self.eq,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.asbinary,
                self.le,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.asbinary,
                self.lt,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 24
            (
                self.asbinary,
                self.ranged,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.eq)

    def test_eq_zero(self):
        tests = [
            (self.zero, self.invalid, False),
            (self.zero, self.asbinary, EqualityExpression((self.bin, 0))),
            (self.zero, self.zero, True),
            (self.zero, self.one, False),
            # 4:
            (self.zero, self.native, False),
            (self.zero, self.npv, EqualityExpression((self.npv, 0))),
            (self.zero, self.param, False),
            (self.zero, self.param_mut, EqualityExpression((self.param_mut, 0))),
            # 8:
            (self.zero, self.var, EqualityExpression((self.var, 0))),
            (self.zero, self.mon_native, EqualityExpression((self.mon_native, 0))),
            (self.zero, self.mon_param, EqualityExpression((self.mon_param, 0))),
            (self.zero, self.mon_npv, EqualityExpression((self.mon_npv, 0))),
            # 12:
            (self.zero, self.linear, EqualityExpression((self.linear, 0))),
            (self.zero, self.sum, EqualityExpression((self.sum, 0))),
            (self.zero, self.other, EqualityExpression((self.other, 0))),
            (self.zero, self.mutable_l0, True),
            # 16:
            (self.zero, self.mutable_l1, EqualityExpression((self.l1, 0))),
            (self.zero, self.mutable_l2, EqualityExpression((self.l2, 0))),
            (self.zero, self.param0, True),
            (self.zero, self.param1, False),
            # 20:
            (self.zero, self.mutable_l3, EqualityExpression((self.l3, 0))),
            (
                self.zero,
                self.eq,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.zero,
                self.le,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.zero,
                self.lt,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 24
            (
                self.zero,
                self.ranged,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.eq)

    def test_eq_one(self):
        tests = [
            (self.one, self.invalid, False),
            (self.one, self.asbinary, EqualityExpression((self.bin, 1))),
            (self.one, self.zero, False),
            (self.one, self.one, True),
            # 4:
            (self.one, self.native, False),
            (self.one, self.npv, EqualityExpression((self.npv, 1))),
            (self.one, self.param, False),
            (self.one, self.param_mut, EqualityExpression((self.param_mut, 1))),
            # 8:
            (self.one, self.var, EqualityExpression((self.var, 1))),
            (self.one, self.mon_native, EqualityExpression((self.mon_native, 1))),
            (self.one, self.mon_param, EqualityExpression((self.mon_param, 1))),
            (self.one, self.mon_npv, EqualityExpression((self.mon_npv, 1))),
            # 12:
            (self.one, self.linear, EqualityExpression((self.linear, 1))),
            (self.one, self.sum, EqualityExpression((self.sum, 1))),
            (self.one, self.other, EqualityExpression((self.other, 1))),
            (self.one, self.mutable_l0, False),
            # 16:
            (self.one, self.mutable_l1, EqualityExpression((self.l1, 1))),
            (self.one, self.mutable_l2, EqualityExpression((self.l2, 1))),
            (self.one, self.param0, False),
            (self.one, self.param1, True),
            # 20:
            (self.one, self.mutable_l3, EqualityExpression((self.l3, 1))),
            (
                self.one,
                self.eq,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.one,
                self.le,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.one,
                self.lt,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 24
            (
                self.one,
                self.ranged,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.eq)

    def test_eq_native(self):
        tests = [
            (self.native, self.invalid, False),
            (self.native, self.asbinary, EqualityExpression((self.bin, 5))),
            (self.native, self.zero, False),
            (self.native, self.one, False),
            # 4:
            (self.native, self.native, True),
            (self.native, self.npv, EqualityExpression((self.npv, 5))),
            (self.native, self.param, False),
            (self.native, self.param_mut, EqualityExpression((self.param_mut, 5))),
            # 8:
            (self.native, self.var, EqualityExpression((self.var, 5))),
            (self.native, self.mon_native, EqualityExpression((self.mon_native, 5))),
            (self.native, self.mon_param, EqualityExpression((self.mon_param, 5))),
            (self.native, self.mon_npv, EqualityExpression((self.mon_npv, 5))),
            # 12:
            (self.native, self.linear, EqualityExpression((self.linear, 5))),
            (self.native, self.sum, EqualityExpression((self.sum, 5))),
            (self.native, self.other, EqualityExpression((self.other, 5))),
            (self.native, self.mutable_l0, False),
            # 16:
            (self.native, self.mutable_l1, EqualityExpression((self.l1, 5))),
            (self.native, self.mutable_l2, EqualityExpression((self.l2, 5))),
            (self.native, self.param0, False),
            (self.native, self.param1, False),
            # 20:
            (self.native, self.mutable_l3, EqualityExpression((self.l3, 5))),
            (
                self.native,
                self.eq,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.native,
                self.le,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.native,
                self.lt,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 24
            (
                self.native,
                self.ranged,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.eq)

    def test_eq_npv(self):
        tests = [
            (self.npv, self.invalid, False),
            (self.npv, self.asbinary, EqualityExpression((self.npv, self.bin))),
            (self.npv, self.zero, EqualityExpression((self.npv, 0))),
            (self.npv, self.one, EqualityExpression((self.npv, 1))),
            # 4:
            (self.npv, self.native, EqualityExpression((self.npv, 5))),
            (self.npv, self.npv, EqualityExpression((self.npv, self.npv))),
            (self.npv, self.param, EqualityExpression((self.npv, 6))),
            (self.npv, self.param_mut, EqualityExpression((self.npv, self.param_mut))),
            # 8:
            (self.npv, self.var, EqualityExpression((self.npv, self.var))),
            (
                self.npv,
                self.mon_native,
                EqualityExpression((self.npv, self.mon_native)),
            ),
            (self.npv, self.mon_param, EqualityExpression((self.npv, self.mon_param))),
            (self.npv, self.mon_npv, EqualityExpression((self.npv, self.mon_npv))),
            # 12:
            (self.npv, self.linear, EqualityExpression((self.npv, self.linear))),
            (self.npv, self.sum, EqualityExpression((self.npv, self.sum))),
            (self.npv, self.other, EqualityExpression((self.npv, self.other))),
            (self.npv, self.mutable_l0, EqualityExpression((self.npv, self.l0))),
            # 16:
            (self.npv, self.mutable_l1, EqualityExpression((self.npv, self.l1))),
            (self.npv, self.mutable_l2, EqualityExpression((self.npv, self.l2))),
            (self.npv, self.param0, EqualityExpression((self.npv, 0))),
            (self.npv, self.param1, EqualityExpression((self.npv, 1))),
            # 20:
            (self.npv, self.mutable_l3, EqualityExpression((self.npv, self.l3))),
            (
                self.npv,
                self.eq,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.npv,
                self.le,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.npv,
                self.lt,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 24
            (
                self.npv,
                self.ranged,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.eq)

    def test_eq_param(self):
        tests = [
            (self.param, self.invalid, False),
            (self.param, self.asbinary, EqualityExpression((self.bin, 6))),
            (self.param, self.zero, False),
            (self.param, self.one, False),
            # 4:
            (self.param, self.native, False),
            (self.param, self.npv, EqualityExpression((self.npv, 6))),
            (self.param, self.param, True),
            (self.param, self.param_mut, EqualityExpression((6, self.param_mut))),
            # 8:
            (self.param, self.var, EqualityExpression((self.var, 6))),
            (self.param, self.mon_native, EqualityExpression((self.mon_native, 6))),
            (self.param, self.mon_param, EqualityExpression((self.mon_param, 6))),
            (self.param, self.mon_npv, EqualityExpression((self.mon_npv, 6))),
            # 12:
            (self.param, self.linear, EqualityExpression((self.linear, 6))),
            (self.param, self.sum, EqualityExpression((self.sum, 6))),
            (self.param, self.other, EqualityExpression((self.other, 6))),
            (self.param, self.mutable_l0, False),
            # 16:
            (self.param, self.mutable_l1, EqualityExpression((self.l1, 6))),
            (self.param, self.mutable_l2, EqualityExpression((self.l2, 6))),
            (self.param, self.param0, False),
            (self.param, self.param1, False),
            # 20:
            (self.param, self.mutable_l3, EqualityExpression((self.l3, 6))),
            (
                self.param,
                self.eq,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.param,
                self.le,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.param,
                self.lt,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 24
            (
                self.param,
                self.ranged,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.eq)

    def test_eq_param_mut(self):
        tests = [
            (self.param_mut, self.invalid, False),
            (
                self.param_mut,
                self.asbinary,
                EqualityExpression((self.param_mut, self.bin)),
            ),
            (self.param_mut, self.zero, EqualityExpression((self.param_mut, 0))),
            (self.param_mut, self.one, EqualityExpression((self.param_mut, 1))),
            # 4:
            (self.param_mut, self.native, EqualityExpression((self.param_mut, 5))),
            (self.param_mut, self.npv, EqualityExpression((self.param_mut, self.npv))),
            (self.param_mut, self.param, EqualityExpression((self.param_mut, 6))),
            (
                self.param_mut,
                self.param_mut,
                EqualityExpression((self.param_mut, self.param_mut)),
            ),
            # 8:
            (self.param_mut, self.var, EqualityExpression((self.param_mut, self.var))),
            (
                self.param_mut,
                self.mon_native,
                EqualityExpression((self.param_mut, self.mon_native)),
            ),
            (
                self.param_mut,
                self.mon_param,
                EqualityExpression((self.param_mut, self.mon_param)),
            ),
            (
                self.param_mut,
                self.mon_npv,
                EqualityExpression((self.param_mut, self.mon_npv)),
            ),
            # 12:
            (
                self.param_mut,
                self.linear,
                EqualityExpression((self.param_mut, self.linear)),
            ),
            (self.param_mut, self.sum, EqualityExpression((self.param_mut, self.sum))),
            (
                self.param_mut,
                self.other,
                EqualityExpression((self.param_mut, self.other)),
            ),
            (
                self.param_mut,
                self.mutable_l0,
                EqualityExpression((self.param_mut, self.l0)),
            ),
            # 16:
            (
                self.param_mut,
                self.mutable_l1,
                EqualityExpression((self.param_mut, self.l1)),
            ),
            (
                self.param_mut,
                self.mutable_l2,
                EqualityExpression((self.param_mut, self.l2)),
            ),
            (self.param_mut, self.param0, EqualityExpression((self.param_mut, 0))),
            (self.param_mut, self.param1, EqualityExpression((self.param_mut, 1))),
            # 20:
            (
                self.param_mut,
                self.mutable_l3,
                EqualityExpression((self.param_mut, self.l3)),
            ),
            (
                self.param_mut,
                self.eq,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.param_mut,
                self.le,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.param_mut,
                self.lt,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 24
            (
                self.param_mut,
                self.ranged,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.eq)

    def test_eq_var(self):
        tests = [
            (self.var, self.invalid, False),
            (self.var, self.asbinary, EqualityExpression((self.var, self.bin))),
            (self.var, self.zero, EqualityExpression((self.var, 0))),
            (self.var, self.one, EqualityExpression((self.var, 1))),
            # 4:
            (self.var, self.native, EqualityExpression((self.var, 5))),
            (self.var, self.npv, EqualityExpression((self.var, self.npv))),
            (self.var, self.param, EqualityExpression((self.var, 6))),
            (self.var, self.param_mut, EqualityExpression((self.var, self.param_mut))),
            # 8:
            (self.var, self.var, EqualityExpression((self.var, self.var))),
            (
                self.var,
                self.mon_native,
                EqualityExpression((self.var, self.mon_native)),
            ),
            (self.var, self.mon_param, EqualityExpression((self.var, self.mon_param))),
            (self.var, self.mon_npv, EqualityExpression((self.var, self.mon_npv))),
            # 12:
            (self.var, self.linear, EqualityExpression((self.var, self.linear))),
            (self.var, self.sum, EqualityExpression((self.var, self.sum))),
            (self.var, self.other, EqualityExpression((self.var, self.other))),
            (self.var, self.mutable_l0, EqualityExpression((self.var, self.l0))),
            # 16:
            (self.var, self.mutable_l1, EqualityExpression((self.var, self.l1))),
            (self.var, self.mutable_l2, EqualityExpression((self.var, self.l2))),
            (self.var, self.param0, EqualityExpression((self.var, 0))),
            (self.var, self.param1, EqualityExpression((self.var, 1))),
            # 20:
            (self.var, self.mutable_l3, EqualityExpression((self.var, self.l3))),
            (
                self.var,
                self.eq,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.var,
                self.le,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.var,
                self.lt,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 24
            (
                self.var,
                self.ranged,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.eq)

    def test_eq_mon_native(self):
        tests = [
            (self.mon_native, self.invalid, False),
            (
                self.mon_native,
                self.asbinary,
                EqualityExpression((self.mon_native, self.bin)),
            ),
            (self.mon_native, self.zero, EqualityExpression((self.mon_native, 0))),
            (self.mon_native, self.one, EqualityExpression((self.mon_native, 1))),
            # 4:
            (self.mon_native, self.native, EqualityExpression((self.mon_native, 5))),
            (
                self.mon_native,
                self.npv,
                EqualityExpression((self.mon_native, self.npv)),
            ),
            (self.mon_native, self.param, EqualityExpression((self.mon_native, 6))),
            (
                self.mon_native,
                self.param_mut,
                EqualityExpression((self.mon_native, self.param_mut)),
            ),
            # 8:
            (
                self.mon_native,
                self.var,
                EqualityExpression((self.mon_native, self.var)),
            ),
            (
                self.mon_native,
                self.mon_native,
                EqualityExpression((self.mon_native, self.mon_native)),
            ),
            (
                self.mon_native,
                self.mon_param,
                EqualityExpression((self.mon_native, self.mon_param)),
            ),
            (
                self.mon_native,
                self.mon_npv,
                EqualityExpression((self.mon_native, self.mon_npv)),
            ),
            # 12:
            (
                self.mon_native,
                self.linear,
                EqualityExpression((self.mon_native, self.linear)),
            ),
            (
                self.mon_native,
                self.sum,
                EqualityExpression((self.mon_native, self.sum)),
            ),
            (
                self.mon_native,
                self.other,
                EqualityExpression((self.mon_native, self.other)),
            ),
            (
                self.mon_native,
                self.mutable_l0,
                EqualityExpression((self.mon_native, self.l0)),
            ),
            # 16:
            (
                self.mon_native,
                self.mutable_l1,
                EqualityExpression((self.mon_native, self.l1)),
            ),
            (
                self.mon_native,
                self.mutable_l2,
                EqualityExpression((self.mon_native, self.l2)),
            ),
            (self.mon_native, self.param0, EqualityExpression((self.mon_native, 0))),
            (self.mon_native, self.param1, EqualityExpression((self.mon_native, 1))),
            # 20:
            (
                self.mon_native,
                self.mutable_l3,
                EqualityExpression((self.mon_native, self.l3)),
            ),
            (
                self.mon_native,
                self.eq,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mon_native,
                self.le,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mon_native,
                self.lt,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 24
            (
                self.mon_native,
                self.ranged,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.eq)

    def test_eq_mon_param(self):
        tests = [
            (self.mon_param, self.invalid, False),
            (
                self.mon_param,
                self.asbinary,
                EqualityExpression((self.mon_param, self.bin)),
            ),
            (self.mon_param, self.zero, EqualityExpression((self.mon_param, 0))),
            (self.mon_param, self.one, EqualityExpression((self.mon_param, 1))),
            # 4:
            (self.mon_param, self.native, EqualityExpression((self.mon_param, 5))),
            (self.mon_param, self.npv, EqualityExpression((self.mon_param, self.npv))),
            (self.mon_param, self.param, EqualityExpression((self.mon_param, 6))),
            (
                self.mon_param,
                self.param_mut,
                EqualityExpression((self.mon_param, self.param_mut)),
            ),
            # 8:
            (self.mon_param, self.var, EqualityExpression((self.mon_param, self.var))),
            (
                self.mon_param,
                self.mon_native,
                EqualityExpression((self.mon_param, self.mon_native)),
            ),
            (
                self.mon_param,
                self.mon_param,
                EqualityExpression((self.mon_param, self.mon_param)),
            ),
            (
                self.mon_param,
                self.mon_npv,
                EqualityExpression((self.mon_param, self.mon_npv)),
            ),
            # 12:
            (
                self.mon_param,
                self.linear,
                EqualityExpression((self.mon_param, self.linear)),
            ),
            (self.mon_param, self.sum, EqualityExpression((self.mon_param, self.sum))),
            (
                self.mon_param,
                self.other,
                EqualityExpression((self.mon_param, self.other)),
            ),
            (
                self.mon_param,
                self.mutable_l0,
                EqualityExpression((self.mon_param, self.l0)),
            ),
            # 16:
            (
                self.mon_param,
                self.mutable_l1,
                EqualityExpression((self.mon_param, self.l1)),
            ),
            (
                self.mon_param,
                self.mutable_l2,
                EqualityExpression((self.mon_param, self.l2)),
            ),
            (self.mon_param, self.param0, EqualityExpression((self.mon_param, 0))),
            (self.mon_param, self.param1, EqualityExpression((self.mon_param, 1))),
            # 20:
            (
                self.mon_param,
                self.mutable_l3,
                EqualityExpression((self.mon_param, self.l3)),
            ),
            (
                self.mon_param,
                self.eq,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mon_param,
                self.le,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mon_param,
                self.lt,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 24
            (
                self.mon_param,
                self.ranged,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.eq)

    def test_eq_mon_npv(self):
        tests = [
            (self.mon_npv, self.invalid, False),
            (self.mon_npv, self.asbinary, EqualityExpression((self.mon_npv, self.bin))),
            (self.mon_npv, self.zero, EqualityExpression((self.mon_npv, 0))),
            (self.mon_npv, self.one, EqualityExpression((self.mon_npv, 1))),
            # 4:
            (self.mon_npv, self.native, EqualityExpression((self.mon_npv, 5))),
            (self.mon_npv, self.npv, EqualityExpression((self.mon_npv, self.npv))),
            (self.mon_npv, self.param, EqualityExpression((self.mon_npv, 6))),
            (
                self.mon_npv,
                self.param_mut,
                EqualityExpression((self.mon_npv, self.param_mut)),
            ),
            # 8:
            (self.mon_npv, self.var, EqualityExpression((self.mon_npv, self.var))),
            (
                self.mon_npv,
                self.mon_native,
                EqualityExpression((self.mon_npv, self.mon_native)),
            ),
            (
                self.mon_npv,
                self.mon_param,
                EqualityExpression((self.mon_npv, self.mon_param)),
            ),
            (
                self.mon_npv,
                self.mon_npv,
                EqualityExpression((self.mon_npv, self.mon_npv)),
            ),
            # 12:
            (
                self.mon_npv,
                self.linear,
                EqualityExpression((self.mon_npv, self.linear)),
            ),
            (self.mon_npv, self.sum, EqualityExpression((self.mon_npv, self.sum))),
            (self.mon_npv, self.other, EqualityExpression((self.mon_npv, self.other))),
            (
                self.mon_npv,
                self.mutable_l0,
                EqualityExpression((self.mon_npv, self.l0)),
            ),
            # 16:
            (
                self.mon_npv,
                self.mutable_l1,
                EqualityExpression((self.mon_npv, self.l1)),
            ),
            (
                self.mon_npv,
                self.mutable_l2,
                EqualityExpression((self.mon_npv, self.l2)),
            ),
            (self.mon_npv, self.param0, EqualityExpression((self.mon_npv, 0))),
            (self.mon_npv, self.param1, EqualityExpression((self.mon_npv, 1))),
            # 20:
            (
                self.mon_npv,
                self.mutable_l3,
                EqualityExpression((self.mon_npv, self.l3)),
            ),
            (
                self.mon_npv,
                self.eq,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mon_npv,
                self.le,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mon_npv,
                self.lt,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 24
            (
                self.mon_npv,
                self.ranged,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.eq)

    def test_eq_linear(self):
        tests = [
            (self.linear, self.invalid, False),
            (self.linear, self.asbinary, EqualityExpression((self.linear, self.bin))),
            (self.linear, self.zero, EqualityExpression((self.linear, 0))),
            (self.linear, self.one, EqualityExpression((self.linear, 1))),
            # 4:
            (self.linear, self.native, EqualityExpression((self.linear, 5))),
            (self.linear, self.npv, EqualityExpression((self.linear, self.npv))),
            (self.linear, self.param, EqualityExpression((self.linear, 6))),
            (
                self.linear,
                self.param_mut,
                EqualityExpression((self.linear, self.param_mut)),
            ),
            # 8:
            (self.linear, self.var, EqualityExpression((self.linear, self.var))),
            (
                self.linear,
                self.mon_native,
                EqualityExpression((self.linear, self.mon_native)),
            ),
            (
                self.linear,
                self.mon_param,
                EqualityExpression((self.linear, self.mon_param)),
            ),
            (
                self.linear,
                self.mon_npv,
                EqualityExpression((self.linear, self.mon_npv)),
            ),
            # 12:
            (self.linear, self.linear, EqualityExpression((self.linear, self.linear))),
            (self.linear, self.sum, EqualityExpression((self.linear, self.sum))),
            (self.linear, self.other, EqualityExpression((self.linear, self.other))),
            (self.linear, self.mutable_l0, EqualityExpression((self.linear, self.l0))),
            # 16:
            (self.linear, self.mutable_l1, EqualityExpression((self.linear, self.l1))),
            (self.linear, self.mutable_l2, EqualityExpression((self.linear, self.l2))),
            (self.linear, self.param0, EqualityExpression((self.linear, 0))),
            (self.linear, self.param1, EqualityExpression((self.linear, 1))),
            # 20:
            (self.linear, self.mutable_l3, EqualityExpression((self.linear, self.l3))),
            (
                self.linear,
                self.eq,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.linear,
                self.le,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.linear,
                self.lt,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 24
            (
                self.linear,
                self.ranged,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.eq)

    def test_eq_sum(self):
        tests = [
            (self.sum, self.invalid, False),
            (self.sum, self.asbinary, EqualityExpression((self.sum, self.bin))),
            (self.sum, self.zero, EqualityExpression((self.sum, 0))),
            (self.sum, self.one, EqualityExpression((self.sum, 1))),
            # 4:
            (self.sum, self.native, EqualityExpression((self.sum, 5))),
            (self.sum, self.npv, EqualityExpression((self.sum, self.npv))),
            (self.sum, self.param, EqualityExpression((self.sum, 6))),
            (self.sum, self.param_mut, EqualityExpression((self.sum, self.param_mut))),
            # 8:
            (self.sum, self.var, EqualityExpression((self.sum, self.var))),
            (
                self.sum,
                self.mon_native,
                EqualityExpression((self.sum, self.mon_native)),
            ),
            (self.sum, self.mon_param, EqualityExpression((self.sum, self.mon_param))),
            (self.sum, self.mon_npv, EqualityExpression((self.sum, self.mon_npv))),
            # 12:
            (self.sum, self.linear, EqualityExpression((self.linear, self.sum))),
            (self.sum, self.sum, EqualityExpression((self.sum, self.sum))),
            (self.sum, self.other, EqualityExpression((self.sum, self.other))),
            (self.sum, self.mutable_l0, EqualityExpression((self.l0, self.sum))),
            # 16:
            (self.sum, self.mutable_l1, EqualityExpression((self.l1, self.sum))),
            (self.sum, self.mutable_l2, EqualityExpression((self.l2, self.sum))),
            (self.sum, self.param0, EqualityExpression((self.sum, 0))),
            (self.sum, self.param1, EqualityExpression((self.sum, 1))),
            # 20:
            (self.sum, self.mutable_l3, EqualityExpression((self.l3, self.sum))),
            (
                self.sum,
                self.eq,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.sum,
                self.le,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.sum,
                self.lt,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 24
            (
                self.sum,
                self.ranged,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.eq)

    def test_eq_other(self):
        tests = [
            (self.other, self.invalid, False),
            (self.other, self.asbinary, EqualityExpression((self.other, self.bin))),
            (self.other, self.zero, EqualityExpression((self.other, 0))),
            (self.other, self.one, EqualityExpression((self.other, 1))),
            # 4:
            (self.other, self.native, EqualityExpression((self.other, 5))),
            (self.other, self.npv, EqualityExpression((self.npv, self.other))),
            (self.other, self.param, EqualityExpression((self.other, 6))),
            (
                self.other,
                self.param_mut,
                EqualityExpression((self.other, self.param_mut)),
            ),
            # 8:
            (self.other, self.var, EqualityExpression((self.other, self.var))),
            (
                self.other,
                self.mon_native,
                EqualityExpression((self.other, self.mon_native)),
            ),
            (
                self.other,
                self.mon_param,
                EqualityExpression((self.other, self.mon_param)),
            ),
            (self.other, self.mon_npv, EqualityExpression((self.other, self.mon_npv))),
            # 12:
            (self.other, self.linear, EqualityExpression((self.other, self.linear))),
            (self.other, self.sum, EqualityExpression((self.other, self.sum))),
            (self.other, self.other, EqualityExpression((self.other, self.other))),
            (self.other, self.mutable_l0, EqualityExpression((self.other, self.l0))),
            # 16:
            (self.other, self.mutable_l1, EqualityExpression((self.other, self.l1))),
            (self.other, self.mutable_l2, EqualityExpression((self.other, self.l2))),
            (self.other, self.param0, EqualityExpression((self.other, 0))),
            (self.other, self.param1, EqualityExpression((self.other, 1))),
            # 20:
            (self.other, self.mutable_l3, EqualityExpression((self.other, self.l3))),
            (
                self.other,
                self.eq,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.other,
                self.le,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.other,
                self.lt,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 24
            (
                self.other,
                self.ranged,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.eq)

    def test_eq_mutable_l0(self):
        tests = [
            (self.mutable_l0, self.invalid, False),
            (self.mutable_l0, self.asbinary, EqualityExpression((self.l0, self.bin))),
            (self.mutable_l0, self.zero, True),
            (self.mutable_l0, self.one, False),
            # 4:
            (self.mutable_l0, self.native, False),
            (self.mutable_l0, self.npv, EqualityExpression((self.l0, self.npv))),
            (self.mutable_l0, self.param, False),
            (
                self.mutable_l0,
                self.param_mut,
                EqualityExpression((self.l0, self.param_mut)),
            ),
            # 8:
            (self.mutable_l0, self.var, EqualityExpression((self.l0, self.var))),
            (
                self.mutable_l0,
                self.mon_native,
                EqualityExpression((self.l0, self.mon_native)),
            ),
            (
                self.mutable_l0,
                self.mon_param,
                EqualityExpression((self.l0, self.mon_param)),
            ),
            (
                self.mutable_l0,
                self.mon_npv,
                EqualityExpression((self.l0, self.mon_npv)),
            ),
            # 12:
            (self.mutable_l0, self.linear, EqualityExpression((self.l0, self.linear))),
            (self.mutable_l0, self.sum, EqualityExpression((self.l0, self.sum))),
            (self.mutable_l0, self.other, EqualityExpression((self.l0, self.other))),
            (self.mutable_l0, self.mutable_l0, True),
            # 16:
            (self.mutable_l0, self.mutable_l1, EqualityExpression((self.l1, self.l0))),
            (self.mutable_l0, self.mutable_l2, EqualityExpression((self.l0, self.l2))),
            (self.mutable_l0, self.param0, True),
            (self.mutable_l0, self.param1, False),
            # 20:
            (self.mutable_l0, self.mutable_l3, EqualityExpression((self.l3, self.l0))),
            (
                self.mutable_l0,
                self.eq,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mutable_l0,
                self.le,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mutable_l0,
                self.lt,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 24
            (
                self.mutable_l0,
                self.ranged,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.eq)

    def test_eq_mutable_l1(self):
        tests = [
            (self.mutable_l1, self.invalid, False),
            (self.mutable_l1, self.asbinary, EqualityExpression((self.l1, self.bin))),
            (self.mutable_l1, self.zero, EqualityExpression((self.l1, 0))),
            (self.mutable_l1, self.one, EqualityExpression((self.l1, 1))),
            # 4:
            (self.mutable_l1, self.native, EqualityExpression((self.l1, 5))),
            (self.mutable_l1, self.npv, EqualityExpression((self.l1, self.npv))),
            (self.mutable_l1, self.param, EqualityExpression((self.l1, 6))),
            (
                self.mutable_l1,
                self.param_mut,
                EqualityExpression((self.l1, self.param_mut)),
            ),
            # 8:
            (self.mutable_l1, self.var, EqualityExpression((self.l1, self.var))),
            (
                self.mutable_l1,
                self.mon_native,
                EqualityExpression((self.l1, self.mon_native)),
            ),
            (
                self.mutable_l1,
                self.mon_param,
                EqualityExpression((self.l1, self.mon_param)),
            ),
            (
                self.mutable_l1,
                self.mon_npv,
                EqualityExpression((self.l1, self.mon_npv)),
            ),
            # 12:
            (self.mutable_l1, self.linear, EqualityExpression((self.l1, self.linear))),
            (self.mutable_l1, self.sum, EqualityExpression((self.l1, self.sum))),
            (self.mutable_l1, self.other, EqualityExpression((self.l1, self.other))),
            (self.mutable_l1, self.mutable_l0, EqualityExpression((self.l1, self.l0))),
            # 16:
            (self.mutable_l1, self.mutable_l1, EqualityExpression((self.l1, self.l1))),
            (self.mutable_l1, self.mutable_l2, EqualityExpression((self.l1, self.l2))),
            (self.mutable_l1, self.param0, EqualityExpression((self.l1, 0))),
            (self.mutable_l1, self.param1, EqualityExpression((self.l1, 1))),
            # 20:
            (self.mutable_l1, self.mutable_l3, EqualityExpression((self.l3, self.l1))),
            (
                self.mutable_l1,
                self.eq,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mutable_l1,
                self.le,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mutable_l1,
                self.lt,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 24
            (
                self.mutable_l1,
                self.ranged,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.eq)

    def test_eq_mutable_l2(self):
        tests = [
            (self.mutable_l2, self.invalid, False),
            (self.mutable_l2, self.asbinary, EqualityExpression((self.l2, self.bin))),
            (self.mutable_l2, self.zero, EqualityExpression((self.l2, 0))),
            (self.mutable_l2, self.one, EqualityExpression((self.l2, 1))),
            # 4:
            (self.mutable_l2, self.native, EqualityExpression((self.l2, 5))),
            (self.mutable_l2, self.npv, EqualityExpression((self.l2, self.npv))),
            (self.mutable_l2, self.param, EqualityExpression((self.l2, 6))),
            (
                self.mutable_l2,
                self.param_mut,
                EqualityExpression((self.l2, self.param_mut)),
            ),
            # 8:
            (self.mutable_l2, self.var, EqualityExpression((self.l2, self.var))),
            (
                self.mutable_l2,
                self.mon_native,
                EqualityExpression((self.l2, self.mon_native)),
            ),
            (
                self.mutable_l2,
                self.mon_param,
                EqualityExpression((self.l2, self.mon_param)),
            ),
            (
                self.mutable_l2,
                self.mon_npv,
                EqualityExpression((self.l2, self.mon_npv)),
            ),
            # 12:
            (self.mutable_l2, self.linear, EqualityExpression((self.l2, self.linear))),
            (self.mutable_l2, self.sum, EqualityExpression((self.l2, self.sum))),
            (self.mutable_l2, self.other, EqualityExpression((self.l2, self.other))),
            (self.mutable_l2, self.mutable_l0, EqualityExpression((self.l2, self.l0))),
            # 16:
            (self.mutable_l2, self.mutable_l1, EqualityExpression((self.l1, self.l2))),
            (self.mutable_l2, self.mutable_l2, EqualityExpression((self.l2, self.l2))),
            (self.mutable_l2, self.param0, EqualityExpression((self.l2, 0))),
            (self.mutable_l2, self.param1, EqualityExpression((self.l2, 1))),
            # 20:
            (self.mutable_l2, self.mutable_l3, EqualityExpression((self.l3, self.l2))),
            (
                self.mutable_l2,
                self.eq,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mutable_l2,
                self.le,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mutable_l2,
                self.lt,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 24
            (
                self.mutable_l2,
                self.ranged,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.eq)

    def test_eq_param0(self):
        tests = [
            (self.param0, self.invalid, False),
            (self.param0, self.asbinary, EqualityExpression((self.bin, 0))),
            (self.param0, self.zero, True),
            (self.param0, self.one, False),
            # 4:
            (self.param0, self.native, False),
            (self.param0, self.npv, EqualityExpression((self.npv, 0))),
            (self.param0, self.param, False),
            (self.param0, self.param_mut, EqualityExpression((0, self.param_mut))),
            # 8:
            (self.param0, self.var, EqualityExpression((self.var, 0))),
            (self.param0, self.mon_native, EqualityExpression((self.mon_native, 0))),
            (self.param0, self.mon_param, EqualityExpression((self.mon_param, 0))),
            (self.param0, self.mon_npv, EqualityExpression((self.mon_npv, 0))),
            # 12:
            (self.param0, self.linear, EqualityExpression((self.linear, 0))),
            (self.param0, self.sum, EqualityExpression((self.sum, 0))),
            (self.param0, self.other, EqualityExpression((self.other, 0))),
            (self.param0, self.mutable_l0, True),
            # 16:
            (self.param0, self.mutable_l1, EqualityExpression((self.l1, 0))),
            (self.param0, self.mutable_l2, EqualityExpression((self.l2, 0))),
            (self.param0, self.param0, True),
            (self.param0, self.param1, False),
            # 20:
            (self.param0, self.mutable_l3, EqualityExpression((self.l3, 0))),
            (
                self.param0,
                self.eq,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.param0,
                self.le,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.param0,
                self.lt,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 24
            (
                self.param0,
                self.ranged,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.eq)

    def test_eq_param1(self):
        tests = [
            (self.param1, self.invalid, False),
            (self.param1, self.asbinary, EqualityExpression((self.bin, 1))),
            (self.param1, self.zero, False),
            (self.param1, self.one, True),
            # 4:
            (self.param1, self.native, False),
            (self.param1, self.npv, EqualityExpression((self.npv, 1))),
            (self.param1, self.param, False),
            (self.param1, self.param_mut, EqualityExpression((1, self.param_mut))),
            # 8:
            (self.param1, self.var, EqualityExpression((self.var, 1))),
            (self.param1, self.mon_native, EqualityExpression((self.mon_native, 1))),
            (self.param1, self.mon_param, EqualityExpression((self.mon_param, 1))),
            (self.param1, self.mon_npv, EqualityExpression((self.mon_npv, 1))),
            # 12:
            (self.param1, self.linear, EqualityExpression((self.linear, 1))),
            (self.param1, self.sum, EqualityExpression((self.sum, 1))),
            (self.param1, self.other, EqualityExpression((self.other, 1))),
            (self.param1, self.mutable_l0, False),
            # 16:
            (self.param1, self.mutable_l1, EqualityExpression((self.l1, 1))),
            (self.param1, self.mutable_l2, EqualityExpression((self.l2, 1))),
            (self.param1, self.param0, False),
            (self.param1, self.param1, True),
            # 20:
            (self.param1, self.mutable_l3, EqualityExpression((self.l3, 1))),
            (
                self.param1,
                self.eq,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.param1,
                self.le,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.param1,
                self.lt,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 24
            (
                self.param1,
                self.ranged,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.eq)

    def test_eq_mutable_l3(self):
        tests = [
            (self.mutable_l3, self.invalid, False),
            (self.mutable_l3, self.asbinary, EqualityExpression((self.l3, self.bin))),
            (self.mutable_l3, self.zero, EqualityExpression((self.l3, 0))),
            (self.mutable_l3, self.one, EqualityExpression((self.l3, 1))),
            # 4:
            (self.mutable_l3, self.native, EqualityExpression((self.l3, 5))),
            (self.mutable_l3, self.npv, EqualityExpression((self.l3, self.npv))),
            (self.mutable_l3, self.param, EqualityExpression((self.l3, 6))),
            (
                self.mutable_l3,
                self.param_mut,
                EqualityExpression((self.l3, self.param_mut)),
            ),
            # 8:
            (self.mutable_l3, self.var, EqualityExpression((self.l3, self.var))),
            (
                self.mutable_l3,
                self.mon_native,
                EqualityExpression((self.l3, self.mon_native)),
            ),
            (
                self.mutable_l3,
                self.mon_param,
                EqualityExpression((self.l3, self.mon_param)),
            ),
            (
                self.mutable_l3,
                self.mon_npv,
                EqualityExpression((self.l3, self.mon_npv)),
            ),
            # 12:
            (self.mutable_l3, self.linear, EqualityExpression((self.l3, self.linear))),
            (self.mutable_l3, self.sum, EqualityExpression((self.l3, self.sum))),
            (self.mutable_l3, self.other, EqualityExpression((self.l3, self.other))),
            (self.mutable_l3, self.mutable_l0, EqualityExpression((self.l3, self.l0))),
            # 16:
            (self.mutable_l3, self.mutable_l1, EqualityExpression((self.l3, self.l1))),
            (self.mutable_l3, self.mutable_l2, EqualityExpression((self.l3, self.l2))),
            (self.mutable_l3, self.param0, EqualityExpression((self.l3, 0))),
            (self.mutable_l3, self.param1, EqualityExpression((self.l3, 1))),
            # 20:
            (self.mutable_l3, self.mutable_l3, EqualityExpression((self.l3, self.l3))),
            (
                self.mutable_l3,
                self.eq,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mutable_l3,
                self.le,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mutable_l3,
                self.lt,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 24
            (
                self.mutable_l3,
                self.ranged,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.eq)

    def test_eq_eq(self):
        tests = [
            (self.eq, self.invalid, False),
            (
                self.eq,
                self.asbinary,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.zero,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.one,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 4:
            (
                self.eq,
                self.native,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.npv,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.param,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.param_mut,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 8:
            (
                self.eq,
                self.var,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.mon_native,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.mon_param,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.mon_npv,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 12:
            (
                self.eq,
                self.linear,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.sum,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.other,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.mutable_l0,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 16:
            (
                self.eq,
                self.mutable_l1,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.mutable_l2,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.param0,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.param1,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 20:
            (
                self.eq,
                self.mutable_l3,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.eq,
                "Cannot create an EqualityExpression where both "
                "sub-expressions are relational expressions",
            ),
            (
                self.eq,
                self.le,
                "Cannot create an EqualityExpression where both "
                "sub-expressions are relational expressions",
            ),
            (
                self.eq,
                self.lt,
                "Cannot create an EqualityExpression where both "
                "sub-expressions are relational expressions",
            ),
            # 24
            (
                self.eq,
                self.ranged,
                "Cannot create an EqualityExpression where both "
                "sub-expressions are relational expressions",
            ),
        ]
        self._run_cases(tests, operator.eq)

    def test_eq_le(self):
        tests = [
            (self.le, self.invalid, False),
            (
                self.le,
                self.asbinary,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.le,
                self.zero,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.le,
                self.one,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 4:
            (
                self.le,
                self.native,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.le,
                self.npv,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.le,
                self.param,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.le,
                self.param_mut,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 8:
            (
                self.le,
                self.var,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.le,
                self.mon_native,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.le,
                self.mon_param,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.le,
                self.mon_npv,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 12:
            (
                self.le,
                self.linear,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.le,
                self.sum,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.le,
                self.other,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.le,
                self.mutable_l0,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 16:
            (
                self.le,
                self.mutable_l1,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.le,
                self.mutable_l2,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.le,
                self.param0,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.le,
                self.param1,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 20:
            (
                self.le,
                self.mutable_l3,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.le,
                self.eq,
                "Cannot create an EqualityExpression where both "
                "sub-expressions are relational expressions",
            ),
            (
                self.le,
                self.le,
                "Cannot create an EqualityExpression where both "
                "sub-expressions are relational expressions",
            ),
            (
                self.le,
                self.lt,
                "Cannot create an EqualityExpression where both "
                "sub-expressions are relational expressions",
            ),
            # 24
            (
                self.le,
                self.ranged,
                "Cannot create an EqualityExpression where both "
                "sub-expressions are relational expressions",
            ),
        ]
        self._run_cases(tests, operator.eq)

    def test_eq_lt(self):
        tests = [
            (self.lt, self.invalid, False),
            (
                self.lt,
                self.asbinary,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.lt,
                self.zero,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.lt,
                self.one,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 4:
            (
                self.lt,
                self.native,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.lt,
                self.npv,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.lt,
                self.param,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.lt,
                self.param_mut,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 8:
            (
                self.lt,
                self.var,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.lt,
                self.mon_native,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.lt,
                self.mon_param,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.lt,
                self.mon_npv,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 12:
            (
                self.lt,
                self.linear,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.lt,
                self.sum,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.lt,
                self.other,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.lt,
                self.mutable_l0,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 16:
            (
                self.lt,
                self.mutable_l1,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.lt,
                self.mutable_l2,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.lt,
                self.param0,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.lt,
                self.param1,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 20:
            (
                self.lt,
                self.mutable_l3,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.lt,
                self.eq,
                "Cannot create an EqualityExpression where both "
                "sub-expressions are relational expressions",
            ),
            (
                self.lt,
                self.le,
                "Cannot create an EqualityExpression where both "
                "sub-expressions are relational expressions",
            ),
            (
                self.lt,
                self.lt,
                "Cannot create an EqualityExpression where both "
                "sub-expressions are relational expressions",
            ),
            # 24
            (
                self.lt,
                self.ranged,
                "Cannot create an EqualityExpression where both "
                "sub-expressions are relational expressions",
            ),
        ]
        self._run_cases(tests, operator.eq)

    def test_eq_ranged(self):
        tests = [
            (self.ranged, self.invalid, False),
            (
                self.ranged,
                self.asbinary,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.zero,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.one,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 4:
            (
                self.ranged,
                self.native,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.npv,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.param,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.param_mut,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 8:
            (
                self.ranged,
                self.var,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.mon_native,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.mon_param,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.mon_npv,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 12:
            (
                self.ranged,
                self.linear,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.sum,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.other,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.mutable_l0,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 16:
            (
                self.ranged,
                self.mutable_l1,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.mutable_l2,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.param0,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.param1,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 20:
            (
                self.ranged,
                self.mutable_l3,
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.eq,
                "Cannot create an EqualityExpression where both "
                "sub-expressions are relational expressions",
            ),
            (
                self.ranged,
                self.le,
                "Cannot create an EqualityExpression where both "
                "sub-expressions are relational expressions",
            ),
            (
                self.ranged,
                self.lt,
                "Cannot create an EqualityExpression where both "
                "sub-expressions are relational expressions",
            ),
            # 24
            (
                self.ranged,
                self.ranged,
                "Cannot create an EqualityExpression where both "
                "sub-expressions are relational expressions",
            ),
        ]
        self._run_cases(tests, operator.eq)


#
#
# INEQUALITY (non-strict)
#
#


class TestInequality(BaseRelational, unittest.TestCase):

    def test_le_invalid(self):
        tests = [
            # "invalid(str) == invalid(str)" is a legitimate Python
            # operation and should never hit the Pyomo expression
            # system
            (self.invalid, self.invalid, True),
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
            (self.invalid, self.eq, NotImplemented),
            (self.invalid, self.le, NotImplemented),
            (self.invalid, self.lt, NotImplemented),
            # 24:
            (self.invalid, self.ranged, NotImplemented),
        ]
        self._run_cases(tests, operator.le)

    def test_le_asbinary(self):
        tests = [
            (self.asbinary, self.invalid, NotImplemented),
            (
                self.asbinary,
                self.asbinary,
                InequalityExpression((self.bin, self.bin), False),
            ),
            (self.asbinary, self.zero, InequalityExpression((self.bin, 0), False)),
            (self.asbinary, self.one, InequalityExpression((self.bin, 1), False)),
            # 4:
            (self.asbinary, self.native, InequalityExpression((self.bin, 5), False)),
            (
                self.asbinary,
                self.npv,
                InequalityExpression((self.bin, self.npv), False),
            ),
            (self.asbinary, self.param, InequalityExpression((self.bin, 6), False)),
            (
                self.asbinary,
                self.param_mut,
                InequalityExpression((self.bin, self.param_mut), False),
            ),
            # 8:
            (
                self.asbinary,
                self.var,
                InequalityExpression((self.bin, self.var), False),
            ),
            (
                self.asbinary,
                self.mon_native,
                InequalityExpression((self.bin, self.mon_native), False),
            ),
            (
                self.asbinary,
                self.mon_param,
                InequalityExpression((self.bin, self.mon_param), False),
            ),
            (
                self.asbinary,
                self.mon_npv,
                InequalityExpression((self.bin, self.mon_npv), False),
            ),
            # 12:
            (
                self.asbinary,
                self.linear,
                InequalityExpression((self.bin, self.linear), False),
            ),
            (
                self.asbinary,
                self.sum,
                InequalityExpression((self.bin, self.sum), False),
            ),
            (
                self.asbinary,
                self.other,
                InequalityExpression((self.bin, self.other), False),
            ),
            (
                self.asbinary,
                self.mutable_l0,
                InequalityExpression((self.bin, self.l0), False),
            ),
            # 16:
            (
                self.asbinary,
                self.mutable_l1,
                InequalityExpression((self.bin, self.l1), False),
            ),
            (
                self.asbinary,
                self.mutable_l2,
                InequalityExpression((self.bin, self.l2), False),
            ),
            (self.asbinary, self.param0, InequalityExpression((self.bin, 0), False)),
            (self.asbinary, self.param1, InequalityExpression((self.bin, 1), False)),
            # 20:
            (
                self.asbinary,
                self.mutable_l3,
                InequalityExpression((self.bin, self.l3), False),
            ),
            (
                self.asbinary,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.asbinary,
                self.le,
                RangedExpression((self.bin,) + self.le.args, (False, False)),
            ),
            (
                self.asbinary,
                self.lt,
                RangedExpression((self.bin,) + self.lt.args, (False, True)),
            ),
            # 24
            (
                self.asbinary,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.le)

    def test_le_zero(self):
        tests = [
            (self.zero, self.invalid, NotImplemented),
            (self.zero, self.asbinary, InequalityExpression((0, self.bin), False)),
            (self.zero, self.zero, True),
            (self.zero, self.one, True),
            # 4:
            (self.zero, self.native, True),
            (self.zero, self.npv, InequalityExpression((0, self.npv), False)),
            (self.zero, self.param, True),
            (
                self.zero,
                self.param_mut,
                InequalityExpression((0, self.param_mut), False),
            ),
            # 8:
            (self.zero, self.var, InequalityExpression((0, self.var), False)),
            (
                self.zero,
                self.mon_native,
                InequalityExpression((0, self.mon_native), False),
            ),
            (
                self.zero,
                self.mon_param,
                InequalityExpression((0, self.mon_param), False),
            ),
            (self.zero, self.mon_npv, InequalityExpression((0, self.mon_npv), False)),
            # 12:
            (self.zero, self.linear, InequalityExpression((0, self.linear), False)),
            (self.zero, self.sum, InequalityExpression((0, self.sum), False)),
            (self.zero, self.other, InequalityExpression((0, self.other), False)),
            (self.zero, self.mutable_l0, True),
            # 16:
            (self.zero, self.mutable_l1, InequalityExpression((0, self.l1), False)),
            (self.zero, self.mutable_l2, InequalityExpression((0, self.l2), False)),
            (self.zero, self.param0, True),
            (self.zero, self.param1, True),
            # 20:
            (self.zero, self.mutable_l3, InequalityExpression((0, self.l3), False)),
            (
                self.zero,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (self.zero, self.le, RangedExpression((0,) + self.le.args, (False, False))),
            (self.zero, self.lt, RangedExpression((0,) + self.lt.args, (False, True))),
            # 24
            (
                self.zero,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.le)

    def test_le_one(self):
        tests = [
            (self.one, self.invalid, NotImplemented),
            (self.one, self.asbinary, InequalityExpression((1, self.bin), False)),
            (self.one, self.zero, False),
            (self.one, self.one, True),
            # 4:
            (self.one, self.native, True),
            (self.one, self.npv, InequalityExpression((1, self.npv), False)),
            (self.one, self.param, True),
            (
                self.one,
                self.param_mut,
                InequalityExpression((1, self.param_mut), False),
            ),
            # 8:
            (self.one, self.var, InequalityExpression((1, self.var), False)),
            (
                self.one,
                self.mon_native,
                InequalityExpression((1, self.mon_native), False),
            ),
            (
                self.one,
                self.mon_param,
                InequalityExpression((1, self.mon_param), False),
            ),
            (self.one, self.mon_npv, InequalityExpression((1, self.mon_npv), False)),
            # 12:
            (self.one, self.linear, InequalityExpression((1, self.linear), False)),
            (self.one, self.sum, InequalityExpression((1, self.sum), False)),
            (self.one, self.other, InequalityExpression((1, self.other), False)),
            (self.one, self.mutable_l0, False),
            # 16:
            (self.one, self.mutable_l1, InequalityExpression((1, self.l1), False)),
            (self.one, self.mutable_l2, InequalityExpression((1, self.l2), False)),
            (self.one, self.param0, False),
            (self.one, self.param1, True),
            # 20:
            (self.one, self.mutable_l3, InequalityExpression((1, self.l3), False)),
            (
                self.one,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (self.one, self.le, RangedExpression((1,) + self.le.args, (False, False))),
            (self.one, self.lt, RangedExpression((1,) + self.lt.args, (False, True))),
            # 24
            (
                self.one,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.le)

    def test_le_native(self):
        tests = [
            (self.native, self.invalid, NotImplemented),
            (self.native, self.asbinary, InequalityExpression((5, self.bin), False)),
            (self.native, self.zero, False),
            (self.native, self.one, False),
            # 4:
            (self.native, self.native, True),
            (self.native, self.npv, InequalityExpression((5, self.npv), False)),
            (self.native, self.param, True),
            (
                self.native,
                self.param_mut,
                InequalityExpression((5, self.param_mut), False),
            ),
            # 8:
            (self.native, self.var, InequalityExpression((5, self.var), False)),
            (
                self.native,
                self.mon_native,
                InequalityExpression((5, self.mon_native), False),
            ),
            (
                self.native,
                self.mon_param,
                InequalityExpression((5, self.mon_param), False),
            ),
            (self.native, self.mon_npv, InequalityExpression((5, self.mon_npv), False)),
            # 12:
            (self.native, self.linear, InequalityExpression((5, self.linear), False)),
            (self.native, self.sum, InequalityExpression((5, self.sum), False)),
            (self.native, self.other, InequalityExpression((5, self.other), False)),
            (self.native, self.mutable_l0, False),
            # 16:
            (self.native, self.mutable_l1, InequalityExpression((5, self.l1), False)),
            (self.native, self.mutable_l2, InequalityExpression((5, self.l2), False)),
            (self.native, self.param0, False),
            (self.native, self.param1, False),
            # 20:
            (self.native, self.mutable_l3, InequalityExpression((5, self.l3), False)),
            (
                self.native,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.native,
                self.le,
                RangedExpression((5,) + self.le.args, (False, False)),
            ),
            (
                self.native,
                self.lt,
                RangedExpression((5,) + self.lt.args, (False, True)),
            ),
            # 24
            (
                self.native,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.le)

    def test_le_npv(self):
        tests = [
            (self.npv, self.invalid, NotImplemented),
            (
                self.npv,
                self.asbinary,
                InequalityExpression((self.npv, self.bin), False),
            ),
            (self.npv, self.zero, InequalityExpression((self.npv, 0), False)),
            (self.npv, self.one, InequalityExpression((self.npv, 1), False)),
            # 4:
            (self.npv, self.native, InequalityExpression((self.npv, 5), False)),
            (self.npv, self.npv, InequalityExpression((self.npv, self.npv), False)),
            (self.npv, self.param, InequalityExpression((self.npv, 6), False)),
            (
                self.npv,
                self.param_mut,
                InequalityExpression((self.npv, self.param_mut), False),
            ),
            # 8:
            (self.npv, self.var, InequalityExpression((self.npv, self.var), False)),
            (
                self.npv,
                self.mon_native,
                InequalityExpression((self.npv, self.mon_native), False),
            ),
            (
                self.npv,
                self.mon_param,
                InequalityExpression((self.npv, self.mon_param), False),
            ),
            (
                self.npv,
                self.mon_npv,
                InequalityExpression((self.npv, self.mon_npv), False),
            ),
            # 12:
            (
                self.npv,
                self.linear,
                InequalityExpression((self.npv, self.linear), False),
            ),
            (self.npv, self.sum, InequalityExpression((self.npv, self.sum), False)),
            (self.npv, self.other, InequalityExpression((self.npv, self.other), False)),
            (
                self.npv,
                self.mutable_l0,
                InequalityExpression((self.npv, self.l0), False),
            ),
            # 16:
            (
                self.npv,
                self.mutable_l1,
                InequalityExpression((self.npv, self.l1), False),
            ),
            (
                self.npv,
                self.mutable_l2,
                InequalityExpression((self.npv, self.l2), False),
            ),
            (self.npv, self.param0, InequalityExpression((self.npv, 0), False)),
            (self.npv, self.param1, InequalityExpression((self.npv, 1), False)),
            # 20:
            (
                self.npv,
                self.mutable_l3,
                InequalityExpression((self.npv, self.l3), False),
            ),
            (
                self.npv,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.npv,
                self.le,
                RangedExpression((self.npv,) + self.le.args, (False, False)),
            ),
            (
                self.npv,
                self.lt,
                RangedExpression((self.npv,) + self.lt.args, (False, True)),
            ),
            # 24
            (
                self.npv,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.le)

    def test_le_param(self):
        tests = [
            (self.param, self.invalid, NotImplemented),
            (self.param, self.asbinary, InequalityExpression((6, self.bin), False)),
            (self.param, self.zero, False),
            (self.param, self.one, False),
            # 4:
            (self.param, self.native, False),
            (self.param, self.npv, InequalityExpression((6, self.npv), False)),
            (self.param, self.param, True),
            (
                self.param,
                self.param_mut,
                InequalityExpression((6, self.param_mut), False),
            ),
            # 8:
            (self.param, self.var, InequalityExpression((6, self.var), False)),
            (
                self.param,
                self.mon_native,
                InequalityExpression((6, self.mon_native), False),
            ),
            (
                self.param,
                self.mon_param,
                InequalityExpression((6, self.mon_param), False),
            ),
            (self.param, self.mon_npv, InequalityExpression((6, self.mon_npv), False)),
            # 12:
            (self.param, self.linear, InequalityExpression((6, self.linear), False)),
            (self.param, self.sum, InequalityExpression((6, self.sum), False)),
            (self.param, self.other, InequalityExpression((6, self.other), False)),
            (self.param, self.mutable_l0, False),
            # 16:
            (self.param, self.mutable_l1, InequalityExpression((6, self.l1), False)),
            (self.param, self.mutable_l2, InequalityExpression((6, self.l2), False)),
            (self.param, self.param0, False),
            (self.param, self.param1, False),
            # 20:
            (self.param, self.mutable_l3, InequalityExpression((6, self.l3), False)),
            (
                self.param,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.param,
                self.le,
                RangedExpression((6,) + self.le.args, (False, False)),
            ),
            (self.param, self.lt, RangedExpression((6,) + self.lt.args, (False, True))),
            # 24
            (
                self.param,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.le)

    def test_le_param_mut(self):
        tests = [
            (self.param_mut, self.invalid, NotImplemented),
            (
                self.param_mut,
                self.asbinary,
                InequalityExpression((self.param_mut, self.bin), False),
            ),
            (
                self.param_mut,
                self.zero,
                InequalityExpression((self.param_mut, 0), False),
            ),
            (
                self.param_mut,
                self.one,
                InequalityExpression((self.param_mut, 1), False),
            ),
            # 4:
            (
                self.param_mut,
                self.native,
                InequalityExpression((self.param_mut, 5), False),
            ),
            (
                self.param_mut,
                self.npv,
                InequalityExpression((self.param_mut, self.npv), False),
            ),
            (
                self.param_mut,
                self.param,
                InequalityExpression((self.param_mut, 6), False),
            ),
            (
                self.param_mut,
                self.param_mut,
                InequalityExpression((self.param_mut, self.param_mut), False),
            ),
            # 8:
            (
                self.param_mut,
                self.var,
                InequalityExpression((self.param_mut, self.var), False),
            ),
            (
                self.param_mut,
                self.mon_native,
                InequalityExpression((self.param_mut, self.mon_native), False),
            ),
            (
                self.param_mut,
                self.mon_param,
                InequalityExpression((self.param_mut, self.mon_param), False),
            ),
            (
                self.param_mut,
                self.mon_npv,
                InequalityExpression((self.param_mut, self.mon_npv), False),
            ),
            # 12:
            (
                self.param_mut,
                self.linear,
                InequalityExpression((self.param_mut, self.linear), False),
            ),
            (
                self.param_mut,
                self.sum,
                InequalityExpression((self.param_mut, self.sum), False),
            ),
            (
                self.param_mut,
                self.other,
                InequalityExpression((self.param_mut, self.other), False),
            ),
            (
                self.param_mut,
                self.mutable_l0,
                InequalityExpression((self.param_mut, self.l0), False),
            ),
            # 16:
            (
                self.param_mut,
                self.mutable_l1,
                InequalityExpression((self.param_mut, self.l1), False),
            ),
            (
                self.param_mut,
                self.mutable_l2,
                InequalityExpression((self.param_mut, self.l2), False),
            ),
            (
                self.param_mut,
                self.param0,
                InequalityExpression((self.param_mut, 0), False),
            ),
            (
                self.param_mut,
                self.param1,
                InequalityExpression((self.param_mut, 1), False),
            ),
            # 20:
            (
                self.param_mut,
                self.mutable_l3,
                InequalityExpression((self.param_mut, self.l3), False),
            ),
            (
                self.param_mut,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.param_mut,
                self.le,
                RangedExpression((self.param_mut,) + self.le.args, (False, False)),
            ),
            (
                self.param_mut,
                self.lt,
                RangedExpression((self.param_mut,) + self.lt.args, (False, True)),
            ),
            # 24
            (
                self.param_mut,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.le)

    def test_le_var(self):
        tests = [
            (self.var, self.invalid, NotImplemented),
            (
                self.var,
                self.asbinary,
                InequalityExpression((self.var, self.bin), False),
            ),
            (self.var, self.zero, InequalityExpression((self.var, 0), False)),
            (self.var, self.one, InequalityExpression((self.var, 1), False)),
            # 4:
            (self.var, self.native, InequalityExpression((self.var, 5), False)),
            (self.var, self.npv, InequalityExpression((self.var, self.npv), False)),
            (self.var, self.param, InequalityExpression((self.var, 6), False)),
            (
                self.var,
                self.param_mut,
                InequalityExpression((self.var, self.param_mut), False),
            ),
            # 8:
            (self.var, self.var, InequalityExpression((self.var, self.var), False)),
            (
                self.var,
                self.mon_native,
                InequalityExpression((self.var, self.mon_native), False),
            ),
            (
                self.var,
                self.mon_param,
                InequalityExpression((self.var, self.mon_param), False),
            ),
            (
                self.var,
                self.mon_npv,
                InequalityExpression((self.var, self.mon_npv), False),
            ),
            # 12:
            (
                self.var,
                self.linear,
                InequalityExpression((self.var, self.linear), False),
            ),
            (self.var, self.sum, InequalityExpression((self.var, self.sum), False)),
            (self.var, self.other, InequalityExpression((self.var, self.other), False)),
            (
                self.var,
                self.mutable_l0,
                InequalityExpression((self.var, self.l0), False),
            ),
            # 16:
            (
                self.var,
                self.mutable_l1,
                InequalityExpression((self.var, self.l1), False),
            ),
            (
                self.var,
                self.mutable_l2,
                InequalityExpression((self.var, self.l2), False),
            ),
            (self.var, self.param0, InequalityExpression((self.var, 0), False)),
            (self.var, self.param1, InequalityExpression((self.var, 1), False)),
            # 20:
            (
                self.var,
                self.mutable_l3,
                InequalityExpression((self.var, self.l3), False),
            ),
            (
                self.var,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.var,
                self.le,
                RangedExpression((self.var,) + self.le.args, (False, False)),
            ),
            (
                self.var,
                self.lt,
                RangedExpression((self.var,) + self.lt.args, (False, True)),
            ),
            # 24
            (
                self.var,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.le)

    def test_le_mon_native(self):
        tests = [
            (self.mon_native, self.invalid, NotImplemented),
            (
                self.mon_native,
                self.asbinary,
                InequalityExpression((self.mon_native, self.bin), False),
            ),
            (
                self.mon_native,
                self.zero,
                InequalityExpression((self.mon_native, 0), False),
            ),
            (
                self.mon_native,
                self.one,
                InequalityExpression((self.mon_native, 1), False),
            ),
            # 4:
            (
                self.mon_native,
                self.native,
                InequalityExpression((self.mon_native, 5), False),
            ),
            (
                self.mon_native,
                self.npv,
                InequalityExpression((self.mon_native, self.npv), False),
            ),
            (
                self.mon_native,
                self.param,
                InequalityExpression((self.mon_native, 6), False),
            ),
            (
                self.mon_native,
                self.param_mut,
                InequalityExpression((self.mon_native, self.param_mut), False),
            ),
            # 8:
            (
                self.mon_native,
                self.var,
                InequalityExpression((self.mon_native, self.var), False),
            ),
            (
                self.mon_native,
                self.mon_native,
                InequalityExpression((self.mon_native, self.mon_native), False),
            ),
            (
                self.mon_native,
                self.mon_param,
                InequalityExpression((self.mon_native, self.mon_param), False),
            ),
            (
                self.mon_native,
                self.mon_npv,
                InequalityExpression((self.mon_native, self.mon_npv), False),
            ),
            # 12:
            (
                self.mon_native,
                self.linear,
                InequalityExpression((self.mon_native, self.linear), False),
            ),
            (
                self.mon_native,
                self.sum,
                InequalityExpression((self.mon_native, self.sum), False),
            ),
            (
                self.mon_native,
                self.other,
                InequalityExpression((self.mon_native, self.other), False),
            ),
            (
                self.mon_native,
                self.mutable_l0,
                InequalityExpression((self.mon_native, self.l0), False),
            ),
            # 16:
            (
                self.mon_native,
                self.mutable_l1,
                InequalityExpression((self.mon_native, self.l1), False),
            ),
            (
                self.mon_native,
                self.mutable_l2,
                InequalityExpression((self.mon_native, self.l2), False),
            ),
            (
                self.mon_native,
                self.param0,
                InequalityExpression((self.mon_native, 0), False),
            ),
            (
                self.mon_native,
                self.param1,
                InequalityExpression((self.mon_native, 1), False),
            ),
            # 20:
            (
                self.mon_native,
                self.mutable_l3,
                InequalityExpression((self.mon_native, self.l3), False),
            ),
            (
                self.mon_native,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mon_native,
                self.le,
                RangedExpression((self.mon_native,) + self.le.args, (False, False)),
            ),
            (
                self.mon_native,
                self.lt,
                RangedExpression((self.mon_native,) + self.lt.args, (False, True)),
            ),
            # 24
            (
                self.mon_native,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.le)

    def test_le_mon_param(self):
        tests = [
            (self.mon_param, self.invalid, NotImplemented),
            (
                self.mon_param,
                self.asbinary,
                InequalityExpression((self.mon_param, self.bin), False),
            ),
            (
                self.mon_param,
                self.zero,
                InequalityExpression((self.mon_param, 0), False),
            ),
            (
                self.mon_param,
                self.one,
                InequalityExpression((self.mon_param, 1), False),
            ),
            # 4:
            (
                self.mon_param,
                self.native,
                InequalityExpression((self.mon_param, 5), False),
            ),
            (
                self.mon_param,
                self.npv,
                InequalityExpression((self.mon_param, self.npv), False),
            ),
            (
                self.mon_param,
                self.param,
                InequalityExpression((self.mon_param, 6), False),
            ),
            (
                self.mon_param,
                self.param_mut,
                InequalityExpression((self.mon_param, self.param_mut), False),
            ),
            # 8:
            (
                self.mon_param,
                self.var,
                InequalityExpression((self.mon_param, self.var), False),
            ),
            (
                self.mon_param,
                self.mon_native,
                InequalityExpression((self.mon_param, self.mon_native), False),
            ),
            (
                self.mon_param,
                self.mon_param,
                InequalityExpression((self.mon_param, self.mon_param), False),
            ),
            (
                self.mon_param,
                self.mon_npv,
                InequalityExpression((self.mon_param, self.mon_npv), False),
            ),
            # 12:
            (
                self.mon_param,
                self.linear,
                InequalityExpression((self.mon_param, self.linear), False),
            ),
            (
                self.mon_param,
                self.sum,
                InequalityExpression((self.mon_param, self.sum), False),
            ),
            (
                self.mon_param,
                self.other,
                InequalityExpression((self.mon_param, self.other), False),
            ),
            (
                self.mon_param,
                self.mutable_l0,
                InequalityExpression((self.mon_param, self.l0), False),
            ),
            # 16:
            (
                self.mon_param,
                self.mutable_l1,
                InequalityExpression((self.mon_param, self.l1), False),
            ),
            (
                self.mon_param,
                self.mutable_l2,
                InequalityExpression((self.mon_param, self.l2), False),
            ),
            (
                self.mon_param,
                self.param0,
                InequalityExpression((self.mon_param, 0), False),
            ),
            (
                self.mon_param,
                self.param1,
                InequalityExpression((self.mon_param, 1), False),
            ),
            # 20:
            (
                self.mon_param,
                self.mutable_l3,
                InequalityExpression((self.mon_param, self.l3), False),
            ),
            (
                self.mon_param,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mon_param,
                self.le,
                RangedExpression((self.mon_param,) + self.le.args, (False, False)),
            ),
            (
                self.mon_param,
                self.lt,
                RangedExpression((self.mon_param,) + self.lt.args, (False, True)),
            ),
            # 24
            (
                self.mon_param,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.le)

    def test_le_mon_npv(self):
        tests = [
            (self.mon_npv, self.invalid, NotImplemented),
            (
                self.mon_npv,
                self.asbinary,
                InequalityExpression((self.mon_npv, self.bin), False),
            ),
            (self.mon_npv, self.zero, InequalityExpression((self.mon_npv, 0), False)),
            (self.mon_npv, self.one, InequalityExpression((self.mon_npv, 1), False)),
            # 4:
            (self.mon_npv, self.native, InequalityExpression((self.mon_npv, 5), False)),
            (
                self.mon_npv,
                self.npv,
                InequalityExpression((self.mon_npv, self.npv), False),
            ),
            (self.mon_npv, self.param, InequalityExpression((self.mon_npv, 6), False)),
            (
                self.mon_npv,
                self.param_mut,
                InequalityExpression((self.mon_npv, self.param_mut), False),
            ),
            # 8:
            (
                self.mon_npv,
                self.var,
                InequalityExpression((self.mon_npv, self.var), False),
            ),
            (
                self.mon_npv,
                self.mon_native,
                InequalityExpression((self.mon_npv, self.mon_native), False),
            ),
            (
                self.mon_npv,
                self.mon_param,
                InequalityExpression((self.mon_npv, self.mon_param), False),
            ),
            (
                self.mon_npv,
                self.mon_npv,
                InequalityExpression((self.mon_npv, self.mon_npv), False),
            ),
            # 12:
            (
                self.mon_npv,
                self.linear,
                InequalityExpression((self.mon_npv, self.linear), False),
            ),
            (
                self.mon_npv,
                self.sum,
                InequalityExpression((self.mon_npv, self.sum), False),
            ),
            (
                self.mon_npv,
                self.other,
                InequalityExpression((self.mon_npv, self.other), False),
            ),
            (
                self.mon_npv,
                self.mutable_l0,
                InequalityExpression((self.mon_npv, self.l0), False),
            ),
            # 16:
            (
                self.mon_npv,
                self.mutable_l1,
                InequalityExpression((self.mon_npv, self.l1), False),
            ),
            (
                self.mon_npv,
                self.mutable_l2,
                InequalityExpression((self.mon_npv, self.l2), False),
            ),
            (self.mon_npv, self.param0, InequalityExpression((self.mon_npv, 0), False)),
            (self.mon_npv, self.param1, InequalityExpression((self.mon_npv, 1), False)),
            # 20:
            (
                self.mon_npv,
                self.mutable_l3,
                InequalityExpression((self.mon_npv, self.l3), False),
            ),
            (
                self.mon_npv,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mon_npv,
                self.le,
                RangedExpression((self.mon_npv,) + self.le.args, (False, False)),
            ),
            (
                self.mon_npv,
                self.lt,
                RangedExpression((self.mon_npv,) + self.lt.args, (False, True)),
            ),
            # 24
            (
                self.mon_npv,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.le)

    def test_le_linear(self):
        tests = [
            (self.linear, self.invalid, NotImplemented),
            (
                self.linear,
                self.asbinary,
                InequalityExpression((self.linear, self.bin), False),
            ),
            (self.linear, self.zero, InequalityExpression((self.linear, 0), False)),
            (self.linear, self.one, InequalityExpression((self.linear, 1), False)),
            # 4:
            (self.linear, self.native, InequalityExpression((self.linear, 5), False)),
            (
                self.linear,
                self.npv,
                InequalityExpression((self.linear, self.npv), False),
            ),
            (self.linear, self.param, InequalityExpression((self.linear, 6), False)),
            (
                self.linear,
                self.param_mut,
                InequalityExpression((self.linear, self.param_mut), False),
            ),
            # 8:
            (
                self.linear,
                self.var,
                InequalityExpression((self.linear, self.var), False),
            ),
            (
                self.linear,
                self.mon_native,
                InequalityExpression((self.linear, self.mon_native), False),
            ),
            (
                self.linear,
                self.mon_param,
                InequalityExpression((self.linear, self.mon_param), False),
            ),
            (
                self.linear,
                self.mon_npv,
                InequalityExpression((self.linear, self.mon_npv), False),
            ),
            # 12:
            (
                self.linear,
                self.linear,
                InequalityExpression((self.linear, self.linear), False),
            ),
            (
                self.linear,
                self.sum,
                InequalityExpression((self.linear, self.sum), False),
            ),
            (
                self.linear,
                self.other,
                InequalityExpression((self.linear, self.other), False),
            ),
            (
                self.linear,
                self.mutable_l0,
                InequalityExpression((self.linear, self.l0), False),
            ),
            # 16:
            (
                self.linear,
                self.mutable_l1,
                InequalityExpression((self.linear, self.l1), False),
            ),
            (
                self.linear,
                self.mutable_l2,
                InequalityExpression((self.linear, self.l2), False),
            ),
            (self.linear, self.param0, InequalityExpression((self.linear, 0), False)),
            (self.linear, self.param1, InequalityExpression((self.linear, 1), False)),
            # 20:
            (
                self.linear,
                self.mutable_l3,
                InequalityExpression((self.linear, self.l3), False),
            ),
            (
                self.linear,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.linear,
                self.le,
                RangedExpression(((self.linear,) + self.le.args), (False, False)),
            ),
            (
                self.linear,
                self.lt,
                RangedExpression(((self.linear,) + self.lt.args), (False, True)),
            ),
            # 24
            (
                self.linear,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.le)

    def test_le_sum(self):
        tests = [
            (self.sum, self.invalid, NotImplemented),
            (
                self.sum,
                self.asbinary,
                InequalityExpression((self.sum, self.bin), False),
            ),
            (self.sum, self.zero, InequalityExpression((self.sum, 0), False)),
            (self.sum, self.one, InequalityExpression((self.sum, 1), False)),
            # 4:
            (self.sum, self.native, InequalityExpression((self.sum, 5), False)),
            (self.sum, self.npv, InequalityExpression((self.sum, self.npv), False)),
            (self.sum, self.param, InequalityExpression((self.sum, 6), False)),
            (
                self.sum,
                self.param_mut,
                InequalityExpression((self.sum, self.param_mut), False),
            ),
            # 8:
            (self.sum, self.var, InequalityExpression((self.sum, self.var), False)),
            (
                self.sum,
                self.mon_native,
                InequalityExpression((self.sum, self.mon_native), False),
            ),
            (
                self.sum,
                self.mon_param,
                InequalityExpression((self.sum, self.mon_param), False),
            ),
            (
                self.sum,
                self.mon_npv,
                InequalityExpression((self.sum, self.mon_npv), False),
            ),
            # 12:
            (
                self.sum,
                self.linear,
                InequalityExpression((self.sum, self.linear), False),
            ),
            (self.sum, self.sum, InequalityExpression((self.sum, self.sum), False)),
            (self.sum, self.other, InequalityExpression((self.sum, self.other), False)),
            (
                self.sum,
                self.mutable_l0,
                InequalityExpression((self.sum, self.l0), False),
            ),
            # 16:
            (
                self.sum,
                self.mutable_l1,
                InequalityExpression((self.sum, self.l1), False),
            ),
            (
                self.sum,
                self.mutable_l2,
                InequalityExpression((self.sum, self.l2), False),
            ),
            (self.sum, self.param0, InequalityExpression((self.sum, 0), False)),
            (self.sum, self.param1, InequalityExpression((self.sum, 1), False)),
            # 20:
            (
                self.sum,
                self.mutable_l3,
                InequalityExpression((self.sum, self.l3), False),
            ),
            (
                self.sum,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.sum,
                self.le,
                RangedExpression((self.sum,) + self.le.args, (False, False)),
            ),
            (
                self.sum,
                self.lt,
                RangedExpression((self.sum,) + self.lt.args, (False, True)),
            ),
            # 24
            (
                self.sum,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.le)

    def test_le_other(self):
        tests = [
            (self.other, self.invalid, NotImplemented),
            (
                self.other,
                self.asbinary,
                InequalityExpression((self.other, self.bin), False),
            ),
            (self.other, self.zero, InequalityExpression((self.other, 0), False)),
            (self.other, self.one, InequalityExpression((self.other, 1), False)),
            # 4:
            (self.other, self.native, InequalityExpression((self.other, 5), False)),
            (self.other, self.npv, InequalityExpression((self.other, self.npv), False)),
            (self.other, self.param, InequalityExpression((self.other, 6), False)),
            (
                self.other,
                self.param_mut,
                InequalityExpression((self.other, self.param_mut), False),
            ),
            # 8:
            (self.other, self.var, InequalityExpression((self.other, self.var), False)),
            (
                self.other,
                self.mon_native,
                InequalityExpression((self.other, self.mon_native), False),
            ),
            (
                self.other,
                self.mon_param,
                InequalityExpression((self.other, self.mon_param), False),
            ),
            (
                self.other,
                self.mon_npv,
                InequalityExpression((self.other, self.mon_npv), False),
            ),
            # 12:
            (
                self.other,
                self.linear,
                InequalityExpression((self.other, self.linear), False),
            ),
            (self.other, self.sum, InequalityExpression((self.other, self.sum), False)),
            (
                self.other,
                self.other,
                InequalityExpression((self.other, self.other), False),
            ),
            (
                self.other,
                self.mutable_l0,
                InequalityExpression((self.other, self.l0), False),
            ),
            # 16:
            (
                self.other,
                self.mutable_l1,
                InequalityExpression((self.other, self.l1), False),
            ),
            (
                self.other,
                self.mutable_l2,
                InequalityExpression((self.other, self.l2), False),
            ),
            (self.other, self.param0, InequalityExpression((self.other, 0), False)),
            (self.other, self.param1, InequalityExpression((self.other, 1), False)),
            # 20:
            (
                self.other,
                self.mutable_l3,
                InequalityExpression((self.other, self.l3), False),
            ),
            (
                self.other,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.other,
                self.le,
                RangedExpression((self.other,) + self.le.args, (False, False)),
            ),
            (
                self.other,
                self.lt,
                RangedExpression((self.other,) + self.lt.args, (False, True)),
            ),
            # 24
            (
                self.other,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.le)

    def test_le_mutable_l0(self):
        tests = [
            (self.mutable_l0, self.invalid, NotImplemented),
            (
                self.mutable_l0,
                self.asbinary,
                InequalityExpression((self.l0, self.bin), False),
            ),
            (self.mutable_l0, self.zero, True),
            (self.mutable_l0, self.one, True),
            # 4:
            (self.mutable_l0, self.native, True),
            (
                self.mutable_l0,
                self.npv,
                InequalityExpression((self.l0, self.npv), False),
            ),
            (self.mutable_l0, self.param, True),
            (
                self.mutable_l0,
                self.param_mut,
                InequalityExpression((self.l0, self.param_mut), False),
            ),
            # 8:
            (
                self.mutable_l0,
                self.var,
                InequalityExpression((self.l0, self.var), False),
            ),
            (
                self.mutable_l0,
                self.mon_native,
                InequalityExpression((self.l0, self.mon_native), False),
            ),
            (
                self.mutable_l0,
                self.mon_param,
                InequalityExpression((self.l0, self.mon_param), False),
            ),
            (
                self.mutable_l0,
                self.mon_npv,
                InequalityExpression((self.l0, self.mon_npv), False),
            ),
            # 12:
            (
                self.mutable_l0,
                self.linear,
                InequalityExpression((self.l0, self.linear), False),
            ),
            (
                self.mutable_l0,
                self.sum,
                InequalityExpression((self.l0, self.sum), False),
            ),
            (
                self.mutable_l0,
                self.other,
                InequalityExpression((self.l0, self.other), False),
            ),
            (self.mutable_l0, self.mutable_l0, True),
            # 16:
            (
                self.mutable_l0,
                self.mutable_l1,
                InequalityExpression((self.l0, self.l1), False),
            ),
            (
                self.mutable_l0,
                self.mutable_l2,
                InequalityExpression((self.l0, self.l2), False),
            ),
            (self.mutable_l0, self.param0, True),
            (self.mutable_l0, self.param1, True),
            # 20:
            (
                self.mutable_l0,
                self.mutable_l3,
                InequalityExpression((self.l0, self.l3), False),
            ),
            (
                self.mutable_l0,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mutable_l0,
                self.le,
                RangedExpression((self.l0,) + self.le.args, (False, False)),
            ),
            (
                self.mutable_l0,
                self.lt,
                RangedExpression((self.l0,) + self.lt.args, (False, True)),
            ),
            # 24
            (
                self.mutable_l0,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.le)

    def test_le_mutable_l1(self):
        tests = [
            (self.mutable_l1, self.invalid, NotImplemented),
            (
                self.mutable_l1,
                self.asbinary,
                InequalityExpression((self.l1, self.bin), False),
            ),
            (self.mutable_l1, self.zero, InequalityExpression((self.l1, 0), False)),
            (self.mutable_l1, self.one, InequalityExpression((self.l1, 1), False)),
            # 4:
            (self.mutable_l1, self.native, InequalityExpression((self.l1, 5), False)),
            (
                self.mutable_l1,
                self.npv,
                InequalityExpression((self.l1, self.npv), False),
            ),
            (self.mutable_l1, self.param, InequalityExpression((self.l1, 6), False)),
            (
                self.mutable_l1,
                self.param_mut,
                InequalityExpression((self.l1, self.param_mut), False),
            ),
            # 8:
            (
                self.mutable_l1,
                self.var,
                InequalityExpression((self.l1, self.var), False),
            ),
            (
                self.mutable_l1,
                self.mon_native,
                InequalityExpression((self.l1, self.mon_native), False),
            ),
            (
                self.mutable_l1,
                self.mon_param,
                InequalityExpression((self.l1, self.mon_param), False),
            ),
            (
                self.mutable_l1,
                self.mon_npv,
                InequalityExpression((self.l1, self.mon_npv), False),
            ),
            # 12:
            (
                self.mutable_l1,
                self.linear,
                InequalityExpression((self.l1, self.linear), False),
            ),
            (
                self.mutable_l1,
                self.sum,
                InequalityExpression((self.l1, self.sum), False),
            ),
            (
                self.mutable_l1,
                self.other,
                InequalityExpression((self.l1, self.other), False),
            ),
            (
                self.mutable_l1,
                self.mutable_l0,
                InequalityExpression((self.l1, self.l0), False),
            ),
            # 16:
            (
                self.mutable_l1,
                self.mutable_l1,
                InequalityExpression((self.l1, self.l1), False),
            ),
            (
                self.mutable_l1,
                self.mutable_l2,
                InequalityExpression((self.l1, self.l2), False),
            ),
            (self.mutable_l1, self.param0, InequalityExpression((self.l1, 0), False)),
            (self.mutable_l1, self.param1, InequalityExpression((self.l1, 1), False)),
            # 20:
            (
                self.mutable_l1,
                self.mutable_l3,
                InequalityExpression((self.l1, self.l3), False),
            ),
            (
                self.mutable_l1,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mutable_l1,
                self.le,
                RangedExpression((self.l1,) + self.le.args, (False, False)),
            ),
            (
                self.mutable_l1,
                self.lt,
                RangedExpression((self.l1,) + self.lt.args, (False, True)),
            ),
            # 24
            (
                self.mutable_l1,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.le)

    def test_le_mutable_l2(self):
        tests = [
            (self.mutable_l2, self.invalid, NotImplemented),
            (
                self.mutable_l2,
                self.asbinary,
                InequalityExpression((self.l2, self.bin), False),
            ),
            (self.mutable_l2, self.zero, InequalityExpression((self.l2, 0), False)),
            (self.mutable_l2, self.one, InequalityExpression((self.l2, 1), False)),
            # 4:
            (self.mutable_l2, self.native, InequalityExpression((self.l2, 5), False)),
            (
                self.mutable_l2,
                self.npv,
                InequalityExpression((self.l2, self.npv), False),
            ),
            (self.mutable_l2, self.param, InequalityExpression((self.l2, 6), False)),
            (
                self.mutable_l2,
                self.param_mut,
                InequalityExpression((self.l2, self.param_mut), False),
            ),
            # 8:
            (
                self.mutable_l2,
                self.var,
                InequalityExpression((self.l2, self.var), False),
            ),
            (
                self.mutable_l2,
                self.mon_native,
                InequalityExpression((self.l2, self.mon_native), False),
            ),
            (
                self.mutable_l2,
                self.mon_param,
                InequalityExpression((self.l2, self.mon_param), False),
            ),
            (
                self.mutable_l2,
                self.mon_npv,
                InequalityExpression((self.l2, self.mon_npv), False),
            ),
            # 12:
            (
                self.mutable_l2,
                self.linear,
                InequalityExpression((self.l2, self.linear), False),
            ),
            (
                self.mutable_l2,
                self.sum,
                InequalityExpression((self.l2, self.sum), False),
            ),
            (
                self.mutable_l2,
                self.other,
                InequalityExpression((self.l2, self.other), False),
            ),
            (
                self.mutable_l2,
                self.mutable_l0,
                InequalityExpression((self.l2, self.l0), False),
            ),
            # 16:
            (
                self.mutable_l2,
                self.mutable_l1,
                InequalityExpression((self.l2, self.l1), False),
            ),
            (
                self.mutable_l2,
                self.mutable_l2,
                InequalityExpression((self.l2, self.l2), False),
            ),
            (self.mutable_l2, self.param0, InequalityExpression((self.l2, 0), False)),
            (self.mutable_l2, self.param1, InequalityExpression((self.l2, 1), False)),
            # 20:
            (
                self.mutable_l2,
                self.mutable_l3,
                InequalityExpression((self.l2, self.l3), False),
            ),
            (
                self.mutable_l2,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mutable_l2,
                self.le,
                RangedExpression((self.l2,) + self.le.args, (False, False)),
            ),
            (
                self.mutable_l2,
                self.lt,
                RangedExpression((self.l2,) + self.lt.args, (False, True)),
            ),
            # 24
            (
                self.mutable_l2,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.le)

    def test_le_param0(self):
        tests = [
            (self.param0, self.invalid, NotImplemented),
            (self.param0, self.asbinary, InequalityExpression((0, self.bin), False)),
            (self.param0, self.zero, True),
            (self.param0, self.one, True),
            # 4:
            (self.param0, self.native, True),
            (self.param0, self.npv, InequalityExpression((0, self.npv), False)),
            (self.param0, self.param, True),
            (
                self.param0,
                self.param_mut,
                InequalityExpression((0, self.param_mut), False),
            ),
            # 8:
            (self.param0, self.var, InequalityExpression((0, self.var), False)),
            (
                self.param0,
                self.mon_native,
                InequalityExpression((0, self.mon_native), False),
            ),
            (
                self.param0,
                self.mon_param,
                InequalityExpression((0, self.mon_param), False),
            ),
            (self.param0, self.mon_npv, InequalityExpression((0, self.mon_npv), False)),
            # 12:
            (self.param0, self.linear, InequalityExpression((0, self.linear), False)),
            (self.param0, self.sum, InequalityExpression((0, self.sum), False)),
            (self.param0, self.other, InequalityExpression((0, self.other), False)),
            (self.param0, self.mutable_l0, True),
            # 16:
            (self.param0, self.mutable_l1, InequalityExpression((0, self.l1), False)),
            (self.param0, self.mutable_l2, InequalityExpression((0, self.l2), False)),
            (self.param0, self.param0, True),
            (self.param0, self.param1, True),
            # 20:
            (self.param0, self.mutable_l3, InequalityExpression((0, self.l3), False)),
            (
                self.param0,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.param0,
                self.le,
                RangedExpression((0,) + self.le.args, (False, False)),
            ),
            (
                self.param0,
                self.lt,
                RangedExpression((0,) + self.lt.args, (False, True)),
            ),
            # 24
            (
                self.param0,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.le)

    def test_le_param1(self):
        tests = [
            (self.param1, self.invalid, NotImplemented),
            (self.param1, self.asbinary, InequalityExpression((1, self.bin), False)),
            (self.param1, self.zero, False),
            (self.param1, self.one, True),
            # 4:
            (self.param1, self.native, True),
            (self.param1, self.npv, InequalityExpression((1, self.npv), False)),
            (self.param1, self.param, True),
            (
                self.param1,
                self.param_mut,
                InequalityExpression((1, self.param_mut), False),
            ),
            # 8:
            (self.param1, self.var, InequalityExpression((1, self.var), False)),
            (
                self.param1,
                self.mon_native,
                InequalityExpression((1, self.mon_native), False),
            ),
            (
                self.param1,
                self.mon_param,
                InequalityExpression((1, self.mon_param), False),
            ),
            (self.param1, self.mon_npv, InequalityExpression((1, self.mon_npv), False)),
            # 12:
            (self.param1, self.linear, InequalityExpression((1, self.linear), False)),
            (self.param1, self.sum, InequalityExpression((1, self.sum), False)),
            (self.param1, self.other, InequalityExpression((1, self.other), False)),
            (self.param1, self.mutable_l0, False),
            # 16:
            (self.param1, self.mutable_l1, InequalityExpression((1, self.l1), False)),
            (self.param1, self.mutable_l2, InequalityExpression((1, self.l2), False)),
            (self.param1, self.param0, False),
            (self.param1, self.param1, True),
            # 20:
            (self.param1, self.mutable_l3, InequalityExpression((1, self.l3), False)),
            (
                self.param1,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.param1,
                self.le,
                RangedExpression((1,) + self.le.args, (False, False)),
            ),
            (
                self.param1,
                self.lt,
                RangedExpression((1,) + self.lt.args, (False, True)),
            ),
            # 24
            (
                self.param1,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.le)

    def test_le_mutable_l3(self):
        tests = [
            (self.mutable_l3, self.invalid, NotImplemented),
            (
                self.mutable_l3,
                self.asbinary,
                InequalityExpression((self.l3, self.bin), False),
            ),
            (self.mutable_l3, self.zero, InequalityExpression((self.l3, 0), False)),
            (self.mutable_l3, self.one, InequalityExpression((self.l3, 1), False)),
            # 4:
            (self.mutable_l3, self.native, InequalityExpression((self.l3, 5), False)),
            (
                self.mutable_l3,
                self.npv,
                InequalityExpression((self.l3, self.npv), False),
            ),
            (self.mutable_l3, self.param, InequalityExpression((self.l3, 6), False)),
            (
                self.mutable_l3,
                self.param_mut,
                InequalityExpression((self.l3, self.param_mut), False),
            ),
            # 8:
            (
                self.mutable_l3,
                self.var,
                InequalityExpression((self.l3, self.var), False),
            ),
            (
                self.mutable_l3,
                self.mon_native,
                InequalityExpression((self.l3, self.mon_native), False),
            ),
            (
                self.mutable_l3,
                self.mon_param,
                InequalityExpression((self.l3, self.mon_param), False),
            ),
            (
                self.mutable_l3,
                self.mon_npv,
                InequalityExpression((self.l3, self.mon_npv), False),
            ),
            # 12:
            (
                self.mutable_l3,
                self.linear,
                InequalityExpression((self.l3, self.linear), False),
            ),
            (
                self.mutable_l3,
                self.sum,
                InequalityExpression((self.l3, self.sum), False),
            ),
            (
                self.mutable_l3,
                self.other,
                InequalityExpression((self.l3, self.other), False),
            ),
            (
                self.mutable_l3,
                self.mutable_l0,
                InequalityExpression((self.l3, self.l0), False),
            ),
            # 16:
            (
                self.mutable_l3,
                self.mutable_l1,
                InequalityExpression((self.l3, self.l1), False),
            ),
            (
                self.mutable_l3,
                self.mutable_l2,
                InequalityExpression((self.l3, self.l2), False),
            ),
            (self.mutable_l3, self.param0, InequalityExpression((self.l3, 0), False)),
            (self.mutable_l3, self.param1, InequalityExpression((self.l3, 1), False)),
            # 20:
            (
                self.mutable_l3,
                self.mutable_l3,
                InequalityExpression((self.l3, self.l3), False),
            ),
            (
                self.mutable_l3,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mutable_l3,
                self.le,
                RangedExpression((self.l3,) + self.le.args, (False, False)),
            ),
            (
                self.mutable_l3,
                self.lt,
                RangedExpression((self.l3,) + self.lt.args, (False, True)),
            ),
            # 24
            (
                self.mutable_l3,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.le)

    def test_le_eq(self):
        tests = [
            (self.eq, self.invalid, NotImplemented),
            (
                self.eq,
                self.asbinary,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.zero,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.one,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 4:
            (
                self.eq,
                self.native,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.npv,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.param,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.param_mut,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 8:
            (
                self.eq,
                self.var,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.mon_native,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.mon_param,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.mon_npv,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 12:
            (
                self.eq,
                self.linear,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.sum,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.other,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.mutable_l0,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 16:
            (
                self.eq,
                self.mutable_l1,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.mutable_l2,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.param0,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.param1,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 20:
            (
                self.eq,
                self.mutable_l3,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.eq,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
            (
                self.eq,
                self.le,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
            (
                self.eq,
                self.lt,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
            # 24
            (
                self.eq,
                self.ranged,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
        ]
        self._run_cases(tests, operator.le)

    def test_le_le(self):
        tests = [
            (self.le, self.invalid, NotImplemented),
            (
                self.le,
                self.asbinary,
                RangedExpression((self.le.args + (self.bin,)), (False, False)),
            ),
            (
                self.le,
                self.zero,
                RangedExpression((self.le.args + (0,)), (False, False)),
            ),
            (
                self.le,
                self.one,
                RangedExpression((self.le.args + (1,)), (False, False)),
            ),
            # 4:
            (
                self.le,
                self.native,
                RangedExpression((self.le.args + (5,)), (False, False)),
            ),
            (
                self.le,
                self.npv,
                RangedExpression((self.le.args + (self.npv,)), (False, False)),
            ),
            (
                self.le,
                self.param,
                RangedExpression((self.le.args + (6,)), (False, False)),
            ),
            (
                self.le,
                self.param_mut,
                RangedExpression((self.le.args + (self.param_mut,)), (False, False)),
            ),
            # 8:
            (
                self.le,
                self.var,
                RangedExpression((self.le.args + (self.var,)), (False, False)),
            ),
            (
                self.le,
                self.mon_native,
                RangedExpression((self.le.args + (self.mon_native,)), (False, False)),
            ),
            (
                self.le,
                self.mon_param,
                RangedExpression((self.le.args + (self.mon_param,)), (False, False)),
            ),
            (
                self.le,
                self.mon_npv,
                RangedExpression((self.le.args + (self.mon_npv,)), (False, False)),
            ),
            # 12:
            (
                self.le,
                self.linear,
                RangedExpression((self.le.args + (self.linear,)), (False, False)),
            ),
            (
                self.le,
                self.sum,
                RangedExpression((self.le.args + (self.sum,)), (False, False)),
            ),
            (
                self.le,
                self.other,
                RangedExpression((self.le.args + (self.other,)), (False, False)),
            ),
            (
                self.le,
                self.mutable_l0,
                RangedExpression((self.le.args + (self.l0,)), (False, False)),
            ),
            # 16:
            (
                self.le,
                self.mutable_l1,
                RangedExpression((self.le.args + (self.l1,)), (False, False)),
            ),
            (
                self.le,
                self.mutable_l2,
                RangedExpression((self.le.args + (self.l2,)), (False, False)),
            ),
            (
                self.le,
                self.param0,
                RangedExpression((self.le.args + (0,)), (False, False)),
            ),
            (
                self.le,
                self.param1,
                RangedExpression((self.le.args + (1,)), (False, False)),
            ),
            # 20:
            (
                self.le,
                self.mutable_l3,
                RangedExpression((self.le.args + (self.l3,)), (False, False)),
            ),
            (
                self.le,
                self.eq,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
            (
                self.le,
                self.le,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
            (
                self.le,
                self.lt,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
            # 24
            (
                self.le,
                self.ranged,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
        ]
        self._run_cases(tests, operator.le)

    def test_le_lt(self):
        tests = [
            (self.lt, self.invalid, NotImplemented),
            (
                self.lt,
                self.asbinary,
                RangedExpression((self.lt.args + (self.bin,)), (True, False)),
            ),
            (
                self.lt,
                self.zero,
                RangedExpression((self.lt.args + (0,)), (True, False)),
            ),
            (self.lt, self.one, RangedExpression((self.lt.args + (1,)), (True, False))),
            # 4:
            (
                self.lt,
                self.native,
                RangedExpression((self.lt.args + (5,)), (True, False)),
            ),
            (
                self.lt,
                self.npv,
                RangedExpression((self.lt.args + (self.npv,)), (True, False)),
            ),
            (
                self.lt,
                self.param,
                RangedExpression((self.lt.args + (6,)), (True, False)),
            ),
            (
                self.lt,
                self.param_mut,
                RangedExpression((self.lt.args + (self.param_mut,)), (True, False)),
            ),
            # 8:
            (
                self.lt,
                self.var,
                RangedExpression((self.lt.args + (self.var,)), (True, False)),
            ),
            (
                self.lt,
                self.mon_native,
                RangedExpression((self.lt.args + (self.mon_native,)), (True, False)),
            ),
            (
                self.lt,
                self.mon_param,
                RangedExpression((self.lt.args + (self.mon_param,)), (True, False)),
            ),
            (
                self.lt,
                self.mon_npv,
                RangedExpression((self.lt.args + (self.mon_npv,)), (True, False)),
            ),
            # 12:
            (
                self.lt,
                self.linear,
                RangedExpression((self.lt.args + (self.linear,)), (True, False)),
            ),
            (
                self.lt,
                self.sum,
                RangedExpression((self.lt.args + (self.sum,)), (True, False)),
            ),
            (
                self.lt,
                self.other,
                RangedExpression((self.lt.args + (self.other,)), (True, False)),
            ),
            (
                self.lt,
                self.mutable_l0,
                RangedExpression((self.lt.args + (self.l0,)), (True, False)),
            ),
            # 16:
            (
                self.lt,
                self.mutable_l1,
                RangedExpression((self.lt.args + (self.l1,)), (True, False)),
            ),
            (
                self.lt,
                self.mutable_l2,
                RangedExpression((self.lt.args + (self.l2,)), (True, False)),
            ),
            (
                self.lt,
                self.param0,
                RangedExpression((self.lt.args + (0,)), (True, False)),
            ),
            (
                self.lt,
                self.param1,
                RangedExpression((self.lt.args + (1,)), (True, False)),
            ),
            # 20:
            (
                self.lt,
                self.mutable_l3,
                RangedExpression((self.lt.args + (self.l3,)), (True, False)),
            ),
            (
                self.lt,
                self.eq,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
            (
                self.lt,
                self.le,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
            (
                self.lt,
                self.lt,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
            # 24
            (
                self.lt,
                self.ranged,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
        ]
        self._run_cases(tests, operator.le)

    def test_le_ranged(self):
        tests = [
            (self.ranged, self.invalid, NotImplemented),
            (
                self.ranged,
                self.asbinary,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.zero,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.one,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 4:
            (
                self.ranged,
                self.native,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.npv,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.param,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.param_mut,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 8:
            (
                self.ranged,
                self.var,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.mon_native,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.mon_param,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.mon_npv,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 12:
            (
                self.ranged,
                self.linear,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.sum,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.other,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.mutable_l0,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 16:
            (
                self.ranged,
                self.mutable_l1,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.mutable_l2,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.param0,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.param1,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 20:
            (
                self.ranged,
                self.mutable_l3,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.eq,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
            (
                self.ranged,
                self.le,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
            (
                self.ranged,
                self.lt,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
            # 24
            (
                self.ranged,
                self.ranged,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
        ]
        self._run_cases(tests, operator.le)


#
#
# INEQUALITY (strict)
#
#


class TestStrictInequality(BaseRelational, unittest.TestCase):

    def test_lt_invalid(self):
        tests = [
            # "invalid(str) == invalid(str)" is a legitimate Python
            # operation and should never hit the Pyomo expression
            # system
            (self.invalid, self.invalid, False),
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
            (self.invalid, self.eq, NotImplemented),
            (self.invalid, self.le, NotImplemented),
            (self.invalid, self.lt, NotImplemented),
            # 24:
            (self.invalid, self.ranged, NotImplemented),
        ]
        self._run_cases(tests, operator.lt)

    def test_lt_asbinary(self):
        tests = [
            (self.asbinary, self.invalid, NotImplemented),
            (
                self.asbinary,
                self.asbinary,
                InequalityExpression((self.bin, self.bin), True),
            ),
            (self.asbinary, self.zero, InequalityExpression((self.bin, 0), True)),
            (self.asbinary, self.one, InequalityExpression((self.bin, 1), True)),
            # 4:
            (self.asbinary, self.native, InequalityExpression((self.bin, 5), True)),
            (self.asbinary, self.npv, InequalityExpression((self.bin, self.npv), True)),
            (self.asbinary, self.param, InequalityExpression((self.bin, 6), True)),
            (
                self.asbinary,
                self.param_mut,
                InequalityExpression((self.bin, self.param_mut), True),
            ),
            # 8:
            (self.asbinary, self.var, InequalityExpression((self.bin, self.var), True)),
            (
                self.asbinary,
                self.mon_native,
                InequalityExpression((self.bin, self.mon_native), True),
            ),
            (
                self.asbinary,
                self.mon_param,
                InequalityExpression((self.bin, self.mon_param), True),
            ),
            (
                self.asbinary,
                self.mon_npv,
                InequalityExpression((self.bin, self.mon_npv), True),
            ),
            # 12:
            (
                self.asbinary,
                self.linear,
                InequalityExpression((self.bin, self.linear), True),
            ),
            (self.asbinary, self.sum, InequalityExpression((self.bin, self.sum), True)),
            (
                self.asbinary,
                self.other,
                InequalityExpression((self.bin, self.other), True),
            ),
            (
                self.asbinary,
                self.mutable_l0,
                InequalityExpression((self.bin, self.l0), True),
            ),
            # 16:
            (
                self.asbinary,
                self.mutable_l1,
                InequalityExpression((self.bin, self.l1), True),
            ),
            (
                self.asbinary,
                self.mutable_l2,
                InequalityExpression((self.bin, self.l2), True),
            ),
            (self.asbinary, self.param0, InequalityExpression((self.bin, 0), True)),
            (self.asbinary, self.param1, InequalityExpression((self.bin, 1), True)),
            # 20:
            (
                self.asbinary,
                self.mutable_l3,
                InequalityExpression((self.bin, self.l3), True),
            ),
            (
                self.asbinary,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.asbinary,
                self.le,
                RangedExpression((self.bin,) + self.le.args, (True, False)),
            ),
            (
                self.asbinary,
                self.lt,
                RangedExpression((self.bin,) + self.lt.args, (True, True)),
            ),
            # 24
            (
                self.asbinary,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.lt)

    def test_lt_zero(self):
        tests = [
            (self.zero, self.invalid, NotImplemented),
            (self.zero, self.asbinary, InequalityExpression((0, self.bin), True)),
            (self.zero, self.zero, False),
            (self.zero, self.one, True),
            # 4:
            (self.zero, self.native, True),
            (self.zero, self.npv, InequalityExpression((0, self.npv), True)),
            (self.zero, self.param, True),
            (
                self.zero,
                self.param_mut,
                InequalityExpression((0, self.param_mut), True),
            ),
            # 8:
            (self.zero, self.var, InequalityExpression((0, self.var), True)),
            (
                self.zero,
                self.mon_native,
                InequalityExpression((0, self.mon_native), True),
            ),
            (
                self.zero,
                self.mon_param,
                InequalityExpression((0, self.mon_param), True),
            ),
            (self.zero, self.mon_npv, InequalityExpression((0, self.mon_npv), True)),
            # 12:
            (self.zero, self.linear, InequalityExpression((0, self.linear), True)),
            (self.zero, self.sum, InequalityExpression((0, self.sum), True)),
            (self.zero, self.other, InequalityExpression((0, self.other), True)),
            (self.zero, self.mutable_l0, False),
            # 16:
            (self.zero, self.mutable_l1, InequalityExpression((0, self.l1), True)),
            (self.zero, self.mutable_l2, InequalityExpression((0, self.l2), True)),
            (self.zero, self.param0, False),
            (self.zero, self.param1, True),
            # 20:
            (self.zero, self.mutable_l3, InequalityExpression((0, self.l3), True)),
            (
                self.zero,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (self.zero, self.le, RangedExpression((0,) + self.le.args, (True, False))),
            (self.zero, self.lt, RangedExpression((0,) + self.lt.args, (True, True))),
            # 24
            (
                self.zero,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.lt)

    def test_lt_one(self):
        tests = [
            (self.one, self.invalid, NotImplemented),
            (self.one, self.asbinary, InequalityExpression((1, self.bin), True)),
            (self.one, self.zero, False),
            (self.one, self.one, False),
            # 4:
            (self.one, self.native, True),
            (self.one, self.npv, InequalityExpression((1, self.npv), True)),
            (self.one, self.param, True),
            (self.one, self.param_mut, InequalityExpression((1, self.param_mut), True)),
            # 8:
            (self.one, self.var, InequalityExpression((1, self.var), True)),
            (
                self.one,
                self.mon_native,
                InequalityExpression((1, self.mon_native), True),
            ),
            (self.one, self.mon_param, InequalityExpression((1, self.mon_param), True)),
            (self.one, self.mon_npv, InequalityExpression((1, self.mon_npv), True)),
            # 12:
            (self.one, self.linear, InequalityExpression((1, self.linear), True)),
            (self.one, self.sum, InequalityExpression((1, self.sum), True)),
            (self.one, self.other, InequalityExpression((1, self.other), True)),
            (self.one, self.mutable_l0, False),
            # 16:
            (self.one, self.mutable_l1, InequalityExpression((1, self.l1), True)),
            (self.one, self.mutable_l2, InequalityExpression((1, self.l2), True)),
            (self.one, self.param0, False),
            (self.one, self.param1, False),
            # 20:
            (self.one, self.mutable_l3, InequalityExpression((1, self.l3), True)),
            (
                self.one,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (self.one, self.le, RangedExpression((1,) + self.le.args, (True, False))),
            (self.one, self.lt, RangedExpression((1,) + self.lt.args, (True, True))),
            # 24
            (
                self.one,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.lt)

    def test_lt_native(self):
        tests = [
            (self.native, self.invalid, NotImplemented),
            (self.native, self.asbinary, InequalityExpression((5, self.bin), True)),
            (self.native, self.zero, False),
            (self.native, self.one, False),
            # 4:
            (self.native, self.native, False),
            (self.native, self.npv, InequalityExpression((5, self.npv), True)),
            (self.native, self.param, True),
            (
                self.native,
                self.param_mut,
                InequalityExpression((5, self.param_mut), True),
            ),
            # 8:
            (self.native, self.var, InequalityExpression((5, self.var), True)),
            (
                self.native,
                self.mon_native,
                InequalityExpression((5, self.mon_native), True),
            ),
            (
                self.native,
                self.mon_param,
                InequalityExpression((5, self.mon_param), True),
            ),
            (self.native, self.mon_npv, InequalityExpression((5, self.mon_npv), True)),
            # 12:
            (self.native, self.linear, InequalityExpression((5, self.linear), True)),
            (self.native, self.sum, InequalityExpression((5, self.sum), True)),
            (self.native, self.other, InequalityExpression((5, self.other), True)),
            (self.native, self.mutable_l0, False),
            # 16:
            (self.native, self.mutable_l1, InequalityExpression((5, self.l1), True)),
            (self.native, self.mutable_l2, InequalityExpression((5, self.l2), True)),
            (self.native, self.param0, False),
            (self.native, self.param1, False),
            # 20:
            (self.native, self.mutable_l3, InequalityExpression((5, self.l3), True)),
            (
                self.native,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.native,
                self.le,
                RangedExpression((5,) + self.le.args, (True, False)),
            ),
            (self.native, self.lt, RangedExpression((5,) + self.lt.args, (True, True))),
            # 24
            (
                self.native,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.lt)

    def test_lt_npv(self):
        tests = [
            (self.npv, self.invalid, NotImplemented),
            (self.npv, self.asbinary, InequalityExpression((self.npv, self.bin), True)),
            (self.npv, self.zero, InequalityExpression((self.npv, 0), True)),
            (self.npv, self.one, InequalityExpression((self.npv, 1), True)),
            # 4:
            (self.npv, self.native, InequalityExpression((self.npv, 5), True)),
            (self.npv, self.npv, InequalityExpression((self.npv, self.npv), True)),
            (self.npv, self.param, InequalityExpression((self.npv, 6), True)),
            (
                self.npv,
                self.param_mut,
                InequalityExpression((self.npv, self.param_mut), True),
            ),
            # 8:
            (self.npv, self.var, InequalityExpression((self.npv, self.var), True)),
            (
                self.npv,
                self.mon_native,
                InequalityExpression((self.npv, self.mon_native), True),
            ),
            (
                self.npv,
                self.mon_param,
                InequalityExpression((self.npv, self.mon_param), True),
            ),
            (
                self.npv,
                self.mon_npv,
                InequalityExpression((self.npv, self.mon_npv), True),
            ),
            # 12:
            (
                self.npv,
                self.linear,
                InequalityExpression((self.npv, self.linear), True),
            ),
            (self.npv, self.sum, InequalityExpression((self.npv, self.sum), True)),
            (self.npv, self.other, InequalityExpression((self.npv, self.other), True)),
            (
                self.npv,
                self.mutable_l0,
                InequalityExpression((self.npv, self.l0), True),
            ),
            # 16:
            (
                self.npv,
                self.mutable_l1,
                InequalityExpression((self.npv, self.l1), True),
            ),
            (
                self.npv,
                self.mutable_l2,
                InequalityExpression((self.npv, self.l2), True),
            ),
            (self.npv, self.param0, InequalityExpression((self.npv, 0), True)),
            (self.npv, self.param1, InequalityExpression((self.npv, 1), True)),
            # 20:
            (
                self.npv,
                self.mutable_l3,
                InequalityExpression((self.npv, self.l3), True),
            ),
            (
                self.npv,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.npv,
                self.le,
                RangedExpression((self.npv,) + self.le.args, (True, False)),
            ),
            (
                self.npv,
                self.lt,
                RangedExpression((self.npv,) + self.lt.args, (True, True)),
            ),
            # 24
            (
                self.npv,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.lt)

    def test_lt_param(self):
        tests = [
            (self.param, self.invalid, NotImplemented),
            (self.param, self.asbinary, InequalityExpression((6, self.bin), True)),
            (self.param, self.zero, False),
            (self.param, self.one, False),
            # 4:
            (self.param, self.native, False),
            (self.param, self.npv, InequalityExpression((6, self.npv), True)),
            (self.param, self.param, False),
            (
                self.param,
                self.param_mut,
                InequalityExpression((6, self.param_mut), True),
            ),
            # 8:
            (self.param, self.var, InequalityExpression((6, self.var), True)),
            (
                self.param,
                self.mon_native,
                InequalityExpression((6, self.mon_native), True),
            ),
            (
                self.param,
                self.mon_param,
                InequalityExpression((6, self.mon_param), True),
            ),
            (self.param, self.mon_npv, InequalityExpression((6, self.mon_npv), True)),
            # 12:
            (self.param, self.linear, InequalityExpression((6, self.linear), True)),
            (self.param, self.sum, InequalityExpression((6, self.sum), True)),
            (self.param, self.other, InequalityExpression((6, self.other), True)),
            (self.param, self.mutable_l0, False),
            # 16:
            (self.param, self.mutable_l1, InequalityExpression((6, self.l1), True)),
            (self.param, self.mutable_l2, InequalityExpression((6, self.l2), True)),
            (self.param, self.param0, False),
            (self.param, self.param1, False),
            # 20:
            (self.param, self.mutable_l3, InequalityExpression((6, self.l3), True)),
            (
                self.param,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (self.param, self.le, RangedExpression((6,) + self.le.args, (True, False))),
            (self.param, self.lt, RangedExpression((6,) + self.lt.args, (True, True))),
            # 24
            (
                self.param,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.lt)

    def test_lt_param_mut(self):
        tests = [
            (self.param_mut, self.invalid, NotImplemented),
            (
                self.param_mut,
                self.asbinary,
                InequalityExpression((self.param_mut, self.bin), True),
            ),
            (
                self.param_mut,
                self.zero,
                InequalityExpression((self.param_mut, 0), True),
            ),
            (self.param_mut, self.one, InequalityExpression((self.param_mut, 1), True)),
            # 4:
            (
                self.param_mut,
                self.native,
                InequalityExpression((self.param_mut, 5), True),
            ),
            (
                self.param_mut,
                self.npv,
                InequalityExpression((self.param_mut, self.npv), True),
            ),
            (
                self.param_mut,
                self.param,
                InequalityExpression((self.param_mut, 6), True),
            ),
            (
                self.param_mut,
                self.param_mut,
                InequalityExpression((self.param_mut, self.param_mut), True),
            ),
            # 8:
            (
                self.param_mut,
                self.var,
                InequalityExpression((self.param_mut, self.var), True),
            ),
            (
                self.param_mut,
                self.mon_native,
                InequalityExpression((self.param_mut, self.mon_native), True),
            ),
            (
                self.param_mut,
                self.mon_param,
                InequalityExpression((self.param_mut, self.mon_param), True),
            ),
            (
                self.param_mut,
                self.mon_npv,
                InequalityExpression((self.param_mut, self.mon_npv), True),
            ),
            # 12:
            (
                self.param_mut,
                self.linear,
                InequalityExpression((self.param_mut, self.linear), True),
            ),
            (
                self.param_mut,
                self.sum,
                InequalityExpression((self.param_mut, self.sum), True),
            ),
            (
                self.param_mut,
                self.other,
                InequalityExpression((self.param_mut, self.other), True),
            ),
            (
                self.param_mut,
                self.mutable_l0,
                InequalityExpression((self.param_mut, self.l0), True),
            ),
            # 16:
            (
                self.param_mut,
                self.mutable_l1,
                InequalityExpression((self.param_mut, self.l1), True),
            ),
            (
                self.param_mut,
                self.mutable_l2,
                InequalityExpression((self.param_mut, self.l2), True),
            ),
            (
                self.param_mut,
                self.param0,
                InequalityExpression((self.param_mut, 0), True),
            ),
            (
                self.param_mut,
                self.param1,
                InequalityExpression((self.param_mut, 1), True),
            ),
            # 20:
            (
                self.param_mut,
                self.mutable_l3,
                InequalityExpression((self.param_mut, self.l3), True),
            ),
            (
                self.param_mut,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.param_mut,
                self.le,
                RangedExpression((self.param_mut,) + self.le.args, (True, False)),
            ),
            (
                self.param_mut,
                self.lt,
                RangedExpression((self.param_mut,) + self.lt.args, (True, True)),
            ),
            # 24
            (
                self.param_mut,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.lt)

    def test_lt_var(self):
        tests = [
            (self.var, self.invalid, NotImplemented),
            (self.var, self.asbinary, InequalityExpression((self.var, self.bin), True)),
            (self.var, self.zero, InequalityExpression((self.var, 0), True)),
            (self.var, self.one, InequalityExpression((self.var, 1), True)),
            # 4:
            (self.var, self.native, InequalityExpression((self.var, 5), True)),
            (self.var, self.npv, InequalityExpression((self.var, self.npv), True)),
            (self.var, self.param, InequalityExpression((self.var, 6), True)),
            (
                self.var,
                self.param_mut,
                InequalityExpression((self.var, self.param_mut), True),
            ),
            # 8:
            (self.var, self.var, InequalityExpression((self.var, self.var), True)),
            (
                self.var,
                self.mon_native,
                InequalityExpression((self.var, self.mon_native), True),
            ),
            (
                self.var,
                self.mon_param,
                InequalityExpression((self.var, self.mon_param), True),
            ),
            (
                self.var,
                self.mon_npv,
                InequalityExpression((self.var, self.mon_npv), True),
            ),
            # 12:
            (
                self.var,
                self.linear,
                InequalityExpression((self.var, self.linear), True),
            ),
            (self.var, self.sum, InequalityExpression((self.var, self.sum), True)),
            (self.var, self.other, InequalityExpression((self.var, self.other), True)),
            (
                self.var,
                self.mutable_l0,
                InequalityExpression((self.var, self.l0), True),
            ),
            # 16:
            (
                self.var,
                self.mutable_l1,
                InequalityExpression((self.var, self.l1), True),
            ),
            (
                self.var,
                self.mutable_l2,
                InequalityExpression((self.var, self.l2), True),
            ),
            (self.var, self.param0, InequalityExpression((self.var, 0), True)),
            (self.var, self.param1, InequalityExpression((self.var, 1), True)),
            # 20:
            (
                self.var,
                self.mutable_l3,
                InequalityExpression((self.var, self.l3), True),
            ),
            (
                self.var,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.var,
                self.le,
                RangedExpression((self.var,) + self.le.args, (True, False)),
            ),
            (
                self.var,
                self.lt,
                RangedExpression((self.var,) + self.lt.args, (True, True)),
            ),
            # 24
            (
                self.var,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.lt)

    def test_lt_mon_native(self):
        tests = [
            (self.mon_native, self.invalid, NotImplemented),
            (
                self.mon_native,
                self.asbinary,
                InequalityExpression((self.mon_native, self.bin), True),
            ),
            (
                self.mon_native,
                self.zero,
                InequalityExpression((self.mon_native, 0), True),
            ),
            (
                self.mon_native,
                self.one,
                InequalityExpression((self.mon_native, 1), True),
            ),
            # 4:
            (
                self.mon_native,
                self.native,
                InequalityExpression((self.mon_native, 5), True),
            ),
            (
                self.mon_native,
                self.npv,
                InequalityExpression((self.mon_native, self.npv), True),
            ),
            (
                self.mon_native,
                self.param,
                InequalityExpression((self.mon_native, 6), True),
            ),
            (
                self.mon_native,
                self.param_mut,
                InequalityExpression((self.mon_native, self.param_mut), True),
            ),
            # 8:
            (
                self.mon_native,
                self.var,
                InequalityExpression((self.mon_native, self.var), True),
            ),
            (
                self.mon_native,
                self.mon_native,
                InequalityExpression((self.mon_native, self.mon_native), True),
            ),
            (
                self.mon_native,
                self.mon_param,
                InequalityExpression((self.mon_native, self.mon_param), True),
            ),
            (
                self.mon_native,
                self.mon_npv,
                InequalityExpression((self.mon_native, self.mon_npv), True),
            ),
            # 12:
            (
                self.mon_native,
                self.linear,
                InequalityExpression((self.mon_native, self.linear), True),
            ),
            (
                self.mon_native,
                self.sum,
                InequalityExpression((self.mon_native, self.sum), True),
            ),
            (
                self.mon_native,
                self.other,
                InequalityExpression((self.mon_native, self.other), True),
            ),
            (
                self.mon_native,
                self.mutable_l0,
                InequalityExpression((self.mon_native, self.l0), True),
            ),
            # 16:
            (
                self.mon_native,
                self.mutable_l1,
                InequalityExpression((self.mon_native, self.l1), True),
            ),
            (
                self.mon_native,
                self.mutable_l2,
                InequalityExpression((self.mon_native, self.l2), True),
            ),
            (
                self.mon_native,
                self.param0,
                InequalityExpression((self.mon_native, 0), True),
            ),
            (
                self.mon_native,
                self.param1,
                InequalityExpression((self.mon_native, 1), True),
            ),
            # 20:
            (
                self.mon_native,
                self.mutable_l3,
                InequalityExpression((self.mon_native, self.l3), True),
            ),
            (
                self.mon_native,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mon_native,
                self.le,
                RangedExpression((self.mon_native,) + self.le.args, (True, False)),
            ),
            (
                self.mon_native,
                self.lt,
                RangedExpression((self.mon_native,) + self.lt.args, (True, True)),
            ),
            # 24
            (
                self.mon_native,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.lt)

    def test_lt_mon_param(self):
        tests = [
            (self.mon_param, self.invalid, NotImplemented),
            (
                self.mon_param,
                self.asbinary,
                InequalityExpression((self.mon_param, self.bin), True),
            ),
            (
                self.mon_param,
                self.zero,
                InequalityExpression((self.mon_param, 0), True),
            ),
            (self.mon_param, self.one, InequalityExpression((self.mon_param, 1), True)),
            # 4:
            (
                self.mon_param,
                self.native,
                InequalityExpression((self.mon_param, 5), True),
            ),
            (
                self.mon_param,
                self.npv,
                InequalityExpression((self.mon_param, self.npv), True),
            ),
            (
                self.mon_param,
                self.param,
                InequalityExpression((self.mon_param, 6), True),
            ),
            (
                self.mon_param,
                self.param_mut,
                InequalityExpression((self.mon_param, self.param_mut), True),
            ),
            # 8:
            (
                self.mon_param,
                self.var,
                InequalityExpression((self.mon_param, self.var), True),
            ),
            (
                self.mon_param,
                self.mon_native,
                InequalityExpression((self.mon_param, self.mon_native), True),
            ),
            (
                self.mon_param,
                self.mon_param,
                InequalityExpression((self.mon_param, self.mon_param), True),
            ),
            (
                self.mon_param,
                self.mon_npv,
                InequalityExpression((self.mon_param, self.mon_npv), True),
            ),
            # 12:
            (
                self.mon_param,
                self.linear,
                InequalityExpression((self.mon_param, self.linear), True),
            ),
            (
                self.mon_param,
                self.sum,
                InequalityExpression((self.mon_param, self.sum), True),
            ),
            (
                self.mon_param,
                self.other,
                InequalityExpression((self.mon_param, self.other), True),
            ),
            (
                self.mon_param,
                self.mutable_l0,
                InequalityExpression((self.mon_param, self.l0), True),
            ),
            # 16:
            (
                self.mon_param,
                self.mutable_l1,
                InequalityExpression((self.mon_param, self.l1), True),
            ),
            (
                self.mon_param,
                self.mutable_l2,
                InequalityExpression((self.mon_param, self.l2), True),
            ),
            (
                self.mon_param,
                self.param0,
                InequalityExpression((self.mon_param, 0), True),
            ),
            (
                self.mon_param,
                self.param1,
                InequalityExpression((self.mon_param, 1), True),
            ),
            # 20:
            (
                self.mon_param,
                self.mutable_l3,
                InequalityExpression((self.mon_param, self.l3), True),
            ),
            (
                self.mon_param,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mon_param,
                self.le,
                RangedExpression((self.mon_param,) + self.le.args, (True, False)),
            ),
            (
                self.mon_param,
                self.lt,
                RangedExpression((self.mon_param,) + self.lt.args, (True, True)),
            ),
            # 24
            (
                self.mon_param,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.lt)

    def test_lt_mon_npv(self):
        tests = [
            (self.mon_npv, self.invalid, NotImplemented),
            (
                self.mon_npv,
                self.asbinary,
                InequalityExpression((self.mon_npv, self.bin), True),
            ),
            (self.mon_npv, self.zero, InequalityExpression((self.mon_npv, 0), True)),
            (self.mon_npv, self.one, InequalityExpression((self.mon_npv, 1), True)),
            # 4:
            (self.mon_npv, self.native, InequalityExpression((self.mon_npv, 5), True)),
            (
                self.mon_npv,
                self.npv,
                InequalityExpression((self.mon_npv, self.npv), True),
            ),
            (self.mon_npv, self.param, InequalityExpression((self.mon_npv, 6), True)),
            (
                self.mon_npv,
                self.param_mut,
                InequalityExpression((self.mon_npv, self.param_mut), True),
            ),
            # 8:
            (
                self.mon_npv,
                self.var,
                InequalityExpression((self.mon_npv, self.var), True),
            ),
            (
                self.mon_npv,
                self.mon_native,
                InequalityExpression((self.mon_npv, self.mon_native), True),
            ),
            (
                self.mon_npv,
                self.mon_param,
                InequalityExpression((self.mon_npv, self.mon_param), True),
            ),
            (
                self.mon_npv,
                self.mon_npv,
                InequalityExpression((self.mon_npv, self.mon_npv), True),
            ),
            # 12:
            (
                self.mon_npv,
                self.linear,
                InequalityExpression((self.mon_npv, self.linear), True),
            ),
            (
                self.mon_npv,
                self.sum,
                InequalityExpression((self.mon_npv, self.sum), True),
            ),
            (
                self.mon_npv,
                self.other,
                InequalityExpression((self.mon_npv, self.other), True),
            ),
            (
                self.mon_npv,
                self.mutable_l0,
                InequalityExpression((self.mon_npv, self.l0), True),
            ),
            # 16:
            (
                self.mon_npv,
                self.mutable_l1,
                InequalityExpression((self.mon_npv, self.l1), True),
            ),
            (
                self.mon_npv,
                self.mutable_l2,
                InequalityExpression((self.mon_npv, self.l2), True),
            ),
            (self.mon_npv, self.param0, InequalityExpression((self.mon_npv, 0), True)),
            (self.mon_npv, self.param1, InequalityExpression((self.mon_npv, 1), True)),
            # 20:
            (
                self.mon_npv,
                self.mutable_l3,
                InequalityExpression((self.mon_npv, self.l3), True),
            ),
            (
                self.mon_npv,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mon_npv,
                self.le,
                RangedExpression((self.mon_npv,) + self.le.args, (True, False)),
            ),
            (
                self.mon_npv,
                self.lt,
                RangedExpression((self.mon_npv,) + self.lt.args, (True, True)),
            ),
            # 24
            (
                self.mon_npv,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.lt)

    def test_lt_linear(self):
        tests = [
            (self.linear, self.invalid, NotImplemented),
            (
                self.linear,
                self.asbinary,
                InequalityExpression((self.linear, self.bin), True),
            ),
            (self.linear, self.zero, InequalityExpression((self.linear, 0), True)),
            (self.linear, self.one, InequalityExpression((self.linear, 1), True)),
            # 4:
            (self.linear, self.native, InequalityExpression((self.linear, 5), True)),
            (
                self.linear,
                self.npv,
                InequalityExpression((self.linear, self.npv), True),
            ),
            (self.linear, self.param, InequalityExpression((self.linear, 6), True)),
            (
                self.linear,
                self.param_mut,
                InequalityExpression((self.linear, self.param_mut), True),
            ),
            # 8:
            (
                self.linear,
                self.var,
                InequalityExpression((self.linear, self.var), True),
            ),
            (
                self.linear,
                self.mon_native,
                InequalityExpression((self.linear, self.mon_native), True),
            ),
            (
                self.linear,
                self.mon_param,
                InequalityExpression((self.linear, self.mon_param), True),
            ),
            (
                self.linear,
                self.mon_npv,
                InequalityExpression((self.linear, self.mon_npv), True),
            ),
            # 12:
            (
                self.linear,
                self.linear,
                InequalityExpression((self.linear, self.linear), True),
            ),
            (
                self.linear,
                self.sum,
                InequalityExpression((self.linear, self.sum), True),
            ),
            (
                self.linear,
                self.other,
                InequalityExpression((self.linear, self.other), True),
            ),
            (
                self.linear,
                self.mutable_l0,
                InequalityExpression((self.linear, self.l0), True),
            ),
            # 16:
            (
                self.linear,
                self.mutable_l1,
                InequalityExpression((self.linear, self.l1), True),
            ),
            (
                self.linear,
                self.mutable_l2,
                InequalityExpression((self.linear, self.l2), True),
            ),
            (self.linear, self.param0, InequalityExpression((self.linear, 0), True)),
            (self.linear, self.param1, InequalityExpression((self.linear, 1), True)),
            # 20:
            (
                self.linear,
                self.mutable_l3,
                InequalityExpression((self.linear, self.l3), True),
            ),
            (
                self.linear,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.linear,
                self.le,
                RangedExpression(((self.linear,) + self.le.args), (True, False)),
            ),
            (
                self.linear,
                self.lt,
                RangedExpression(((self.linear,) + self.lt.args), (True, True)),
            ),
            # 24
            (
                self.linear,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.lt)

    def test_lt_sum(self):
        tests = [
            (self.sum, self.invalid, NotImplemented),
            (self.sum, self.asbinary, InequalityExpression((self.sum, self.bin), True)),
            (self.sum, self.zero, InequalityExpression((self.sum, 0), True)),
            (self.sum, self.one, InequalityExpression((self.sum, 1), True)),
            # 4:
            (self.sum, self.native, InequalityExpression((self.sum, 5), True)),
            (self.sum, self.npv, InequalityExpression((self.sum, self.npv), True)),
            (self.sum, self.param, InequalityExpression((self.sum, 6), True)),
            (
                self.sum,
                self.param_mut,
                InequalityExpression((self.sum, self.param_mut), True),
            ),
            # 8:
            (self.sum, self.var, InequalityExpression((self.sum, self.var), True)),
            (
                self.sum,
                self.mon_native,
                InequalityExpression((self.sum, self.mon_native), True),
            ),
            (
                self.sum,
                self.mon_param,
                InequalityExpression((self.sum, self.mon_param), True),
            ),
            (
                self.sum,
                self.mon_npv,
                InequalityExpression((self.sum, self.mon_npv), True),
            ),
            # 12:
            (
                self.sum,
                self.linear,
                InequalityExpression((self.sum, self.linear), True),
            ),
            (self.sum, self.sum, InequalityExpression((self.sum, self.sum), True)),
            (self.sum, self.other, InequalityExpression((self.sum, self.other), True)),
            (
                self.sum,
                self.mutable_l0,
                InequalityExpression((self.sum, self.l0), True),
            ),
            # 16:
            (
                self.sum,
                self.mutable_l1,
                InequalityExpression((self.sum, self.l1), True),
            ),
            (
                self.sum,
                self.mutable_l2,
                InequalityExpression((self.sum, self.l2), True),
            ),
            (self.sum, self.param0, InequalityExpression((self.sum, 0), True)),
            (self.sum, self.param1, InequalityExpression((self.sum, 1), True)),
            # 20:
            (
                self.sum,
                self.mutable_l3,
                InequalityExpression((self.sum, self.l3), True),
            ),
            (
                self.sum,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.sum,
                self.le,
                RangedExpression((self.sum,) + self.le.args, (True, False)),
            ),
            (
                self.sum,
                self.lt,
                RangedExpression((self.sum,) + self.lt.args, (True, True)),
            ),
            # 24
            (
                self.sum,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.lt)

    def test_lt_other(self):
        tests = [
            (self.other, self.invalid, NotImplemented),
            (
                self.other,
                self.asbinary,
                InequalityExpression((self.other, self.bin), True),
            ),
            (self.other, self.zero, InequalityExpression((self.other, 0), True)),
            (self.other, self.one, InequalityExpression((self.other, 1), True)),
            # 4:
            (self.other, self.native, InequalityExpression((self.other, 5), True)),
            (self.other, self.npv, InequalityExpression((self.other, self.npv), True)),
            (self.other, self.param, InequalityExpression((self.other, 6), True)),
            (
                self.other,
                self.param_mut,
                InequalityExpression((self.other, self.param_mut), True),
            ),
            # 8:
            (self.other, self.var, InequalityExpression((self.other, self.var), True)),
            (
                self.other,
                self.mon_native,
                InequalityExpression((self.other, self.mon_native), True),
            ),
            (
                self.other,
                self.mon_param,
                InequalityExpression((self.other, self.mon_param), True),
            ),
            (
                self.other,
                self.mon_npv,
                InequalityExpression((self.other, self.mon_npv), True),
            ),
            # 12:
            (
                self.other,
                self.linear,
                InequalityExpression((self.other, self.linear), True),
            ),
            (self.other, self.sum, InequalityExpression((self.other, self.sum), True)),
            (
                self.other,
                self.other,
                InequalityExpression((self.other, self.other), True),
            ),
            (
                self.other,
                self.mutable_l0,
                InequalityExpression((self.other, self.l0), True),
            ),
            # 16:
            (
                self.other,
                self.mutable_l1,
                InequalityExpression((self.other, self.l1), True),
            ),
            (
                self.other,
                self.mutable_l2,
                InequalityExpression((self.other, self.l2), True),
            ),
            (self.other, self.param0, InequalityExpression((self.other, 0), True)),
            (self.other, self.param1, InequalityExpression((self.other, 1), True)),
            # 20:
            (
                self.other,
                self.mutable_l3,
                InequalityExpression((self.other, self.l3), True),
            ),
            (
                self.other,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.other,
                self.le,
                RangedExpression((self.other,) + self.le.args, (True, False)),
            ),
            (
                self.other,
                self.lt,
                RangedExpression((self.other,) + self.lt.args, (True, True)),
            ),
            # 24
            (
                self.other,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.lt)

    def test_lt_mutable_l0(self):
        tests = [
            (self.mutable_l0, self.invalid, NotImplemented),
            (
                self.mutable_l0,
                self.asbinary,
                InequalityExpression((self.l0, self.bin), True),
            ),
            (self.mutable_l0, self.zero, False),
            (self.mutable_l0, self.one, True),
            # 4:
            (self.mutable_l0, self.native, True),
            (
                self.mutable_l0,
                self.npv,
                InequalityExpression((self.l0, self.npv), True),
            ),
            (self.mutable_l0, self.param, True),
            (
                self.mutable_l0,
                self.param_mut,
                InequalityExpression((self.l0, self.param_mut), True),
            ),
            # 8:
            (
                self.mutable_l0,
                self.var,
                InequalityExpression((self.l0, self.var), True),
            ),
            (
                self.mutable_l0,
                self.mon_native,
                InequalityExpression((self.l0, self.mon_native), True),
            ),
            (
                self.mutable_l0,
                self.mon_param,
                InequalityExpression((self.l0, self.mon_param), True),
            ),
            (
                self.mutable_l0,
                self.mon_npv,
                InequalityExpression((self.l0, self.mon_npv), True),
            ),
            # 12:
            (
                self.mutable_l0,
                self.linear,
                InequalityExpression((self.l0, self.linear), True),
            ),
            (
                self.mutable_l0,
                self.sum,
                InequalityExpression((self.l0, self.sum), True),
            ),
            (
                self.mutable_l0,
                self.other,
                InequalityExpression((self.l0, self.other), True),
            ),
            (self.mutable_l0, self.mutable_l0, False),
            # 16:
            (
                self.mutable_l0,
                self.mutable_l1,
                InequalityExpression((self.l0, self.l1), True),
            ),
            (
                self.mutable_l0,
                self.mutable_l2,
                InequalityExpression((self.l0, self.l2), True),
            ),
            (self.mutable_l0, self.param0, False),
            (self.mutable_l0, self.param1, True),
            # 20:
            (
                self.mutable_l0,
                self.mutable_l3,
                InequalityExpression((self.l0, self.l3), True),
            ),
            (
                self.mutable_l0,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mutable_l0,
                self.le,
                RangedExpression((self.l0,) + self.le.args, (True, False)),
            ),
            (
                self.mutable_l0,
                self.lt,
                RangedExpression((self.l0,) + self.lt.args, (True, True)),
            ),
            # 24
            (
                self.mutable_l0,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.lt)

    def test_lt_mutable_l1(self):
        tests = [
            (self.mutable_l1, self.invalid, NotImplemented),
            (
                self.mutable_l1,
                self.asbinary,
                InequalityExpression((self.l1, self.bin), True),
            ),
            (self.mutable_l1, self.zero, InequalityExpression((self.l1, 0), True)),
            (self.mutable_l1, self.one, InequalityExpression((self.l1, 1), True)),
            # 4:
            (self.mutable_l1, self.native, InequalityExpression((self.l1, 5), True)),
            (
                self.mutable_l1,
                self.npv,
                InequalityExpression((self.l1, self.npv), True),
            ),
            (self.mutable_l1, self.param, InequalityExpression((self.l1, 6), True)),
            (
                self.mutable_l1,
                self.param_mut,
                InequalityExpression((self.l1, self.param_mut), True),
            ),
            # 8:
            (
                self.mutable_l1,
                self.var,
                InequalityExpression((self.l1, self.var), True),
            ),
            (
                self.mutable_l1,
                self.mon_native,
                InequalityExpression((self.l1, self.mon_native), True),
            ),
            (
                self.mutable_l1,
                self.mon_param,
                InequalityExpression((self.l1, self.mon_param), True),
            ),
            (
                self.mutable_l1,
                self.mon_npv,
                InequalityExpression((self.l1, self.mon_npv), True),
            ),
            # 12:
            (
                self.mutable_l1,
                self.linear,
                InequalityExpression((self.l1, self.linear), True),
            ),
            (
                self.mutable_l1,
                self.sum,
                InequalityExpression((self.l1, self.sum), True),
            ),
            (
                self.mutable_l1,
                self.other,
                InequalityExpression((self.l1, self.other), True),
            ),
            (
                self.mutable_l1,
                self.mutable_l0,
                InequalityExpression((self.l1, self.l0), True),
            ),
            # 16:
            (
                self.mutable_l1,
                self.mutable_l1,
                InequalityExpression((self.l1, self.l1), True),
            ),
            (
                self.mutable_l1,
                self.mutable_l2,
                InequalityExpression((self.l1, self.l2), True),
            ),
            (self.mutable_l1, self.param0, InequalityExpression((self.l1, 0), True)),
            (self.mutable_l1, self.param1, InequalityExpression((self.l1, 1), True)),
            # 20:
            (
                self.mutable_l1,
                self.mutable_l3,
                InequalityExpression((self.l1, self.l3), True),
            ),
            (
                self.mutable_l1,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mutable_l1,
                self.le,
                RangedExpression((self.l1,) + self.le.args, (True, False)),
            ),
            (
                self.mutable_l1,
                self.lt,
                RangedExpression((self.l1,) + self.lt.args, (True, True)),
            ),
            # 24
            (
                self.mutable_l1,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.lt)

    def test_lt_mutable_l2(self):
        tests = [
            (self.mutable_l2, self.invalid, NotImplemented),
            (
                self.mutable_l2,
                self.asbinary,
                InequalityExpression((self.l2, self.bin), True),
            ),
            (self.mutable_l2, self.zero, InequalityExpression((self.l2, 0), True)),
            (self.mutable_l2, self.one, InequalityExpression((self.l2, 1), True)),
            # 4:
            (self.mutable_l2, self.native, InequalityExpression((self.l2, 5), True)),
            (
                self.mutable_l2,
                self.npv,
                InequalityExpression((self.l2, self.npv), True),
            ),
            (self.mutable_l2, self.param, InequalityExpression((self.l2, 6), True)),
            (
                self.mutable_l2,
                self.param_mut,
                InequalityExpression((self.l2, self.param_mut), True),
            ),
            # 8:
            (
                self.mutable_l2,
                self.var,
                InequalityExpression((self.l2, self.var), True),
            ),
            (
                self.mutable_l2,
                self.mon_native,
                InequalityExpression((self.l2, self.mon_native), True),
            ),
            (
                self.mutable_l2,
                self.mon_param,
                InequalityExpression((self.l2, self.mon_param), True),
            ),
            (
                self.mutable_l2,
                self.mon_npv,
                InequalityExpression((self.l2, self.mon_npv), True),
            ),
            # 12:
            (
                self.mutable_l2,
                self.linear,
                InequalityExpression((self.l2, self.linear), True),
            ),
            (
                self.mutable_l2,
                self.sum,
                InequalityExpression((self.l2, self.sum), True),
            ),
            (
                self.mutable_l2,
                self.other,
                InequalityExpression((self.l2, self.other), True),
            ),
            (
                self.mutable_l2,
                self.mutable_l0,
                InequalityExpression((self.l2, self.l0), True),
            ),
            # 16:
            (
                self.mutable_l2,
                self.mutable_l1,
                InequalityExpression((self.l2, self.l1), True),
            ),
            (
                self.mutable_l2,
                self.mutable_l2,
                InequalityExpression((self.l2, self.l2), True),
            ),
            (self.mutable_l2, self.param0, InequalityExpression((self.l2, 0), True)),
            (self.mutable_l2, self.param1, InequalityExpression((self.l2, 1), True)),
            # 20:
            (
                self.mutable_l2,
                self.mutable_l3,
                InequalityExpression((self.l2, self.l3), True),
            ),
            (
                self.mutable_l2,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mutable_l2,
                self.le,
                RangedExpression((self.l2,) + self.le.args, (True, False)),
            ),
            (
                self.mutable_l2,
                self.lt,
                RangedExpression((self.l2,) + self.lt.args, (True, True)),
            ),
            # 24
            (
                self.mutable_l2,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.lt)

    def test_lt_param0(self):
        tests = [
            (self.param0, self.invalid, NotImplemented),
            (self.param0, self.asbinary, InequalityExpression((0, self.bin), True)),
            (self.param0, self.zero, False),
            (self.param0, self.one, True),
            # 4:
            (self.param0, self.native, True),
            (self.param0, self.npv, InequalityExpression((0, self.npv), True)),
            (self.param0, self.param, True),
            (
                self.param0,
                self.param_mut,
                InequalityExpression((0, self.param_mut), True),
            ),
            # 8:
            (self.param0, self.var, InequalityExpression((0, self.var), True)),
            (
                self.param0,
                self.mon_native,
                InequalityExpression((0, self.mon_native), True),
            ),
            (
                self.param0,
                self.mon_param,
                InequalityExpression((0, self.mon_param), True),
            ),
            (self.param0, self.mon_npv, InequalityExpression((0, self.mon_npv), True)),
            # 12:
            (self.param0, self.linear, InequalityExpression((0, self.linear), True)),
            (self.param0, self.sum, InequalityExpression((0, self.sum), True)),
            (self.param0, self.other, InequalityExpression((0, self.other), True)),
            (self.param0, self.mutable_l0, False),
            # 16:
            (self.param0, self.mutable_l1, InequalityExpression((0, self.l1), True)),
            (self.param0, self.mutable_l2, InequalityExpression((0, self.l2), True)),
            (self.param0, self.param0, False),
            (self.param0, self.param1, True),
            # 20:
            (self.param0, self.mutable_l3, InequalityExpression((0, self.l3), True)),
            (
                self.param0,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.param0,
                self.le,
                RangedExpression((0,) + self.le.args, (True, False)),
            ),
            (self.param0, self.lt, RangedExpression((0,) + self.lt.args, (True, True))),
            # 24
            (
                self.param0,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.lt)

    def test_lt_param1(self):
        tests = [
            (self.param1, self.invalid, NotImplemented),
            (self.param1, self.asbinary, InequalityExpression((1, self.bin), True)),
            (self.param1, self.zero, False),
            (self.param1, self.one, False),
            # 4:
            (self.param1, self.native, True),
            (self.param1, self.npv, InequalityExpression((1, self.npv), True)),
            (self.param1, self.param, True),
            (
                self.param1,
                self.param_mut,
                InequalityExpression((1, self.param_mut), True),
            ),
            # 8:
            (self.param1, self.var, InequalityExpression((1, self.var), True)),
            (
                self.param1,
                self.mon_native,
                InequalityExpression((1, self.mon_native), True),
            ),
            (
                self.param1,
                self.mon_param,
                InequalityExpression((1, self.mon_param), True),
            ),
            (self.param1, self.mon_npv, InequalityExpression((1, self.mon_npv), True)),
            # 12:
            (self.param1, self.linear, InequalityExpression((1, self.linear), True)),
            (self.param1, self.sum, InequalityExpression((1, self.sum), True)),
            (self.param1, self.other, InequalityExpression((1, self.other), True)),
            (self.param1, self.mutable_l0, False),
            # 16:
            (self.param1, self.mutable_l1, InequalityExpression((1, self.l1), True)),
            (self.param1, self.mutable_l2, InequalityExpression((1, self.l2), True)),
            (self.param1, self.param0, False),
            (self.param1, self.param1, False),
            # 20:
            (self.param1, self.mutable_l3, InequalityExpression((1, self.l3), True)),
            (
                self.param1,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.param1,
                self.le,
                RangedExpression((1,) + self.le.args, (True, False)),
            ),
            (self.param1, self.lt, RangedExpression((1,) + self.lt.args, (True, True))),
            # 24
            (
                self.param1,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.lt)

    def test_lt_mutable_l3(self):
        tests = [
            (self.mutable_l3, self.invalid, NotImplemented),
            (
                self.mutable_l3,
                self.asbinary,
                InequalityExpression((self.l3, self.bin), True),
            ),
            (self.mutable_l3, self.zero, InequalityExpression((self.l3, 0), True)),
            (self.mutable_l3, self.one, InequalityExpression((self.l3, 1), True)),
            # 4:
            (self.mutable_l3, self.native, InequalityExpression((self.l3, 5), True)),
            (
                self.mutable_l3,
                self.npv,
                InequalityExpression((self.l3, self.npv), True),
            ),
            (self.mutable_l3, self.param, InequalityExpression((self.l3, 6), True)),
            (
                self.mutable_l3,
                self.param_mut,
                InequalityExpression((self.l3, self.param_mut), True),
            ),
            # 8:
            (
                self.mutable_l3,
                self.var,
                InequalityExpression((self.l3, self.var), True),
            ),
            (
                self.mutable_l3,
                self.mon_native,
                InequalityExpression((self.l3, self.mon_native), True),
            ),
            (
                self.mutable_l3,
                self.mon_param,
                InequalityExpression((self.l3, self.mon_param), True),
            ),
            (
                self.mutable_l3,
                self.mon_npv,
                InequalityExpression((self.l3, self.mon_npv), True),
            ),
            # 12:
            (
                self.mutable_l3,
                self.linear,
                InequalityExpression((self.l3, self.linear), True),
            ),
            (
                self.mutable_l3,
                self.sum,
                InequalityExpression((self.l3, self.sum), True),
            ),
            (
                self.mutable_l3,
                self.other,
                InequalityExpression((self.l3, self.other), True),
            ),
            (
                self.mutable_l3,
                self.mutable_l0,
                InequalityExpression((self.l3, self.l0), True),
            ),
            # 16:
            (
                self.mutable_l3,
                self.mutable_l1,
                InequalityExpression((self.l3, self.l1), True),
            ),
            (
                self.mutable_l3,
                self.mutable_l2,
                InequalityExpression((self.l3, self.l2), True),
            ),
            (self.mutable_l3, self.param0, InequalityExpression((self.l3, 0), True)),
            (self.mutable_l3, self.param1, InequalityExpression((self.l3, 1), True)),
            # 20:
            (
                self.mutable_l3,
                self.mutable_l3,
                InequalityExpression((self.l3, self.l3), True),
            ),
            (
                self.mutable_l3,
                self.eq,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.mutable_l3,
                self.le,
                RangedExpression((self.l3,) + self.le.args, (True, False)),
            ),
            (
                self.mutable_l3,
                self.lt,
                RangedExpression((self.l3,) + self.lt.args, (True, True)),
            ),
            # 24
            (
                self.mutable_l3,
                self.ranged,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
        ]
        self._run_cases(tests, operator.lt)

    def test_lt_eq(self):
        tests = [
            (self.eq, self.invalid, NotImplemented),
            (
                self.eq,
                self.asbinary,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.zero,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.one,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 4:
            (
                self.eq,
                self.native,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.npv,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.param,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.param_mut,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 8:
            (
                self.eq,
                self.var,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.mon_native,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.mon_param,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.mon_npv,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 12:
            (
                self.eq,
                self.linear,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.sum,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.other,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.mutable_l0,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 16:
            (
                self.eq,
                self.mutable_l1,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.mutable_l2,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.param0,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.param1,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 20:
            (
                self.eq,
                self.mutable_l3,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.eq,
                self.eq,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
            (
                self.eq,
                self.le,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
            (
                self.eq,
                self.lt,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
            # 24
            (
                self.eq,
                self.ranged,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
        ]
        self._run_cases(tests, operator.lt)

    def test_lt_le(self):
        tests = [
            (self.le, self.invalid, NotImplemented),
            (
                self.le,
                self.asbinary,
                RangedExpression((self.le.args + (self.bin,)), (False, True)),
            ),
            (
                self.le,
                self.zero,
                RangedExpression((self.le.args + (0,)), (False, True)),
            ),
            (self.le, self.one, RangedExpression((self.le.args + (1,)), (False, True))),
            # 4:
            (
                self.le,
                self.native,
                RangedExpression((self.le.args + (5,)), (False, True)),
            ),
            (
                self.le,
                self.npv,
                RangedExpression((self.le.args + (self.npv,)), (False, True)),
            ),
            (
                self.le,
                self.param,
                RangedExpression((self.le.args + (6,)), (False, True)),
            ),
            (
                self.le,
                self.param_mut,
                RangedExpression((self.le.args + (self.param_mut,)), (False, True)),
            ),
            # 8:
            (
                self.le,
                self.var,
                RangedExpression((self.le.args + (self.var,)), (False, True)),
            ),
            (
                self.le,
                self.mon_native,
                RangedExpression((self.le.args + (self.mon_native,)), (False, True)),
            ),
            (
                self.le,
                self.mon_param,
                RangedExpression((self.le.args + (self.mon_param,)), (False, True)),
            ),
            (
                self.le,
                self.mon_npv,
                RangedExpression((self.le.args + (self.mon_npv,)), (False, True)),
            ),
            # 12:
            (
                self.le,
                self.linear,
                RangedExpression((self.le.args + (self.linear,)), (False, True)),
            ),
            (
                self.le,
                self.sum,
                RangedExpression((self.le.args + (self.sum,)), (False, True)),
            ),
            (
                self.le,
                self.other,
                RangedExpression((self.le.args + (self.other,)), (False, True)),
            ),
            (
                self.le,
                self.mutable_l0,
                RangedExpression((self.le.args + (self.l0,)), (False, True)),
            ),
            # 16:
            (
                self.le,
                self.mutable_l1,
                RangedExpression((self.le.args + (self.l1,)), (False, True)),
            ),
            (
                self.le,
                self.mutable_l2,
                RangedExpression((self.le.args + (self.l2,)), (False, True)),
            ),
            (
                self.le,
                self.param0,
                RangedExpression((self.le.args + (0,)), (False, True)),
            ),
            (
                self.le,
                self.param1,
                RangedExpression((self.le.args + (1,)), (False, True)),
            ),
            # 20:
            (
                self.le,
                self.mutable_l3,
                RangedExpression((self.le.args + (self.l3,)), (False, True)),
            ),
            (
                self.le,
                self.eq,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
            (
                self.le,
                self.le,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
            (
                self.le,
                self.lt,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
            # 24
            (
                self.le,
                self.ranged,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
        ]
        self._run_cases(tests, operator.lt)

    def test_lt_lt(self):
        tests = [
            (self.lt, self.invalid, NotImplemented),
            (
                self.lt,
                self.asbinary,
                RangedExpression((self.lt.args + (self.bin,)), (True, True)),
            ),
            (self.lt, self.zero, RangedExpression((self.lt.args + (0,)), (True, True))),
            (self.lt, self.one, RangedExpression((self.lt.args + (1,)), (True, True))),
            # 4:
            (
                self.lt,
                self.native,
                RangedExpression((self.lt.args + (5,)), (True, True)),
            ),
            (
                self.lt,
                self.npv,
                RangedExpression((self.lt.args + (self.npv,)), (True, True)),
            ),
            (
                self.lt,
                self.param,
                RangedExpression((self.lt.args + (6,)), (True, True)),
            ),
            (
                self.lt,
                self.param_mut,
                RangedExpression((self.lt.args + (self.param_mut,)), (True, True)),
            ),
            # 8:
            (
                self.lt,
                self.var,
                RangedExpression((self.lt.args + (self.var,)), (True, True)),
            ),
            (
                self.lt,
                self.mon_native,
                RangedExpression((self.lt.args + (self.mon_native,)), (True, True)),
            ),
            (
                self.lt,
                self.mon_param,
                RangedExpression((self.lt.args + (self.mon_param,)), (True, True)),
            ),
            (
                self.lt,
                self.mon_npv,
                RangedExpression((self.lt.args + (self.mon_npv,)), (True, True)),
            ),
            # 12:
            (
                self.lt,
                self.linear,
                RangedExpression((self.lt.args + (self.linear,)), (True, True)),
            ),
            (
                self.lt,
                self.sum,
                RangedExpression((self.lt.args + (self.sum,)), (True, True)),
            ),
            (
                self.lt,
                self.other,
                RangedExpression((self.lt.args + (self.other,)), (True, True)),
            ),
            (
                self.lt,
                self.mutable_l0,
                RangedExpression((self.lt.args + (self.l0,)), (True, True)),
            ),
            # 16:
            (
                self.lt,
                self.mutable_l1,
                RangedExpression((self.lt.args + (self.l1,)), (True, True)),
            ),
            (
                self.lt,
                self.mutable_l2,
                RangedExpression((self.lt.args + (self.l2,)), (True, True)),
            ),
            (
                self.lt,
                self.param0,
                RangedExpression((self.lt.args + (0,)), (True, True)),
            ),
            (
                self.lt,
                self.param1,
                RangedExpression((self.lt.args + (1,)), (True, True)),
            ),
            # 20:
            (
                self.lt,
                self.mutable_l3,
                RangedExpression((self.lt.args + (self.l3,)), (True, True)),
            ),
            (
                self.lt,
                self.eq,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
            (
                self.lt,
                self.le,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
            (
                self.lt,
                self.lt,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
            # 24
            (
                self.lt,
                self.ranged,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
        ]
        self._run_cases(tests, operator.lt)

    def test_lt_ranged(self):
        tests = [
            (self.ranged, self.invalid, NotImplemented),
            (
                self.ranged,
                self.asbinary,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.zero,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.one,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 4:
            (
                self.ranged,
                self.native,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.npv,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.param,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.param_mut,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 8:
            (
                self.ranged,
                self.var,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.mon_native,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.mon_param,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.mon_npv,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 12:
            (
                self.ranged,
                self.linear,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.sum,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.other,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.mutable_l0,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 16:
            (
                self.ranged,
                self.mutable_l1,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.mutable_l2,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.param0,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.param1,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            # 20:
            (
                self.ranged,
                self.mutable_l3,
                "Cannot create an InequalityExpression where one of the "
                "sub-expressions is a relational expression",
            ),
            (
                self.ranged,
                self.eq,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
            (
                self.ranged,
                self.le,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
            (
                self.ranged,
                self.lt,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
            # 24
            (
                self.ranged,
                self.ranged,
                "Cannot create an InequalityExpression where both "
                "sub-expressions are relational expressions",
            ),
        ]
        self._run_cases(tests, operator.lt)
