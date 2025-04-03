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

from pyomo.core.expr import inequality, EqualityExpression

from pyomo.core.tests.unit.test_numeric_expr_dispatcher import Base

logger = logging.getLogger(__name__)


class BaseRelational(Base, unittest.TestCase):
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
        self.lt = self.m.x < self.m.q
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
