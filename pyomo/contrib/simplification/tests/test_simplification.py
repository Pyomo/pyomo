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

from pyomo.common import unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.contrib.simplification import Simplifier
from pyomo.contrib.simplification.simplify import ginac_available
from pyomo.core.expr.compare import assertExpressionsEqual, compare_expressions
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd
from pyomo.core.expr.sympy_tools import sympy_available

import pyomo.environ as pyo


class SimplificationMixin:
    def compare_against_possible_results(self, got, expected_list):
        success = False
        for exp in expected_list:
            if compare_expressions(got, exp):
                success = True
                break
        self.assertTrue(success)

    def test_simplify(self):
        m = pyo.ConcreteModel()
        x = m.x = pyo.Var(bounds=(0, None))
        e = x * pyo.log(x)
        der1 = reverse_sd(e)[x]
        der2 = reverse_sd(der1)[x]
        der2_simp = self.simp.simplify(der2)
        expected = x**-1.0
        assertExpressionsEqual(self, expected, der2_simp)

    def test_mul(self):
        m = pyo.ConcreteModel()
        x = m.x = pyo.Var()
        e = 2 * x
        e2 = self.simp.simplify(e)
        expected = 2.0 * x
        assertExpressionsEqual(self, expected, e2)

    def test_sum(self):
        m = pyo.ConcreteModel()
        x = m.x = pyo.Var()
        e = 2 + x
        e2 = self.simp.simplify(e)
        self.compare_against_possible_results(e2, [2.0 + x, x + 2.0])

    def test_neg(self):
        m = pyo.ConcreteModel()
        x = m.x = pyo.Var()
        e = -pyo.log(x)
        e2 = self.simp.simplify(e)
        self.compare_against_possible_results(
            e2, [(-1.0) * pyo.log(x), pyo.log(x) * (-1.0), -pyo.log(x)]
        )

    def test_pow(self):
        m = pyo.ConcreteModel()
        x = m.x = pyo.Var()
        e = x**2.0
        e2 = self.simp.simplify(e)
        assertExpressionsEqual(self, e, e2)

    def test_div(self):
        m = pyo.ConcreteModel()
        x = m.x = pyo.Var()
        y = m.y = pyo.Var()
        e = x / y + y / x - x / y
        e2 = self.simp.simplify(e)
        self.compare_against_possible_results(
            e2, [y / x, y * (1.0 / x), y * x**-1.0, x**-1.0 * y]
        )

    def test_unary(self):
        m = pyo.ConcreteModel()
        x = m.x = pyo.Var()
        func_list = [pyo.log, pyo.sin, pyo.cos, pyo.tan, pyo.asin, pyo.acos, pyo.atan]
        for func in func_list:
            e = func(x)
            e2 = self.simp.simplify(e)
            assertExpressionsEqual(self, e, e2)

    def test_param(self):
        m = pyo.ConcreteModel()
        x = m.x = pyo.Var()
        p = m.p = pyo.Param(mutable=True)
        e1 = p * x**2 + p * x + p * x**2
        e2 = self.simp.simplify(e1)
        self.compare_against_possible_results(
            e2,
            [
                p * x**2.0 * 2.0 + p * x,
                p * x + p * x**2.0 * 2.0,
                2.0 * p * x**2.0 + p * x,
                p * x + 2.0 * p * x**2.0,
                x**2.0 * p * 2.0 + p * x,
                p * x + x**2.0 * p * 2.0,
                p * x * (1 + 2 * x),
            ],
        )


@unittest.skipUnless(sympy_available, 'sympy is not available')
class TestSimplificationSympy(unittest.TestCase, SimplificationMixin):
    def setUp(self):
        self.simp = Simplifier(mode=Simplifier.Mode.sympy)


@unittest.pytest.mark.default
@unittest.pytest.mark.builders
@unittest.skipUnless(ginac_available, 'GiNaC is not available')
class TestSimplificationGiNaC(unittest.TestCase, SimplificationMixin):
    def setUp(self):
        self.simp = Simplifier(mode=Simplifier.Mode.ginac)
