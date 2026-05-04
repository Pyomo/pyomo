# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from pyomo.common.unittest import TestCase, skipUnless
import pyomo.environ as pyo
from pyomo.contrib import piecewise
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.common.dependencies import numpy_available, numpy
from pyomo.core.expr.numeric_expr import ProductExpression


def _get_trans():
    return pyo.TransformationFactory(
        'contrib.piecewise.univariate_nonlinear_decomposition'
    )


class TestUnivariateNonlinearDecomposition(TestCase):
    def test_multiterm(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.c = pyo.Constraint(expr=m.x + pyo.log(m.y + m.z) + 1 / pyo.exp(m.x**0.5) <= 0)

        trans = _get_trans()
        trans.apply_to(m)
        aux = m.auxiliary

        assertExpressionsEqual(self, m.c.body, m.x + aux.x[3] + aux.x[2])
        assertExpressionsEqual(self, aux.c[1].expr, aux.x[1] == (m.y + m.z))
        assertExpressionsEqual(self, aux.c[2].expr, aux.x[2] == 1 / pyo.exp(m.x**0.5))
        assertExpressionsEqual(self, aux.c[3].expr, aux.x[3] == pyo.log(aux.x[1]))
        self.assertEqual(m.x.lb, 0)
        self.assertIsNone(m.x.ub)
        self.assertIsNone(m.y.lb)
        self.assertIsNone(m.y.ub)
        self.assertIsNone(m.z.lb)
        self.assertIsNone(m.z.ub)
        self.assertTrue(aux.x[1].lb is None or aux.x[1].lb <= 0)
        self.assertIsNone(aux.x[1].ub)
        self.assertEqual(aux.x[2].lb, 0)
        self.assertEqual(aux.x[2].ub, 1)
        self.assertIsNone(aux.x[3].lb)
        self.assertIsNone(aux.x[3].ub)

    def test_common_subexpressions(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z1 = pyo.Var()
        m.z2 = pyo.Var()
        e = -pyo.log(m.x + m.y)
        m.c1 = pyo.Constraint(expr=m.z1 + e == 0)
        m.c2 = pyo.Constraint(expr=m.z2 + e == 0)

        trans = _get_trans()
        trans.apply_to(m)
        aux = m.auxiliary

        assertExpressionsEqual(self, m.c1.expr, m.z1 + aux.x[2] == 0)
        assertExpressionsEqual(self, m.c2.expr, m.z2 + aux.x[2] == 0)
        assertExpressionsEqual(self, aux.c[1].expr, aux.x[1] == m.x + m.y)
        assertExpressionsEqual(self, aux.c[2].expr, aux.x[2] == -pyo.log(aux.x[1]))

    def test_product_fixed_variable(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.c = pyo.Constraint(expr=2 * pyo.log(m.x + m.y) <= 0)

        trans = _get_trans()
        trans.apply_to(m)
        aux = m.auxiliary

        assertExpressionsEqual(self, m.c.expr, 2 * pyo.log(aux.x[1]) <= 0)
        assertExpressionsEqual(self, aux.c[1].expr, aux.x[1] == m.x + m.y)

    def test_product_variable_fixed(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.c = pyo.Constraint(expr=pyo.log(m.x + m.y) * 2 <= 0)

        trans = _get_trans()
        trans.apply_to(m)
        aux = m.auxiliary

        assertExpressionsEqual(self, m.c.expr, pyo.log(aux.x[1]) * 2 <= 0)
        assertExpressionsEqual(self, aux.c[1].expr, aux.x[1] == m.x + m.y)

    def test_prod_sum_sum(self):
        m = pyo.ConcreteModel()
        m.x1 = pyo.Var()
        m.x2 = pyo.Var()
        m.x3 = pyo.Var()
        m.x4 = pyo.Var()
        m.c = pyo.Constraint(expr=(m.x1 + m.x2) * (m.x3 + m.x4) <= 1)

        trans = _get_trans()
        trans.apply_to(m)
        aux = m.auxiliary

        assertExpressionsEqual(self, m.c.expr, aux.x[1] * aux.x[2] <= 1)
        assertExpressionsEqual(self, aux.c[1].expr, aux.x[1] == m.x1 + m.x2)
        assertExpressionsEqual(self, aux.c[2].expr, aux.x[2] == m.x3 + m.x4)

    def test_pow_sum_sum(self):
        m = pyo.ConcreteModel()
        m.x1 = pyo.Var()
        m.x2 = pyo.Var()
        m.x3 = pyo.Var()
        m.x4 = pyo.Var()
        m.c = pyo.Constraint(expr=(m.x1 + m.x2) ** (m.x3 + m.x4) <= 1)

        trans = _get_trans()
        trans.apply_to(m)
        aux = m.auxiliary

        assertExpressionsEqual(self, m.c.expr, aux.x[1] ** aux.x[2] <= 1)
        assertExpressionsEqual(self, aux.c[1].expr, aux.x[1] == m.x1 + m.x2)
        assertExpressionsEqual(self, aux.c[2].expr, aux.x[2] == m.x3 + m.x4)

    def test_division_var_const(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.c = pyo.Constraint(expr=(m.x + m.y) / 2 <= 0)

        trans = _get_trans()
        trans.apply_to(m)
        aux = m.auxiliary

        assertExpressionsEqual(self, m.c.expr, (m.x + m.y) / 2 <= 0)

    def test_division_sum_sum(self):
        m = pyo.ConcreteModel()
        m.x1 = pyo.Var()
        m.x2 = pyo.Var()
        m.x3 = pyo.Var()
        m.x4 = pyo.Var()
        m.c = pyo.Constraint(expr=(m.x1 + m.x2) / (m.x3 + m.x4) <= 1)

        trans = _get_trans()
        trans.apply_to(m)
        aux = m.auxiliary

        assertExpressionsEqual(self, m.c.expr, aux.x[1] * aux.x[3] <= 1)
        assertExpressionsEqual(self, aux.c[1].expr, aux.x[1] == m.x1 + m.x2)
        assertExpressionsEqual(self, aux.c[2].expr, aux.x[2] == m.x3 + m.x4)
        assertExpressionsEqual(self, aux.c[3].expr, aux.x[3] * aux.x[2] == 1)

    @skipUnless(numpy_available, "Numpy is not available")
    def test_numpy_float(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.c = pyo.Constraint(
            expr=ProductExpression((numpy.float64(2.5), pyo.log(m.x + m.y))) <= 0
        )

        trans = _get_trans()
        trans.apply_to(m)
        aux = m.auxiliary

        assertExpressionsEqual(self, m.c.expr, 2.5 * pyo.log(aux.x[1]) <= 0)
        assertExpressionsEqual(self, aux.c[1].expr, aux.x[1] == m.x + m.y)
