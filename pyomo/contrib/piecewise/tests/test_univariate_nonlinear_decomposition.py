from pyomo.common.unittest import TestCase, skipUnless
import pyomo.environ as pyo
from pyomo.contrib import piecewise
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.common.dependencies import numpy_available, numpy
from pyomo.core.expr.numeric_expr import ProductExpression


pe = pyo


def _get_trans():
    return pyo.TransformationFactory('contrib.piecewise.univariate_nonlinear_decomposition')


class TestUnivariateNonlinearDecomposition(TestCase):
    def test_multiterm(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.c = pe.Constraint(expr=m.x + pe.log(m.y + m.z) + 1/pe.exp(m.x**0.5) <= 0)

        trans = _get_trans()
        trans.apply_to(m)
        aux = m.auxiliary

        assertExpressionsEqual(
            self,
            m.c.body,
            m.x + aux.x[3] + aux.x[2],
        )
        assertExpressionsEqual(
            self,
            aux.c[1].expr,
            aux.x[1] == (m.y + m.z),
        )
        assertExpressionsEqual(
            self,
            aux.c[2].expr,
            aux.x[2] == 1/pyo.exp(m.x**0.5),
        )
        assertExpressionsEqual(
            self,
            aux.c[3].expr,
            aux.x[3] == pyo.log(aux.x[1]),
        )
        self.assertEqual(m.x.lb, 0)
        self.assertIsNone(m.x.ub)
        self.assertIsNone(m.y.lb)
        self.assertIsNone(m.y.ub)
        self.assertIsNone(m.z.lb)
        self.assertIsNone(m.z.ub)
        self.assertEqual(aux.x[1].lb, 0)
        self.assertIsNone(aux.x[1].ub)
        self.assertEqual(aux.x[2].lb, 0)
        self.assertEqual(aux.x[2].ub, 1)
        self.assertIsNone(aux.x[3].lb)
        self.assertIsNone(aux.x[3].ub)

    def test_common_subexpressions(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z1 = pe.Var()
        m.z2 = pe.Var()
        e = -pe.log(m.x + m.y)
        m.c1 = pe.Constraint(expr=m.z1 + e == 0)
        m.c2 = pe.Constraint(expr=m.z2 + e == 0)

        trans = _get_trans()
        trans.apply_to(m)
        aux = m.auxiliary

        assertExpressionsEqual(
            self,
            m.c1.expr,
            m.z1 + aux.x[2] == 0,
        )
        assertExpressionsEqual(
            self,
            m.c2.expr,
            m.z2 + aux.x[2] == 0,
        )
        assertExpressionsEqual(
            self,
            aux.c[1].expr,
            aux.x[1] == m.x + m.y,
        )
        assertExpressionsEqual(
            self,
            aux.c[2].expr,
            aux.x[2] == -pe.log(aux.x[1]),
        )

    def test_product_fixed_variable(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.c = pe.Constraint(expr=2*pe.log(m.x + m.y)  <= 0)

        trans = _get_trans()
        trans.apply_to(m)
        aux = m.auxiliary

        assertExpressionsEqual(
            self,
            m.c.expr,
            2*pe.log(aux.x[1]) <= 0,
        )
        assertExpressionsEqual(
            self,
            aux.c[1].expr,
            aux.x[1] == m.x + m.y,
        )

    def test_product_variable_fixed(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.c = pe.Constraint(expr=pe.log(m.x + m.y)*2  <= 0)

        trans = _get_trans()
        trans.apply_to(m)
        aux = m.auxiliary

        assertExpressionsEqual(
            self,
            m.c.expr,
            pe.log(aux.x[1])*2 <= 0,
        )
        assertExpressionsEqual(
            self,
            aux.c[1].expr,
            aux.x[1] == m.x + m.y,
        )

    def test_prod_sum_sum(self):
        m = pe.ConcreteModel()
        m.x1 = pe.Var()
        m.x2 = pe.Var()
        m.x3 = pe.Var()
        m.x4 = pe.Var()
        m.c = pe.Constraint(expr=(m.x1 + m.x2) * (m.x3 + m.x4) <= 1)

        trans = _get_trans()
        trans.apply_to(m)
        aux = m.auxiliary

        assertExpressionsEqual(
            self,
            m.c.expr,
            aux.x[1] * aux.x[2] <= 1,
        )
        assertExpressionsEqual(
            self,
            aux.c[1].expr,
            aux.x[1] == m.x1 + m.x2,
        )
        assertExpressionsEqual(
            self,
            aux.c[2].expr,
            aux.x[2] == m.x3 + m.x4,
        )

    def test_pow_sum_sum(self):
        m = pe.ConcreteModel()
        m.x1 = pe.Var()
        m.x2 = pe.Var()
        m.x3 = pe.Var()
        m.x4 = pe.Var()
        m.c = pe.Constraint(expr=(m.x1 + m.x2) ** (m.x3 + m.x4) <= 1)

        trans = _get_trans()
        trans.apply_to(m)
        aux = m.auxiliary

        assertExpressionsEqual(
            self,
            m.c.expr,
            aux.x[1] ** aux.x[2] <= 1,
        )
        assertExpressionsEqual(
            self,
            aux.c[1].expr,
            aux.x[1] == m.x1 + m.x2,
        )
        assertExpressionsEqual(
            self,
            aux.c[2].expr,
            aux.x[2] == m.x3 + m.x4,
        )

    def test_division_var_const(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.c = pe.Constraint(expr=(m.x + m.y) / 2 <= 0)

        trans = _get_trans()
        trans.apply_to(m)
        aux = m.auxiliary

        assertExpressionsEqual(
            self,
            m.c.expr,
            (m.x + m.y) / 2 <= 0,
        )

    def test_division_sum_sum(self):
        m = pe.ConcreteModel()
        m.x1 = pe.Var()
        m.x2 = pe.Var()
        m.x3 = pe.Var()
        m.x4 = pe.Var()
        m.c = pe.Constraint(expr=(m.x1 + m.x2) / (m.x3 + m.x4) <= 1)

        trans = _get_trans()
        trans.apply_to(m)
        aux = m.auxiliary

        assertExpressionsEqual(
            self,
            m.c.expr,
            aux.x[1] * aux.x[3] <= 1,
        )
        assertExpressionsEqual(
            self,
            aux.c[1].expr,
            aux.x[1] == m.x1 + m.x2,
        )
        assertExpressionsEqual(
            self,
            aux.c[2].expr,
            aux.x[2] == m.x3 + m.x4,
        )
        assertExpressionsEqual(
            self,
            aux.c[3].expr,
            aux.x[3] * aux.x[2] == 1,
        )

    @skipUnless(numpy_available, "Numpy is not available")
    def test_numpy_float(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.c = pe.Constraint(expr=ProductExpression((numpy.float64(2.5), pe.log(m.x + m.y)))  <= 0)

        trans = _get_trans()
        trans.apply_to(m)
        aux = m.auxiliary

        assertExpressionsEqual(
            self,
            m.c.expr,
            2.5*pe.log(aux.x[1]) <= 0,
        )
        assertExpressionsEqual(
            self,
            aux.c[1].expr,
            aux.x[1] == m.x + m.y,
        )
