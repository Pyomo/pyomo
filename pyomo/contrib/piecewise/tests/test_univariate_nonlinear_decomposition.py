from pyomo.common.unittest import TestCase
import pyomo.environ as pyo
from pyomo.contrib import piecewise
from pyomo.core.expr.compare import assertExpressionsEqual


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
            aux.x[3] + aux.x[2] + m.x,
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
