import pyutilib.th as unittest
import pyomo.environ as pe
from pyomo.core.expr.taylor_series.taylor import taylor_series
from pyomo.core.expr.current import polynomial_degree


class TestDerivs(unittest.TestCase):
    def test_taylor_series(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.x.value = 1
        exprs_to_test = [m.x**2, pe.exp(m.x), (m.x + 2)**2]
        for e in exprs_to_test:
            tsa = taylor_series(e)
            self.assertAlmostEqual(pe.differentiate(e, wrt=m.x), pe.differentiate(tsa, wrt=m.x))
            self.assertAlmostEqual(pe.value(e), pe.value(tsa))
            self.assertEqual(polynomial_degree(tsa), 1)
