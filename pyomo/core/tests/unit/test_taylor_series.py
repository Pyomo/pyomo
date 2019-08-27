import pyutilib.th as unittest
import pyomo.environ as pe
from pyomo.core.expr.taylor_series import taylor_series_expansion
from pyomo.core.expr.current import polynomial_degree
from pyomo.core.expr.calculus.derivatives import differentiate


class TestTaylorSeries(unittest.TestCase):
    def test_first_order_taylor_series(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.x.value = 1
        exprs_to_test = [m.x**2, pe.exp(m.x), (m.x + 2)**2]
        for e in exprs_to_test:
            tsa = taylor_series_expansion(e)
            self.assertAlmostEqual(pe.differentiate(e, wrt=m.x), pe.differentiate(tsa, wrt=m.x))
            self.assertAlmostEqual(pe.value(e), pe.value(tsa))
            self.assertEqual(polynomial_degree(tsa), 1)

    def test_higher_order_taylor_series(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(initialize=0.5)
        m.y = pe.Var(initialize=1.5)

        e = m.x * m.y
        tse = taylor_series_expansion(e, diff_mode=differentiate.Modes.reverse_symbolic, order=2)
        for _x in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
            for _y in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
                m.x.value = _x
                m.y.value = _y
                self.assertAlmostEqual(pe.value(e), pe.value(tse))

        e = m.x**3 + m.y**3
        tse = taylor_series_expansion(e, diff_mode=differentiate.Modes.reverse_symbolic, order=3)
        for _x in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
            for _y in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
                m.x.value = _x
                m.y.value = _y
                self.assertAlmostEqual(pe.value(e), pe.value(tse))

        e = (m.x*m.y)**2
        tse = taylor_series_expansion(e, diff_mode=differentiate.Modes.reverse_symbolic, order=4)
        for _x in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
            for _y in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
                m.x.value = _x
                m.y.value = _y
                self.assertAlmostEqual(pe.value(e), pe.value(tse))
