from pyomo.common.unittest import TestCase
from pyomo.contrib.simplification import Simplifier
from pyomo.core.expr.compare import assertExpressionsEqual, compare_expressions
import pyomo.environ as pe
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd


class TestSimplification(TestCase):
    def test_simplify(self):
        m = pe.ConcreteModel()
        x = m.x = pe.Var(bounds=(0, None))
        e = x*pe.log(x)
        der1 = reverse_sd(e)[x]
        der2 = reverse_sd(der1)[x]
        simp = Simplifier()
        der2_simp = simp.simplify(der2)
        expected = x**-1.0
        assertExpressionsEqual(self, expected, der2_simp)

    def test_param(self):
        m = pe.ConcreteModel()
        x = m.x = pe.Var()
        p = m.p = pe.Param(mutable=True)
        e1 = p*x**2 + p*x + p*x**2
        simp = Simplifier()
        e2 = simp.simplify(e1)
        exp1 = p*x**2.0*2.0 + p*x
        exp2 = p*x + p*x**2.0*2.0
        self.assertTrue(
            compare_expressions(e2, exp1) 
            or compare_expressions(e2, exp2)
            or compare_expressions(e2, p*x + x**2.0*p*2.0)
            or compare_expressions(e2, x**2.0*p*2.0 + p*x)
        )

    def test_mul(self):
        m = pe.ConcreteModel()
        x = m.x = pe.Var()
        e = 2*x
        simp = Simplifier()
        e2 = simp.simplify(e)
        expected = 2.0*x
        assertExpressionsEqual(self, expected, e2)

    def test_sum(self):
        m = pe.ConcreteModel()
        x = m.x = pe.Var()
        e = 2 + x
        simp = Simplifier()
        e2 = simp.simplify(e)
        expected = x + 2.0
        assertExpressionsEqual(self, expected, e2)

    def test_neg(self):
        m = pe.ConcreteModel()
        x = m.x = pe.Var()
        e = -pe.log(x)
        simp = Simplifier()
        e2 = simp.simplify(e)
        expected = pe.log(x)*(-1.0)
        assertExpressionsEqual(self, expected, e2)

