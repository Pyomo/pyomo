import pyutilib.th as unittest

from pyomo.core import *
from pyomo.repn.standard_repn import generate_standard_repn as gar


class AmplRepnTests(unittest.TestCase):

    def test_divide_by_mutable(self):
        #
        # Test from https://github.com/Pyomo/pyomo/issues/153
        #
        m = ConcreteModel()
        m.x = Var(bounds=(1,5))
        m.p = Param(initialize=100, mutable=True)
        m.con = Constraint(expr=exp(5*(1/m.x - 1/m.p))<=10)
        m.obj = Objective(expr=m.x**2)

        test = gar(m.con.body)
        self.assertEqual(test.constant, 0)
        self.assertEqual(test.linear_vars, {})
        self.assertEqual(test.linear_coefs, {})
        self.assertEqual(test.nonlinear_vars, {id(m.x): m.x})
        self.assertIs(test.nonlinear_expr, m.con.body)

if __name__ == "__main__":
    unittest.main()

