import pyutilib.th as unittest

from pyomo.core import *
from pyomo.repn.ampl_repn import _generate_ampl_repn as gar
from pyomo.repn.ampl_repn import AmplRepn


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
        self.assertEqual(test._constant, 0)
        self.assertEqual(test._linear_vars, {})
        self.assertEqual(test._linear_terms_coef, {})
        self.assertEqual(test._nonlinear_vars, {id(m.x): m.x})
        self.assertIs(test._nonlinear_expr, m.con.body)
