#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.th as unittest

from pyomo.core import ConcreteModel, Var, Param, Constraint, Objective, exp
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
        self.assertEqual(test.linear_vars, tuple())
        self.assertEqual(test.linear_coefs, tuple())
        self.assertEqual(set(id(v) for v in test.nonlinear_vars), set([id(m.x)]))
        self.assertIs(test.nonlinear_expr, m.con.body)

if __name__ == "__main__":
    unittest.main()
