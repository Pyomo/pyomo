#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


from pyomo.environ import SolverFactory, ConcreteModel, Var, Constraint, Objective, Integers, Boolean
import pyutilib.th as unittest
from pyutilib.misc import capture_output

opt_cbc = SolverFactory('cbc')
cbc_available = opt_cbc.available(exception_flag=False)


class CBCTests(unittest.TestCase):

    @unittest.skipIf(not cbc_available,
                     "The CBC solver is not available")
    def test_warm_start(self):

        m = ConcreteModel()
        m.x = Var()
        m.z = Var(domain=Integers)
        m.w = Var(domain=Boolean)
        m.c = Constraint(expr=m.x + m.z + m.w >= 0)
        m.o = Objective(expr=m.x + m.z + m.w)

        # Set some initial values for warm start.
        m.x.set_value(10)
        m.z.set_value(5)
        m.w.set_value(1)

        with SolverFactory("cbc") as opt:
            with capture_output() as output:
                opt.solve(m, tee=True, warmstart=True)

                # Check if CBC loaded the warmstart file.
                self.assertIn('opening mipstart file', output.getvalue())
                # Only integer and binary variables are considered by CBC.
                self.assertIn('MIPStart values read for 2 variables.', output.getvalue())
                # m.x is ignored because it is continuous, so cost should be 5+1
                self.assertIn('MIPStart provided solution with cost 6', output.getvalue())


if __name__ == "__main__":
    unittest.main()
