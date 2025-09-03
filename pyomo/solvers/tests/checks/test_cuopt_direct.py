#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
from pyomo.environ import (
    SolverFactory,
    ConcreteModel,
    Block,
    Var,
    Constraint,
    Objective,
    NonNegativeReals,
    Suffix,
    value,
    minimize,
)
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.unittest as unittest

try:
    import cuopt

    cuopt_available = True
except:
    cuopt_available = False


class CUOPTTests(unittest.TestCase):
    @unittest.skipIf(not cuopt_available, "The CuOpt solver is not available")
    def test_values_and_rc(self):
        m = ConcreteModel()

        m.dual = Suffix(direction=Suffix.IMPORT)
        m.rc = Suffix(direction=Suffix.IMPORT)

        m.x = Var(domain=NonNegativeReals)
        m.top_con = Constraint(expr=m.x >= 0)

        m.b1 = Block()
        m.b1.y = Var(domain=NonNegativeReals)
        m.b1.con1 = Constraint(expr=m.x + m.b1.y <= 10)

        m.b1.subb = Block()
        m.b1.subb.z = Var(domain=NonNegativeReals)
        m.b1.subb.con2 = Constraint(expr=2 * m.b1.y + m.b1.subb.z >= 8)

        m.b2 = Block()
        m.b2.w = Var(domain=NonNegativeReals)
        m.b2.con3 = Constraint(expr=m.b1.subb.z - m.b2.w == 2)

        # Minimize cost = 1*x + 2*y + 3*z + 0.5*w
        m.obj = Objective(
            expr=1.0 * m.x + 2.0 * m.b1.y + 3.0 * m.b1.subb.z + 0.5 * m.b2.w,
            sense=minimize,
        )

        opt = SolverFactory('cuopt')
        res = opt.solve(m)

        expected_rc = [1.0, 0.0, 0.0, 2.5]
        expected_val = [0.0, 3.0, 2.0, 0.0]
        expected_dual = [0.0, 0.0, 1.0, 2.0]

        for i, v in enumerate([m.x, m.b1.y, m.b1.subb.z, m.b2.w]):
            self.assertAlmostEqual(m.rc[v], expected_rc[i], places=5)
            self.assertAlmostEqual(value(v), expected_val[i], places=5)

        for i, c in enumerate([m.top_con, m.b1.con1, m.b1.subb.con2, m.b2.con3]):
            self.assertAlmostEqual(m.dual[c], expected_dual[i], places=5)


if __name__ == "__main__":
    unittest.main()
