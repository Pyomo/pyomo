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
    Var,
    Constraint,
    Objective,
    Integers,
    Boolean,
    Suffix,
    maximize,
)
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.unittest as unittest

opt_cbc = SolverFactory('cbc')
cbc_available = opt_cbc.available(exception_flag=False)


class CBCTests(unittest.TestCase):
    @unittest.skipIf(not cbc_available, "The CBC solver is not available")
    def test_warm_start(self):
        m = ConcreteModel()
        m.x = Var()
        m.z = Var(domain=Integers)
        m.w = Var(domain=Boolean)
        m.c = Constraint(expr=m.x + m.z + m.w >= 0)
        m.o = Objective(expr=m.x + m.z + m.w)

        TempfileManager.push()
        tempdir = os.path.dirname(TempfileManager.create_tempfile())
        TempfileManager.pop()

        sameDrive = os.path.splitdrive(tempdir)[0] == os.path.splitdrive(os.getcwd())[0]

        # At the moment, CBC does not cleanly handle windows-style drive
        # names in the MIPSTART file name (though at least 2.10.5).
        #
        # See https://github.com/coin-or/Cbc/issues/32
        # The problematic source is https://github.com/coin-or/Cbc/blob/3dcedb27664ae458990e9d4d50bc11c2c55917a0/src/CbcSolver.cpp#L9445-L9459
        #
        # We will try two different situations: running from the current
        # directory (which may or may not be on the same drive), and
        # then from the tempdir (which will be on the same drive).

        # Set some initial values for warm start.
        m.x.set_value(10)
        m.z.set_value(5)
        m.w.set_value(1)

        with SolverFactory("cbc") as opt, capture_output() as output:
            opt.solve(
                m, tee=True, warmstart=True, options={'sloglevel': 2, 'loglevel': 2}
            )

        log = output.getvalue()
        # Check if CBC loaded the warmstart file.
        self.assertIn('opening mipstart file', log)

        if sameDrive:
            # Only integer and binary variables are considered by CBC.
            self.assertIn('MIPStart values read for 2 variables.', log)
            # m.x is ignored because it is continuous, so cost should be 5+1
            self.assertIn('MIPStart provided solution with cost 6', log)
        else:
            self.assertNotIn('MIPStart values read', log)

        # Set some initial values for warm start.
        m.x.set_value(10)
        m.z.set_value(5)
        m.w.set_value(1)

        try:
            _origDir = os.getcwd()
            os.chdir(tempdir)
            with SolverFactory("cbc") as opt, capture_output() as output:
                opt.solve(
                    m, tee=True, warmstart=True, options={'sloglevel': 2, 'loglevel': 2}
                )
        finally:
            os.chdir(_origDir)

        log = output.getvalue()
        # Check if CBC loaded the warmstart file.
        self.assertIn('opening mipstart file', log)
        # Only integer and binary variables are considered by CBC.
        self.assertIn('MIPStart values read for 2 variables.', log)
        # m.x is ignored because it is continuous, so cost should be 5+1
        self.assertIn('MIPStart provided solution with cost 6', log)

    @unittest.skipIf(not cbc_available, "The CBC solver is not available")
    def test_duals_signs(self):
        m = ConcreteModel()
        m.x = Var()
        m.obj = Objective(expr=m.x)
        m.c = Constraint(expr=(-1, m.x, 1))
        m.dual = Suffix(direction=Suffix.IMPORT)
        opt = SolverFactory('cbc')
        res = opt.solve(m)
        self.assertAlmostEqual(res.problem.lower_bound, -1)
        self.assertAlmostEqual(res.problem.upper_bound, -1)
        self.assertAlmostEqual(m.dual[m.c], 1)
        m.obj.sense = maximize
        res = opt.solve(m)
        self.assertAlmostEqual(res.problem.lower_bound, 1)
        self.assertAlmostEqual(res.problem.upper_bound, 1)
        self.assertAlmostEqual(m.dual[m.c], 1)

    @unittest.skipIf(not cbc_available, "The CBC solver is not available")
    def test_rc_signs(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-1, 1))
        m.obj = Objective(expr=m.x)
        m.rc = Suffix(direction=Suffix.IMPORT)
        opt = SolverFactory('cbc')
        res = opt.solve(m)
        self.assertAlmostEqual(res.problem.lower_bound, -1)
        self.assertAlmostEqual(res.problem.upper_bound, -1)
        self.assertAlmostEqual(m.rc[m.x], 1)
        m.obj.sense = maximize
        res = opt.solve(m)
        self.assertAlmostEqual(res.problem.lower_bound, 1)
        self.assertAlmostEqual(res.problem.upper_bound, 1)
        self.assertAlmostEqual(m.rc[m.x], 1)


if __name__ == "__main__":
    unittest.main()
