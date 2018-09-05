#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
import re
import pyutilib.th as unittest
from pyutilib.misc.redirect_io import capture_output

from pyomo.environ import SolverFactory
from pyomo.scripting.driver_help import help_solvers


class Test(unittest.TestCase):

    def test_help_solvers(self):
        with capture_output() as OUT:
            help_solvers()
        OUT = OUT.getvalue()
        self.assertTrue(re.search('Pyomo Solvers and Solver Managers', OUT))
        self.assertTrue(re.search('Serial Solver', OUT))
        # Test known solvers and metasolver flags
        # ASL is a metasolver
        self.assertTrue(re.search('asl +\+', OUT))
        # PS is bundles with Pyomo so should always be available
        self.assertTrue(re.search('ps +\*', OUT))
        for solver in ('ipopt','baron','cbc','glpk'):
            s = SolverFactory(solver)
            if s.available():
                self.assertTrue(re.search("%s +\* [a-zA-Z]" % solver, OUT))
            else:
                self.assertTrue(re.search("%s +[a-zA-Z]" % solver, OUT))


if __name__ == "__main__":
    unittest.main()
