#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
import re
import pyomo.common.unittest as unittest
from pyomo.common.tee import capture_output
from pyomo.common.log import LoggingIntercept

from pyomo.environ import SolverFactory
from pyomo.scripting.driver_help import help_solvers, help_transformations
from pyomo.scripting.pyomo_main import main


class Test(unittest.TestCase):
    def test_pyomo_main_deprecation(self):
        with LoggingIntercept() as LOG:
            with unittest.pytest.raises(SystemExit) as e:
                main(args=['--solvers=glpk', 'foo.py'])
        self.assertIn("Running the 'pyomo' script with no subcommand", LOG.getvalue())

    def test_help_solvers(self):
        with capture_output() as OUT:
            help_solvers()
        OUT = OUT.getvalue()
        self.assertTrue(re.search('Pyomo Solvers and Solver Managers', OUT))
        self.assertTrue(re.search('Serial Solver', OUT))
        # Test known solvers and metasolver flags
        # ASL is a metasolver
        self.assertTrue(re.search(r'\n   \*asl ', OUT))
        # MindtPY is bundled with Pyomo so should always be available
        self.assertTrue(re.search(r'\n   \+mindtpy ', OUT))
        for solver in ('ipopt', 'cbc', 'glpk'):
            s = SolverFactory(solver)
            if s.available():
                self.assertTrue(
                    re.search(r"\n   \+%s " % solver, OUT),
                    "'   +%s' not found in help --solvers" % solver,
                )
            else:
                self.assertTrue(
                    re.search(r"\n    %s " % solver, OUT),
                    "'    %s' not found in help --solvers" % solver,
                )
        for solver in ('baron',):
            s = SolverFactory(solver)
            if s.license_is_valid():
                self.assertTrue(
                    re.search(r"\n   \+%s " % solver, OUT),
                    "'   +%s' not found in help --solvers" % solver,
                )
            elif s.available():
                self.assertTrue(
                    re.search(r"\n   \-%s " % solver, OUT),
                    "'   -%s' not found in help --solvers" % solver,
                )
            else:
                self.assertTrue(
                    re.search(r"\n    %s " % solver, OUT),
                    "'    %s' not found in help --solvers" % solver,
                )

    def test_help_transformations(self):
        with capture_output() as OUT:
            help_transformations()
        OUT = OUT.getvalue()
        self.assertTrue(re.search('Pyomo Model Transformations', OUT))
        self.assertTrue(re.search('core.relax_integer_vars', OUT))
        # test a transformation that we know is deprecated
        self.assertTrue(re.search(r'duality.linear_dual\s+\[DEPRECATED\]', OUT))


if __name__ == "__main__":
    unittest.main()
