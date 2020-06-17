#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Tests the BARON interface."""

from six import StringIO

import pyutilib.th as unittest

from pyomo.common.log import LoggingIntercept
from pyomo.environ import (
    ConcreteModel, Constraint, Objective, Var, log10, minimize,
)
from pyomo.opt import SolverFactory, TerminationCondition

# check if BARON is available
from pyomo.solvers.tests.solvers import test_solver_cases
baron_available = test_solver_cases('baron', 'bar').available


@unittest.skipIf(not baron_available,
                 "The 'BARON' solver is not available")
class BaronTest(unittest.TestCase):
    """Test the BARON interface."""

    def test_log10(self):
        # Tests the special transformation for log10
        with SolverFactory("baron") as opt:

            m = ConcreteModel()
            m.x = Var()
            m.c = Constraint(expr=log10(m.x) >= 2)
            m.obj = Objective(expr=m.x, sense=minimize)

            results = opt.solve(m)

            self.assertEqual(results.solver.termination_condition,
                             TerminationCondition.optimal)

    def test_abs(self):
        # Tests the special transformation for abs
        with SolverFactory("baron") as opt:

            m = ConcreteModel()
            m.x = Var(bounds=(-100,1))
            m.c = Constraint(expr=abs(m.x) >= 2)
            m.obj = Objective(expr=m.x, sense=minimize)

            results = opt.solve(m)

            self.assertEqual(results.solver.termination_condition,
                             TerminationCondition.optimal)

    def test_pow(self):
        # Tests the special transformation for x ^ y (both variables)
        with SolverFactory("baron") as opt:

            m = ConcreteModel()
            m.x = Var(bounds=(10,100))
            m.y = Var(bounds=(1,10))
            m.c = Constraint(expr=m.x ** m.y >= 20)
            m.obj = Objective(expr=m.x, sense=minimize)

            results = opt.solve(m)

            self.assertEqual(results.solver.termination_condition,
                             TerminationCondition.optimal)

    def test_BARON_option_warnings(self):
        os = StringIO()
        with LoggingIntercept(os, 'pyomo.solvers'):
            m = ConcreteModel()
            m.x = Var()
            m.obj = Objective(expr=m.x**2)

            with SolverFactory("baron") as opt:
                results = opt.solve(m, options={'ResName': 'results.lst',
                                                'TimName': 'results.tim'})

            self.assertEqual(results.solver.termination_condition,
                             TerminationCondition.optimal)
        self.assertIn('Ignoring user-specified option "ResName=results.lst"',
                      os.getvalue())
        self.assertIn('Ignoring user-specified option "TimName=results.tim"',
                      os.getvalue())

if __name__ == '__main__':
    unittest.main()
