import logging

import pyutilib.th as unittest
from six import StringIO

from pyomo.common.log import LoggingIntercept
from pyomo.environ import ConcreteModel, Var, Objective
from pyomo.gdp import Disjunction
from pyomo.opt import SolverFactory


class TestGDPBB(unittest.TestCase):
    """Tests for logic-based branch and bound."""

    @unittest.skipUnless(SolverFactory('glpk').available(exception_flag=False), "glpk is not available.")
    def test_deprecation_warning(self):
        """Test for deprecation warning with small infeasible GDP."""
        m = ConcreteModel()
        m.x = Var(bounds=(0, 2))
        m.d = Disjunction(expr=[
            [m.x ** 2 >= 3, m.x >= 3],
            [m.x ** 2 <= -1, m.x <= -1]])
        m.o = Objective(expr=m.x)

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.solvers', logging.WARNING):
            SolverFactory('gdpbb').solve(m, tee=False, solver='glpk',)

        self.assertIn(
            "GDPbb has been merged into GDPopt.",
            output.getvalue()
        )
