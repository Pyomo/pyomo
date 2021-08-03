# -*- coding: utf-8 -*-
"""Tests for the MindtPy solver."""
from math import fabs
import pyomo.core.base.symbolic
import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import \
    EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.MINLP_simple import SimpleMINLP as SimpleMINLP
from pyomo.contrib.mindtpy.tests.MINLP2_simple import SimpleMINLP as SimpleMINLP2
from pyomo.contrib.mindtpy.tests.MINLP3_simple import SimpleMINLP as SimpleMINLP3
from pyomo.contrib.mindtpy.tests.from_proposal import ProposalModel
from pyomo.contrib.mindtpy.tests.constraint_qualification_example import ConstraintQualificationExample
from pyomo.contrib.mindtpy.tests.online_doc_example import OnlineDocExample
from pyomo.contrib.mindtpy.tests.nonconvex1 import Nonconvex1
from pyomo.contrib.mindtpy.tests.nonconvex2 import Nonconvex2
from pyomo.contrib.mindtpy.tests.nonconvex3 import Nonconvex3
from pyomo.contrib.mindtpy.tests.nonconvex4 import Nonconvex4
from pyomo.environ import SolverFactory, value
from pyomo.environ import *
from pyomo.solvers.tests.models.LP_unbounded import LP_unbounded
from pyomo.solvers.tests.models.QCP_simple import QCP_simple
from pyomo.solvers.tests.models.MIQCP_simple import MIQCP_simple

from pyomo.opt import TerminationCondition

required_solvers = ('baron', 'cplex_persistent')
if not all(SolverFactory(s).available(False) for s in required_solvers):
    subsolvers_available = False
elif not SolverFactory('baron').license_is_valid():
    subsolvers_available = False
else:
    subsolvers_available = True


model_list = [EightProcessFlowsheet(convex=False),
              Nonconvex1(),
              Nonconvex2(),
              Nonconvex3(),
              Nonconvex4(),
              ]


@unittest.skipIf(not subsolvers_available,
                 'Required subsolvers %s are not available'
                 % (required_solvers,))
@unittest.skipIf(not pyomo.core.base.symbolic.differentiate_available,
                 'Symbolic differentiation is not available')
@unittest.skipIf(not pyomo.contrib.mcpp.pyomo_mcpp.mcpp_available(),
                 'MC++ is not available')
class TestMindtPy(unittest.TestCase):
    """Tests for the MindtPy solver plugin."""

    def test_GOA(self):
        """Test the global outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                results = opt.solve(model, strategy='GOA',
                                    mip_solver=required_solvers[1],
                                    nlp_solver=required_solvers[0],
                                    single_tree=True)

                self.assertIn(results.solver.termination_condition, [
                    TerminationCondition.optimal, TerminationCondition.feasible])
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1)

    def test_GOA_tabu_list(self):
        """Test the global outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                results = opt.solve(model, strategy='GOA',
                                    mip_solver=required_solvers[1],
                                    nlp_solver=required_solvers[0],
                                    single_tree=True,
                                    use_tabu_list=True)

                self.assertIn(results.solver.termination_condition, [
                    TerminationCondition.optimal, TerminationCondition.feasible])
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1)


if __name__ == '__main__':
    unittest.main()
