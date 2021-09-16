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
from pyomo.environ import SolverFactory, value
from pyomo.environ import *
from pyomo.solvers.tests.models.LP_unbounded import LP_unbounded
from pyomo.solvers.tests.models.QCP_simple import QCP_simple
from pyomo.solvers.tests.models.MIQCP_simple import MIQCP_simple
from pyomo.opt import TerminationCondition

required_solvers = ('ipopt', 'glpk')
# required_solvers = ('gams', 'gams')
if all(SolverFactory(s).available() for s in required_solvers):
    subsolvers_available = True
else:
    subsolvers_available = False

model_list = [EightProcessFlowsheet(convex=True),
              ConstraintQualificationExample(),
              SimpleMINLP(),
              SimpleMINLP2(),
              SimpleMINLP3(),
              ProposalModel(),
              ]


@unittest.skipIf(not subsolvers_available,
                 'Required subsolvers %s are not available'
                 % (required_solvers,))
@unittest.skipIf(not pyomo.core.base.symbolic.differentiate_available,
                 'Symbolic differentiation is not available')
class TestMindtPy(unittest.TestCase):
    """Tests for the MindtPy solver plugin."""

    def test_ECP(self):
        """Test the extended cutting plane decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                results = opt.solve(model, strategy='ECP',
                                    init_strategy='rNLP',
                                    mip_solver=required_solvers[1],
                                    nlp_solver=required_solvers[0],
                                    bound_tolerance=1E-5)

                self.assertIs(results.solver.termination_condition,
                              TerminationCondition.optimal)
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1)

    def test_ECP_add_slack(self):
        """Test the extended cutting plane decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                results = opt.solve(model, strategy='ECP',
                                    init_strategy='rNLP',
                                    mip_solver=required_solvers[1],
                                    nlp_solver=required_solvers[0],
                                    bound_tolerance=1E-5,
                                    add_slack=True)

                self.assertIs(results.solver.termination_condition,
                              TerminationCondition.optimal)
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1)


if __name__ == '__main__':
    unittest.main()
