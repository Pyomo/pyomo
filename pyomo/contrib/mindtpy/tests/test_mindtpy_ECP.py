# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

# -*- coding: utf-8 -*-
"""Tests for the MindtPy solver."""

import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.minlp_simple import MinlpSimple
from pyomo.contrib.mindtpy.tests.minlp2_simple import Minlp2Simple
from pyomo.contrib.mindtpy.tests.minlp3_simple import Minlp3Simple
from pyomo.contrib.mindtpy.tests.from_proposal import FromProposalModel
from pyomo.contrib.mindtpy.tests.constraint_qualification_example import (
    ConstraintQualificationExample,
)
from pyomo.environ import SolverFactory, value
from pyomo.opt import TerminationCondition

if SolverFactory('appsi_highs').available(exception_flag=False) and SolverFactory(
    'appsi_highs'
).version() >= (1, 7, 0):
    required_solvers = ('ipopt', 'appsi_highs')
else:
    required_solvers = ('ipopt', 'glpk')

if all(SolverFactory(s).available(exception_flag=False) for s in required_solvers):
    subsolvers_available = True
else:
    subsolvers_available = False

model_list = [
    EightProcessFlowsheet(convex=True),
    ConstraintQualificationExample(),
    MinlpSimple(),
    Minlp2Simple(),
    Minlp3Simple(),
    FromProposalModel(),
]


@unittest.skipIf(
    not subsolvers_available,
    'Required subsolvers %s are not available' % (required_solvers,),
)
class TestMindtPy(unittest.TestCase):
    """Tests for the MindtPy solver."""

    def check_optimal_solution(self, model, places=1):
        """Assert that variable values match the model's known optimum.

        Parameters
        ----------
        model : Block
            Model containing ``optimal_solution`` values for comparison.
        places : int, optional
            Decimal places used by ``assertAlmostEqual``.
        """
        for var in model.optimal_solution:
            self.assertAlmostEqual(
                var.value, model.optimal_solution[var], places=places
            )

    def test_ECP(self):
        """Test the extended cutting plane decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='ECP',
                    init_strategy='rNLP',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                    absolute_bound_tolerance=1e-5,
                )

                self.assertIs(
                    results.solver.termination_condition, TerminationCondition.optimal
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)

    def test_ECP_add_slack(self):
        """Test the extended cutting plane decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='ECP',
                    init_strategy='rNLP',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                    absolute_bound_tolerance=1e-5,
                    add_slack=True,
                )

                self.assertIs(
                    results.solver.termination_condition, TerminationCondition.optimal
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)


if __name__ == '__main__':
    unittest.main()
