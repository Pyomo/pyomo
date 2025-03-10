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

# -*- coding: utf-8 -*-
"""Tests for the MindtPy solver."""
import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.MINLP_simple import SimpleMINLP as SimpleMINLP
from pyomo.contrib.mindtpy.tests.MINLP2_simple import SimpleMINLP as SimpleMINLP2
from pyomo.contrib.mindtpy.tests.MINLP3_simple import SimpleMINLP as SimpleMINLP3
from pyomo.contrib.mindtpy.tests.from_proposal import ProposalModel
from pyomo.contrib.mindtpy.tests.constraint_qualification_example import (
    ConstraintQualificationExample,
)
from pyomo.contrib.mindtpy.tests.online_doc_example import OnlineDocExample
from pyomo.environ import SolverFactory, value
from pyomo.opt import TerminationCondition
from pyomo.contrib.gdpopt.util import is_feasible
from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.contrib.mindtpy.tests.feasibility_pump1 import FeasPump1
from pyomo.contrib.mindtpy.tests.feasibility_pump2 import FeasPump2

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
    FeasPump1(),
    FeasPump2(),
    SimpleMINLP(),
    SimpleMINLP2(),
    SimpleMINLP3(),
    ProposalModel(),
    OnlineDocExample(),
]


@unittest.skipIf(
    not subsolvers_available,
    'Required subsolvers %s are not available' % (required_solvers,),
)
class TestMindtPy(unittest.TestCase):
    """Tests for the MindtPy solver."""

    def check_optimal_solution(self, model, places=1):
        for var in model.optimal_solution:
            self.assertAlmostEqual(
                var.value, model.optimal_solution[var], places=places
            )

    def get_config(self, solver):
        config = solver.CONFIG
        return config

    def test_FP(self):
        """Test the feasibility pump algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='FP',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                    absolute_bound_tolerance=1e-5,
                )
                log_infeasible_constraints(model)
                self.assertTrue(is_feasible(model, self.get_config(opt)))

    def test_FP_L1_norm(self):
        """Test the feasibility pump algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='FP',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                    absolute_bound_tolerance=1e-5,
                    fp_main_norm='L1',
                )
                log_infeasible_constraints(model)
                self.assertTrue(is_feasible(model, self.get_config(opt)))

    def test_FP_OA_8PP(self):
        """Test the FP-OA algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    init_strategy='FP',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                    # absolute_bound_tolerance=1E-5
                )
                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)


if __name__ == '__main__':
    unittest.main()
