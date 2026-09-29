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
from pyomo.contrib.mindtpy.tests.online_doc_example import OnlineDocExample
from pyomo.environ import SolverFactory, value
from pyomo.opt import TerminationCondition
from pyomo.contrib.gdpopt.util import is_feasible
from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.contrib.mindtpy.tests.feasibility_pump1 import FeasibilityPump1
from pyomo.contrib.mindtpy.tests.feasibility_pump2 import FeasibilityPump2

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
    FeasibilityPump1(),
    FeasibilityPump2(),
    MinlpSimple(),
    Minlp2Simple(),
    Minlp3Simple(),
    FromProposalModel(),
    OnlineDocExample(),
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

    def get_config(self, solver):
        """Return the active MindtPy configuration block for ``solver``.

        Parameters
        ----------
        solver : SolverFactory
            Instantiated MindtPy solver object.

        Returns
        -------
        ConfigBlock
            Active configuration block associated with ``solver``.
        """
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
