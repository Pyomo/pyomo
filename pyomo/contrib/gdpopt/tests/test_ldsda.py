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
from pyomo.environ import (
    SolverFactory,
    value,
    Var,
    Constraint,
    TransformationFactory,
    ConcreteModel,
    BooleanVar,
    LogicalConstraint,
    Block,
)
from pyomo.gdp import Disjunct, Disjunction
import pyomo.common.unittest as unittest
from pyomo.contrib.gdpopt.tests.four_stage_dynamic_model import build_model
from unittest.mock import MagicMock
from pyomo.core.expr.logical_expr import exactly
from pyomo.contrib.gdpopt.ldsda import GDP_LDSDA_Solver


class TestGDPoptLDSDA(unittest.TestCase):
    """Real unit tests for GDPopt"""

    @unittest.skipUnless(
        SolverFactory('gams').available(False)
        and SolverFactory('gams').license_is_valid(),
        "gams solver not available",
    )
    def test_solve_four_stage_dynamic_model(self):

        model = build_model(mode_transfer=True)

        # Discretize the model using dae.collocation
        discretizer = TransformationFactory('dae.collocation')
        discretizer.apply_to(model, nfe=10, ncp=3, scheme='LAGRANGE-RADAU')
        # We need to reconstruct the constraints in disjuncts after discretization.
        # This is a bug in Pyomo.dae. https://github.com/Pyomo/pyomo/issues/3101
        for disjunct in model.component_data_objects(ctype=Disjunct):
            for constraint in disjunct.component_objects(ctype=Constraint):
                constraint._constructed = False
                constraint.construct()

        for dxdt in model.component_data_objects(ctype=Var, descend_into=True):
            if 'dxdt' in dxdt.name:
                dxdt.setlb(-300)
                dxdt.setub(300)

        for direction_norm in ['L2', 'Linf']:
            result = SolverFactory('gdpopt.ldsda').solve(
                model,
                direction_norm=direction_norm,
                minlp_solver='gams',
                minlp_solver_args=dict(solver='ipopth'),
                starting_point=[1, 2],
                logical_constraint_list=[
                    model.mode_transfer_lc1,
                    model.mode_transfer_lc2,
                ],
                time_limit=100,
            )
            self.assertAlmostEqual(value(model.obj), -23.305325, places=4)


class TestLDSDALinearSearchUnit(unittest.TestCase):
    def test_line_search_tuple_unpacking(self):
        """
        Test that line_search correctly unpacks the (bool, float) tuple
        returned by _solve_GDP_subproblem.
        """
        # 1. Instantiate the solver class directly
        solver = GDP_LDSDA_Solver()

        # 2. Set up the fake internal state required for line_search
        solver.current_point = (0, 0)
        solver.best_direction = (1, 1)

        # 3. Mock the internal methods
        # check_valid_neighbor: Always say the neighbor is valid
        solver._check_valid_neighbor = MagicMock(return_value=True)

        # solve_GDP_subproblem: SIMULATE THE RETURN VALUES
        # Call 1: Returns (True, 10.0) -> Improvement found. Loop should continue.
        # Call 2: Returns (False, 10.0) -> No improvement. Loop SHOULD break.
        # If your fix works, the loop breaks here. If broken, it loops infinitely or crashes.
        solver._solve_GDP_subproblem = MagicMock(
            side_effect=[(True, 10.0), (False, 10.0)]
        )

        # 4. Run the method
        config = MagicMock()
        solver.line_search(config)

        # 5. Verify results
        # The solver should have moved exactly ONCE (from 0,0 to 1,1)
        self.assertEqual(solver.current_point, (1, 1))

        # The solver should have attempted to solve exactly TWICE
        # (Once for the success, once for the failure that stops the loop)
        self.assertEqual(solver._solve_GDP_subproblem.call_count, 2)

class TestLDSDAUnits(unittest.TestCase):
    """
    Unit tests for solver-independent LDSDA logic, targeting critical control-flow,
    validation, and regression-sensitive code paths rather than full coverage.
    """

    def setUp(self):
        self.solver = GDP_LDSDA_Solver()
        self.config = self.solver.CONFIG()
        # self.config.logger = MagicMock()
        
        self.model = ConcreteModel()
        self.model.util_block = Block()
        self.model.util_block.external_var_info_list = []
        self.model.util_block.parent_block = MagicMock(return_value=self.model)


    def test_any_termination_criterion_met(self):
        """
        Covers line 147: 'break' when termination criterion is met.
        """

        # 1. Mock critical methods to prevent real solving
        self.solver._solve_GDP_subproblem = MagicMock(return_value=(True, 0))
        # This forces the 'break' at line 147 to execute immediately
        self.solver.any_termination_criterion_met = MagicMock(return_value=True)
        self.solver.neighbor_search = MagicMock()
        
        # 2. Mock internal setup methods
        self.solver._get_external_information = MagicMock()
        self.number_of_external_variables = []
        self.solver._get_directions = MagicMock(return_value=[])

        # 3. FIX: Manually set the attribute that _get_external_information would have set
        self.solver.number_of_external_variables = 1
        self.config.starting_point = [0]

        # 4. Run the method
        self.solver._solve_gdp(self.model, self.config)
        
        # 5. Verify we hit the break (neighbor_search was skipped)
        self.solver.neighbor_search.assert_not_called()
        self.solver.any_termination_criterion_met.assert_called()

    def test_invalid_logical_constraint_type(self):
        """Covers line 230: ValueError for non-Exactly logical constraints"""
        self.model.b = BooleanVar()
        self.model.lc = LogicalConstraint(expr=self.model.b.implies(True))
        
        self.config.logical_constraint_list = [self.model.lc]
        self.model.util_block.config_logical_constraint_list = [self.model.lc]

        with self.assertRaisesRegex(ValueError, "should be a list of ExactlyExpression"):
            self.solver._get_external_information(self.model.util_block, self.config)

    def test_exactly_number_greater_than_one(self):
        """Covers line 236: ValueError for Exactly(N) where N > 1"""
        self.model.b1 = BooleanVar()
        self.model.b2 = BooleanVar()
        self.model.lc = LogicalConstraint(expr=exactly(2, self.model.b1, self.model.b2))
        
        self.config.logical_constraint_list = [self.model.lc]
        self.model.util_block.config_logical_constraint_list = [self.model.lc]

        with self.assertRaisesRegex(ValueError, "only works for exactly_number = 1"):
            self.solver._get_external_information(self.model.util_block, self.config)

    def test_starting_point_mismatch(self):
        """Covers line 287: Starting point length mismatch"""
        self.model.b1 = BooleanVar()
        self.model.lc = LogicalConstraint(expr=exactly(1, self.model.b1))
        
        self.config.logical_constraint_list = [self.model.lc]
        self.model.util_block.config_logical_constraint_list = [self.model.lc]
        
        # Mismatch: 1 external var vs 2 starting points
        self.config.starting_point = [1, 2]

        with self.assertRaisesRegex(ValueError, "length of the provided starting point"):
            self.solver._get_external_information(self.model.util_block, self.config)
    
    def test_disjunction_list_processing(self):
        """Covers lines 254-266: Processing config.disjunction_list"""
        self.model.d1 = Disjunct()
        self.model.d2 = Disjunct()
        self.model.disj = Disjunction(expr=[self.model.d1, self.model.d2])
        
        self.config.disjunction_list = [self.model.disj]
        self.model.util_block.config_disjunction_list = [self.model.disj]
        self.config.starting_point = [1] # Correct length

        self.solver._get_external_information(self.model.util_block, self.config)

        # Verify it processed the disjunction
        self.assertEqual(len(self.model.util_block.external_var_info_list), 1)
        self.assertEqual(self.model.util_block.external_var_info_list[0].UB, 2)

    def test_neighbor_search_tiebreaker_logic(self):
        """Covers lines 398-406: Tie-breaking logic in neighbor_search"""
        self.solver.current_point = (0, 0)
        self.config.integer_tolerance = 1e-5
        
        # Manually define neighbors:
        # 1. (1,0) - Distance 1
        # 2. (2,0) - Distance 4 (Further away)
        self.solver.directions = [(1, 0), (1, 1)]
        self.solver._check_valid_neighbor = MagicMock(return_value=True)

        # Mock subproblems to return IDENTICAL objectives
        # This forces the code to check the distance to break the tie
        self.solver._solve_GDP_subproblem = MagicMock(side_effect=[
            (True, 100.0), 
            (True, 100.0)
        ])

        self.solver.neighbor_search(self.config)

        # It should pick (1,1) because it is further away (Tiebreaker rule)
        self.assertEqual(self.solver.current_point, (1, 1))

    def test_handle_subproblem_result_none(self):
        """Covers line 470: Returns False when subproblem result is None"""
        result = self.solver._handle_subproblem_result(None, (False, None), None, None, None)
        self.assertFalse(result)
    
    def test_handle_subproblem_result_termination_failure(self):
        """
        Covers line 496: return False when termination condition is not optimal/feasible.
        """
        # 1. Create a mock result object
        mock_result = MagicMock()
        
        # 2. Set termination condition to something NOT in the 'success' set
        # The code checks for: optimal, feasible, globallyOptimal, locallyOptimal, 
        # maxTimeLimit, maxIterations, maxEvaluations.
        # We use 'infeasible' or 'error' to trigger the False return.
        from pyomo.opt import TerminationCondition as tc
        mock_result.solver.termination_condition = tc.error
        
        # 3. Run the method
        # We don't need real model objects for the other args since the code exits early
        result = self.solver._handle_subproblem_result(
            mock_result, None, None, None, None
        )
        
        # 4. Verify it returns False (because the solver failed)
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()
