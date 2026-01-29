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
from pyomo.environ import SolverFactory, value, Var, Constraint, TransformationFactory
from pyomo.gdp import Disjunct
import pyomo.common.unittest as unittest
from pyomo.contrib.gdpopt.tests.four_stage_dynamic_model import build_model
from unittest.mock import MagicMock
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


class TestLDSDAUnit(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
