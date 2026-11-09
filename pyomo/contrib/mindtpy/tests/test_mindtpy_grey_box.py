# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Tests for the MindtPy solver."""

from pyomo.core.expr.calculus.diff_with_sympy import differentiate_available
import pyomo.common.unittest as unittest
from pyomo.environ import SolverFactory, value, maximize
from pyomo.opt import TerminationCondition
from pyomo.common.dependencies import numpy_available, scipy_available
from pyomo.contrib.mindtpy.tests.minlp_simple import MinlpSimple
from pyomo.contrib.mindtpy.tests.minlp_simple_grey_box import GreyBoxModel

model_list = [MinlpSimple]
grey_box_available = GreyBoxModel is not None and numpy_available and scipy_available

if SolverFactory('appsi_highs').available(exception_flag=False) and SolverFactory(
    'appsi_highs'
).version() >= (1, 7, 0):
    required_solvers = ('cyipopt', 'appsi_highs')
else:
    required_solvers = ('cyipopt', 'glpk')

if all(SolverFactory(s).available(exception_flag=False) for s in required_solvers):
    subsolvers_available = True
else:
    subsolvers_available = False


@unittest.skipIf(not grey_box_available, 'Unable to generate the Grey Box model.')
@unittest.skipIf(
    not subsolvers_available,
    'Required subsolvers %s are not available' % (required_solvers,),
)
@unittest.skipIf(
    not differentiate_available, 'Symbolic differentiation is not available'
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

    def test_OA_rNLP(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model_factory in model_list:
                model = model_factory(grey_box=True)
                results = opt.solve(
                    model,
                    strategy='OA',
                    init_strategy='rNLP',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                    calculate_dual_at_solution=True,
                    nlp_solver_args={
                        'options': {
                            'hessian_approximation': 'limited-memory',
                            'linear_solver': 'mumps',
                        }
                    },
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
