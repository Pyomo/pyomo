# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software. This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.contrib.piecewise.transform.epigraph_hypograph import (
    PWLToEpigraphOrHypograph,
)
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Var,
    Objective,
    TransformationFactory,
    SolverFactory,
    minimize,
    value,
)
from pyomo.contrib.piecewise.transform.piecewise_linear_transformation_base import (
    PiecewiseLinearTransformationBase,
)

gurobi_available = (
    SolverFactory('gurobi').available(exception_flag=False)
    and SolverFactory('gurobi').license_is_valid()
)


def x_squared(x):
    return x**2


class TestEpigraphHypographTransformation(unittest.TestCase):
    """Test the epigraph/hypograph transformation for piecewise linear functions."""

    def pwl_x_squared(self):
        # Create a model with a convex piecewise linear function
        m = ConcreteModel()
        m.x = Var(bounds=(0, 4))

        # Define breakpoints for x in [0, 4]
        breakpoints = [0, 1, 2, 3, 4]

        # Create piecewise linear approximation of x²
        m.pw = PiecewiseLinearFunction(
            points=breakpoints,
            function=x_squared,
            convex=True,  # Specify that this is convex
        )

        m.c = Constraint(expr=m.x >= 1.5)

        # Use the piecewise function in an objective - minimize to test the piecewise part
        m.obj = Objective(expr=m.pw(m.x), sense=minimize)

        return m

    def test_convex_pwl_minimization(self):
        """Test transformation of a convex piecewise linear function (x² approximation)."""
        m = self.pwl_x_squared()
        # Apply the epigraph transformation
        transformation = TransformationFactory('contrib.piecewise.epigraph_hypograph')
        transformation.apply_to(m)

        # Verify the transformation created the expected components
        # The transformation should have created:
        # 1. A substitute variable
        # 2. Epigraphical constraints (one for each linear segment)

        # Get the substitute var and the transformed function block
        sub_var = m.pw.get_transformation_var(m.obj.expr)
        self.assertIsInstance(sub_var, Var)
        pw_block = sub_var.parent_block()

        # Check that epigraphical constraints exist
        epigraphical_constraints = pw_block.component('epigraphical_constraints')
        self.assertIsInstance(epigraphical_constraints, Constraint)

        # For 5 breakpoints, we have 4 segments, so we should have 4 constraints
        self.assertEqual(len(epigraphical_constraints), 4)

        # Verify the objective now uses the substitute variable
        # The original pw(m.x) should be replaced with substitute_var
        obj_expr = m.obj.expr.expr
        self.assertIs(obj_expr, sub_var)

        exprs = [
            m.x - sub_var,
            3.0 * m.x - 2.0 - sub_var,
            5.0 * m.x - 6.0 - sub_var,
            7.0 * m.x - 12.0 - sub_var,
        ]

        # Verify each constraint is of the form: linear_func_expr <= substitute_var
        # (since this is a convex function, we use epigraph)
        for idx, cons in epigraphical_constraints.items():
            constraint = pw_block.epigraphical_constraints[idx]
            # For epigraph of convex function, we expect <= constraints
            # The constraint is stored as: linear_func_expr <= substitute_var
            # which Pyomo represents as: linear_func_expr - substitute_var <= 0
            # So the upper bound should be 0
            self.assertEqual(constraint.upper, 0)
            self.assertIsNone(constraint.lower)
            print(constraint.body)
            assertExpressionsEqual(self, constraint.body, exprs[idx])

    @unittest.skipUnless(gurobi_available, "Gurobi is not available")
    def test_solve_pwl_minimization(self):
        m = self.pwl_x_squared()
        # Apply the epigraph transformation
        transformation = TransformationFactory('contrib.piecewise.epigraph_hypograph')
        transformation.apply_to(m)

        # Now solve the model to verify it produces the correct solution
        # The minimum of x² for x >= 1.5 is at x = 1.5, with value 2.25
        solver = SolverFactory('gurobi')
        results = solver.solve(m)

        # Check that the solve was successful
        self.assertTrue(
            results.solver.termination_condition == 'optimal'
            or results.solver.termination_condition == 'feasible'
        )

        # Check the solution
        # The optimal x should be 1.5 (the lower bound)
        self.assertAlmostEqual(m.x.value, 1.5, places=4)

        # The objective value should be approximately 2.25 (1.5²)
        expected_obj = 3 * 1.5 - 2
        self.assertAlmostEqual(value(m.obj.expr), expected_obj, places=4)


if __name__ == '__main__':
    unittest.main()
