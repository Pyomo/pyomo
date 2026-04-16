# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
"""
Tests for example usage of Pyomo.DoE advanced diagnostics entry points.
"""

from pyomo.common.dependencies import numpy_available, scipy_available
import pyomo.common.unittest as unittest

from pyomo.opt import SolverFactory

from pyomo.contrib.doe.examples.reactor_example import run_reactor_doe
from pyomo.contrib.doe.examples.reactor_optimize_debug_example import (
    run_reactor_trace_debug_example,
)

ipopt_available = SolverFactory("ipopt").available()


@unittest.skipIf(not numpy_available, "Numpy is not available")
@unittest.skipIf(not scipy_available, "scipy is not available")
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestDoEExamples(unittest.TestCase):
    """Tests for standard and debug-oriented Pyomo.DoE example entry points."""

    def test_reactor_example_minimal_optimize_run(self):
        """
        Execute the standard reactor example in a minimal optimize configuration.

        The test:
        - runs the real example function,
        - skips optional factorial and plotting stages for speed,
        - verifies optimize mode returns solver/status fields and design results.
        """
        doe_obj = run_reactor_doe(
            compute_FIM_full_factorial=False,
            plot_factorial_results=False,
            run_optimal_doe=True,
            solver=SolverFactory("ipopt"),
        )

        self.assertEqual(doe_obj.results["Solver Status"], "ok")
        self.assertIn("Termination Condition", doe_obj.results)
        self.assertIn("Experiment Design", doe_obj.results)
        self.assertGreater(len(doe_obj.results["Experiment Design"]), 0)

    def test_reactor_debug_example_initialization_inspection(self):
        """
        Execute debug example in assemble-and-inspect mode (no final solve).

        This directly exercises the NLP initialization diagnostics logic exposed
        through ``run_config`` and checks that structured residual results are
        returned as documented by the debug example.
        """
        doe_obj = run_reactor_trace_debug_example(
            nfe=10,
            ncp=3,
            top_constraints=15,
            solve_final_model=False,
            solver=SolverFactory("ipopt"),
        )

        self.assertEqual(doe_obj.results["Solver Status"], "not_run")
        self.assertIn("Constraint Residuals", doe_obj.results)
        self.assertIn("post_initialization", doe_obj.results["Constraint Residuals"])
        residuals = doe_obj.results["Constraint Residuals"]["post_initialization"]
        self.assertGreater(len(residuals), 0)
        self.assertLessEqual(len(residuals), 15)
        self.assertEqual(
            set(residuals[0].keys()),
            {
                "constraint_name",
                "body",
                "lower_bound",
                "upper_bound",
                "violation",
                "constraint_type",
            },
        )


if __name__ == "__main__":
    unittest.main()
