# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import pyomo.environ as pyo
import pyomo.common.unittest as unittest
import gc
from pyomo.contrib.appsi.cmodel import cmodel_available

from pyomo.common.dependencies import attempt_import

# Note: tracemalloc is not always available, e.g., under PyPy
tracemalloc, tracemalloc_available = attempt_import('tracemalloc')


class TestAppsiLegacyLeak(unittest.TestCase):
    @unittest.skipIf(not tracemalloc_available, "tracemalloc not available")
    @unittest.skipIf(not cmodel_available, "APPSI C-extension not available")
    def test_legacy_solver_wrapper_memory_leak(self):
        tracemalloc.start()
        # 1. Create a minimal structure
        model = pyo.ConcreteModel()
        model.I = pyo.RangeSet(100)
        model.x = pyo.Var(model.I)
        model.obj = pyo.Objective(expr=sum(model.x[i] for i in model.I))
        model.c = pyo.ConstraintList()
        for i in model.I:
            model.c.add(model.x[i] >= 0)

        # 2. Instantiate Legacy Wrapper
        solver = pyo.SolverFactory('appsi_cbc')
        if not solver.available():
            raise unittest.SkipTest("appsi_cbc solver is not available")

        solver.set_instance(model)

        # Warm-up solve
        solver.solve(model)
        gc.collect()

        # 3. Take initial memory snapshot

        s1 = tracemalloc.take_snapshot()

        # 4. Perform iterative solves
        iterations = 10
        for _ in range(iterations):
            solver.solve(model)

        gc.collect()
        s2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # 5. Measure memory delta
        stats = s2.compare_to(s1, 'lineno')
        total_leak = sum(stat.size_diff for stat in stats)

        initial_size = sum(stat.size for stat in s1.statistics('lineno'))
        final_size = sum(stat.size for stat in s2.statistics('lineno'))
        percentage_increase_per_solve = (total_leak / iterations) / initial_size * 100

        # We allow a small tolerance for memory use growth, set here
        threshold_pct = 3
        print(f"Percentage increase per solve: {percentage_increase_per_solve}%")
        # Check if the leak is substantial
        self.assertLess(
            percentage_increase_per_solve,
            threshold_pct,
            f"More than {threshold_pct}% memory leak detected across iterations",
        )


if __name__ == "__main__":
    unittest.main()
