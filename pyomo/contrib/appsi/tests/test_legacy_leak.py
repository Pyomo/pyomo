import pyomo.environ as pyo
import pyomo.common.unittest as unittest
import tracemalloc
import gc
from pyomo.contrib.appsi.cmodel import cmodel_available
"""
With the fix in place, 
"""
class TestAppsiLegacyLeak(unittest.TestCase):
    @unittest.skipIf(not cmodel_available, "APPSI C-extension not available")
    def test_legacy_solver_wrapper_memory_leak(self):
        # 1. Create a minimal structure
        model = pyo.ConcreteModel()
        model.I = pyo.RangeSet(10000)
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
        tracemalloc.start()
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
        
        # We allow a small tolerance, but not 10-100KB per loop
        # Check if the leak is substantial
        self.assertLess(total_leak, 50 * 1024, "Substantial memory leak detected across iterations")

if __name__ == "__main__":
    unittest.main()
