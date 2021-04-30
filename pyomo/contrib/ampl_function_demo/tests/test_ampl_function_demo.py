import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.core.base.external import ExternalFunction
from pyomo.common.fileutils import find_library
from pyomo.opt import check_available_solvers

flib = find_library("function_ASL")

class TestAMPLExternalFunction(unittest.TestCase):
    def test_eval_function(self):
        if not flib:
            self.skipTest("Could not find the function_ASL.dll library")
        m = pyo.ConcreteModel()
        m.tf = pyo.ExternalFunction(library=flib, function="testing_only")
        assert abs(pyo.value(m.tf("whatevs", 1, 2, 3) - 6))/6 < 1e-5
        assert abs(pyo.value(m.tf("inv", 1, 2, 3) - 1.8333333))/1.8333 < 1e-5

    @unittest.skipIf(not check_available_solvers('ipopt'),
                     "The 'ipopt' solver is not available")
    def test_solve_function(self):
        if not flib:
            self.skipTest("Could not find the function_ASL.dll library")
        m = pyo.ConcreteModel()
        m.tf = pyo.ExternalFunction(library=flib, function="testing_only")
        m.x = pyo.Var(initialize=0.5)
        m.x.fix()
        m.y = pyo.Var(initialize=4.0)
        m.c = pyo.Constraint(expr=m.x == m.tf("inv", 3, m.y))
        solver = pyo.SolverFactory("ipopt")
        solver.solve(m, tee=True)
        assert abs(pyo.value(m.y - 6))/6 < 1e-5
