import pyutilib.th as unittest

from pyomo.opt import (TerminationCondition,
                       SolutionStatus,
                       SolverStatus)
import pyomo.environ as aml
import pyomo.kernel as pmo
import sys
from test_MOSEKDirect import *

try:
    import mosek
    mosek_available = True
    mosek_version = mosek.Env().getversion()
except ImportError:
    mosek_available = False
    modek_version = None

diff_tol = 1e-3

@unittest.skipIf(not mosek_available, "MOSEK's python bindings are missing.")
class MOSEKPersistentTests(MOSEKDirectTests):

    def test_model_modification(self):
        m = aml.ConcreteModel()
        m.x = aml.Var()
        m.y = aml.Var()

        opt = aml.SolverFactory('mosek_persistent')
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getnumvar(), 0)

        opt.remove_var(m.x)
        self.assertEqual(opt._solver_model.getnumvar(), 1)

        opt.add_var(m.x)
        self.assertEqual(opt._solver_modelgetnumvar(), 1)
        self.assertEqual(opt._solver_modelgetnumvar(), 2)

        opt.remove_var(m.x)
        opt.add_var(m.x)
        opt.remove_var(m.x)
        self.assertEqual(opt._solver_modelgetnumvar(), 1)