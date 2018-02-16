"""Tests for the MINDT solver plugin."""
from math import fabs

import pyutilib.th as unittest

from pyomo.contrib.mindtpy.tests.batchdes import *

from pyomo.environ import SolverFactory, value

required_solvers = ('ipopt', 'gurobi')
if all(SolverFactory(s).available() for s in required_solvers):
    subsolvers_available = True
else:
    subsolvers_available = False

import pyomo.core.base.symbolic

class TestMindtPy(unittest.TestCase):
    """Tests for the MINDT solver plugin."""
    
    def test_model(self):
        """Test the MindtPy implementation."""
        with SolverFactory('mindtpy') as opt:
            print('\n Solving problem with selected strategy')
            opt.solve(model, strategy='ECP', init_strategy = 'initial_binary', mip = 'glpk')
            model.pprint()
    
            # self.assertIs(results.solver.termination_condition,
            #               TerminationCondition.optimal)
            # self.assertTrue(abs(value(model.cost.expr) - 3.5) <= 1E-2)
    
    

if __name__ == "__main__":
    unittest.main()
