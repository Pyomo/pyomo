#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import pyutilib.th as unittest
from pyomo.opt import *
from pyomo.core import *
import sys

try:
    import cplex
    cplexpy_available = True
except ImportError:
    cplexpy_available = False

diff_tol = 1e-4

class CPLEXDirectTests(unittest.TestCase):

    def setUp(self):
        self.stderr = sys.stderr
        sys.stderr = None

    def tearDown(self):
        sys.stderr = self.stderr

    @unittest.skipIf(not cplexpy_available,"The 'cplex' python bindings are not available")
    def test_infeasible_lp(self):
        self.opt = SolverFactory("cplex",solver_io="python")
        
        model = AbstractModel()
        model.X = Var(within=NonNegativeReals)
        model.C1 = Constraint(expr= model.X==1)
        model.C2 = Constraint(expr= model.X==2)
        model.O = Objective(expr= model.X)

        instance = model.create()
        results = self.opt.solve(instance)

        self.assertEqual(results.solver.termination_condition, TerminationCondition.infeasible)

    @unittest.skipIf(not cplexpy_available,"The 'cplex' python bindings are not available")
    def test_unbounded_lp(self):
        self.opt = SolverFactory("cplex",solver_io="python")
        
        model = AbstractModel()
        model.X = Var()
        model.O = Objective(expr= model.X)

        instance = model.create()
        results = self.opt.solve(instance)

        self.assertEqual(results.solver.termination_condition, TerminationCondition.unbounded)

    @unittest.skipIf(not cplexpy_available,"The 'cplex' python bindings are not available")
    def test_optimal_lp(self):
        self.opt = SolverFactory("cplex",solver_io="python")
        
        model = AbstractModel()
        model.X = Var(within=NonNegativeReals)
        model.O = Objective(expr= model.X)

        instance = model.create()
        results = self.opt.solve(instance)

        self.assertEqual(results.solution.status, SolutionStatus.optimal)

    @unittest.skipIf(not cplexpy_available,"The 'cplex' python bindings are not available")
    def test_get_duals_lp(self):
        self.opt = SolverFactory("cplex",solver_io="python")
        
        model = AbstractModel()
        model.X = Var(within=NonNegativeReals)
        model.Y = Var(within=NonNegativeReals)

        model.C1 = Constraint(expr= 2*model.X + model.Y >= 8 )
        model.C2 = Constraint(expr= model.X + 3*model.Y >= 6 )

        model.O = Objective(expr= model.X + model.Y)

        instance = model.create()
        instance.load(self.opt.solve(instance,suffixes=['dual']))

        self.assertAlmostEqual(instance.dual[instance.C1], 0.4)
        self.assertAlmostEqual(instance.dual[instance.C2], 0.2)




    @unittest.skipIf(not cplexpy_available,"The 'cplex' python bindings are not available")
    def test_infeasible_mip(self):
        self.opt = SolverFactory("cplex",solver_io="python")
        
        model = AbstractModel()
        model.X = Var(within=NonNegativeIntegers)
        model.C1 = Constraint(expr= model.X==1)
        model.C2 = Constraint(expr= model.X==2)
        model.O = Objective(expr= model.X)

        instance = model.create()
        results = self.opt.solve(instance)

        self.assertEqual(results.solver.termination_condition, TerminationCondition.infeasible)

    @unittest.skipIf(not cplexpy_available,"The 'cplex' python bindings are not available")
    def test_unbounded_mip(self):
        self.opt = SolverFactory("cplex",solver_io="python")
        
        model = AbstractModel()
        model.X = Var(within=Integers)
        model.O = Objective(expr= model.X)

        instance = model.create()
        results = self.opt.solve(instance)

        self.assertEqual(results.solver.termination_condition, TerminationCondition.unbounded)

    @unittest.skipIf(not cplexpy_available,"The 'cplex' python bindings are not available")
    def test_optimal_mip(self):
        self.opt = SolverFactory("cplex",solver_io="python")
        
        model = AbstractModel()
        model.X = Var(within=NonNegativeIntegers)
        model.O = Objective(expr= model.X)

        instance = model.create()
        results = self.opt.solve(instance)

        self.assertEqual(results.solution.status, SolutionStatus.optimal)





if __name__ == "__main__":
    unittest.main()
