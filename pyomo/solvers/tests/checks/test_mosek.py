#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.th as unittest

from pyomo.opt import (TerminationCondition,
                       SolutionStatus, SolverFactory)
from pyomo.environ import ConcreteModel, AbstractModel, Var, Objective, Constraint, Suffix, NonNegativeReals, NonNegativeIntegers, Integers 
from pyomo.kernel import block, objective, constraint, variable, variable_tuple, conic
import sys

try:
    import mosek
    mosek_available = True
    mosek_version = mosek.Env().getversion()
except ImportError:
    mosek_available = False
    modek_version = None

diff_tol = 1e-3


@unittest.skipIf(not mosek_available,
                 "The 'mosek' python bindings are not available")
class MosekDirectTests(unittest.TestCase):

    def setUp(self):
        self.stderr = sys.stderr
        sys.stderr = None

    def tearDown(self):
        sys.stderr = self.stderr

    def test_infeasible_lp(self):

        model =  ConcreteModel()
        model.X =  Var(within= NonNegativeReals)
        model.C1 =  Constraint(expr=model.X == 1)
        model.C2 =  Constraint(expr=model.X == 2)
        model.O =  Objective(expr=model.X)

        opt =  SolverFactory("mosek")
        results = opt.solve(model)

        self.assertIn(results.solver.termination_condition,
                      (TerminationCondition.infeasible,
                       TerminationCondition.infeasibleOrUnbounded))

    def test_unbounded_lp(self):

        model =  ConcreteModel()
        model.X =  Var()
        model.O =  Objective(expr=model.X)

        opt =  SolverFactory("mosek")
        results = opt.solve(model)

        self.assertIn(results.solver.termination_condition,
                      (TerminationCondition.unbounded,
                       TerminationCondition.infeasibleOrUnbounded))

    def test_optimal_lp(self):

        model =  ConcreteModel()
        model.X =  Var(within= NonNegativeReals)
        model.O =  Objective(expr=model.X)

        opt =  SolverFactory("mosek")
        results = opt.solve(model, load_solutions=False)

        self.assertEqual(results.solution.status,
                         SolutionStatus.optimal)

    def test_get_duals_lp(self):

        model =  ConcreteModel()
        model.X =  Var(within= NonNegativeReals)
        model.Y =  Var(within= NonNegativeReals)

        model.C1 =  Constraint(expr=2*model.X + model.Y >= 8)
        model.C2 =  Constraint(expr=model.X + 3*model.Y >= 6)

        model.O =  Objective(expr=model.X + model.Y)

        opt =  SolverFactory("mosek")
        results = opt.solve(model, suffixes=['dual'], load_solutions=False)

        model.dual =  Suffix(direction= Suffix.IMPORT)
        model.solutions.load_from(results)

        self.assertAlmostEqual(model.dual[model.C1], 0.4, 4)
        self.assertAlmostEqual(model.dual[model.C2], 0.2, 4)

    def test_infeasible_mip(self):

        model =  ConcreteModel()
        model.X =  Var(within= NonNegativeIntegers)
        model.C1 =  Constraint(expr=model.X == 1)
        model.C2 =  Constraint(expr=model.X == 2)
        model.O =  Objective(expr=model.X)

        opt =  SolverFactory("mosek")
        results = opt.solve(model)

        self.assertIn(results.solver.termination_condition,
                      (TerminationCondition.infeasibleOrUnbounded,
                       TerminationCondition.infeasible))

    def test_unbounded_mip(self):

        model =  AbstractModel()
        model.X =  Var(within= Integers)
        model.O =  Objective(expr=model.X)

        instance = model.create_instance()
        opt =  SolverFactory("mosek")
        results = opt.solve(instance)

        self.assertIn(results.solver.termination_condition,
                      (TerminationCondition.unbounded,
                       TerminationCondition.infeasibleOrUnbounded))

    def test_optimal_mip(self):

        model =  ConcreteModel()
        model.X =  Var(within= NonNegativeIntegers)
        model.O =  Objective(expr=model.X)

        opt =  SolverFactory("mosek")
        results = opt.solve(model, load_solutions=False)

        self.assertEqual(results.solution.status,
                         SolutionStatus.optimal)

    def test_optimal_mip(self):

        model =  ConcreteModel()
        model.X =  Var(within= NonNegativeIntegers)
        model.O =  Objective(expr=model.X)

        opt =  SolverFactory("mosek")
        results = opt.solve(model, load_solutions=False)

        self.assertEqual(results.solution.status,
                         SolutionStatus.optimal)

    def test_conic(self):

        model =  block()
        model.o =  objective(0.0)
        model.c =  constraint(body=0.0,
                                 rhs=1)

        b = model.quadratic =  block()
        b.x =  variable_tuple(( variable(),
                                   variable()))
        b.r =  variable(lb=0)
        b.c =  conic.quadratic(x=b.x,
                                  r=b.r)
        model.o.expr += b.r
        model.c.body += b.r
        del b

        b = model.rotated_quadratic =  block()
        b.x =  variable_tuple(( variable(),
                                   variable()))
        b.r1 =  variable(lb=0)
        b.r2 =  variable(lb=0)
        b.c =  conic.rotated_quadratic(x=b.x,
                                          r1=b.r1,
                                          r2=b.r2)
        model.o.expr += b.r1 + b.r2
        model.c.body += b.r1 + b.r2
        del b

        if mosek_version >= (9,0,0):
            b = model.primal_exponential =  block()
            b.x1 =  variable(lb=0)
            b.x2 =  variable()
            b.r =  variable(lb=0)
            b.c =  conic.primal_exponential(x1=b.x1,
                                               x2=b.x2,
                                               r=b.r)
            model.o.expr += b.r
            model.c.body += b.r
            del b

            b = model.primal_power =  block()
            b.x =  variable_tuple(( variable(),
                                       variable()))
            b.r1 =  variable(lb=0)
            b.r2 =  variable(lb=0)
            b.c =  conic.primal_power(x=b.x,
                                         r1=b.r1,
                                         r2=b.r2,
                                         alpha=0.6)
            model.o.expr += b.r1 + b.r2
            model.c.body += b.r1 + b.r2
            del b

            b = model.dual_exponential =  block()
            b.x1 =  variable()
            b.x2 =  variable(ub=0)
            b.r =  variable(lb=0)
            b.c =  conic.dual_exponential(x1=b.x1,
                                             x2=b.x2,
                                             r=b.r)
            model.o.expr += b.r
            model.c.body += b.r
            del b

            b = model.dual_power =  block()
            b.x =  variable_tuple(( variable(),
                                       variable()))
            b.r1 =  variable(lb=0)
            b.r2 =  variable(lb=0)
            b.c =  conic.dual_power(x=b.x,
                                       r1=b.r1,
                                       r2=b.r2,
                                       alpha=0.4)
            model.o.expr += b.r1 + b.r2
            model.c.body += b.r1 + b.r2

        opt =  SolverFactory("mosek")
        results = opt.solve(model)

        self.assertEqual(results.solution.status,
                         SolutionStatus.optimal)

if __name__ == "__main__":
    unittest.main()
