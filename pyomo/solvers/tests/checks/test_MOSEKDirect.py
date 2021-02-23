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

from pyomo.opt import (
    TerminationCondition, SolutionStatus, check_available_solvers,
)
import pyomo.environ as pyo
import pyomo.kernel as pmo
import sys

diff_tol = 1e-3

mosek_available = check_available_solvers('mosek_direct')

@unittest.skipIf(not mosek_available ,
                 "MOSEK's python bindings are not available")
class MOSEKDirectTests(unittest.TestCase):

    def setUp(self):
        self.stderr = sys.stderr
        sys.stderr = None

    def tearDown(self):
        sys.stderr = self.stderr

    def test_interface_call(self):

        interface_instance = type(pyo.SolverFactory('mosek_direct'))
        alt_1 = pyo.SolverFactory('mosek')
        alt_2 = pyo.SolverFactory('mosek', solver_io='python')
        alt_3 = pyo.SolverFactory('mosek', solver_io='direct')
        self.assertIsInstance(alt_1, interface_instance)
        self.assertIsInstance(alt_2, interface_instance)
        self.assertIsInstance(alt_3, interface_instance)

    def test_infeasible_lp(self):

        model = pyo.ConcreteModel()
        model.X = pyo.Var(within=pyo.NonNegativeReals)
        model.C1 = pyo.Constraint(expr=model.X == 1)
        model.C2 = pyo.Constraint(expr=model.X == 2)
        model.O = pyo.Objective(expr=model.X)

        opt = pyo.SolverFactory("mosek_direct")
        results = opt.solve(model)

        self.assertIn(results.solver.termination_condition,
                      (TerminationCondition.infeasible,
                       TerminationCondition.infeasibleOrUnbounded))

    def test_unbounded_lp(self):

        model = pyo.ConcreteModel()
        model.X = pyo.Var()
        model.O = pyo.Objective(expr=model.X)

        opt = pyo.SolverFactory("mosek_direct")
        results = opt.solve(model)

        self.assertIn(results.solver.termination_condition,
                      (TerminationCondition.unbounded,
                       TerminationCondition.infeasibleOrUnbounded))

    def test_optimal_lp(self):

        model = pyo.ConcreteModel()
        model.X = pyo.Var(within=pyo.NonNegativeReals)
        model.O = pyo.Objective(expr=model.X)

        opt = pyo.SolverFactory("mosek_direct")
        results = opt.solve(model, load_solutions=False)

        self.assertEqual(results.solution.status,
                         SolutionStatus.optimal)

    def test_get_duals_lp(self):

        model = pyo.ConcreteModel()
        model.X = pyo.Var(within=pyo.NonNegativeReals)
        model.Y = pyo.Var(within=pyo.NonNegativeReals)

        model.C1 = pyo.Constraint(expr=2*model.X + model.Y >= 8)
        model.C2 = pyo.Constraint(expr=model.X + 3*model.Y >= 6)

        model.O = pyo.Objective(expr=model.X + model.Y)

        opt = pyo.SolverFactory("mosek_direct")
        results = opt.solve(model, suffixes=['dual'], load_solutions=False)

        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        model.solutions.load_from(results)

        self.assertAlmostEqual(model.dual[model.C1], 0.4, 4)
        self.assertAlmostEqual(model.dual[model.C2], 0.2, 4)

    def test_infeasible_mip(self):

        model = pyo.ConcreteModel()
        model.X = pyo.Var(within=pyo.NonNegativeIntegers)
        model.C1 = pyo.Constraint(expr=model.X == 1)
        model.C2 = pyo.Constraint(expr=model.X == 2)
        model.O = pyo.Objective(expr=model.X)

        opt = pyo.SolverFactory("mosek_direct")
        results = opt.solve(model)

        self.assertIn(results.solver.termination_condition,
                      (TerminationCondition.infeasibleOrUnbounded,
                       TerminationCondition.infeasible))

    def test_unbounded_mip(self):

        model = pyo.AbstractModel()
        model.X = pyo.Var(within=pyo.Integers)
        model.O = pyo.Objective(expr=model.X)

        instance = model.create_instance()
        opt = pyo.SolverFactory("mosek_direct")
        results = opt.solve(instance)

        self.assertIn(results.solver.termination_condition,
                      (TerminationCondition.unbounded,
                       TerminationCondition.infeasibleOrUnbounded))

    def test_optimal_mip(self):

        model = pyo.ConcreteModel()
        model.X = pyo.Var(within=pyo.NonNegativeIntegers)
        model.O = pyo.Objective(expr=model.X)

        opt = pyo.SolverFactory("mosek_direct")
        results = opt.solve(model, load_solutions=False)

        self.assertEqual(results.solution.status,
                         SolutionStatus.optimal)

    def test_conic(self):

        model = pmo.block()
        model.o = pmo.objective(0.0)
        model.c = pmo.constraint(body=0.0,
                                 rhs=1)

        b = model.quadratic = pmo.block()
        b.x = pmo.variable_tuple((pmo.variable(),
                                  pmo.variable()))
        b.r = pmo.variable(lb=0)
        b.c = pmo.conic.quadratic(x=b.x,
                                  r=b.r)
        model.o.expr += b.r
        model.c.body += b.r
        del b

        b = model.rotated_quadratic = pmo.block()
        b.x = pmo.variable_tuple((pmo.variable(),
                                  pmo.variable()))
        b.r1 = pmo.variable(lb=0)
        b.r2 = pmo.variable(lb=0)
        b.c = pmo.conic.rotated_quadratic(x=b.x,
                                          r1=b.r1,
                                          r2=b.r2)
        model.o.expr += b.r1 + b.r2
        model.c.body += b.r1 + b.r2
        del b

        import mosek
        if mosek.Env().getversion() >= (9, 0, 0):
            b = model.primal_exponential = pmo.block()
            b.x1 = pmo.variable(lb=0)
            b.x2 = pmo.variable()
            b.r = pmo.variable(lb=0)
            b.c = pmo.conic.primal_exponential(x1=b.x1,
                                               x2=b.x2,
                                               r=b.r)
            model.o.expr += b.r
            model.c.body += b.r
            del b

            b = model.primal_power = pmo.block()
            b.x = pmo.variable_tuple((pmo.variable(),
                                      pmo.variable()))
            b.r1 = pmo.variable(lb=0)
            b.r2 = pmo.variable(lb=0)
            b.c = pmo.conic.primal_power(x=b.x,
                                         r1=b.r1,
                                         r2=b.r2,
                                         alpha=0.6)
            model.o.expr += b.r1 + b.r2
            model.c.body += b.r1 + b.r2
            del b

            b = model.dual_exponential = pmo.block()
            b.x1 = pmo.variable()
            b.x2 = pmo.variable(ub=0)
            b.r = pmo.variable(lb=0)
            b.c = pmo.conic.dual_exponential(x1=b.x1,
                                             x2=b.x2,
                                             r=b.r)
            model.o.expr += b.r
            model.c.body += b.r
            del b

            b = model.dual_power = pmo.block()
            b.x = pmo.variable_tuple((pmo.variable(),
                                      pmo.variable()))
            b.r1 = pmo.variable(lb=0)
            b.r2 = pmo.variable(lb=0)
            b.c = pmo.conic.dual_power(x=b.x,
                                       r1=b.r1,
                                       r2=b.r2,
                                       alpha=0.4)
            model.o.expr += b.r1 + b.r2
            model.c.body += b.r1 + b.r2

        opt = pmo.SolverFactory("mosek_direct")
        results = opt.solve(model)

        self.assertEqual(results.solution.status,
                         SolutionStatus.optimal)


if __name__ == "__main__":
    unittest.main()
