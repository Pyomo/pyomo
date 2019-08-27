import pyutilib.th as unittest

from pyomo.opt import (TerminationCondition,
                       SolutionStatus,
                       SolverStatus)
import pyomo.environ as aml
import pyomo.kernel as pmo
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

        model = aml.ConcreteModel()
        model.X = aml.Var(within=aml.NonNegativeReals)
        model.C1 = aml.Constraint(expr=model.X == 1)
        model.C2 = aml.Constraint(expr=model.X == 2)
        model.O = aml.Objective(expr=model.X)

        opt = aml.SolverFactory("mosek")
        results = opt.solve(model)

        self.assertIn(results.solver.termination_condition,
                      (TerminationCondition.infeasible,
                       TerminationCondition.infeasibleOrUnbounded))

    def test_unbounded_lp(self):

        model = aml.ConcreteModel()
        model.X = aml.Var()
        model.O = aml.Objective(expr=model.X)

        opt = aml.SolverFactory("mosek")
        results = opt.solve(model)

        self.assertIn(results.solver.termination_condition,
                      (TerminationCondition.unbounded,
                       TerminationCondition.infeasibleOrUnbounded))

    def test_optimal_lp(self):

        model = aml.ConcreteModel()
        model.X = aml.Var(within=aml.NonNegativeReals)
        model.O = aml.Objective(expr=model.X)

        opt = aml.SolverFactory("mosek")
        results = opt.solve(model, load_solutions=False)

        self.assertEqual(results.solution.status,
                         SolutionStatus.optimal)

    def test_get_duals_lp(self):

        model = aml.ConcreteModel()
        model.X = aml.Var(within=aml.NonNegativeReals)
        model.Y = aml.Var(within=aml.NonNegativeReals)

        model.C1 = aml.Constraint(expr=2*model.X + model.Y >= 8)
        model.C2 = aml.Constraint(expr=model.X + 3*model.Y >= 6)

        model.O = aml.Objective(expr=model.X + model.Y)

        opt = aml.SolverFactory("mosek")
        results = opt.solve(model, suffixes=['dual'], load_solutions=False)

        model.dual = aml.Suffix(direction=aml.Suffix.IMPORT)
        model.solutions.load_from(results)

        self.assertAlmostEqual(model.dual[model.C1], 0.4, 4)
        self.assertAlmostEqual(model.dual[model.C2], 0.2, 4)

    def test_infeasible_mip(self):

        model = aml.ConcreteModel()
        model.X = aml.Var(within=aml.NonNegativeIntegers)
        model.C1 = aml.Constraint(expr=model.X == 1)
        model.C2 = aml.Constraint(expr=model.X == 2)
        model.O = aml.Objective(expr=model.X)

        opt = aml.SolverFactory("mosek")
        results = opt.solve(model)

        self.assertIn(results.solver.termination_condition,
                      (TerminationCondition.infeasibleOrUnbounded,
                       TerminationCondition.infeasible))

    def test_unbounded_mip(self):

        model = aml.AbstractModel()
        model.X = aml.Var(within=aml.Integers)
        model.O = aml.Objective(expr=model.X)

        instance = model.create_instance()
        opt = aml.SolverFactory("mosek")
        results = opt.solve(instance)

        self.assertIn(results.solver.termination_condition,
                      (TerminationCondition.unbounded,
                       TerminationCondition.infeasibleOrUnbounded))

    def test_optimal_mip(self):

        model = aml.ConcreteModel()
        model.X = aml.Var(within=aml.NonNegativeIntegers)
        model.O = aml.Objective(expr=model.X)

        opt = aml.SolverFactory("mosek")
        results = opt.solve(model, load_solutions=False)

        self.assertEqual(results.solution.status,
                         SolutionStatus.optimal)

    def test_optimal_mip(self):

        model = aml.ConcreteModel()
        model.X = aml.Var(within=aml.NonNegativeIntegers)
        model.O = aml.Objective(expr=model.X)

        opt = aml.SolverFactory("mosek")
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

        if mosek_version >= (9,0,0):
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

        opt = pmo.SolverFactory("mosek")
        results = opt.solve(model)

        self.assertEqual(results.solution.status,
                         SolutionStatus.optimal)

if __name__ == "__main__":
    unittest.main()
