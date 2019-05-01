import pyutilib.th as unittest
from pyomo.opt import *
from pyomo.environ import *
import sys

try:
    import mosek
    mosek_available = True
except ImportError:
    mosek_available = False

diff_tol = 1e-3


class MosekDirectTests(unittest.TestCase):

    def setUp(self):
        self.stderr = sys.stderr
        sys.stderr = None

    def tearDown(self):
        sys.stderr = self.stderr

    @unittest.skipIf(not mosek_available,
                     "The 'mosek' python bindings are not available")
    def test_infeasible_lp(self):
        with SolverFactory("mosek") as opt:

            model = ConcreteModel()
            model.X = Var(within=NonNegativeReals)
            model.C1 = Constraint(expr=model.X == 1)
            model.C2 = Constraint(expr=model.X == 2)
            model.O = Objective(expr=model.X)

            results = opt.solve(model)

            self.assertIn(results.solver.termination_condition,
                          (TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded))

    @unittest.skipIf(not mosek_available,
                     "The 'mosek' python bindings are not available")
    def test_unbounded_lp(self):
        with SolverFactory("mosek") as opt:

            model = ConcreteModel()
            model.X = Var()
            model.O = Objective(expr=model.X)

            results = opt.solve(model)

            self.assertIn(results.solver.termination_condition,
                          (TerminationCondition.unbounded,
                           TerminationCondition.infeasibleOrUnbounded))

    @unittest.skipIf(not mosek_available,
                     "The 'mosek' python bindings are not available")
    def test_optimal_lp(self):
        with SolverFactory("mosek") as opt:

            model = ConcreteModel()
            model.X = Var(within=NonNegativeReals)
            model.O = Objective(expr=model.X)

            results = opt.solve(model, load_solutions=False)

            self.assertEqual(results.solution.status,
                             SolutionStatus.optimal)

    @unittest.skipIf(not mosek_available,
                     "The 'mosek' python bindings are not available")
    def test_get_duals_lp(self):
        with SolverFactory("mosek") as opt:

            model = ConcreteModel()
            model.X = Var(within=NonNegativeReals)
            model.Y = Var(within=NonNegativeReals)

            model.C1 = Constraint(expr=2*model.X + model.Y >= 8)
            model.C2 = Constraint(expr=model.X + 3*model.Y >= 6)

            model.O = Objective(expr=model.X + model.Y)

            results = opt.solve(model, suffixes=['dual'], load_solutions=False)

            model.dual = Suffix(direction=Suffix.IMPORT)
            model.solutions.load_from(results)

            self.assertAlmostEqual(model.dual[model.C1], 0.4, 4)
            self.assertAlmostEqual(model.dual[model.C2], 0.2, 4)

    @unittest.skipIf(not mosek_available,
                     "The 'mosek' python bindings are not available")
    def test_infeasible_mip(self):
        with SolverFactory("mosek") as opt:

            model = ConcreteModel()
            model.X = Var(within=NonNegativeIntegers)
            model.C1 = Constraint(expr=model.X == 1)
            model.C2 = Constraint(expr=model.X == 2)
            model.O = Objective(expr=model.X)

            results = opt.solve(model)

            self.assertIn(results.solver.termination_condition,
                          (TerminationCondition.infeasibleOrUnbounded, TerminationCondition.infeasible))

    @unittest.skipIf(not mosek_available,
                     "The 'mosek' python bindings are not available")
    def test_unbounded_mip(self):
        with SolverFactory("mosek") as opt:

            model = AbstractModel()
            model.X = Var(within=Integers)
            model.O = Objective(expr=model.X)

            instance = model.create_instance()
            results = opt.solve(instance)

            self.assertIn(results.solver.termination_condition,
                          (TerminationCondition.unbounded,
                           TerminationCondition.infeasibleOrUnbounded))

    @unittest.skipIf(not mosek_available,
                     "The 'mosek' python bindings are not available")
    def test_optimal_mip(self):
        with SolverFactory("mosek") as opt:

            model = ConcreteModel()
            model.X = Var(within=NonNegativeIntegers)
            model.O = Objective(expr=model.X)

            results = opt.solve(model, load_solutions=False)

            self.assertEqual(results.solution.status,
                             SolutionStatus.optimal)


if __name__ == "__main__":
    unittest.main()
