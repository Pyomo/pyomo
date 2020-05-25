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
from pyomo.opt import *
from pyomo.environ import *
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

    @unittest.skipIf(not cplexpy_available,
                     "The 'cplex' python bindings are not available")
    def test_infeasible_lp(self):
        with SolverFactory("cplex", solver_io="python") as opt:

            model = ConcreteModel()
            model.X = Var(within=NonNegativeReals)
            model.C1 = Constraint(expr= model.X==1)
            model.C2 = Constraint(expr= model.X==2)
            model.O = Objective(expr= model.X)

            results = opt.solve(model)

            self.assertEqual(results.solver.termination_condition,
                             TerminationCondition.infeasible)

    @unittest.skipIf(not cplexpy_available,
                     "The 'cplex' python bindings are not available")
    def test_unbounded_lp(self):
        with SolverFactory("cplex", solver_io="python") as opt:

            model = ConcreteModel()
            model.X = Var()
            model.O = Objective(expr= model.X)

            results = opt.solve(model)

            self.assertIn(results.solver.termination_condition,
                          (TerminationCondition.unbounded,
                           TerminationCondition.infeasibleOrUnbounded))

    @unittest.skipIf(not cplexpy_available,
                     "The 'cplex' python bindings are not available")
    def test_optimal_lp(self):
        with SolverFactory("cplex", solver_io="python") as opt:

            model = ConcreteModel()
            model.X = Var(within=NonNegativeReals)
            model.O = Objective(expr= model.X)

            results = opt.solve(model, load_solutions=False)

            self.assertEqual(results.solution.status,
                             SolutionStatus.optimal)

    @unittest.skipIf(not cplexpy_available,
                     "The 'cplex' python bindings are not available")
    def test_get_duals_lp(self):
        with SolverFactory("cplex", solver_io="python") as opt:

            model = ConcreteModel()
            model.X = Var(within=NonNegativeReals)
            model.Y = Var(within=NonNegativeReals)

            model.C1 = Constraint(expr= 2*model.X + model.Y >= 8 )
            model.C2 = Constraint(expr= model.X + 3*model.Y >= 6 )

            model.O = Objective(expr= model.X + model.Y)

            results = opt.solve(model, suffixes=['dual'], load_solutions=False)

            model.dual = Suffix(direction=Suffix.IMPORT)
            model.solutions.load_from(results)

            self.assertAlmostEqual(model.dual[model.C1], 0.4)
            self.assertAlmostEqual(model.dual[model.C2], 0.2)

    @unittest.skipIf(not cplexpy_available,
                     "The 'cplex' python bindings are not available")
    def test_infeasible_mip(self):
        with SolverFactory("cplex", solver_io="python") as opt:

            model = ConcreteModel()
            model.X = Var(within=NonNegativeIntegers)
            model.C1 = Constraint(expr= model.X==1)
            model.C2 = Constraint(expr= model.X==2)
            model.O = Objective(expr= model.X)

            results = opt.solve(model)

            self.assertEqual(results.solver.termination_condition,
                             TerminationCondition.infeasible)

    @unittest.skipIf(not cplexpy_available,
                     "The 'cplex' python bindings are not available")
    def test_unbounded_mip(self):
        with SolverFactory("cplex", solver_io="python") as opt:

            model = AbstractModel()
            model.X = Var(within=Integers)
            model.O = Objective(expr= model.X)

            instance = model.create_instance()
            results = opt.solve(instance)

            self.assertIn(results.solver.termination_condition,
                          (TerminationCondition.unbounded,
                           TerminationCondition.infeasibleOrUnbounded))

    @unittest.skipIf(not cplexpy_available,
                     "The 'cplex' python bindings are not available")
    def test_optimal_mip(self):
        with SolverFactory("cplex", solver_io="python") as opt:

            model = ConcreteModel()
            model.X = Var(within=NonNegativeIntegers)
            model.O = Objective(expr= model.X)

            results = opt.solve(model, load_solutions=False)

            self.assertEqual(results.solution.status,
                             SolutionStatus.optimal)


@unittest.skipIf(not unittest.mock_available, "'mock' is not available")
@unittest.skipIf(not cplexpy_available, "The 'cplex' python bindings are not available")
class TestIsFixedCallCount(unittest.TestCase):
    """ Tests for PR#1402 (669e7b2b) """
    def setup(self, skip_trivial_constraints):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.c1 = Constraint(expr=m.x + m.y == 1)
        m.c2 = Constraint(expr=m.x <= 1)
        self.assertFalse(m.c2.has_lb())
        self.assertTrue(m.c2.has_ub())
        self._model = m

        self._opt = SolverFactory("cplex_persistent")
        self._opt.set_instance(
            self._model, skip_trivial_constraints=skip_trivial_constraints
        )

    def test_skip_trivial_and_call_count_for_fixed_con_is_one(self):
        self.setup(skip_trivial_constraints=True)
        self._model.x.fix(1)
        self.assertTrue(self._opt._skip_trivial_constraints)
        self.assertTrue(self._model.c2.body.is_fixed())

        with unittest.mock.patch(
            "pyomo.solvers.plugins.solvers.cplex_direct.is_fixed", wraps=is_fixed
        ) as mock_is_fixed:
            self.assertEqual(mock_is_fixed.call_count, 0)
            self._opt.add_constraint(self._model.c2)
            self.assertEqual(mock_is_fixed.call_count, 1)

    def test_skip_trivial_and_call_count_for_unfixed_con_is_two(self):
        self.setup(skip_trivial_constraints=True)
        self.assertTrue(self._opt._skip_trivial_constraints)
        self.assertFalse(self._model.c2.body.is_fixed())

        with unittest.mock.patch(
            "pyomo.solvers.plugins.solvers.cplex_direct.is_fixed", wraps=is_fixed
        ) as mock_is_fixed:
            self.assertEqual(mock_is_fixed.call_count, 0)
            self._opt.add_constraint(self._model.c2)
            self.assertEqual(mock_is_fixed.call_count, 2)

    def test_skip_trivial_and_call_count_for_unfixed_equality_con_is_three(self):
        self.setup(skip_trivial_constraints=True)
        self._model.c2 = Constraint(expr=self._model.x == 1)
        self.assertTrue(self._opt._skip_trivial_constraints)
        self.assertFalse(self._model.c2.body.is_fixed())

        with unittest.mock.patch(
            "pyomo.solvers.plugins.solvers.cplex_direct.is_fixed", wraps=is_fixed
        ) as mock_is_fixed:
            self.assertEqual(mock_is_fixed.call_count, 0)
            self._opt.add_constraint(self._model.c2)
            self.assertEqual(mock_is_fixed.call_count, 3)

    def test_dont_skip_trivial_and_call_count_for_fixed_con_is_one(self):
        self.setup(skip_trivial_constraints=False)
        self._model.x.fix(1)
        self.assertFalse(self._opt._skip_trivial_constraints)
        self.assertTrue(self._model.c2.body.is_fixed())

        with unittest.mock.patch(
            "pyomo.solvers.plugins.solvers.cplex_direct.is_fixed", wraps=is_fixed
        ) as mock_is_fixed:
            self.assertEqual(mock_is_fixed.call_count, 0)
            self._opt.add_constraint(self._model.c2)
            self.assertEqual(mock_is_fixed.call_count, 1)

    def test_dont_skip_trivial_and_call_count_for_unfixed_con_is_one(self):
        self.setup(skip_trivial_constraints=False)
        self.assertFalse(self._opt._skip_trivial_constraints)
        self.assertFalse(self._model.c2.body.is_fixed())

        with unittest.mock.patch(
            "pyomo.solvers.plugins.solvers.cplex_direct.is_fixed", wraps=is_fixed
        ) as mock_is_fixed:
            self.assertEqual(mock_is_fixed.call_count, 0)
            self._opt.add_constraint(self._model.c2)
            self.assertEqual(mock_is_fixed.call_count, 1)


if __name__ == "__main__":
    unittest.main()
