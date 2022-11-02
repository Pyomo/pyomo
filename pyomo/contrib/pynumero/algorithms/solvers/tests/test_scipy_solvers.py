#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
from pyomo.common.dependencies import scipy, scipy_available
import pyomo.environ as pyo

if not scipy_available:
    raise unittest.SkipTest("SciPy is needed to test the SciPy solvers")

from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
    SquareNlpSolverBase,
)
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
    FsolveNlpSolver,
    RootNlpSolver,
    PyomoScipySolver,
)


def make_simple_model():
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=[1, 2, 3])
    m.x = pyo.Var(m.I, initialize=1.0)
    m.con1 = pyo.Constraint(expr=m.x[1]**2 + m.x[2]**2 + m.x[3]**2 == 1)
    m.con2 = pyo.Constraint(expr=2*m.x[1] + 3*m.x[2] - 4*m.x[3] == 0)
    m.con3 = pyo.Constraint(expr=m.x[1] == 2*pyo.exp(m.x[2]/m.x[3]))
    m.obj = pyo.Objective(expr=0.0)
    nlp = PyomoNLP(m)
    return m, nlp


@unittest.skipUnless(AmplInterface.available(), "AmplInterface is not available")
class TestSquareSolverBase(unittest.TestCase):

    def test_not_implemented_solve(self):
        m, nlp = make_simple_model()
        solver = SquareNlpSolverBase(nlp)
        msg = "has not implemented the solve method"
        with self.assertRaisesRegex(NotImplementedError, msg):
            solver.solve()

    def test_not_square(self):
        m, _ = make_simple_model()
        m.con4 = pyo.Constraint(expr=m.x[1] == m.x[2])
        nlp = PyomoNLP(m)
        msg = "same numbers of variables as equality constraints"
        with self.assertRaisesRegex(RuntimeError, msg):
            solver = SquareNlpSolverBase(nlp)

    def test_bounds_and_ineq_okay(self):
        m, _ = make_simple_model()
        m.x[1].setlb(0.0)
        m.x[1].setub(1.0)
        m.con4 = pyo.Constraint(expr=m.x[1] <= m.x[2])
        nlp = PyomoNLP(m)
        # Just construct the solver and get no error
        solver = SquareNlpSolverBase(nlp)


@unittest.skipUnless(AmplInterface.available(), "AmplInterface is not available")
class TestFsolveNLP(unittest.TestCase):

    def test_solve_simple_nlp(self):
        m, nlp = make_simple_model()
        solver = FsolveNlpSolver(nlp, options=dict(
            xtol=1e-9,
            maxfev=20,
            tol=1e-8,
        ))
        x, info, ier, msg = solver.solve()
        self.assertEqual(ier, 1)

        variables = [m.x[1], m.x[2], m.x[3]]
        predicted_xorder = [0.92846891, -0.22610731, 0.29465397]
        indices = nlp.get_primal_indices(variables)
        nlp_to_x_indices = [None]*len(variables)
        for i, j in enumerate(indices):
            nlp_to_x_indices[j] = i
        predicted_nlporder = [predicted_xorder[i] for i in nlp_to_x_indices]
        self.assertStructuredAlmostEqual(
            nlp.get_primals().tolist(),
            predicted_nlporder,
        )

    def test_solve_max_iter(self):
        m, nlp = make_simple_model()
        solver = FsolveNlpSolver(nlp, options=dict(
            xtol=1e-9,
            maxfev=10,
        ))
        x, info, ier, msg = solver.solve()
        self.assertNotEqual(ier, 1)
        self.assertIn("has reached maxfev", msg)

    def test_solve_too_tight_tol(self):
        m, nlp = make_simple_model()
        solver = FsolveNlpSolver(nlp, options=dict(
            xtol=1e-3,
            maxfev=20,
            tol=1e-8,
        ))
        msg = "does not satisfy the function tolerance"
        with self.assertRaisesRegex(RuntimeError, msg):
            x, info, ier, msg = solver.solve()


@unittest.skipUnless(AmplInterface.available(), "AmplInterface is not available")
class TestPyomoScipySolver(unittest.TestCase):

    def test_available_and_version(self):
        solver = PyomoScipySolver()
        self.assertTrue(solver.available())
        self.assertTrue(solver.license_is_valid())

        sp_version = tuple(
            int(num) for num in scipy.__version__.split('.')
        )
        self.assertEqual(sp_version, solver.version())


@unittest.skipUnless(AmplInterface.available(), "AmplInterface is not available")
class TestFsolvePyomo(unittest.TestCase):

    def test_available_and_version(self):
        solver = pyo.SolverFactory("fsolve")
        self.assertTrue(solver.available())
        self.assertTrue(solver.license_is_valid())

        sp_version = tuple(
            int(num) for num in scipy.__version__.split('.')
        )
        self.assertEqual(sp_version, solver.version())

    def test_solve_simple_nlp(self):
        m, _ = make_simple_model()
        solver = pyo.SolverFactory("fsolve")

        # Just want to make sure this option works
        solver.set_options(dict(full_output=False))

        results = solver.solve(m)
        solution = [m.x[1].value, m.x[2].value, m.x[3].value]
        predicted = [0.92846891, -0.22610731, 0.29465397]
        self.assertStructuredAlmostEqual(solution, predicted)

    def test_solve_results_obj(self):
        m, _ = make_simple_model()
        solver = pyo.SolverFactory("fsolve")
        results = solver.solve(m)
        solution = [m.x[1].value, m.x[2].value, m.x[3].value]
        predicted = [0.92846891, -0.22610731, 0.29465397]
        self.assertStructuredAlmostEqual(solution, predicted)

        self.assertEqual(results.problem.number_of_constraints, 3)
        self.assertEqual(results.problem.number_of_variables, 3)

        # Note that the solver returns termination condition feasible
        # rather than optimal...
        self.assertEqual(
            results.solver.termination_condition,
            pyo.TerminationCondition.feasible,
        )
        msg = "Solver failed to return an optimal solution"
        with self.assertRaisesRegex(RuntimeError, msg):
            pyo.assert_optimal_termination(results)
        self.assertEqual(
            results.solver.status, pyo.SolverStatus.ok
        )

    def test_solve_max_iter(self):
        m, _ = make_simple_model()
        solver = pyo.SolverFactory("fsolve")
        solver.set_options(dict(xtol=1e-9, maxfev=10))
        res = solver.solve(m)
        self.assertNotEqual(res.solver.return_code, 1)
        self.assertIn("has reached maxfev", res.solver.message)

    def test_solve_too_tight_tol(self):
        m, _ = make_simple_model()
        solver = pyo.SolverFactory("fsolve", options=dict(
            xtol=1e-3,
            maxfev=20,
            tol=1e-8,
        ))
        msg = "does not satisfy the function tolerance"
        with self.assertRaisesRegex(RuntimeError, msg):
            res = solver.solve(m)


@unittest.skipUnless(AmplInterface.available(), "AmplInterface is not available")
class TestRootNLP(unittest.TestCase):

    def test_solve_simple_nlp(self):
        m, nlp = make_simple_model()
        solver = RootNlpSolver(nlp)
        results = solver.solve()
        self.assertTrue(results.success)

        variables = [m.x[1], m.x[2], m.x[3]]
        predicted_xorder = [0.92846891, -0.22610731, 0.29465397]
        indices = nlp.get_primal_indices(variables)
        nlp_to_x_indices = [None]*len(variables)
        for i, j in enumerate(indices):
            nlp_to_x_indices[j] = i
        predicted_nlporder = [predicted_xorder[i] for i in nlp_to_x_indices]
        self.assertStructuredAlmostEqual(
            results.x.tolist(),
            predicted_nlporder,
        )


@unittest.skipUnless(AmplInterface.available(), "AmplInterface is not available")
class TestRootPyomo(unittest.TestCase):

    def test_available_and_version(self):
        solver = pyo.SolverFactory("root")
        self.assertTrue(solver.available())
        self.assertTrue(solver.license_is_valid())

        sp_version = tuple(
            int(num) for num in scipy.__version__.split('.')
        )
        self.assertEqual(sp_version, solver.version())

    def test_solve_simple_nlp(self):
        m, _ = make_simple_model()
        solver = pyo.SolverFactory("root")

        solver.set_options(dict(tol=1e-7))

        results = solver.solve(m)
        solution = [m.x[1].value, m.x[2].value, m.x[3].value]
        predicted = [0.92846891, -0.22610731, 0.29465397]
        self.assertStructuredAlmostEqual(solution, predicted)

    def test_solve_simple_nlp_levenberg_marquardt(self):
        m, _ = make_simple_model()
        solver = pyo.SolverFactory("root")

        solver.set_options(dict(tol=1e-7, method="lm"))

        results = solver.solve(m)
        solution = [m.x[1].value, m.x[2].value, m.x[3].value]
        predicted = [0.92846891, -0.22610731, 0.29465397]
        self.assertStructuredAlmostEqual(solution, predicted)

    def test_bad_method(self):
        m, _ = make_simple_model()
        solver = pyo.SolverFactory("root")

        solver.set_options(dict(tol=1e-7, method="some-solver"))
        with self.assertRaisesRegex(ValueError, "not in domain"):
            results = solver.solve(m)

    def test_solver_results_obj(self):
        m, _ = make_simple_model()
        solver = pyo.SolverFactory("root")

        solver.set_options(dict(tol=1e-7))

        results = solver.solve(m)
        solution = [m.x[1].value, m.x[2].value, m.x[3].value]
        predicted = [0.92846891, -0.22610731, 0.29465397]
        self.assertStructuredAlmostEqual(solution, predicted)

        self.assertEqual(results.problem.number_of_constraints, 3)
        self.assertEqual(results.problem.number_of_variables, 3)
        self.assertEqual(results.solver.return_code, 1)
        self.assertEqual(
            results.solver.termination_condition,
            pyo.TerminationCondition.feasible
        )
        self.assertEqual(results.solver.message, "The solution converged.")

    def test_solver_results_obj_levenberg_marquardt(self):
        m, _ = make_simple_model()
        solver = pyo.SolverFactory("root")

        solver.set_options(dict(tol=1e-7, method="lm"))

        results = solver.solve(m)
        solution = [m.x[1].value, m.x[2].value, m.x[3].value]
        predicted = [0.92846891, -0.22610731, 0.29465397]
        self.assertStructuredAlmostEqual(solution, predicted)

        self.assertEqual(results.problem.number_of_constraints, 3)
        self.assertEqual(results.problem.number_of_variables, 3)
        self.assertEqual(results.solver.return_code, 1)
        self.assertEqual(
            results.solver.termination_condition,
            pyo.TerminationCondition.feasible
        )
        self.assertEqual(results.solver.message, "The solution converged.")


if __name__ == "__main__":
    unittest.main()
