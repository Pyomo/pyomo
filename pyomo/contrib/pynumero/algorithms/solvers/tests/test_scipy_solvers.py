import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
    SquareNlpSolverBase,
)
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
    FsolveNlpSolver,
    RootNlpSolver,
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


class TestFsolveNLP(unittest.TestCase):

    def test_solve_simple_nlp(self):
        m, nlp = make_simple_model()
        solver = FsolveNlpSolver(nlp, options=dict(
            xtol=1e-9,
            maxiter=20,
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
            maxiter=10,
        ))
        x, info, ier, msg = solver.solve()
        self.assertNotEqual(ier, 1)
        self.assertIn("has reached maxfev", msg)

    def test_solve_too_tight_tol(self):
        m, nlp = make_simple_model()
        solver = FsolveNlpSolver(nlp, options=dict(
            xtol=1e-3,
            maxiter=20,
            tol=1e-8,
        ))
        msg = "does not satisfy the function tolerance"
        with self.assertRaisesRegex(RuntimeError, msg):
            x, info, ier, msg = solver.solve()


class TestFsolvePyomo(unittest.TestCase):

    def test_solve_simple_nlp(self):
        m, _ = make_simple_model()
        solver = pyo.SolverFactory("fsolve")
        results = solver.solve(m)
        solution = [m.x[1].value, m.x[2].value, m.x[3].value]
        predicted = [0.92846891, -0.22610731, 0.29465397]
        self.assertStructuredAlmostEqual(solution, predicted)


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


if __name__ == "__main__":
    unittest.main()
