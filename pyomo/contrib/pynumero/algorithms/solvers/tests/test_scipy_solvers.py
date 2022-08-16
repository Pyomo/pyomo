import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
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
    pass


class TestFsolveNLP(unittest.TestCase):

    def test_solve_simple_nlp(self):
        m, nlp = make_simple_model()
        solver = FsolveNlpSolver(nlp)
        results = solver.solve()

        variables = [m.x[1], m.x[2], m.x[3]]
        predicted_xorder = [0.92846891, -0.22610731, 0.29465397]
        indices = nlp.get_primal_indices(variables)
        nlp_to_x_indices = [None]*len(variables)
        for i, j in enumerate(indices):
            nlp_to_x_indices[j] = i
        predicted_nlporder = [predicted_xorder[i] for i in nlp_to_x_indices]
        self.assertStructuredAlmostEqual(
            results.tolist(),
            predicted_nlporder,
        )


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
    m, nlp = make_simple_model()
