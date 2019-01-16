import pyutilib.th as unittest
try:
    import numpy as np
    from scipy.sparse import coo_matrix, tril
except ImportError:
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.contrib.pynumero.extensions.hsl import MA27_LinearSolver
if not MA27_LinearSolver.available():
    found_hsl = False
else:
    found_hsl = True

try:
    from pyomo.contrib.pynumero.linalg.solvers.mumps_solver import MUMPSSymLinearSolver
    found_mumps = True
except ImportError as e:
    found_mumps = False

if not found_mumps and not found_hsl:
    raise unittest.SkipTest("Pynumero needs pymumps or ma27 to run kkt solver tests")

from pyomo.contrib.pynumero.linalg.solvers.kkt_solver import FullKKTSolver, SchurComplementKKTSolver

from pyomo.contrib.pynumero.sparse import (BlockSymMatrix,
                                           BlockVector,
                                           empty_matrix)

from pyomo.contrib.pynumero.interfaces.qp import EqualityQP, EqualityQuadraticModel
from pyomo.contrib.pynumero.interfaces.nlp_compositions import TwoStageStochasticNLP


class TestFullKKTSolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        G = np.array([[6, 2, 1], [2, 5, 2], [1, 2, 4]])
        A = np.array([[1, 0, 1], [0, 1, 1]])

        Q = coo_matrix(G)
        A = coo_matrix(A)
        c = np.array([-8, -3, -3])
        b = np.array([3, 0])

        cls.qp_model1 = EqualityQuadraticModel(Q, c, 0.0, A, b)
        cls.qp1 = EqualityQP(cls.qp_model1)

    def test_solve(self):

        nlp = self.qp1
        x = nlp.create_vector_x()
        y = nlp.create_vector_y()

        # create KKT system
        kkt = BlockSymMatrix(4)
        kkt[0, 0] = nlp.hessian_lag(x, y)
        kkt[1, 1] = empty_matrix(nlp.nd, nlp.nd)
        kkt[2, 0] = nlp.jacobian_c(x)
        kkt[3, 0] = nlp.jacobian_d(x)
        kkt[3, 1] = empty_matrix(nlp.nd, nlp.nd)

        grad_x_lag_bar = nlp.grad_objective(x) + nlp.jacobian_c(x).transpose() * nlp.y_init()
        rhs = -BlockVector([grad_x_lag_bar, np.zeros(0), nlp.evaluate_c(x), np.zeros(0)])

        if found_mumps:
            solver = FullKKTSolver('mumps')
            sol, info = solver.solve(kkt, rhs, nlp=nlp)
            x = sol[0]
            yc = sol[2]
            self.assertTrue(np.allclose(x, np.array([2.0, -1.0, 1.0])))
            self.assertTrue(np.allclose(yc, np.array([-4.0, 1.0])))

        if found_hsl:
            solver = FullKKTSolver('ma27')
            sol, info = solver.solve(kkt, rhs, nlp=nlp)
            x = sol[0]
            yc = sol[2]
            self.assertTrue(np.allclose(x, np.array([2.0, -1.0, 1.0])))
            self.assertTrue(np.allclose(yc, np.array([-4.0, 1.0])))

@unittest.skipIf(not found_hsl, "Need ma27")
class TestSchurComplementKKTSolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        G = np.array([[6, 2, 1], [2, 5, 2], [1, 2, 4]])
        A = np.array([[1, 0, 1], [0, 1, 1]])
        b = np.array([3, 0])


        scenarios = dict()
        coupling_vars = dict()
        n_scenarios = 2
        np.random.seed(seed=985739465)
        bs = [b + np.random.normal(scale=10.0, size=2) for i in range(n_scenarios)]

        for i in range(n_scenarios):

            Q = coo_matrix(G)
            A = coo_matrix(A)
            c = np.array([-8, -3, -3])
            qp_model = EqualityQuadraticModel(Q, c, 0.0, A, bs[i])
            nlp = EqualityQP(qp_model)

            scenario_name = "s{}".format(i)
            scenarios[scenario_name] = nlp
            coupling_vars[scenario_name] = [0]

        cls.qp1 = TwoStageStochasticNLP(scenarios, coupling_vars)

    def test_solve(self):

        nlp = self.qp1
        x = nlp.create_vector_x()
        y = nlp.create_vector_y()

        # create KKT system
        kkt = BlockSymMatrix(4)
        kkt[0, 0] = nlp.hessian_lag(x, y)
        Ds = BlockSymMatrix(2)
        Ds[0, 0] = empty_matrix(nlp.nd, nlp.nd)
        Ds[1, 1] = empty_matrix(nlp.nd, nlp.nd)
        kkt[1, 1] = Ds
        kkt[2, 0] = nlp.jacobian_c(x)
        kkt[3, 0] = nlp.jacobian_d(x)
        kkt[3, 1] = empty_matrix(nlp.nd, nlp.nd)

        grad_x_lag_bar = nlp.grad_objective(x) + nlp.jacobian_c(x).transpose() * nlp.y_init()
        rhs = -BlockVector([grad_x_lag_bar,
                            BlockVector([np.zeros(0), np.zeros(0)]),
                            nlp.evaluate_c(x),
                            BlockVector([np.zeros(0), np.zeros(0)])
                            ])

        solver = SchurComplementKKTSolver('ma27')
        sol, info = solver.solve(kkt, rhs, nlp=nlp)
        x = sol[0]
        yc = sol[2]

        x_sol = np.array([8.44301052, -19.90217852, 6.98749613, 8.44301052, -0.73386561, -7.96556946, 8.44301052])
        y_sol = np.array([-66.23851729, 70.6498793, 21.17255261, 4.71444595, 55.39731506, -55.39731506])

        self.assertTrue(np.allclose(x.flatten(), x_sol))
        self.assertTrue(np.allclose(yc.flatten(), y_sol))
