import pyutilib.th as unittest
import pyomo.environ as aml
import os

try:
    import scipy.sparse as spa
    import numpy as np
except ImportError:
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.contrib.pynumero.extensions.asl import AmplInterface
if not AmplInterface.available():
    raise unittest.SkipTest(
        "Pynumero needs the ASL extension to run NLP tests")

from pyomo.contrib.pynumero.interfaces.nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_transformations import AdmmNLP


def create_basic_dense_qp(G, A, b, c):

    nx = G.shape[0]
    nl = A.shape[0]

    model = aml.ConcreteModel()
    model.var_ids = range(nx)
    model.con_ids = range(nl)

    model.x = aml.Var(model.var_ids, initialize=0.0)
    model.hessian_f = G
    model.jacobian_c = A
    model.rhs = b
    model.grad_f = c

    def equality_constraint_rule(m, i):
        return sum(m.jacobian_c[i, j] * m.x[j] for j in m.var_ids) == m.rhs[i]
    model.equalities = aml.Constraint(model.con_ids, rule=equality_constraint_rule)

    def objective_rule(m):
        accum = 0.0
        for i in m.var_ids:
            accum += m.x[i] * sum(m.hessian_f[i, j] * m.x[j] for j in m.var_ids)
        accum *= 0.5
        accum += sum(m.x[j] * m.grad_f[j] for j in m.var_ids)
        return accum

    model.obj = aml.Objective(rule=objective_rule, sense=aml.minimize)

    return model


class TestAdmmNLP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # test problem 1
        G = np.array([[6, 2, 1], [2, 5, 2], [1, 2, 4]])
        A = np.array([[1, 0, 1], [0, 1, 1]])
        b = np.array([3, 0])
        c = np.array([-8, -3, -3])

        cls.model = create_basic_dense_qp(G, A, b, c)
        cls.pyomo_nlp = PyomoNLP(cls.model)
        cls.coupling_vars = [cls.pyomo_nlp.variable_idx(cls.model.x[0]),
                             cls.pyomo_nlp.variable_idx(cls.model.x[2])]
        cls.nlp = AdmmNLP(cls.pyomo_nlp, cls.coupling_vars, rho=2.0)

    def test_nx(self):
        self.assertEqual(self.nlp.nx, 3)

    def test_ng(self):
        self.assertEqual(self.nlp.ng, 2)

    def test_nc(self):
        self.assertEqual(self.nlp.nc, 2)

    def test_nd(self):
        self.assertEqual(self.nlp.nd, 0)

    def test_nz(self):
        self.assertEqual(self.nlp.nz, 2)

    def test_nw(self):
        self.assertEqual(self.nlp.nw, 2)

    def test_x_maps(self):
        lower_mask = [False, False, False]
        self.assertListEqual(self.nlp._lower_x_mask.tolist(), lower_mask)
        upper_mask = [False, False, False]
        self.assertListEqual(self.nlp._upper_x_mask.tolist(), upper_mask)
        self.assertFalse(self.nlp._lower_x_map.tolist())
        self.assertFalse(self.nlp._upper_x_map.tolist())

    def test_g_maps(self):
        c_mask = [True, True]
        self.assertListEqual(self.nlp._c_mask.tolist(), c_mask)
        d_mask = [False, False]
        self.assertListEqual(self.nlp._d_mask.tolist(), d_mask)
        c_map = [0, 1]
        self.assertListEqual(self.nlp._c_map.tolist(), c_map)
        self.assertFalse(self.nlp._d_map.tolist())
        lower_g_mask = [False, False]
        self.assertListEqual(self.nlp._lower_g_mask.tolist(), lower_g_mask)
        upper_g_mask = [False, False]
        self.assertListEqual(self.nlp._upper_g_mask.tolist(), upper_g_mask)
        self.assertFalse(self.nlp._lower_g_map.tolist())
        self.assertFalse(self.nlp._upper_g_map.tolist())


    def test_nnz_hessian_lag(self):
        self.assertEqual(self.nlp.nnz_hessian_lag, 6)

    def test_nnz_jacobian_g(self):
        self.assertEqual(self.nlp.nnz_jacobian_c, 4)

    def test_rho(self):
        self.assertEqual(self.nlp.rho, 2.0)
        self.nlp.rho = 5.0
        self.assertEqual(self.nlp.rho, 5.0)
        self.nlp.rho = 2.0

    def test_w_estimates(self):

        w_estimates = np.array([5.0, 5.0])
        nlp = AdmmNLP(self.pyomo_nlp,
                       self.coupling_vars,
                       rho=2.0,
                       w_estimates=w_estimates)
        self.assertTrue(np.allclose(nlp.w_estimates, w_estimates))
        w_estimates = np.array([6.0, 5.0])
        nlp.w_estimates = w_estimates
        self.assertTrue(np.allclose(nlp.w_estimates, w_estimates))
        self.assertEqual(len(nlp.create_vector_w()), 2)

    def test_z_estimates(self):

        z_estimates = np.array([5.0, 5.0])
        nlp = AdmmNLP(self.pyomo_nlp,
                       self.coupling_vars,
                       rho=2.0,
                       z_estimates=z_estimates)
        self.assertTrue(np.allclose(nlp.z_estimates, z_estimates))
        z_estimates = np.array([6.0, 5.0])
        nlp.z_estimates = z_estimates
        self.assertTrue(np.allclose(nlp.z_estimates, z_estimates))
        self.assertEqual(len(nlp.create_vector_z()), 2)


    def test_hessian_lag(self):

        hessian_base = self.model.hessian_f
        ata = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.double)
        rho = self.nlp.rho
        admm_hessian = hessian_base + ata * rho
        x = self.nlp.create_vector_x()
        y = self.nlp.create_vector_y()
        hess_lag = self.nlp.hessian_lag(x, y)
        dense_hess_lag = hess_lag.todense()
        self.assertTrue(np.allclose(dense_hess_lag, admm_hessian))


    def test_objective(self):

        w_estimates = np.array([5.0, 5.0])
        z_estimates = np.array([2.0, 2.0])
        rho = 2.0
        nlp = AdmmNLP(self.pyomo_nlp,
                       self.coupling_vars,
                       rho=rho,
                       w_estimates=w_estimates,
                       z_estimates=z_estimates)

        hessian_base = self.model.hessian_f
        c = self.model.grad_f
        A = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.double)
        x = nlp.create_vector_x()
        x.fill(1.0)
        f = 0.5 * x.transpose().dot(hessian_base.dot(x)) + c.dot(x)
        difference = A.dot(x) - z_estimates
        f += w_estimates.dot(difference)
        f += 0.5 * rho * np.linalg.norm(difference)**2
        self.assertEqual(f, nlp.objective(x))


    def test_grad_objective(self):

        w_estimates = np.array([5.0, 5.0])
        z_estimates = np.array([2.0, 2.0])
        rho = 2.0
        nlp = AdmmNLP(self.pyomo_nlp,
                      self.coupling_vars,
                      rho=rho,
                      w_estimates=w_estimates,
                      z_estimates=z_estimates)

        hessian_base = self.model.hessian_f
        c = self.model.grad_f
        A = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.double)
        x = nlp.create_vector_x()
        x.fill(1.0)

        df = hessian_base.dot(x) + c
        df += A.transpose().dot(w_estimates)
        df += rho * (A.transpose().dot(A).dot(x) - A.transpose().dot(z_estimates))

        self.assertTrue(np.allclose(df, nlp.grad_objective(x)))

    def test_jacobian_g(self):

        nlp = self.nlp
        jac_g = self.model.jacobian_c
        x = nlp.create_vector_x()
        self.assertTrue(np.allclose(nlp.jacobian_g(x).todense(), jac_g))

    def test_jacobian_c(self):
        nlp = self.nlp
        jac_c = self.model.jacobian_c
        x = nlp.create_vector_x()
        self.assertTrue(np.allclose(nlp.jacobian_c(x).todense(), jac_c))

    def test_jacobian_d(self):
        nlp = self.nlp
        x = nlp.create_vector_x()
        self.assertEqual(nlp.jacobian_d(x).shape, (0, 3))

    # ToDo: add test that modifies number of nonzeros in hessian