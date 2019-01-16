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
from pyomo.common.plugin import alias
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

from pyomo.contrib.pynumero.interfaces.qp import EqualityQP, EqualityQuadraticModel
from pyomo.contrib.pynumero.sparse import empty_matrix
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.interfaces import PyomoNLP
import pyomo.environ as aml


def create_dense_qp():

    G = np.array([[6, 2, 1], [2, 5, 2], [1, 2, 4]])
    A = np.array([[1, 0, 1], [0, 1, 1]])
    b = np.array([3, 0])
    c = np.array([-8, -3, -3])

    nx = G.shape[0]
    nl = A.shape[0]

    model = aml.ConcreteModel()
    model.var_ids = range(nx)
    model.con_ids = range(nl)

    model.x = aml.Var(model.var_ids, initialize=0.0)
    model.hessian_f = aml.Param(model.var_ids, model.var_ids, mutable=True, rule=lambda m, i, j: G[i, j])
    model.jacobian_c = aml.Param(model.con_ids, model.var_ids, mutable=True, rule=lambda m, i, j: A[i, j])
    model.rhs = aml.Param(model.con_ids, mutable=True, rule=lambda m, i: b[i])
    model.grad_f = aml.Param(model.var_ids, mutable=True, rule=lambda m, i: c[i])

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


class TestQP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.model1 = create_dense_qp()
        cls.nlp1 = PyomoNLP(cls.model1)
        x = cls.nlp1.create_vector_x()
        x.fill(0.0)
        y = cls.nlp1.create_vector_y()
        Q = cls.nlp1.hessian_lag(x, y)
        A = cls.nlp1.jacobian_g(x)
        c = cls.nlp1.grad_objective(x)
        b = -cls.nlp1.evaluate_g(x)
        x.fill(1.0)

        cls.qp_model1 = EqualityQuadraticModel(Q, c, 0.0, A, b)
        cls.qp1 = EqualityQP(cls.qp_model1)

    def test_nx(self):

        nlp = self.nlp1
        qp = self.qp1
        self.assertEqual(nlp.nx, qp.nx)

    def test_ng(self):
        nlp = self.nlp1
        qp = self.qp1
        self.assertEqual(nlp.ng, qp.ng)

    def test_nc(self):
        nlp = self.nlp1
        qp = self.qp1
        self.assertEqual(nlp.nc, qp.nc)

    def test_nd(self):
        nlp = self.nlp1
        qp = self.qp1
        self.assertEqual(nlp.nd, qp.nd)

    def test_x_maps(self):
        nlp = self.nlp1
        qp = self.qp1
        self.assertListEqual(nlp._lower_x_mask.tolist(),
                             qp._lower_x_mask.tolist())

        self.assertListEqual(nlp._lower_x_map.tolist(),
                             qp._lower_x_map.tolist())

        self.assertListEqual(nlp._upper_x_mask.tolist(),
                             qp._upper_x_mask.tolist())

        self.assertListEqual(nlp._upper_x_map.tolist(),
                             qp._upper_x_map.tolist())

    def test_g_maps(self):
        nlp = self.nlp1
        qp = self.qp1
        self.assertListEqual(nlp._lower_g_mask.tolist(),
                             qp._lower_g_mask.tolist())

        self.assertListEqual(nlp._lower_g_map.tolist(),
                             qp._lower_g_map.tolist())

        self.assertListEqual(nlp._upper_g_mask.tolist(),
                             qp._upper_g_mask.tolist())

        self.assertListEqual(nlp._upper_g_map.tolist(),
                             qp._upper_g_map.tolist())

    def test_c_maps(self):
        nlp = self.nlp1
        qp = self.qp1
        self.assertTrue(np.allclose(nlp._c_mask, qp._c_mask))
        self.assertTrue(np.allclose(nlp._c_map, qp._c_map))

    def test_d_maps(self):
        nlp = self.nlp1
        qp = self.qp1
        self.assertTrue(np.allclose(nlp._d_mask, qp._d_mask))
        self.assertTrue(np.allclose(nlp._d_map, qp._d_map))

    def test_xl(self):
        qp = self.qp1
        xl = np.full((1, qp.nx), -np.inf)
        self.assertTrue(np.allclose(xl, qp.xl()))
        self.assertTrue(np.allclose(np.zeros(0), qp.xl(condensed=True)))

    def test_xu(self):
        qp = self.qp1
        xu = np.full((1, qp.nx), np.inf)
        self.assertTrue(np.allclose(xu, qp.xu()))
        self.assertTrue(np.allclose(np.zeros(0), qp.xu(condensed=True)))

    def test_gl(self):
        qp = self.qp1
        gl = np.zeros(qp.ng)
        self.assertTrue(np.allclose(gl, qp.gl()))
        self.assertTrue(np.allclose(gl, qp.gl(condensed=True)))

    def test_gu(self):
        qp = self.qp1
        gu = np.zeros(qp.ng)
        self.assertTrue(np.allclose(gu, qp.gu()))
        self.assertTrue(np.allclose(gu, qp.gu(condensed=True)))

    def test_dl(self):
        qp = self.qp1
        dl = np.zeros(qp.nd)
        self.assertTrue(np.allclose(dl, qp.dl()))
        self.assertTrue(np.allclose(dl, qp.dl(condensed=True)))

    def test_du(self):
        qp = self.qp1
        du = np.zeros(qp.nd)
        self.assertTrue(np.allclose(du, qp.du()))
        self.assertTrue(np.allclose(du, qp.du(condensed=True)))

    def test_x_init(self):
        qp = self.qp1
        nlp = self.nlp1
        x = nlp.create_vector_x()
        x1 = qp.create_vector_x()
        self.assertTrue(np.allclose(x1, x))
        x = nlp.create_vector_x(subset='l')
        x1 = qp.create_vector_x(subset='l')
        self.assertTrue(np.allclose(x1, x))
        x = nlp.create_vector_x(subset='u')
        x1 = qp.create_vector_x(subset='u')
        self.assertTrue(np.allclose(x1, x))

    def test_y_init(self):
        qp = self.qp1
        nlp = self.nlp1
        x = nlp.create_vector_y()
        x1 = qp.create_vector_y()
        self.assertTrue(np.allclose(x1, x))
        x = nlp.create_vector_y(subset='dl')
        x1 = qp.create_vector_y(subset='dl')
        self.assertTrue(np.allclose(x1, x))
        x = nlp.create_vector_y(subset='du')
        x1 = qp.create_vector_y(subset='du')
        self.assertTrue(np.allclose(x1, x))
        x = nlp.create_vector_y(subset='c')
        x1 = qp.create_vector_y(subset='c')
        self.assertTrue(np.allclose(x1, x))

    def test_nnz_jacobian_g(self):
        qp = self.qp1
        nlp = self.nlp1
        self.assertEqual(qp.nnz_jacobian_g, nlp.nnz_jacobian_g)

    def test_nnz_jacobian_c(self):
        qp = self.qp1
        nlp = self.nlp1
        self.assertEqual(qp.nnz_jacobian_c, nlp.nnz_jacobian_c)

    def test_nnz_jacobian_d(self):
        qp = self.qp1
        nlp = self.nlp1
        self.assertEqual(qp.nnz_jacobian_d, nlp.nnz_jacobian_d)

    def test_nnz_hessian_lag(self):
        qp = self.qp1
        nlp = self.nlp1
        self.assertEqual(qp.nnz_hessian_lag, nlp.nnz_hessian_lag)

    def test_objective(self):
        qp = self.qp1
        nlp = self.nlp1
        x = nlp.create_vector_x()
        x.fill(1.0)
        self.assertEqual(qp.objective(x), nlp.objective(x))

    def test_grad_objective(self):
        qp = self.qp1
        nlp = self.nlp1
        x = nlp.create_vector_x()
        x.fill(1.0)
        self.assertTrue(np.allclose(qp.grad_objective(x),
                                    nlp.grad_objective(x)))
        res = nlp.create_vector_x()
        qp.grad_objective(x, out=res)
        self.assertTrue(np.allclose(res,
                                    nlp.grad_objective(x)))

    def test_evaluate_g(self):
        qp = self.qp1
        nlp = self.nlp1
        x = nlp.create_vector_x()
        x.fill(1.0)
        self.assertTrue(np.allclose(qp.evaluate_g(x),
                                    nlp.evaluate_g(x)))
        res = nlp.create_vector_y()
        qp.evaluate_g(x, out=res)
        self.assertTrue(np.allclose(res,
                                    nlp.evaluate_g(x)))

    def test_evaluate_c(self):
        qp = self.qp1
        nlp = self.nlp1
        x = nlp.create_vector_x()
        x.fill(1.0)
        self.assertTrue(np.allclose(qp.evaluate_c(x),
                                    nlp.evaluate_c(x)))
        res = nlp.create_vector_y()
        qp.evaluate_c(x, out=res)
        self.assertTrue(np.allclose(res,
                                    nlp.evaluate_c(x)))

    def test_evaluate_d(self):
        qp = self.qp1
        nlp = self.nlp1
        x = nlp.create_vector_x()
        x.fill(1.0)
        self.assertTrue(np.allclose(qp.evaluate_d(x),
                                    nlp.evaluate_d(x)))
        res = nlp.create_vector_y('d')
        qp.evaluate_d(x, out=res)
        self.assertTrue(np.allclose(res,
                                    nlp.evaluate_d(x)))

    def test_jacobian_g(self):
        qp = self.qp1
        nlp = self.nlp1
        x = nlp.create_vector_x()
        jac = qp.jacobian_g(x)
        jac_nlp = nlp.jacobian_g(x)
        self.assertTrue(np.allclose(jac.toarray(),
                                    jac_nlp.toarray()))
        jac = jac * 2
        qp.jacobian_g(x, out=jac)
        self.assertTrue(np.allclose(jac.toarray(),
                                    jac_nlp.toarray()))

    def test_jacobian_c(self):
        qp = self.qp1
        nlp = self.nlp1
        x = nlp.create_vector_x()
        jac = qp.jacobian_c(x)
        jac_nlp = nlp.jacobian_c(x)
        self.assertTrue(np.allclose(jac.toarray(),
                                    jac_nlp.toarray()))
        jac = jac * 2
        qp.jacobian_c(x, out=jac)
        self.assertTrue(np.allclose(jac.toarray(),
                                    jac_nlp.toarray()))

    def test_jacobian_d(self):
        qp = self.qp1
        nlp = self.nlp1
        x = nlp.create_vector_x()
        jac = qp.jacobian_d(x)
        jac_nlp = nlp.jacobian_d(x)
        self.assertTrue(np.allclose(jac.toarray(),
                                    jac_nlp.toarray()))
        jac = jac * 2
        qp.jacobian_d(x, out=jac)
        self.assertTrue(np.allclose(jac.toarray(),
                                    jac_nlp.toarray()))

    def test_hessian_lag(self):
        qp = self.qp1
        nlp = self.nlp1
        x = nlp.create_vector_x()
        y = nlp.create_vector_y()
        hess = qp.hessian_lag(x, y)
        hess_nlp = nlp.hessian_lag(x, y)
        self.assertTrue(np.allclose(hess.toarray(),
                                    hess_nlp.toarray()))

        hess = hess * 2.0
        qp.hessian_lag(x, y, out=hess)
        self.assertTrue(np.allclose(hess.toarray(),
                                    hess_nlp.toarray()))

    def test_expansion_matrix_xl(self):
        qp = self.qp1
        nlp = self.nlp1
        exp1 = qp.expansion_matrix_xl()
        exp2 = nlp.expansion_matrix_xl()
        self.assertTrue(np.allclose(exp1.toarray(),
                                    exp2.toarray()))

    def test_expansion_matrix_xu(self):
        qp = self.qp1
        nlp = self.nlp1
        exp1 = qp.expansion_matrix_xu()
        exp2 = nlp.expansion_matrix_xu()
        self.assertTrue(np.allclose(exp1.toarray(),
                                    exp2.toarray()))

    def test_expansion_matrix_dl(self):
        qp = self.qp1
        nlp = self.nlp1
        exp1 = qp.expansion_matrix_dl()
        exp2 = nlp.expansion_matrix_dl()
        self.assertTrue(np.allclose(exp1.toarray(),
                                    exp2.toarray()))

    def test_expansion_matrix_du(self):
        qp = self.qp1
        nlp = self.nlp1
        exp1 = qp.expansion_matrix_du()
        exp2 = nlp.expansion_matrix_du()
        self.assertTrue(np.allclose(exp1.toarray(),
                                    exp2.toarray()))

    def test_expansion_matrix_d(self):
        qp = self.qp1
        nlp = self.nlp1
        exp1 = qp.expansion_matrix_d()
        exp2 = nlp.expansion_matrix_d()
        self.assertTrue(np.allclose(exp1.toarray(),
                                    exp2.toarray()))

    def test_expansion_matrix_c(self):
        qp = self.qp1
        nlp = self.nlp1
        exp1 = qp.expansion_matrix_c()
        exp2 = nlp.expansion_matrix_c()
        self.assertTrue(np.allclose(exp1.toarray(),
                                    exp2.toarray()))