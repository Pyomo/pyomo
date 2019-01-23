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

import pyomo.environ as pe
from pyomo.contrib.pynumero.interfaces.nlp import PyomoNLP, AmplNLP
import tempfile

from scipy.sparse import coo_matrix


def create_basic_model():

    m = pe.ConcreteModel()
    m.x = pe.Var([1, 2, 3], domain=pe.Reals)
    for i in range(1, 4):
        m.x[i].value = i
    m.c1 = pe.Constraint(expr=m.x[1] ** 2 - m.x[2] - 1 == 0)
    m.c2 = pe.Constraint(expr=m.x[1] - m.x[3] - 0.5 == 0)
    m.d1 = pe.Constraint(expr=m.x[1] + m.x[2] <= 100.0)
    m.d2 = pe.Constraint(expr=m.x[2] + m.x[3] >= -100.0)
    m.d3 = pe.Constraint(expr=m.x[2] + m.x[3] + m.x[1] >= -500.0)
    m.x[2].setlb(0.0)
    m.x[3].setlb(0.0)
    m.x[2].setub(100.0)
    m.obj = pe.Objective(expr=m.x[2]**2)
    return m


def create_rosenbrock_model(n_vars):

    model = pe.ConcreteModel()
    model.nx = n_vars
    model.lb = -10
    model.ub = 20
    model.init = 2.0
    model.index_vars = range(n_vars)
    model.x = pe.Var(model.index_vars, initialize=model.init, bounds=(model.lb, model.ub))

    def rule_constraint(m, i):
        return m.x[i]**3-1.0 == 0
    model.c = pe.Constraint(model.index_vars, rule=rule_constraint)

    model.obj = pe.Objective(expr=sum((model.x[i]-2)**2 for i in model.index_vars))

    return model


@unittest.skipIf(os.name in ['nt', 'dos'], "Do not test on windows")
class TestPyomoNLP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # test problem 1
        cls.p1 = create_basic_model()
        cls.nlp1 = PyomoNLP(cls.p1)
        cls.p2 = create_rosenbrock_model(10)
        cls.nlp2 = PyomoNLP(cls.p2)

    def test_nx(self):
        self.assertEqual(self.nlp1.nx, 3)
        self.assertEqual(self.nlp2.nx, self.p2.nx)

    def test_ng(self):
        self.assertEqual(self.nlp1.ng, 5)
        self.assertEqual(self.nlp2.ng, self.p2.nx)

    def test_nc(self):
        self.assertEqual(self.nlp1.nc, 2)
        self.assertEqual(self.nlp2.nc, self.p2.nx)

    def test_nd(self):
        self.assertEqual(self.nlp1.nd, 3)
        self.assertEqual(self.nlp2.nd, 0)

    def test_x_maps(self):
        lower_mask = [False, True, True]
        self.assertListEqual(self.nlp1._lower_x_mask.tolist(), lower_mask)
        upper_mask = [False, True, False]
        self.assertListEqual(self.nlp1._upper_x_mask.tolist(), upper_mask)
        lower_map = [1, 2]
        self.assertListEqual(self.nlp1._lower_x_map.tolist(), lower_map)
        upper_map = [1]
        self.assertListEqual(self.nlp1._upper_x_map.tolist(), upper_map)

    def test_g_maps(self):
        lower_mask = np.array([True, True, False, True, True], dtype=bool)
        self.assertTrue(np.allclose(self.nlp1._lower_g_mask, lower_mask))
        lower_map = [0, 1, 3, 4]
        self.assertTrue(np.allclose(self.nlp1._lower_g_map, lower_map))
        upper_mask = np.array([True, True, True, False, False], dtype=bool)
        self.assertTrue(np.allclose(self.nlp1._upper_g_mask, upper_mask))
        upper_map = [0, 1, 2]
        self.assertTrue(np.allclose(self.nlp1._upper_g_map, upper_map))

    def test_c_maps(self):
        c_mask = np.array([True, True, False, False, False], dtype=bool)
        self.assertTrue(np.allclose(self.nlp1._c_mask, c_mask))
        c_map = np.array([0, 1])
        self.assertTrue(np.allclose(self.nlp1._c_map, c_map))

    def test_d_maps(self):

        d_mask = np.array([False, False, True, True, True], dtype=bool)
        self.assertTrue(np.allclose(self.nlp1._d_mask, d_mask))
        d_map = np.array([2, 3, 4])
        self.assertTrue(np.allclose(self.nlp1._d_map, d_map))

        lower_mask = np.array([False, True, True], dtype=bool)
        self.assertTrue(np.allclose(self.nlp1._lower_d_mask, lower_mask))
        lower_map = [1, 2]
        self.assertTrue(np.allclose(self.nlp1._lower_d_map, lower_map))
        upper_mask = np.array([True, False, False], dtype=bool)
        self.assertTrue(np.allclose(self.nlp1._upper_d_mask, upper_mask))
        upper_map = [0]
        self.assertTrue(np.allclose(self.nlp1._upper_d_map, upper_map))

    def test_xl(self):
        xl = np.array([-np.inf, 0, 0])
        self.assertTrue(np.allclose(xl, self.nlp1.xl()))
        xl = np.zeros(2)
        self.assertTrue(np.allclose(xl, self.nlp1.xl(condensed=True)))
        xl = np.array([self.p2.lb] * self.p2.nx)
        self.assertTrue(np.allclose(xl, self.nlp2.xl()))
        self.assertTrue(np.allclose(xl, self.nlp2.xl(condensed=True)))

    def test_xu(self):
        xu = [np.inf, 100.0, np.inf]
        self.assertTrue(np.allclose(xu, self.nlp1.xu()))
        xu = np.array([100.0])
        self.assertTrue(np.allclose(xu, self.nlp1.xu(condensed=True)))
        xu = [self.p2.ub] * self.p2.nx
        self.assertTrue(np.allclose(xu, self.nlp2.xu()))
        self.assertTrue(np.allclose(xu, self.nlp2.xu(condensed=True)))

    def test_gl(self):
        gl = [0.0, 0.0, -np.inf, -100., -500.]
        self.assertTrue(np.allclose(gl, self.nlp1.gl()))
        gl = [0.0, 0.0, -100., -500.]
        self.assertTrue(np.allclose(gl, self.nlp1.gl(condensed=True)))

    def test_gu(self):
        gu = [0.0, 0.0, 100., np.inf, np.inf]
        self.assertTrue(np.allclose(gu, self.nlp1.gu()))
        gu = [0.0, 0.0, 100.]
        self.assertTrue(np.allclose(gu, self.nlp1.gu(condensed=True)))

    def test_dl(self):
        dl = [-np.inf, -100., -500.]
        self.assertTrue(np.allclose(dl, self.nlp1.dl()))
        dl = [-100., -500.]
        self.assertTrue(np.allclose(dl, self.nlp1.dl(condensed=True)))

    def test_du(self):
        du = [100., np.inf, np.inf]
        self.assertTrue(np.allclose(du, self.nlp1.du()))
        du = [100.]
        self.assertTrue(np.allclose(du, self.nlp1.du(condensed=True)))

    def test_x_init(self):
        x_init = np.array(range(1, 4))
        self.assertTrue(np.allclose(self.nlp1.x_init(), x_init))
        x_init = np.array([self.p2.init] * self.p2.nx)
        self.assertTrue(np.allclose(self.nlp2.x_init(), x_init))

    def test_y_init(self):
        y_init = np.zeros(self.nlp1.ng)
        self.assertTrue(np.allclose(self.nlp1.y_init(), y_init))
        y_init = np.zeros(self.p2.nx)
        self.assertTrue(np.allclose(self.nlp2.y_init(), y_init))

    def test_create_vector_x(self):
        x = np.zeros(3)
        self.assertTrue(np.allclose(x, self.nlp1.create_vector_x()))
        x = np.zeros(2)
        self.assertTrue(np.allclose(x, self.nlp1.create_vector_x(subset='l')))
        x = np.zeros(1)
        self.assertTrue(np.allclose(x, self.nlp1.create_vector_x(subset='u')))
        x = np.zeros(self.p2.nx)
        self.assertTrue(np.allclose(x, self.nlp2.create_vector_x()))
        self.assertTrue(np.allclose(x, self.nlp2.create_vector_x(subset='l')))
        self.assertTrue(np.allclose(x, self.nlp2.create_vector_x(subset='u')))

    def test_create_vector_y(self):
        g = np.zeros(5)
        self.assertTrue(np.allclose(g, self.nlp1.create_vector_y()))
        c = np.zeros(2)
        self.assertTrue(np.allclose(c, self.nlp1.create_vector_y(subset='c')))
        d = np.zeros(3)
        self.assertTrue(np.allclose(d, self.nlp1.create_vector_y(subset='d')))
        dl = np.zeros(2)
        self.assertTrue(np.allclose(dl, self.nlp1.create_vector_y(subset='dl')))
        du = np.zeros(1)
        self.assertTrue(np.allclose(du, self.nlp1.create_vector_y(subset='du')))
        g = np.zeros(self.p2.nx)
        self.assertTrue(np.allclose(g, self.nlp2.create_vector_y()))
        c = np.zeros(self.p2.nx)
        self.assertTrue(np.allclose(c, self.nlp2.create_vector_y(subset='c')))
        d = np.zeros(0)
        self.assertTrue(np.allclose(d, self.nlp2.create_vector_y(subset='d')))
        self.assertTrue(np.allclose(d, self.nlp2.create_vector_y(subset='dl')))
        self.assertTrue(np.allclose(d, self.nlp2.create_vector_y(subset='du')))

    def test_create_vector_s(self):
        g = np.zeros(3)
        self.assertTrue(np.allclose(g, self.nlp1.create_vector_s()))
        dl = np.zeros(2)
        self.assertTrue(np.allclose(dl, self.nlp1.create_vector_s(subset='l')))
        du = np.zeros(1)
        self.assertTrue(np.allclose(du, self.nlp1.create_vector_s(subset='u')))
        d = np.zeros(0)
        self.assertTrue(np.allclose(d, self.nlp2.create_vector_s()))
        self.assertTrue(np.allclose(d, self.nlp2.create_vector_s(subset='l')))
        self.assertTrue(np.allclose(d, self.nlp2.create_vector_s(subset='u')))

    def test_nnz_jacobian_g(self):
        self.assertEqual(self.nlp1.nnz_jacobian_g, 11)
        self.assertEqual(self.nlp2.nnz_jacobian_g, self.p2.nx)

    def test_nnz_jacobian_c(self):
        self.assertEqual(self.nlp1.nnz_jacobian_c, 4)
        self.assertEqual(self.nlp2.nnz_jacobian_c, self.p2.nx)

    def test_nnz_jacobian_d(self):
        self.assertEqual(self.nlp1.nnz_jacobian_d, 7)
        self.assertEqual(self.nlp2.nnz_jacobian_d, 0)

    def test_nnz_hessian_lag(self):
        self.assertEqual(self.nlp1.nnz_hessian_lag, 2)
        self.assertEqual(self.nlp2.nnz_hessian_lag, self.p2.nx)

    def test_irows_jacobian_g(self):
        irows = np.array([0, 1, 2, 4, 0, 2, 3, 4, 1, 3, 4])
        self.assertTrue(np.allclose(self.nlp1._irows_jac_g, irows))
        irows = np.arange(self.p2.nx)
        self.assertTrue(np.allclose(self.nlp2._irows_jac_g, irows))

    def test_jcols_jacobian_g(self):
        jcols = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
        self.assertTrue(np.allclose(self.nlp1._jcols_jac_g, jcols))
        jcols = np.arange(self.p2.nx)
        self.assertTrue(np.allclose(self.nlp2._jcols_jac_g, jcols))

    def test_irows_hessian_lag(self):
        irows = np.arange(2)
        self.assertTrue(np.allclose(self.nlp1._irows_hess, irows))
        irows = np.arange(self.p2.nx)
        self.assertTrue(np.allclose(self.nlp2._irows_hess, irows))

    def test_jcols_hessian_lag(self):
        jcols = np.arange(2)
        self.assertTrue(np.allclose(self.nlp1._jcols_hess, jcols))
        jcols = np.arange(self.p2.nx)
        self.assertTrue(np.allclose(self.nlp2._jcols_hess, jcols))

    def test_objective(self):
        x = self.nlp1.create_vector_x()
        x[1] = 5
        self.assertEqual(25.0, self.nlp1.objective(x))
        x = self.nlp2.create_vector_x()
        self.assertEqual(4 * self.nlp2.nx, self.nlp2.objective(x))

    def test_grad_objective(self):
        x = self.nlp1.create_vector_x()
        x[1] = 1
        df = np.zeros(3)
        df[1] = 2
        self.assertTrue(np.allclose(self.nlp1.grad_objective(x), df))
        df_ = self.nlp1.create_vector_x()
        self.nlp1.grad_objective(x, out=df_)
        self.assertTrue(np.allclose(df_, df))

        x = self.nlp2.create_vector_x() + 3.0
        df = np.ones(self.nlp2.nx) * 2.0
        self.assertTrue(np.allclose(self.nlp2.grad_objective(x), df))
        df_ = self.nlp2.create_vector_x()
        self.nlp2.grad_objective(x, out=df_)
        self.assertTrue(np.allclose(df_, df))

        x = self.nlp1.create_vector_x()
        x.fill(1.0)
        sdf = np.array([0, 2], np.double)
        self.assertTrue(np.allclose(sdf, self.nlp1.grad_objective(x, subset_variables=[self.p1.x[1],
                                                                                       self.p1.x[2]])))

        sdf = np.array([2, 0], np.double)
        self.assertTrue(np.allclose(sdf, self.nlp1.grad_objective(x, subset_variables=[self.p1.x[2],
                                                                                       self.p1.x[3]])))

    def test_eval_g(self):
        x = np.ones(self.nlp1.nx)
        res = np.array([-1.0, -0.5, 2, 2, 3])
        self.assertTrue(np.allclose(self.nlp1.evaluate_g(x), res))
        res_ = self.nlp1.create_vector_y()
        self.nlp1.evaluate_g(x, out=res_)
        self.assertTrue(np.allclose(res_, res))

        x = self.nlp2.create_vector_x()
        res = -np.ones(self.p2.nx)
        self.assertTrue(np.allclose(self.nlp2.evaluate_g(x), res))
        res_ = self.nlp2.create_vector_y()
        self.nlp2.evaluate_g(x, out=res_)
        self.assertTrue(np.allclose(res_, res))

        x = self.nlp1.create_vector_x()
        res = np.array([-1.0, 0.0, 0.0])
        subset_g = [self.p1.c1, self.p1.d1, self.p1.d3]
        self.assertTrue(np.allclose(self.nlp1.evaluate_g(x, subset_constraints=subset_g), res))

    def test_eval_c(self):
        x = np.ones(self.nlp1.nx)
        res = np.array([-1.0, -0.5])
        self.assertTrue(np.allclose(self.nlp1.evaluate_c(x), res))
        res_ = self.nlp1.create_vector_y(subset='c')
        self.nlp1.evaluate_c(x, out=res_)
        self.assertTrue(np.allclose(res_, res))

        x = np.ones(self.nlp1.nx)
        res = np.array([-1.0, -0.5])
        g = self.nlp1.evaluate_g(x)
        res_eval = self.nlp1.evaluate_c(x, evaluated_g=g)
        self.assertTrue(np.allclose(res_eval, res))
        res_ = self.nlp1.create_vector_y(subset='c')
        self.nlp1.evaluate_c(x, out=res_, evaluated_g=g)
        self.assertTrue(np.allclose(res_, res))

        x = self.nlp2.create_vector_x()
        res = -np.ones(self.p2.nx)
        self.assertTrue(np.allclose(self.nlp2.evaluate_c(x), res))
        res_ = self.nlp2.create_vector_y(subset='c')
        self.nlp2.evaluate_c(x, out=res_)
        self.assertTrue(np.allclose(res_, res))

    def test_eval_d(self):
        x = np.ones(self.nlp1.nx)
        res = np.array([2.0, 2.0, 3.0])
        self.assertTrue(np.allclose(self.nlp1.evaluate_d(x), res))
        res_ = self.nlp1.create_vector_y(subset='d')
        self.nlp1.evaluate_d(x, out=res_)
        self.assertTrue(np.allclose(res_, res))

        x = np.ones(self.nlp1.nx)
        res = np.array([2.0, 2.0, 3.0])
        g = self.nlp1.evaluate_g(x)
        res_eval = self.nlp1.evaluate_d(x, evaluated_g=g)
        self.assertTrue(np.allclose(res_eval, res))
        res_ = self.nlp1.create_vector_y(subset='d')
        self.nlp1.evaluate_d(x, out=res_, evaluated_g=g)
        self.assertTrue(np.allclose(res_, res))

        x = self.nlp2.create_vector_x()
        res = np.zeros(0)
        self.assertTrue(np.allclose(self.nlp2.evaluate_d(x), res))
        res_ = self.nlp2.create_vector_y(subset='d')
        self.nlp2.evaluate_d(x, out=res_)
        self.assertTrue(np.allclose(res_, res))

    def test_jacobian_g(self):

        x = self.nlp1.create_vector_x()
        x.fill(1.0)
        jac = self.nlp1.jacobian_g(x)
        self.assertEqual(3, jac.shape[1])
        self.assertEqual(5, jac.shape[0])
        self.assertIsInstance(jac, coo_matrix)
        values = np.array([2.0, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1])
        self.assertTrue(np.allclose(values, jac.data))

        jac.data.fill(0.0)
        new_jac = self.nlp1.jacobian_g(x, out=jac)
        self.assertTrue(np.allclose(values, new_jac.data))

        jac = self.nlp1.jacobian_g(x, subset_constraints=[self.p1.c1])
        self.assertEqual(3, jac.shape[1])
        self.assertEqual(1, jac.shape[0])
        self.assertIsInstance(jac, coo_matrix)
        values = np.array([2.0, -1])
        self.assertTrue(np.allclose(values, jac.data))

        jac = self.nlp1.jacobian_g(x, subset_variables=[self.p1.x[1], self.p1.x[2]])
        self.assertEqual(2, jac.shape[1])
        self.assertEqual(5, jac.shape[0])
        self.assertIsInstance(jac, coo_matrix)
        values = np.array([2.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0])
        self.assertTrue(np.allclose(values, jac.data))

        jac = self.nlp1.jacobian_g(x,
                                   subset_variables=[self.p1.x[1], self.p1.x[2]],
                                   subset_constraints=[self.p1.c1])
        self.assertEqual(2, jac.shape[1])
        self.assertEqual(1, jac.shape[0])
        self.assertIsInstance(jac, coo_matrix)
        values = np.array([2.0, -1.0])
        self.assertTrue(np.allclose(values, jac.data))


        # tests rosenbrock jacobian
        x = self.nlp2.create_vector_x() + 1.0
        jac = self.nlp2.jacobian_g(x)
        self.assertEqual(self.p2.nx, jac.shape[1])
        self.assertEqual(self.p2.nx, jac.shape[0])
        self.assertIsInstance(jac, coo_matrix)
        values = 3.0 * np.ones(self.p2.nx)
        self.assertTrue(np.allclose(values, jac.data))

    def test_jacobian_c(self):
        x = self.nlp1.create_vector_x()
        x.fill(1.0)
        jac = self.nlp1.jacobian_c(x)
        self.assertEqual(3, jac.shape[1])
        self.assertEqual(2, jac.shape[0])
        self.assertIsInstance(jac, coo_matrix)
        values = np.array([2.0, 1.0, -1.0, -1.0])
        self.assertTrue(np.allclose(values, jac.data))

        new_jac = self.nlp1.jacobian_c(x)
        new_jac.data.fill(0.0)
        self.nlp1.jacobian_c(x, out=new_jac)
        self.assertTrue(np.allclose(values, new_jac.data))

        jac_g = self.nlp1.jacobian_g(x)
        new_jac = self.nlp1.jacobian_c(x, evaluated_jac_g=jac_g)
        self.assertTrue(np.allclose(values, new_jac.data))

        new_jac = self.nlp1.jacobian_c(x)
        new_jac.data.fill(0.0)
        self.nlp1.jacobian_c(x, out=new_jac, evaluated_jac_g=jac_g)
        self.assertTrue(np.allclose(values, new_jac.data))

        # tests rosenbrock jacobian
        x = self.nlp2.create_vector_x() + 1.0
        jac = self.nlp2.jacobian_c(x)
        self.assertEqual(self.p2.nx, jac.shape[1])
        self.assertEqual(self.p2.nx, jac.shape[0])
        self.assertIsInstance(jac, coo_matrix)
        values = 3.0 * np.ones(self.p2.nx)
        self.assertTrue(np.allclose(values, jac.data))

        new_jac = self.nlp2.jacobian_c(x)
        new_jac.data.fill(0.0)
        self.nlp2.jacobian_c(x, out=new_jac)
        self.assertTrue(np.allclose(values, new_jac.data))

        jac_g = self.nlp2.jacobian_g(x)
        new_jac = self.nlp2.jacobian_c(x, evaluated_jac_g=jac_g)
        self.assertTrue(np.allclose(values, new_jac.data))

        new_jac = self.nlp2.jacobian_c(x)
        new_jac.data.fill(0.0)
        self.nlp2.jacobian_c(x, out=new_jac, evaluated_jac_g=jac_g)
        self.assertTrue(np.allclose(values, new_jac.data))

    def test_jacobian_d(self):
        x = self.nlp1.create_vector_x()
        x.fill(1.0)
        jac = self.nlp1.jacobian_d(x)
        self.assertEqual(3, jac.shape[1])
        self.assertEqual(3, jac.shape[0])
        self.assertIsInstance(jac, coo_matrix)
        values = np.ones(7)
        self.assertTrue(np.allclose(values, jac.data))

        new_jac = self.nlp1.jacobian_d(x)
        new_jac.data.fill(0.0)
        self.nlp1.jacobian_d(x, out=new_jac)
        self.assertTrue(np.allclose(values, new_jac.data))

        jac_g = self.nlp1.jacobian_g(x)
        new_jac = self.nlp1.jacobian_d(x, evaluated_jac_g=jac_g)
        self.assertTrue(np.allclose(values, new_jac.data))

        new_jac = self.nlp1.jacobian_d(x)
        new_jac.data.fill(0.0)
        self.nlp1.jacobian_d(x, out=new_jac, evaluated_jac_g=jac_g)
        self.assertTrue(np.allclose(values, new_jac.data))

        x = self.nlp2.create_vector_x() + 1.0
        jac = self.nlp2.jacobian_d(x)
        self.assertEqual(self.p2.nx, jac.shape[1])
        self.assertEqual(0, jac.shape[0])
        self.assertIsInstance(jac, coo_matrix)
        values = np.zeros(0)
        self.assertTrue(np.allclose(values, jac.data))

        jac_g = self.nlp2.jacobian_g(x)
        new_jac = self.nlp2.jacobian_d(x, evaluated_jac_g=jac_g)
        self.assertTrue(np.allclose(values, new_jac.data))

        new_jac = self.nlp2.jacobian_d(x)
        new_jac.data.fill(0.0)
        self.nlp2.jacobian_d(x, out=new_jac, evaluated_jac_g=jac_g)
        self.assertTrue(np.allclose(values, new_jac.data))

    def test_hessian_lag(self):
        x = self.nlp1.create_vector_x()
        l = self.nlp1.create_vector_y()
        l[0] = 1.0
        values = np.array([2.0, 2.0])
        hes = self.nlp1.hessian_lag(x, l)
        self.assertEqual(hes.shape[0], 3)
        self.assertEqual(hes.shape[1], 3)
        self.assertTrue(np.allclose(values, hes.data))

        # these are QPs so it does not make a difference eval_f_c
        hes = self.nlp1.hessian_lag(x, l, eval_f_c=False)
        self.assertTrue(np.allclose(values, hes.data))

        hes.data.fill(0.0)
        self.nlp1.hessian_lag(x, l, out=hes)
        self.assertTrue(np.allclose(values, hes.data))

        x = self.nlp1.create_vector_x()
        l = self.nlp1.create_vector_y()
        l[0] = 1
        values = np.array([2.0, 2.0])

        hes = self.nlp1.hessian_lag(x, l)
        self.assertEqual(hes.shape[0], 3)
        self.assertEqual(hes.shape[1], 3)
        self.assertTrue(np.allclose(values, hes.data))

        hes = self.nlp1.hessian_lag(x, l,
                                    subset_variables_row=[self.p1.x[1], self.p1.x[2]],
                                    subset_variables_col=[self.p1.x[1], self.p1.x[2]])
        self.assertEqual(hes.shape[0], 2)
        self.assertEqual(hes.shape[1], 2)
        self.assertTrue(np.allclose(values, hes.data))

        hes = self.nlp1.hessian_lag(x, l,
                                    subset_variables_row=[self.p1.x[1], self.p1.x[2]])
        self.assertEqual(hes.shape[0], 2)
        self.assertEqual(hes.shape[1], self.nlp1.nx)
        self.assertTrue(np.allclose(values, hes.data))

        hes = self.nlp1.hessian_lag(x, l,
                                    subset_variables_col=[self.p1.x[2], self.p1.x[3]])
        self.assertEqual(hes.shape[0], self.nlp1.nx)
        self.assertEqual(hes.shape[1], 2)
        self.assertTrue(np.allclose(np.ones(1) * 2.0, hes.data))

        x = self.nlp2.create_vector_x()
        l = self.nlp2.create_vector_y()
        hes = self.nlp2.hessian_lag(x, l)
        values = 2.0 * np.ones(self.p2.nx)
        self.assertEqual(hes.shape[0], self.p2.nx)
        self.assertEqual(hes.shape[1], self.p2.nx)
        self.assertTrue(np.allclose(values, hes.data))

        # these are QPs so it does not make a difference eval_f_c
        hes = self.nlp2.hessian_lag(x, l, eval_f_c=False)
        self.assertTrue(np.allclose(values, hes.data))

        hes.data.fill(0.0)
        self.nlp2.hessian_lag(x, l, out=hes)
        self.assertTrue(np.allclose(values, hes.data))

    def test_expansion_matrix_xl(self):

        xl = self.nlp1.xl(condensed=True)
        Pxl = self.nlp1.expansion_matrix_xl()
        all_xl = Pxl * xl
        xx = np.copy(self.nlp1.xl())
        xx[xx == -np.inf] = 0
        self.assertTrue(np.allclose(all_xl, xx))

    def test_expansion_matrix_xu(self):
        xu = self.nlp1.xu(condensed=True)
        Pxu = self.nlp1.expansion_matrix_xu()
        all_xu = Pxu * xu
        xx = np.copy(self.nlp1.xu())
        xx[xx == np.inf] = 0
        self.assertTrue(np.allclose(all_xu, xx))

    def test_expansion_matrix_dl(self):
        dl = self.nlp1.dl(condensed=True)
        Pdl = self.nlp1.expansion_matrix_dl()
        all_dl = Pdl * dl
        dd = np.copy(self.nlp1.dl())
        dd[dd == -np.inf] = 0
        self.assertTrue(np.allclose(all_dl, dd))

    def test_expansion_matrix_du(self):
        du = self.nlp1.du(condensed=True)
        Pdu = self.nlp1.expansion_matrix_du()
        all_du = Pdu * du
        dd = np.copy(self.nlp1.du())
        dd[dd == np.inf] = 0
        self.assertTrue(np.allclose(all_du, dd))

    def test_expansion_matrix_d(self):

        d = self.nlp1.create_vector_s()
        d.fill(1.0)
        Pd = self.nlp1.expansion_matrix_d()
        g = Pd * d
        cnames = self.nlp1.constraint_order()
        dd = np.zeros(self.nlp1.ng)
        for i in range(self.nlp1.ng):
            if 'd' in cnames[i]:
                dd[i] = 1.0
        self.assertTrue(np.allclose(g, dd))

    def test_expansion_matrix_c(self):

        c = self.nlp1.create_vector_y(subset='c')
        c.fill(1.0)
        Pc = self.nlp1.expansion_matrix_c()
        g = Pc * c
        cnames = self.nlp1.constraint_order()
        cc = np.zeros(self.nlp1.ng)
        for i in range(self.nlp1.ng):
            if 'c' in cnames[i]:
                cc[i] = 1.0
        self.assertTrue(np.allclose(g, cc))

    def test_variable_order(self):
        self.assertEqual(len(self.nlp1.variable_order()), 3)

    def test_constraint_order(self):
        self.assertEqual(len(self.nlp1.constraint_order()), 5)

    def test_variable_idx(self):
        self.assertTrue(0 <= self.nlp1.variable_idx(self.p1.x[1]) <= 3)

    def test_constraint_idx(self):
        self.assertTrue(0 <= self.nlp1.constraint_idx(self.p1.c1) <= 5)

@unittest.skipIf(os.name in ['nt', 'dos'], "Do not test on windows")
class TestAmplNLP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # test problem 1
        cls.p1 = create_basic_model()
        temporal_dir = tempfile.mkdtemp()
        filename = os.path.join(temporal_dir, "pynumero_pyomo")
        cls.p1.write(filename+'.nl', io_options={"symbolic_solver_labels": True})
        cls.nlp = AmplNLP(filename+'.nl',
                          row_filename=filename+'.row',
                          col_filename=filename+'.col')

    def test_constructor(self):

        self.assertEqual(len(self.nlp._name_to_cid), 5)
        self.assertEqual(len(self.nlp._name_to_vid), 3)

    def test_variable_order(self):

        self.assertEqual(len(self.nlp.variable_order()), 3)
