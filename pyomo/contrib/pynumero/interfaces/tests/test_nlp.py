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
class TestNLP(unittest.TestCase):

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
        c_mask = [True, True, False, False, False]
        self.assertListEqual(self.nlp1._c_mask.tolist(), c_mask)
        d_mask = [False, False, True, True, True]
        self.assertListEqual(self.nlp1._d_mask.tolist(), d_mask)
        c_map = [0, 1]
        self.assertListEqual(self.nlp1._c_map.tolist(), c_map)
        d_map = [2, 3, 4]
        self.assertListEqual(self.nlp1._d_map.tolist(), d_map)
        lower_d_mask = [False, False, False, True, True]
        self.assertListEqual(self.nlp1._lower_d_mask.tolist(), lower_d_mask)
        upper_d_mask = [False, False, True, False, False]
        self.assertListEqual(self.nlp1._upper_d_mask.tolist(), upper_d_mask)
        lower_d_map = [3, 4]
        self.assertListEqual(self.nlp1._lower_d_map.tolist(), lower_d_map)
        upper_d_map = [2]
        self.assertListEqual(self.nlp1._upper_d_map.tolist(), upper_d_map)

    def test_xl(self):
        xl = [-np.inf, 0, 0]
        self.assertListEqual(list(self.nlp1.xl), xl)
        xl = [self.p2.lb] * self.p2.nx
        self.assertListEqual(list(self.nlp2.xl), xl)

    def test_xu(self):
        xu = [np.inf, 100.0, np.inf]
        self.assertListEqual(list(self.nlp1.xu), xu)
        xu = [self.p2.ub] * self.p2.nx
        self.assertListEqual(list(self.nlp2.xu), xu)

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
        irows = [0, 1, 2, 4, 0, 2, 3, 4, 1, 3, 4]
        self.assertListEqual(list(self.nlp1._irows_jac_g), irows)
        irows = list(range(self.p2.nx))
        self.assertListEqual(list(self.nlp2._irows_jac_g), irows)

    def test_jcols_jacobian_g(self):
        jcols = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
        self.assertListEqual(list(self.nlp1._jcols_jac_g), jcols)
        jcols = list(range(self.p2.nx))
        self.assertListEqual(list(self.nlp2._jcols_jac_g), jcols)

    def test_x_init(self):
        x_init = list(range(1, 4))
        self.assertListEqual(list(self.nlp1.x_init), x_init)
        x_init = [self.p2.init] * self.p2.nx
        self.assertListEqual(list(self.nlp2.x_init), x_init)

    def test_y_init(self):
        lam_init = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertListEqual(list(self.nlp1.y_init), lam_init)
        lam_init = [0.0] * self.p2.nx
        self.assertListEqual(list(self.nlp2.y_init), lam_init)

    def test_irows_hessian_lag(self):
        irows = [0, 1]
        self.assertListEqual(list(self.nlp1._irows_hess), irows)
        irows = list(range(self.p2.nx))
        self.assertListEqual(list(self.nlp2._irows_hess), irows)

    def test_jcols_hessian_lag(self):
        jcols = [0, 1]
        self.assertListEqual(list(self.nlp1._jcols_hess), jcols)
        jcols = list(range(self.p2.nx))
        self.assertListEqual(list(self.nlp2._jcols_hess), jcols)

    def test_create_vector_x(self):
        x = np.zeros(3)
        self.assertListEqual(list(x), list(self.nlp1.create_vector_x()))
        x = np.zeros(self.p2.nx)
        self.assertListEqual(list(x), list(self.nlp2.create_vector_x()))

    def test_create_vector_y(self):
        c = np.zeros(5)
        self.assertListEqual(list(c), list(self.nlp1.create_vector_y()))
        c = np.zeros(self.p2.nx)
        self.assertListEqual(list(c), list(self.nlp2.create_vector_y()))

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
        self.assertListEqual(list(self.nlp1.grad_objective(x)), list(df))
        df_ = self.nlp1.create_vector_x()
        self.assertListEqual(list(self.nlp1.grad_objective(x, out=df_)), list(df))

        x = self.nlp2.create_vector_x() + 3.0
        df = np.ones(self.nlp2.nx) * 2.0
        self.assertListEqual(list(self.nlp2.grad_objective(x)), list(df))
        df_ = self.nlp2.create_vector_x()
        self.assertListEqual(list(self.nlp2.grad_objective(x, out=df_)), list(df))

    def test_set_x_init(self):
        nlp = PyomoNLP(create_basic_model())
        n_vars = 3
        new_xinit = np.ones(n_vars)
        nlp.x_init = new_xinit
        self.assertListEqual(list(nlp.x_init), list(new_xinit))
        new_xinit = np.ones(4)
        self.assertRaises(RuntimeError, setattr, nlp, "x_init", new_xinit)

    def test_set_y_init(self):
        nlp = PyomoNLP(create_basic_model())
        n_cons = 5
        new_y_init = np.ones(n_cons)
        nlp.y_init = new_y_init
        self.assertListEqual(list(nlp.y_init), list(new_y_init))
        new_y_init = np.ones(4)
        self.assertRaises(RuntimeError, setattr, nlp, "y_init", new_y_init)

    def test_eval_g(self):
        x = np.ones(self.nlp1.nx)
        res = [-1.0, -0.5, 2, 2, 3]
        self.assertListEqual(list(self.nlp1.evaluate_g(x)), res)
        res_ = self.nlp1.create_vector_y()
        self.assertListEqual(list(self.nlp1.evaluate_g(x, out=res_)), res)

        x = self.nlp2.create_vector_x()
        res = [-1.0] * self.p2.nx
        self.assertListEqual(list(self.nlp2.evaluate_g(x)), res)
        res_ = self.nlp2.create_vector_y()
        self.assertListEqual(list(self.nlp2.evaluate_g(x, out=res_)), res)

    def test_eval_c(self):
        x = np.ones(self.nlp1.nx)
        res = [-1.0, -0.5]
        self.assertListEqual(list(self.nlp1.evaluate_c(x)), res)

        x = self.nlp2.create_vector_x()
        res = [-1.0] * self.p2.nx
        self.assertListEqual(list(self.nlp2.evaluate_c(x)), res)

    def test_eval_d(self):
        x = np.ones(self.nlp1.nx)
        res = [2.0, 2.0, 3.0]
        self.assertListEqual(list(self.nlp1.evaluate_d(x)), res)

        x = self.nlp2.create_vector_x()
        res = []
        self.assertListEqual(list(self.nlp2.evaluate_d(x)), res)

    def test_jacobian_g(self):

        x = self.nlp1.create_vector_x()
        x.fill(1.0)
        jac = self.nlp1.jacobian_g(x)
        self.assertEqual(3, jac.shape[1])
        self.assertEqual(5, jac.shape[0])
        self.assertTrue(spa.isspmatrix_coo(jac))
        values = [2.0, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1]
        self.assertListEqual(values, list(jac.data))

        jac.data.fill(0.0)
        new_jac = self.nlp1.jacobian_g(x, out=jac)
        self.assertListEqual(values, list(new_jac.data))

        # tests rosenbrock jacobian
        x = self.nlp2.create_vector_x() + 1.0
        jac = self.nlp2.jacobian_g(x)
        self.assertEqual(self.p2.nx, jac.shape[1])
        self.assertEqual(self.p2.nx, jac.shape[0])
        self.assertTrue(spa.isspmatrix_coo(jac))
        values = [3.0] * self.p2.nx
        self.assertListEqual(values, list(jac.data))

    def test_jacobian_c(self):
        x = self.nlp1.create_vector_x()
        x.fill(1.0)
        jac = self.nlp1.jacobian_c(x)
        self.assertEqual(3, jac.shape[1])
        self.assertEqual(2, jac.shape[0])
        self.assertTrue(spa.isspmatrix_coo(jac))
        values = [2.0, 1.0, -1.0, -1.0]
        self.assertListEqual(values, list(jac.data))

        jac_g = self.nlp1.jacobian_g(x)
        new_jac = self.nlp1.jacobian_c(x, evaluated_jac_g=jac_g)
        self.assertListEqual(values, list(new_jac.data))

        # tests rosenbrock jacobian
        x = self.nlp2.create_vector_x() + 1.0
        jac = self.nlp2.jacobian_c(x)
        self.assertEqual(self.p2.nx, jac.shape[1])
        self.assertEqual(self.p2.nx, jac.shape[0])
        self.assertTrue(spa.isspmatrix_coo(jac))
        values = [3.0] * self.p2.nx
        self.assertListEqual(values, list(jac.data))

        jac_g = self.nlp2.jacobian_g(x)
        new_jac = self.nlp2.jacobian_c(x, evaluated_jac_g=jac_g)
        self.assertListEqual(values, list(new_jac.data))

    def test_jacobian_d(self):
        x = self.nlp1.create_vector_x()
        x.fill(1.0)
        jac = self.nlp1.jacobian_d(x)
        self.assertEqual(3, jac.shape[1])
        self.assertEqual(3, jac.shape[0])
        self.assertTrue(spa.isspmatrix_coo(jac))
        values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.assertListEqual(values, list(jac.data))

        jac_g = self.nlp1.jacobian_g(x)
        new_jac = self.nlp1.jacobian_d(x, evaluated_jac_g=jac_g)
        self.assertListEqual(values, list(new_jac.data))

        x = self.nlp2.create_vector_x() + 1.0
        jac = self.nlp2.jacobian_d(x)
        self.assertEqual(self.p2.nx, jac.shape[1])
        self.assertEqual(0, jac.shape[0])
        self.assertTrue(spa.isspmatrix_coo(jac))
        values = []
        self.assertListEqual(values, list(jac.data))

    def test_hessian_lag(self):
        x = self.nlp1.create_vector_x()
        l = self.nlp1.create_vector_y()
        l[0] = 1.0
        values = [2.0, 2.0]
        hes = self.nlp1.hessian_lag(x, l)
        self.assertEqual(hes.shape[0], 3)
        self.assertEqual(hes.shape[1], 3)
        self.assertListEqual(values, list(hes.data))

        hes.data.fill(0.0)
        new_hes = self.nlp1.hessian_lag(x, l, other=hes)
        self.assertListEqual(values, list(new_hes.data))

    def test_Jacobian_g(self):
        x = self.nlp1.create_vector_x()
        x.fill(1.0)
        jac = self.nlp1.Jacobian_g(x)
        self.assertEqual(3, jac.shape[1])
        self.assertEqual(5, jac.shape[0])
        self.assertTrue(spa.isspmatrix_coo(jac))
        values = [2.0, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1]
        self.assertListEqual(values, list(jac.data))

        jac = self.nlp1.Jacobian_g(x, constraints=[self.p1.c1])
        self.assertEqual(3, jac.shape[1])
        self.assertEqual(1, jac.shape[0])
        self.assertTrue(spa.isspmatrix_coo(jac))
        values = [2.0, -1]
        self.assertListEqual(values, list(jac.data))

        jac = self.nlp1.Jacobian_g(x, variables=[self.p1.x[1], self.p1.x[2]])
        self.assertEqual(2, jac.shape[1])
        self.assertEqual(5, jac.shape[0])
        self.assertTrue(spa.isspmatrix_coo(jac))
        values = [2.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0]
        self.assertListEqual(values, list(jac.data))

        jac = self.nlp1.Jacobian_g(x, variables=[self.p1.x[1], self.p1.x[2]], constraints=[self.p1.c1])
        self.assertEqual(2, jac.shape[1])
        self.assertEqual(1, jac.shape[0])
        self.assertTrue(spa.isspmatrix_coo(jac))
        values = [2.0, -1.0]
        self.assertListEqual(values, list(jac.data))

    def test_Hessian_lag(self):
        x = self.nlp1.create_vector_x()
        l = self.nlp1.create_vector_y()
        l[0] = 1
        values = [2.0, 2.0]

        hes = self.nlp1.Hessian_lag(x, l)
        self.assertEqual(hes.shape[0], 3)
        self.assertEqual(hes.shape[1], 3)
        self.assertListEqual(values, list(hes.data))

        hes = self.nlp1.Hessian_lag(x, l,
                                    variables_rows=[self.p1.x[1], self.p1.x[2]],
                                    variables_cols=[self.p1.x[1], self.p1.x[2]])
        self.assertEqual(hes.shape[0], 2)
        self.assertEqual(hes.shape[1], 2)
        self.assertListEqual(values, list(hes.data))

        hes = self.nlp1.Hessian_lag(x, l, variables_rows=[self.p1.x[1], self.p1.x[2]])
        self.assertEqual(hes.shape[0], 2)
        self.assertEqual(hes.shape[1], self.nlp1.nx)
        self.assertListEqual(values, list(hes.data))

        hes = self.nlp1.Hessian_lag(x, l, variables_cols=[self.p1.x[2], self.p1.x[3]])
        self.assertEqual(hes.shape[0], self.nlp1.nx)
        self.assertEqual(hes.shape[1], 2)
        self.assertListEqual([2], list(hes.data))

    def test_Grad_objective(self):
        x = self.nlp1.create_vector_x()
        x.fill(1.0)
        sdf = np.array([0, 2], np.double)
        self.assertListEqual(list(sdf), list(self.nlp1.Grad_objective(x, variables=[self.p1.x[1], self.p1.x[2]])))

        sdf = np.array([2, 0], np.double)
        self.assertListEqual(list(sdf), list(self.nlp1.Grad_objective(x, variables=[self.p1.x[2], self.p1.x[3]])))

    def test_Evaluate_g(self):
        x = self.nlp1.create_vector_x()
        res = [-1.0, -0.5, 0.0, 0.0, 0.0]
        self.assertListEqual(list(self.nlp1.Evaluate_g(x)), res)
        x = self.nlp2.create_vector_x()
        res = [-1.0] * self.p2.nx
        self.assertListEqual(list(self.nlp2.Evaluate_g(x)), res)




