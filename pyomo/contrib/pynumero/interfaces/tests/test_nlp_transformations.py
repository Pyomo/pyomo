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

from pyomo.contrib.pynumero.interfaces.nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_transformations import AdmmNLP
from pyomo.core.plugins.transform.hierarchy import Transformation
import pyomo.environ as aml


class AdmmModel(Transformation):
    """
    Transformation to add augmented lagrangian terms to a model.
    """

    def __init__(self, **kwds):
        kwds['name'] = "aug_lag"
        super(AdmmModel, self).__init__(**kwds)

    def _create_using(self, original_model, **kwds):
        augmented_model = original_model.clone()
        self._apply_to(augmented_model, **kwds)
        return augmented_model

    def _apply_to(self, model, **kwds):

        complicating_vars = kwds.pop('complicating_vars', None)
        z_estimates = kwds.pop('z_estimates', None)
        w_estimates = kwds.pop('w_estimates', None)
        rho = kwds.pop('rho', 1.0)

        if complicating_vars is None:
            raise RuntimeError('need to pass list of complicating variables')

        assert isinstance(complicating_vars, list)

        cloned_vars = []
        original_vars = []
        for v in complicating_vars:
            vid = aml.ComponentUID(v)
            vv = vid.find_component_on(model)
            if v.is_indexed():
                raise RuntimeError('Indexed variables not supported')
            else:
                cloned_vars.append(vv)
                original_vars.append(v)

        nz = len(cloned_vars)
        z_vals = np.zeros(nz)
        if z_estimates is not None:
            assert len(z_estimates) == nz
            z_vals = z_estimates

        w_vals = np.zeros(nz)
        if w_estimates is not None:
            assert len(w_estimates) == nz
            w_vals = w_estimates

        model._z = aml.Param(range(nz), initialize=0.0, mutable=True)
        model._w = aml.Param(range(nz), initialize=0.0, mutable=True)
        for i in range(nz):
            model._z[i].value = z_vals[i]
            model._w[i].value = w_vals[i]

        model._rho = aml.Param(initialize=rho, mutable=True)

        # defines objective
        objectives = model.component_map(aml.Objective, active=True)
        if len(objectives) > 1:
            raise RuntimeError('Multiple objectives not supported')
        obj = list(objectives.values())[0]

        def rule_linkin_exprs(m, i):
            return cloned_vars[i] - m._z[i]
        # store non-anticipativity expression
        model._linking_residuals = aml.Expression(range(nz), rule=rule_linkin_exprs)

        dual_term = 0.0
        penalty_term = 0.0
        for zid in range(nz):
            dual_term += (model._linking_residuals[zid]) * model._w[zid]
            penalty_term += (model._linking_residuals[zid])**2

        # multiplier terms in objective
        model._dual_obj_term = aml.Expression(expr=dual_term)
        # penalty term
        model._penalty_obj_term = aml.Expression(expr=0.5 * model._rho * penalty_term)

        model._aug_obj = aml.Objective(expr=obj.expr +
                                            model._dual_obj_term +
                                            model._penalty_obj_term)

        obj.deactivate()

    def propagate_solution(self, augmented_model, original_model):

        for avar in augmented_model.component_objects(ctype=aml.Var, descend_into=True):
            cuid = aml.ComponentUID(avar)
            original_v = cuid.find_component_on(original_model)
            for k in avar:
                original_v[k].value = aml.value(avar[k])


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


def create_model2():
    model = aml.ConcreteModel()
    model.indices = [i for i in range(1, 6)]
    model.x = aml.Var(model.indices, initialize=2)

    def rule_obj(m):
        expr = (m.x[1] - m.x[2]) ** 2 + (m.x[2] + m.x[3] - 2) ** 2 + (m.x[4] - 1) ** 2 + (m.x[5] - 1) **2
        return expr

    model.obj = aml.Objective(rule=rule_obj)

    model.c1 = aml.Constraint(expr=model.x[1] + 3 * model.x[2] == 0.0)
    model.c2 = aml.Constraint(expr=model.x[3] + model.x[4] - 2 * model.x[5] == 0.0)
    model.c3 = aml.Constraint(expr=model.x[2] - model.x[5] == 0.0)

    return model


def create_basic_model():

    m = aml.ConcreteModel()
    m.x = aml.Var([1, 2, 3], domain=aml.Reals)
    for i in range(1, 4):
        m.x[i].value = i
    m.c1 = aml.Constraint(expr=m.x[1] ** 2 - m.x[2] - 1 == 0)
    m.c2 = aml.Constraint(expr=m.x[1] - m.x[3] - 0.5 == 0)
    m.d1 = aml.Constraint(expr=m.x[1] + m.x[2] <= 100.0)
    m.d2 = aml.Constraint(expr=m.x[2] + m.x[3] >= -100.0)
    m.d3 = aml.Constraint(expr=m.x[2] + m.x[3] + m.x[1] >= -500.0)
    m.x[2].setlb(0.0)
    m.x[3].setlb(0.0)
    m.x[2].setub(100.0)
    m.obj = aml.Objective(expr=m.x[1]**2 + m.x[2]**2 + m.x[3]**2)
    return m


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

        # test problem 2
        cls.model2 = create_model2()
        cls.pyomo_nlp2 = PyomoNLP(cls.model2)
        cls.coupling_vars2 = [cls.pyomo_nlp2.variable_idx(cls.model2.x[1]),
                             cls.pyomo_nlp2.variable_idx(cls.model2.x[3]),
                             cls.pyomo_nlp2.variable_idx(cls.model2.x[5])]
        cls.nlp2 = AdmmNLP(cls.pyomo_nlp2,
                           cls.coupling_vars2,
                           rho=1.0)

        # test problem 3
        cls.model3 = create_basic_model()
        cls.pyomo_nlp3 = PyomoNLP(cls.model3)
        cls.coupling_vars3 = [cls.pyomo_nlp3.variable_idx(cls.model3.x[1])]
        cls.nlp3 = AdmmNLP(cls.pyomo_nlp3,
                           cls.coupling_vars3,
                           rho=1.0)

    def test_nx(self):
        self.assertEqual(self.nlp.nx, 3)
        self.assertEqual(self.nlp2.nx, 5)
        self.assertEqual(self.nlp3.nx, 3)

    def test_ng(self):
        self.assertEqual(self.nlp.ng, 2)
        self.assertEqual(self.nlp2.ng, 3)
        self.assertEqual(self.nlp3.ng, 5)

    def test_nc(self):
        self.assertEqual(self.nlp.nc, 2)
        self.assertEqual(self.nlp2.nc, 3)
        self.assertEqual(self.nlp3.nc, 2)

    def test_nd(self):
        self.assertEqual(self.nlp.nd, 0)
        self.assertEqual(self.nlp2.nd, 0)
        self.assertEqual(self.nlp3.nd, 3)

    def test_nz(self):
        self.assertEqual(self.nlp.nz, 2)
        self.assertEqual(self.nlp2.nz, 3)
        self.assertEqual(self.nlp3.nz, 1)

    def test_nw(self):
        self.assertEqual(self.nlp.nw, 2)
        self.assertEqual(self.nlp2.nw, 3)
        self.assertEqual(self.nlp3.nw, 1)

    def test_x_maps(self):
        lower_mask = [False, False, False]
        self.assertListEqual(self.nlp._lower_x_mask.tolist(), lower_mask)
        upper_mask = [False, False, False]
        self.assertListEqual(self.nlp._upper_x_mask.tolist(), upper_mask)
        self.assertFalse(self.nlp._lower_x_map.tolist())
        self.assertFalse(self.nlp._upper_x_map.tolist())

        lower_mask = [False, True, True]
        self.assertListEqual(self.nlp3._lower_x_mask.tolist(), lower_mask)
        upper_mask = [False, True, False]
        self.assertListEqual(self.nlp3._upper_x_mask.tolist(), upper_mask)
        lower_map = [1, 2]
        self.assertListEqual(self.nlp3._lower_x_map.tolist(), lower_map)
        upper_map = [1]
        self.assertListEqual(self.nlp3._upper_x_map.tolist(), upper_map)

    def test_g_maps(self):
        c_mask = [True, True]
        self.assertListEqual(self.nlp._c_mask.tolist(), c_mask)
        d_mask = [False, False]
        self.assertListEqual(self.nlp._d_mask.tolist(), d_mask)
        c_map = [0, 1]
        self.assertListEqual(self.nlp._c_map.tolist(), c_map)
        self.assertFalse(self.nlp._d_map.tolist())
        lower_g_mask = [True, True]
        self.assertListEqual(self.nlp._lower_g_mask.tolist(), lower_g_mask)
        upper_g_mask = [True, True]
        self.assertListEqual(self.nlp._upper_g_mask.tolist(), upper_g_mask)
        self.assertTrue(self.nlp._lower_g_map.tolist())
        self.assertTrue(self.nlp._upper_g_map.tolist())

        lower_mask = np.array([True, True, False, True, True], dtype=bool)
        self.assertTrue(np.allclose(self.nlp3._lower_g_mask, lower_mask))
        lower_map = [0, 1, 3, 4]
        self.assertTrue(np.allclose(self.nlp3._lower_g_map, lower_map))
        upper_mask = np.array([True, True, True, False, False], dtype=bool)
        self.assertTrue(np.allclose(self.nlp3._upper_g_mask, upper_mask))
        upper_map = [0, 1, 2]
        self.assertTrue(np.allclose(self.nlp3._upper_g_map, upper_map))

    def test_c_maps(self):
        c_mask = np.array([True, True, False, False, False], dtype=bool)
        self.assertTrue(np.allclose(self.nlp3._c_mask, c_mask))
        c_map = np.array([0, 1])
        self.assertTrue(np.allclose(self.nlp3._c_map, c_map))

    def test_d_maps(self):
        d_mask = np.array([False, False, True, True, True], dtype=bool)
        self.assertTrue(np.allclose(self.nlp3._d_mask, d_mask))
        d_map = np.array([2, 3, 4])
        self.assertTrue(np.allclose(self.nlp3._d_map, d_map))

        lower_mask = np.array([False, True, True], dtype=bool)
        self.assertTrue(np.allclose(self.nlp3._lower_d_mask, lower_mask))
        lower_map = [1, 2]
        self.assertTrue(np.allclose(self.nlp3._lower_d_map, lower_map))
        upper_mask = np.array([True, False, False], dtype=bool)
        self.assertTrue(np.allclose(self.nlp3._upper_d_mask, upper_mask))
        upper_map = [0]
        self.assertTrue(np.allclose(self.nlp3._upper_d_map, upper_map))

    def test_xl(self):
        xl = np.array([-np.inf, 0, 0])
        self.assertTrue(np.allclose(xl, self.nlp3.xl()))
        xl = np.zeros(2)
        self.assertTrue(np.allclose(xl, self.nlp3.xl(condensed=True)))

    def test_xu(self):
        xu = [np.inf, 100.0, np.inf]
        self.assertTrue(np.allclose(xu, self.nlp3.xu()))
        xu = np.array([100.0])
        self.assertTrue(np.allclose(xu, self.nlp3.xu(condensed=True)))

    def test_gl(self):
        gl = [0.0, 0.0, -np.inf, -100., -500.]
        self.assertTrue(np.allclose(gl, self.nlp3.gl()))
        gl = [0.0, 0.0, -100., -500.]
        self.assertTrue(np.allclose(gl, self.nlp3.gl(condensed=True)))

    def test_gu(self):
        gu = [0.0, 0.0, 100., np.inf, np.inf]
        self.assertTrue(np.allclose(gu, self.nlp3.gu()))
        gu = [0.0, 0.0, 100.]
        self.assertTrue(np.allclose(gu, self.nlp3.gu(condensed=True)))

    def test_dl(self):
        dl = [-np.inf, -100., -500.]
        self.assertTrue(np.allclose(dl, self.nlp3.dl()))
        dl = [-100., -500.]
        self.assertTrue(np.allclose(dl, self.nlp3.dl(condensed=True)))

    def test_du(self):
        du = [100., np.inf, np.inf]
        self.assertTrue(np.allclose(du, self.nlp3.du()))
        du = [100.]
        self.assertTrue(np.allclose(du, self.nlp3.du(condensed=True)))

    def test_create_vector_x(self):
        self.assertTrue(np.allclose(self.pyomo_nlp.create_vector_x(),
                                    self.nlp.create_vector_x()))
        self.assertTrue(np.allclose(self.pyomo_nlp.create_vector_x(subset='l'),
                                    self.nlp.create_vector_x(subset='l')))
        self.assertTrue(np.allclose(self.pyomo_nlp.create_vector_x(subset='u'),
                                    self.nlp.create_vector_x(subset='u')))

        self.assertTrue(np.allclose(self.pyomo_nlp2.create_vector_x(),
                                    self.nlp2.create_vector_x()))
        self.assertTrue(np.allclose(self.pyomo_nlp2.create_vector_x(subset='l'),
                                    self.nlp2.create_vector_x(subset='l')))
        self.assertTrue(np.allclose(self.pyomo_nlp2.create_vector_x(subset='u'),
                                    self.nlp2.create_vector_x(subset='u')))

        self.assertTrue(np.allclose(self.pyomo_nlp3.create_vector_x(),
                                    self.nlp3.create_vector_x()))
        self.assertTrue(np.allclose(self.pyomo_nlp3.create_vector_x(subset='l'),
                                    self.nlp3.create_vector_x(subset='l')))
        self.assertTrue(np.allclose(self.pyomo_nlp3.create_vector_x(subset='u'),
                                    self.nlp3.create_vector_x(subset='u')))

    def test_create_vector_y(self):
        self.assertTrue(np.allclose(self.pyomo_nlp.create_vector_y(),
                                    self.nlp.create_vector_y()))
        self.assertTrue(np.allclose(self.pyomo_nlp.create_vector_y(subset='c'),
                                    self.nlp.create_vector_y(subset='c')))
        self.assertTrue(np.allclose(self.pyomo_nlp.create_vector_y(subset='d'),
                                    self.nlp.create_vector_y(subset='d')))
        self.assertTrue(np.allclose(self.pyomo_nlp.create_vector_y(subset='dl'),
                                    self.nlp.create_vector_y(subset='dl')))
        self.assertTrue(np.allclose(self.pyomo_nlp.create_vector_y(subset='du'),
                                    self.nlp.create_vector_y(subset='du')))

        self.assertTrue(np.allclose(self.pyomo_nlp3.create_vector_y(),
                                    self.nlp3.create_vector_y()))
        self.assertTrue(np.allclose(self.pyomo_nlp3.create_vector_y(subset='c'),
                                    self.nlp3.create_vector_y(subset='c')))
        self.assertTrue(np.allclose(self.pyomo_nlp3.create_vector_y(subset='d'),
                                    self.nlp3.create_vector_y(subset='d')))
        self.assertTrue(np.allclose(self.pyomo_nlp3.create_vector_y(subset='dl'),
                                    self.nlp3.create_vector_y(subset='dl')))
        self.assertTrue(np.allclose(self.pyomo_nlp3.create_vector_y(subset='du'),
                                    self.nlp3.create_vector_y(subset='du')))

    def test_x_init(self):
        x_init = np.array(range(1, 4))
        self.assertTrue(np.allclose(self.nlp3.x_init(), x_init))
        self.assertTrue(np.allclose(self.nlp.x_init(), self.pyomo_nlp.x_init()))
        self.assertTrue(np.allclose(self.nlp2.x_init(), self.pyomo_nlp2.x_init()))

    def test_y_init(self):
        y_init = np.zeros(self.nlp3.ng)
        self.assertTrue(np.allclose(self.nlp3.y_init(), y_init))
        self.assertTrue(np.allclose(self.nlp2.y_init(), self.pyomo_nlp2.y_init()))
        self.assertTrue(np.allclose(self.nlp3.y_init(), self.pyomo_nlp3.y_init()))

    def test_nnz_jacobian_g(self):
        self.assertEqual(self.nlp.nnz_jacobian_g, self.pyomo_nlp.nnz_jacobian_g)
        self.assertEqual(self.nlp2.nnz_jacobian_g, self.pyomo_nlp2.nnz_jacobian_g)
        self.assertEqual(self.nlp3.nnz_jacobian_g, self.pyomo_nlp3.nnz_jacobian_g)

    def test_nnz_jacobian_c(self):
        self.assertEqual(self.nlp.nnz_jacobian_c, self.pyomo_nlp.nnz_jacobian_c)
        self.assertEqual(self.nlp2.nnz_jacobian_c, self.pyomo_nlp2.nnz_jacobian_c)
        self.assertEqual(self.nlp3.nnz_jacobian_c, self.pyomo_nlp3.nnz_jacobian_c)

    def test_nnz_jacobian_d(self):
        self.assertEqual(self.nlp.nnz_jacobian_d, self.pyomo_nlp.nnz_jacobian_d)
        self.assertEqual(self.nlp2.nnz_jacobian_d, self.pyomo_nlp2.nnz_jacobian_d)
        self.assertEqual(self.nlp3.nnz_jacobian_d, self.pyomo_nlp3.nnz_jacobian_d)

    def test_eval_g(self):
        x = np.ones(self.nlp3.nx)
        res = np.array([-1.0, -0.5, 2, 2, 3])
        self.assertTrue(np.allclose(self.nlp3.evaluate_g(x), res))
        res_ = self.nlp3.create_vector_y()
        self.nlp3.evaluate_g(x, out=res_)
        self.assertTrue(np.allclose(res_, res))

    def test_eval_c(self):
        x = np.ones(self.nlp3.nx)
        res = np.array([-1.0, -0.5])
        self.assertTrue(np.allclose(self.nlp3.evaluate_c(x), res))
        res_ = self.nlp3.create_vector_y(subset='c')
        self.nlp3.evaluate_c(x, out=res_)
        self.assertTrue(np.allclose(res_, res))

        x = np.ones(self.nlp3.nx)
        res = np.array([-1.0, -0.5])
        g = self.nlp3.evaluate_g(x)
        res_eval = self.nlp3.evaluate_c(x, evaluated_g=g)
        self.assertTrue(np.allclose(res_eval, res))
        res_ = self.nlp3.create_vector_y(subset='c')
        self.nlp3.evaluate_c(x, out=res_, evaluated_g=g)
        self.assertTrue(np.allclose(res_, res))

    def test_eval_d(self):
        x = np.ones(self.nlp3.nx)
        res = np.array([2.0, 2.0, 3.0])
        self.assertTrue(np.allclose(self.nlp3.evaluate_d(x), res))
        res_ = self.nlp3.create_vector_y(subset='d')
        self.nlp3.evaluate_d(x, out=res_)
        self.assertTrue(np.allclose(res_, res))

        x = np.ones(self.nlp3.nx)
        res = np.array([2.0, 2.0, 3.0])
        g = self.nlp3.evaluate_g(x)
        res_eval = self.nlp3.evaluate_d(x, evaluated_g=g)
        self.assertTrue(np.allclose(res_eval, res))
        res_ = self.nlp3.create_vector_y(subset='d')
        self.nlp3.evaluate_d(x, out=res_, evaluated_g=g)
        self.assertTrue(np.allclose(res_, res))

    def test_jacobian_g(self):
        nlp = self.nlp
        jac_g = self.model.jacobian_c
        x = nlp.create_vector_x()
        self.assertTrue(np.allclose(nlp.jacobian_g(x).todense(), jac_g))

        x = self.nlp3.create_vector_x()
        x.fill(1.0)
        jac = self.nlp3.jacobian_g(x)
        self.assertEqual(3, jac.shape[1])
        self.assertEqual(5, jac.shape[0])
        values = np.array([2.0, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1])
        self.assertTrue(np.allclose(values, jac.data))

        jac.data.fill(0.0)
        new_jac = self.nlp3.jacobian_g(x, out=jac)
        self.assertTrue(np.allclose(values, new_jac.data))

        jac = self.nlp3.jacobian_g(x, subset_constraints=[self.model3.c1])
        self.assertEqual(3, jac.shape[1])
        self.assertEqual(1, jac.shape[0])
        values = np.array([2.0, -1])
        self.assertTrue(np.allclose(values, jac.data))

        jac = self.nlp3.jacobian_g(x, subset_variables=[self.model3.x[1], self.model3.x[2]])
        self.assertEqual(2, jac.shape[1])
        self.assertEqual(5, jac.shape[0])
        values = np.array([2.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0])
        self.assertTrue(np.allclose(values, jac.data))

        jac = self.nlp3.jacobian_g(x,
                                   subset_variables=[self.model3.x[1], self.model3.x[2]],
                                   subset_constraints=[self.model3.c1])
        self.assertEqual(2, jac.shape[1])
        self.assertEqual(1, jac.shape[0])
        values = np.array([2.0, -1.0])
        self.assertTrue(np.allclose(values, jac.data))

    def test_jacobian_c(self):

        nlp = self.nlp
        jac_c = self.model.jacobian_c
        x = nlp.create_vector_x()
        self.assertTrue(np.allclose(nlp.jacobian_c(x).todense(), jac_c))

        x = self.nlp3.create_vector_x()
        x.fill(1.0)
        jac = self.nlp3.jacobian_c(x)
        self.assertEqual(3, jac.shape[1])
        self.assertEqual(2, jac.shape[0])
        values = np.array([2.0, 1.0, -1.0, -1.0])
        self.assertTrue(np.allclose(values, jac.data))

        new_jac = self.nlp3.jacobian_c(x)
        new_jac.data.fill(0.0)
        self.nlp3.jacobian_c(x, out=new_jac)
        self.assertTrue(np.allclose(values, new_jac.data))

        jac_g = self.nlp3.jacobian_g(x)
        new_jac = self.nlp3.jacobian_c(x, evaluated_jac_g=jac_g)
        self.assertTrue(np.allclose(values, new_jac.data))

        new_jac = self.nlp3.jacobian_c(x)
        new_jac.data.fill(0.0)
        self.nlp3.jacobian_c(x, out=new_jac, evaluated_jac_g=jac_g)
        self.assertTrue(np.allclose(values, new_jac.data))

    def test_jacobian_d(self):

        nlp = self.nlp
        x = nlp.create_vector_x()
        self.assertEqual(nlp.jacobian_d(x).shape, (0, 3))

        x = self.nlp3.create_vector_x()
        x.fill(1.0)
        jac = self.nlp3.jacobian_d(x)
        self.assertEqual(3, jac.shape[1])
        self.assertEqual(3, jac.shape[0])
        values = np.ones(7)
        self.assertTrue(np.allclose(values, jac.data))

        new_jac = self.nlp3.jacobian_d(x)
        new_jac.data.fill(0.0)
        self.nlp3.jacobian_d(x, out=new_jac)
        self.assertTrue(np.allclose(values, new_jac.data))

        jac_g = self.nlp3.jacobian_g(x)
        new_jac = self.nlp3.jacobian_d(x, evaluated_jac_g=jac_g)
        self.assertTrue(np.allclose(values, new_jac.data))

        new_jac = self.nlp3.jacobian_d(x)
        new_jac.data.fill(0.0)
        self.nlp3.jacobian_d(x, out=new_jac, evaluated_jac_g=jac_g)
        self.assertTrue(np.allclose(values, new_jac.data))

    def test_nnz_hessian_lag(self):
        self.assertEqual(self.nlp.nnz_hessian_lag, 9)

        m = self.model2
        transform = AdmmModel()
        aug_model = transform.create_using(m,
                                           complicating_vars=[m.x[1], m.x[3], m.x[5]],
                                           # z_estimates=[1, 2, 3],
                                           # w_estimates=[1, 2, 3],
                                           rho=1.0)
        nl = PyomoNLP(aug_model)
        self.assertEqual(self.nlp2.nnz_hessian_lag, nl.nnz_hessian_lag)

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
        self.assertTrue(np.allclose(nlp.w_estimates(), w_estimates))
        w_estimates = np.array([6.0, 5.0])
        nlp.set_w_estimates(w_estimates)
        self.assertTrue(np.allclose(nlp.w_estimates(), w_estimates))
        self.assertEqual(len(nlp.create_vector_w()), 2)

    def test_z_estimates(self):

        z_estimates = np.array([5.0, 5.0])
        nlp = AdmmNLP(self.pyomo_nlp,
                       self.coupling_vars,
                       rho=2.0,
                       z_estimates=z_estimates)
        self.assertTrue(np.allclose(nlp.z_estimates(), z_estimates))
        z_estimates = np.array([6.0, 5.0])
        nlp.set_z_estimates(z_estimates)
        self.assertTrue(np.allclose(nlp.z_estimates(), z_estimates))
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

        # second nlp
        w_estimates = np.array([1.0, 2.0, 3.0])
        z_estimates = np.array([3.0, 4.0, 5.0])
        nlp = AdmmNLP(self.pyomo_nlp2,
                      self.coupling_vars2,
                      rho=7.0,
                      w_estimates=w_estimates,
                      z_estimates=z_estimates)

        m = self.model2
        transform = AdmmModel()
        aug_model = transform.create_using(m,
                                           complicating_vars=[m.x[1], m.x[3], m.x[5]],
                                           z_estimates=z_estimates,
                                           w_estimates=w_estimates,
                                           rho=7.0)
        nl = PyomoNLP(aug_model)

        x = nlp.create_vector_x()
        y = nlp.create_vector_y()
        hess_lag = nlp.hessian_lag(x, y)
        dense_hess_lag = hess_lag.todense()
        hess_lagp = nl.hessian_lag(x, y)
        dense_hess_lagp = hess_lagp.todense()
        self.assertTrue(np.allclose(dense_hess_lag, dense_hess_lagp))

        # third nlp
        w_estimates = np.array([1.0])
        z_estimates = np.array([3.0])
        nlp = AdmmNLP(self.pyomo_nlp3,
                      self.coupling_vars3,
                      rho=1.0,
                      w_estimates=w_estimates,
                      z_estimates=z_estimates)

        m = self.model3
        transform = AdmmModel()
        aug_model = transform.create_using(m,
                                           complicating_vars=[m.x[1]],
                                           z_estimates=z_estimates,
                                           w_estimates=w_estimates,
                                           rho=1.0)
        nl = PyomoNLP(aug_model)
        x = nlp.create_vector_x()
        y = nlp.create_vector_y()
        hess_lag = nlp.hessian_lag(x, y)
        dense_hess_lag = hess_lag.todense()
        hess_lagp = nl.hessian_lag(x, y)
        dense_hess_lagp = hess_lagp.todense()

        self.assertTrue(np.allclose(dense_hess_lag, dense_hess_lagp))

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

        # second nlp
        w_estimates = np.array([1.0, 2.0, 3.0])
        z_estimates = np.array([3.0, 4.0, 5.0])
        nlp = AdmmNLP(self.pyomo_nlp2,
                      self.coupling_vars2,
                      rho=5.0,
                      w_estimates=w_estimates,
                      z_estimates=z_estimates)

        m = self.model2
        transform = AdmmModel()
        aug_model = transform.create_using(m,
                                           complicating_vars=[m.x[1], m.x[3], m.x[5]],
                                           z_estimates=z_estimates,
                                           w_estimates=w_estimates,
                                           rho=5.0)
        nl = PyomoNLP(aug_model)

        x = nlp.create_vector_x()
        self.assertAlmostEqual(nlp.objective(x), nl.objective((x)))

        # third nlp
        w_estimates = np.array([1.0])
        z_estimates = np.array([3.0])
        nlp = AdmmNLP(self.pyomo_nlp3,
                      self.coupling_vars3,
                      rho=7.0,
                      w_estimates=w_estimates,
                      z_estimates=z_estimates)

        m = self.model3
        transform = AdmmModel()
        aug_model = transform.create_using(m,
                                           complicating_vars=[m.x[1]],
                                           z_estimates=z_estimates,
                                           w_estimates=w_estimates,
                                           rho=7.0)
        nl = PyomoNLP(aug_model)

        x = nlp.create_vector_x()
        self.assertAlmostEqual(nlp.objective(x), nl.objective((x)))

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

        # second nlp
        w_estimates = np.array([1.0, 2.0, 3.0])
        z_estimates = np.array([3.0, 4.0, 5.0])
        nlp = AdmmNLP(self.pyomo_nlp2,
                      self.coupling_vars2,
                      rho=3.0,
                      w_estimates=w_estimates,
                      z_estimates=z_estimates)

        m = self.model2
        transform = AdmmModel()
        aug_model = transform.create_using(m,
                                           complicating_vars=[m.x[1], m.x[3], m.x[5]],
                                           z_estimates=z_estimates,
                                           w_estimates=w_estimates,
                                           rho=3.0)
        nl = PyomoNLP(aug_model)

        x = nlp.create_vector_x()
        self.assertTrue(np.allclose(nlp.grad_objective(x), nl.grad_objective(x)))

        # third nlp
        w_estimates = np.array([1.0])
        z_estimates = np.array([3.0])
        nlp = AdmmNLP(self.pyomo_nlp3,
                      self.coupling_vars3,
                      rho=8.0,
                      w_estimates=w_estimates,
                      z_estimates=z_estimates)

        m = self.model3
        transform = AdmmModel()
        aug_model = transform.create_using(m,
                                           complicating_vars=[m.x[1]],
                                           z_estimates=z_estimates,
                                           w_estimates=w_estimates,
                                           rho=8.0)
        nl = PyomoNLP(aug_model)

        x = nlp.create_vector_x()
        x.fill(1.0)
        self.assertTrue(np.allclose(nlp.grad_objective(x), nl.grad_objective(x)))

