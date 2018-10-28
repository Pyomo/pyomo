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


from pyomo.contrib.pynumero.interfaces.nlp import PyomoNLP, NLP
from pyomo.contrib.pynumero.interfaces.nlp_compositions import TwoStageStochasticNLP
from pyomo.contrib.pynumero.sparse import (BlockVector,
                                           BlockMatrix,
                                           COOSymMatrix,
                                           COOMatrix,
                                           BlockSymMatrix,
                                           IdentityMatrix,
                                           EmptyMatrix)


def create_basic_dense_qp(G, A, b, c, complicated_var_ids):

    nx = G.shape[0]
    nl = A.shape[0]

    model = aml.ConcreteModel()
    model.var_ids = range(nx)
    model.complicated_var_ids = complicated_var_ids
    model.con_ids = range(nl)

    model.x = aml.Var(model.var_ids, initialize=0.0)

    model.x[0].value = 1.0
    model.x[0].setlb(-100.0)
    model.x[0].setub(100.0)

    model.z = aml.Var(model.complicated_var_ids, initialize=0.0)
    model.hessian_f = aml.Param(model.var_ids, model.var_ids, mutable=True, rule=lambda m, i, j: G[i, j])
    model.jacobian_c = aml.Param(model.con_ids, model.var_ids, mutable=True, rule=lambda m, i, j: A[i, j])
    model.rhs = aml.Param(model.con_ids, mutable=True, rule=lambda m, i: b[i])
    model.grad_f = aml.Param(model.var_ids, mutable=True, rule=lambda m, i: c[i])

    def equality_constraint_rule(m, i):
        return sum(m.jacobian_c[i, j] * m.x[j] for j in m.var_ids) == m.rhs[i]
    model.equalities = aml.Constraint(model.con_ids, rule=equality_constraint_rule)

    def fixing_constraints_rule(m, i):
        return m.z[i] == m.x[i]
    model.fixing_constraints = aml.Constraint(model.complicated_var_ids, rule=fixing_constraints_rule)

    def second_stage_cost_rule(m):
        accum = 0.0
        for i in m.var_ids:
            accum += m.x[i] * sum(m.hessian_f[i, j] * m.x[j] for j in m.var_ids)
        accum *= 0.5
        accum += sum(m.x[j] * m.grad_f[j] for j in m.var_ids)
        return accum

    model.FirstStageCost = aml.Expression(expr=0.0)
    model.SecondStageCost = aml.Expression(rule=second_stage_cost_rule)

    model.obj = aml.Objective(expr=model.FirstStageCost + model.SecondStageCost,
                             sense=aml.minimize)

    return model


@unittest.skipIf(os.name in ['nt', 'dos'], "Do not test on windows")
class TestTwoStageStochasticNLP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # Hessian
        cls.G = np.array([[36, 17, 19, 12, 8, 15],
                          [17, 33, 18, 11, 7, 14],
                          [19, 18, 43, 13, 8, 16],
                          [12, 11, 13, 18, 6, 11],
                          [8, 7, 8, 6, 9, 8],
                          [15, 14, 16, 11, 8, 29]])

        # jacobian
        cls.A = np.array([[7, 1, 8, 3, 3, 3],
                          [5, 0, 5, 1, 5, 8],
                          [2, 6, 7, 1, 1, 8],
                          [1, 5, 0, 6, 1, 0]])

        cls.b = np.array([84, 62, 65, 1])
        cls.c = np.array([20, 15, 21, 18, 29, 24])

        cls.complicated_vars_ids = [4, 5]

        cls.scenarios = dict()
        cls.coupling_vars = dict()
        cls.n_scenarios = 3
        for i in range(cls.n_scenarios):
            instance = create_basic_dense_qp(cls.G,
                                             cls.A,
                                             cls.b,
                                             cls.c,
                                             cls.complicated_vars_ids)

            nlp = PyomoNLP(instance)
            scenario_name = "s{}".format(i)
            cls.scenarios[scenario_name] = nlp

            cvars = list()
            for k in cls.complicated_vars_ids:
                cvars.append(nlp.variable_idx(instance.z[k]))
            cls.coupling_vars[scenario_name] = cvars

        cls.nlp = TwoStageStochasticNLP(cls.scenarios, cls.coupling_vars)

    def test_nx(self):
        nz = len(self.complicated_vars_ids)
        nx_i = (self.G.shape[0] + nz)
        nx = self.n_scenarios * nx_i + nz
        self.assertEqual(self.nlp.nx, nx)

    def test_ng(self):
        nz = len(self.complicated_vars_ids)
        ng_i = self.A.shape[0] + nz
        ng_z = nz * self.n_scenarios
        ng = ng_i * self.n_scenarios + ng_z
        self.assertEqual(self.nlp.ng, ng)

    # ToDo: nc, nd, maps and masks

    def test_nblocks(self):
        self.assertEqual(self.n_scenarios, self.nlp.nblocks)

    def test_nz(self):
        nz = len(self.complicated_vars_ids)
        self.assertEqual(self.nlp.nz, nz)

    def test_xl(self):
        xl = BlockVector(self.n_scenarios + 1)
        nz = len(self.complicated_vars_ids)
        nx_i = (self.G.shape[0] + nz)
        for i in range(self.n_scenarios):
            xl[i] = np.array([-np.inf]*nx_i)
            xl[i][0] = -100.0
        xl[self.n_scenarios] = np.array([-np.inf] * nz)
        self.assertIsInstance(self.nlp.xl(), BlockVector)
        xl_flat = xl.flatten()
        self.assertEqual(xl.nblocks, self.nlp.xl().nblocks)
        self.assertTrue(np.allclose(xl.block_sizes(), self.nlp.xl().block_sizes()))
        self.assertListEqual(list(xl_flat), list(self.nlp.xl().flatten()))

    def test_xu(self):
        xu = BlockVector(self.n_scenarios + 1)
        nz = len(self.complicated_vars_ids)
        nx_i = (self.G.shape[0] + nz)
        for i in range(self.n_scenarios):
            xu[i] = np.array([np.inf]*nx_i)
            xu[i][0] = 100.0
        xu[self.n_scenarios] = np.array([np.inf] * nz)
        self.assertIsInstance(self.nlp.xu(), BlockVector)
        xu_flat = xu.flatten()
        self.assertEqual(xu.nblocks, self.nlp.xu().nblocks)
        self.assertTrue(np.allclose(xu.block_sizes(), self.nlp.xu().block_sizes()))
        self.assertListEqual(list(xu_flat), list(self.nlp.xu().flatten()))

    def test_x_init(self):
        x_init = BlockVector(self.n_scenarios + 1)
        nz = len(self.complicated_vars_ids)
        nx_i = (self.G.shape[0] + nz)
        for i in range(self.n_scenarios):
            x_init[i] = np.zeros(nx_i)
            x_init[i][0] = 1.0
        x_init[self.n_scenarios] = np.zeros(nz)
        self.assertIsInstance(self.nlp.x_init(), BlockVector)
        x_init_flat = x_init.flatten()
        self.assertEqual(x_init.nblocks, self.nlp.x_init().nblocks)
        self.assertTrue(np.allclose(x_init.block_sizes(), self.nlp.x_init().block_sizes()))
        self.assertListEqual(list(x_init_flat), list(self.nlp.x_init().flatten()))

    def test_create_vector_x(self):

        x_ = BlockVector(self.n_scenarios + 1)
        nz = len(self.complicated_vars_ids)
        nx_i = (self.G.shape[0] + nz)
        for i in range(self.n_scenarios):
            x_[i] = np.zeros(nx_i)
        x_[self.n_scenarios] = np.zeros(nz)
        self.assertEqual(x_.shape, self.nlp.create_vector_x().shape)
        self.assertEqual(x_.nblocks,
                         self.nlp.create_vector_x().nblocks)
        self.assertTrue(np.allclose(x_.block_sizes(),
                                    self.nlp.create_vector_x().block_sizes()))
        self.assertListEqual(list(x_.flatten()),
                             list(self.nlp.create_vector_x().flatten()))

        # check for subset
        for s in ['l', 'u']:
            xs = self.nlp.create_vector_x(subset=s)
            xs_ = BlockVector(self.n_scenarios + 1)
            for i in range(self.n_scenarios):
                xs_[i] = np.zeros(1)
            xs_[self.n_scenarios] = np.zeros(0)
            self.assertEqual(xs_.shape, xs.shape)
            self.assertEqual(xs_.nblocks, xs.nblocks)
            self.assertTrue(np.allclose(xs_.block_sizes(),
                                        xs.block_sizes()))
            self.assertListEqual(list(xs_.flatten()),
                                 list(xs.flatten()))

    def test_create_vector_y(self):
        nz = len(self.complicated_vars_ids)
        ng_i = self.A.shape[0] + nz

        y_ = BlockVector(2 * self.n_scenarios)
        for i in range(self.n_scenarios):
            y_[i] = np.zeros(ng_i)
            y_[self.n_scenarios + i] = np.zeros(nz)
        y = self.nlp.create_vector_y()

        self.assertEqual(y_.shape, y.shape)
        self.assertEqual(y_.nblocks, y.nblocks)
        self.assertTrue(np.allclose(y_.block_sizes(),
                                    y.block_sizes()))
        self.assertListEqual(list(y_.flatten()),
                             list(y.flatten()))

        # check for equalities
        ys_ = BlockVector(2 * self.n_scenarios)
        for i in range(self.n_scenarios):
            ys_[i] = np.zeros(ng_i)
            ys_[self.n_scenarios + i] = np.zeros(nz)
        ys = self.nlp.create_vector_y(subset='c')
        self.assertEqual(ys_.shape, ys.shape)
        self.assertEqual(ys_.nblocks, ys.nblocks)
        self.assertTrue(np.allclose(ys_.block_sizes(),
                                    ys.block_sizes()))
        self.assertListEqual(list(ys_.flatten()),
                             list(ys.flatten()))

        # check for inequalities
        ys_ = BlockVector(self.n_scenarios)
        for i in range(self.n_scenarios):
            ys_[i] = np.zeros(0)
        ys = self.nlp.create_vector_y(subset='d')
        self.assertEqual(ys_.shape, ys.shape)
        self.assertEqual(ys_.nblocks, ys.nblocks)
        self.assertTrue(np.allclose(ys_.block_sizes(),
                                    ys.block_sizes()))
        self.assertListEqual(list(ys_.flatten()),
                             list(ys.flatten()))

    def test_nlps(self):

        counter = 0
        for name, nlp in self.nlp.nlps():
            counter += 1
            self.assertIsInstance(nlp, NLP)
        self.assertEqual(counter, self.n_scenarios)

    def test_objective(self):

        G = self.G
        c = self.c
        x_ = np.ones(G.shape[1])
        single_obj = 0.5 * x_.transpose().dot(G.dot(x_)) + x_.dot(c)
        obj_ = single_obj * self.n_scenarios
        x = self.nlp.create_vector_x()
        x.fill(1.0)
        obj = self.nlp.objective(x)
        self.assertEqual(obj, obj_)
        obj = self.nlp.objective(x.flatten())
        self.assertEqual(obj, obj_)

    def test_grad_objective(self):

        G = self.G
        c = self.c
        nz = len(self.complicated_vars_ids)
        x_ = np.ones(G.shape[1])
        single_grad = G.dot(x_) + c
        single_grad = np.append(single_grad, np.zeros(nz))
        x = self.nlp.create_vector_x()
        x.fill(1.0)
        grad_obj = self.nlp.grad_objective(x)
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(grad_obj[i], single_grad))
        self.assertTrue(np.allclose(grad_obj[self.n_scenarios],
                                    np.zeros(nz)))

        x.fill(1.0)
        grad_obj = self.nlp.grad_objective(x.flatten())
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(grad_obj[i], single_grad))
        self.assertTrue(np.allclose(grad_obj[self.n_scenarios],
                                    np.zeros(nz)))

    def test_evaluate_g(self):

        nz = len(self.complicated_vars_ids)
        x = self.nlp.create_vector_x()
        x.fill(1.0)
        gi = np.array([-59, -38, -40, 12, 0, 0])
        g = self.nlp.evaluate_g(x)
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(g[i], gi))
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(g[i+self.n_scenarios], np.zeros(nz)))

        g = self.nlp.evaluate_g(x.flatten())
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(g[i], gi))
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(g[i + self.n_scenarios], np.zeros(nz)))

    def test_evaluate_c(self):

        nz = len(self.complicated_vars_ids)
        x = self.nlp.create_vector_x()
        x.fill(1.0)
        ci = np.array([-59, -38, -40, 12, 0, 0])
        c = self.nlp.evaluate_c(x)
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(c[i], ci))
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(c[i+self.n_scenarios], np.zeros(nz)))

        c = self.nlp.evaluate_c(x.flatten())
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(c[i], ci))
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(c[i + self.n_scenarios], np.zeros(nz)))

    def test_jacobian_g(self):

        nz = len(self.complicated_vars_ids)
        nxi = nz + self.G.shape[1]
        ngi = nz + self.A.shape[0]
        Ji = BlockMatrix(2, 2)
        Ji[0, 0] = COOMatrix(self.A)
        B1 = np.zeros((nz, self.A.shape[1]))
        B2 = np.zeros((nz, nz))
        for i, v in enumerate(self.complicated_vars_ids):
            B1[i, v] = -1.0
            B2[i, i] = 1.0
        Ji[1, 0] = COOMatrix(B1)
        Ji[1, 1] = COOMatrix(B2)
        dense_Ji = Ji.todense()

        x = self.nlp.create_vector_x()
        jac_g = self.nlp.jacobian_g(x)

        total_nx = nxi * self.n_scenarios + nz
        total_ng = (ngi + nz) * self.n_scenarios
        self.assertEqual(jac_g.shape, (total_ng, total_nx))

        # check block jacobians
        for i in range(self.n_scenarios):
            jac_gi = jac_g[i, i].todense()
            self.assertTrue(np.allclose(jac_gi, dense_Ji))

        # check coupling jacobians
        Ai_ = BlockMatrix(1, 2)
        Ai_[0, 1] = IdentityMatrix(nz)
        Ai_[0, 0] = EmptyMatrix(nz, self.G.shape[1])
        Ai_ = Ai_.todense()
        Bi_ = -IdentityMatrix(nz).todense()
        for i in range(self.n_scenarios):
            Ai = jac_g[self.n_scenarios + i, i]
            self.assertTrue(np.allclose(Ai.todense(), Ai_))
            Bi = jac_g[self.n_scenarios + i, self.n_scenarios]
            self.assertTrue(np.allclose(Bi.todense(), Bi_))

        # test flattened vector
        jac_g = self.nlp.jacobian_g(x.flatten())

        total_nx = nxi * self.n_scenarios + nz
        total_ng = (ngi + nz) * self.n_scenarios
        self.assertEqual(jac_g.shape, (total_ng, total_nx))

        # check block jacobians
        for i in range(self.n_scenarios):
            jac_gi = jac_g[i, i].todense()
            self.assertTrue(np.allclose(jac_gi, dense_Ji))

        # check coupling jacobians
        Ai_ = BlockMatrix(1, 2)
        Ai_[0, 1] = IdentityMatrix(nz)
        Ai_[0, 0] = EmptyMatrix(nz, self.G.shape[1])
        Ai_ = Ai_.todense()
        Bi_ = -IdentityMatrix(nz).todense()
        for i in range(self.n_scenarios):
            Ai = jac_g[self.n_scenarios + i, i]
            self.assertTrue(np.allclose(Ai.todense(), Ai_))
            Bi = jac_g[self.n_scenarios + i, self.n_scenarios]
            self.assertTrue(np.allclose(Bi.todense(), Bi_))

    def test_jacobian_c(self):

        nz = len(self.complicated_vars_ids)
        nxi = nz + self.G.shape[1]
        ngi = nz + self.A.shape[0]
        Ji = BlockMatrix(2, 2)
        Ji[0, 0] = COOMatrix(self.A)
        B1 = np.zeros((nz, self.A.shape[1]))
        B2 = np.zeros((nz, nz))
        for i, v in enumerate(self.complicated_vars_ids):
            B1[i, v] = -1.0
            B2[i, i] = 1.0
        Ji[1, 0] = COOMatrix(B1)
        Ji[1, 1] = COOMatrix(B2)
        dense_Ji = Ji.todense()

        x = self.nlp.create_vector_x()
        jac_c = self.nlp.jacobian_c(x)

        total_nx = nxi * self.n_scenarios + nz
        total_nc = (ngi + nz) * self.n_scenarios
        self.assertEqual(jac_c.shape, (total_nc, total_nx))

        # check block jacobians
        for i in range(self.n_scenarios):
            jac_ci = jac_c[i, i].todense()
            self.assertTrue(np.allclose(jac_ci, dense_Ji))

        # check coupling jacobians
        Ai_ = BlockMatrix(1, 2)
        Ai_[0, 1] = IdentityMatrix(nz)
        Ai_[0, 0] = EmptyMatrix(nz, self.G.shape[1])
        Ai_ = Ai_.todense()
        Bi_ = -IdentityMatrix(nz).todense()
        for i in range(self.n_scenarios):
            Ai = jac_c[self.n_scenarios + i, i]
            self.assertTrue(np.allclose(Ai.todense(), Ai_))
            Bi = jac_c[self.n_scenarios + i, self.n_scenarios]
            self.assertTrue(np.allclose(Bi.todense(), Bi_))

        # test flattened vector
        jac_g = self.nlp.jacobian_c(x.flatten())

        total_nx = nxi * self.n_scenarios + nz
        total_nc = (ngi + nz) * self.n_scenarios
        self.assertEqual(jac_c.shape, (total_nc, total_nx))

        # check block jacobians
        for i in range(self.n_scenarios):
            jac_ci = jac_c[i, i].todense()
            self.assertTrue(np.allclose(jac_ci, dense_Ji))

        # check coupling jacobians
        Ai_ = BlockMatrix(1, 2)
        Ai_[0, 1] = IdentityMatrix(nz)
        Ai_[0, 0] = EmptyMatrix(nz, self.G.shape[1])
        Ai_ = Ai_.todense()
        Bi_ = -IdentityMatrix(nz).todense()
        for i in range(self.n_scenarios):
            Ai = jac_c[self.n_scenarios + i, i]
            self.assertTrue(np.allclose(Ai.todense(), Ai_))
            Bi = jac_c[self.n_scenarios + i, self.n_scenarios]
            self.assertTrue(np.allclose(Bi.todense(), Bi_))

    def test_hessian(self):

        nz = len(self.complicated_vars_ids)
        Hi = BlockSymMatrix(2)
        Hi[0, 0] = COOSymMatrix(self.G)
        Hi[1, 1] = EmptyMatrix(nz, nz) # this is because of the way the test problem was setup

        Hi = Hi.todense()
        x = self.nlp.create_vector_x()
        y = self.nlp.create_vector_y()
        H = self.nlp.hessian_lag(x, y)
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(H[i, i].todense(), Hi))
        self.assertTrue(np.allclose(H[self.n_scenarios, self.n_scenarios].todense(),
                                    EmptyMatrix(nz, nz).todense()))

        # ToDo: add tests for flattened x and/or flattened y


    # ToDo: add test jacobian_d
    # ToDo: add masks and maps and tests those
    # ToDo: add expansion matrices and test those

