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

from pyomo.contrib.pynumero import numpy_available, scipy_available
if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

import pyomo.environ as aml
import scipy.sparse as spa
import numpy as np

from pyomo.contrib.pynumero.sparse import (BlockVector,
                                           BlockMatrix,
                                           BlockSymMatrix,
                                           diagonal_matrix)

from pyomo.contrib.pynumero.interfaces import (PyomoNLP,
                                               TwoStageStochasticNLP,
                                               NLPState,
                                               IPNLPState)

import pyomo.contrib.pynumero as pn


def create_basic_model():
    m = aml.ConcreteModel()
    m._name = 'model1'
    m.x = aml.Var([1, 2, 3], initialize=4.0)
    m.c = aml.Constraint(expr=m.x[3] ** 2 + m.x[1] == 25)
    m.d = aml.Constraint(expr=m.x[2] ** 2 + m.x[1] <= 28.0)
    # m.d = aml.Constraint(expr=aml.inequality(-18, m.x[2] ** 2 + m.x[1],  28))
    m.o = aml.Objective(expr=m.x[1] ** 4 - 3 * m.x[1] * m.x[2] ** 3 + m.x[3] ** 2 - 8.0)
    m.x[1].setlb(0.0)
    m.x[2].setlb(0.0)
    m.x[3].setub(10.0)

    return m


@unittest.skipIf(os.name in ['nt', 'dos'], "Do not test on windows")
class TestNLPState(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # test problem 1
        cls.scenario1 = create_basic_model()
        cls.scenario2 = create_basic_model()
        cls.nlp1 = PyomoNLP(cls.scenario1)
        cls.nlp2 = PyomoNLP(cls.scenario2)
        # this problem is not necessarily feasible
        # we use it only to test functionality in NLPState interface
        cls.nlp = TwoStageStochasticNLP({'s1':cls.nlp1, 's2':cls.nlp2},
                                        {'s1':[0, 1], 's2':[0, 1]})

    def test_x(self):
        state = NLPState(self.nlp1)
        init_x = self.nlp1.x_init()
        self.assertTrue(np.allclose(init_x, state.x))

        state2 = NLPState(self.nlp)
        init_x2 = self.nlp.x_init()
        self.assertIsInstance(state2.x, BlockVector)
        self.assertTrue(np.allclose(init_x2.flatten(), state2.x.flatten()))

    def test_yc(self):
        state = NLPState(self.nlp1)
        init_yc = self.nlp1.create_vector_y(subset='c')
        self.assertTrue(np.allclose(init_yc, state.yc))

        state2 = NLPState(self.nlp)
        init_yc2 = self.nlp.create_vector_y(subset='c')
        self.assertIsInstance(state2.yc, BlockVector)
        self.assertTrue(np.allclose(init_yc2.flatten(), state2.yc.flatten()))

    def test_yd(self):
        state = NLPState(self.nlp1)
        init_yd = self.nlp1.create_vector_y(subset='d')
        self.assertTrue(np.allclose(init_yd, state.yd))

        state2 = NLPState(self.nlp)
        init_yd2 = self.nlp.create_vector_y(subset='d')
        self.assertIsInstance(state2.yd, BlockVector)
        self.assertTrue(np.allclose(init_yd2.flatten(), state2.yd.flatten()))

    def test_s(self):
        state = NLPState(self.nlp1, disable_bound_push=True)
        x_init = self.nlp1.x_init()
        init_s = self.nlp1.evaluate_d(x_init)
        self.assertTrue(np.allclose(init_s, state.s))

        state2 = NLPState(self.nlp, disable_bound_push=True)
        x_init = self.nlp.x_init()
        init_s = self.nlp.evaluate_d(x_init)
        self.assertIsInstance(state2.s, BlockVector)
        self.assertTrue(np.allclose(init_s.flatten(), state2.s.flatten()))

    def test_zl(self):
        state = NLPState(self.nlp1)
        init_zl = self.nlp1.create_vector_x(subset='l')
        init_zl.fill(1.0)
        self.assertTrue(np.allclose(init_zl, state.zl))

        state2 = NLPState(self.nlp)
        init_zl = self.nlp.create_vector_x(subset='l')
        init_zl.fill(1.0)
        self.assertIsInstance(state2.zl, BlockVector)
        self.assertTrue(np.allclose(init_zl.flatten(), state2.zl.flatten()))

    def test_zu(self):
        state = NLPState(self.nlp1)
        init_zu = self.nlp1.create_vector_x(subset='u')
        init_zu.fill(1.0)
        self.assertTrue(np.allclose(init_zu, state.zu))

        state2 = NLPState(self.nlp)
        init_zu = self.nlp.create_vector_x(subset='u')
        init_zu.fill(1.0)
        self.assertIsInstance(state2.zu, BlockVector)
        self.assertTrue(np.allclose(init_zu.flatten(), state2.zu.flatten()))

    def test_vl(self):
        state = NLPState(self.nlp1)
        init_vl = self.nlp1.create_vector_s(subset='l')
        init_vl.fill(1.0)
        self.assertTrue(np.allclose(init_vl, state.vl))

        state2 = NLPState(self.nlp)
        init_vl = self.nlp.create_vector_s(subset='l')
        init_vl.fill(1.0)
        self.assertIsInstance(state2.vl, BlockVector)
        self.assertTrue(np.allclose(init_vl.flatten(), state2.vl.flatten()))

    def test_vu(self):
        state = NLPState(self.nlp1)
        init_vu = self.nlp1.create_vector_s(subset='u')
        init_vu.fill(1.0)
        self.assertTrue(np.allclose(init_vu, state.vu))

        state2 = NLPState(self.nlp)
        init_vu = self.nlp.create_vector_s(subset='u')
        init_vu.fill(1.0)
        self.assertIsInstance(state2.vu, BlockVector)
        self.assertTrue(np.allclose(init_vu.flatten(), state2.vu.flatten()))

    def test_objective(self):
        state = NLPState(self.nlp1)
        x_init = self.nlp1.x_init()
        obj = self.nlp1.objective(x_init)
        self.assertAlmostEqual(obj, state.objective())

        state = NLPState(self.nlp)
        x_init = self.nlp.x_init()
        obj = self.nlp.objective(x_init)
        self.assertAlmostEqual(obj, state.objective())

    def test_grad_objective(self):
        state = NLPState(self.nlp1)
        x_init = self.nlp1.x_init()
        grad_obj = self.nlp1.grad_objective(x_init)
        self.assertTrue(np.allclose(grad_obj, state.grad_objective()))

        state = NLPState(self.nlp)
        x_init = self.nlp.x_init()
        grad_obj = self.nlp.grad_objective(x_init)
        self.assertIsInstance(state.grad_objective(), BlockVector)
        self.assertTrue(np.allclose(grad_obj.flatten(), state.grad_objective().flatten()))

    def test_evaluate_g(self):
        state = NLPState(self.nlp1)
        x_init = self.nlp1.x_init()
        g = self.nlp1.evaluate_g(x_init)
        self.assertTrue(np.allclose(g, state.evaluate_g()))

        state = NLPState(self.nlp)
        x_init = self.nlp.x_init()
        g = self.nlp.evaluate_g(x_init)
        self.assertIsInstance(state.evaluate_g(), BlockVector)
        self.assertTrue(np.allclose(g.flatten(), state.evaluate_g().flatten()))

    def test_evaluate_c(self):
        state = NLPState(self.nlp1)
        x_init = self.nlp1.x_init()
        c = self.nlp1.evaluate_c(x_init)
        self.assertTrue(np.allclose(c, state.evaluate_c()))

        state = NLPState(self.nlp)
        x_init = self.nlp.x_init()
        c = self.nlp.evaluate_c(x_init)
        self.assertIsInstance(state.evaluate_c(), BlockVector)
        self.assertTrue(np.allclose(c.flatten(), state.evaluate_c().flatten()))

    def test_evaluate_d(self):
        state = NLPState(self.nlp1)
        x_init = self.nlp1.x_init()
        d = self.nlp1.evaluate_d(x_init)
        self.assertTrue(np.allclose(d, state.evaluate_d()))

        state = NLPState(self.nlp)
        x_init = self.nlp.x_init()
        d = self.nlp.evaluate_d(x_init)
        self.assertIsInstance(state.evaluate_d(), BlockVector)
        self.assertTrue(np.allclose(d.flatten(), state.evaluate_d().flatten()))

    def test_residual_d(self):
        state = NLPState(self.nlp1, disable_bound_push=True)
        self.assertTrue(np.allclose(np.zeros(self.nlp1.nd), state.residual_d()))

        state = NLPState(self.nlp, disable_bound_push=True)
        x_init = self.nlp.x_init()
        zeros = BlockVector([np.zeros(self.nlp1.nd),
                            np.zeros(self.nlp1.nd)])
        self.assertIsInstance(state.residual_d(), BlockVector)
        self.assertTrue(np.allclose(zeros.flatten(), state.residual_d().flatten()))

    def test_jacobian_g(self):
        state = NLPState(self.nlp1)
        x_init = self.nlp1.x_init()
        jac = self.nlp1.jacobian_g(x_init)
        self.assertTrue(np.allclose(jac.toarray(), state.jacobian_g().toarray()))

        state = NLPState(self.nlp)
        x_init = self.nlp.x_init()
        jac = self.nlp.jacobian_g(x_init)
        self.assertIsInstance(state.jacobian_g(), BlockMatrix)
        self.assertTrue(np.allclose(jac.toarray(), state.jacobian_g().toarray()))

    def test_jacobian_c(self):
        state = NLPState(self.nlp1)
        x_init = self.nlp1.x_init()
        jac = self.nlp1.jacobian_c(x_init)
        self.assertTrue(np.allclose(jac.toarray(), state.jacobian_c().toarray()))

        state = NLPState(self.nlp)
        x_init = self.nlp.x_init()
        jac = self.nlp.jacobian_c(x_init)
        self.assertIsInstance(state.jacobian_c(), BlockMatrix)
        self.assertTrue(np.allclose(jac.toarray(), state.jacobian_c().toarray()))

    def test_jacobian_d(self):
        state = NLPState(self.nlp1)
        x_init = self.nlp1.x_init()
        jac = self.nlp1.jacobian_d(x_init)
        self.assertTrue(np.allclose(jac.toarray(), state.jacobian_d().toarray()))

        state = NLPState(self.nlp)
        x_init = self.nlp.x_init()
        jac = self.nlp.jacobian_d(x_init)
        self.assertIsInstance(state.jacobian_d(), BlockMatrix)
        self.assertTrue(np.allclose(jac.toarray(), state.jacobian_d().toarray()))

    def test_hessian_lag(self):
        state = NLPState(self.nlp1)
        x_init = self.nlp1.x_init()
        y_init = self.nlp1.y_init()
        hess = self.nlp1.hessian_lag(x_init, y_init)
        self.assertTrue(np.allclose(hess.toarray(), state.hessian_lag().toarray()))

        state = NLPState(self.nlp)
        x_init = self.nlp.x_init()
        y_init = self.nlp.y_init()
        hess = self.nlp.hessian_lag(x_init, y_init)
        self.assertIsInstance(state.hessian_lag(), BlockMatrix)
        self.assertTrue(np.allclose(hess.toarray(), state.hessian_lag().toarray()))

    def test_slack_xl(self):
        state = NLPState(self.nlp1)
        x_init = self.nlp1.x_init()
        Pl = self.nlp1.expansion_matrix_xl()
        slack = Pl.T.dot(x_init) - self.nlp1.xl(condensed=True)
        self.assertTrue(np.allclose(slack, state.slack_xl()))

        state = NLPState(self.nlp)
        x_init = self.nlp.x_init()
        Pl = self.nlp.expansion_matrix_xl()
        slack = Pl.T.dot(x_init) - self.nlp.xl(condensed=True)
        self.assertTrue(np.allclose(slack, state.slack_xl()))

    def test_slack_xu(self):
        state = NLPState(self.nlp1)
        x_init = self.nlp1.x_init()
        Pu = self.nlp1.expansion_matrix_xu()
        slack = self.nlp1.xu(condensed=True) - Pu.T.dot(x_init)
        self.assertTrue(np.allclose(slack, state.slack_xu()))

        state = NLPState(self.nlp)
        x_init = self.nlp.x_init()
        Pu = self.nlp.expansion_matrix_xu()
        slack = self.nlp.xu(condensed=True) - Pu.T.dot(x_init)
        self.assertTrue(np.allclose(slack, state.slack_xu()))

    def test_slack_sl(self):
        state = NLPState(self.nlp1, disable_bound_push=True)
        x_init = self.nlp1.x_init()
        s_init = self.nlp1.evaluate_d(x_init)
        Pl = self.nlp1.expansion_matrix_dl()
        slack = Pl.T.dot(s_init) - self.nlp1.dl(condensed=True)
        self.assertTrue(np.allclose(slack, state.slack_sl()))

        state = NLPState(self.nlp)
        x_init = self.nlp.x_init()
        s_init = self.nlp.evaluate_d(x_init)
        Pl = self.nlp.expansion_matrix_dl()
        slack = Pl.T.dot(s_init) - self.nlp.dl(condensed=True)
        self.assertTrue(np.allclose(slack, state.slack_sl()))

    def test_slack_su(self):
        state = NLPState(self.nlp1, disable_bound_push=True)
        x_init = self.nlp1.x_init()
        s_init = self.nlp1.evaluate_d(x_init)
        Pu = self.nlp1.expansion_matrix_du()
        slack = self.nlp1.du(condensed=True) - Pu.T.dot(s_init)
        self.assertTrue(np.allclose(slack, state.slack_su()))

        state = NLPState(self.nlp)
        x_init = self.nlp.x_init()
        s_init = self.nlp.evaluate_d(x_init)
        Pu = self.nlp.expansion_matrix_du()
        slack = self.nlp.du(condensed=True) - Pu.T.dot(s_init)
        self.assertTrue(np.allclose(slack, state.slack_su()))

    def test_grad_lag_x(self):
        state = NLPState(self.nlp1)
        x_init = self.nlp1.x_init()
        y_init = self.nlp1.y_init()
        yc = self.nlp1.expansion_matrix_c().T.dot(y_init)
        yd = self.nlp1.expansion_matrix_d().T.dot(y_init)
        zl = self.nlp1.create_vector_x(subset='l')
        zl.fill(1.0)
        zu = self.nlp1.create_vector_x(subset='u')
        zu.fill(1.0)
        nlp = self.nlp1
        grad_lag = nlp.grad_objective(x_init) + \
              nlp.jacobian_c(x_init).T.dot(yc) + \
              nlp.jacobian_d(x_init).T.dot(yd) + \
              nlp.expansion_matrix_xu().dot(zu) - \
              nlp.expansion_matrix_xl().dot(zl)
        self.assertTrue(np.allclose(grad_lag, state.grad_lag_x()))

        state = NLPState(self.nlp)
        x_init = self.nlp.x_init()
        y_init = self.nlp.y_init()
        yc = self.nlp.expansion_matrix_c().T.dot(y_init)
        yd = self.nlp.expansion_matrix_d().T.dot(y_init)
        zl = self.nlp.create_vector_x(subset='l')
        zl.fill(1.0)
        zu = self.nlp.create_vector_x(subset='u')
        zu.fill(1.0)
        nlp = self.nlp
        grad_lag = nlp.grad_objective(x_init) + \
              nlp.jacobian_c(x_init).T.dot(yc) + \
              nlp.jacobian_d(x_init).T.dot(yd) + \
              nlp.expansion_matrix_xu().dot(zu) - \
              nlp.expansion_matrix_xl().dot(zl)
        self.assertTrue(np.allclose(grad_lag.flatten(), state.grad_lag_x().flatten()))

    def test_grad_lag_s(self):
        state = NLPState(self.nlp1)
        x_init = self.nlp1.x_init()
        y_init = self.nlp1.y_init()
        yd = self.nlp1.expansion_matrix_d().T.dot(y_init)
        vl = self.nlp1.create_vector_s(subset='l')
        vl.fill(1.0)
        vu = self.nlp1.create_vector_s(subset='u')
        vu.fill(1.0)
        nlp = self.nlp1
        grad_lag = nlp.expansion_matrix_du().dot(vu) - \
                   nlp.expansion_matrix_dl().dot(vl) - \
                   yd
        self.assertTrue(np.allclose(grad_lag, state.grad_lag_s()))

        state = NLPState(self.nlp)
        x_init = self.nlp.x_init()
        y_init = self.nlp.y_init()
        yd = self.nlp.expansion_matrix_d().T.dot(y_init)
        vl = self.nlp.create_vector_s(subset='l')
        vl.fill(1.0)
        vu = self.nlp.create_vector_s(subset='u')
        vu.fill(1.0)
        nlp = self.nlp
        grad_lag = nlp.expansion_matrix_du().dot(vu) - \
                   nlp.expansion_matrix_dl().dot(vl) - \
                   yd
        self.assertTrue(np.allclose(grad_lag, state.grad_lag_s()))

    def test_grad_lag_x(self):
        state = NLPState(self.nlp1)
        x_init = self.nlp1.x_init()
        y_init = self.nlp1.y_init()
        yc = self.nlp1.expansion_matrix_c().T.dot(y_init)
        yd = self.nlp1.expansion_matrix_d().T.dot(y_init)
        nlp = self.nlp1
        grad_lag = nlp.grad_objective(x_init) + \
              nlp.jacobian_c(x_init).T.dot(yc) + \
              nlp.jacobian_d(x_init).T.dot(yd)
        self.assertTrue(np.allclose(grad_lag, state.grad_lag_bar_x()))

        state = NLPState(self.nlp)
        x_init = self.nlp.x_init()
        y_init = self.nlp.y_init()
        yc = self.nlp.expansion_matrix_c().T.dot(y_init)
        yd = self.nlp.expansion_matrix_d().T.dot(y_init)
        nlp = self.nlp
        grad_lag = nlp.grad_objective(x_init) + \
              nlp.jacobian_c(x_init).T.dot(yc) + \
              nlp.jacobian_d(x_init).T.dot(yd)
        self.assertTrue(np.allclose(grad_lag.flatten(), state.grad_lag_bar_x().flatten()))

    def test_grad_lag_s(self):
        state = NLPState(self.nlp1)
        x_init = self.nlp1.x_init()
        y_init = self.nlp1.y_init()
        yd = self.nlp1.expansion_matrix_d().T.dot(y_init)
        nlp = self.nlp1
        grad_lag = yd
        self.assertTrue(np.allclose(grad_lag, state.grad_lag_bar_s()))

        state = NLPState(self.nlp)
        x_init = self.nlp.x_init()
        y_init = self.nlp.y_init()
        yd = self.nlp.expansion_matrix_d().T.dot(y_init)
        nlp = self.nlp
        grad_lag = yd
        self.assertTrue(np.allclose(grad_lag, state.grad_lag_bar_s()))

    def test_Dx_matrix(self):

        state = NLPState(self.nlp1)
        x_init = self.nlp1.x_init()
        y_init = self.nlp1.y_init()
        zl = self.nlp1.create_vector_x(subset='l')
        zl.fill(1.0)
        zu = self.nlp1.create_vector_x(subset='u')
        zu.fill(1.0)
        nlp = self.nlp1

        Pl = self.nlp1.expansion_matrix_xl()
        Pu = self.nlp1.expansion_matrix_xu()

        slack_l = Pl.T.dot(x_init) - self.nlp1.xl(condensed=True)
        slack_u = self.nlp1.xu(condensed=True) - Pu.T.dot(x_init)

        diff = Pl * np.divide(zl, slack_l) + Pu * np.divide(zu, slack_u)
        Dx = diagonal_matrix(diff)

        self.assertTrue(np.allclose(Dx.toarray(), state.Dx_matrix().toarray()))

        state = NLPState(self.nlp)
        x_init = self.nlp.x_init()
        y_init = self.nlp.y_init()
        zl = self.nlp.create_vector_x(subset='l')
        zl.fill(1.0)
        zu = self.nlp.create_vector_x(subset='u')
        zu.fill(1.0)
        nlp = self.nlp

        Pl = self.nlp.expansion_matrix_xl()
        Pu = self.nlp.expansion_matrix_xu()

        slack_l = Pl.T.dot(x_init) - self.nlp.xl(condensed=True)
        slack_u = self.nlp.xu(condensed=True) - Pu.T.dot(x_init)

        diff = Pl * np.divide(zl, slack_l) + Pu * np.divide(zu, slack_u)
        Dx = BlockSymMatrix(diff.nblocks)
        for bid in range(diff.nblocks):
            Dx[bid, bid] = diagonal_matrix(diff[bid])

        self.assertTrue(np.allclose(Dx.toarray(), state.Dx_matrix().toarray()))

    def test_Ds_matrix(self):

        state = NLPState(self.nlp1)
        x_init = self.nlp1.x_init()
        y_init = self.nlp1.y_init()
        vl = self.nlp1.create_vector_s(subset='l')
        vl.fill(1.0)
        vu = self.nlp1.create_vector_s(subset='u')
        vu.fill(1.0)
        nlp = self.nlp1

        Pl = self.nlp1.expansion_matrix_dl()
        Pu = self.nlp1.expansion_matrix_du()

        s_init = self.nlp1.evaluate_d(x_init)
        slack_l = Pl.T.dot(s_init) - self.nlp1.dl(condensed=True)
        slack_u = self.nlp1.du(condensed=True) - Pu.T.dot(s_init)


        diff = Pl * np.divide(vl, slack_l) + Pu * np.divide(vu, slack_u)
        Ds = diagonal_matrix(diff)

        self.assertTrue(np.allclose(Ds.toarray(), state.Ds_matrix().toarray()))

        state = NLPState(self.nlp)
        x_init = self.nlp.x_init()
        y_init = self.nlp.y_init()
        vl = self.nlp.create_vector_s(subset='l')
        vl.fill(1.0)
        vu = self.nlp.create_vector_s(subset='u')
        vu.fill(1.0)
        nlp = self.nlp

        Pl = self.nlp.expansion_matrix_dl()
        Pu = self.nlp.expansion_matrix_du()

        s_init = self.nlp.evaluate_d(x_init)
        slack_l = Pl.T.dot(s_init) - self.nlp.dl(condensed=True)
        slack_u = self.nlp.du(condensed=True) - Pu.T.dot(s_init)

        diff = Pl * np.divide(vl, slack_l) + Pu * np.divide(vu, slack_u)

        Ds = BlockSymMatrix(diff.nblocks)
        for bid in range(diff.nblocks):
            Ds[bid, bid] = diagonal_matrix(diff[bid])

        self.assertTrue(np.allclose(Ds.toarray(), state.Ds_matrix().toarray()))

    def test_max_alpha_primal(self):

        state = NLPState(self.nlp1)
        x_init = self.nlp1.x_init()
        y_init = self.nlp1.y_init()

        delta_x = np.ones(self.nlp1.nx)
        delta_s = np.zeros(self.nlp1.nd)

        alpha = state.max_alpha_primal(delta_x, delta_s, 1.0)
        self.assertAlmostEqual(alpha, 1.0) # no negative directions. full step

        delta_x = np.ones(self.nlp1.nx)
        delta_s = np.zeros(self.nlp1.nd)
        delta_x[0] = -3.0

        alpha = state.max_alpha_primal(delta_x, delta_s, 1.0)
        self.assertAlmostEqual(alpha, 1.0) # don't reach bound

        delta_x = np.ones(self.nlp1.nx)
        delta_s = np.zeros(self.nlp1.nd)
        delta_x[0] = -8.0

        alpha = state.max_alpha_primal(delta_x, delta_s, 1.0)
        self.assertAlmostEqual(alpha, 0.5) # went to far need to go back half

        delta_x = np.ones(self.nlp1.nx)
        delta_s = np.zeros(self.nlp1.nd)
        delta_x[1] = 8.0

        alpha = state.max_alpha_primal(delta_x, delta_s, 1.0)
        self.assertAlmostEqual(alpha, 0.75) # went to far need to go back a quater

        s_init = self.nlp1.evaluate_d(x_init)
        delta_x = np.zeros(self.nlp1.nx)
        delta_s = np.ones(self.nlp1.nd)

        alpha = state.max_alpha_primal(delta_x, delta_s, 1.0)
        self.assertAlmostEqual(alpha, 1.0) # still have room to go

        delta_x = np.ones(self.nlp1.nx)
        delta_s = np.ones(self.nlp1.nd)
        delta_s[0] = 10.0

        alpha = state.max_alpha_primal(delta_x, delta_s, 1.0)
        self.assertAlmostEqual(alpha, 0.8) # the slack went to far

        delta_x = np.ones(self.nlp1.nx)
        delta_s = np.ones(self.nlp1.nd)
        delta_s[0] = 10.0
        delta_x[1] = 10.0

        alpha = state.max_alpha_primal(delta_x, delta_s, 1.0)
        self.assertAlmostEqual(alpha, 0.6) # the slack went to far but the primal went farther

        state = NLPState(self.nlp)
        x_init = self.nlp.x_init()
        y_init = self.nlp.y_init()

        delta_x = self.nlp.create_vector_x()
        delta_s = self.nlp.create_vector_s()

        alpha = state.max_alpha_primal(delta_x, delta_s, 1.0)
        self.assertAlmostEqual(alpha, 1.0) # no negative directions. full step

        delta_x = self.nlp.create_vector_x()
        delta_s = self.nlp.create_vector_s()
        delta_x[0][0] = -8.0

        alpha = state.max_alpha_primal(delta_x, delta_s, 1.0)
        self.assertAlmostEqual(alpha, 0.5) # went to far need to go back a half

        delta_x = self.nlp.create_vector_x()
        delta_s = self.nlp.create_vector_s()
        delta_x[0][1] = 8.0

        alpha = state.max_alpha_primal(delta_x, delta_s, 1.0)
        self.assertAlmostEqual(alpha, 0.75) # went to far need to go back a quarter

        delta_x = self.nlp.create_vector_x()
        delta_s = self.nlp.create_vector_s()
        delta_s[0][0] = 10.0

        alpha = state.max_alpha_primal(delta_x, delta_s, 1.0)
        self.assertAlmostEqual(alpha, 0.8) # the slack went to far

        delta_x = self.nlp.create_vector_x()
        delta_s = self.nlp.create_vector_s()
        delta_s[1][0] = 10.0
        delta_x[0][1] = 10.0

        alpha = state.max_alpha_primal(delta_x, delta_s, 1.0)
        self.assertAlmostEqual(alpha, 0.6) # the slack went to far but the primal went farther

    def test_max_alpha_dual(self):

        state = NLPState(self.nlp1)
        delta_zl = self.nlp1.create_vector_x(subset='l')
        delta_zu = self.nlp1.create_vector_x(subset='u')
        delta_vl = self.nlp1.create_vector_s(subset='l')
        delta_vu = self.nlp1.create_vector_s(subset='u')

        alpha = state.max_alpha_dual(delta_zl, delta_zu, delta_vl, delta_vu, 1.0)
        self.assertAlmostEqual(alpha, 1.0)

        delta_zl = self.nlp1.create_vector_x(subset='l')
        delta_zu = self.nlp1.create_vector_x(subset='u')
        delta_vl = self.nlp1.create_vector_s(subset='l')
        delta_vu = self.nlp1.create_vector_s(subset='u')
        delta_zl[0] = -2.0

        alpha = state.max_alpha_dual(delta_zl, delta_zu, delta_vl, delta_vu, 1.0)
        self.assertAlmostEqual(alpha, 0.5)

        delta_zl = self.nlp1.create_vector_x(subset='l')
        delta_zu = self.nlp1.create_vector_x(subset='u')
        delta_vl = self.nlp1.create_vector_s(subset='l')
        delta_vu = self.nlp1.create_vector_s(subset='u')
        delta_zl[0] = -2.0
        delta_zu[0] = -4.0

        alpha = state.max_alpha_dual(delta_zl, delta_zu, delta_vl, delta_vu, 1.0)
        self.assertAlmostEqual(alpha, 0.25)

        delta_zl = self.nlp1.create_vector_x(subset='l')
        delta_zu = self.nlp1.create_vector_x(subset='u')
        delta_vl = self.nlp1.create_vector_s(subset='l')
        delta_vu = self.nlp1.create_vector_s(subset='u')
        delta_zl[0] = -2.0
        delta_zu[0] = -4.0
        delta_vu[0] = -10.0

        alpha = state.max_alpha_dual(delta_zl, delta_zu, delta_vl, delta_vu, 1.0)
        self.assertAlmostEqual(alpha, 0.1)

        delta_zl = self.nlp1.create_vector_x(subset='l')
        delta_zu = self.nlp1.create_vector_x(subset='u')
        delta_vl = self.nlp1.create_vector_s(subset='l')
        delta_vu = self.nlp1.create_vector_s(subset='u')
        delta_zl[0] = 2.0
        delta_zu[0] = 4.0
        delta_vu[0] = 10.0

        alpha = state.max_alpha_dual(delta_zl, delta_zu, delta_vl, delta_vu, 1.0)
        self.assertAlmostEqual(alpha, 1.0)

        state = NLPState(self.nlp)
        delta_zl = self.nlp.create_vector_x(subset='l')
        delta_zu = self.nlp.create_vector_x(subset='u')
        delta_vl = self.nlp.create_vector_s(subset='l')
        delta_vu = self.nlp.create_vector_s(subset='u')

        alpha = state.max_alpha_dual(delta_zl, delta_zu, delta_vl, delta_vu, 1.0)
        self.assertAlmostEqual(alpha, 1.0)

        delta_zl = self.nlp.create_vector_x(subset='l')
        delta_zu = self.nlp.create_vector_x(subset='u')
        delta_vl = self.nlp.create_vector_s(subset='l')
        delta_vu = self.nlp.create_vector_s(subset='u')
        delta_zl[0][0] = -2.0

        alpha = state.max_alpha_dual(delta_zl, delta_zu, delta_vl, delta_vu, 1.0)
        self.assertAlmostEqual(alpha, 0.5)

        delta_zl = self.nlp.create_vector_x(subset='l')
        delta_zu = self.nlp.create_vector_x(subset='u')
        delta_vl = self.nlp.create_vector_s(subset='l')
        delta_vu = self.nlp.create_vector_s(subset='u')
        delta_zl[0][0] = -2.0
        delta_zu[0][0] = -4.0

        alpha = state.max_alpha_dual(delta_zl, delta_zu, delta_vl, delta_vu, 1.0)
        self.assertAlmostEqual(alpha, 0.25)

        delta_zl = self.nlp.create_vector_x(subset='l')
        delta_zu = self.nlp.create_vector_x(subset='u')
        delta_vl = self.nlp.create_vector_s(subset='l')
        delta_vu = self.nlp.create_vector_s(subset='u')
        delta_zl[1][0] = -2.0
        delta_zu[1][0] = -4.0
        delta_vu[1][0] = -10.0

        alpha = state.max_alpha_dual(delta_zl, delta_zu, delta_vl, delta_vu, 1.0)
        self.assertAlmostEqual(alpha, 0.1)

        delta_zl = self.nlp.create_vector_x(subset='l')
        delta_zu = self.nlp.create_vector_x(subset='u')
        delta_vl = self.nlp.create_vector_s(subset='l')
        delta_vu = self.nlp.create_vector_s(subset='u')
        delta_zl[0][0] = 2.0
        delta_zu[0][0] = 4.0
        delta_vu[0][0] = 10.0

        alpha = state.max_alpha_dual(delta_zl, delta_zu, delta_vl, delta_vu, 1.0)
        self.assertAlmostEqual(alpha, 1.0)

    def test_primal_infesibility(self):

        state = NLPState(self.nlp1)
        x_init = self.nlp1.x_init()
        nlp = self.nlp1

        c = pn.linalg.norm(nlp.evaluate_c(x_init), ord=np.inf)
        self.assertEqual(state.primal_infeasibility(), c)

    def test_dual_infeasibility(self):
        state = NLPState(self.nlp1)
        x_init = self.nlp1.x_init()
        y_init = self.nlp1.y_init()
        yc = self.nlp1.expansion_matrix_c().T.dot(y_init)
        yd = self.nlp1.expansion_matrix_d().T.dot(y_init)
        zl = self.nlp1.create_vector_x(subset='l')
        zl.fill(1.0)
        zu = self.nlp1.create_vector_x(subset='u')
        zu.fill(1.0)
        nlp = self.nlp1
        grad_lag = nlp.grad_objective(x_init) + \
              nlp.jacobian_c(x_init).T.dot(yc) + \
              nlp.jacobian_d(x_init).T.dot(yd) + \
              nlp.expansion_matrix_xu().dot(zu) - \
              nlp.expansion_matrix_xl().dot(zl)

        n1 = pn.linalg.norm(grad_lag, ord=np.inf)
        n2 = state.dual_infeasibility()
        self.assertAlmostEqual(n1, n2)

        state = NLPState(self.nlp)
        x_init = self.nlp.x_init()
        y_init = self.nlp.y_init()
        yc = self.nlp.expansion_matrix_c().T.dot(y_init)
        yd = self.nlp.expansion_matrix_d().T.dot(y_init)
        zl = self.nlp.create_vector_x(subset='l')
        zl.fill(1.0)
        zu = self.nlp.create_vector_x(subset='u')
        zu.fill(1.0)
        nlp = self.nlp
        grad_lag = nlp.grad_objective(x_init) + \
              nlp.jacobian_c(x_init).T.dot(yc) + \
              nlp.jacobian_d(x_init).T.dot(yd) + \
              nlp.expansion_matrix_xu().dot(zu) - \
              nlp.expansion_matrix_xl().dot(zl)

        n1 = pn.linalg.norm(grad_lag, ord=np.inf)
        n2 = state.dual_infeasibility()
        self.assertAlmostEqual(n1, n2)

    def test_norm_primal_step(self):

        state = NLPState(self.nlp1)
        self.assertEqual(0.0, state.norm_primal_step())

        state = NLPState(self.nlp)
        self.assertEqual(0.0, state.norm_primal_step())
