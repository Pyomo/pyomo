#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import itertools
import pyomo.common.unittest as unittest
import pyomo.environ as pyo

from pyomo.contrib.pynumero.dependencies import (
    numpy as np,
    numpy_available,
    scipy,
    scipy_available,
)

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.common.dependencies.scipy import sparse as sps

from pyomo.contrib.pynumero.asl import AmplInterface

if not AmplInterface.available():
    raise unittest.SkipTest("Pynumero needs the ASL extension to run cyipopt tests")

from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import (
    ExternalPyomoModel,
    get_hessian_of_constraint,
)
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
    PyomoNLPWithGreyBoxBlocks,
)
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import CyIpoptNLP

if not pyo.SolverFactory("ipopt").available():
    raise unittest.SkipTest("Need IPOPT to run ExternalPyomoModel tests")


class SimpleModel1(object):
    def make_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=2.0)
        m.y = pyo.Var(initialize=2.0)
        m.residual_eqn = pyo.Constraint(expr=m.x**2 + m.y**2 == 1.0)
        m.external_eqn = pyo.Constraint(expr=m.x * m.y == 0.2)
        # The "external function constraint" exposed by the ExternalPyomoModel
        # will look like: x**2 + 0.04/x**2 - 1 == 0
        return m

    def evaluate_external_variables(self, x):
        # y(x)
        return 0.2 / x

    def evaluate_external_jacobian(self, x):
        # dydx
        return -0.2 / (x**2)

    def evaluate_external_hessian(self, x):
        # d2ydx2
        return 0.4 / (x**3)

    def evaluate_residual(self, x):
        return x**2 + 0.04 / x**2 - 1

    def evaluate_jacobian(self, x):
        return 2 * x - 0.08 / x**3

    def evaluate_hessian(self, x):
        return 2 + 0.24 / x**4


class SimpleModel2(object):
    """
    The purpose of this model is to exercise each term in the computation
    of the d2ydx2 Hessian.
    """

    def make_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=2.0)
        m.y = pyo.Var(initialize=2.0)
        m.residual_eqn = pyo.Constraint(expr=m.x**2 + m.y**2 == 1.0)
        m.external_eqn = pyo.Constraint(expr=(m.x**3) * (m.y**3) == 0.2)
        return m

    def evaluate_external_variables(self, x):
        return 0.2 ** (1 / 3) / x

    def evaluate_external_jacobian(self, x):
        return -(0.2 ** (1 / 3)) / (x**2)

    def evaluate_external_hessian(self, x):
        return 2 * 0.2 ** (1 / 3) / (x**3)

    def evaluate_residual(self, x):
        return x**2 + 0.2 ** (2 / 3) / x**2 - 1

    def evaluate_jacobian(self, x):
        return 2 * x - 2 * 0.2 ** (2 / 3) / x**3

    def evaluate_hessian(self, x):
        return 2 + 6 * 0.2 ** (2 / 3) / x**4


class SimpleModel2by2_1(object):
    """
    The purpose of this model is to test second derivative computation
    when the external model is nonlinear only in x. This exercises
    the first term in the second derivative implicit function theorem.
    """

    def make_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([0, 1], initialize=2.0)
        m.y = pyo.Var([0, 1], initialize=2.0)

        def residual_eqn_rule(m, i):
            # These equations are chosen to exercise every term in the
            # equality hessian calculation, i.e. to have nonlinearities
            # in x, y, and xy.
            if i == 0:
                return m.x[0] ** 2 + m.x[0] * m.y[0] + m.y[0] ** 2 == 1.0
            elif i == 1:
                return m.x[1] ** 2 + m.x[1] * m.y[1] == 2.0

        m.residual_eqn = pyo.Constraint([0, 1], rule=residual_eqn_rule)

        def external_eqn_rule(m, i):
            if i == 0:
                return m.y[0] + m.y[1] + m.x[0] * m.x[1] + m.x[0] ** 2 == 1.0
            elif i == 1:
                return m.y[0] + 2.0 * m.x[0] * m.x[1] + m.x[1] ** 2 == 2.0

        m.external_eqn = pyo.Constraint([0, 1], rule=external_eqn_rule)

        return m

    # The following three methods are evaluation and derivatives of
    # the equality constraints exposed by the ExternalPyomoModel
    def evaluate_residual(self, x):
        f0 = (
            x[0] ** 2
            + 2 * x[0]
            - 2 * x[0] ** 2 * x[1]
            - x[1] ** 2 * x[0]
            + 4
            - 8 * x[0] * x[1]
            - 4 * x[1] ** 2
            + 4 * x[0] ** 2 * x[1] ** 2
            + 4 * x[0] * x[1] ** 3
            + x[1] ** 4
            - 1.0
        )
        f1 = x[1] ** 2 - x[1] + x[0] * x[1] ** 2 + x[1] ** 3 - x[0] ** 2 * x[1] - 2.0
        return (f0, f1)

    def evaluate_jacobian(self, x):
        df0dx0 = (
            2 * x[0]
            + 2
            - 4 * x[0] * x[1]
            - x[1] ** 2
            - 8 * x[1]
            + 8 * x[0] * x[1] ** 2
            + 4 * x[1] ** 3
        )
        df0dx1 = (
            -2 * x[0] ** 2
            - 2 * x[0] * x[1]
            - 8 * x[0]
            - 8 * x[1]
            + 8 * x[0] ** 2 * x[1]
            + 12 * x[0] * x[1] ** 2
            + 4 * x[1] ** 3
        )
        df1dx0 = x[1] ** 2 - 2 * x[0] * x[1]
        df1dx1 = 2 * x[1] - 1 + 2 * x[0] * x[1] - x[0] ** 2 + 3 * x[1] ** 2
        return np.array([[df0dx0, df0dx1], [df1dx0, df1dx1]])

    def evaluate_hessian(self, x):
        df0dx0dx0 = 2 - 4 * x[1] + 8 * x[1] ** 2
        df0dx0dx1 = -4 * x[0] - 2 * x[1] - 8 + 16 * x[0] * x[1] + 12 * x[1] ** 2
        df0dx1dx1 = -2 * x[0] - 8 + 8 * x[0] ** 2 + 24 * x[0] * x[1] + 12 * x[1] ** 2

        df1dx0dx0 = -2 * x[1]
        df1dx0dx1 = 2 * x[1] - 2 * x[0]
        df1dx1dx1 = 2 + 2 * x[0] + 6 * x[1]
        d2f0 = np.array([[df0dx0dx0, df0dx0dx1], [df0dx0dx1, df0dx1dx1]])
        d2f1 = np.array([[df1dx0dx0, df1dx0dx1], [df1dx0dx1, df1dx1dx1]])
        return [d2f0, d2f1]

    # The following three methods are evaluation and derivatives of
    # the external function "hidden by" the ExternalPyomoModel
    def evaluate_external_variables(self, x):
        y0 = 2.0 - 2.0 * x[0] * x[1] - x[1] ** 2
        y1 = 1.0 - y0 - x[0] * x[1] - x[0] ** 2
        return (y0, y1)

    def evaluate_external_jacobian(self, x):
        dy0dx0 = -2.0 * x[1]
        dy0dx1 = -2.0 * x[0] - 2.0 * x[1]
        dy1dx0 = -dy0dx0 - x[1] - 2.0 * x[0]
        dy1dx1 = -dy0dx1 - x[0]
        return np.array([[dy0dx0, dy0dx1], [dy1dx0, dy1dx1]])

    def evaluate_external_hessian(self, x):
        dy0dx0dx0 = 0.0
        dy0dx0dx1 = -2.0
        dy0dx1dx1 = -2.0

        dy1dx0dx0 = -dy0dx0dx0 - 2.0
        dy1dx0dx1 = -dy0dx0dx1 - 1.0
        dy1dx1dx1 = -dy0dx1dx1

        dy0dxdx = np.array([[dy0dx0dx0, dy0dx0dx1], [dy0dx0dx1, dy0dx1dx1]])
        dy1dxdx = np.array([[dy1dx0dx0, dy1dx0dx1], [dy1dx0dx1, dy1dx1dx1]])
        return [dy0dxdx, dy1dxdx]


class Model2by2(object):
    """
    The purpose of this model is to test d2ydx2 Hessian computation when
    transposes result in a nontrivial modification of Hessian/Jacobian
    matrices.
    """

    def make_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([0, 1], initialize=2.0)
        m.y = pyo.Var([0, 1], initialize=2.0)

        m.residual_eqn = pyo.Constraint(
            expr=m.x[0] ** 2 + m.x[1] ** 2 + m.y[0] ** 2 + m.y[1] ** 2 == 1.0
        )

        def external_eqn_rule(m, i):
            if i == 0:
                return (m.x[0] ** 2) * m.y[0] * (m.x[1] ** 0.5) * m.y[1] - 0.1 == 0
            elif i == 1:
                return m.x[0] * m.y[0] * m.x[1] - 0.2 == 0

        m.external_eqn = pyo.Constraint([0, 1], rule=external_eqn_rule)

        return m

    def evaluate_external_variables(self, x):
        y0 = 0.2 / (x[0] * x[1])
        y1 = 0.1 / (x[0] ** 2 * y0 * x[1] ** 0.5)
        return [y0, y1]

    def evaluate_external_jacobian(self, x):
        dy0dx0 = -0.2 / (x[0] ** 2 * x[1])
        dy0dx1 = -0.2 / (x[0] * x[1] ** 2)
        dy1dx0 = -0.5 * x[1] ** 0.5 / x[0] ** 2
        dy1dx1 = 0.25 / (x[0] * x[1] ** 0.5)
        return np.array([[dy0dx0, dy0dx1], [dy1dx0, dy1dx1]])

    def evaluate_external_hessian(self, x):
        d2y0dx0dx0 = 0.4 / (x[0] ** 3 * x[1])
        d2y0dx0dx1 = 0.2 / (x[0] ** 2 * x[1] ** 2)
        d2y0dx1dx1 = 0.4 / (x[0] * x[1] ** 3)

        d2y1dx0dx0 = x[1] ** 0.5 / (x[0] ** 3)
        d2y1dx0dx1 = -0.25 / (x[0] ** 2 * x[1] ** 0.5)
        d2y1dx1dx1 = -0.125 / (x[0] * x[1] ** 1.5)

        d2y0dxdx = np.array([[d2y0dx0dx0, d2y0dx0dx1], [d2y0dx0dx1, d2y0dx1dx1]])
        d2y1dxdx = np.array([[d2y1dx0dx0, d2y1dx0dx1], [d2y1dx0dx1, d2y1dx1dx1]])
        return [d2y0dxdx, d2y1dxdx]

    def calculate_external_multipliers(self, lam, x):
        r"""
        Calculates the multipliers of the external constraints
        from the multipliers of the residual constraints,
        assuming zero dual infeasibility in the coordinates of
        the external variables.
        This is calculated analytically from:

        \nabla_y f^T \lambda_f + \nabla_y g^T \lambda_g = 0

        """
        y = self.evaluate_external_variables(x)
        lg0 = -2 * y[1] * lam[0] / (x[0] ** 2 * x[1] ** 0.5 * y[0])
        lg1 = -(2 * y[0] * lam[0] + x[0] ** 2 * x[1] ** 0.5 * y[1] * lg0) / (
            x[0] * x[1]
        )
        return [lg0, lg1]

    def calculate_full_space_lagrangian_hessians(self, lam, x):
        y = self.evaluate_external_variables(x)
        lam_g = self.calculate_external_multipliers(lam, x)
        d2fdx0dx0 = 2.0
        d2fdx1dx1 = 2.0
        d2fdy0dy0 = 2.0
        d2fdy1dy1 = 2.0
        hfxx = np.array([[d2fdx0dx0, 0], [0, d2fdx1dx1]])
        hfxy = np.array([[0, 0], [0, 0]])
        hfyy = np.array([[d2fdy0dy0, 0], [0, d2fdy1dy1]])

        dg0dx0dx0 = 2 * y[0] * x[1] ** 0.5 * y[1]
        dg0dx0dx1 = x[0] * y[0] * y[1] / x[1] ** 0.5
        dg0dx1dx1 = -1 / 4 * x[0] ** 2 * y[0] * y[1] / x[1] ** (3 / 2)
        dg0dx0dy0 = 2 * x[0] * x[1] ** 0.5 * y[1]
        dg0dx0dy1 = 2 * x[0] * y[0] * x[1] ** 0.5
        dg0dx1dy0 = 0.5 * x[0] ** 2 * y[1] / x[1] ** 0.5
        dg0dx1dy1 = 0.5 * x[0] ** 2 * y[0] / x[1] ** 0.5
        dg0dy0dy1 = x[0] ** 2 * x[1] ** 0.5
        hg0xx = np.array([[dg0dx0dx0, dg0dx0dx1], [dg0dx0dx1, dg0dx1dx1]])
        hg0xy = np.array([[dg0dx0dy0, dg0dx0dy1], [dg0dx1dy0, dg0dx1dy1]])
        hg0yy = np.array([[0, dg0dy0dy1], [dg0dy0dy1, 0]])

        dg1dx0dx1 = y[0]
        dg1dx0dy0 = x[1]
        dg1dx1dy0 = x[0]
        hg1xx = np.array([[0, dg1dx0dx1], [dg1dx0dx1, 0]])
        hg1xy = np.array([[dg1dx0dy0, 0], [dg1dx1dy0, 0]])
        hg1yy = np.zeros((2, 2))

        hlxx = lam[0] * hfxx + lam_g[0] * hg0xx + lam_g[1] * hg1xx
        hlxy = lam[0] * hfxy + lam_g[0] * hg0xy + lam_g[1] * hg1xy
        hlyy = lam[0] * hfyy + lam_g[0] * hg0yy + lam_g[1] * hg1yy
        return hlxx, hlxy, hlyy

    def calculate_reduced_lagrangian_hessian(self, lam, x):
        dydx = self.evaluate_external_jacobian(x)
        hlxx, hlxy, hlyy = self.calculate_full_space_lagrangian_hessians(lam, x)
        return (
            hlxx
            + (hlxy.dot(dydx)).transpose()
            + hlxy.dot(dydx)
            + dydx.transpose().dot(hlyy).dot(dydx)
        )


class TestGetHessianOfConstraint(unittest.TestCase):
    def test_simple_model_1(self):
        model = SimpleModel1()
        m = model.make_model()
        m.x.set_value(2.0)
        m.y.set_value(2.0)

        con = m.residual_eqn
        expected_hess = np.array([[2.0, 0.0], [0.0, 2.0]])
        hess = get_hessian_of_constraint(con)
        self.assertTrue(np.all(expected_hess == hess.toarray()))

        expected_hess = np.array([[2.0]])
        hess = get_hessian_of_constraint(con, [m.x])
        self.assertTrue(np.all(expected_hess == hess.toarray()))

        con = m.external_eqn
        expected_hess = np.array([[0.0, 1.0], [1.0, 0.0]])
        hess = get_hessian_of_constraint(con)
        self.assertTrue(np.all(expected_hess == hess.toarray()))

    def test_polynomial(self):
        m = pyo.ConcreteModel()

        n_x = 3
        x1 = 1.1
        x2 = 1.2
        x3 = 1.3
        m.x = pyo.Var(range(1, n_x + 1), initialize={1: x1, 2: x2, 3: x3})
        m.eqn = pyo.Constraint(
            expr=5 * (m.x[1] ** 5)  # T1
            + 5 * (m.x[1] ** 4) * (m.x[2])  # T2
            + 5 * (m.x[1] ** 3) * (m.x[2]) * (m.x[3])  # T3
            + 5 * (m.x[1]) * (m.x[2] ** 2) * (m.x[3] ** 2)  # T4
            + 4 * (m.x[1] ** 2) * (m.x[2]) * (m.x[3])  # T5
            + 4 * (m.x[2] ** 2) * (m.x[3] ** 2)  # T6
            + 4 * (m.x[3] ** 4)  # T7
            + 3 * (m.x[1]) * (m.x[2]) * (m.x[3])  # T8
            + 3 * (m.x[2] ** 3)  # T9
            + 3 * (m.x[2] ** 2) * (m.x[3])  # T10
            + 2 * (m.x[1]) * (m.x[2])  # T11
            + 2 * (m.x[2]) * (m.x[3])  # T12
            == 0
        )

        rcd = []
        rcd.append(
            (
                0,
                0,
                (
                    # wrt x1, x1
                    5 * 5 * 4 * x1**3  # T1
                    + 5 * 4 * 3 * x1**2 * x2  # T2
                    + 5 * 3 * 2 * x1 * x2 * x3  # T3
                    + 4 * 2 * 1 * x2 * x3  # T5
                ),
            )
        )
        rcd.append(
            (
                1,
                1,
                (
                    # wrt x2, x2
                    5 * x1 * 2 * x3**2  # T4
                    + 4 * 2 * x3**2  # T6
                    + 3 * 3 * 2 * x2  # T9
                    + 3 * 2 * x3  # T10
                ),
            )
        )
        rcd.append(
            (
                2,
                2,
                (
                    # wrt x3, x3
                    5 * x1 * x2**2 * 2  # T4
                    + 4 * x2**2 * 2  # T6
                    + 4 * 4 * 3 * x3**2  # T7
                ),
            )
        )
        rcd.append(
            (
                1,
                0,
                (
                    # wrt x2, x1
                    5 * 4 * x1**3  # T2
                    + 5 * 3 * x1**2 * x3  # T3
                    + 5 * 2 * x2 * x3**2  # T4
                    + 4 * 2 * x1 * x3  # T5
                    + 3 * x3  # T8
                    + 2  # T11
                ),
            )
        )
        rcd.append(
            (
                2,
                0,
                (
                    # wrt x3, x1
                    5 * 3 * x1**2 * x2  # T3
                    + 5 * x2**2 * 2 * x3  # T4
                    + 4 * 2 * x1 * x2  # T5
                    + 3 * x2  # T8
                ),
            )
        )
        rcd.append(
            (
                2,
                1,
                (
                    # wrt x3, x2
                    5 * x1**3  # T3
                    + 5 * x1 * 2 * x2 * 2 * x3  # T4
                    + 4 * x1**2  # T5
                    + 4 * 2 * x2 * 2 * x3  # T6
                    + 3 * x1  # T8
                    + 3 * 2 * x2  # T10
                    + 2  # T12
                ),
            )
        )

        row = [r for r, _, _ in rcd]
        col = [c for _, c, _ in rcd]
        data = [d for _, _, d in rcd]
        expected_hess = sps.coo_matrix((data, (row, col)), shape=(n_x, n_x))
        expected_hess_array = expected_hess.toarray()
        expected_hess_array = (
            expected_hess_array
            + np.transpose(expected_hess_array)
            - np.diag(np.diagonal(expected_hess_array))
        )
        hess = get_hessian_of_constraint(m.eqn, list(m.x.values()))
        hess_array = hess.toarray()
        np.testing.assert_allclose(expected_hess_array, hess_array, rtol=1e-8)

    def test_unused_variable(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.0)
        m.y = pyo.Var(initialize=1.0)
        m.z = pyo.Var(initialize=1.0)
        m.eqn = pyo.Constraint(expr=m.x**2 + m.y**2 == 1.0)
        variables = [m.x, m.y, m.z]
        expected_hess = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 0]])
        hess = get_hessian_of_constraint(m.eqn, variables).toarray()
        np.testing.assert_allclose(hess, expected_hess, rtol=1e-8)

    def test_explicit_zeros(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.0)
        m.y = pyo.Var(initialize=0.0)
        m.eqn = pyo.Constraint(expr=m.x**2 + m.y**3 == 1.0)
        variables = [m.x, m.y]

        row = np.array([0, 1])
        col = np.array([0, 1])
        data = np.array([2.0, 0.0])
        expected_hess = sps.coo_matrix((data, (row, col)), shape=(2, 2))
        hess = get_hessian_of_constraint(m.eqn, variables)
        np.testing.assert_allclose(hess.row, row, atol=0)
        np.testing.assert_allclose(hess.col, col, atol=0)
        np.testing.assert_allclose(hess.data, data, rtol=1e-8)


class TestExternalPyomoModel(unittest.TestCase):
    def test_evaluate_SimpleModel1(self):
        model = SimpleModel1()
        m = model.make_model()
        x_init_list = [[-5.0], [-4.0], [-3.0], [-1.5], [0.5], [1.0], [2.0], [3.5]]
        external_model = ExternalPyomoModel(
            [m.x], [m.y], [m.residual_eqn], [m.external_eqn]
        )

        for x in x_init_list:
            external_model.set_input_values(x)
            resid = external_model.evaluate_equality_constraints()
            self.assertAlmostEqual(resid[0], model.evaluate_residual(x[0]), delta=1e-8)

    def test_jacobian_SimpleModel1(self):
        model = SimpleModel1()
        m = model.make_model()
        x_init_list = [[-5.0], [-4.0], [-3.0], [-1.5], [0.5], [1.0], [2.0], [3.5]]
        external_model = ExternalPyomoModel(
            [m.x], [m.y], [m.residual_eqn], [m.external_eqn]
        )

        for x in x_init_list:
            external_model.set_input_values(x)
            jac = external_model.evaluate_jacobian_equality_constraints()
            self.assertAlmostEqual(
                jac.toarray()[0][0], model.evaluate_jacobian(x[0]), delta=1e-8
            )

    def test_hessian_SimpleModel1(self):
        model = SimpleModel1()
        m = model.make_model()
        x_init_list = [[-5.0], [-4.0], [-3.0], [-1.5], [0.5], [1.0], [2.0], [3.5]]
        external_model = ExternalPyomoModel(
            [m.x], [m.y], [m.residual_eqn], [m.external_eqn]
        )

        for x in x_init_list:
            external_model.set_input_values(x)
            hess = external_model.evaluate_hessians_of_residuals()
            self.assertAlmostEqual(
                hess[0][0, 0], model.evaluate_hessian(x[0]), delta=1e-8
            )

    def test_external_hessian_SimpleModel1(self):
        model = SimpleModel1()
        m = model.make_model()
        x_init_list = [[-5.0], [-4.0], [-3.0], [-1.5], [0.5], [1.0], [2.0], [3.5]]
        external_model = ExternalPyomoModel(
            [m.x], [m.y], [m.residual_eqn], [m.external_eqn]
        )

        for x in x_init_list:
            external_model.set_input_values(x)
            hess = external_model.evaluate_hessian_external_variables()
            expected_hess = model.evaluate_external_hessian(x[0])
            self.assertAlmostEqual(hess[0][0, 0], expected_hess, delta=1e-8)

    def test_evaluate_SimpleModel2(self):
        model = SimpleModel2()
        m = model.make_model()
        x_init_list = [[-5.0], [-4.0], [-3.0], [-1.5], [0.5], [1.0], [2.0], [3.5]]
        external_model = ExternalPyomoModel(
            [m.x], [m.y], [m.residual_eqn], [m.external_eqn]
        )

        for x in x_init_list:
            external_model.set_input_values(x)
            resid = external_model.evaluate_equality_constraints()
            self.assertAlmostEqual(resid[0], model.evaluate_residual(x[0]), delta=1e-8)

    def test_jacobian_SimpleModel2(self):
        model = SimpleModel2()
        m = model.make_model()
        x_init_list = [[-5.0], [-4.0], [-3.0], [-1.5], [0.5], [1.0], [2.0], [3.5]]
        external_model = ExternalPyomoModel(
            [m.x], [m.y], [m.residual_eqn], [m.external_eqn]
        )

        for x in x_init_list:
            external_model.set_input_values(x)
            jac = external_model.evaluate_jacobian_equality_constraints()
            # evaluate_jacobian_equality_constraints involves an LU
            # factorization and repeated back-solve. SciPy returns a
            # dense matrix from this operation. I am not sure if I should
            # cast it to a sparse matrix. For now it is dense...
            self.assertAlmostEqual(
                jac.toarray()[0][0], model.evaluate_jacobian(x[0]), delta=1e-7
            )

    def test_hessian_SimpleModel2(self):
        model = SimpleModel2()
        m = model.make_model()
        x_init_list = [[-5.0], [-4.0], [-3.0], [-1.5], [0.5], [1.0], [2.0], [3.5]]
        external_model = ExternalPyomoModel(
            [m.x], [m.y], [m.residual_eqn], [m.external_eqn]
        )

        for x in x_init_list:
            external_model.set_input_values(x)
            hess = external_model.evaluate_hessians_of_residuals()
            self.assertAlmostEqual(
                hess[0][0, 0], model.evaluate_hessian(x[0]), delta=1e-7
            )

    def test_external_jacobian_SimpleModel2(self):
        model = SimpleModel2()
        m = model.make_model()
        x_init_list = [[-5.0], [-4.0], [-3.0], [-1.5], [0.5], [1.0], [2.0], [3.5]]
        external_model = ExternalPyomoModel(
            [m.x], [m.y], [m.residual_eqn], [m.external_eqn]
        )

        for x in x_init_list:
            external_model.set_input_values(x)
            jac = external_model.evaluate_jacobian_external_variables()
            expected_jac = model.evaluate_external_jacobian(x[0])
            self.assertAlmostEqual(jac[0, 0], expected_jac, delta=1e-8)

    def test_external_hessian_SimpleModel2(self):
        model = SimpleModel2()
        m = model.make_model()
        x_init_list = [[-5.0], [-4.0], [-3.0], [-1.5], [0.5], [1.0], [2.0], [3.5]]
        external_model = ExternalPyomoModel(
            [m.x], [m.y], [m.residual_eqn], [m.external_eqn]
        )

        for x in x_init_list:
            external_model.set_input_values(x)
            hess = external_model.evaluate_hessian_external_variables()
            expected_hess = model.evaluate_external_hessian(x[0])
            self.assertAlmostEqual(hess[0][0, 0], expected_hess, delta=1e-7)

    def test_external_jacobian_Model2by2(self):
        model = Model2by2()
        m = model.make_model()
        m.x[0].set_value(1.0)
        m.x[1].set_value(2.0)
        m.y[0].set_value(3.0)
        m.y[1].set_value(4.0)
        x0_init_list = [-5.0, -3.0, 0.5, 1.0, 2.5]
        x1_init_list = [0.5, 1.0, 1.5, 2.5, 4.1]
        x_init_list = list(itertools.product(x0_init_list, x1_init_list))
        external_model = ExternalPyomoModel(
            list(m.x.values()),
            list(m.y.values()),
            list(m.residual_eqn.values()),
            list(m.external_eqn.values()),
        )

        for x in x_init_list:
            external_model.set_input_values(x)
            jac = external_model.evaluate_jacobian_external_variables()
            expected_jac = model.evaluate_external_jacobian(x)
            np.testing.assert_allclose(jac, expected_jac, rtol=1e-8)

    def test_external_hessian_Model2by2(self):
        model = Model2by2()
        m = model.make_model()
        m.x[0].set_value(1.0)
        m.x[1].set_value(2.0)
        m.y[0].set_value(3.0)
        m.y[1].set_value(4.0)
        x0_init_list = [-5.0, -3.0, 0.5, 1.0, 2.5]
        x1_init_list = [0.5, 1.0, 1.5, 2.5, 4.1]
        x_init_list = list(itertools.product(x0_init_list, x1_init_list))
        external_model = ExternalPyomoModel(
            list(m.x.values()),
            list(m.y.values()),
            list(m.residual_eqn.values()),
            list(m.external_eqn.values()),
        )

        for x in x_init_list:
            external_model.set_input_values(x)
            hess = external_model.evaluate_hessian_external_variables()
            expected_hess = model.evaluate_external_hessian(x)
            for matrix1, matrix2 in zip(hess, expected_hess):
                np.testing.assert_allclose(matrix1, matrix2, rtol=1e-8)

    def test_external_jacobian_SimpleModel2x2_1(self):
        model = SimpleModel2by2_1()
        m = model.make_model()
        m.x[0].set_value(1.0)
        m.x[1].set_value(2.0)
        m.y[0].set_value(3.0)
        m.y[1].set_value(4.0)
        x0_init_list = [-5.0, -3.0, 0.5, 1.0, 2.5]
        x1_init_list = [-4.5, -2.3, 0.0, 1.0, 4.1]
        x_init_list = list(itertools.product(x0_init_list, x1_init_list))
        external_model = ExternalPyomoModel(
            list(m.x.values()),
            list(m.y.values()),
            list(m.residual_eqn.values()),
            list(m.external_eqn.values()),
        )

        for x in x_init_list:
            external_model.set_input_values(x)
            jac = external_model.evaluate_jacobian_external_variables()
            expected_jac = model.evaluate_external_jacobian(x)
            np.testing.assert_allclose(jac, expected_jac, rtol=1e-8)

    def test_external_hessian_SimpleModel2x2_1(self):
        model = SimpleModel2by2_1()
        m = model.make_model()
        m.x[0].set_value(1.0)
        m.x[1].set_value(2.0)
        m.y[0].set_value(3.0)
        m.y[1].set_value(4.0)
        x0_init_list = [-5.0, -3.0, 0.5, 1.0, 2.5]
        x1_init_list = [-4.5, -2.3, 0.0, 1.0, 4.1]
        x_init_list = list(itertools.product(x0_init_list, x1_init_list))
        external_model = ExternalPyomoModel(
            list(m.x.values()),
            list(m.y.values()),
            list(m.residual_eqn.values()),
            list(m.external_eqn.values()),
        )

        for x in x_init_list:
            external_model.set_input_values(x)
            hess = external_model.evaluate_hessian_external_variables()
            expected_hess = model.evaluate_external_hessian(x)
            for matrix1, matrix2 in zip(hess, expected_hess):
                np.testing.assert_allclose(matrix1, matrix2, rtol=1e-8)

    def test_evaluate_SimpleModel2x2_1(self):
        model = SimpleModel2by2_1()
        m = model.make_model()
        m.x[0].set_value(1.0)
        m.x[1].set_value(2.0)
        m.y[0].set_value(3.0)
        m.y[1].set_value(4.0)
        x0_init_list = [-5.0, -3.0, 0.5, 1.0, 2.5]
        x1_init_list = [-4.5, -2.3, 0.0, 1.0, 4.1]
        x_init_list = list(itertools.product(x0_init_list, x1_init_list))
        external_model = ExternalPyomoModel(
            list(m.x.values()),
            list(m.y.values()),
            list(m.residual_eqn.values()),
            list(m.external_eqn.values()),
        )

        for x in x_init_list:
            external_model.set_input_values(x)
            resid = external_model.evaluate_equality_constraints()
            expected_resid = model.evaluate_residual(x)
            np.testing.assert_allclose(resid, expected_resid, rtol=1e-8)

    def test_jacobian_SimpleModel2x2_1(self):
        model = SimpleModel2by2_1()
        m = model.make_model()
        m.x[0].set_value(1.0)
        m.x[1].set_value(2.0)
        m.y[0].set_value(3.0)
        m.y[1].set_value(4.0)
        x0_init_list = [-5.0, -3.0, 0.5, 1.0, 2.5]
        x1_init_list = [-4.5, -2.3, 0.0, 1.0, 4.1]
        x_init_list = list(itertools.product(x0_init_list, x1_init_list))
        external_model = ExternalPyomoModel(
            list(m.x.values()),
            list(m.y.values()),
            list(m.residual_eqn.values()),
            list(m.external_eqn.values()),
        )

        for x in x_init_list:
            external_model.set_input_values(x)
            jac = external_model.evaluate_jacobian_equality_constraints()
            expected_jac = model.evaluate_jacobian(x)
            np.testing.assert_allclose(
                jac.toarray(), expected_jac, rtol=1e-8, atol=1e-8
            )

    def test_hessian_SimpleModel2x2_1(self):
        model = SimpleModel2by2_1()
        m = model.make_model()
        m.x[0].set_value(1.0)
        m.x[1].set_value(2.0)
        m.y[0].set_value(3.0)
        m.y[1].set_value(4.0)
        x0_init_list = [-5.0, -3.0, 0.5, 1.0, 2.5]
        x1_init_list = [-4.5, -2.3, 0.0, 1.0, 4.1]
        x_init_list = list(itertools.product(x0_init_list, x1_init_list))
        external_model = ExternalPyomoModel(
            list(m.x.values()),
            list(m.y.values()),
            list(m.residual_eqn.values()),
            list(m.external_eqn.values()),
        )

        for x in x_init_list:
            external_model.set_input_values(x)
            hess = external_model.evaluate_hessians_of_residuals()
            expected_hess = model.evaluate_hessian(x)
            np.testing.assert_allclose(hess, expected_hess, rtol=1e-8)

    def test_evaluate_hessian_lagrangian_SimpleModel2x2_1(self):
        model = SimpleModel2by2_1()
        m = model.make_model()
        m.x[0].set_value(1.0)
        m.x[1].set_value(2.0)
        m.y[0].set_value(3.0)
        m.y[1].set_value(4.0)
        x0_init_list = [-5.0, -3.0, 0.5, 1.0, 2.5]
        x1_init_list = [-4.5, -2.3, 0.0, 1.0, 4.1]
        x_init_list = list(itertools.product(x0_init_list, x1_init_list))
        external_model = ExternalPyomoModel(
            list(m.x.values()),
            list(m.y.values()),
            list(m.residual_eqn.values()),
            list(m.external_eqn.values()),
        )

        for x in x_init_list:
            external_model.set_input_values(x)
            external_model.set_equality_constraint_multipliers([1.0, 1.0])
            hess_lag = external_model.evaluate_hessian_equality_constraints()
            hess_lag = hess_lag.toarray()
            expected_hess = model.evaluate_hessian(x)
            expected_hess_lag = np.tril(expected_hess[0] + expected_hess[1])
            np.testing.assert_allclose(hess_lag, expected_hess_lag, rtol=1e-8)


class TestUpdatedHessianCalculationMethods(unittest.TestCase):
    """
    These tests exercise the methods for fast Hessian-of-Lagrangian
    computation.
    They use Model2by2 because it has constraints that are nonlinear
    in both x and y.

    """

    def test_external_multipliers_from_residual_multipliers(self):
        model = Model2by2()
        m = model.make_model()
        m.x[0].set_value(1.0)
        m.x[1].set_value(2.0)
        m.y[0].set_value(3.0)
        m.y[1].set_value(4.0)
        x0_init_list = [-5.0, -3.0, 0.5, 1.0, 2.5]
        x1_init_list = [0.5, 1.0, 1.5, 2.5, 4.1]
        lam_init_list = [-2.5, -0.5, 0.0, 1.0, 2.0]
        init_list = list(itertools.product(x0_init_list, x1_init_list, lam_init_list))
        external_model = ExternalPyomoModel(
            list(m.x.values()),
            list(m.y.values()),
            list(m.residual_eqn.values()),
            list(m.external_eqn.values()),
        )

        for x0, x1, lam in init_list:
            x = [x0, x1]
            lam = [lam]
            external_model.set_input_values(x)
            lam_g = external_model.calculate_external_constraint_multipliers(lam)
            pred_lam_g = model.calculate_external_multipliers(lam, x)
            np.testing.assert_allclose(lam_g, pred_lam_g, rtol=1e-8)

    def test_full_space_lagrangian_hessians(self):
        model = Model2by2()
        m = model.make_model()
        m.x[0].set_value(1.0)
        m.x[1].set_value(2.0)
        m.y[0].set_value(3.0)
        m.y[1].set_value(4.0)
        x0_init_list = [-5.0, -3.0, 0.5, 1.0, 2.5]
        x1_init_list = [0.5, 1.0, 1.5, 2.5, 4.1]
        lam_init_list = [-2.5, -0.5, 0.0, 1.0, 2.0]
        init_list = list(itertools.product(x0_init_list, x1_init_list, lam_init_list))
        external_model = ExternalPyomoModel(
            list(m.x.values()),
            list(m.y.values()),
            list(m.residual_eqn.values()),
            list(m.external_eqn.values()),
        )

        for x0, x1, lam in init_list:
            x = [x0, x1]
            lam = [lam]
            external_model.set_input_values(x)
            # Note that these multiplier calculations are dependent on x,
            # so if we switch their order, we will get "wrong" answers.
            # (This is wrong in the sense that the residual and external
            # multipliers won't necessarily correspond).
            external_model.set_external_constraint_multipliers(lam)
            hlxx, hlxy, hlyy = external_model.get_full_space_lagrangian_hessians()
            (
                pred_hlxx,
                pred_hlxy,
                pred_hlyy,
            ) = model.calculate_full_space_lagrangian_hessians(lam, x)

            # TODO: Is comparing the array representation sufficient here?
            # Should I make sure I get the sparse representation I expect?
            np.testing.assert_allclose(hlxx.toarray(), pred_hlxx, rtol=1e-8)
            np.testing.assert_allclose(hlxy.toarray(), pred_hlxy, rtol=1e-8)
            np.testing.assert_allclose(hlyy.toarray(), pred_hlyy, rtol=1e-8)

    def test_reduced_hessian_lagrangian(self):
        model = Model2by2()
        m = model.make_model()
        m.x[0].set_value(1.0)
        m.x[1].set_value(2.0)
        m.y[0].set_value(3.0)
        m.y[1].set_value(4.0)
        x0_init_list = [-5.0, -3.0, 0.5, 1.0, 2.5]
        x1_init_list = [0.5, 1.0, 1.5, 2.5, 4.1]
        lam_init_list = [-2.5, -0.5, 0.0, 1.0, 2.0]
        init_list = list(itertools.product(x0_init_list, x1_init_list, lam_init_list))
        external_model = ExternalPyomoModel(
            list(m.x.values()),
            list(m.y.values()),
            list(m.residual_eqn.values()),
            list(m.external_eqn.values()),
        )

        for x0, x1, lam in init_list:
            x = [x0, x1]
            lam = [lam]
            external_model.set_input_values(x)
            # Same comment as previous test regarding calculation order
            external_model.set_external_constraint_multipliers(lam)
            hlxx, hlxy, hlyy = external_model.get_full_space_lagrangian_hessians()
            hess = external_model.calculate_reduced_hessian_lagrangian(hlxx, hlxy, hlyy)
            pred_hess = model.calculate_reduced_lagrangian_hessian(lam, x)
            # This test asserts that we are doing the block reduction properly.
            np.testing.assert_allclose(np.array(hess), pred_hess, rtol=1e-8)

            from_individual = external_model.evaluate_hessians_of_residuals()
            hl_from_individual = sum(l * h for l, h in zip(lam, from_individual))
            # This test asserts that the block reduction is correct.
            np.testing.assert_allclose(np.array(hess), hl_from_individual, rtol=1e-8)

    def test_evaluate_hessian_equality_constraints(self):
        model = Model2by2()
        m = model.make_model()
        m.x[0].set_value(1.0)
        m.x[1].set_value(2.0)
        m.y[0].set_value(3.0)
        m.y[1].set_value(4.0)
        x0_init_list = [-5.0, -3.0, 0.5, 1.0, 2.5]
        x1_init_list = [0.5, 1.0, 1.5, 2.5, 4.1]
        lam_init_list = [-2.5, -0.5, 0.0, 1.0, 2.0]
        init_list = list(itertools.product(x0_init_list, x1_init_list, lam_init_list))
        external_model = ExternalPyomoModel(
            list(m.x.values()),
            list(m.y.values()),
            list(m.residual_eqn.values()),
            list(m.external_eqn.values()),
        )

        for x0, x1, lam in init_list:
            x = [x0, x1]
            lam = [lam]
            external_model.set_input_values(x)
            external_model.set_equality_constraint_multipliers(lam)
            hess = external_model.evaluate_hessian_equality_constraints()
            pred_hess = model.calculate_reduced_lagrangian_hessian(lam, x)
            # This test asserts that we are doing the block reduction properly.
            np.testing.assert_allclose(hess.toarray(), np.tril(pred_hess), rtol=1e-8)

            from_individual = external_model.evaluate_hessians_of_residuals()
            hl_from_individual = sum(l * h for l, h in zip(lam, from_individual))
            # This test asserts that the block reduction is correct.
            np.testing.assert_allclose(
                hess.toarray(), np.tril(hl_from_individual), rtol=1e-8
            )

    def test_evaluate_hessian_equality_constraints_order(self):
        model = Model2by2()
        m = model.make_model()
        m.x[0].set_value(1.0)
        m.x[1].set_value(2.0)
        m.y[0].set_value(3.0)
        m.y[1].set_value(4.0)
        x0_init_list = [-5.0, -3.0, 0.5, 1.0, 2.5]
        x1_init_list = [0.5, 1.0, 1.5, 2.5, 4.1]
        lam_init_list = [-2.5, -0.5, 0.0, 1.0, 2.0]
        init_list = list(itertools.product(x0_init_list, x1_init_list, lam_init_list))
        external_model = ExternalPyomoModel(
            list(m.x.values()),
            list(m.y.values()),
            list(m.residual_eqn.values()),
            list(m.external_eqn.values()),
        )

        for x0, x1, lam in init_list:
            x = [x0, x1]
            lam = [lam]
            external_model.set_equality_constraint_multipliers(lam)
            external_model.set_input_values(x)
            # Using evaluate_hessian_equality_constraints, which calculates
            # external multiplier values, we can calculate the correct Hessian
            # regardless of the order in which primal and dual variables are
            # set.
            hess = external_model.evaluate_hessian_equality_constraints()
            pred_hess = model.calculate_reduced_lagrangian_hessian(lam, x)
            # This test asserts that we are doing the block reduction properly.
            np.testing.assert_allclose(hess.toarray(), np.tril(pred_hess), rtol=1e-8)

            from_individual = external_model.evaluate_hessians_of_residuals()
            hl_from_individual = sum(l * h for l, h in zip(lam, from_individual))
            # This test asserts that the block reduction is correct.
            np.testing.assert_allclose(
                hess.toarray(), np.tril(hl_from_individual), rtol=1e-8
            )


class TestScaling(unittest.TestCase):
    def con_3_body(self, x, y, u, v):
        return 1e5 * x**2 + 1e4 * y**2 + 1e1 * u**2 + 1e0 * v**2

    def con_3_rhs(self):
        return 2.0e4

    def con_4_body(self, x, y, u, v):
        return 1e-2 * x + 1e-3 * y + 1e-4 * u + 1e-4 * v

    def con_4_rhs(self):
        return 3.0e-4

    def make_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.0)
        m.y = pyo.Var(initialize=1.0)
        m.u = pyo.Var(initialize=1.0)
        m.v = pyo.Var(initialize=1.0)
        m.con_1 = pyo.Constraint(expr=m.x * m.y == m.u)
        m.con_2 = pyo.Constraint(expr=m.x**2 * m.y**3 == m.v)
        m.con_3 = pyo.Constraint(
            expr=self.con_3_body(m.x, m.y, m.u, m.v) == self.con_3_rhs()
        )
        m.con_4 = pyo.Constraint(
            expr=self.con_4_body(m.x, m.y, m.u, m.v) == self.con_4_rhs()
        )

        epm_model = pyo.ConcreteModel()
        epm_model.x = pyo.Reference(m.x)
        epm_model.y = pyo.Reference(m.y)
        epm_model.u = pyo.Reference(m.u)
        epm_model.v = pyo.Reference(m.v)
        epm_model.epm = ExternalPyomoModel(
            [m.u, m.v], [m.x, m.y], [m.con_3, m.con_4], [m.con_1, m.con_2]
        )
        epm_model.obj = pyo.Objective(expr=m.x**2 + m.y**2 + m.u**2 + m.v**2)
        epm_model.egb = ExternalGreyBoxBlock()
        epm_model.egb.set_external_model(epm_model.epm, inputs=[m.u, m.v])
        return epm_model

    def test_get_set_scaling_factors(self):
        m = self.make_model()
        scaling_factors = [1e-4, 1e4]
        m.epm.set_equality_constraint_scaling_factors(scaling_factors)
        epm_sf = m.epm.get_equality_constraint_scaling_factors()
        np.testing.assert_array_equal(scaling_factors, epm_sf)

    def test_pyomo_nlp(self):
        m = self.make_model()
        scaling_factors = [1e-4, 1e4]
        m.epm.set_equality_constraint_scaling_factors(scaling_factors)
        nlp = PyomoNLPWithGreyBoxBlocks(m)
        nlp_sf = nlp.get_constraints_scaling()
        np.testing.assert_array_equal(scaling_factors, nlp_sf)

    @unittest.skipUnless(cyipopt_available, "cyipopt is not available")
    def test_cyipopt_nlp(self):
        m = self.make_model()
        scaling_factors = [1e-4, 1e4]
        m.epm.set_equality_constraint_scaling_factors(scaling_factors)
        nlp = PyomoNLPWithGreyBoxBlocks(m)
        cyipopt_nlp = CyIpoptNLP(nlp)
        obj_scaling, x_scaling, g_scaling = cyipopt_nlp.scaling_factors()
        np.testing.assert_array_equal(scaling_factors, g_scaling)

    @unittest.skipUnless(cyipopt_available, "cyipopt is not available")
    def test_cyipopt_callback(self):
        # Use a callback to check that the reported infeasibility is
        # due to the scaled equality constraints.
        # Note that the scaled infeasibility is not what we see if we
        # call solve with tee=True, as by default the displayed infeasibility
        # is unscaled. Luckily, we can still access the scaled infeasibility
        # with a callback.
        m = self.make_model()
        scaling_factors = [1e-4, 1e4]
        m.epm.set_equality_constraint_scaling_factors(scaling_factors)
        nlp = PyomoNLPWithGreyBoxBlocks(m)

        def callback(
            local_nlp,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials,
        ):
            primals = tuple(local_nlp.get_primals())
            # I happen to know the order of the primals here
            u, v, x, y = primals

            # Calculate the scaled residuals I expect
            con_3_resid = scaling_factors[0] * abs(
                self.con_3_body(x, y, u, v) - self.con_3_rhs()
            )
            con_4_resid = scaling_factors[1] * abs(
                self.con_4_body(x, y, u, v) - self.con_4_rhs()
            )
            pred_inf_pr = max(con_3_resid, con_4_resid)

            # Make sure Ipopt is using the scaled constraints internally
            self.assertAlmostEqual(inf_pr, pred_inf_pr)

        cyipopt_nlp = CyIpoptNLP(nlp, intermediate_callback=callback)
        x0 = nlp.get_primals()
        cyipopt = CyIpoptSolver(
            cyipopt_nlp, options={"max_iter": 0, "nlp_scaling_method": "user-scaling"}
        )
        cyipopt.solve(x0=x0)


if __name__ == '__main__':
    unittest.main()
