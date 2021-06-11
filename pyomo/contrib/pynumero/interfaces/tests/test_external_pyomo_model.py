#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import itertools
import pyomo.common.unittest as unittest
import pyomo.environ as pyo

from pyomo.contrib.pynumero.dependencies import (
    numpy as np, numpy_available, scipy, scipy_available
)
from pyomo.common.dependencies.scipy import sparse as sps

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.contrib.pynumero.asl import AmplInterface
if not AmplInterface.available():
    raise unittest.SkipTest(
        "Pynumero needs the ASL extension to run cyipopt tests")

from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
    cyipopt_available,
)
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import (
    ExternalPyomoModel,
    get_hessian_of_constraint,
)

if not pyo.SolverFactory("ipopt").available():
    raise unittest.SkipTest(
        "Need IPOPT to run ExternalPyomoModel tests"
        )


class SimpleModel1(object):

    def make_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=2.0)
        m.y = pyo.Var(initialize=2.0)
        m.residual_eqn = pyo.Constraint(expr=m.x**2 + m.y**2 == 1.0)
        m.external_eqn = pyo.Constraint(expr=m.x*m.y == 0.2)
        # The "external function constraint" exposed by the ExternalPyomoModel
        # will look like: x**2 + 0.04/x**2 - 1 == 0
        return m

    def evaluate_external_variables(self, x):
        # y(x)
        return 0.2/x

    def evaluate_external_jacobian(self, x):
        # dydx
        return -0.2/(x**2)

    def evaluate_external_hessian(self, x):
        # d2ydx2
        return 0.4/(x**3)

    def evaluate_residual(self, x):
        return x**2 + 0.04/x**2 - 1

    def evaluate_jacobian(self, x):
        return 2*x - 0.08/x**3


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
        m.external_eqn = pyo.Constraint(expr=(m.x**3)*(m.y**3) == 0.2)
        return m

    def evaluate_external_variables(self, x):
        return 0.2**(1/3)/x

    def evaluate_external_jacobian(self, x):
        return -(0.2**(1/3))/(x**2)

    def evaluate_external_hessian(self, x):
        return 2*0.2**(1/3)/(x**3)


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

        m.residual_eqn = pyo.Constraint(expr=
                m.x[0]**2 + m.x[1]**2 + m.y[0]**2 + m.y[1]**2 == 1.0
                )

        def external_eqn_rule(m, i):
            if i == 0:
                return (m.x[0]**2) * m.y[0] * (m.x[1]**0.5) * m.y[1] - 0.1 == 0
            elif i == 1:
                return m.x[0] * m.y[0] * m.x[1] - 0.2 == 0

        m.external_eqn = pyo.Constraint([0, 1], rule=external_eqn_rule)

        return m

    def evaluate_external_variables(self, x):
        y0 = 0.2/(x[0]*x[1])
        y1 = 0.1/(x[0]**2 * y0 * x[1]**0.5)
        return [y0, y1]

    def evaluate_external_jacobian(self, x):
        dy0dx0 = -0.2/(x[0]**2 * x[1])
        dy0dx1 = -0.2/(x[0] * x[1]**2)
        dy1dx0 = -0.5*x[1]**0.5/x[0]**2
        dy1dx1 = 0.25/(x[0] * x[1]**0.5)
        return np.array([[dy0dx0, dy0dx1], [dy1dx0, dy1dx1]])

    def evaluate_external_hessian(self, x):
        d2y0dx0dx0 = 0.4/(x[0]**3 * x[1])
        d2y0dx0dx1 = 0.2/(x[0]**2 * x[1]**2)
        d2y0dx1dx1 = 0.4/(x[0] * x[1]**3)

        d2y1dx0dx0 = x[1]**0.5/(x[0]**3)
        d2y1dx0dx1 = -0.25/(x[0]**2 * x[1]**0.5)
        d2y1dx1dx1 = -0.125/(x[0] * x[1]**1.5)

        d2y0dxdx = np.array([
            [d2y0dx0dx0, d2y0dx0dx1],
            [d2y0dx0dx1, d2y0dx1dx1],
            ])
        d2y1dxdx = np.array([
            [d2y1dx0dx0, d2y1dx0dx1],
            [d2y1dx0dx1, d2y1dx1dx1],
            ])
        return [d2y0dxdx, d2y1dxdx]

"""
Tests should cover:
    1. Residual, Jacobian, and Hessian evaluation
    2. Embed in an ExternalGreyBoxBlock
    3. Convert to a CyIpoptNLP and solve
"""


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
        m.x = pyo.Var(range(1, n_x+1), initialize={1: x1, 2: x2, 3: x3})
        m.eqn = pyo.Constraint(expr=
                5*(m.x[1]**5) +                        # T1
                5*(m.x[1]**4)*(m.x[2]) +               # T2
                5*(m.x[1]**3)*(m.x[2])*(m.x[3]) +      # T3
                5*(m.x[1])*(m.x[2]**2)*(m.x[3]**2) +   # T4
                4*(m.x[1]**2)*(m.x[2])*(m.x[3]) +      # T5
                4*(m.x[2]**2)*(m.x[3]**2) +            # T6
                4*(m.x[3]**4) +                        # T7
                3*(m.x[1])*(m.x[2])*(m.x[3]) +         # T8
                3*(m.x[2]**3) +                        # T9
                3*(m.x[2]**2)*(m.x[3]) +               # T10
                2*(m.x[1])*(m.x[2]) +                  # T11
                2*(m.x[2])*(m.x[3])                    # T12
                == 0
                )

        rcd = []
        rcd.append((0, 0, (
            # wrt x1, x1
            5*5*4*x1**3 +    # T1
            5*4*3*x1**2*x2 + # T2
            5*3*2*x1*x2*x3 + # T3
            4*2*1*x2*x3      # T5
            )))
        rcd.append((1, 1, (
            # wrt x2, x2
            5*x1*2*x3**2 +   # T4
            4*2*x3**2 +      # T6
            3*3*2*x2 +       # T9
            3*2*x3           # T10
            )))
        rcd.append((2, 2, (
            # wrt x3, x3
            5*x1*x2**2*2 +   # T4
            4*x2**2*2 +      # T6
            4*4*3*x3**2      # T7
            )))
        rcd.append((1, 0, (
            # wrt x2, x1
            5*4*x1**3 +      # T2
            5*3*x1**2*x3 +   # T3
            5*2*x2*x3**2 +   # T4
            4*2*x1*x3 +      # T5
            3*x3 +           # T8
            2                # T11
            )))
        rcd.append((2, 0, (
            # wrt x3, x1
            5*3*x1**2*x2 +   # T3
            5*x2**2*2*x3 +   # T4
            4*2*x1*x2 +      # T5
            3*x2             # T8
            )))
        rcd.append((2, 1, (
            # wrt x3, x2
            5*x1**3 +        # T3
            5*x1*2*x2*2*x3 + # T4
            4*x1**2 +        # T5
            4*2*x2*2*x3 +    # T6
            3*x1 +           # T8
            3*2*x2 +         # T10
            2                # T12
            )))

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


class TestExternalPyomoModel(unittest.TestCase):

    def test_evaluate_SimpleModel1(self):
        model = SimpleModel1()
        m = model.make_model()
        x_init_list = [
                [-5.0], [-4.0], [-3.0], [-1.5], [0.5], [1.0], [2.0], [3.5]
                ]
        external_model = ExternalPyomoModel(
                [m.x], [m.y], [m.residual_eqn], [m.external_eqn],
                )

        for x in x_init_list:
            external_model.set_input_values(x)
            resid = external_model.evaluate_equality_constraints()
            self.assertAlmostEqual(
                    resid[0],
                    model.evaluate_residual(x[0]),
                    delta=1e-8,
                    )

    def test_jacobian_SimpleModel1(self):
        model = SimpleModel1()
        m = model.make_model()
        x_init_list = [
                [-5.0], [-4.0], [-3.0], [-1.5], [0.5], [1.0], [2.0], [3.5]
                ]
        external_model = ExternalPyomoModel(
                [m.x], [m.y], [m.residual_eqn], [m.external_eqn],
                )

        for x in x_init_list:
            external_model.set_input_values(x)
            jac = external_model.evaluate_jacobian_equality_constraints()
            # evaluate_jacobian_equality_constraints involves an LU
            # factorization and repeated back-solve. SciPy returns a
            # dense matrix from this operation. I am not sure if I should
            # cast it to a sparse matrix. For now it is dense...
            self.assertAlmostEqual(
                    jac[0][0],
                    model.evaluate_jacobian(x[0]),
                    delta=1e-8,
                    )

    def test_external_hessian_SimpleModel1(self):
        model = SimpleModel1()
        m = model.make_model()
        x_init_list = [
                [-5.0], [-4.0], [-3.0], [-1.5], [0.5], [1.0], [2.0], [3.5]
                ]
        external_model = ExternalPyomoModel(
                [m.x], [m.y], [m.residual_eqn], [m.external_eqn],
                )

        for x in x_init_list:
            external_model.set_input_values(x)
            hess = external_model.evaluate_hessian_external_variables()
            expected_hess = model.evaluate_external_hessian(x[0])
            self.assertAlmostEqual(
                    hess[0][0,0],
                    expected_hess,
                    delta=1e-8,
                    )

    def test_external_jacobian_SimpleModel2(self):
        model = SimpleModel2()
        m = model.make_model()
        x_init_list = [
                [-5.0], [-4.0], [-3.0], [-1.5], [0.5], [1.0], [2.0], [3.5]
                ]
        external_model = ExternalPyomoModel(
                [m.x], [m.y], [m.residual_eqn], [m.external_eqn],
                )

        for x in x_init_list:
            external_model.set_input_values(x)
            jac = external_model.evaluate_jacobian_external_variables()
            expected_jac = model.evaluate_external_jacobian(x[0])
            self.assertAlmostEqual(
                    jac[0,0],
                    expected_jac,
                    delta=1e-8,
                    )

    def test_external_hessian_SimpleModel2(self):
        model = SimpleModel2()
        m = model.make_model()
        x_init_list = [
                [-5.0], [-4.0], [-3.0], [-1.5], [0.5], [1.0], [2.0], [3.5]
                ]
        external_model = ExternalPyomoModel(
                [m.x], [m.y], [m.residual_eqn], [m.external_eqn],
                )

        for x in x_init_list:
            external_model.set_input_values(x)
            hess = external_model.evaluate_hessian_external_variables()
            expected_hess = model.evaluate_external_hessian(x[0])
            self.assertAlmostEqual(
                    hess[0][0,0],
                    expected_hess,
                    delta=1e-7,
                    )

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
                matrix2 = np.matrix(matrix2)
                np.testing.assert_allclose(matrix1, matrix2, rtol=1e-8)


if __name__ == '__main__':
    #unittest.main()
    test = TestExternalPyomoModel()
    test.test_external_hessian_Model2by2()
