#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

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

def make_simple_model_1():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=2.0)
    m.y = pyo.Var(initialize=2.0)
    m.residual_eqn = pyo.Constraint(expr=m.x**2 + m.y**2 == 1.0)
    m.external_eqn = pyo.Constraint(expr=m.x*m.y == 0.2)
    # The "external function constraint" exposed by the ExternalPyomoModel
    # will look like: x**2 + 0.04/x**2 - 1 == 0
    return m

# Could construct an equivalent ExternalGreyBoxModel...
def simple_model_1_residual(x):
    try:
        x = x[0]
    except TypeError:
        pass
    return x**2 + 0.04/x**2 - 1

def simple_model_1_jacobian(x):
    try:
        x = x[0]
    except TypeError:
        pass
    return 2*x - 0.08/x**3

"""
Tests should cover:
    1. Residual, Jacobian, and Hessian evaluation
    2. Embed in an ExternalGreyBoxBlock
    3. Convert to a CyIpoptNLP and solve
"""


class TestGetHessianOfConstraint(unittest.TestCase):

    def test_simple_model_1(self):
        m = make_simple_model_1()
        m.x.set_value(2.0)
        m.y.set_value(2.0)

        con = m.residual_eqn
        expected_hess = np.array([[2.0, 0.0], [0.0, 2.0]])
        hess = get_hessian_of_constraint(con)
        self.assertTrue(np.all(expected_hess == hess.toarray()))

        expected_hess = np.array([[2.0]])
        hess = get_hessian_of_constraint(con, wrt=[m.x])
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
        hess = get_hessian_of_constraint(m.eqn, wrt=list(m.x.values()))
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
        hess = get_hessian_of_constraint(m.eqn, wrt=variables).toarray()
        np.testing.assert_allclose(hess, expected_hess, rtol=1e-8)


class TestExternalPyomoModel(unittest.TestCase):

    def test_evaluate_simple_model_1(self):
        m = make_simple_model_1()
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
                    simple_model_1_residual(x),
                    delta=1e-7,
                    )

    def test_jacobian_simple_model_1(self):
        m = make_simple_model_1()
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
                    simple_model_1_jacobian(x),
                    delta=1e-7,
                    )


if __name__ == '__main__':
    unittest.main()
