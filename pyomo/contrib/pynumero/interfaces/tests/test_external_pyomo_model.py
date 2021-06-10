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
    # will look like: x**2 + 1/x**2 - 1 == 0
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
