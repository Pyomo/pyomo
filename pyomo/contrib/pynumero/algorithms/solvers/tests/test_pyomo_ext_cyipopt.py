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
import pyomo.environ as pyo

from pyomo.contrib.pynumero.dependencies import (
    numpy as np, numpy_available, scipy_sparse as spa, scipy_available
)
if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.contrib.pynumero.asl import AmplInterface
if not AmplInterface.available():
    raise unittest.SkipTest(
        "Pynumero needs the ASL extension to run CyIpoptSolver tests")

try:
    import ipopt
except ImportError:
    raise unittest.SkipTest("Pynumero needs cyipopt to run CyIpoptSolver tests")

from pyomo.contrib.pynumero.algorithms.solvers.pyomo_ext_cyipopt import ExternalInputOutputModel, PyomoExternalCyIpoptProblem
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver

class PressureDropModel(ExternalInputOutputModel):
    def __init__(self):
        self._Pin = None
        self._c1 = None
        self._c2 = None
        self._F = None

    def set_inputs(self, input_values):
        assert len(input_values) == 4
        self._Pin = input_values[0]
        self._c1 = input_values[1]
        self._c2 = input_values[2]
        self._F = input_values[3]

    def evaluate_outputs(self):
        P1 = self._Pin - self._c1*self._F**2
        P2 = P1 - self._c2*self._F**2
        return np.asarray([P1, P2], dtype=np.float64)

    def evaluate_derivatives(self):
        jac = [[1, -self._F**2, 0, -2*self._c1*self._F],
               [1, -self._F**2, -self._F**2, -2*self._F*(self._c1 + self._c2)]]
        jac = np.asarray(jac, dtype=np.float64)
        return spa.coo_matrix(jac)

class TestExternalInputOutputModel(unittest.TestCase):

    def test_interface(self):
        # weird, this is really a test of the test class above
        # but we could add code later, so...
        iom = PressureDropModel()
        iom.set_inputs(np.ones(4))
        o = iom.evaluate_outputs()
        expected_o = np.asarray([0.0, -1.0], dtype=np.float64)
        self.assertTrue(np.array_equal(o, expected_o))

        jac = iom.evaluate_derivatives()
        expected_jac = np.asarray([[1, -1, 0, -2], [1, -1, -1, -4]], dtype=np.float64)
        self.assertTrue(np.array_equal(jac.todense(), expected_jac))

    def test_pyomo_external_model(self):
        m = pyo.ConcreteModel()
        m.Pin = pyo.Var(initialize=100, bounds=(0,None))
        m.c1 = pyo.Var(initialize=1.0, bounds=(0,None))
        m.c2 = pyo.Var(initialize=1.0, bounds=(0,None))
        m.F = pyo.Var(initialize=10, bounds=(0,None))

        m.P1 = pyo.Var()
        m.P2 = pyo.Var()

        m.F_con = pyo.Constraint(expr = m.F == 10)
        m.Pin_con = pyo.Constraint(expr = m.Pin == 100)

        # simple parameter estimation test
        m.obj = pyo.Objective(expr= (m.P1 - 90)**2 + (m.P2 - 40)**2)

        cyipopt_problem = \
            PyomoExternalCyIpoptProblem(m,
                                        PressureDropModel(),
                                        [m.Pin, m.c1, m.c2, m.F],
                                        [m.P1, m.P2]
                                        )

        # check that the dummy variable is initialized
        expected_dummy_var_value = pyo.value(m.Pin) + pyo.value(m.c1) + pyo.value(m.c2) + pyo.value(m.F) \
            + 0 + 0
            # + pyo.value(m.P1) + pyo.value(m.P2) # not initialized - therefore should use zero
        self.assertAlmostEqual(pyo.value(m._dummy_variable_CyIpoptPyomoExNLP), expected_dummy_var_value)

        # solve the problem
        solver = CyIpoptSolver(cyipopt_problem, {'hessian_approximation':'limited-memory'})
        x, info = solver.solve(tee=False)
        cyipopt_problem.load_x_into_pyomo(x)
        self.assertAlmostEqual(pyo.value(m.c1), 0.1, places=5)
        self.assertAlmostEqual(pyo.value(m.c2), 0.5, places=5)

    def test_pyomo_external_model_dummy_var_initialization(self):
        m = pyo.ConcreteModel()
        m.Pin = pyo.Var(initialize=100, bounds=(0,None))
        m.c1 = pyo.Var(initialize=1.0, bounds=(0,None))
        m.c2 = pyo.Var(initialize=1.0, bounds=(0,None))
        m.F = pyo.Var(initialize=10, bounds=(0,None))

        m.P1 = pyo.Var(initialize=75.0)
        m.P2 = pyo.Var(initialize=50.0)

        m.F_con = pyo.Constraint(expr = m.F == 10)
        m.Pin_con = pyo.Constraint(expr = m.Pin == 100)

        # simple parameter estimation test
        m.obj = pyo.Objective(expr= (m.P1 - 90)**2 + (m.P2 - 40)**2)

        cyipopt_problem = \
            PyomoExternalCyIpoptProblem(m,
                                        PressureDropModel(),
                                        [m.Pin, m.c1, m.c2, m.F],
                                        [m.P1, m.P2]
                                        )

        # check that the dummy variable is initialized
        expected_dummy_var_value = pyo.value(m.Pin) + pyo.value(m.c1) + pyo.value(m.c2) + pyo.value(m.F) \
            + pyo.value(m.P1) + pyo.value(m.P2)
        self.assertAlmostEqual(pyo.value(m._dummy_variable_CyIpoptPyomoExNLP), expected_dummy_var_value)
        # check that the dummy constraint is satisfied
        self.assertAlmostEqual(pyo.value(m._dummy_constraint_CyIpoptPyomoExNLP.body),pyo.value(m._dummy_constraint_CyIpoptPyomoExNLP.lower))
        self.assertAlmostEqual(pyo.value(m._dummy_constraint_CyIpoptPyomoExNLP.body),pyo.value(m._dummy_constraint_CyIpoptPyomoExNLP.upper))

        # solve the problem
        solver = CyIpoptSolver(cyipopt_problem, {'hessian_approximation':'limited-memory'})
        x, info = solver.solve(tee=False)
        cyipopt_problem.load_x_into_pyomo(x)
        self.assertAlmostEqual(pyo.value(m.c1), 0.1, places=5)
        self.assertAlmostEqual(pyo.value(m.c2), 0.5, places=5)

