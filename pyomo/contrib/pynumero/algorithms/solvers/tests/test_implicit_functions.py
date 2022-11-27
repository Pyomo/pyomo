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
from pyomo.common.dependencies import (
    scipy,
    scipy_available,
    numpy as np,
    numpy_available,
)
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
    ImplicitFunctionSolver,
    SccImplicitFunctionSolver,
)

if not scipy_available or not numpy_available:
    # SciPy is only really necessary as it is a dependency of AmplInterface.
    # NumPy is directly used by the implicit function solvers.
    raise unittest.SkipTest(
        "NumPy and SciPy are needed to test the implicit function solvers"
    )
if not AmplInterface.available():
    # AmplInterface is not theoretically necessary for these solvers,
    # however it is the only AD backend implemented for PyomoNLP.
    raise unittest.SkipTest(
        "PyNumero ASL extension is necessary to test implicit"
        " function solvers"
    )


def make_simple_model():
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=[1, 2, 3])
    m.x = pyo.Var(m.I, initialize=1.0)
    m.con1 = pyo.Constraint(expr=m.x[1]**2 + m.x[2]**2 + m.x[3]**2 == 1)
    m.con2 = pyo.Constraint(expr=2*m.x[1] + 3*m.x[2] - 4*m.x[3] == 0)
    m.con3 = pyo.Constraint(expr=m.x[1] == 2*pyo.exp(m.x[2]/m.x[3]))
    m.obj = pyo.Objective(expr=0.0)
    nlp = PyomoNLP(m)
    return m, nlp


class ImplicitFunction1(object):

    def __init__(self):
        self._model = self._make_model()

    def _make_model(self):
        m = pyo.ConcreteModel()
        m.I = pyo.Set(initialize=[1, 2, 3])
        m.J = pyo.Set(initialize=[1, 2])
        m.x = pyo.Var(m.I, initialize=1.0)
        m.p = pyo.Var(m.J, initialize=1.0)
        # Note that this system of constraints decomposes. First con1
        # and con3 are used to solve for x[2] and x[3], then con2
        # is used to solve for x[1]
        m.con1 = pyo.Constraint(
            expr=m.x[2]**2 + m.x[3]**2 == m.p[1]
        )
        m.con2 = pyo.Constraint(
            expr=2*m.x[1] + 3*m.x[2] - 4*m.x[3] == m.p[1]**2 - m.p[2]
        )
        m.con3 = pyo.Constraint(
            expr=m.p[2]**1.5 == 2*pyo.exp(m.x[2]/m.x[3])
        )
        m.obj = pyo.Objective(expr=0.0)
        return m

    def get_parameters(self):
        m = self._model
        return [m.p[1], m.p[2]]

    def get_variables(self):
        m = self._model
        return [m.x[1], m.x[2], m.x[3]]

    def get_equations(self):
        m = self._model
        return [m.con1, m.con2, m.con3]

    def get_input_output_sequence(self):
        p1_inputs = [1.0, 2.0, 3.0]
        p2_inputs = [1.0, 2.0, 3.0]
        inputs = list(itertools.product(p1_inputs, p2_inputs))

        outputs = [
            # Outputs computed by solving system with Ipopt
            (2.498253, -0.569676, 0.821869),
            (0.898530, 0.327465, 0.944863),
            (-0.589294, 0.690561, 0.723274),
            (5.033063, -0.805644, 1.162299),
            (2.977820, 0.463105, 1.336239),
            (1.080826, 0.976601, 1.022864),
            (8.327101, -0.986708, 1.423519),
            (5.922325, 0.567186, 1.636551),
            (3.711364, 1.196087, 1.252747),
        ]
        # We will iterate over these input/output pairs, set inputs,
        # solve, and check outputs. Note that these values are computed
        # with Ipopt, and the default solver for the system defining the
        # implicit function is scipy.optimize.fsolve. There is no guarantee
        # that these algorithms converge to the same solution for the
        # highly nonlinear system defining this implicit function. If some
        # of these tests start failing (e.g. because one of these algorithms
        # changes), some of these inputs may need to be omitted.
        return list(zip(inputs, outputs))


class _TestSolver(unittest.TestCase):
    """A suite of basic tests for implicit function solvers.

    A "concrete" subclass should be defined for each implicit function
    solver. This subclass should implement get_solver_class, then
    add "test" methods that call the following methods:

        _test_implicit_function_1

    These methods are private so they don't get picked up on the base
    class by pytest.

    """

    def get_solver_class(self):
        raise NotImplementedError()

    def _test_implicit_function_1(self):
        SolverClass = self.get_solver_class()
        fcn = ImplicitFunction1()
        variables = fcn.get_variables()
        parameters = fcn.get_parameters()
        equations = fcn.get_equations()

        solver = SolverClass(variables, equations, parameters)

        for inputs, pred_outputs in fcn.get_input_output_sequence():
            solver.set_parameters(inputs)
            outputs = solver.evaluate_outputs()
            self.assertStructuredAlmostEqual(
                list(outputs), list(pred_outputs), reltol=1e-5, abstol=1e-5
            )

            solver.update_pyomo_model()
            for i, var in enumerate(variables):
                self.assertAlmostEqual(
                    var.value, pred_outputs[i], delta=1e-5
                )


class TestImplicitFunctionSolver(_TestSolver):

    def get_solver_class(self):
        return ImplicitFunctionSolver

    def test_implicit_function_1(self):
        self._test_implicit_function_1()


class TestSccImplicitFunctionSolver(_TestSolver):

    def get_solver_class(self):
        return SccImplicitFunctionSolver

    def test_implicit_function_1(self):
        self._test_implicit_function_1()


#def _solve_with_ipopt():
#    from pyomo.util.subsystems import TemporarySubsystemManager
#    ipopt = pyo.SolverFactory("ipopt")
#    fcn = ImplicitFunction1()
#    m = fcn._model
#    params = fcn.get_parameters()
#    variables = fcn.get_variables()
#    input_list = []
#    output_list = []
#    error_list = []
#    for (p1, p2), _ in fcn.get_input_output_sequence():
#        params[0].set_value(p1)
#        params[1].set_value(p2)
#        input_list.append((p1, p2))
#        with TemporarySubsystemManager(to_fix=params):
#            try:
#                res = ipopt.solve(m, tee=True)
#                pyo.assert_optimal_termination(res)
#                error_list.append(False)
#            except (ValueError, AssertionError, RuntimeError):
#                error_list.append(True)
#            output_list.append(tuple(var.value for var in variables))
#    for i, (inputs, outputs) in enumerate(zip(input_list, output_list)):
#        print(inputs, outputs, error_list[i])
#    for outputs in output_list:
#        print("(%1.6f, %1.6f, %1.6f)," % outputs)


if __name__ == "__main__":
    unittest.main()
    #_solve_with_ipopt()
