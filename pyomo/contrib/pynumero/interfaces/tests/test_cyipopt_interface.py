#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
import pyomo.environ as pyo

from pyomo.contrib.pynumero.dependencies import (
    numpy as np,
    numpy_available,
    scipy_available,
)

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run CyIpopt tests")

from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
from pyomo.contrib.pynumero.asl import AmplInterface

if not AmplInterface.available():
    raise unittest.SkipTest("Pynumero needs the ASL extension to run CyIpopt tests")

from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import (
    cyipopt,
    cyipopt_available,
    CyIpoptProblemInterface,
    CyIpoptNLP,
)

if not cyipopt_available:
    raise unittest.SkipTest("CyIpopt is not available")


cyipopt_ge_1_3 = hasattr(cyipopt, "CyIpoptEvaluationError")


class TestSubclassCyIpoptInterface(unittest.TestCase):
    def test_subclass_no_init(self):
        class MyCyIpoptProblem(CyIpoptProblemInterface):
            def __init__(self):
                # This subclass implements __init__ but does not call
                # super().__init__
                pass

            def x_init(self):
                pass

            def x_lb(self):
                pass

            def x_ub(self):
                pass

            def g_lb(self):
                pass

            def g_ub(self):
                pass

            def scaling_factors(self):
                pass

            def objective(self, x):
                pass

            def gradient(self, x):
                pass

            def constraints(self, x):
                pass

            def jacobianstructure(self):
                pass

            def jacobian(self, x):
                pass

            def hessianstructure(self):
                pass

            def hessian(self, x, y, obj_factor):
                pass

        problem = MyCyIpoptProblem()
        x0 = []
        msg = "__init__ has not been called"
        with self.assertRaisesRegex(RuntimeError, msg):
            problem.solve(x0)


def _get_model_nlp_interface(halt_on_evaluation_error=None, intermediate_callback=None):
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3], initialize=1.0)
    m.obj = pyo.Objective(expr=m.x[1] * pyo.sqrt(m.x[2]) + m.x[1] * m.x[3])
    m.eq1 = pyo.Constraint(expr=m.x[1] * pyo.sqrt(m.x[2]) == 1.0)
    nlp = PyomoNLP(m)
    interface = CyIpoptNLP(
        nlp,
        halt_on_evaluation_error=halt_on_evaluation_error,
        intermediate_callback=intermediate_callback,
    )
    bad_primals = np.array([1.0, -2.0, 3.0])
    indices = nlp.get_primal_indices([m.x[1], m.x[2], m.x[3]])
    bad_primals = bad_primals[indices]
    return m, nlp, interface, bad_primals


class TestCyIpoptVersionDependentConfig(unittest.TestCase):
    @unittest.skipIf(cyipopt_ge_1_3, "cyipopt version >= 1.3.0")
    def test_config_error(self):
        _, nlp, _, _ = _get_model_nlp_interface()
        with self.assertRaisesRegex(ValueError, "halt_on_evaluation_error"):
            interface = CyIpoptNLP(nlp, halt_on_evaluation_error=False)

    @unittest.skipIf(cyipopt_ge_1_3, "cyipopt version >= 1.3.0")
    def test_default_config_with_old_cyipopt(self):
        _, nlp, _, bad_x = _get_model_nlp_interface()
        interface = CyIpoptNLP(nlp)
        msg = "Error in AMPL evaluation"
        with self.assertRaisesRegex(PyNumeroEvaluationError, msg):
            interface.objective(bad_x)

    @unittest.skipIf(not cyipopt_ge_1_3, "cyipopt version < 1.3.0")
    def test_default_config_with_new_cyipopt(self):
        _, nlp, _, bad_x = _get_model_nlp_interface()
        interface = CyIpoptNLP(nlp)
        msg = "Error in objective function"
        with self.assertRaisesRegex(cyipopt.CyIpoptEvaluationError, msg):
            interface.objective(bad_x)


class TestCyIpoptEvaluationErrors(unittest.TestCase):
    @unittest.skipUnless(cyipopt_ge_1_3, "cyipopt version < 1.3.0")
    def test_error_in_objective(self):
        m, nlp, interface, bad_x = _get_model_nlp_interface(
            halt_on_evaluation_error=False
        )
        msg = "Error in objective function"
        with self.assertRaisesRegex(cyipopt.CyIpoptEvaluationError, msg):
            interface.objective(bad_x)

    def test_error_in_objective_halt(self):
        m, nlp, interface, bad_x = _get_model_nlp_interface(
            halt_on_evaluation_error=True
        )
        msg = "Error in AMPL evaluation"
        with self.assertRaisesRegex(PyNumeroEvaluationError, msg):
            interface.objective(bad_x)

    @unittest.skipUnless(cyipopt_ge_1_3, "cyipopt version < 1.3.0")
    def test_error_in_gradient(self):
        m, nlp, interface, bad_x = _get_model_nlp_interface(
            halt_on_evaluation_error=False
        )
        msg = "Error in objective gradient"
        with self.assertRaisesRegex(cyipopt.CyIpoptEvaluationError, msg):
            interface.gradient(bad_x)

    def test_error_in_gradient_halt(self):
        m, nlp, interface, bad_x = _get_model_nlp_interface(
            halt_on_evaluation_error=True
        )
        msg = "Error in AMPL evaluation"
        with self.assertRaisesRegex(PyNumeroEvaluationError, msg):
            interface.gradient(bad_x)

    @unittest.skipUnless(cyipopt_ge_1_3, "cyipopt version < 1.3.0")
    def test_error_in_constraints(self):
        m, nlp, interface, bad_x = _get_model_nlp_interface(
            halt_on_evaluation_error=False
        )
        msg = "Error in constraint evaluation"
        with self.assertRaisesRegex(cyipopt.CyIpoptEvaluationError, msg):
            interface.constraints(bad_x)

    def test_error_in_constraints_halt(self):
        m, nlp, interface, bad_x = _get_model_nlp_interface(
            halt_on_evaluation_error=True
        )
        msg = "Error in AMPL evaluation"
        with self.assertRaisesRegex(PyNumeroEvaluationError, msg):
            interface.constraints(bad_x)

    @unittest.skipUnless(cyipopt_ge_1_3, "cyipopt version < 1.3.0")
    def test_error_in_jacobian(self):
        m, nlp, interface, bad_x = _get_model_nlp_interface(
            halt_on_evaluation_error=False
        )
        msg = "Error in constraint Jacobian"
        with self.assertRaisesRegex(cyipopt.CyIpoptEvaluationError, msg):
            interface.jacobian(bad_x)

    def test_error_in_jacobian_halt(self):
        m, nlp, interface, bad_x = _get_model_nlp_interface(
            halt_on_evaluation_error=True
        )
        msg = "Error in AMPL evaluation"
        with self.assertRaisesRegex(PyNumeroEvaluationError, msg):
            interface.jacobian(bad_x)

    @unittest.skipUnless(cyipopt_ge_1_3, "cyipopt version < 1.3.0")
    def test_error_in_hessian(self):
        m, nlp, interface, bad_x = _get_model_nlp_interface(
            halt_on_evaluation_error=False
        )
        msg = "Error in Lagrangian Hessian"
        with self.assertRaisesRegex(cyipopt.CyIpoptEvaluationError, msg):
            interface.hessian(bad_x, [1.0], 0.0)

    def test_error_in_hessian_halt(self):
        m, nlp, interface, bad_x = _get_model_nlp_interface(
            halt_on_evaluation_error=True
        )
        msg = "Error in AMPL evaluation"
        with self.assertRaisesRegex(PyNumeroEvaluationError, msg):
            interface.hessian(bad_x, [1.0], 0.0)

    def test_intermediate_12arg(self):
        iterate_data = []

        def intermediate(
            nlp,
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
            self.assertIsInstance(nlp, PyomoNLP)
            iterate_data.append((inf_pr, inf_du))

        m, nlp, interface, bad_x = _get_model_nlp_interface(
            intermediate_callback=intermediate
        )
        # The interface's callback is always called with 11 arguments (by CyIpopt/Ipopt)
        # but we add the NLP object to the arguments.
        interface.intermediate(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
        self.assertEqual(iterate_data, [(4, 5)])

    def test_intermediate_13arg(self):
        iterate_data = []

        def intermediate(
            nlp,
            problem,
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
            self.assertIsInstance(nlp, PyomoNLP)
            self.assertIsInstance(problem, cyipopt.Problem)
            iterate_data.append((inf_pr, inf_du))

        m, nlp, interface, bad_x = _get_model_nlp_interface(
            intermediate_callback=intermediate
        )
        # The interface's callback is always called with 11 arguments (by CyIpopt/Ipopt)
        # but we add the NLP object *and the cyipopt.Problem object* to the arguments.
        interface.intermediate(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
        self.assertEqual(iterate_data, [(4, 5)])


if __name__ == "__main__":
    unittest.main()
