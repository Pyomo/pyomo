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
from pyomo.common.tempfiles import TempfileManager

from pyomo.contrib.pynumero.dependencies import (
    numpy as np,
    numpy_available,
    scipy,
    scipy_available,
)
from pyomo.common.dependencies.scipy import sparse as spa

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.contrib.pynumero.asl import AmplInterface

if not AmplInterface.available():
    raise unittest.SkipTest("Pynumero needs the ASL extension to run cyipopt tests")

from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available

from pyomo.contrib.pynumero.interfaces.external_grey_box import (
    ExternalGreyBoxModel,
    ExternalGreyBoxBlock,
)
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoGreyBoxNLP
from pyomo.contrib.pynumero.interfaces.tests.compare_utils import (
    check_vectors_specific_order,
    check_sparse_matrix_specific_order,
)
import pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models as ex_models


class TestExternalGreyBoxModel(unittest.TestCase):
    def test_pressure_drop_single_output(self):
        egbm = ex_models.PressureDropSingleOutput()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual(eq_con_names, [])
        output_names = egbm.output_names()
        self.assertEqual(output_names, ['Pout'])

        egbm.set_input_values(np.asarray([100, 2, 3], dtype=np.float64))
        egbm.set_equality_constraint_multipliers(np.asarray([], dtype=np.float64))
        with self.assertRaises(AssertionError):
            egbm.set_equality_constraint_multipliers(np.asarray([1], dtype=np.float64))
        egbm.set_output_constraint_multipliers(np.asarray([5], dtype=np.float64))

        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_equality_constraints()

        o = egbm.evaluate_outputs()
        self.assertTrue(np.array_equal(o, np.asarray([28], dtype=np.float64)))

        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_jacobian_equality_constraints()

        jac_o = egbm.evaluate_jacobian_outputs()
        self.assertTrue(
            np.array_equal(jac_o.row, np.asarray([0, 0, 0], dtype=np.int64))
        )
        self.assertTrue(
            np.array_equal(jac_o.col, np.asarray([0, 1, 2], dtype=np.int64))
        )
        self.assertTrue(
            np.array_equal(jac_o.data, np.asarray([1, -36, -48], dtype=np.float64))
        )

        with self.assertRaises(AttributeError):
            eq_hess = egbm.evaluate_hessian_equality_constraints()

        with self.assertRaises(AttributeError):
            outputs_hess = egbm.evaluate_hessian_outputs()

    def test_pressure_drop_single_output_with_hessian(self):
        egbm = ex_models.PressureDropSingleOutputWithHessian()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual(eq_con_names, [])
        output_names = egbm.output_names()
        self.assertEqual(output_names, ['Pout'])

        egbm.set_input_values(np.asarray([100, 2, 3], dtype=np.float64))
        egbm.set_equality_constraint_multipliers(np.asarray([], dtype=np.float64))
        with self.assertRaises(AssertionError):
            egbm.set_equality_constraint_multipliers(np.asarray([1], dtype=np.float64))
        egbm.set_output_constraint_multipliers(np.asarray([5], dtype=np.float64))

        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_equality_constraints()

        o = egbm.evaluate_outputs()
        self.assertTrue(np.array_equal(o, np.asarray([28], dtype=np.float64)))

        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_jacobian_equality_constraints()

        jac_o = egbm.evaluate_jacobian_outputs()
        self.assertTrue(
            np.array_equal(jac_o.row, np.asarray([0, 0, 0], dtype=np.int64))
        )
        self.assertTrue(
            np.array_equal(jac_o.col, np.asarray([0, 1, 2], dtype=np.int64))
        )
        self.assertTrue(
            np.array_equal(jac_o.data, np.asarray([1, -36, -48], dtype=np.float64))
        )

        with self.assertRaises(AttributeError):
            eq_hess = egbm.evaluate_hessian_equality_constraints()
        outputs_hess = egbm.evaluate_hessian_outputs()
        self.assertTrue(
            np.array_equal(outputs_hess.row, np.asarray([2, 2], dtype=np.int64))
        )
        self.assertTrue(
            np.array_equal(outputs_hess.col, np.asarray([1, 2], dtype=np.int64))
        )
        self.assertTrue(
            np.array_equal(
                outputs_hess.data,
                np.asarray([5 * (-8 * 3), 5 * (-8 * 2)], dtype=np.int64),
            )
        )

    def test_pressure_drop_single_equality(self):
        egbm = ex_models.PressureDropSingleEquality()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F', 'Pout'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual(eq_con_names, ['pdrop'])
        output_names = egbm.output_names()
        self.assertEqual(output_names, [])

        egbm.set_input_values(np.asarray([100, 2, 3, 50], dtype=np.float64))
        egbm.set_equality_constraint_multipliers(np.asarray([5], dtype=np.float64))
        with self.assertRaises(AssertionError):
            egbm.set_output_constraint_multipliers(np.asarray([1], dtype=np.float64))

        eq = egbm.evaluate_equality_constraints()
        self.assertTrue(np.array_equal(eq, np.asarray([22], dtype=np.float64)))

        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_outputs()

        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_jacobian_outputs()

        jac_eq = egbm.evaluate_jacobian_equality_constraints()
        self.assertTrue(
            np.array_equal(jac_eq.row, np.asarray([0, 0, 0, 0], dtype=np.int64))
        )
        self.assertTrue(
            np.array_equal(jac_eq.col, np.asarray([0, 1, 2, 3], dtype=np.int64))
        )
        self.assertTrue(
            np.array_equal(jac_eq.data, np.asarray([-1, 36, 48, 1], dtype=np.float64))
        )

        with self.assertRaises(AttributeError):
            eq_hess = egbm.evaluate_hessian_equality_constraints()
        with self.assertRaises(AttributeError):
            outputs_hess = egbm.evaluate_hessian_outputs()

    def test_pressure_drop_single_equality_with_hessian(self):
        egbm = ex_models.PressureDropSingleEqualityWithHessian()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F', 'Pout'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual(eq_con_names, ['pdrop'])
        output_names = egbm.output_names()
        self.assertEqual(output_names, [])

        egbm.set_input_values(np.asarray([100, 2, 3, 50], dtype=np.float64))
        egbm.set_equality_constraint_multipliers(np.asarray([5], dtype=np.float64))
        with self.assertRaises(AssertionError):
            egbm.set_output_constraint_multipliers(np.asarray([1], dtype=np.float64))

        eq = egbm.evaluate_equality_constraints()
        self.assertTrue(np.array_equal(eq, np.asarray([22], dtype=np.float64)))

        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_outputs()

        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_jacobian_outputs()

        jac_eq = egbm.evaluate_jacobian_equality_constraints()
        self.assertTrue(
            np.array_equal(jac_eq.row, np.asarray([0, 0, 0, 0], dtype=np.int64))
        )
        self.assertTrue(
            np.array_equal(jac_eq.col, np.asarray([0, 1, 2, 3], dtype=np.int64))
        )
        self.assertTrue(
            np.array_equal(jac_eq.data, np.asarray([-1, 36, 48, 1], dtype=np.float64))
        )

        eq_hess = egbm.evaluate_hessian_equality_constraints()
        with self.assertRaises(AttributeError):
            outputs_hess = egbm.evaluate_hessian_outputs()
        self.assertTrue(np.array_equal(eq_hess.row, np.asarray([2, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(eq_hess.col, np.asarray([1, 2], dtype=np.int64)))
        self.assertTrue(
            np.array_equal(
                eq_hess.data, np.asarray([5 * (8 * 3), 5 * (8 * 2)], dtype=np.float64)
            )
        )

    def test_pressure_drop_two_outputs(self):
        egbm = ex_models.PressureDropTwoOutputs()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual([], eq_con_names)
        output_names = egbm.output_names()
        self.assertEqual(output_names, ['P2', 'Pout'])

        egbm.set_input_values(np.asarray([100, 2, 3], dtype=np.float64))
        egbm.set_equality_constraint_multipliers(np.asarray([], dtype=np.float64))
        # this one should fail
        with self.assertRaises(AssertionError):
            egbm.set_equality_constraint_multipliers(np.asarray([1], dtype=np.float64))

        egbm.set_output_constraint_multipliers(np.asarray([3.0, 5.0], dtype=np.float64))

        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_equality_constraints()

        #   u = [Pin, c, F]
        #   o = [P2, Pout]
        #   h_eq(u) = {empty}
        #   h_o(u) = [Pin - 2*c*F^2]
        #            [Pin - 4*c*F^2]

        o = egbm.evaluate_outputs()
        self.assertTrue(np.array_equal(o, np.asarray([64, 28], dtype=np.float64)))

        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_jacobian_equality_constraints()

        jac_o = egbm.evaluate_jacobian_outputs()
        self.assertTrue(
            np.array_equal(jac_o.row, np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64))
        )
        self.assertTrue(
            np.array_equal(jac_o.col, np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64))
        )
        self.assertTrue(
            np.array_equal(
                jac_o.data, np.asarray([1, -18, -24, 1, -36, -48], dtype=np.float64)
            )
        )

        with self.assertRaises(AttributeError):
            hess_eq = egbm.evaluate_hessian_equality_constraints()
        with self.assertRaises(AttributeError):
            hess_outputs = egbm.evaluate_hessian_outputs()

    def test_pressure_drop_two_outputs_with_hessian(self):
        egbm = ex_models.PressureDropTwoOutputsWithHessian()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual([], eq_con_names)
        output_names = egbm.output_names()
        self.assertEqual(output_names, ['P2', 'Pout'])

        egbm.set_input_values(np.asarray([100, 2, 3], dtype=np.float64))
        egbm.set_equality_constraint_multipliers(np.asarray([], dtype=np.float64))
        # this one should fail
        with self.assertRaises(AssertionError):
            egbm.set_equality_constraint_multipliers(np.asarray([1], dtype=np.float64))

        egbm.set_output_constraint_multipliers(np.asarray([3.0, 5.0], dtype=np.float64))

        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_equality_constraints()

        #   u = [Pin, c, F]
        #   o = [P2, Pout]
        #   h_eq(u) = {empty}
        #   h_o(u) = [Pin - 2*c*F^2]
        #            [Pin - 4*c*F^2]

        o = egbm.evaluate_outputs()
        self.assertTrue(np.array_equal(o, np.asarray([64, 28], dtype=np.float64)))

        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_jacobian_equality_constraints()

        jac_o = egbm.evaluate_jacobian_outputs()
        self.assertTrue(
            np.array_equal(jac_o.row, np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64))
        )
        self.assertTrue(
            np.array_equal(jac_o.col, np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64))
        )
        self.assertTrue(
            np.array_equal(
                jac_o.data, np.asarray([1, -18, -24, 1, -36, -48], dtype=np.float64)
            )
        )

        with self.assertRaises(AttributeError):
            hess_eq = egbm.evaluate_hessian_equality_constraints()
        hess = egbm.evaluate_hessian_outputs()
        self.assertTrue(np.array_equal(hess.row, np.asarray([2, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(hess.col, np.asarray([1, 2], dtype=np.int64)))
        self.assertTrue(
            np.array_equal(hess.data, np.asarray([-156.0, -104.0], dtype=np.float64))
        )

    def test_pressure_drop_two_equalities(self):
        egbm = ex_models.PressureDropTwoEqualities()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F', 'P2', 'Pout'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual(eq_con_names, ['pdrop2', 'pdropout'])
        output_names = egbm.output_names()
        self.assertEqual([], output_names)

        egbm.set_input_values(np.asarray([100, 2, 3, 20, 50], dtype=np.float64))
        egbm.set_equality_constraint_multipliers(np.asarray([3, 5], dtype=np.float64))
        egbm.set_output_constraint_multipliers(np.asarray([]))
        # this one should fail
        with self.assertRaises(AssertionError):
            egbm.set_output_constraint_multipliers(np.asarray([1], dtype=np.float64))

        #   u = [Pin, c, F, P2, Pout]
        #   o = {empty}
        #   h_eq(u) = [P2 - (Pin - 2*c*F^2]
        #             [Pout - (P2 - 2*c*F^2]
        #   h_o(u) = {empty}
        eq = egbm.evaluate_equality_constraints()
        self.assertTrue(np.array_equal(eq, np.asarray([-44, 66], dtype=np.float64)))

        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_outputs()

        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_jacobian_outputs()

        jac_eq = egbm.evaluate_jacobian_equality_constraints()
        self.assertTrue(
            np.array_equal(
                jac_eq.row, np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
            )
        )
        self.assertTrue(
            np.array_equal(
                jac_eq.col, np.asarray([0, 1, 2, 3, 1, 2, 3, 4], dtype=np.int64)
            )
        )
        self.assertTrue(
            np.array_equal(
                jac_eq.data,
                np.asarray([-1, 18, 24, 1, 18, 24, -1, 1], dtype=np.float64),
            )
        )

        with self.assertRaises(AttributeError):
            hess_outputs = egbm.evaluate_hessian_outputs()
        with self.assertRaises(AttributeError):
            hess = egbm.evaluate_hessian_equality_constraints()

    def test_pressure_drop_two_equalities_with_hessian(self):
        egbm = ex_models.PressureDropTwoEqualitiesWithHessian()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F', 'P2', 'Pout'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual(eq_con_names, ['pdrop2', 'pdropout'])
        output_names = egbm.output_names()
        self.assertEqual([], output_names)

        egbm.set_input_values(np.asarray([100, 2, 3, 20, 50], dtype=np.float64))
        egbm.set_equality_constraint_multipliers(np.asarray([3, 5], dtype=np.float64))
        egbm.set_output_constraint_multipliers(np.asarray([]))
        # this one should fail
        with self.assertRaises(AssertionError):
            egbm.set_output_constraint_multipliers(np.asarray([1], dtype=np.float64))

        #   u = [Pin, c, F, P2, Pout]
        #   o = {empty}
        #   h_eq(u) = [P2 - (Pin - 2*c*F^2]
        #             [Pout - (P2 - 2*c*F^2]
        #   h_o(u) = {empty}
        eq = egbm.evaluate_equality_constraints()
        self.assertTrue(np.array_equal(eq, np.asarray([-44, 66], dtype=np.float64)))

        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_outputs()

        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_jacobian_outputs()

        jac_eq = egbm.evaluate_jacobian_equality_constraints()
        self.assertTrue(
            np.array_equal(
                jac_eq.row, np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
            )
        )
        self.assertTrue(
            np.array_equal(
                jac_eq.col, np.asarray([0, 1, 2, 3, 1, 2, 3, 4], dtype=np.int64)
            )
        )
        self.assertTrue(
            np.array_equal(
                jac_eq.data,
                np.asarray([-1, 18, 24, 1, 18, 24, -1, 1], dtype=np.float64),
            )
        )

        with self.assertRaises(AttributeError):
            hess_outputs = egbm.evaluate_hessian_outputs()
        hess = egbm.evaluate_hessian_equality_constraints()
        self.assertTrue(np.array_equal(hess.row, np.asarray([2, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(hess.col, np.asarray([1, 2], dtype=np.int64)))
        self.assertTrue(
            np.array_equal(hess.data, np.asarray([96.0, 64.0], dtype=np.float64))
        )

    def test_pressure_drop_two_equalities_two_outputs(self):
        #   u = [Pin, c, F, P1, P3]
        #   o = {P2, Pout}
        #   h_eq(u) = [P1 - (Pin - c*F^2]
        #             [P3 - (Pin - 2*c*F^2]
        #   h_o(u) = [P1 - c*F^2]
        #            [Pin - 4*c*F^2]
        egbm = ex_models.PressureDropTwoEqualitiesTwoOutputs()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F', 'P1', 'P3'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual(eq_con_names, ['pdrop1', 'pdrop3'])
        output_names = egbm.output_names()
        self.assertEqual(output_names, ['P2', 'Pout'])

        egbm.set_input_values(np.asarray([100, 2, 3, 80, 70], dtype=np.float64))
        egbm.set_equality_constraint_multipliers(np.asarray([2, 4], dtype=np.float64))
        egbm.set_output_constraint_multipliers(np.asarray([7, 9], dtype=np.float64))
        eq = egbm.evaluate_equality_constraints()
        self.assertTrue(np.array_equal(eq, np.asarray([-2, 26], dtype=np.float64)))

        o = egbm.evaluate_outputs()
        self.assertTrue(np.array_equal(o, np.asarray([62, 28], dtype=np.float64)))

        jac_eq = egbm.evaluate_jacobian_equality_constraints()
        self.assertTrue(
            np.array_equal(
                jac_eq.row, np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
            )
        )
        self.assertTrue(
            np.array_equal(
                jac_eq.col, np.asarray([0, 1, 2, 3, 1, 2, 3, 4], dtype=np.int64)
            )
        )
        self.assertTrue(
            np.array_equal(
                jac_eq.data, np.asarray([-1, 9, 12, 1, 18, 24, -1, 1], dtype=np.float64)
            )
        )

        jac_o = egbm.evaluate_jacobian_outputs()
        self.assertTrue(
            np.array_equal(jac_o.row, np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64))
        )
        self.assertTrue(
            np.array_equal(jac_o.col, np.asarray([1, 2, 3, 0, 1, 2], dtype=np.int64))
        )
        self.assertTrue(
            np.array_equal(
                jac_o.data, np.asarray([-9, -12, 1, 1, -36, -48], dtype=np.float64)
            )
        )

        with self.assertRaises(AttributeError):
            hess = egbm.evaluate_hessian_equality_constraints()
        with self.assertRaises(AttributeError):
            hess = egbm.evaluate_hessian_outputs()

    def test_pressure_drop_two_equalities_two_outputs_with_hessian(self):
        #   u = [Pin, c, F, P1, P3]
        #   o = {P2, Pout}
        #   h_eq(u) = [P1 - (Pin - c*F^2]
        #             [P3 - (Pin - 2*c*F^2]
        #   h_o(u) = [P1 - c*F^2]
        #            [Pin - 4*c*F^2]
        egbm = ex_models.PressureDropTwoEqualitiesTwoOutputsWithHessian()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F', 'P1', 'P3'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual(eq_con_names, ['pdrop1', 'pdrop3'])
        output_names = egbm.output_names()
        self.assertEqual(output_names, ['P2', 'Pout'])

        egbm.set_input_values(np.asarray([100, 2, 3, 80, 70], dtype=np.float64))
        egbm.set_equality_constraint_multipliers(np.asarray([2, 4], dtype=np.float64))
        egbm.set_output_constraint_multipliers(np.asarray([7, 9], dtype=np.float64))
        eq = egbm.evaluate_equality_constraints()
        self.assertTrue(np.array_equal(eq, np.asarray([-2, 26], dtype=np.float64)))

        o = egbm.evaluate_outputs()
        self.assertTrue(np.array_equal(o, np.asarray([62, 28], dtype=np.float64)))

        jac_eq = egbm.evaluate_jacobian_equality_constraints()
        self.assertTrue(
            np.array_equal(
                jac_eq.row, np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
            )
        )
        self.assertTrue(
            np.array_equal(
                jac_eq.col, np.asarray([0, 1, 2, 3, 1, 2, 3, 4], dtype=np.int64)
            )
        )
        self.assertTrue(
            np.array_equal(
                jac_eq.data, np.asarray([-1, 9, 12, 1, 18, 24, -1, 1], dtype=np.float64)
            )
        )

        jac_o = egbm.evaluate_jacobian_outputs()
        self.assertTrue(
            np.array_equal(jac_o.row, np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64))
        )
        self.assertTrue(
            np.array_equal(jac_o.col, np.asarray([1, 2, 3, 0, 1, 2], dtype=np.int64))
        )
        self.assertTrue(
            np.array_equal(
                jac_o.data, np.asarray([-9, -12, 1, 1, -36, -48], dtype=np.float64)
            )
        )

        hess = egbm.evaluate_hessian_equality_constraints()
        self.assertTrue(np.array_equal(hess.row, np.asarray([2, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(hess.col, np.asarray([1, 2], dtype=np.int64)))
        self.assertTrue(
            np.array_equal(hess.data, np.asarray([60.0, 40.0], dtype=np.float64))
        )

        hess = egbm.evaluate_hessian_outputs()
        self.assertTrue(np.array_equal(hess.row, np.asarray([2, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(hess.col, np.asarray([1, 2], dtype=np.int64)))
        self.assertTrue(
            np.array_equal(hess.data, np.asarray([-258, -172], dtype=np.float64))
        )


"""
    def test_pressure_drop_two_equalities_two_outputs_no_hessian(self):
        #   u = [Pin, c, F, P1, P3]
        #   o = {P2, Pout}
        #   h_eq(u) = [P1 - (Pin - c*F^2]
        #             [P3 - (Pin - 2*c*F^2]
        #   h_o(u) = [P1 - c*F^2]
        #            [Pin - 4*c*F^2]
        egbm = ex_models.PressureDropTwoEqualitiesTwoOutputsNoHessian()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F', 'P1', 'P3'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual(eq_con_names, ['pdrop1', 'pdrop3'])
        output_names = egbm.output_names()
        self.assertEqual(output_names, ['P2', 'Pout'])

        egbm.set_input_values(np.asarray([100, 2, 3, 80, 70], dtype=np.float64))
        with self.assertRaises(NotImplementedError):
            egbm.set_equality_constraint_multipliers(np.asarray([2, 4], dtype=np.float64))
        with self.assertRaises(NotImplementedError):
            egbm.set_output_constraint_multipliers(np.asarray([7, 9], dtype=np.float64))
            
        eq = egbm.evaluate_equality_constraints()
        self.assertTrue(np.array_equal(eq, np.asarray([-2, 26], dtype=np.float64)))

        o = egbm.evaluate_outputs()
        self.assertTrue(np.array_equal(o, np.asarray([62, 28], dtype=np.float64)))

        jac_eq = egbm.evaluate_jacobian_equality_constraints()
        self.assertTrue(np.array_equal(jac_eq.row, np.asarray([0,0,0,0,1,1,1,1], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_eq.col, np.asarray([0,1,2,3,1,2,3,4], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_eq.data, np.asarray([-1, 9, 12, 1, 18, 24, -1, 1], dtype=np.float64)))

        jac_o = egbm.evaluate_jacobian_outputs()
        self.assertTrue(np.array_equal(jac_o.row, np.asarray([0,0,0,1,1,1], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_o.col, np.asarray([1,2,3,0,1,2], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_o.data, np.asarray([-9, -12, 1, 1, -36, -48], dtype=np.float64)))

        with self.assertRaises(NotImplementedError):
            hess = egbm.evaluate_hessian_equality_constraints()

        with self.assertRaises(NotImplementedError):
            hess = egbm.evaluate_hessian_outputs()
"""


# TODO: make this work even if there is only external and no variables anywhere in pyomo part
class TestPyomoGreyBoxNLP(unittest.TestCase):
    def test_error_no_variables(self):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(ex_models.PressureDropSingleOutput())
        m.obj = pyo.Objective(expr=1)
        with self.assertRaises(ValueError):
            pyomo_nlp = PyomoGreyBoxNLP(m)

    def test_error_fixed_inputs_outputs(self):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(ex_models.PressureDropSingleOutput())
        m.egb.inputs['Pin'].fix(100)
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout'] - 20) ** 2)
        with self.assertRaises(NotImplementedError):
            pyomo_nlp = PyomoGreyBoxNLP(m)

        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(ex_models.PressureDropTwoOutputs())
        m.egb.outputs['P2'].fix(50)
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout'] - 20) ** 2)
        with self.assertRaises(NotImplementedError):
            pyomo_nlp = PyomoGreyBoxNLP(m)

    def test_pressure_drop_single_output(self):
        self._test_pressure_drop_single_output(
            ex_models.PressureDropSingleOutput(), False
        )
        self._test_pressure_drop_single_output(
            ex_models.PressureDropSingleOutputWithHessian(), True
        )

    def _test_pressure_drop_single_output(self, ex_model, hessian_support):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(ex_model)
        m.egb.inputs['Pin'].value = 100
        m.egb.inputs['Pin'].setlb(50)
        m.egb.inputs['Pin'].setub(150)
        m.egb.inputs['c'].value = 2
        m.egb.inputs['c'].setlb(1)
        m.egb.inputs['c'].setub(5)
        m.egb.inputs['F'].value = 3
        m.egb.inputs['F'].setlb(1)
        m.egb.inputs['F'].setub(5)
        m.egb.outputs['Pout'].value = 50
        m.egb.outputs['Pout'].setlb(0)
        m.egb.outputs['Pout'].setub(100)
        # m.dummy = pyo.Constraint(expr=sum(m.egb.inputs[i] for i in m.egb.inputs) + sum(m.egb.outputs[i] for i in m.egb.outputs) <= 1e6)
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout'] - 20) ** 2)

        pyomo_nlp = PyomoGreyBoxNLP(m)

        self.assertEqual(4, pyomo_nlp.n_primals())
        self.assertEqual(1, pyomo_nlp.n_constraints())
        self.assertEqual(4, pyomo_nlp.nnz_jacobian())
        if hessian_support:
            self.assertEqual(3, pyomo_nlp.nnz_hessian_lag())

        comparison_x_order = [
            'egb.inputs[Pin]',
            'egb.inputs[c]',
            'egb.inputs[F]',
            'egb.outputs[Pout]',
        ]
        x_order = pyomo_nlp.variable_names()
        comparison_c_order = ['egb.Pout_con']
        c_order = pyomo_nlp.constraint_names()

        xlb = pyomo_nlp.primals_lb()
        comparison_xlb = np.asarray([50, 1, 1, 0], dtype=np.float64)
        check_vectors_specific_order(
            self, xlb, x_order, comparison_xlb, comparison_x_order
        )
        xub = pyomo_nlp.primals_ub()
        comparison_xub = np.asarray([150, 5, 5, 100], dtype=np.float64)
        check_vectors_specific_order(
            self, xub, x_order, comparison_xub, comparison_x_order
        )
        clb = pyomo_nlp.constraints_lb()
        comparison_clb = np.asarray([0], dtype=np.float64)
        check_vectors_specific_order(
            self, clb, c_order, comparison_clb, comparison_c_order
        )
        cub = pyomo_nlp.constraints_ub()
        comparison_cub = np.asarray([0], dtype=np.float64)
        check_vectors_specific_order(
            self, cub, c_order, comparison_cub, comparison_c_order
        )

        xinit = pyomo_nlp.init_primals()
        comparison_xinit = np.asarray([100, 2, 3, 50], dtype=np.float64)
        check_vectors_specific_order(
            self, xinit, x_order, comparison_xinit, comparison_x_order
        )
        duals_init = pyomo_nlp.init_duals()
        comparison_duals_init = np.asarray([0], dtype=np.float64)
        check_vectors_specific_order(
            self, duals_init, c_order, comparison_duals_init, comparison_c_order
        )

        self.assertEqual(4, len(pyomo_nlp.create_new_vector('primals')))
        self.assertEqual(1, len(pyomo_nlp.create_new_vector('constraints')))
        self.assertEqual(1, len(pyomo_nlp.create_new_vector('duals')))
        self.assertEqual(1, len(pyomo_nlp.create_new_vector('eq_constraints')))
        self.assertEqual(0, len(pyomo_nlp.create_new_vector('ineq_constraints')))
        self.assertEqual(1, len(pyomo_nlp.create_new_vector('duals_eq')))
        self.assertEqual(0, len(pyomo_nlp.create_new_vector('duals_ineq')))

        pyomo_nlp.set_primals(np.asarray([1, 2, 3, 4], dtype=np.float64))
        x = pyomo_nlp.get_primals()
        self.assertTrue(np.array_equal(x, np.asarray([1, 2, 3, 4], dtype=np.float64)))
        pyomo_nlp.set_primals(pyomo_nlp.init_primals())

        pyomo_nlp.set_duals(np.asarray([42], dtype=np.float64))
        y = pyomo_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([42], dtype=np.float64)))
        pyomo_nlp.set_duals(np.asarray([21], dtype=np.float64))
        y = pyomo_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([21], dtype=np.float64)))

        fac = pyomo_nlp.get_obj_factor()
        self.assertEqual(fac, 1)
        pyomo_nlp.set_obj_factor(42)
        self.assertEqual(pyomo_nlp.get_obj_factor(), 42)
        pyomo_nlp.set_obj_factor(1)

        f = pyomo_nlp.evaluate_objective()
        self.assertEqual(f, 900)

        gradf = pyomo_nlp.evaluate_grad_objective()
        comparison_gradf = np.asarray([0, 0, 0, 60], dtype=np.float64)
        check_vectors_specific_order(
            self, gradf, x_order, comparison_gradf, comparison_x_order
        )
        c = pyomo_nlp.evaluate_constraints()
        comparison_c = np.asarray([-22], dtype=np.float64)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        c = np.zeros(1)
        pyomo_nlp.evaluate_constraints(out=c)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)

        j = pyomo_nlp.evaluate_jacobian()
        comparison_j = np.asarray([[1, -36, -48, -1]])
        check_sparse_matrix_specific_order(
            self,
            j,
            c_order,
            x_order,
            comparison_j,
            comparison_c_order,
            comparison_x_order,
        )

        j = 2.0 * j
        pyomo_nlp.evaluate_jacobian(out=j)
        check_sparse_matrix_specific_order(
            self,
            j,
            c_order,
            x_order,
            comparison_j,
            comparison_c_order,
            comparison_x_order,
        )

        if hessian_support:
            h = pyomo_nlp.evaluate_hessian_lag()
            self.assertTrue(h.shape == (4, 4))
            comparison_h = np.asarray(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, -8 * 3 * 21, -8 * 2 * 21, 0],
                    [0, 0, 0, 2 * 1],
                ],
                dtype=np.float64,
            )
            check_sparse_matrix_specific_order(
                self,
                h,
                x_order,
                x_order,
                comparison_h,
                comparison_x_order,
                comparison_x_order,
            )
        else:
            with self.assertRaises(AttributeError):
                h = pyomo_nlp.evaluate_hessian_lag()

    def test_pressure_drop_single_equality(self):
        self._test_pressure_drop_single_equality(
            ex_models.PressureDropSingleEquality(), False
        )
        self._test_pressure_drop_single_equality(
            ex_models.PressureDropSingleEqualityWithHessian(), True
        )

    def _test_pressure_drop_single_equality(self, ex_model, hessian_support):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(ex_model)
        m.egb.inputs['Pin'].value = 100
        m.egb.inputs['Pin'].setlb(50)
        m.egb.inputs['Pin'].setub(150)
        m.egb.inputs['c'].value = 2
        m.egb.inputs['c'].setlb(1)
        m.egb.inputs['c'].setub(5)
        m.egb.inputs['F'].value = 3
        m.egb.inputs['F'].setlb(1)
        m.egb.inputs['F'].setub(5)
        m.egb.inputs['Pout'].value = 50
        m.egb.inputs['Pout'].setlb(0)
        m.egb.inputs['Pout'].setub(100)
        m.obj = pyo.Objective(expr=(m.egb.inputs['Pout'] - 20) ** 2)
        pyomo_nlp = PyomoGreyBoxNLP(m)

        self.assertEqual(4, pyomo_nlp.n_primals())
        self.assertEqual(1, pyomo_nlp.n_constraints())
        self.assertEqual(4, pyomo_nlp.nnz_jacobian())
        if hessian_support:
            self.assertEqual(3, pyomo_nlp.nnz_hessian_lag())

        comparison_x_order = [
            'egb.inputs[Pin]',
            'egb.inputs[c]',
            'egb.inputs[F]',
            'egb.inputs[Pout]',
        ]
        x_order = pyomo_nlp.variable_names()
        comparison_c_order = ['egb.pdrop']
        c_order = pyomo_nlp.constraint_names()

        xlb = pyomo_nlp.primals_lb()
        comparison_xlb = np.asarray([50, 1, 1, 0], dtype=np.float64)
        check_vectors_specific_order(
            self, xlb, x_order, comparison_xlb, comparison_x_order
        )
        xub = pyomo_nlp.primals_ub()
        comparison_xub = np.asarray([150, 5, 5, 100], dtype=np.float64)
        check_vectors_specific_order(
            self, xub, x_order, comparison_xub, comparison_x_order
        )
        clb = pyomo_nlp.constraints_lb()
        comparison_clb = np.asarray([0], dtype=np.float64)
        check_vectors_specific_order(
            self, clb, c_order, comparison_clb, comparison_c_order
        )
        cub = pyomo_nlp.constraints_ub()
        comparison_cub = np.asarray([0], dtype=np.float64)
        check_vectors_specific_order(
            self, cub, c_order, comparison_cub, comparison_c_order
        )

        xinit = pyomo_nlp.init_primals()
        comparison_xinit = np.asarray([100, 2, 3, 50], dtype=np.float64)
        check_vectors_specific_order(
            self, xinit, x_order, comparison_xinit, comparison_x_order
        )
        duals_init = pyomo_nlp.init_duals()
        comparison_duals_init = np.asarray([0], dtype=np.float64)
        check_vectors_specific_order(
            self, duals_init, c_order, comparison_duals_init, comparison_c_order
        )

        self.assertEqual(4, len(pyomo_nlp.create_new_vector('primals')))
        self.assertEqual(1, len(pyomo_nlp.create_new_vector('constraints')))
        self.assertEqual(1, len(pyomo_nlp.create_new_vector('duals')))
        self.assertEqual(1, len(pyomo_nlp.create_new_vector('eq_constraints')))
        self.assertEqual(0, len(pyomo_nlp.create_new_vector('ineq_constraints')))
        self.assertEqual(1, len(pyomo_nlp.create_new_vector('duals_eq')))
        self.assertEqual(0, len(pyomo_nlp.create_new_vector('duals_ineq')))

        pyomo_nlp.set_primals(np.asarray([1, 2, 3, 4], dtype=np.float64))
        x = pyomo_nlp.get_primals()
        self.assertTrue(np.array_equal(x, np.asarray([1, 2, 3, 4], dtype=np.float64)))
        pyomo_nlp.set_primals(pyomo_nlp.init_primals())

        pyomo_nlp.set_duals(np.asarray([42], dtype=np.float64))
        y = pyomo_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([42], dtype=np.float64)))
        pyomo_nlp.set_duals(np.asarray([21], dtype=np.float64))
        y = pyomo_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([21], dtype=np.float64)))

        fac = pyomo_nlp.get_obj_factor()
        self.assertEqual(fac, 1)
        pyomo_nlp.set_obj_factor(42)
        self.assertEqual(pyomo_nlp.get_obj_factor(), 42)
        pyomo_nlp.set_obj_factor(1)

        f = pyomo_nlp.evaluate_objective()
        self.assertEqual(f, 900)

        gradf = pyomo_nlp.evaluate_grad_objective()
        comparison_gradf = np.asarray([0, 0, 0, 60], dtype=np.float64)
        check_vectors_specific_order(
            self, gradf, x_order, comparison_gradf, comparison_x_order
        )
        c = pyomo_nlp.evaluate_constraints()
        comparison_c = np.asarray([22], dtype=np.float64)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        c = np.zeros(1)
        pyomo_nlp.evaluate_constraints(out=c)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)

        j = pyomo_nlp.evaluate_jacobian()
        comparison_j = np.asarray([[-1, 36, 48, 1]])
        check_sparse_matrix_specific_order(
            self,
            j,
            c_order,
            x_order,
            comparison_j,
            comparison_c_order,
            comparison_x_order,
        )

        j = 2.0 * j
        pyomo_nlp.evaluate_jacobian(out=j)
        check_sparse_matrix_specific_order(
            self,
            j,
            c_order,
            x_order,
            comparison_j,
            comparison_c_order,
            comparison_x_order,
        )

        if hessian_support:
            h = pyomo_nlp.evaluate_hessian_lag()
            self.assertTrue(h.shape == (4, 4))
            comparison_h = np.asarray(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 8 * 3 * 21, 8 * 2 * 21, 0],
                    [0, 0, 0, 2 * 1],
                ],
                dtype=np.float64,
            )
            check_sparse_matrix_specific_order(
                self,
                h,
                x_order,
                x_order,
                comparison_h,
                comparison_x_order,
                comparison_x_order,
            )
        else:
            with self.assertRaises(AttributeError):
                h = pyomo_nlp.evaluate_hessian_lag()

    def test_pressure_drop_two_outputs(self):
        self._test_pressure_drop_two_outputs(ex_models.PressureDropTwoOutputs(), False)
        self._test_pressure_drop_two_outputs(
            ex_models.PressureDropTwoOutputsWithHessian(), True
        )

    def _test_pressure_drop_two_outputs(self, ex_model, hessian_support):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(ex_model)
        m.egb.inputs['Pin'].value = 100
        m.egb.inputs['Pin'].setlb(50)
        m.egb.inputs['Pin'].setub(150)
        m.egb.inputs['c'].value = 2
        m.egb.inputs['c'].setlb(1)
        m.egb.inputs['c'].setub(5)
        m.egb.inputs['F'].value = 3
        m.egb.inputs['F'].setlb(1)
        m.egb.inputs['F'].setub(5)
        m.egb.outputs['P2'].value = 80
        m.egb.outputs['P2'].setlb(10)
        m.egb.outputs['P2'].setub(90)
        m.egb.outputs['Pout'].value = 50
        m.egb.outputs['Pout'].setlb(0)
        m.egb.outputs['Pout'].setub(100)
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout'] - 20) ** 2)
        pyomo_nlp = PyomoGreyBoxNLP(m)

        self.assertEqual(5, pyomo_nlp.n_primals())
        self.assertEqual(2, pyomo_nlp.n_constraints())
        self.assertEqual(8, pyomo_nlp.nnz_jacobian())
        if hessian_support:
            self.assertEqual(3, pyomo_nlp.nnz_hessian_lag())

        comparison_x_order = [
            'egb.inputs[Pin]',
            'egb.inputs[c]',
            'egb.inputs[F]',
            'egb.outputs[P2]',
            'egb.outputs[Pout]',
        ]
        x_order = pyomo_nlp.variable_names()
        comparison_c_order = ['egb.P2_con', 'egb.Pout_con']
        c_order = pyomo_nlp.constraint_names()

        xlb = pyomo_nlp.primals_lb()
        comparison_xlb = np.asarray([50, 1, 1, 10, 0], dtype=np.float64)
        check_vectors_specific_order(
            self, xlb, x_order, comparison_xlb, comparison_x_order
        )
        xub = pyomo_nlp.primals_ub()
        comparison_xub = np.asarray([150, 5, 5, 90, 100], dtype=np.float64)
        check_vectors_specific_order(
            self, xub, x_order, comparison_xub, comparison_x_order
        )
        clb = pyomo_nlp.constraints_lb()
        comparison_clb = np.asarray([0, 0], dtype=np.float64)
        check_vectors_specific_order(
            self, clb, c_order, comparison_clb, comparison_c_order
        )
        cub = pyomo_nlp.constraints_ub()
        comparison_cub = np.asarray([0, 0], dtype=np.float64)
        check_vectors_specific_order(
            self, cub, c_order, comparison_cub, comparison_c_order
        )

        xinit = pyomo_nlp.init_primals()
        comparison_xinit = np.asarray([100, 2, 3, 80, 50], dtype=np.float64)
        check_vectors_specific_order(
            self, xinit, x_order, comparison_xinit, comparison_x_order
        )
        duals_init = pyomo_nlp.init_duals()
        comparison_duals_init = np.asarray([0, 0], dtype=np.float64)
        check_vectors_specific_order(
            self, duals_init, c_order, comparison_duals_init, comparison_c_order
        )

        self.assertEqual(5, len(pyomo_nlp.create_new_vector('primals')))
        self.assertEqual(2, len(pyomo_nlp.create_new_vector('constraints')))
        self.assertEqual(2, len(pyomo_nlp.create_new_vector('duals')))
        self.assertEqual(2, len(pyomo_nlp.create_new_vector('eq_constraints')))
        self.assertEqual(0, len(pyomo_nlp.create_new_vector('ineq_constraints')))
        self.assertEqual(2, len(pyomo_nlp.create_new_vector('duals_eq')))
        self.assertEqual(0, len(pyomo_nlp.create_new_vector('duals_ineq')))

        pyomo_nlp.set_primals(np.asarray([1, 2, 3, 4, 5], dtype=np.float64))
        x = pyomo_nlp.get_primals()
        self.assertTrue(
            np.array_equal(x, np.asarray([1, 2, 3, 4, 5], dtype=np.float64))
        )
        pyomo_nlp.set_primals(pyomo_nlp.init_primals())

        pyomo_nlp.set_duals(np.asarray([42, 10], dtype=np.float64))
        y = pyomo_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([42, 10], dtype=np.float64)))
        pyomo_nlp.set_duals(np.asarray([21, 5], dtype=np.float64))
        y = pyomo_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([21, 5], dtype=np.float64)))

        fac = pyomo_nlp.get_obj_factor()
        self.assertEqual(fac, 1)
        pyomo_nlp.set_obj_factor(42)
        self.assertEqual(pyomo_nlp.get_obj_factor(), 42)
        pyomo_nlp.set_obj_factor(1)

        f = pyomo_nlp.evaluate_objective()
        self.assertEqual(f, 900)

        gradf = pyomo_nlp.evaluate_grad_objective()
        comparison_gradf = np.asarray([0, 0, 0, 0, 60], dtype=np.float64)
        check_vectors_specific_order(
            self, gradf, x_order, comparison_gradf, comparison_x_order
        )
        c = pyomo_nlp.evaluate_constraints()
        comparison_c = np.asarray([-16, -22], dtype=np.float64)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        c = np.zeros(2)
        pyomo_nlp.evaluate_constraints(out=c)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)

        j = pyomo_nlp.evaluate_jacobian()
        comparison_j = np.asarray([[1, -18, -24, -1, 0], [1, -36, -48, 0, -1]])
        check_sparse_matrix_specific_order(
            self,
            j,
            c_order,
            x_order,
            comparison_j,
            comparison_c_order,
            comparison_x_order,
        )

        j = 2.0 * j
        pyomo_nlp.evaluate_jacobian(out=j)
        check_sparse_matrix_specific_order(
            self,
            j,
            c_order,
            x_order,
            comparison_j,
            comparison_c_order,
            comparison_x_order,
        )

        if hessian_support:
            h = pyomo_nlp.evaluate_hessian_lag()
            self.assertTrue(h.shape == (5, 5))
            comparison_h = np.asarray(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [
                        0,
                        (-4 * 3 * 21) + (-8 * 3 * 5),
                        (-4 * 2 * 21) + (-8 * 2 * 5),
                        0,
                        0,
                    ],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 2 * 1],
                ],
                dtype=np.float64,
            )
            check_sparse_matrix_specific_order(
                self,
                h,
                x_order,
                x_order,
                comparison_h,
                comparison_x_order,
                comparison_x_order,
            )
        else:
            with self.assertRaises(AttributeError):
                h = pyomo_nlp.evaluate_hessian_lag()

    def test_pressure_drop_two_equalities(self):
        self._test_pressure_drop_two_equalities(
            ex_models.PressureDropTwoEqualities(), False
        )
        self._test_pressure_drop_two_equalities(
            ex_models.PressureDropTwoEqualitiesWithHessian(), True
        )

    def _test_pressure_drop_two_equalities(self, ex_model, hessian_support):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(ex_model)
        m.egb.inputs['Pin'].value = 100
        m.egb.inputs['Pin'].setlb(50)
        m.egb.inputs['Pin'].setub(150)
        m.egb.inputs['c'].value = 2
        m.egb.inputs['c'].setlb(1)
        m.egb.inputs['c'].setub(5)
        m.egb.inputs['F'].value = 3
        m.egb.inputs['F'].setlb(1)
        m.egb.inputs['F'].setub(5)
        m.egb.inputs['P2'].value = 80
        m.egb.inputs['P2'].setlb(10)
        m.egb.inputs['P2'].setub(90)
        m.egb.inputs['Pout'].value = 50
        m.egb.inputs['Pout'].setlb(0)
        m.egb.inputs['Pout'].setub(100)
        m.obj = pyo.Objective(expr=(m.egb.inputs['Pout'] - 20) ** 2)
        pyomo_nlp = PyomoGreyBoxNLP(m)

        self.assertEqual(5, pyomo_nlp.n_primals())
        self.assertEqual(2, pyomo_nlp.n_constraints())
        self.assertEqual(8, pyomo_nlp.nnz_jacobian())
        if hessian_support:
            self.assertEqual(3, pyomo_nlp.nnz_hessian_lag())

        comparison_x_order = [
            'egb.inputs[Pin]',
            'egb.inputs[c]',
            'egb.inputs[F]',
            'egb.inputs[P2]',
            'egb.inputs[Pout]',
        ]
        x_order = pyomo_nlp.variable_names()
        comparison_c_order = ['egb.pdrop2', 'egb.pdropout']
        c_order = pyomo_nlp.constraint_names()

        xlb = pyomo_nlp.primals_lb()
        comparison_xlb = np.asarray([50, 1, 1, 10, 0], dtype=np.float64)
        check_vectors_specific_order(
            self, xlb, x_order, comparison_xlb, comparison_x_order
        )
        xub = pyomo_nlp.primals_ub()
        comparison_xub = np.asarray([150, 5, 5, 90, 100], dtype=np.float64)
        check_vectors_specific_order(
            self, xub, x_order, comparison_xub, comparison_x_order
        )
        clb = pyomo_nlp.constraints_lb()
        comparison_clb = np.asarray([0, 0], dtype=np.float64)
        check_vectors_specific_order(
            self, clb, c_order, comparison_clb, comparison_c_order
        )
        cub = pyomo_nlp.constraints_ub()
        comparison_cub = np.asarray([0, 0], dtype=np.float64)
        check_vectors_specific_order(
            self, cub, c_order, comparison_cub, comparison_c_order
        )

        xinit = pyomo_nlp.init_primals()
        comparison_xinit = np.asarray([100, 2, 3, 80, 50], dtype=np.float64)
        check_vectors_specific_order(
            self, xinit, x_order, comparison_xinit, comparison_x_order
        )
        duals_init = pyomo_nlp.init_duals()
        comparison_duals_init = np.asarray([0, 0], dtype=np.float64)
        check_vectors_specific_order(
            self, duals_init, c_order, comparison_duals_init, comparison_c_order
        )

        self.assertEqual(5, len(pyomo_nlp.create_new_vector('primals')))
        self.assertEqual(2, len(pyomo_nlp.create_new_vector('constraints')))
        self.assertEqual(2, len(pyomo_nlp.create_new_vector('duals')))
        self.assertEqual(2, len(pyomo_nlp.create_new_vector('eq_constraints')))
        self.assertEqual(0, len(pyomo_nlp.create_new_vector('ineq_constraints')))
        self.assertEqual(2, len(pyomo_nlp.create_new_vector('duals_eq')))
        self.assertEqual(0, len(pyomo_nlp.create_new_vector('duals_ineq')))

        pyomo_nlp.set_primals(np.asarray([1, 2, 3, 4, 5], dtype=np.float64))
        x = pyomo_nlp.get_primals()
        self.assertTrue(
            np.array_equal(x, np.asarray([1, 2, 3, 4, 5], dtype=np.float64))
        )
        pyomo_nlp.set_primals(pyomo_nlp.init_primals())

        pyomo_nlp.set_duals(np.asarray([42, 10], dtype=np.float64))
        y = pyomo_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([42, 10], dtype=np.float64)))
        pyomo_nlp.set_duals(np.asarray([21, 5], dtype=np.float64))
        y = pyomo_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([21, 5], dtype=np.float64)))

        fac = pyomo_nlp.get_obj_factor()
        self.assertEqual(fac, 1)
        pyomo_nlp.set_obj_factor(42)
        self.assertEqual(pyomo_nlp.get_obj_factor(), 42)
        pyomo_nlp.set_obj_factor(1)

        f = pyomo_nlp.evaluate_objective()
        self.assertEqual(f, 900)

        gradf = pyomo_nlp.evaluate_grad_objective()
        comparison_gradf = np.asarray([0, 0, 0, 0, 60], dtype=np.float64)
        check_vectors_specific_order(
            self, gradf, x_order, comparison_gradf, comparison_x_order
        )
        c = pyomo_nlp.evaluate_constraints()
        comparison_c = np.asarray([16, 6], dtype=np.float64)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        c = np.zeros(2)
        pyomo_nlp.evaluate_constraints(out=c)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)

        j = pyomo_nlp.evaluate_jacobian()
        comparison_j = np.asarray([[-1, 18, 24, 1, 0], [0, 18, 24, -1, 1]])
        check_sparse_matrix_specific_order(
            self,
            j,
            c_order,
            x_order,
            comparison_j,
            comparison_c_order,
            comparison_x_order,
        )

        j = 2.0 * j
        pyomo_nlp.evaluate_jacobian(out=j)
        check_sparse_matrix_specific_order(
            self,
            j,
            c_order,
            x_order,
            comparison_j,
            comparison_c_order,
            comparison_x_order,
        )

        if hessian_support:
            h = pyomo_nlp.evaluate_hessian_lag()
            self.assertTrue(h.shape == (5, 5))
            comparison_h = np.asarray(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, (4 * 3 * 21) + (4 * 3 * 5), (4 * 2 * 21) + (4 * 2 * 5), 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 2 * 1],
                ],
                dtype=np.float64,
            )
            check_sparse_matrix_specific_order(
                self,
                h,
                x_order,
                x_order,
                comparison_h,
                comparison_x_order,
                comparison_x_order,
            )
        else:
            with self.assertRaises(AttributeError):
                h = pyomo_nlp.evaluate_hessian_lag()

    def test_pressure_drop_two_equalities_two_outputs(self):
        self._test_pressure_drop_two_equalities_two_outputs(
            ex_models.PressureDropTwoEqualitiesTwoOutputs(), False
        )
        self._test_pressure_drop_two_equalities_two_outputs(
            ex_models.PressureDropTwoEqualitiesTwoOutputsWithHessian(), True
        )

    def _test_pressure_drop_two_equalities_two_outputs(self, ex_model, hessian_support):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(ex_model)
        m.egb.inputs['Pin'].value = 100
        m.egb.inputs['Pin'].setlb(50)
        m.egb.inputs['Pin'].setub(150)
        m.egb.inputs['c'].value = 2
        m.egb.inputs['c'].setlb(1)
        m.egb.inputs['c'].setub(5)
        m.egb.inputs['F'].value = 3
        m.egb.inputs['F'].setlb(1)
        m.egb.inputs['F'].setub(5)
        m.egb.inputs['P1'].value = 80
        m.egb.inputs['P1'].setlb(10)
        m.egb.inputs['P1'].setub(90)
        m.egb.inputs['P3'].value = 70
        m.egb.inputs['P3'].setlb(20)
        m.egb.inputs['P3'].setub(80)
        m.egb.outputs['P2'].value = 75
        m.egb.outputs['P2'].setlb(15)
        m.egb.outputs['P2'].setub(85)
        m.egb.outputs['Pout'].value = 50
        m.egb.outputs['Pout'].setlb(30)
        m.egb.outputs['Pout'].setub(70)
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout'] - 20) ** 2)
        pyomo_nlp = PyomoGreyBoxNLP(m)

        self.assertEqual(7, pyomo_nlp.n_primals())
        self.assertEqual(4, pyomo_nlp.n_constraints())
        self.assertEqual(16, pyomo_nlp.nnz_jacobian())
        if hessian_support:
            self.assertEqual(5, pyomo_nlp.nnz_hessian_lag())

        comparison_x_order = [
            'egb.inputs[Pin]',
            'egb.inputs[c]',
            'egb.inputs[F]',
            'egb.inputs[P1]',
            'egb.inputs[P3]',
            'egb.outputs[P2]',
            'egb.outputs[Pout]',
        ]
        x_order = pyomo_nlp.variable_names()
        comparison_c_order = ['egb.pdrop1', 'egb.pdrop3', 'egb.P2_con', 'egb.Pout_con']
        c_order = pyomo_nlp.constraint_names()

        xlb = pyomo_nlp.primals_lb()
        comparison_xlb = np.asarray([50, 1, 1, 10, 20, 15, 30], dtype=np.float64)
        check_vectors_specific_order(
            self, xlb, x_order, comparison_xlb, comparison_x_order
        )
        xub = pyomo_nlp.primals_ub()
        comparison_xub = np.asarray([150, 5, 5, 90, 80, 85, 70], dtype=np.float64)
        check_vectors_specific_order(
            self, xub, x_order, comparison_xub, comparison_x_order
        )
        clb = pyomo_nlp.constraints_lb()
        comparison_clb = np.asarray([0, 0, 0, 0], dtype=np.float64)
        check_vectors_specific_order(
            self, clb, c_order, comparison_clb, comparison_c_order
        )
        cub = pyomo_nlp.constraints_ub()
        comparison_cub = np.asarray([0, 0, 0, 0], dtype=np.float64)
        check_vectors_specific_order(
            self, cub, c_order, comparison_cub, comparison_c_order
        )

        xinit = pyomo_nlp.init_primals()
        comparison_xinit = np.asarray([100, 2, 3, 80, 70, 75, 50], dtype=np.float64)
        check_vectors_specific_order(
            self, xinit, x_order, comparison_xinit, comparison_x_order
        )
        duals_init = pyomo_nlp.init_duals()
        comparison_duals_init = np.asarray([0, 0, 0, 0], dtype=np.float64)
        check_vectors_specific_order(
            self, duals_init, c_order, comparison_duals_init, comparison_c_order
        )

        self.assertEqual(7, len(pyomo_nlp.create_new_vector('primals')))
        self.assertEqual(4, len(pyomo_nlp.create_new_vector('constraints')))
        self.assertEqual(4, len(pyomo_nlp.create_new_vector('duals')))
        self.assertEqual(4, len(pyomo_nlp.create_new_vector('eq_constraints')))
        self.assertEqual(0, len(pyomo_nlp.create_new_vector('ineq_constraints')))
        self.assertEqual(4, len(pyomo_nlp.create_new_vector('duals_eq')))
        self.assertEqual(0, len(pyomo_nlp.create_new_vector('duals_ineq')))

        pyomo_nlp.set_primals(np.asarray([1, 2, 3, 4, 5, 6, 7], dtype=np.float64))
        x = pyomo_nlp.get_primals()
        self.assertTrue(
            np.array_equal(x, np.asarray([1, 2, 3, 4, 5, 6, 7], dtype=np.float64))
        )
        pyomo_nlp.set_primals(pyomo_nlp.init_primals())

        pyomo_nlp.set_duals(np.asarray([42, 10, 11, 12], dtype=np.float64))
        y = pyomo_nlp.get_duals()
        self.assertTrue(
            np.array_equal(y, np.asarray([42, 10, 11, 12], dtype=np.float64))
        )
        pyomo_nlp.set_duals(np.asarray([21, 5, 6, 7], dtype=np.float64))
        y = pyomo_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([21, 5, 6, 7], dtype=np.float64)))

        fac = pyomo_nlp.get_obj_factor()
        self.assertEqual(fac, 1)
        pyomo_nlp.set_obj_factor(42)
        self.assertEqual(pyomo_nlp.get_obj_factor(), 42)
        pyomo_nlp.set_obj_factor(1)

        f = pyomo_nlp.evaluate_objective()
        self.assertEqual(f, 900)

        gradf = pyomo_nlp.evaluate_grad_objective()
        comparison_gradf = np.asarray([0, 0, 0, 0, 0, 0, 60], dtype=np.float64)
        check_vectors_specific_order(
            self, gradf, x_order, comparison_gradf, comparison_x_order
        )
        c = pyomo_nlp.evaluate_constraints()
        comparison_c = np.asarray([-2, 26, -13, -22], dtype=np.float64)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        c = np.zeros(4)
        pyomo_nlp.evaluate_constraints(out=c)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)

        j = pyomo_nlp.evaluate_jacobian()
        comparison_j = np.asarray(
            [
                [-1, 9, 12, 1, 0, 0, 0],
                [0, 18, 24, -1, 1, 0, 0],
                [0, -9, -12, 1, 0, -1, 0],
                [1, -36, -48, 0, 0, 0, -1],
            ]
        )
        check_sparse_matrix_specific_order(
            self,
            j,
            c_order,
            x_order,
            comparison_j,
            comparison_c_order,
            comparison_x_order,
        )

        j = 2.0 * j
        pyomo_nlp.evaluate_jacobian(out=j)
        check_sparse_matrix_specific_order(
            self,
            j,
            c_order,
            x_order,
            comparison_j,
            comparison_c_order,
            comparison_x_order,
        )

        if hessian_support:
            h = pyomo_nlp.evaluate_hessian_lag()
            self.assertTrue(h.shape == (7, 7))
            comparison_h = np.asarray(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [
                        0,
                        (2 * 3 * 21) + (4 * 3 * 5) + (-2 * 3 * 6) + (-8 * 3 * 7),
                        (2 * 2 * 21) + (4 * 2 * 5) + (-2 * 2 * 6) + (-8 * 2 * 7),
                        0,
                        0,
                        0,
                        0,
                    ],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 2 * 1],
                ],
                dtype=np.float64,
            )
            check_sparse_matrix_specific_order(
                self,
                h,
                x_order,
                x_order,
                comparison_h,
                comparison_x_order,
                comparison_x_order,
            )
        else:
            with self.assertRaises(AttributeError):
                h = pyomo_nlp.evaluate_hessian_lag()

    def test_external_additional_constraints_vars(self):
        self._test_external_additional_constraints_vars(
            ex_models.PressureDropTwoEqualitiesTwoOutputs(), False
        )
        self._test_external_additional_constraints_vars(
            ex_models.PressureDropTwoEqualitiesTwoOutputsWithHessian(), True
        )

    def _test_external_additional_constraints_vars(self, ex_model, hessian_support):
        m = pyo.ConcreteModel()
        m.hin = pyo.Var(bounds=(0, None), initialize=10)
        m.hout = pyo.Var(bounds=(0, None))
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(ex_model)
        m.incon = pyo.Constraint(expr=0 <= m.egb.inputs['Pin'] - 10 * m.hin)
        m.outcon = pyo.Constraint(expr=0 == m.egb.outputs['Pout'] - 10 * m.hout)
        m.egb.inputs['Pin'].value = 100
        m.egb.inputs['Pin'].setlb(50)
        m.egb.inputs['Pin'].setub(150)
        m.egb.inputs['c'].value = 2
        m.egb.inputs['c'].setlb(1)
        m.egb.inputs['c'].setub(5)
        m.egb.inputs['F'].value = 3
        m.egb.inputs['F'].setlb(1)
        m.egb.inputs['F'].setub(5)
        m.egb.inputs['P1'].value = 80
        m.egb.inputs['P1'].setlb(10)
        m.egb.inputs['P1'].setub(90)
        m.egb.inputs['P3'].value = 70
        m.egb.inputs['P3'].setlb(20)
        m.egb.inputs['P3'].setub(80)
        m.egb.outputs['P2'].value = 75
        m.egb.outputs['P2'].setlb(15)
        m.egb.outputs['P2'].setub(85)
        m.egb.outputs['Pout'].value = 50
        m.egb.outputs['Pout'].setlb(30)
        m.egb.outputs['Pout'].setub(70)
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout'] - 20) ** 2)
        pyomo_nlp = PyomoGreyBoxNLP(m)

        self.assertEqual(9, pyomo_nlp.n_primals())
        self.assertEqual(6, pyomo_nlp.n_constraints())
        self.assertEqual(20, pyomo_nlp.nnz_jacobian())
        if hessian_support:
            self.assertEqual(5, pyomo_nlp.nnz_hessian_lag())

        comparison_x_order = [
            'egb.inputs[Pin]',
            'egb.inputs[c]',
            'egb.inputs[F]',
            'egb.inputs[P1]',
            'egb.inputs[P3]',
            'egb.outputs[P2]',
            'egb.outputs[Pout]',
            'hin',
            'hout',
        ]
        x_order = pyomo_nlp.variable_names()
        comparison_c_order = [
            'egb.pdrop1',
            'egb.pdrop3',
            'egb.P2_con',
            'egb.Pout_con',
            'incon',
            'outcon',
        ]
        c_order = pyomo_nlp.constraint_names()

        xlb = pyomo_nlp.primals_lb()
        comparison_xlb = np.asarray([50, 1, 1, 10, 20, 15, 30, 0, 0], dtype=np.float64)
        check_vectors_specific_order(
            self, xlb, x_order, comparison_xlb, comparison_x_order
        )
        xub = pyomo_nlp.primals_ub()
        comparison_xub = np.asarray(
            [150, 5, 5, 90, 80, 85, 70, np.inf, np.inf], dtype=np.float64
        )
        check_vectors_specific_order(
            self, xub, x_order, comparison_xub, comparison_x_order
        )
        clb = pyomo_nlp.constraints_lb()
        comparison_clb = np.asarray([0, 0, 0, 0, 0, 0], dtype=np.float64)
        check_vectors_specific_order(
            self, clb, c_order, comparison_clb, comparison_c_order
        )
        cub = pyomo_nlp.constraints_ub()
        comparison_cub = np.asarray([0, 0, 0, 0, np.inf, 0], dtype=np.float64)
        check_vectors_specific_order(
            self, cub, c_order, comparison_cub, comparison_c_order
        )

        xinit = pyomo_nlp.init_primals()
        comparison_xinit = np.asarray(
            [100, 2, 3, 80, 70, 75, 50, 10, 0], dtype=np.float64
        )
        check_vectors_specific_order(
            self, xinit, x_order, comparison_xinit, comparison_x_order
        )
        duals_init = pyomo_nlp.init_duals()
        comparison_duals_init = np.asarray([0, 0, 0, 0, 0, 0], dtype=np.float64)
        check_vectors_specific_order(
            self, duals_init, c_order, comparison_duals_init, comparison_c_order
        )

        self.assertEqual(9, len(pyomo_nlp.create_new_vector('primals')))
        self.assertEqual(6, len(pyomo_nlp.create_new_vector('constraints')))
        self.assertEqual(6, len(pyomo_nlp.create_new_vector('duals')))
        self.assertEqual(5, len(pyomo_nlp.create_new_vector('eq_constraints')))
        self.assertEqual(1, len(pyomo_nlp.create_new_vector('ineq_constraints')))
        self.assertEqual(5, len(pyomo_nlp.create_new_vector('duals_eq')))
        self.assertEqual(1, len(pyomo_nlp.create_new_vector('duals_ineq')))

        pyomo_nlp.set_primals(np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64))
        x = pyomo_nlp.get_primals()
        self.assertTrue(
            np.array_equal(x, np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64))
        )
        pyomo_nlp.set_primals(pyomo_nlp.init_primals())

        pyomo_nlp.set_duals(np.asarray([42, 10, 11, 12, 13, 14], dtype=np.float64))
        y = pyomo_nlp.get_duals()
        self.assertTrue(
            np.array_equal(y, np.asarray([42, 10, 11, 12, 13, 14], dtype=np.float64))
        )
        pyomo_nlp.set_duals(np.asarray([0, 0, 21, 5, 6, 7], dtype=np.float64))
        y = pyomo_nlp.get_duals()
        self.assertTrue(
            np.array_equal(y, np.asarray([0, 0, 21, 5, 6, 7], dtype=np.float64))
        )

        fac = pyomo_nlp.get_obj_factor()
        self.assertEqual(fac, 1)
        pyomo_nlp.set_obj_factor(42)
        self.assertEqual(pyomo_nlp.get_obj_factor(), 42)
        pyomo_nlp.set_obj_factor(1)

        f = pyomo_nlp.evaluate_objective()
        self.assertEqual(f, 900)

        gradf = pyomo_nlp.evaluate_grad_objective()
        comparison_gradf = np.asarray([0, 0, 0, 0, 0, 0, 60, 0, 0], dtype=np.float64)
        check_vectors_specific_order(
            self, gradf, x_order, comparison_gradf, comparison_x_order
        )
        c = pyomo_nlp.evaluate_constraints()
        comparison_c = np.asarray([-2, 26, -13, -22, 0, 50], dtype=np.float64)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        c = np.zeros(6)
        pyomo_nlp.evaluate_constraints(out=c)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)

        j = pyomo_nlp.evaluate_jacobian()
        comparison_j = np.asarray(
            [
                [-1, 9, 12, 1, 0, 0, 0, 0, 0],
                [0, 18, 24, -1, 1, 0, 0, 0, 0],
                [0, -9, -12, 1, 0, -1, 0, 0, 0],
                [1, -36, -48, 0, 0, 0, -1, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, -10, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, -10],
            ]
        )

        check_sparse_matrix_specific_order(
            self,
            j,
            c_order,
            x_order,
            comparison_j,
            comparison_c_order,
            comparison_x_order,
        )

        j = 2.0 * j
        pyomo_nlp.evaluate_jacobian(out=j)
        check_sparse_matrix_specific_order(
            self,
            j,
            c_order,
            x_order,
            comparison_j,
            comparison_c_order,
            comparison_x_order,
        )

        if hessian_support:
            h = pyomo_nlp.evaluate_hessian_lag()
            self.assertTrue(h.shape == (9, 9))
            comparison_h = np.asarray(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [
                        0,
                        (2 * 3 * 21) + (4 * 3 * 5) + (-2 * 3 * 6) + (-8 * 3 * 7),
                        (2 * 2 * 21) + (4 * 2 * 5) + (-2 * 2 * 6) + (-8 * 2 * 7),
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 2 * 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.float64,
            )
            check_sparse_matrix_specific_order(
                self,
                h,
                x_order,
                x_order,
                comparison_h,
                comparison_x_order,
                comparison_x_order,
            )
        else:
            with self.assertRaises(AttributeError):
                h = pyomo_nlp.evaluate_hessian_lag()

    @unittest.skipIf(not cyipopt_available, "CyIpopt needed to run tests with solve")
    def test_external_greybox_solve(self):
        self._test_external_greybox_solve(
            ex_models.PressureDropTwoEqualitiesTwoOutputs(), False
        )
        self._test_external_greybox_solve(
            ex_models.PressureDropTwoEqualitiesTwoOutputsWithHessian(), True
        )

    def _test_external_greybox_solve(self, ex_model, hessian_support):
        m = pyo.ConcreteModel()
        m.mu = pyo.Var(bounds=(0, None), initialize=1)
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(ex_model)
        m.ccon = pyo.Constraint(
            expr=m.egb.inputs['c'] == 128 / (3.14 * 1e-4) * m.mu * m.egb.inputs['F']
        )
        m.pcon = pyo.Constraint(expr=m.egb.inputs['Pin'] - m.egb.outputs['Pout'] <= 72)
        m.pincon = pyo.Constraint(expr=m.egb.inputs['Pin'] == 100.0)
        m.egb.inputs['Pin'].value = 100
        m.egb.inputs['Pin'].setlb(50)
        m.egb.inputs['Pin'].setub(150)
        m.egb.inputs['c'].value = 2
        m.egb.inputs['c'].setlb(1)
        m.egb.inputs['c'].setub(5)
        m.egb.inputs['F'].value = 3
        m.egb.inputs['F'].setlb(1)
        m.egb.inputs['F'].setub(5)
        m.egb.inputs['P1'].value = 80
        m.egb.inputs['P1'].setlb(10)
        m.egb.inputs['P1'].setub(90)
        m.egb.inputs['P3'].value = 70
        m.egb.inputs['P3'].setlb(20)
        m.egb.inputs['P3'].setub(80)
        m.egb.outputs['P2'].value = 75
        m.egb.outputs['P2'].setlb(15)
        m.egb.outputs['P2'].setub(85)
        m.egb.outputs['Pout'].value = 50
        m.egb.outputs['Pout'].setlb(10)
        m.egb.outputs['Pout'].setub(70)
        m.obj = pyo.Objective(
            expr=(m.egb.outputs['Pout'] - 20) ** 2 + (m.egb.inputs['F'] - 3) ** 2
        )

        solver = pyo.SolverFactory('cyipopt')

        if not hessian_support:
            solver.config.options = {'hessian_approximation': 'limited-memory'}
        status = solver.solve(m, tee=False)

        self.assertAlmostEqual(pyo.value(m.egb.inputs['F']), 3.0, places=3)
        self.assertAlmostEqual(pyo.value(m.mu), 1.63542e-6, places=3)
        self.assertAlmostEqual(pyo.value(m.egb.outputs['Pout']), 28.0, places=3)
        self.assertAlmostEqual(pyo.value(m.egb.inputs['Pin']), 100.0, places=3)
        self.assertAlmostEqual(pyo.value(m.egb.inputs['c']), 2.0, places=3)
        self.assertAlmostEqual(pyo.value(m.egb.inputs['P1']), 82.0, places=3)
        self.assertAlmostEqual(pyo.value(m.egb.inputs['P3']), 46.0, places=3)
        self.assertAlmostEqual(pyo.value(m.egb.outputs['P2']), 64.0, places=3)
        self.assertAlmostEqual(pyo.value(m.egb.inputs['F']), 3.0, places=3)

    def create_model_two_equalities_two_outputs(self, external_model):
        m = pyo.ConcreteModel()
        m.hin = pyo.Var(bounds=(0, None), initialize=10)
        m.hout = pyo.Var(bounds=(0, None))
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(external_model)
        m.incon = pyo.Constraint(expr=0 <= m.egb.inputs['Pin'] - 10 * m.hin)
        m.outcon = pyo.Constraint(expr=0 == m.egb.outputs['Pout'] - 10 * m.hout)
        m.egb.inputs['Pin'].value = 100
        m.egb.inputs['Pin'].setlb(50)
        m.egb.inputs['Pin'].setub(150)
        m.egb.inputs['c'].value = 2
        m.egb.inputs['c'].setlb(1)
        m.egb.inputs['c'].setub(5)
        m.egb.inputs['F'].value = 3
        m.egb.inputs['F'].setlb(1)
        m.egb.inputs['F'].setub(5)
        m.egb.inputs['P1'].value = 80
        m.egb.inputs['P1'].setlb(10)
        m.egb.inputs['P1'].setub(90)
        m.egb.inputs['P3'].value = 70
        m.egb.inputs['P3'].setlb(20)
        m.egb.inputs['P3'].setub(80)
        m.egb.outputs['P2'].value = 75
        m.egb.outputs['P2'].setlb(15)
        m.egb.outputs['P2'].setub(85)
        m.egb.outputs['Pout'].value = 50
        m.egb.outputs['Pout'].setlb(30)
        m.egb.outputs['Pout'].setub(70)
        return m

    def test_scaling_all_missing(self):
        m = self.create_model_two_equalities_two_outputs(
            ex_models.PressureDropTwoEqualitiesTwoOutputs()
        )
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout'] - 20) ** 2)
        pyomo_nlp = PyomoGreyBoxNLP(m)
        fs = pyomo_nlp.get_obj_scaling()
        xs = pyomo_nlp.get_primals_scaling()
        cs = pyomo_nlp.get_constraints_scaling()
        self.assertIsNone(fs)
        self.assertIsNone(xs)
        self.assertIsNone(cs)

    def test_scaling_pyomo_model_only(self):
        m = self.create_model_two_equalities_two_outputs(
            ex_models.PressureDropTwoEqualitiesTwoOutputs()
        )
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout'] - 20) ** 2)
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        # m.scaling_factor[m.obj] = 0.1 # scale the objective
        m.scaling_factor[m.egb.inputs['Pin']] = 1.1  # scale the variable
        m.scaling_factor[m.egb.inputs['c']] = 1.2  # scale the variable
        m.scaling_factor[m.egb.inputs['F']] = 1.3  # scale the variable
        # m.scaling_factor[m.egb.inputs['P1']] = 1.4 # scale the variable
        m.scaling_factor[m.egb.inputs['P3']] = 1.5  # scale the variable
        m.scaling_factor[m.egb.outputs['P2']] = 1.6  # scale the variable
        m.scaling_factor[m.egb.outputs['Pout']] = 1.7  # scale the variable
        # m.scaling_factor[m.hin] = 1.8
        m.scaling_factor[m.hout] = 1.9
        # m.scaling_factor[m.incon] = 2.1
        m.scaling_factor[m.outcon] = 2.2
        pyomo_nlp = PyomoGreyBoxNLP(m)

        comparison_x_order = [
            'egb.inputs[Pin]',
            'egb.inputs[c]',
            'egb.inputs[F]',
            'egb.inputs[P1]',
            'egb.inputs[P3]',
            'egb.outputs[P2]',
            'egb.outputs[Pout]',
            'hin',
            'hout',
        ]
        x_order = pyomo_nlp.variable_names()
        comparison_c_order = [
            'egb.pdrop1',
            'egb.pdrop3',
            'egb.P2_con',
            'egb.Pout_con',
            'incon',
            'outcon',
        ]
        c_order = pyomo_nlp.constraint_names()

        fs = pyomo_nlp.get_obj_scaling()
        self.assertEqual(fs, 1.0)

        xs = pyomo_nlp.get_primals_scaling()
        comparison_xs = np.asarray(
            [1.1, 1.2, 1.3, 1.0, 1.5, 1.6, 1.7, 1.0, 1.9], dtype=np.float64
        )
        check_vectors_specific_order(
            self, xs, x_order, comparison_xs, comparison_x_order
        )

        cs = pyomo_nlp.get_constraints_scaling()
        comparison_cs = np.asarray([1, 1, 1, 1, 1, 2.2], dtype=np.float64)
        check_vectors_specific_order(
            self, cs, c_order, comparison_cs, comparison_c_order
        )

    def test_scaling_greybox_only(self):
        m = self.create_model_two_equalities_two_outputs(
            ex_models.PressureDropTwoEqualitiesTwoOutputsScaleBoth()
        )
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout'] - 20) ** 2)
        pyomo_nlp = PyomoGreyBoxNLP(m)

        comparison_x_order = [
            'egb.inputs[Pin]',
            'egb.inputs[c]',
            'egb.inputs[F]',
            'egb.inputs[P1]',
            'egb.inputs[P3]',
            'egb.outputs[P2]',
            'egb.outputs[Pout]',
            'hin',
            'hout',
        ]
        x_order = pyomo_nlp.variable_names()
        comparison_c_order = [
            'egb.pdrop1',
            'egb.pdrop3',
            'egb.P2_con',
            'egb.Pout_con',
            'incon',
            'outcon',
        ]
        c_order = pyomo_nlp.constraint_names()

        fs = pyomo_nlp.get_obj_scaling()
        self.assertEqual(fs, 1.0)

        xs = pyomo_nlp.get_primals_scaling()
        comparison_xs = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)
        check_vectors_specific_order(
            self, xs, x_order, comparison_xs, comparison_x_order
        )

        cs = pyomo_nlp.get_constraints_scaling()
        comparison_cs = np.asarray([3.1, 3.2, 4.1, 4.2, 1, 1], dtype=np.float64)
        check_vectors_specific_order(
            self, cs, c_order, comparison_cs, comparison_c_order
        )

        m = self.create_model_two_equalities_two_outputs(
            ex_models.PressureDropTwoEqualitiesTwoOutputsScaleEqualities()
        )
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout'] - 20) ** 2)
        pyomo_nlp = PyomoGreyBoxNLP(m)
        cs = pyomo_nlp.get_constraints_scaling()
        comparison_cs = np.asarray([3.1, 3.2, 1, 1, 1, 1], dtype=np.float64)
        check_vectors_specific_order(
            self, cs, c_order, comparison_cs, comparison_c_order
        )

        m = self.create_model_two_equalities_two_outputs(
            ex_models.PressureDropTwoEqualitiesTwoOutputsScaleOutputs()
        )
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout'] - 20) ** 2)
        pyomo_nlp = PyomoGreyBoxNLP(m)
        cs = pyomo_nlp.get_constraints_scaling()
        comparison_cs = np.asarray([1, 1, 4.1, 4.2, 1, 1], dtype=np.float64)
        check_vectors_specific_order(
            self, cs, c_order, comparison_cs, comparison_c_order
        )

    def test_scaling_pyomo_model_and_greybox(self):
        m = self.create_model_two_equalities_two_outputs(
            ex_models.PressureDropTwoEqualitiesTwoOutputsScaleBoth()
        )
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout'] - 20) ** 2)
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        # m.scaling_factor[m.obj] = 0.1 # scale the objective
        m.scaling_factor[m.egb.inputs['Pin']] = 1.1  # scale the variable
        m.scaling_factor[m.egb.inputs['c']] = 1.2  # scale the variable
        m.scaling_factor[m.egb.inputs['F']] = 1.3  # scale the variable
        # m.scaling_factor[m.egb.inputs['P1']] = 1.4 # scale the variable
        m.scaling_factor[m.egb.inputs['P3']] = 1.5  # scale the variable
        m.scaling_factor[m.egb.outputs['P2']] = 1.6  # scale the variable
        m.scaling_factor[m.egb.outputs['Pout']] = 1.7  # scale the variable
        # m.scaling_factor[m.hin] = 1.8
        m.scaling_factor[m.hout] = 1.9
        # m.scaling_factor[m.incon] = 2.1
        m.scaling_factor[m.outcon] = 2.2
        pyomo_nlp = PyomoGreyBoxNLP(m)

        comparison_x_order = [
            'egb.inputs[Pin]',
            'egb.inputs[c]',
            'egb.inputs[F]',
            'egb.inputs[P1]',
            'egb.inputs[P3]',
            'egb.outputs[P2]',
            'egb.outputs[Pout]',
            'hin',
            'hout',
        ]
        x_order = pyomo_nlp.variable_names()
        comparison_c_order = [
            'egb.pdrop1',
            'egb.pdrop3',
            'egb.P2_con',
            'egb.Pout_con',
            'incon',
            'outcon',
        ]
        c_order = pyomo_nlp.constraint_names()

        fs = pyomo_nlp.get_obj_scaling()
        self.assertEqual(fs, 1.0)

        xs = pyomo_nlp.get_primals_scaling()
        comparison_xs = np.asarray(
            [1.1, 1.2, 1.3, 1.0, 1.5, 1.6, 1.7, 1.0, 1.9], dtype=np.float64
        )
        check_vectors_specific_order(
            self, xs, x_order, comparison_xs, comparison_x_order
        )

        cs = pyomo_nlp.get_constraints_scaling()
        comparison_cs = np.asarray([3.1, 3.2, 4.1, 4.2, 1, 2.2], dtype=np.float64)
        check_vectors_specific_order(
            self, cs, c_order, comparison_cs, comparison_c_order
        )

    @unittest.skipIf(not cyipopt_available, "CyIpopt needed to run tests with solve")
    def test_external_greybox_solve_scaling(self):
        m = pyo.ConcreteModel()
        m.mu = pyo.Var(bounds=(0, None), initialize=1)
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(
            ex_models.PressureDropTwoEqualitiesTwoOutputsScaleBoth()
        )
        m.ccon = pyo.Constraint(
            expr=m.egb.inputs['c'] == 128 / (3.14 * 1e-4) * m.mu * m.egb.inputs['F']
        )
        m.pcon = pyo.Constraint(expr=m.egb.inputs['Pin'] - m.egb.outputs['Pout'] <= 72)
        m.pincon = pyo.Constraint(expr=m.egb.inputs['Pin'] == 100.0)
        m.egb.inputs['Pin'].value = 100
        m.egb.inputs['Pin'].setlb(50)
        m.egb.inputs['Pin'].setub(150)
        m.egb.inputs['c'].value = 2
        m.egb.inputs['c'].setlb(1)
        m.egb.inputs['c'].setub(5)
        m.egb.inputs['F'].value = 3
        m.egb.inputs['F'].setlb(1)
        m.egb.inputs['F'].setub(5)
        m.egb.inputs['P1'].value = 80
        m.egb.inputs['P1'].setlb(10)
        m.egb.inputs['P1'].setub(90)
        m.egb.inputs['P3'].value = 70
        m.egb.inputs['P3'].setlb(20)
        m.egb.inputs['P3'].setub(80)
        m.egb.outputs['P2'].value = 75
        m.egb.outputs['P2'].setlb(15)
        m.egb.outputs['P2'].setub(85)
        m.egb.outputs['Pout'].value = 50
        m.egb.outputs['Pout'].setlb(10)
        m.egb.outputs['Pout'].setub(70)
        m.obj = pyo.Objective(
            expr=(m.egb.outputs['Pout'] - 20) ** 2 + (m.egb.inputs['F'] - 3) ** 2
        )

        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.scaling_factor[m.obj] = 0.1  # scale the objective
        m.scaling_factor[m.egb.inputs['Pin']] = 1.1  # scale the variable
        m.scaling_factor[m.egb.inputs['c']] = 1.2  # scale the variable
        m.scaling_factor[m.egb.inputs['F']] = 1.3  # scale the variable
        # m.scaling_factor[m.egb.inputs['P1']] = 1.4 # scale the variable
        m.scaling_factor[m.egb.inputs['P3']] = 1.5  # scale the variable
        m.scaling_factor[m.egb.outputs['P2']] = 1.6  # scale the variable
        m.scaling_factor[m.egb.outputs['Pout']] = 1.7  # scale the variable
        m.scaling_factor[m.mu] = 1.9
        m.scaling_factor[m.pincon] = 2.2

        with TempfileManager.new_context() as temp:
            logfile = temp.create_tempfile('_cyipopt-external-greybox-scaling.log')
            solver = pyo.SolverFactory('cyipopt')
            solver.config.options = {
                'hessian_approximation': 'limited-memory',
                'nlp_scaling_method': 'user-scaling',
                'output_file': logfile,
                'file_print_level': 10,
                'max_iter': 0,
            }
            status = solver.solve(m, tee=False)

            with open(logfile, 'r') as fd:
                solver_trace = fd.read()

        self.assertIn('nlp_scaling_method = user-scaling', solver_trace)
        self.assertIn(f'output_file = {logfile}', solver_trace)
        self.assertIn('objective scaling factor = 0.1', solver_trace)
        self.assertIn('x scaling provided', solver_trace)
        self.assertIn('c scaling provided', solver_trace)
        self.assertIn('d scaling provided', solver_trace)
        # x order: ['egb.inputs[F]', 'mu', 'egb.outputs[Pout]', 'egb.inputs[Pin]', 'egb.inputs[c]', 'egb.inputs[P1]', 'egb.inputs[P3]', 'egb.outputs[P2]']
        # c order: ['ccon', 'pcon', 'pincon', 'egb.pdrop1', 'egb.pdrop3', 'egb.P2_con', 'egb.Pout_con']
        self.assertIn('DenseVector "x scaling vector" with 8 elements:', solver_trace)
        self.assertIn(
            'x scaling vector[    1]= 1.3000000000000000e+00', solver_trace
        )  # F
        self.assertIn(
            'x scaling vector[    8]= 1.8999999999999999e+00', solver_trace
        )  # mu
        self.assertIn(
            'x scaling vector[    7]= 1.7000000000000000e+00', solver_trace
        )  # Pout
        self.assertIn(
            'x scaling vector[    4]= 1.1000000000000001e+00', solver_trace
        )  # Pin
        self.assertIn(
            'x scaling vector[    5]= 1.2000000000000000e+00', solver_trace
        )  # c
        self.assertIn(
            'x scaling vector[    2]= 1.0000000000000000e+00', solver_trace
        )  # P1
        self.assertIn(
            'x scaling vector[    3]= 1.5000000000000000e+00', solver_trace
        )  # P3
        self.assertIn(
            'x scaling vector[    6]= 1.6000000000000001e+00', solver_trace
        )  # P2
        self.assertIn('DenseVector "c scaling vector" with 6 elements:', solver_trace)
        self.assertIn(
            'c scaling vector[    1]= 1.0000000000000000e+00', solver_trace
        )  # ccon
        self.assertIn(
            'c scaling vector[    2]= 2.2000000000000002e+00', solver_trace
        )  # pincon
        self.assertIn(
            'c scaling vector[    3]= 3.1000000000000001e+00', solver_trace
        )  # pdrop1
        self.assertIn(
            'c scaling vector[    4]= 3.2000000000000002e+00', solver_trace
        )  # pdrop3
        self.assertIn(
            'c scaling vector[    5]= 4.0999999999999996e+00', solver_trace
        )  # P2_con
        self.assertIn(
            'c scaling vector[    6]= 4.2000000000000002e+00', solver_trace
        )  # Pout_con
        self.assertIn('DenseVector "d scaling vector" with 1 elements:', solver_trace)
        self.assertIn(
            'd scaling vector[    1]= 1.0000000000000000e+00', solver_trace
        )  # pcon


if __name__ == '__main__':
    TestPyomoGreyBoxNLP().test_external_greybox_solve()
