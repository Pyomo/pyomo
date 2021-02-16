#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
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

from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
    CyIpoptSolver, CyIpoptNLP, ipopt, ipopt_available,
)

from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import _ExternalGreyBoxAsNLP
from pyomo.contrib.pynumero.interfaces.tests.compare_utils import check_vectors_specific_order, check_sparse_matrix_specific_order
import pyomo.contrib.pynumero.interfaces.tests._external_grey_box_models as ex_models

class Test_ExternalGreyBoxAsNLP(unittest.TestCase):
    def test_pressure_drop_single_output(self):
        self._test_pressure_drop_single_output(ex_models.PressureDropSingleOutput(),False)
        self._test_pressure_drop_single_output(ex_models.PressureDropSingleOutputWithHessian(),True)

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
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout']-20)**2)
        egb_nlp = _ExternalGreyBoxAsNLP(m.egb)

        self.assertEqual(4, egb_nlp.n_primals())
        self.assertEqual(1, egb_nlp.n_constraints())
        self.assertEqual(4, egb_nlp.nnz_jacobian())
        if hessian_support:
            self.assertEqual(3, egb_nlp.nnz_hessian_lag())

        comparison_x_order = ['egb.inputs[Pin]', 'egb.inputs[c]', 'egb.inputs[F]', 'egb.outputs[Pout]']
        x_order = egb_nlp.primals_names()
        comparison_c_order = ['egb.output_constraints[Pout]']
        c_order = egb_nlp.constraint_names()

        xlb = egb_nlp.primals_lb()
        comparison_xlb = np.asarray([50, 1, 1, 0], dtype=np.float64)
        check_vectors_specific_order(self, xlb, x_order, comparison_xlb, comparison_x_order)
        xub = egb_nlp.primals_ub()
        comparison_xub = np.asarray([150, 5, 5, 100], dtype=np.float64)
        check_vectors_specific_order(self, xub, x_order, comparison_xub, comparison_x_order)
        clb = egb_nlp.constraints_lb()
        comparison_clb = np.asarray([0], dtype=np.float64)
        check_vectors_specific_order(self, clb, c_order, comparison_clb, comparison_c_order)
        cub = egb_nlp.constraints_ub()
        comparison_cub = np.asarray([0], dtype=np.float64)
        check_vectors_specific_order(self, cub, c_order, comparison_cub, comparison_c_order)

        xinit = egb_nlp.init_primals()
        comparison_xinit = np.asarray([100, 2, 3, 50], dtype=np.float64)
        check_vectors_specific_order(self, xinit, x_order, comparison_xinit, comparison_x_order)
        duals_init = egb_nlp.init_duals()
        comparison_duals_init = np.asarray([0], dtype=np.float64)
        check_vectors_specific_order(self, duals_init, c_order, comparison_duals_init, comparison_c_order)

        self.assertEqual(4, len(egb_nlp.create_new_vector('primals')))
        self.assertEqual(1, len(egb_nlp.create_new_vector('constraints')))
        self.assertEqual(1, len(egb_nlp.create_new_vector('duals')))

        egb_nlp.set_primals(np.asarray([1, 2, 3, 4], dtype=np.float64))
        x = egb_nlp.get_primals()
        self.assertTrue(np.array_equal(x, np.asarray([1,2,3,4], dtype=np.float64)))
        egb_nlp.set_primals(egb_nlp.init_primals())

        egb_nlp.set_duals(np.asarray([42], dtype=np.float64))
        y = egb_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([42], dtype=np.float64)))
        egb_nlp.set_duals(np.asarray([21], dtype=np.float64))
        y = egb_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([21], dtype=np.float64)))

        with self.assertRaises(NotImplementedError):
            fac = egb_nlp.get_obj_factor()
        with self.assertRaises(NotImplementedError):
            egb_nlp.set_obj_factor(42)
        with self.assertRaises(NotImplementedError):
            egb_nlp.set_obj_factor(1)
        with self.assertRaises(NotImplementedError):
            f = egb_nlp.evaluate_objective()
        with self.assertRaises(NotImplementedError):
            gradf = egb_nlp.evaluate_grad_objective()

        c = egb_nlp.evaluate_constraints()
        comparison_c = np.asarray([-22], dtype=np.float64)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        c = np.zeros(1, dtype=np.float64)
        egb_nlp.evaluate_constraints(out=c)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        
        j = egb_nlp.evaluate_jacobian()
        comparison_j = np.asarray([[1, -36, -48, -1]])
        check_sparse_matrix_specific_order(self, j, c_order, x_order, comparison_j, comparison_c_order, comparison_x_order)

        j = 2.0*j
        egb_nlp.evaluate_jacobian(out=j)
        check_sparse_matrix_specific_order(self, j, c_order, x_order, comparison_j, comparison_c_order, comparison_x_order)

        if hessian_support:
            h = egb_nlp.evaluate_hessian_lag()
            self.assertTrue(h.shape == (4,4))
            # hessian should be "full", not lower or upper triangular
            comparison_h = np.asarray([[0, 0, 0, 0],[0, 0, -8*3*21, 0], [0, -8*3*21, -8*2*21, 0], [0, 0, 0, 0]], dtype=np.float64)
            check_sparse_matrix_specific_order(self, h, x_order, x_order, comparison_h, comparison_x_order, comparison_x_order)
        else:
            with self.assertRaises(AttributeError):
                h = egb_nlp.evaluate_hessian_lag()

    def test_pressure_drop_single_equality(self):
        self._test_pressure_drop_single_equality(ex_models.PressureDropSingleEquality(), False)
        self._test_pressure_drop_single_equality(ex_models.PressureDropSingleEqualityWithHessian(), True)

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
        m.obj = pyo.Objective(expr=(m.egb.inputs['Pout']-20)**2)
        egb_nlp = _ExternalGreyBoxAsNLP(m.egb)

        self.assertEqual(4, egb_nlp.n_primals())
        self.assertEqual(1, egb_nlp.n_constraints())
        self.assertEqual(4, egb_nlp.nnz_jacobian())
        if hessian_support:
            self.assertEqual(3, egb_nlp.nnz_hessian_lag())

        comparison_x_order = ['egb.inputs[Pin]', 'egb.inputs[c]', 'egb.inputs[F]', 'egb.inputs[Pout]']
        x_order = egb_nlp.primals_names()
        comparison_c_order = ['egb.pdrop']
        c_order = egb_nlp.constraint_names()

        xlb = egb_nlp.primals_lb()
        comparison_xlb = np.asarray([50, 1, 1, 0], dtype=np.float64)
        check_vectors_specific_order(self, xlb, x_order, comparison_xlb, comparison_x_order)
        xub = egb_nlp.primals_ub()
        comparison_xub = np.asarray([150, 5, 5, 100], dtype=np.float64)
        check_vectors_specific_order(self, xub, x_order, comparison_xub, comparison_x_order)
        clb = egb_nlp.constraints_lb()
        comparison_clb = np.asarray([0], dtype=np.float64)
        check_vectors_specific_order(self, clb, c_order, comparison_clb, comparison_c_order)
        cub = egb_nlp.constraints_ub()
        comparison_cub = np.asarray([0], dtype=np.float64)
        check_vectors_specific_order(self, cub, c_order, comparison_cub, comparison_c_order)

        xinit = egb_nlp.init_primals()
        comparison_xinit = np.asarray([100, 2, 3, 50], dtype=np.float64)
        check_vectors_specific_order(self, xinit, x_order, comparison_xinit, comparison_x_order)
        duals_init = egb_nlp.init_duals()
        comparison_duals_init = np.asarray([0], dtype=np.float64)
        check_vectors_specific_order(self, duals_init, c_order, comparison_duals_init, comparison_c_order)

        self.assertEqual(4, len(egb_nlp.create_new_vector('primals')))
        self.assertEqual(1, len(egb_nlp.create_new_vector('constraints')))
        self.assertEqual(1, len(egb_nlp.create_new_vector('duals')))

        egb_nlp.set_primals(np.asarray([1, 2, 3, 4], dtype=np.float64))
        x = egb_nlp.get_primals()
        self.assertTrue(np.array_equal(x, np.asarray([1,2,3,4], dtype=np.float64)))
        egb_nlp.set_primals(egb_nlp.init_primals())

        egb_nlp.set_duals(np.asarray([42], dtype=np.float64))
        y = egb_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([42], dtype=np.float64)))
        egb_nlp.set_duals(np.asarray([21], dtype=np.float64))
        y = egb_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([21], dtype=np.float64)))

        with self.assertRaises(NotImplementedError):
            fac = egb_nlp.get_obj_factor()
        with self.assertRaises(NotImplementedError):
            egb_nlp.set_obj_factor(42)
        with self.assertRaises(NotImplementedError):
            egb_nlp.set_obj_factor(1)
        with self.assertRaises(NotImplementedError):
            f = egb_nlp.evaluate_objective()
        with self.assertRaises(NotImplementedError):
            gradf = egb_nlp.evaluate_grad_objective()

        c = egb_nlp.evaluate_constraints()
        comparison_c = np.asarray([22], dtype=np.float64)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        c = np.zeros(1, dtype=np.float64)
        egb_nlp.evaluate_constraints(out=c)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)

        j = egb_nlp.evaluate_jacobian()
        comparison_j = np.asarray([[-1, 36, 48, 1]])
        check_sparse_matrix_specific_order(self, j, c_order, x_order, comparison_j, comparison_c_order, comparison_x_order)

        j = 2.0*j
        egb_nlp.evaluate_jacobian(out=j)
        check_sparse_matrix_specific_order(self, j, c_order, x_order, comparison_j, comparison_c_order, comparison_x_order)

        if hessian_support:
            h = egb_nlp.evaluate_hessian_lag()
            self.assertTrue(h.shape == (4,4))
            comparison_h = np.asarray([[0, 0, 0, 0],[0, 0,  8*3*21, 0], [0, 8*3*21, 8*2*21, 0], [0, 0, 0, 0]], dtype=np.float64)
            check_sparse_matrix_specific_order(self, h, x_order, x_order, comparison_h, comparison_x_order, comparison_x_order)
        else:
            with self.assertRaises(AttributeError):
                h = egb_nlp.evaluate_hessian_lag()

    def test_pressure_drop_two_outputs(self):
        self._test_pressure_drop_two_outputs(ex_models.PressureDropTwoOutputs(), False)
        self._test_pressure_drop_two_outputs(ex_models.PressureDropTwoOutputsWithHessian(), True)

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
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout']-20)**2)
        egb_nlp = _ExternalGreyBoxAsNLP(m.egb)

        self.assertEqual(5, egb_nlp.n_primals())
        self.assertEqual(2, egb_nlp.n_constraints())
        self.assertEqual(8, egb_nlp.nnz_jacobian())
        if hessian_support:
            self.assertEqual(3, egb_nlp.nnz_hessian_lag())

        comparison_x_order = ['egb.inputs[Pin]', 'egb.inputs[c]', 'egb.inputs[F]', 'egb.outputs[P2]', 'egb.outputs[Pout]']
        x_order = egb_nlp.primals_names()
        comparison_c_order = ['egb.output_constraints[P2]', 'egb.output_constraints[Pout]']
        c_order = egb_nlp.constraint_names()

        xlb = egb_nlp.primals_lb()
        comparison_xlb = np.asarray([50, 1, 1, 10, 0], dtype=np.float64)
        check_vectors_specific_order(self, xlb, x_order, comparison_xlb, comparison_x_order)
        xub = egb_nlp.primals_ub()
        comparison_xub = np.asarray([150, 5, 5, 90, 100], dtype=np.float64)
        check_vectors_specific_order(self, xub, x_order, comparison_xub, comparison_x_order)
        clb = egb_nlp.constraints_lb()
        comparison_clb = np.asarray([0, 0], dtype=np.float64)
        check_vectors_specific_order(self, clb, c_order, comparison_clb, comparison_c_order)
        cub = egb_nlp.constraints_ub()
        comparison_cub = np.asarray([0, 0], dtype=np.float64)
        check_vectors_specific_order(self, cub, c_order, comparison_cub, comparison_c_order)

        xinit = egb_nlp.init_primals()
        comparison_xinit = np.asarray([100, 2, 3, 80, 50], dtype=np.float64)
        check_vectors_specific_order(self, xinit, x_order, comparison_xinit, comparison_x_order)
        duals_init = egb_nlp.init_duals()
        comparison_duals_init = np.asarray([0, 0], dtype=np.float64)
        check_vectors_specific_order(self, duals_init, c_order, comparison_duals_init, comparison_c_order)

        self.assertEqual(5, len(egb_nlp.create_new_vector('primals')))
        self.assertEqual(2, len(egb_nlp.create_new_vector('constraints')))
        self.assertEqual(2, len(egb_nlp.create_new_vector('duals')))

        egb_nlp.set_primals(np.asarray([1, 2, 3, 4, 5], dtype=np.float64))
        x = egb_nlp.get_primals()
        self.assertTrue(np.array_equal(x, np.asarray([1,2,3,4,5], dtype=np.float64)))
        egb_nlp.set_primals(egb_nlp.init_primals())

        egb_nlp.set_duals(np.asarray([42, 10], dtype=np.float64))
        y = egb_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([42, 10], dtype=np.float64)))
        egb_nlp.set_duals(np.asarray([21, 5], dtype=np.float64))
        y = egb_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([21, 5], dtype=np.float64)))

        with self.assertRaises(NotImplementedError):
            fac = egb_nlp.get_obj_factor()
        with self.assertRaises(NotImplementedError):
            egb_nlp.set_obj_factor(42)
        with self.assertRaises(NotImplementedError):
            egb_nlp.set_obj_factor(1)
        with self.assertRaises(NotImplementedError):
            f = egb_nlp.evaluate_objective()
        with self.assertRaises(NotImplementedError):
            gradf = egb_nlp.evaluate_grad_objective()

        c = egb_nlp.evaluate_constraints()
        comparison_c = np.asarray([-16, -22], dtype=np.float64)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        c = np.zeros(2)
        egb_nlp.evaluate_constraints(out=c)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)

        j = egb_nlp.evaluate_jacobian()
        comparison_j = np.asarray([[1, -18, -24, -1, 0], [1, -36, -48, 0, -1]])
        check_sparse_matrix_specific_order(self, j, c_order, x_order, comparison_j, comparison_c_order, comparison_x_order)

        j = 2.0*j
        egb_nlp.evaluate_jacobian(out=j)
        check_sparse_matrix_specific_order(self, j, c_order, x_order, comparison_j, comparison_c_order, comparison_x_order)

        if hessian_support:
            h = egb_nlp.evaluate_hessian_lag()
            self.assertTrue(h.shape == (5,5))
            comparison_h = np.asarray([[0, 0, 0, 0, 0],
                                       [0, 0, (-4*3*21) + (-8*3*5), 0, 0],
                                       [0, (-4*3*21) + (-8*3*5), (-4*2*21) + (-8*2*5), 0, 0],
                                       [0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0]], dtype=np.float64)
            check_sparse_matrix_specific_order(self, h, x_order, x_order, comparison_h, comparison_x_order, comparison_x_order)
        else:
            with self.assertRaises(AttributeError):
                h = egb_nlp.evaluate_hessian_lag()

    def test_pressure_drop_two_equalities(self):
        self._test_pressure_drop_two_equalities(ex_models.PressureDropTwoEqualities(), False)
        self._test_pressure_drop_two_equalities(ex_models.PressureDropTwoEqualitiesWithHessian(), True)

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
        m.obj = pyo.Objective(expr=(m.egb.inputs['Pout']-20)**2)
        egb_nlp = _ExternalGreyBoxAsNLP(m.egb)

        self.assertEqual(5, egb_nlp.n_primals())
        self.assertEqual(2, egb_nlp.n_constraints())
        self.assertEqual(8, egb_nlp.nnz_jacobian())
        if hessian_support:
            self.assertEqual(3, egb_nlp.nnz_hessian_lag())

        comparison_x_order = ['egb.inputs[Pin]', 'egb.inputs[c]', 'egb.inputs[F]', 'egb.inputs[P2]', 'egb.inputs[Pout]']
        x_order = egb_nlp.primals_names()
        comparison_c_order = ['egb.pdrop2', 'egb.pdropout']
        c_order = egb_nlp.constraint_names()

        xlb = egb_nlp.primals_lb()
        comparison_xlb = np.asarray([50, 1, 1, 10, 0], dtype=np.float64)
        check_vectors_specific_order(self, xlb, x_order, comparison_xlb, comparison_x_order)
        xub = egb_nlp.primals_ub()
        comparison_xub = np.asarray([150, 5, 5, 90, 100], dtype=np.float64)
        check_vectors_specific_order(self, xub, x_order, comparison_xub, comparison_x_order)
        clb = egb_nlp.constraints_lb()
        comparison_clb = np.asarray([0, 0], dtype=np.float64)
        check_vectors_specific_order(self, clb, c_order, comparison_clb, comparison_c_order)
        cub = egb_nlp.constraints_ub()
        comparison_cub = np.asarray([0, 0], dtype=np.float64)
        check_vectors_specific_order(self, cub, c_order, comparison_cub, comparison_c_order)

        xinit = egb_nlp.init_primals()
        comparison_xinit = np.asarray([100, 2, 3, 80, 50], dtype=np.float64)
        check_vectors_specific_order(self, xinit, x_order, comparison_xinit, comparison_x_order)
        duals_init = egb_nlp.init_duals()
        comparison_duals_init = np.asarray([0, 0], dtype=np.float64)
        check_vectors_specific_order(self, duals_init, c_order, comparison_duals_init, comparison_c_order)

        self.assertEqual(5, len(egb_nlp.create_new_vector('primals')))
        self.assertEqual(2, len(egb_nlp.create_new_vector('constraints')))
        self.assertEqual(2, len(egb_nlp.create_new_vector('duals')))

        egb_nlp.set_primals(np.asarray([1, 2, 3, 4, 5], dtype=np.float64))
        x = egb_nlp.get_primals()
        self.assertTrue(np.array_equal(x, np.asarray([1,2,3,4,5], dtype=np.float64)))
        egb_nlp.set_primals(egb_nlp.init_primals())

        egb_nlp.set_duals(np.asarray([42, 10], dtype=np.float64))
        y = egb_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([42, 10], dtype=np.float64)))
        egb_nlp.set_duals(np.asarray([21, 5], dtype=np.float64))
        y = egb_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([21, 5], dtype=np.float64)))

        with self.assertRaises(NotImplementedError):
            fac = egb_nlp.get_obj_factor()
        with self.assertRaises(NotImplementedError):
            egb_nlp.set_obj_factor(42)
        with self.assertRaises(NotImplementedError):
            egb_nlp.set_obj_factor(1)
        with self.assertRaises(NotImplementedError):
            f = egb_nlp.evaluate_objective()
        with self.assertRaises(NotImplementedError):
            gradf = egb_nlp.evaluate_grad_objective()

        c = egb_nlp.evaluate_constraints()
        comparison_c = np.asarray([16, 6], dtype=np.float64)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        c = np.zeros(2)
        egb_nlp.evaluate_constraints(out=c)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        
        j = egb_nlp.evaluate_jacobian()
        comparison_j = np.asarray([[-1, 18, 24, 1, 0], [0, 18, 24, -1, 1]])
        check_sparse_matrix_specific_order(self, j, c_order, x_order, comparison_j, comparison_c_order, comparison_x_order)

        j = 2.0*j
        egb_nlp.evaluate_jacobian(out=j)
        check_sparse_matrix_specific_order(self, j, c_order, x_order, comparison_j, comparison_c_order, comparison_x_order)

        if hessian_support:
            h = egb_nlp.evaluate_hessian_lag()
            self.assertTrue(h.shape == (5,5))
            comparison_h = np.asarray([[0, 0, 0, 0, 0],
                                       [0, 0, (4*3*21) + (4*3*5), 0, 0],
                                       [0, (4*3*21) + (4*3*5), (4*2*21) + (4*2*5), 0, 0],
                                       [0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0]], dtype=np.float64)
            check_sparse_matrix_specific_order(self, h, x_order, x_order, comparison_h, comparison_x_order, comparison_x_order)
        else:
            with self.assertRaises(AttributeError):
                h = egb_nlp.evaluate_hessian_lag()

    def test_pressure_drop_two_equalities_two_outputs(self):
        self._test_pressure_drop_two_equalities_two_outputs(ex_models.PressureDropTwoEqualitiesTwoOutputs(), False)
        self._test_pressure_drop_two_equalities_two_outputs(ex_models.PressureDropTwoEqualitiesTwoOutputsWithHessian(), True)

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
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout']-20)**2)
        egb_nlp = _ExternalGreyBoxAsNLP(m.egb)

        self.assertEqual(7, egb_nlp.n_primals())
        self.assertEqual(4, egb_nlp.n_constraints())
        self.assertEqual(16, egb_nlp.nnz_jacobian())
        if hessian_support:
            # this number is larger than expected because the nnz for the
            # hessian of equality and output constraints are concatenated
            # even if they occur in the same place
            self.assertEqual(6, egb_nlp.nnz_hessian_lag())

        comparison_x_order = ['egb.inputs[Pin]', 'egb.inputs[c]', 'egb.inputs[F]',
                              'egb.inputs[P1]', 'egb.inputs[P3]',
                              'egb.outputs[P2]', 'egb.outputs[Pout]']
        x_order = egb_nlp.primals_names()
        comparison_c_order = ['egb.pdrop1', 'egb.pdrop3', 'egb.output_constraints[P2]', 'egb.output_constraints[Pout]']
        c_order = egb_nlp.constraint_names()

        xlb = egb_nlp.primals_lb()
        comparison_xlb = np.asarray([50, 1, 1, 10, 20, 15, 30], dtype=np.float64)
        check_vectors_specific_order(self, xlb, x_order, comparison_xlb, comparison_x_order)
        xub = egb_nlp.primals_ub()
        comparison_xub = np.asarray([150, 5, 5, 90, 80, 85, 70], dtype=np.float64)
        check_vectors_specific_order(self, xub, x_order, comparison_xub, comparison_x_order)
        clb = egb_nlp.constraints_lb()
        comparison_clb = np.asarray([0, 0, 0, 0], dtype=np.float64)
        check_vectors_specific_order(self, clb, c_order, comparison_clb, comparison_c_order)
        cub = egb_nlp.constraints_ub()
        comparison_cub = np.asarray([0, 0, 0, 0], dtype=np.float64)
        check_vectors_specific_order(self, cub, c_order, comparison_cub, comparison_c_order)

        xinit = egb_nlp.init_primals()
        comparison_xinit = np.asarray([100, 2, 3, 80, 70, 75, 50], dtype=np.float64)
        check_vectors_specific_order(self, xinit, x_order, comparison_xinit, comparison_x_order)
        duals_init = egb_nlp.init_duals()
        comparison_duals_init = np.asarray([0, 0, 0, 0], dtype=np.float64)
        check_vectors_specific_order(self, duals_init, c_order, comparison_duals_init, comparison_c_order)

        self.assertEqual(7, len(egb_nlp.create_new_vector('primals')))
        self.assertEqual(4, len(egb_nlp.create_new_vector('constraints')))
        self.assertEqual(4, len(egb_nlp.create_new_vector('duals')))

        egb_nlp.set_primals(np.asarray([1, 2, 3, 4, 5, 6, 7], dtype=np.float64))
        x = egb_nlp.get_primals()
        self.assertTrue(np.array_equal(x, np.asarray([1,2,3,4,5,6,7], dtype=np.float64)))
        egb_nlp.set_primals(egb_nlp.init_primals())

        egb_nlp.set_duals(np.asarray([42, 10, 11, 12], dtype=np.float64))
        y = egb_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([42, 10, 11, 12], dtype=np.float64)))
        egb_nlp.set_duals(np.asarray([21, 5, 6, 7], dtype=np.float64))
        y = egb_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([21, 5, 6, 7], dtype=np.float64)))

        with self.assertRaises(NotImplementedError):
            fac = egb_nlp.get_obj_factor()
        with self.assertRaises(NotImplementedError):
            egb_nlp.set_obj_factor(42)
        with self.assertRaises(NotImplementedError):
            egb_nlp.set_obj_factor(1)
        with self.assertRaises(NotImplementedError):
            f = egb_nlp.evaluate_objective()
        with self.assertRaises(NotImplementedError):
            gradf = egb_nlp.evaluate_grad_objective()

        c = egb_nlp.evaluate_constraints()
        comparison_c = np.asarray([-2, 26, -13, -22], dtype=np.float64)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        c = np.zeros(4)
        egb_nlp.evaluate_constraints(out=c)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        

        j = egb_nlp.evaluate_jacobian()
        comparison_j = np.asarray([[-1,   9,  12,  1, 0,  0,  0],
                                   [ 0,  18,  24, -1, 1,  0,  0],
                                   [ 0,  -9, -12,  1, 0, -1,  0],
                                   [ 1, -36, -48,  0, 0,  0, -1]])
        check_sparse_matrix_specific_order(self, j, c_order, x_order, comparison_j, comparison_c_order, comparison_x_order)

        j = 2.0*j
        egb_nlp.evaluate_jacobian(out=j)
        check_sparse_matrix_specific_order(self, j, c_order, x_order, comparison_j, comparison_c_order, comparison_x_order)

        if hessian_support:
            h = egb_nlp.evaluate_hessian_lag()
            self.assertTrue(h.shape == (7,7))
            comparison_h = np.asarray([[0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, (2*3*21) + (4*3*5) + (-2*3*6) + (-8*3*7), 0, 0, 0, 0],
                                       [0, (2*3*21) + (4*3*5) + (-2*3*6) + (-8*3*7), (2*2*21) + (4*2*5) + (-2*2*6) + (-8*2*7), 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0]],
                                      dtype=np.float64)
            check_sparse_matrix_specific_order(self, h, x_order, x_order, comparison_h, comparison_x_order, comparison_x_order)
        else:
            with self.assertRaises(AttributeError):
                h = egb_nlp.evaluate_hessian_lag()

    def create_model_two_equalities_two_outputs(self, external_model):
        m = pyo.ConcreteModel()
        m.hin = pyo.Var(bounds=(0,None), initialize=10)
        m.hout = pyo.Var(bounds=(0,None))
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(external_model)
        m.incon = pyo.Constraint(expr= 0 <= m.egb.inputs['Pin'] - 10*m.hin)
        m.outcon = pyo.Constraint(expr= 0 == m.egb.outputs['Pout'] - 10*m.hout)
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
        m = self.create_model_two_equalities_two_outputs(ex_models.PressureDropTwoEqualitiesTwoOutputs())
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout']-20)**2)
        egb_nlp = _ExternalGreyBoxAsNLP(m.egb)
        with self.assertRaises(NotImplementedError):
            fs = egb_nlp.get_obj_scaling()
        with self.assertRaises(NotImplementedError):
            xs = egb_nlp.get_primals_scaling()
        cs = egb_nlp.get_constraints_scaling()
        self.assertIsNone(cs)

    def test_scaling_pyomo_model_only(self):
        m = self.create_model_two_equalities_two_outputs(ex_models.PressureDropTwoEqualitiesTwoOutputs())
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout']-20)**2)
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        #m.scaling_factor[m.obj] = 0.1 # scale the objective
        m.scaling_factor[m.egb.inputs['Pin']] = 1.1 # scale the variable
        m.scaling_factor[m.egb.inputs['c']] = 1.2 # scale the variable
        m.scaling_factor[m.egb.inputs['F']] = 1.3 # scale the variable
        #m.scaling_factor[m.egb.inputs['P1']] = 1.4 # scale the variable
        m.scaling_factor[m.egb.inputs['P3']] = 1.5 # scale the variable
        m.scaling_factor[m.egb.outputs['P2']] = 1.6 # scale the variable
        m.scaling_factor[m.egb.outputs['Pout']] = 1.7 # scale the variable
        #m.scaling_factor[m.hin] = 1.8
        m.scaling_factor[m.hout] = 1.9
        #m.scaling_factor[m.incon] = 2.1
        m.scaling_factor[m.outcon] = 2.2
        egb_nlp = _ExternalGreyBoxAsNLP(m.egb)

        with self.assertRaises(NotImplementedError):
            fs = egb_nlp.get_obj_scaling()
        with self.assertRaises(NotImplementedError):
            xs = egb_nlp.get_primals_scaling()

        cs = egb_nlp.get_constraints_scaling()
        self.assertIsNone(cs)

    def test_scaling_greybox_only(self):
        m = self.create_model_two_equalities_two_outputs(ex_models.PressureDropTwoEqualitiesTwoOutputsScaleBoth())
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout']-20)**2)
        egb_nlp = _ExternalGreyBoxAsNLP(m.egb)

        comparison_c_order = ['egb.pdrop1', 'egb.pdrop3', 'egb.output_constraints[P2]', 'egb.output_constraints[Pout]']
        c_order = egb_nlp.constraint_names()

        with self.assertRaises(NotImplementedError):
            fs = egb_nlp.get_obj_scaling()
        with self.assertRaises(NotImplementedError):
            xs = egb_nlp.get_primals_scaling()

        cs = egb_nlp.get_constraints_scaling()
        comparison_cs = np.asarray([3.1, 3.2, 4.1, 4.2], dtype=np.float64)
        check_vectors_specific_order(self, cs, c_order, comparison_cs, comparison_c_order)

        m = self.create_model_two_equalities_two_outputs(ex_models.PressureDropTwoEqualitiesTwoOutputsScaleEqualities())
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout']-20)**2)
        egb_nlp = _ExternalGreyBoxAsNLP(m.egb)
        cs = egb_nlp.get_constraints_scaling()
        comparison_cs = np.asarray([3.1, 3.2, 1, 1], dtype=np.float64)
        check_vectors_specific_order(self, cs, c_order, comparison_cs, comparison_c_order)

        m = self.create_model_two_equalities_two_outputs(ex_models.PressureDropTwoEqualitiesTwoOutputsScaleOutputs())
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout']-20)**2)
        egb_nlp = _ExternalGreyBoxAsNLP(m.egb)
        cs = egb_nlp.get_constraints_scaling()
        comparison_cs = np.asarray([1, 1, 4.1, 4.2], dtype=np.float64)
        check_vectors_specific_order(self, cs, c_order, comparison_cs, comparison_c_order)



if __name__ == '__main__':
    Test_ExternalGreyBoxAsNLP().test_pressure_drop_single_equality()
    Test_ExternalGreyBoxAsNLP().test_pressure_drop_two_outputs()
    Test_ExternalGreyBoxAsNLP().test_pressure_drop_two_equalities()
    Test_ExternalGreyBoxAsNLP().test_pressure_drop_two_equalities_two_outputs()
    Test_ExternalGreyBoxAsNLP().test_scaling_all_missing()
    Test_ExternalGreyBoxAsNLP().test_scaling_pyomo_model_only()
    Test_ExternalGreyBoxAsNLP().test_scaling_greybox_only()
