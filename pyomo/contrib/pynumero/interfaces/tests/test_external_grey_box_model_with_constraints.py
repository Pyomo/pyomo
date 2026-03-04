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
#
#  Additional contributions Copyright (c) 2026 OLI Systems, Inc.
#  ___________________________________________________________________________


import pyomo.common.unittest as unittest
import pyomo.environ as pyo

from pyomo.contrib.pynumero.dependencies import (
    numpy as np,
    numpy_available,
    scipy_available,
)

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.interfaces.external_grey_box_constraint import (
    ExternalGreyBoxConstraint,
)
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoGreyBoxNLP
from pyomo.contrib.pynumero.interfaces.tests.compare_utils import (
    check_vectors_specific_order,
    check_sparse_matrix_specific_order,
)
import pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models as ex_models
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface


class TestExternalGreyBoxModelWithConstraints(unittest.TestCase):
    """Tests for ExternalGreyBoxBlock with build_implicit_constraint_objects=True"""

    def test_pressure_drop_single_output_constraint_creation(self):
        """Test that constraint objects are created for outputs"""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(
            ex_models.PressureDropSingleOutput(), build_implicit_constraint_objects=True
        )

        # Check that the constraint object was created for the output
        self.assertTrue(hasattr(m.egb, 'Pout_constraint'))
        self.assertIsInstance(m.egb.Pout_constraint, ExternalGreyBoxConstraint)

        # Check that no equality constraint objects were created (no equality constraints)
        egbm = m.egb.get_external_model()
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual(eq_con_names, [])

    def test_pressure_drop_single_equality_constraint_creation(self):
        """Test that constraint objects are created for equality constraints"""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(
            ex_models.PressureDropSingleEquality(),
            build_implicit_constraint_objects=True,
        )

        # Check that the constraint object was created for the equality constraint
        self.assertTrue(hasattr(m.egb, 'pdrop'))
        self.assertIsInstance(m.egb.pdrop, ExternalGreyBoxConstraint)

        # Check that no output constraint objects were created (no outputs)
        egbm = m.egb.get_external_model()
        output_names = egbm.output_names()
        self.assertEqual(output_names, [])

    def test_pressure_drop_two_outputs_constraint_creation(self):
        """Test that constraint objects are created for multiple outputs"""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(
            ex_models.PressureDropTwoOutputs(), build_implicit_constraint_objects=True
        )

        # Check that constraint objects were created for both outputs
        self.assertTrue(hasattr(m.egb, 'P2_constraint'))
        self.assertIsInstance(m.egb.P2_constraint, ExternalGreyBoxConstraint)
        self.assertTrue(hasattr(m.egb, 'Pout_constraint'))
        self.assertIsInstance(m.egb.Pout_constraint, ExternalGreyBoxConstraint)

    def test_pressure_drop_two_equalities_constraint_creation(self):
        """Test that constraint objects are created for multiple equality constraints"""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(
            ex_models.PressureDropTwoEqualities(),
            build_implicit_constraint_objects=True,
        )

        # Check that constraint objects were created for both equality constraints
        self.assertTrue(hasattr(m.egb, 'pdrop2'))
        self.assertIsInstance(m.egb.pdrop2, ExternalGreyBoxConstraint)
        self.assertTrue(hasattr(m.egb, 'pdropout'))
        self.assertIsInstance(m.egb.pdropout, ExternalGreyBoxConstraint)

    def test_pressure_drop_two_equalities_two_outputs_constraint_creation(self):
        """Test that constraint objects are created for both equality constraints and outputs"""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(
            ex_models.PressureDropTwoEqualitiesTwoOutputs(),
            build_implicit_constraint_objects=True,
        )

        # Check that constraint objects were created for equality constraints
        self.assertTrue(hasattr(m.egb, 'pdrop1'))
        self.assertIsInstance(m.egb.pdrop1, ExternalGreyBoxConstraint)
        self.assertTrue(hasattr(m.egb, 'pdrop3'))
        self.assertIsInstance(m.egb.pdrop3, ExternalGreyBoxConstraint)

        # Check that constraint objects were created for outputs
        self.assertTrue(hasattr(m.egb, 'P2_constraint'))
        self.assertIsInstance(m.egb.P2_constraint, ExternalGreyBoxConstraint)
        self.assertTrue(hasattr(m.egb, 'Pout_constraint'))
        self.assertIsInstance(m.egb.Pout_constraint, ExternalGreyBoxConstraint)

    def test_pressure_drop_single_equality_with_constraints(self):
        """Test PyomoGreyBoxNLP with single equality constraint and constraint objects"""
        self._test_pressure_drop_single_equality(
            ex_models.PressureDropSingleEquality(), False
        )
        self._test_pressure_drop_single_equality(
            ex_models.PressureDropSingleEqualityWithHessian(), True
        )

    def _test_pressure_drop_single_equality(self, ex_model, hessian_support):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(ex_model, build_implicit_constraint_objects=True)
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

    def test_pressure_drop_two_equalities_with_constraints(self):
        """Test PyomoGreyBoxNLP with two equality constraints and constraint objects"""
        self._test_pressure_drop_two_equalities(
            ex_models.PressureDropTwoEqualities(), False
        )
        self._test_pressure_drop_two_equalities(
            ex_models.PressureDropTwoEqualitiesWithHessian(), True
        )

    def _test_pressure_drop_two_equalities(self, ex_model, hessian_support):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(ex_model, build_implicit_constraint_objects=True)
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


class TestExternalGreyBoxModelWithConstraints(unittest.TestCase):
    """Tests for integration of ExternalGreyBoxBlock with incidence analysis"""

    def build_model(self):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoEqualitiesTwoOutputsWithHessian()
        m.egb.set_external_model(external_model, build_implicit_constraint_objects=True)

        return m

    def build_model_with_pyomo_components(self):
        m = self.build_model()

        # Add Vars and linking constraints to m
        m.Pin = pyo.Var()
        m.c = pyo.Var()
        m.F = pyo.Var()
        m.P1 = pyo.Var()
        m.P3 = pyo.Var()
        m.P2 = pyo.Var()
        m.Pout = pyo.Var()

        m.link_Pin = pyo.Constraint(expr=m.Pin == m.egb.inputs['Pin'])
        m.link_c = pyo.Constraint(expr=m.c == m.egb.inputs['c'])
        m.link_F = pyo.Constraint(expr=m.F == m.egb.inputs['F'])
        m.link_P1 = pyo.Constraint(expr=m.P1 == m.egb.inputs['P1'])
        m.link_P3 = pyo.Constraint(expr=m.P3 == m.egb.inputs['P3'])
        m.link_P2 = pyo.Constraint(expr=m.P2 == m.egb.outputs['P2'])
        m.link_Pout = pyo.Constraint(expr=m.Pout == m.egb.outputs['Pout'])

        return m

    def test_grey_box_only(self):
        """
        Test that the incidence analysis correctly determines the DM partition for
        a grey box model with two equality constraints and two outputs
        """
        m = self.build_model()

        assert m.egb.has_implicit_constraint_objects

        # Check that the get_incident_variables method on the implicit constraint body returns the correct variables
        # Implicit constraint: 'pdrop1'
        body_obj1 = m.egb.pdrop1.body
        incident_vars1 = body_obj1.get_incident_variables(use_jacobian=False)
        assert len(incident_vars1) == 5
        expected_names = [
            'egb.inputs[Pin]',
            'egb.inputs[c]',
            'egb.inputs[F]',
            'egb.inputs[P1]',
            'egb.inputs[P3]',
        ]
        for v in incident_vars1:
            assert v.name in expected_names

        # Implicit constraint: 'pdrop3'
        body_obj1 = m.egb.pdrop3.body
        incident_vars1 = body_obj1.get_incident_variables(use_jacobian=False)
        assert len(incident_vars1) == 5
        expected_names = [
            'egb.inputs[Pin]',
            'egb.inputs[c]',
            'egb.inputs[F]',
            'egb.inputs[P1]',
            'egb.inputs[P3]',
        ]
        for v in incident_vars1:
            assert v.name in expected_names

        # Implicit constraint: 'P2_constraint'
        body_obj1 = m.egb.P2_constraint.body
        incident_vars1 = body_obj1.get_incident_variables(use_jacobian=False)
        assert len(incident_vars1) == 6
        expected_names = [
            'egb.inputs[Pin]',
            'egb.inputs[c]',
            'egb.inputs[F]',
            'egb.inputs[P1]',
            'egb.inputs[P3]',
            'egb.outputs[P2]',
        ]
        for v in incident_vars1:
            assert v.name in expected_names

        # Implicit constraint: 'Pout_constraint'
        body_obj1 = m.egb.Pout_constraint.body
        incident_vars1 = body_obj1.get_incident_variables(use_jacobian=False)
        assert len(incident_vars1) == 6
        expected_names = [
            'egb.inputs[Pin]',
            'egb.inputs[c]',
            'egb.inputs[F]',
            'egb.inputs[P1]',
            'egb.inputs[P3]',
            'egb.outputs[Pout]',
        ]
        for v in incident_vars1:
            assert v.name in expected_names

        # Check Dulmage-Mendelsohn partitioning of the incidence graph
        igraph = IncidenceGraphInterface(m, include_inequality=False)
        var_dm_partition, con_dm_partition = igraph.dulmage_mendelsohn()

        # In this case, as we have not fixed any variables, we expect the system to be under-constrained.
        # All variables will be in the under-constrained or unmatched sets
        # All constraints will be in the under-constrained set
        assert var_dm_partition.overconstrained == []
        assert var_dm_partition.square == []
        assert con_dm_partition.unmatched == []
        assert con_dm_partition.overconstrained == []
        assert con_dm_partition.square == []

        assert len(var_dm_partition.underconstrained) == 4
        assert len(var_dm_partition.unmatched) == 3
        var_names = [
            v.name
            for v in var_dm_partition.underconstrained + var_dm_partition.unmatched
        ]
        for v in var_names:
            assert v in [
                'egb.inputs[Pin]',
                'egb.inputs[c]',
                'egb.inputs[F]',
                'egb.inputs[P1]',
                'egb.inputs[P3]',
                'egb.outputs[P2]',
                'egb.outputs[Pout]',
            ]

        assert len(con_dm_partition.underconstrained) == 4
        con_names = [c.name for c in con_dm_partition.underconstrained]
        for c in con_names:
            assert c in [
                'egb.pdrop1',
                'egb.pdrop3',
                'egb.P2_constraint',
                'egb.Pout_constraint',
            ]

    def test_grey_box_w_pyomo_components(self):
        """
        Test that the incidence analysis correctly determines the DM partition for
        a model containing both grey box and other components
        """
        m = self.build_model_with_pyomo_components()
        assert m.egb.has_implicit_constraint_objects

        # Check Dulmage-Mendelsohn partitioning of the incidence graph
        igraph = IncidenceGraphInterface(m, include_inequality=False)
        var_dm_partition, con_dm_partition = igraph.dulmage_mendelsohn()

        # In this case, as we have not fixed any variables, we expect the system to be under-constrained.
        # All variables will be in the under-constrained or unmatched sets
        # All constraints will be in the under-constrained set
        assert var_dm_partition.overconstrained == []
        assert var_dm_partition.square == []
        assert con_dm_partition.unmatched == []
        assert con_dm_partition.overconstrained == []
        assert con_dm_partition.square == []

        assert len(var_dm_partition.underconstrained) == 11
        assert len(var_dm_partition.unmatched) == 3
        var_names = [
            v.name
            for v in var_dm_partition.underconstrained + var_dm_partition.unmatched
        ]
        for v in var_names:
            assert v in [
                'egb.inputs[Pin]',
                'egb.inputs[c]',
                'egb.inputs[F]',
                'egb.inputs[P1]',
                'egb.inputs[P3]',
                'egb.outputs[P2]',
                'egb.outputs[Pout]',
                'Pin',
                'c',
                'F',
                'P1',
                'P3',
                'P2',
                'Pout',
            ]

        assert len(con_dm_partition.underconstrained) == 11
        con_names = [c.name for c in con_dm_partition.underconstrained]
        for c in con_names:
            assert c in [
                'egb.pdrop1',
                'egb.pdrop3',
                'egb.P2_constraint',
                'egb.Pout_constraint',
                'link_Pin',
                'link_c',
                'link_F',
                'link_P1',
                'link_P3',
                'link_P2',
                'link_Pout',
            ]

    def test_grey_box_w_pyomo_components_square(self):
        """
        Test that the incidence analysis correctly determines the DM partition for
        a model containing both grey box and other components
        """
        m = self.build_model_with_pyomo_components()

        # Fix 3 inputs
        # Note that we have 2 implicit constraints that cross-link inputs
        m.Pin.fix(1)
        m.c.fix(1)
        m.F.fix(1)

        # Check Dulmage-Mendelsohn partitioning of the incidence graph
        igraph = IncidenceGraphInterface(m, include_inequality=False)
        var_dm_partition, con_dm_partition = igraph.dulmage_mendelsohn()

        # In this case, as we have fixed all input variables, we expect the system to be square.
        # All variables and constraints will be in the square sets
        assert var_dm_partition.unmatched == []
        assert var_dm_partition.overconstrained == []
        assert var_dm_partition.underconstrained == []
        assert con_dm_partition.unmatched == []
        assert con_dm_partition.overconstrained == []
        assert con_dm_partition.underconstrained == []

        assert len(var_dm_partition.square) == 11
        var_names = [v.name for v in var_dm_partition.square]
        for v in var_names:
            assert v in [
                'egb.inputs[Pin]',
                'egb.inputs[c]',
                'egb.inputs[F]',
                'egb.inputs[P1]',
                'egb.inputs[P3]',
                'egb.outputs[P2]',
                'egb.outputs[Pout]',
                # These three re fixed, so do not appear by default
                # 'Pin',
                # 'c',
                # 'F',
                'P1',
                'P3',
                'P2',
                'Pout',
            ]

        assert len(con_dm_partition.square) == 11
        con_names = [c.name for c in con_dm_partition.square]
        for c in con_names:
            assert c in [
                'egb.pdrop1',
                'egb.pdrop3',
                'egb.P2_constraint',
                'egb.Pout_constraint',
                'link_Pin',
                'link_c',
                'link_F',
                'link_P1',
                'link_P3',
                'link_P2',
                'link_Pout',
            ]


if __name__ == '__main__':
    unittest.main()
