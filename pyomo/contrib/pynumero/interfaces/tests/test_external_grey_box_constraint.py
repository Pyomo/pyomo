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

from pyomo.contrib.pynumero.dependencies import numpy as np

from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.interfaces.external_grey_box_constraint import (
    ExternalGreyBoxConstraint,
    ScalarExternalGreyBoxConstraint,
    EGBConstraintBody,
)
import pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models as ex_models


class TestExternalGreyBoxConstraintConstruction(unittest.TestCase):
    """Test construction and initialization of ExternalGreyBoxConstraint."""

    def test_construction_without_implicit_constraint_id_raises(self):
        """Test that constructing without implicit_constraint_id raises ValueError."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        with self.assertRaises(ValueError) as context:
            m.egb.c = ExternalGreyBoxConstraint()
        self.assertIn("implicit_constraint_id", str(context.exception))

    def test_construction_with_rule_raises(self):
        """Test that passing 'rule' argument raises TypeError."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        with self.assertRaises(TypeError) as context:
            m.egb.c = ExternalGreyBoxConstraint(
                implicit_constraint_id='pdrop', rule=lambda m: None
            )
        self.assertIn("rule", str(context.exception))

    def test_construction_with_expr_raises(self):
        """Test that passing 'expr' argument raises TypeError."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        with self.assertRaises(TypeError) as context:
            m.egb.c = ExternalGreyBoxConstraint(
                implicit_constraint_id='pdrop', expr=pyo.Constraint.Skip
            )
        self.assertIn("expr", str(context.exception))

    def test_construction_with_invalid_implicit_constraint_id_raises(self):
        """Test that invalid implicit_constraint_id raises ValueError on construct."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        # Construction happens automatically when adding to block
        with self.assertRaises(ValueError) as context:
            m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='nonexistent')
        self.assertIn("does not exist", str(context.exception))

    def test_construction_not_in_external_grey_box_block_raises(self):
        """Test that construction outside ExternalGreyBoxBlock raises ValueError."""
        m = pyo.ConcreteModel()

        # Construction happens automatically when added to block
        with self.assertRaises(ValueError) as context:
            m.c = ExternalGreyBoxConstraint(implicit_constraint_id='test')
        self.assertIn("ExternalGreyBoxBlock", str(context.exception))

    def test_scalar_construction_with_equality_constraint(self):
        """Test scalar constraint construction with equality constraint."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        self.assertIsInstance(m.egb.c, ScalarExternalGreyBoxConstraint)
        self.assertEqual(m.egb.c.implicit_constraint_id, 'pdrop')
        self.assertTrue(m.egb.c.active)

    def test_scalar_construction_with_output(self):
        """Test scalar constraint construction with output variable."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleOutput()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='Pout')

        self.assertIsInstance(m.egb.c, ScalarExternalGreyBoxConstraint)
        self.assertEqual(m.egb.c.implicit_constraint_id, 'Pout')


class TestExternalGreyBoxConstraintProperties(unittest.TestCase):
    """Test properties of ExternalGreyBoxConstraint."""

    def test_body_with_equality_constraint(self):
        """Test body property returns EGBConstraintBody that evaluates to residual."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        # Set input values directly on external model: Pin=100, c=2, F=3, Pout=50
        # Expected residual: Pout - (Pin - 4*c*F^2) = 50 - (100 - 4*2*9) = 50 - 28 = 22
        external_model.set_input_values(np.asarray([100, 2, 3, 50], dtype=np.float64))

        body_value = pyo.value(m.egb.c.body)
        self.assertAlmostEqual(body_value, 22.0, places=6)

    def test_body_with_output(self):
        """Test body property returns EGBConstraintBody that evaluates residual for outputs."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleOutput()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='Pout')

        # Set input values directly on external model
        # Pin=100, c=2, F=3 => Pout_evaluated = 100 - 4*2*3^2 = 100 - 72 = 28
        external_model.set_input_values(np.asarray([100, 2, 3], dtype=np.float64))

        # Set output variable to match the evaluated value
        m.egb.outputs['Pout'].set_value(28.0)

        # For outputs, when variable matches evaluated value, residual should be 0.0
        body_value = pyo.value(m.egb.c.body)
        self.assertAlmostEqual(body_value, 0.0, places=6)

    def test_body_with_invalid_constraint_id_raises(self):
        """Test body property raises ValueError for invalid constraint ID."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        # Create constraint with valid id, then manually change it
        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')
        m.egb.c._implicit_constraint_id = 'invalid_id'

        with self.assertRaises(ValueError) as context:
            _ = m.egb.c.body
        self.assertIn("invalid_id", str(context.exception))

    def test_body_without_inputs_set_evaluates_with_defaults(self):
        """Test body property evaluates with default zero inputs when not explicitly set."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        # External model initializes with zeros, so evaluation should work
        # Expected: 0 - (0 - 4*0*0) = 0
        body_value = pyo.value(m.egb.c.body)
        self.assertAlmostEqual(body_value, 0.0, places=6)

    def test_lower_property(self):
        """Test lower bound is always 0.0."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        self.assertEqual(m.egb.c.lower, 0.0)

    def test_upper_property(self):
        """Test upper bound is always 0.0."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        self.assertEqual(m.egb.c.upper, 0.0)

    def test_lb_property(self):
        """Test lb (lower bound value) is always 0.0."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        self.assertEqual(m.egb.c.lb, 0.0)

    def test_ub_property(self):
        """Test ub (upper bound value) is always 0.0."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        self.assertEqual(m.egb.c.ub, 0.0)

    def test_equality_property(self):
        """Test equality property is always True."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        self.assertTrue(m.egb.c.equality)

    def test_strict_lower_property(self):
        """Test strict_lower is always False."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        self.assertFalse(m.egb.c.strict_lower)

    def test_strict_upper_property(self):
        """Test strict_upper is always False."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        self.assertFalse(m.egb.c.strict_upper)

    def test_has_lb_method(self):
        """Test has_lb() returns True."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        self.assertTrue(m.egb.c.has_lb())

    def test_has_ub_method(self):
        """Test has_ub() returns True."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        self.assertTrue(m.egb.c.has_ub())

    def test_expr_property_raises(self):
        """Test expr property raises TypeError."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        with self.assertRaises(TypeError) as context:
            _ = m.egb.c.expr
        self.assertIn("do not have an explicit expression", str(context.exception))

    def test_get_value_raises(self):
        """Test get_value() raises TypeError."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        with self.assertRaises(TypeError) as context:
            m.egb.c.get_value()
        self.assertIn("do not have an explicit expression", str(context.exception))

    def test_set_value_raises(self):
        """Test set_value() raises TypeError."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        with self.assertRaises(TypeError) as context:
            m.egb.c.set_value(None)
        self.assertIn("do not have an explicit expression", str(context.exception))

    def test_to_bounded_expression_raises(self):
        """Test to_bounded_expression() raises TypeError."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        with self.assertRaises(TypeError) as context:
            m.egb.c.to_bounded_expression()
        self.assertIn("do not have an explicit expression", str(context.exception))


class TestEGBConstraintBody(unittest.TestCase):
    """Test the EGBConstraintBody object returned by the body property."""

    def test_body_returns_egb_constraint_body_object(self):
        """Test that body property returns an EGBConstraintBody object."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        body_obj = m.egb.c.body
        self.assertIsInstance(body_obj, EGBConstraintBody)

    def test_body_object_is_numeric_type(self):
        """Test that EGBConstraintBody object has is_numeric_type property set to True."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        body_obj = m.egb.c.body
        self.assertTrue(body_obj.is_numeric_type)

    def test_body_object_can_be_called(self):
        """Test that EGBConstraintBody object can be called directly to get residual."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        external_model.set_input_values(np.asarray([100, 2, 3, 50], dtype=np.float64))

        body_obj = m.egb.c.body
        body_value = body_obj()
        self.assertAlmostEqual(body_value, 22.0, places=6)

    def test_body_object_with_pyo_value(self):
        """Test that pyo.value() works with EGBConstraintBody object."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        external_model.set_input_values(np.asarray([100, 2, 3, 50], dtype=np.float64))

        body_obj = m.egb.c.body
        body_value = pyo.value(body_obj)
        self.assertAlmostEqual(body_value, 22.0, places=6)

    def test_body_object_caching(self):
        """Test that body property returns the same object on repeated access."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        body_obj1 = m.egb.c.body
        body_obj2 = m.egb.c.body
        self.assertIs(body_obj1, body_obj2)

    def test_body_object_evaluates_with_different_inputs(self):
        """Test that body object evaluates correctly with different external model inputs."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        body_obj = m.egb.c.body

        # First evaluation
        external_model.set_input_values(np.asarray([100, 2, 3, 50], dtype=np.float64))
        body_value1 = pyo.value(body_obj)
        self.assertAlmostEqual(body_value1, 22.0, places=6)

        # Second evaluation with different inputs
        external_model.set_input_values(np.asarray([100, 2, 3, 28], dtype=np.float64))
        body_value2 = pyo.value(body_obj)
        self.assertAlmostEqual(body_value2, 0.0, places=6)

    def test_body_object_for_output_constraint(self):
        """Test EGBConstraintBody object for output-based constraints."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleOutput()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='Pout')

        # Pin=100, c=2, F=3 => Pout_evaluated = 100 - 4*2*3^2 = 28
        external_model.set_input_values(np.asarray([100, 2, 3], dtype=np.float64))

        # Set output variable to match evaluated value
        m.egb.outputs['Pout'].set_value(28.0)

        body_obj = m.egb.c.body
        # For outputs, when variable matches evaluated value, residual should be 0
        self.assertAlmostEqual(pyo.value(body_obj), 0.0, places=6)

    def test_output_constraint_residual_when_variable_matches_evaluated(self):
        """Test that residual is zero when output variable value matches evaluated value."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleOutput()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='Pout')

        # Set inputs: Pin=100, c=2, F=3 => Pout_evaluated = 100 - 4*2*3^2 = 28
        external_model.set_input_values(np.asarray([100, 2, 3], dtype=np.float64))

        # Set the output variable to match the evaluated value
        evaluated_value = external_model.evaluate_outputs()[0]
        m.egb.outputs['Pout'].set_value(evaluated_value)

        # Residual should be: var_value - evaluated_value = 28 - 28 = 0
        residual = pyo.value(m.egb.c.body)
        self.assertAlmostEqual(residual, 0.0, places=6)

    def test_output_constraint_residual_when_variable_does_not_match_evaluated(self):
        """Test that residual is non-zero when output variable value does not match evaluated value."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleOutput()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='Pout')

        # Set inputs: Pin=100, c=2, F=3 => Pout_evaluated = 100 - 4*2*3^2 = 28
        external_model.set_input_values(np.asarray([100, 2, 3], dtype=np.float64))

        # Set the output variable to a different value than evaluated
        m.egb.outputs['Pout'].set_value(50.0)

        # Residual should be: var_value - evaluated_value = 50 - 28 = 22
        residual = pyo.value(m.egb.c.body)
        self.assertAlmostEqual(residual, 22.0, places=6)

        # Test with another non-matching value
        m.egb.outputs['Pout'].set_value(20.0)
        
        # Residual should be: var_value - evaluated_value = 20 - 28 = -8
        residual = pyo.value(m.egb.c.body)
        self.assertAlmostEqual(residual, -8.0, places=6)

    def test_output_constraint_residual_updates_with_inputs_and_variable(self):
        """Test that residual updates correctly when inputs or output variable changes."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleOutput()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='Pout')

        # Initial inputs: Pin=100, c=2, F=3 => Pout_evaluated = 100 - 4*2*9 = 28
        external_model.set_input_values(np.asarray([100, 2, 3], dtype=np.float64))
        m.egb.outputs['Pout'].set_value(30.0)
        
        # Residual should be: 30 - 28 = 2
        residual = pyo.value(m.egb.c.body)
        self.assertAlmostEqual(residual, 2.0, places=6)

        # Change inputs: Pin=100, c=2, F=2 => Pout_evaluated = 100 - 4*2*4 = 68
        external_model.set_input_values(np.asarray([100, 2, 2], dtype=np.float64))
        
        # Residual should be: 30 - 68 = -38
        residual = pyo.value(m.egb.c.body)
        self.assertAlmostEqual(residual, -38.0, places=6)

        # Now set variable to match the new evaluated value
        m.egb.outputs['Pout'].set_value(68.0)
        
        # Residual should be: 68 - 68 = 0
        residual = pyo.value(m.egb.c.body)
        self.assertAlmostEqual(residual, 0.0, places=6)

    def test_body_object_invalid_constraint_id_raises(self):
        """Test that body object raises error for invalid constraint ID."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        # Directly create body object with invalid constraint ID
        with self.assertRaises(ValueError) as context:
            body_obj = EGBConstraintBody(m.egb, 'invalid_constraint_id')
        self.assertIn("invalid_constraint_id", str(context.exception))

    def test_get_incident_variables_without_jacobian(self):
        """Test get_incident_variables returns all input variables when use_jacobian=False."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        body_obj = m.egb.c.body
        incident_vars = body_obj.get_incident_variables(use_jacobian=False)

        # Should return all 4 input variables
        self.assertEqual(len(incident_vars), 4)
        expected_names = ['Pin', 'c', 'F', 'Pout']
        actual_names = [var.name for var in incident_vars]
        self.assertEqual(
            actual_names, [f'egb.inputs[{name}]' for name in expected_names]
        )

    def test_get_incident_variables_with_jacobian_all_nonzero(self):
        """Test get_incident_variables with use_jacobian=True when all Jacobian entries are non-zero."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        # Set inputs to non-zero values so all Jacobian entries are non-zero
        external_model.set_input_values(np.asarray([100, 2, 3, 50], dtype=np.float64))

        body_obj = m.egb.c.body
        incident_vars = body_obj.get_incident_variables(use_jacobian=True)

        # Should return all 4 input variables since all Jacobian entries are non-zero
        self.assertEqual(len(incident_vars), 4)
        expected_names = ['Pin', 'c', 'F', 'Pout']
        actual_names = [var.name for var in incident_vars]
        self.assertEqual(
            actual_names, [f'egb.inputs[{name}]' for name in expected_names]
        )

    def test_get_incident_variables_with_jacobian_some_zero(self):
        """Test get_incident_variables with use_jacobian=True when some Jacobian entries are zero."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoEqualities()
        m.egb.set_external_model(external_model)

        m.egb.c1 = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop2')
        m.egb.c2 = ExternalGreyBoxConstraint(implicit_constraint_id='pdropout')

        # Set F=0 so that derivatives with respect to c and F are zero
        external_model.set_input_values(
            np.asarray([100, 2, 0, 90, 80], dtype=np.float64)
        )

        # For first constraint (pdrop2): P2 - (Pin - 2*c*F^2)
        # Jacobian: [-1, 0, 0, 1, 0] (only Pin and P2 are non-zero)
        body_obj1 = m.egb.c1.body
        incident_vars1 = body_obj1.get_incident_variables(use_jacobian=True)
        self.assertEqual(len(incident_vars1), 2)
        names1 = [var.name for var in incident_vars1]
        self.assertIn('egb.inputs[Pin]', names1)
        self.assertIn('egb.inputs[P2]', names1)

        # For second constraint (pdropout): Pout - (P2 - 2*c*F^2)
        # Jacobian: [0, 0, 0, -1, 1] (only P2 and Pout are non-zero)
        body_obj2 = m.egb.c2.body
        incident_vars2 = body_obj2.get_incident_variables(use_jacobian=True)
        self.assertEqual(len(incident_vars2), 2)
        names2 = [var.name for var in incident_vars2]
        self.assertIn('egb.inputs[P2]', names2)
        self.assertIn('egb.inputs[Pout]', names2)

    def test_get_incident_variables_with_output_constraint(self):
        """Test get_incident_variables for output-based constraints."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoOutputs()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='Pout')

        body_obj = m.egb.c.body

        # Test without Jacobian
        incident_vars = body_obj.get_incident_variables(use_jacobian=False)
        self.assertEqual(len(incident_vars), 4)
        expected_names = [
            'egb.inputs[Pin]',
            'egb.inputs[c]',
            'egb.inputs[F]',
            'egb.outputs[Pout]',
        ]
        assert all(var.name in expected_names for var in incident_vars)

        # Test with Jacobian (all non-zero)
        external_model.set_input_values(np.asarray([100, 2, 3], dtype=np.float64))
        incident_vars = body_obj.get_incident_variables(use_jacobian=True)
        self.assertEqual(len(incident_vars), 4)
        assert all(var.name in expected_names for var in incident_vars)

    def test_get_incident_variables_with_custom_tolerance(self):
        """Test get_incident_variables with custom Jacobian tolerance."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        # Set inputs so that some Jacobian entries are small but non-zero
        # With c=0.001 and F=0.001, the derivative w.r.t. c is 4*F^2 = 4e-6
        external_model.set_input_values(
            np.asarray([100, 0.001, 0.001, 50], dtype=np.float64)
        )

        body_obj = m.egb.c.body

        # With default tolerance (1e-8), should include variables with Jacobian entry 4e-6
        incident_vars_default = body_obj.get_incident_variables(use_jacobian=True)
        self.assertEqual(len(incident_vars_default), 4)

        # With higher tolerance (1e-5), should exclude variable with Jacobian entry 4e-6
        incident_vars_high_tol = body_obj.get_incident_variables(
            use_jacobian=True, jac_tolerance=1e-5
        )
        # c and F derivatives should be filtered out
        self.assertEqual(len(incident_vars_high_tol), 2)
        names = [var.name for var in incident_vars_high_tol]
        self.assertIn('egb.inputs[Pin]', names)
        self.assertIn('egb.inputs[Pout]', names)

    def test_get_incident_variables_multiple_outputs(self):
        """Test get_incident_variables for different output constraints in same model."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoOutputs()
        m.egb.set_external_model(external_model)

        m.egb.c1 = ExternalGreyBoxConstraint(implicit_constraint_id='P2')
        m.egb.c2 = ExternalGreyBoxConstraint(implicit_constraint_id='Pout')

        external_model.set_input_values(np.asarray([100, 2, 3], dtype=np.float64))

        body_obj1 = m.egb.c1.body
        body_obj2 = m.egb.c2.body

        incident_vars1 = body_obj1.get_incident_variables(use_jacobian=True)
        incident_vars2 = body_obj2.get_incident_variables(use_jacobian=True)

        self.assertEqual(len(incident_vars1), 4)
        self.assertEqual(len(incident_vars2), 4)

        # Compare variable names
        for v in incident_vars1:
            expected1 = [
                "egb.inputs[Pin]",
                "egb.inputs[c]",
                "egb.inputs[F]",
                "egb.outputs[P2]",
            ]
            assert v.name in expected1
        for v in incident_vars2:
            expected2 = [
                "egb.inputs[Pin]",
                "egb.inputs[c]",
                "egb.inputs[F]",
                "egb.outputs[Pout]",
            ]
            assert v.name in expected2

    def test_get_incident_variables_multiple_constraints_and_outputs(self):
        """Test get_incident_variables for different implicit constraints in same model."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoEqualitiesTwoOutputs()
        m.egb.set_external_model(external_model, build_implicit_constraint_objects=True)

        # Implicit constraint: 'pdrop1'
        body_obj1 = m.egb.pdrop1.body
        incident_vars1 = body_obj1.get_incident_variables(use_jacobian=False)
        self.assertEqual(len(incident_vars1), 5)
        expected_names = [
            'egb.inputs[Pin]',
            'egb.inputs[c]',
            'egb.inputs[F]',
            'egb.inputs[P1]',
            'egb.inputs[P3]',
        ]
        for v in incident_vars1:
            self.assertIn(v.name, expected_names)

        # Implicit constraint: 'pdrop3'
        body_obj1 = m.egb.pdrop3.body
        incident_vars1 = body_obj1.get_incident_variables(use_jacobian=False)
        self.assertEqual(len(incident_vars1), 5)
        expected_names = [
            'egb.inputs[Pin]',
            'egb.inputs[c]',
            'egb.inputs[F]',
            'egb.inputs[P1]',
            'egb.inputs[P3]',
        ]
        for v in incident_vars1:
            self.assertIn(v.name, expected_names)

        # Implicit constraint: 'P2_constraint'
        body_obj1 = m.egb.P2_constraint.body
        incident_vars1 = body_obj1.get_incident_variables(use_jacobian=False)
        self.assertEqual(len(incident_vars1), 6)
        expected_names = [
            'egb.inputs[Pin]',
            'egb.inputs[c]',
            'egb.inputs[F]',
            'egb.inputs[P1]',
            'egb.inputs[P3]',
            'egb.outputs[P2]',
        ]
        for v in incident_vars1:
            self.assertIn(v.name, expected_names)

        # Implicit constraint: 'Pout_constraint'
        body_obj1 = m.egb.Pout_constraint.body
        incident_vars1 = body_obj1.get_incident_variables(use_jacobian=False)
        self.assertEqual(len(incident_vars1), 6)
        expected_names = [
            'egb.inputs[Pin]',
            'egb.inputs[c]',
            'egb.inputs[F]',
            'egb.inputs[P1]',
            'egb.inputs[P3]',
            'egb.outputs[Pout]',
        ]
        for v in incident_vars1:
            self.assertIn(v.name, expected_names)

    def test_get_incident_variables_default_parameters(self):
        """Test get_incident_variables with default parameters (use_jacobian=False)."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        body_obj = m.egb.c.body
        # Call without parameters (should default to use_jacobian=False)
        incident_vars = body_obj.get_incident_variables()

        # Should return all 4 input variables
        self.assertEqual(len(incident_vars), 4)

    def test_get_incident_variables_returns_var_data_objects(self):
        """Test that get_incident_variables returns actual variable data objects."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        body_obj = m.egb.c.body
        incident_vars = body_obj.get_incident_variables()

        # Verify that each element is a Pyomo VarData object
        for var in incident_vars:
            self.assertTrue(hasattr(var, 'value'))
            self.assertTrue(hasattr(var, 'fixed'))
            self.assertTrue(hasattr(var, 'lb'))
            self.assertTrue(hasattr(var, 'ub'))

    def test_get_incident_variables_with_zero_jacobian_entries(self):
        """Test get_incident_variables when all Jacobian entries are exactly zero."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        # Set c=0 and F=0, making derivatives w.r.t. c and F equal to zero
        external_model.set_input_values(np.asarray([100, 0, 0, 50], dtype=np.float64))

        body_obj = m.egb.c.body
        incident_vars = body_obj.get_incident_variables(use_jacobian=True)

        # Should only include Pin and Pout (which have Jacobian entries -1 and 1)
        self.assertEqual(len(incident_vars), 2)
        names = [var.name for var in incident_vars]
        self.assertIn('egb.inputs[Pin]', names)
        self.assertIn('egb.inputs[Pout]', names)
        self.assertNotIn('egb.inputs[c]', names)
        self.assertNotIn('egb.inputs[F]', names)


class TestExternalGreyBoxConstraintSlack(unittest.TestCase):
    """Test slack methods of ExternalGreyBoxConstraint."""

    def test_lslack(self):
        """Test lslack() returns body value."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        external_model.set_input_values(np.asarray([100, 2, 3, 50], dtype=np.float64))

        lslack_value = m.egb.c.lslack()
        body_value = pyo.value(m.egb.c.body)
        self.assertAlmostEqual(lslack_value, body_value, places=6)

    def test_uslack(self):
        """Test uslack() returns negative body value."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        external_model.set_input_values(np.asarray([100, 2, 3, 50], dtype=np.float64))

        uslack_value = m.egb.c.uslack()
        body_value = pyo.value(m.egb.c.body)
        self.assertAlmostEqual(uslack_value, -body_value, places=6)

    def test_slack(self):
        """Test slack() returns negative absolute value of body."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        external_model.set_input_values(np.asarray([100, 2, 3, 50], dtype=np.float64))

        slack_value = m.egb.c.slack()
        body_value = pyo.value(m.egb.c.body)
        self.assertAlmostEqual(slack_value, -abs(body_value), places=6)

    def test_call_method(self):
        """Test __call__() method returns body value."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        external_model.set_input_values(np.asarray([100, 2, 3, 50], dtype=np.float64))

        call_value = m.egb.c()
        body_value = pyo.value(m.egb.c.body)
        self.assertAlmostEqual(call_value, body_value, places=6)


class TestExternalGreyBoxConstraintActive(unittest.TestCase):
    """Test active status methods of ExternalGreyBoxConstraint."""

    def test_active_property(self):
        """Test active property follows parent block."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        self.assertTrue(m.egb.c.active)

        m.egb.deactivate()
        self.assertFalse(m.egb.c.active)

        m.egb.activate()
        self.assertTrue(m.egb.c.active)

    def test_activate_raises(self):
        """Test activate() raises TypeError."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        with self.assertRaises(TypeError) as context:
            m.egb.c.activate()
        self.assertIn("cannot be activated or deactivated", str(context.exception))

    def test_deactivate_raises(self):
        """Test deactivate() raises TypeError."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        with self.assertRaises(TypeError) as context:
            m.egb.c.deactivate()
        self.assertIn("cannot be activated or deactivated", str(context.exception))


class TestScalarExternalGreyBoxConstraint(unittest.TestCase):
    """Test ScalarExternalGreyBoxConstraint specific functionality."""

    def test_scalar_body_before_assignment_raises(self):
        """Test accessing body before assignment raises ValueError."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')
        m.egb.c.clear()  # Clear the data

        with self.assertRaises(ValueError) as context:
            _ = m.egb.c.body
        self.assertIn(
            "before the ExternalGreyBoxConstraint has been assigned",
            str(context.exception),
        )

    def test_scalar_lower_before_assignment_raises(self):
        """Test accessing lower before assignment raises ValueError."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')
        m.egb.c.clear()

        with self.assertRaises(ValueError) as context:
            _ = m.egb.c.lower
        self.assertIn(
            "before the ExternalGreyBoxConstraint has been assigned",
            str(context.exception),
        )

    def test_scalar_upper_before_assignment_raises(self):
        """Test accessing upper before assignment raises ValueError."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')
        m.egb.c.clear()

        with self.assertRaises(ValueError) as context:
            _ = m.egb.c.upper
        self.assertIn(
            "before the ExternalGreyBoxConstraint has been assigned",
            str(context.exception),
        )

    def test_scalar_equality_before_assignment_raises(self):
        """Test accessing equality before assignment raises ValueError."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')
        m.egb.c.clear()

        with self.assertRaises(ValueError) as context:
            _ = m.egb.c.equality
        self.assertIn(
            "before the ExternalGreyBoxConstraint has been assigned",
            str(context.exception),
        )

    def test_scalar_strict_lower_before_assignment_raises(self):
        """Test accessing strict_lower before assignment raises ValueError."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')
        m.egb.c.clear()

        with self.assertRaises(ValueError) as context:
            _ = m.egb.c.strict_lower
        self.assertIn(
            "before the ExternalGreyBoxConstraint has been assigned",
            str(context.exception),
        )

    def test_scalar_strict_upper_before_assignment_raises(self):
        """Test accessing strict_upper before assignment raises ValueError."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')
        m.egb.c.clear()

        with self.assertRaises(ValueError) as context:
            _ = m.egb.c.strict_upper
        self.assertIn(
            "before the ExternalGreyBoxConstraint has been assigned",
            str(context.exception),
        )

    def test_scalar_add_with_invalid_index_raises(self):
        """Test add() with non-None index raises ValueError."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        with self.assertRaises(ValueError) as context:
            m.egb.c.add(1, None)
        self.assertIn(
            "does not accept index values other than None", str(context.exception)
        )


class TestExternalGreyBoxConstraintMultipleConstraints(unittest.TestCase):
    """Test with models having multiple equality constraints."""

    def test_two_equality_constraints(self):
        """Test with model having two equality constraints."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoEqualities()
        m.egb.set_external_model(external_model)

        m.egb.c1 = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop2')
        m.egb.c2 = ExternalGreyBoxConstraint(implicit_constraint_id='pdropout')

        # Set input values directly on external model: Pin=100, c=2, F=3, P2=82, Pout=64
        external_model.set_input_values(
            np.asarray([100, 2, 3, 82, 64], dtype=np.float64)
        )

        # Expected residual for pdrop2: P2 - (Pin - 2*c*F^2) = 82 - (100 - 2*2*9) = 82 - 64 = 18
        body1 = pyo.value(m.egb.c1.body)
        self.assertAlmostEqual(body1, 18.0, places=6)

        # Expected residual for pdropout: Pout - (P2 - 2*c*F^2) = 64 - (82 - 2*2*9) = 64 - 46 = 18
        body2 = pyo.value(m.egb.c2.body)
        self.assertAlmostEqual(body2, 18.0, places=6)

    def test_two_outputs(self):
        """Test with model having two outputs."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoOutputs()
        m.egb.set_external_model(external_model)

        m.egb.c1 = ExternalGreyBoxConstraint(implicit_constraint_id='P2')
        m.egb.c2 = ExternalGreyBoxConstraint(implicit_constraint_id='Pout')

        # Set input values directly on external model
        # Pin=100, c=2, F=3
        # P2_evaluated = 100 - 2*2*9 = 64
        # Pout_evaluated = 100 - 4*2*9 = 28
        external_model.set_input_values(np.asarray([100, 2, 3], dtype=np.float64))

        # Set output variables to match evaluated values
        m.egb.outputs['P2'].set_value(64.0)
        m.egb.outputs['Pout'].set_value(28.0)

        # For outputs, when variables match evaluated values, residuals should be 0.0
        self.assertAlmostEqual(pyo.value(m.egb.c1.body), 0.0, places=6)
        self.assertAlmostEqual(pyo.value(m.egb.c2.body), 0.0, places=6)


class TestExternalGreyBoxConstraintDisplay(unittest.TestCase):
    """Test display and printing methods."""

    def test_display_method(self):
        """Test display() method executes without error."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        external_model.set_input_values(np.asarray([100, 2, 3, 50], dtype=np.float64))

        # Should not raise an exception
        import io

        output = io.StringIO()
        m.egb.c.display(ostream=output)
        result = output.getvalue()

        # Check that output contains expected elements
        self.assertIn('Lower', result)
        self.assertIn('Body', result)
        self.assertIn('Upper', result)

    def test_display_inactive_does_nothing(self):
        """Test display() on inactive component produces no output."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')
        m.egb.deactivate()

        import io

        output = io.StringIO()
        m.egb.c.display(ostream=output)
        result = output.getvalue()

        # Should produce empty or minimal output
        self.assertEqual(result, '')

    def test_pprint_method(self):
        """Test _pprint() returns expected data structure."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        external_model.set_input_values(np.asarray([100, 2, 3, 50], dtype=np.float64))

        data = m.egb.c._pprint()

        # Check structure
        self.assertEqual(len(data), 4)
        headers, items_method, columns, formatter = data

        # Check headers
        self.assertIn(("Size", 1), headers)
        self.assertIn(("Active", True), headers)

        # Check columns
        self.assertEqual(columns, ("Lower", "Body", "Upper", "Active"))


class TestExternalGreyBoxConstraintImplicitConstraintId(unittest.TestCase):
    """Test implicit_constraint_id property."""

    def test_implicit_constraint_id_property(self):
        """Test implicit_constraint_id property returns correct value."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        constraint_id = 'pdrop'
        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id=constraint_id)

        self.assertEqual(m.egb.c.implicit_constraint_id, constraint_id)

    def test_implicit_constraint_id_stored_correctly(self):
        """Test implicit_constraint_id is stored in _implicit_constraint_id."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoEqualities()
        m.egb.set_external_model(external_model)

        m.egb.c1 = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop2')
        m.egb.c2 = ExternalGreyBoxConstraint(implicit_constraint_id='pdropout')

        self.assertEqual(m.egb.c1._implicit_constraint_id, 'pdrop2')
        self.assertEqual(m.egb.c2._implicit_constraint_id, 'pdropout')


class TestExternalGreyBoxConstraintIntegration(unittest.TestCase):
    """Integration tests with various external models."""

    def test_with_hessian_model_equality(self):
        """Test with model that supports Hessian evaluation."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEqualityWithHessian()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        external_model.set_input_values(np.asarray([100, 2, 3, 28], dtype=np.float64))

        # At correct solution, residual should be 0
        body_value = pyo.value(m.egb.c.body)
        self.assertAlmostEqual(body_value, 0.0, places=6)

    def test_with_hessian_model_output(self):
        """Test with output model that supports Hessian evaluation."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleOutputWithHessian()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='Pout')

        # Pin=100, c=2, F=3 => Pout_evaluated = 100 - 4*2*9 = 28
        external_model.set_input_values(np.asarray([100, 2, 3], dtype=np.float64))

        # Set output variable to match evaluated value
        m.egb.outputs['Pout'].set_value(28.0)

        # For outputs, when variable matches evaluated value, residual should be 0
        self.assertAlmostEqual(pyo.value(m.egb.c.body), 0.0, places=6)

    def test_constraint_in_different_blocks(self):
        """Test constraints in multiple ExternalGreyBoxBlocks."""
        m = pyo.ConcreteModel()

        m.egb1 = ExternalGreyBoxBlock()
        external_model1 = ex_models.PressureDropSingleEquality()
        m.egb1.set_external_model(external_model1)
        m.egb1.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        m.egb2 = ExternalGreyBoxBlock()
        external_model2 = ex_models.PressureDropSingleOutput()
        m.egb2.set_external_model(external_model2)
        m.egb2.c = ExternalGreyBoxConstraint(implicit_constraint_id='Pout')

        # Set inputs for first block
        external_model1.set_input_values(np.asarray([100, 2, 3, 28], dtype=np.float64))

        # Set inputs for second block
        # Pin=50, c=1, F=2 => Pout_evaluated = 50 - 4*1*4 = 34
        external_model2.set_input_values(np.asarray([50, 1, 2], dtype=np.float64))

        # Set output variable for second block to match evaluated value
        m.egb2.outputs['Pout'].set_value(34.0)

        # Check both constraints work independently
        self.assertAlmostEqual(pyo.value(m.egb1.c.body), 0.0, places=6)
        self.assertAlmostEqual(pyo.value(m.egb2.c.body), 0.0, places=6)


class TestExternalGreyBoxConstraintEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_constraint_with_zero_inputs(self):
        """Test constraint evaluation with zero input values."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        # Set all inputs to zero directly on external model
        external_model.set_input_values(np.asarray([0, 0, 0, 0], dtype=np.float64))

        # Residual should be: 0 - (0 - 4*0*0) = 0
        body_value = pyo.value(m.egb.c.body)
        self.assertAlmostEqual(body_value, 0.0, places=6)

    def test_constraint_with_negative_inputs(self):
        """Test constraint evaluation with negative input values."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        # Set negative inputs directly on external model
        external_model.set_input_values(
            np.asarray([-100, -2, -3, -50], dtype=np.float64)
        )

        # Should evaluate without error
        body_value = pyo.value(m.egb.c.body)
        # Expected: -50 - (-100 - 4*(-2)*(-3)^2) = -50 - (-100 - 4*(-2)*9) = -50 - (-100 + 72) = -50 + 28 = -22
        self.assertAlmostEqual(body_value, -22.0, places=6)

    def test_constraint_with_large_inputs(self):
        """Test constraint evaluation with large input values."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        # Set large inputs directly on external model
        external_model.set_input_values(
            np.asarray([1e6, 1e3, 1e2, 1e5], dtype=np.float64)
        )

        # Should evaluate without error
        body_value = pyo.value(m.egb.c.body)
        self.assertIsInstance(body_value, (float, np.floating))


def test_component_data_objects_with_EGBC():
    """Test that ExternalGreyBoxConstraints can be iterated over using component_data_objects."""
    m = pyo.ConcreteModel()
    m.egb = ExternalGreyBoxBlock()
    external_model = ex_models.PressureDropTwoEqualitiesTwoOutputsWithHessian()
    m.egb.set_external_model(external_model, build_implicit_constraint_objects=True)

    count = 0
    for c in m.egb.component_data_objects(
        ctype=ExternalGreyBoxConstraint, descend_into=False
    ):
        assert isinstance(c, ScalarExternalGreyBoxConstraint)
        assert c.local_name in ['P2_constraint', 'Pout_constraint', 'pdrop1', 'pdrop3']
        count += 1
    assert count == 4


if __name__ == '__main__':
    unittest.main()
