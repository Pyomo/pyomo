# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
#
#  Additional contributions Copyright (c) 2026 OLI Systems, Inc.
#  ___________________________________________________________________________________

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from unittest.mock import patch

from pyomo.contrib.pynumero.dependencies import numpy as np

from pyomo.contrib.pynumero.interfaces.external_grey_box import (
    ExternalGreyBoxBlock,
    ExternalGreyBoxBlockData,
)
from pyomo.contrib.pynumero.interfaces.external_grey_box_constraint import (
    ExternalGreyBoxConstraint,
    ExternalGreyBoxConstraintData,
    ScalarExternalGreyBoxConstraint,
    EGBConstraintBody,
)
import pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models as ex_models


def _no_op_construct_implicit_constraints(self):
    """
    No-op implementation of _construct_implicit_constraints.

    This prevents automatic construction of ExternalGreyBoxConstraints,
    allowing tests to manually construct them to test the constraint
    construction logic itself.
    """
    pass


# Decorator to patch _construct_implicit_constraints with a no-op for test classes
def skip_implicit_constraint_construction(test_class):
    """
    Decorator that patches _construct_implicit_constraints to prevent
    automatic construction of implicit constraints in tests.

    This is scoped to the decorated test class only and won't affect other tests.
    """
    return patch.object(
        ExternalGreyBoxBlockData,
        '_construct_implicit_constraints',
        _no_op_construct_implicit_constraints,
    )(test_class)


@skip_implicit_constraint_construction
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
        """Test that construction outside ExternalGreyBoxBlock raises an error."""
        m = pyo.ConcreteModel()

        # Construction happens automatically when added to block
        # This should raise either ValueError or AttributeError depending on validation order
        with self.assertRaises((ValueError, AttributeError)) as context:
            m.c = ExternalGreyBoxConstraint(implicit_constraint_id='test')
        # Check that error message indicates the problem is related to ExternalGreyBoxBlock
        self.assertTrue(
            "ExternalGreyBoxBlock" in str(context.exception)
            or "get_external_model" in str(context.exception)
        )

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


@skip_implicit_constraint_construction
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
        """Test get_incident_variables returns all input variables."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropSingleEquality()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop')

        body_obj = m.egb.c.body
        incident_vars = body_obj.get_incident_variables()

        # Should return all 4 input variables
        self.assertEqual(len(incident_vars), 4)
        expected_names = ['Pin', 'c', 'F', 'Pout']
        actual_names = [var.name for var in incident_vars]
        self.assertEqual(
            actual_names, [f'egb.inputs[{name}]' for name in expected_names]
        )

    def test_get_incident_variables_with_output_constraint(self):
        """Test get_incident_variables for output-based constraints."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoOutputs()
        m.egb.set_external_model(external_model)

        m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_id='Pout')

        body_obj = m.egb.c.body

        incident_vars = body_obj.get_incident_variables()
        self.assertEqual(len(incident_vars), 4)
        expected_names = [
            'egb.inputs[Pin]',
            'egb.inputs[c]',
            'egb.inputs[F]',
            'egb.outputs[Pout]',
        ]
        assert all(var.name in expected_names for var in incident_vars)

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

        incident_vars1 = body_obj1.get_incident_variables()
        incident_vars2 = body_obj2.get_incident_variables()

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
        m.egb.set_external_model(external_model)

        # Manually create the implicit constraint objects
        m.egb.pdrop1 = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop1')
        m.egb.pdrop3 = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop3')
        m.egb.P2_constraint = ExternalGreyBoxConstraint(implicit_constraint_id='P2')
        m.egb.Pout_constraint = ExternalGreyBoxConstraint(implicit_constraint_id='Pout')

        # Implicit constraint: 'pdrop1'
        body_obj1 = m.egb.pdrop1.body
        incident_vars1 = body_obj1.get_incident_variables()
        self.assertEqual(len(incident_vars1), 4)
        expected_names = [
            'egb.inputs[Pin]',
            'egb.inputs[c]',
            'egb.inputs[F]',
            'egb.inputs[P1]',
        ]
        for v in incident_vars1:
            self.assertIn(v.name, expected_names)

        # Implicit constraint: 'pdrop3'
        body_obj1 = m.egb.pdrop3.body
        incident_vars1 = body_obj1.get_incident_variables()
        self.assertEqual(len(incident_vars1), 4)
        expected_names = [
            'egb.inputs[c]',
            'egb.inputs[F]',
            'egb.inputs[P1]',
            'egb.inputs[P3]',
        ]
        for v in incident_vars1:
            self.assertIn(v.name, expected_names)

        # Implicit constraint: 'P2_constraint'
        body_obj1 = m.egb.P2_constraint.body
        incident_vars1 = body_obj1.get_incident_variables()
        self.assertEqual(len(incident_vars1), 4)
        expected_names = [
            'egb.inputs[c]',
            'egb.inputs[F]',
            'egb.inputs[P1]',
            'egb.outputs[P2]',
        ]
        for v in incident_vars1:
            self.assertIn(v.name, expected_names)

        # Implicit constraint: 'Pout_constraint'
        body_obj1 = m.egb.Pout_constraint.body
        incident_vars1 = body_obj1.get_incident_variables()
        self.assertEqual(len(incident_vars1), 4)
        expected_names = [
            'egb.inputs[c]',
            'egb.inputs[F]',
            'egb.inputs[Pin]',
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


@skip_implicit_constraint_construction
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


@skip_implicit_constraint_construction
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


@skip_implicit_constraint_construction
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


@skip_implicit_constraint_construction
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


@skip_implicit_constraint_construction
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


@skip_implicit_constraint_construction
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


@skip_implicit_constraint_construction
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


@skip_implicit_constraint_construction
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

        # Expected residual: 0 - (0 - 4*0*0) = 0
        body_value = pyo.value(m.egb.c.body)
        self.assertAlmostEqual(body_value, 0.0, places=6)


@skip_implicit_constraint_construction
class TestIndexedExternalGreyBoxConstraint(unittest.TestCase):
    """Test indexed ExternalGreyBoxConstraint functionality."""

    def test_indexed_with_explicit_mapping(self):
        """Test indexed constraint with explicit implicit_constraint_id mapping."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoEqualitiesTwoOutputs()
        m.egb.set_external_model(external_model)

        m.set = pyo.Set(initialize=['P2', 'Pout', 'pdrop1', 'pdrop3'])

        # Create indexed constraint with explicit mapping
        m.egb.c = ExternalGreyBoxConstraint(
            m.set, implicit_constraint_id={i: i for i in m.set}
        )

        # Verify construction
        self.assertTrue(m.egb.c.is_indexed())
        self.assertEqual(len(m.egb.c), 4)

        # Verify each index has correct implicit_constraint_id
        for idx in m.set:
            self.assertEqual(m.egb.c[idx]._implicit_constraint_id, idx)

    def test_indexed_with_implicit_ids(self):
        """Test indexed constraint with inferred implicit_constraint_id from index."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoEqualitiesTwoOutputs()
        m.egb.set_external_model(external_model)

        m.set = pyo.Set(initialize=['P2', 'Pout', 'pdrop1', 'pdrop3'])

        # Create indexed constraint without explicit mapping (ids inferred from index)
        m.egb.c = ExternalGreyBoxConstraint(m.set)

        # Verify construction
        self.assertTrue(m.egb.c.is_indexed())
        self.assertEqual(len(m.egb.c), 4)

        # Verify each index has implicit_constraint_id equal to the index
        for idx in m.set:
            self.assertEqual(m.egb.c[idx]._implicit_constraint_id, idx)

    def test_indexed_body_evaluation(self):
        """Test body evaluation for indexed constraints."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoOutputs()
        m.egb.set_external_model(external_model)

        m.set = pyo.Set(initialize=['P2', 'Pout'])

        m.egb.c = ExternalGreyBoxConstraint(m.set)

        # Set inputs: Pin=100, c=2, F=3
        # P2_evaluated = 100 - 2*2*9 = 64
        # Pout_evaluated = 100 - 4*2*9 = 28
        external_model.set_input_values(np.asarray([100, 2, 3], dtype=np.float64))

        # Set output variables
        m.egb.outputs['P2'].set_value(70.0)
        m.egb.outputs['Pout'].set_value(30.0)

        # Check residuals
        # P2: 70 - 64 = 6
        self.assertAlmostEqual(pyo.value(m.egb.c['P2'].body), 6.0, places=6)
        # Pout: 30 - 28 = 2
        self.assertAlmostEqual(pyo.value(m.egb.c['Pout'].body), 2.0, places=6)

    def test_indexed_iteration(self):
        """Test iterating over indexed constraints."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoEqualitiesTwoOutputs()
        m.egb.set_external_model(external_model)

        m.set = pyo.Set(initialize=['P2', 'Pout', 'pdrop1', 'pdrop3'])

        m.egb.c = ExternalGreyBoxConstraint(m.set)

        # Test iteration
        count = 0
        indices = []
        for idx in m.egb.c:
            count += 1
            indices.append(idx)

        self.assertEqual(count, 4)
        self.assertEqual(set(indices), set(m.set))

    def test_indexed_items_method(self):
        """Test items() method for indexed constraints."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoEqualitiesTwoOutputs()
        m.egb.set_external_model(external_model)

        m.set = pyo.Set(initialize=['P2', 'Pout'])

        m.egb.c = ExternalGreyBoxConstraint(m.set)

        # Test items() method
        items = list(m.egb.c.items())
        self.assertEqual(len(items), 2)

        for idx, constraint_data in items:
            self.assertIn(idx, m.set)
            self.assertEqual(constraint_data._implicit_constraint_id, idx)

    def test_indexed_properties(self):
        """Test properties work correctly for indexed constraints."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoOutputs()
        m.egb.set_external_model(external_model)

        m.set = pyo.Set(initialize=['P2', 'Pout'])

        m.egb.c = ExternalGreyBoxConstraint(m.set)

        for idx in m.set:
            # Test bounds
            self.assertEqual(m.egb.c[idx].lower, 0.0)
            self.assertEqual(m.egb.c[idx].upper, 0.0)
            self.assertEqual(m.egb.c[idx].lb, 0.0)
            self.assertEqual(m.egb.c[idx].ub, 0.0)

            # Test equality
            self.assertTrue(m.egb.c[idx].equality)

            # Test strict bounds
            self.assertFalse(m.egb.c[idx].strict_lower)
            self.assertFalse(m.egb.c[idx].strict_upper)

            # Test has_lb/has_ub
            self.assertTrue(m.egb.c[idx].has_lb())
            self.assertTrue(m.egb.c[idx].has_ub())

    def test_indexed_with_tuple_index(self):
        """Test indexed constraint with tuple indices."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoEqualitiesTwoOutputs()
        m.egb.set_external_model(external_model)

        m.set = pyo.Set(
            initialize=[(1, 'P2'), (1, 'Pout'), (2, 'pdrop1'), (2, 'pdrop3')]
        )

        # Create mapping from tuple indices to string constraint ids
        id_map = {
            (1, 'P2'): 'P2',
            (1, 'Pout'): 'Pout',
            (2, 'pdrop1'): 'pdrop1',
            (2, 'pdrop3'): 'pdrop3',
        }

        m.egb.c = ExternalGreyBoxConstraint(m.set, implicit_constraint_id=id_map)

        # Verify construction
        self.assertEqual(len(m.egb.c), 4)

        for idx in m.set:
            self.assertEqual(m.egb.c[idx]._implicit_constraint_id, id_map[idx])


@skip_implicit_constraint_construction
class TestIndexedExternalGreyBoxConstraintValidation(unittest.TestCase):
    """Test validation errors for indexed ExternalGreyBoxConstraint."""

    def test_indexed_missing_keys_raises(self):
        """Test that missing keys in implicit_constraint_id mapping raises ValueError."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoEqualitiesTwoOutputs()
        m.egb.set_external_model(external_model)

        m.set = pyo.Set(initialize=['P2', 'Pout', 'pdrop1', 'pdrop3'])

        # Create mapping missing some keys
        incomplete_map = {'P2': 'P2', 'Pout': 'Pout'}  # Missing 'pdrop1' and 'pdrop3'

        with self.assertRaises(ValueError) as context:
            m.egb.c = ExternalGreyBoxConstraint(
                m.set, implicit_constraint_id=incomplete_map
            )

        self.assertIn("Missing keys", str(context.exception))
        self.assertIn("pdrop1", str(context.exception))
        self.assertIn("pdrop3", str(context.exception))

    def test_indexed_extra_keys_raises(self):
        """Test that extra keys in implicit_constraint_id mapping raises ValueError."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoEqualitiesTwoOutputs()
        m.egb.set_external_model(external_model)

        m.set = pyo.Set(initialize=['P2', 'Pout'])

        # Create mapping with extra keys
        mapping_with_extras = {
            'P2': 'P2',
            'Pout': 'Pout',
            'extra1': 'pdrop1',
            'extra2': 'pdrop3',
        }

        with self.assertRaises(ValueError) as context:
            m.egb.c = ExternalGreyBoxConstraint(
                m.set, implicit_constraint_id=mapping_with_extras
            )

        self.assertIn("Invalid keys", str(context.exception))
        self.assertIn("extra1", str(context.exception))
        self.assertIn("extra2", str(context.exception))

    def test_indexed_missing_and_extra_keys_raises(self):
        """Test error message when both missing and extra keys exist."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoEqualitiesTwoOutputs()
        m.egb.set_external_model(external_model)

        m.set = pyo.Set(initialize=['P2', 'Pout', 'pdrop1'])

        # Create mapping with missing and extra keys
        bad_map = {
            'P2': 'P2',
            'extra': 'Pout',
        }  # Missing 'Pout' and 'pdrop1', extra 'extra'

        with self.assertRaises(ValueError) as context:
            m.egb.c = ExternalGreyBoxConstraint(m.set, implicit_constraint_id=bad_map)

        error_msg = str(context.exception)
        self.assertIn("Missing keys", error_msg)
        self.assertIn("Invalid keys", error_msg)

    def test_indexed_invalid_type_raises(self):
        """Test that invalid type for implicit_constraint_id raises TypeError."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoEqualitiesTwoOutputs()
        m.egb.set_external_model(external_model)

        m.set = pyo.Set(initialize=['P2', 'Pout'])

        # Pass a string instead of mapping (invalid for indexed)
        with self.assertRaises(TypeError) as context:
            m.egb.c = ExternalGreyBoxConstraint(m.set, implicit_constraint_id='P2')

        self.assertIn("must be a mapping", str(context.exception))

    def test_indexed_invalid_constraint_id_value_raises(self):
        """Test that invalid constraint id value in mapping raises ValueError."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoOutputs()
        m.egb.set_external_model(external_model)

        m.set = pyo.Set(initialize=['P2', 'Pout'])

        # Create mapping with invalid constraint id
        bad_map = {
            'P2': 'P2',
            'Pout': 'invalid_constraint',  # This constraint doesn't exist
        }

        with self.assertRaises(ValueError) as context:
            m.egb.c = ExternalGreyBoxConstraint(m.set, implicit_constraint_id=bad_map)

        self.assertIn("invalid_constraint", str(context.exception))
        self.assertIn("does not exist", str(context.exception))

    def test_indexed_non_string_constraint_id_raises(self):
        """Test that non-string implicit_constraint_id value raises TypeError."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoOutputs()
        m.egb.set_external_model(external_model)

        m.set = pyo.Set(initialize=['P2', 'Pout'])

        # Create mapping with non-string value
        bad_map = {'P2': 123, 'Pout': 'Pout'}  # Not a string

        with self.assertRaises(TypeError) as context:
            m.egb.c = ExternalGreyBoxConstraint(m.set, implicit_constraint_id=bad_map)

        self.assertIn("must be strings", str(context.exception))

    def test_indexed_with_inferred_id_invalid_raises(self):
        """Test that inferred implicit_constraint_id that doesn't exist raises ValueError."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoOutputs()
        m.egb.set_external_model(external_model)

        # Use indices that don't match any constraint/output names
        m.set = pyo.Set(initialize=['invalid1', 'invalid2'])

        with self.assertRaises(ValueError) as context:
            m.egb.c = ExternalGreyBoxConstraint(m.set)

        self.assertIn("does not exist", str(context.exception))


@skip_implicit_constraint_construction
class TestIndexedExternalGreyBoxConstraintAdvanced(unittest.TestCase):
    """Advanced tests for indexed ExternalGreyBoxConstraint."""

    def test_indexed_add_method(self):
        """Test add() method for indexed constraints."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoOutputs()
        m.egb.set_external_model(external_model)

        m.set = pyo.Set(initialize=['P2', 'Pout'])

        m.egb.c = ExternalGreyBoxConstraint(m.set)

        # The add method should work the same as __setitem__
        # Test that we can access via indexing
        self.assertIsNotNone(m.egb.c['P2'])
        self.assertIsNotNone(m.egb.c['Pout'])

    def test_indexed_getitem(self):
        """Test __getitem__ for indexed constraints."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoOutputs()
        m.egb.set_external_model(external_model)

        m.set = pyo.Set(initialize=['P2', 'Pout'])

        m.egb.c = ExternalGreyBoxConstraint(m.set)

        # Test getitem
        c_p2 = m.egb.c['P2']
        c_pout = m.egb.c['Pout']

        self.assertEqual(c_p2._implicit_constraint_id, 'P2')
        self.assertEqual(c_pout._implicit_constraint_id, 'Pout')

    def test_indexed_with_different_constraint_types(self):
        """Test indexed constraints mixing equality constraints and outputs."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoEqualitiesTwoOutputs()
        m.egb.set_external_model(external_model)

        # Mix of outputs and equality constraints
        m.set = pyo.Set(initialize=['P2', 'pdrop1'])

        id_map = {'P2': 'P2', 'pdrop1': 'pdrop1'}  # output  # equality constraint

        m.egb.c = ExternalGreyBoxConstraint(m.set, implicit_constraint_id=id_map)

        # Set inputs
        # Using  6 inputs: Pin, c, F, P1, P3, and one for the missing input of the equality constraint
        # Inputs: Pin=100, c=2, F=3, P1=82, P3=46, plus empty output placeholders
        external_model.set_input_values(
            np.asarray([100, 2, 3, 82, 46], dtype=np.float64)
        )

        # For output P2, set value
        m.egb.outputs['P2'].set_value(64.0)

        # Both constraints should be accessible and evaluatable
        self.assertIsNotNone(m.egb.c['P2'].body)
        self.assertIsNotNone(m.egb.c['pdrop1'].body)

    def test_indexed_component_property(self):
        """Test that indexed constraint data has correct component reference."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoOutputs()
        m.egb.set_external_model(external_model)

        m.set = pyo.Set(initialize=['P2', 'Pout'])

        m.egb.c = ExternalGreyBoxConstraint(m.set)

        for idx in m.set:
            constraint_data = m.egb.c[idx]
            # Verify parent component reference
            self.assertIs(constraint_data.parent_component(), m.egb.c)

    def test_indexed_active_status(self):
        """Test active status for indexed constraints."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoOutputs()
        m.egb.set_external_model(external_model)

        m.set = pyo.Set(initialize=['P2', 'Pout'])

        m.egb.c = ExternalGreyBoxConstraint(m.set)

        # All should be active
        for idx in m.set:
            self.assertTrue(m.egb.c[idx].active)

        # Deactivate parent block
        m.egb.deactivate()

        # All should now be inactive
        for idx in m.set:
            self.assertFalse(m.egb.c[idx].active)

    def test_indexed_slack_methods(self):
        """Test slack methods for indexed constraints."""
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        external_model = ex_models.PressureDropTwoOutputs()
        m.egb.set_external_model(external_model)

        m.set = pyo.Set(initialize=['P2', 'Pout'])

        m.egb.c = ExternalGreyBoxConstraint(m.set)

        # Set inputs
        external_model.set_input_values(np.asarray([100, 2, 3], dtype=np.float64))
        m.egb.outputs['P2'].set_value(70.0)
        m.egb.outputs['Pout'].set_value(30.0)

        for idx in m.set:
            body_val = pyo.value(m.egb.c[idx].body)

            # Test lslack
            self.assertAlmostEqual(m.egb.c[idx].lslack(), body_val, places=6)

            # Test uslack
            self.assertAlmostEqual(m.egb.c[idx].uslack(), -body_val, places=6)

            # Test slack
            self.assertAlmostEqual(m.egb.c[idx].slack(), -abs(body_val), places=6)

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


@skip_implicit_constraint_construction
def test_component_data_objects_with_EGBC():
    """Test that ExternalGreyBoxConstraints can be iterated over using component_data_objects."""
    m = pyo.ConcreteModel()
    m.egb = ExternalGreyBoxBlock()
    external_model = ex_models.PressureDropTwoEqualitiesTwoOutputsWithHessian()
    m.egb.set_external_model(external_model)

    # Manually create the implicit constraint objects
    m.egb.pdrop1 = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop1')
    m.egb.pdrop3 = ExternalGreyBoxConstraint(implicit_constraint_id='pdrop3')
    m.egb.P2_constraint = ExternalGreyBoxConstraint(implicit_constraint_id='P2')
    m.egb.Pout_constraint = ExternalGreyBoxConstraint(implicit_constraint_id='Pout')

    count = 0
    for c in m.egb.component_data_objects(
        ctype=ExternalGreyBoxConstraint, descend_into=False
    ):
        assert isinstance(c, ExternalGreyBoxConstraintData)
        assert c.local_name in ['P2_constraint', 'Pout_constraint', 'pdrop1', 'pdrop3']
        count += 1
    assert count == 4


@skip_implicit_constraint_construction
def test_indexed_egbc_no_implicit_constraint_id():
    m = pyo.ConcreteModel()
    m.egb = ExternalGreyBoxBlock()
    external_model = ex_models.PressureDropTwoEqualitiesTwoOutputs()
    m.egb.set_external_model(external_model)

    m.set = pyo.Set(initialize=['P2', 'Pout', 'pdrop1', 'pdrop3'])

    m.egb.c = ExternalGreyBoxConstraint(m.set)

    assert m.egb.c["P2"]._implicit_constraint_id == "P2"
    assert m.egb.c["Pout"]._implicit_constraint_id == "Pout"
    assert m.egb.c["pdrop1"]._implicit_constraint_id == "pdrop1"
    assert m.egb.c["pdrop3"]._implicit_constraint_id == "pdrop3"


@skip_implicit_constraint_construction
def test_indexed_egbc_implicit_constraint_id_mapping():
    m = pyo.ConcreteModel()
    m.egb = ExternalGreyBoxBlock()
    external_model = ex_models.PressureDropTwoEqualitiesTwoOutputs()
    m.egb.set_external_model(external_model)

    m.set = pyo.Set(initialize=['P2', 'Pout', 'pdrop1', 'pdrop3'])

    m.egb.c = ExternalGreyBoxConstraint(
        m.set, implicit_constraint_id={i: i for i in m.set}
    )

    assert m.egb.c["P2"]._implicit_constraint_id == "P2"
    assert m.egb.c["Pout"]._implicit_constraint_id == "Pout"
    assert m.egb.c["pdrop1"]._implicit_constraint_id == "pdrop1"
    assert m.egb.c["pdrop3"]._implicit_constraint_id == "pdrop3"


if __name__ == '__main__':
    unittest.main()
