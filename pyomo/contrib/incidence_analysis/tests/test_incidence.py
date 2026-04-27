# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.incidence_analysis.incidence import (
    IncidenceMethod,
    get_incident_variables,
    get_variables_incident_to_constraint,
)
from pyomo.contrib.incidence_analysis.config import get_config_from_kwds
from pyomo.repn.ampl import AMPLRepnVisitor
from unittest.mock import MagicMock


class TestAssumedBehavior(unittest.TestCase):
    """Tests for non-obvious behavior we rely on

    If this behavior changes, these tests will fail and hopefully we won't
    waste time debugging the "real" tests. This behavior includes:
    - The error message when we try to evaluate an expression with
      uninitialized variables

    """

    def test_uninitialized_value_error_message(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2])
        m.x[1].set_value(5)
        msg = "No value for uninitialized VarData"
        with self.assertRaisesRegex(ValueError, msg):
            pyo.value(1 + m.x[1] * m.x[2])


class _TestIncidence:
    """Base class with tests for get_incident_variables that should be
    independent of the method used

    """

    def _get_incident_variables(self, expr):
        raise NotImplementedError("_TestIncidence should not be used directly")

    def test_basic_incidence(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = m.x[1] + m.x[1] * m.x[2] + m.x[1] * pyo.exp(m.x[3])
        variables = self._get_incident_variables(expr)
        self.assertEqual(ComponentSet(variables), ComponentSet(m.x[:]))

    def test_incidence_with_fixed_variable(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=1.0)
        expr = m.x[1] + m.x[1] * m.x[2] + m.x[1] * pyo.exp(m.x[3])
        m.x[2].fix()
        variables = self._get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet([m.x[1], m.x[3]]))

    def test_incidence_with_named_expression(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.subexpr = pyo.Expression(pyo.Integers)
        m.subexpr[1] = m.x[1] * pyo.exp(m.x[3])
        expr = m.x[1] + m.x[1] * m.x[2] + m.subexpr[1]
        variables = self._get_incident_variables(expr)
        self.assertEqual(ComponentSet(variables), ComponentSet(m.x[:]))


class _TestIncidenceLinearOnly:
    """Tests for methods that support linear_only"""

    def _get_incident_variables(self, expr):
        raise NotImplementedError(
            "_TestIncidenceLinearOnly should not be used directly"
        )

    def test_linear_only(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])

        expr = 2 * m.x[1] + 4 * m.x[2] * m.x[1] - m.x[1] * pyo.exp(m.x[3])
        variables = self._get_incident_variables(expr, linear_only=True)
        self.assertEqual(len(variables), 0)

        expr = 2 * m.x[1] + 2 * m.x[2] * m.x[3] + 3 * m.x[2]
        variables = self._get_incident_variables(expr, linear_only=True)
        self.assertEqual(ComponentSet(variables), ComponentSet([m.x[1]]))

        m.x[3].fix(2.5)
        expr = 2 * m.x[1] + 2 * m.x[2] * m.x[3] + 3 * m.x[2]
        variables = self._get_incident_variables(expr, linear_only=True)
        self.assertEqual(ComponentSet(variables), ComponentSet([m.x[1], m.x[2]]))


class _TestIncidenceLinearCancellation:
    """Tests for methods that perform linear cancellation"""

    def _get_incident_variables(self, expr):
        raise NotImplementedError(
            "_TestIncidenceLinearCancellation should not be used directly"
        )

    def test_zero_coef(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])

        # generate_standard_repn filters subexpressions with zero coefficients
        expr = 0 * m.x[1] + 0 * m.x[1] * m.x[2] + 0 * pyo.exp(m.x[3])
        variables = self._get_incident_variables(expr)
        self.assertEqual(len(variables), 0)

    def test_variable_minus_itself(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        # standard repn will recognize the zero coefficient and filter x[1]
        expr = m.x[1] + m.x[2] * m.x[3] - m.x[1]
        variables = self._get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet([m.x[2], m.x[3]]))

    def test_fixed_zero_linear_coefficient(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.p = pyo.Param([1, 2], mutable=True, initialize=1.0)
        m.p[1].set_value(0)
        expr = 2 * m.x[1] + m.p[1] * m.p[2] * m.x[2] + m.p[2] * m.x[3] ** 2
        variables = self._get_incident_variables(expr)
        self.assertEqual(ComponentSet(variables), ComponentSet([m.x[1], m.x[3]]))

        m.x[3].fix(0.0)
        expr = 2 * m.x[1] + 3 * m.x[3] * m.p[2] * m.x[2] + m.x[1] ** 2
        variables = self._get_incident_variables(expr)
        self.assertEqual(ComponentSet(variables), ComponentSet([m.x[1]]))

        m.x[3].fix(1.0)
        variables = self._get_incident_variables(expr)
        self.assertEqual(ComponentSet(variables), ComponentSet([m.x[1], m.x[2]]))

    # NOTE: This test assumes that all methods that support linear cancellation
    # accept a linear_only argument. If this changes, this test will need to be
    # moved.
    def test_fixed_zero_coefficient_linear_only(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = m.x[1] * m.x[2] + 2 * m.x[3]
        m.x[2].fix(0)
        variables = get_incident_variables(
            expr, method=IncidenceMethod.standard_repn, linear_only=True
        )
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[3])


class TestIncidenceStandardRepn(
    unittest.TestCase,
    _TestIncidence,
    _TestIncidenceLinearOnly,
    _TestIncidenceLinearCancellation,
):
    def _get_incident_variables(self, expr, **kwds):
        method = IncidenceMethod.standard_repn
        return get_incident_variables(expr, method=method, **kwds)

    def test_assumed_standard_repn_behavior(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2])
        m.p = pyo.Param(initialize=0.0)

        # We rely on variables with constant coefficients of zero not appearing
        # in the standard repn (as opposed to appearing with explicit
        # coefficients of zero).
        expr = m.x[1] + 0 * m.x[2]
        repn = generate_standard_repn(expr)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertIs(repn.linear_vars[0], m.x[1])

        expr = m.p * m.x[1] + m.x[2]
        repn = generate_standard_repn(expr)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertIs(repn.linear_vars[0], m.x[2])

    def test_fixed_none_linear_coefficient(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.p = pyo.Param([1, 2], mutable=True, initialize=1.0)
        m.x[3].fix(None)
        expr = 2 * m.x[1] + 3 * m.x[3] * m.p[2] * m.x[2] + m.x[1] ** 2
        variables = self._get_incident_variables(expr)
        self.assertEqual(ComponentSet(variables), ComponentSet([m.x[1], m.x[2]]))

    def test_incidence_with_mutable_parameter(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.p = pyo.Param(mutable=True, initialize=None)
        expr = m.x[1] + m.p * m.x[1] * m.x[2] + m.x[1] * pyo.exp(m.x[3])
        variables = self._get_incident_variables(expr)
        self.assertEqual(ComponentSet(variables), ComponentSet(m.x[:]))


class TestIncidenceIdentifyVariables(unittest.TestCase, _TestIncidence):
    def _get_incident_variables(self, expr, **kwds):
        method = IncidenceMethod.identify_variables
        return get_incident_variables(expr, method=method, **kwds)

    def test_zero_coef(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])

        # identify_variables does not eliminate expressions times zero
        expr = 0 * m.x[1] + 0 * m.x[1] * m.x[2] + 0 * pyo.exp(m.x[3])
        variables = self._get_incident_variables(expr)
        self.assertEqual(ComponentSet(variables), ComponentSet(m.x[:]))

    def test_variable_minus_itself(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        # identify_variables will not filter x[1]
        expr = m.x[1] + m.x[2] * m.x[3] - m.x[1]
        variables = self._get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet(m.x[:]))

    def test_incidence_with_mutable_parameter(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.p = pyo.Param(mutable=True, initialize=None)
        expr = m.x[1] + m.p * m.x[1] * m.x[2] + m.x[1] * pyo.exp(m.x[3])
        variables = self._get_incident_variables(expr)
        self.assertEqual(ComponentSet(variables), ComponentSet(m.x[:]))


class TestIncidenceAmplRepn(
    unittest.TestCase,
    _TestIncidence,
    _TestIncidenceLinearOnly,
    _TestIncidenceLinearCancellation,
):
    def _get_incident_variables(self, expr, **kwds):
        method = IncidenceMethod.ampl_repn
        return get_incident_variables(expr, method=method, **kwds)


class TestIncidenceStandardRepnComputeValues(
    unittest.TestCase,
    _TestIncidence,
    _TestIncidenceLinearOnly,
    _TestIncidenceLinearCancellation,
):
    def _get_incident_variables(self, expr, **kwds):
        method = IncidenceMethod.standard_repn_compute_values
        return get_incident_variables(expr, method=method, **kwds)


class TestGetVariablesIncidentToConstraint(unittest.TestCase):
    """
    Unit tests for the get_variables_incident_to_constraint helper function.
    This function was extracted to handle the logic of dispatching to different
    methods for identifying incident variables.
    """

    def test_identify_variables_method(self):
        """Test using identify_variables method"""
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = m.x[1] + m.x[1] * m.x[2] + m.x[1] * pyo.exp(m.x[3])
        
        variables = get_variables_incident_to_constraint(
            expr, 
            method=IncidenceMethod.identify_variables,
            include_fixed=False
        )
        self.assertEqual(ComponentSet(variables), ComponentSet(m.x[:]))

    def test_identify_variables_with_fixed(self):
        """Test identify_variables method with include_fixed=True"""
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=1.0)
        expr = m.x[1] + m.x[1] * m.x[2] + m.x[1] * pyo.exp(m.x[3])
        m.x[2].fix()
        
        # With include_fixed=True, should include all variables
        variables = get_variables_incident_to_constraint(
            expr, 
            method=IncidenceMethod.identify_variables,
            include_fixed=True
        )
        self.assertEqual(ComponentSet(variables), ComponentSet(m.x[:]))
        
        # With include_fixed=False, should exclude fixed variables
        variables = get_variables_incident_to_constraint(
            expr, 
            method=IncidenceMethod.identify_variables,
            include_fixed=False
        )
        self.assertEqual(ComponentSet(variables), ComponentSet([m.x[1], m.x[3]]))

    def test_standard_repn_method(self):
        """Test using standard_repn method"""
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = m.x[1] + 2 * m.x[2] + 3 * m.x[3] ** 2
        
        variables = get_variables_incident_to_constraint(
            expr,
            method=IncidenceMethod.standard_repn,
            include_fixed=False,
            linear_only=False
        )
        self.assertEqual(ComponentSet(variables), ComponentSet(m.x[:]))

    def test_standard_repn_linear_only(self):
        """Test standard_repn method with linear_only=True"""
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = 2 * m.x[1] + 2 * m.x[2] * m.x[3] + 3 * m.x[2]
        
        variables = get_variables_incident_to_constraint(
            expr,
            method=IncidenceMethod.standard_repn,
            include_fixed=False,
            linear_only=True
        )
        # Only x[1] is purely linear
        self.assertEqual(ComponentSet(variables), ComponentSet([m.x[1]]))

    def test_standard_repn_compute_values_method(self):
        """Test using standard_repn_compute_values method"""
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.p = pyo.Param([1, 2], mutable=True, initialize=1.0)
        expr = m.p[1] * m.x[1] + m.p[2] * m.x[2] + m.x[3] ** 2
        
        variables = get_variables_incident_to_constraint(
            expr,
            method=IncidenceMethod.standard_repn_compute_values,
            include_fixed=False,
            linear_only=False
        )
        self.assertEqual(ComponentSet(variables), ComponentSet(m.x[:]))

    def test_standard_repn_compute_values_zero_coefficient(self):
        """Test standard_repn_compute_values filters zero coefficients"""
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.p = pyo.Param([1, 2], mutable=True, initialize=1.0)
        m.p[1].set_value(0)
        expr = m.p[1] * m.x[1] + m.p[2] * m.x[2] + m.x[3] ** 2
        
        variables = get_variables_incident_to_constraint(
            expr,
            method=IncidenceMethod.standard_repn_compute_values,
            include_fixed=False,
            linear_only=False
        )
        # x[1] should be filtered out due to zero coefficient
        self.assertEqual(ComponentSet(variables), ComponentSet([m.x[2], m.x[3]]))

    def test_ampl_repn_method(self):
        """Test using ampl_repn method"""
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = m.x[1] + 2 * m.x[2] + 3 * m.x[3] ** 2
        
        # Create AMPLRepnVisitor
        config = get_config_from_kwds(method=IncidenceMethod.ampl_repn)
        visitor = config._ampl_repn_visitor
        
        variables = get_variables_incident_to_constraint(
            expr,
            method=IncidenceMethod.ampl_repn,
            linear_only=False,
            amplrepnvisitor=visitor
        )
        self.assertEqual(ComponentSet(variables), ComponentSet(m.x[:]))

    def test_ampl_repn_linear_only(self):
        """Test ampl_repn method with linear_only=True"""
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = 2 * m.x[1] + 2 * m.x[2] * m.x[3] + 3 * m.x[2]
        
        # Create AMPLRepnVisitor
        config = get_config_from_kwds(method=IncidenceMethod.ampl_repn)
        visitor = config._ampl_repn_visitor
        
        variables = get_variables_incident_to_constraint(
            expr,
            method=IncidenceMethod.ampl_repn,
            linear_only=True,
            amplrepnvisitor=visitor
        )
        # Only x[1] is purely linear
        self.assertEqual(ComponentSet(variables), ComponentSet([m.x[1]]))

    def test_egb_constraint_body(self):
        """Test special handling for EGBConstraintBody expressions"""
        # Create a mock EGBConstraintBody with a get_incident_variables method
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        
        mock_egb_body = MagicMock()
        mock_egb_body.get_incident_variables.return_value = [m.x[1], m.x[2]]
        
        # Import EGBConstraintBody to use isinstance check
        from pyomo.contrib.pynumero.interfaces.external_grey_box_constraint import (
            EGBConstraintBody,
        )
        
        # Make the mock an instance of EGBConstraintBody
        mock_egb_body.__class__ = EGBConstraintBody
        
        variables = get_variables_incident_to_constraint(
            mock_egb_body,
            method=IncidenceMethod.standard_repn,  # Method is ignored for EGBConstraintBody
            include_fixed=False,
            linear_only=False
        )
        
        # Should call get_incident_variables on the EGBConstraintBody object
        mock_egb_body.get_incident_variables.assert_called_once()
        self.assertEqual(variables, [m.x[1], m.x[2]])

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError"""
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = m.x[1] + m.x[2]
        
        # Create an invalid method (not a real IncidenceMethod)
        invalid_method = "not_a_real_method"
        
        with self.assertRaises(ValueError) as cm:
            get_variables_incident_to_constraint(
                expr,
                method=invalid_method,
                include_fixed=False,
                linear_only=False
            )
        
        self.assertIn("Unrecognized value", str(cm.exception))
        self.assertIn("for the method used to identify incident", str(cm.exception))

    def test_standard_repn_with_fixed_variables(self):
        """Test standard_repn with include_fixed parameter"""
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=1.0)
        expr = m.x[1] + m.x[2] + m.x[3] ** 2
        m.x[2].fix()
        
        # With include_fixed=True, temporarily unfix variables
        variables = get_variables_incident_to_constraint(
            expr,
            method=IncidenceMethod.standard_repn,
            include_fixed=True,
            linear_only=False
        )
        self.assertEqual(ComponentSet(variables), ComponentSet(m.x[:]))
        # Variable should still be fixed after the call
        self.assertTrue(m.x[2].fixed)
        
        # With include_fixed=False, exclude fixed variables
        variables = get_variables_incident_to_constraint(
            expr,
            method=IncidenceMethod.standard_repn,
            include_fixed=False,
            linear_only=False
        )
        self.assertEqual(ComponentSet(variables), ComponentSet([m.x[1], m.x[3]]))

    def test_zero_coefficient_filtering(self):
        """Test that variables with zero coefficients are properly filtered"""
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        
        # Expression where x[1] cancels out
        expr = m.x[1] + m.x[2] * m.x[3] - m.x[1]
        
        variables = get_variables_incident_to_constraint(
            expr,
            method=IncidenceMethod.standard_repn,
            include_fixed=False,
            linear_only=False
        )
        # x[1] should be filtered out
        self.assertEqual(ComponentSet(variables), ComponentSet([m.x[2], m.x[3]]))

    def test_nonlinear_variables_filtering(self):
        """Test filtering of variables that only appear nonlinearly when linear_only=True"""
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        
        # x[1] is linear, x[2] is both linear and nonlinear, x[3] is only nonlinear
        expr = 2 * m.x[1] + 3 * m.x[2] + m.x[2] * m.x[3] + m.x[3] ** 2
        
        variables = get_variables_incident_to_constraint(
            expr,
            method=IncidenceMethod.standard_repn,
            include_fixed=False,
            linear_only=True
        )
        # Only x[1] should be included (purely linear)
        self.assertEqual(ComponentSet(variables), ComponentSet([m.x[1]]))




if __name__ == "__main__":
    unittest.main()
