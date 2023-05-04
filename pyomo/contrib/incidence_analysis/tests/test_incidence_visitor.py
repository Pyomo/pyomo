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

import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.incidence_analysis.visitor import (
    get_incident_variables,
    _get_ampl_expr,
)


class TestUninitialized(unittest.TestCase):
    def test_assumed_behavior(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2])
        self.assertIs(m.x[1].value, None)

    def test_product_one_fixed(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2])
        m.x[1].fix()

        variables = get_incident_variables(m.x[1] * m.x[2])
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[2])

        variables = get_incident_variables(m.x[2] * m.x[1])
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[2])

    def test_uninit_named_expr_times_linear(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.expr = pyo.Expression(expr=m.x[1] + 2 * m.x[2])
        m.x[1].fix()
        m.x[2].fix()
        variables = get_incident_variables(m.x[3] * m.expr)
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[3])

    def test_nonlinear(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = m.x[1] * m.x[2] * m.x[3]
        m.x[2].fix()
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(len(var_set), 2)
        self.assertIn(m.x[1], var_set)
        self.assertIn(m.x[3], var_set)

        expr = m.x[3] * pyo.exp(m.x[1] ** m.x[2])
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(len(var_set), 2)
        self.assertIn(m.x[1], var_set)
        self.assertIn(m.x[3], var_set)

        expr = m.x[1] * m.x[2] * m.x[3] - m.x[3] * pyo.exp(m.x[1] ** m.x[2])
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(len(var_set), 2)
        self.assertIn(m.x[1], var_set)
        self.assertIn(m.x[3], var_set)

    def test_nonlinear_fixed_to_power(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.x[2].fix()
        expr = 5 * m.x[1] + m.x[2] ** 3 * m.x[3] ** 2
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(len(var_set), 2)
        self.assertIn(m.x[1], var_set)
        self.assertIn(m.x[3], var_set)

    def test_named_expr(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.named_expr1 = pyo.Expression(expr=m.x[1] ** 2 + m.x[2] ** 2)
        m.named_expr2 = pyo.Expression(expr=5 * m.x[2] + m.x[2] ** 3)
        expr = (
            m.named_expr1 * m.x[3]
            + m.x[1] ** 2 * m.x[2] * m.named_expr2
            + m.named_expr1
        )
        m.x[1].fix()
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet([m.x[2], m.x[3]]))

    def test_trivial_named_expr(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.named_expr1 = pyo.Expression(expr=m.x[1])
        m.named_expr2 = pyo.Expression(expr=3)
        # This was meant to test direct replacement that occurs even
        # when used_named_exprs=True, but we never use this option at
        # this point.
        expr = (
            m.named_expr1 * m.x[2]
            + m.x[2] ** 2
            + m.named_expr1 * m.named_expr2
            + m.named_expr2
        )
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet([m.x[1], m.x[2]]))

    def test_combine_like_linear(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = (m.x[1] + m.x[1] ** 2) + (m.x[2] * m.x[1] + 2 * m.x[1] ** 3)
        m.x[2].fix()
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[1])

        expr = m.x[1] + 3 * m.x[3] + m.x[2] * m.x[1]
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(len(variables), 2)
        self.assertIn(m.x[1], var_set)
        self.assertIn(m.x[3], var_set)

    def test_fixed_var_pow_zero(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = m.x[1] ** 0 * m.x[2]
        m.x[1].fix()
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet([m.x[2]]))

        expr = m.x[1] ** 0 * m.x[2] - m.x[2]
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 0)

    def test_one_pow_fixed_var(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = 1 ** m.x[1] * m.x[2]
        m.x[1].fix()
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet([m.x[2]]))

        expr = 1 ** m.x[1] * m.x[2] - m.x[2]
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 0)

    def test_zero_pow_fixed_var(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        # This does not induce a cancellation as 0**x[1] can be 1
        # if x[1] == 0
        expr = 0 ** m.x[1] * m.x[2]
        m.x[1].fix()
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet([m.x[2]]))

    def test_const_pow_fixed_var(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = 1.5 ** m.x[1] * m.x[2]
        m.x[1].fix()
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet([m.x[2]]))

    def test_abs_fixed_var(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.x[1].fix()
        expr = abs(m.x[1]) * m.x[2]
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet([m.x[2]]))

    def test_abs(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = abs(m.x[1]) * m.x[2]
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet([m.x[1], m.x[2]]))

    def test_sin_fixed_var(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.x[1].fix()
        expr = pyo.sin(m.x[1]) * m.x[2]
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 1)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet([m.x[2]]))

    def test_expr_if(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.x[1].fix()

        # Cancellation is branch-dependent, so x[2] "is incident"
        expr = m.x[2] * pyo.Expr_if(m.x[1], 2, 3) - 3 * m.x[2]
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[2])

        # Cancellation is not branch-dependent
        expr = m.x[2] * pyo.Expr_if(m.x[1], 3, 3) - 3 * m.x[2]
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 0)

        nan = float('nan')
        expr = m.x[2] * pyo.Expr_if(m.x[1], nan, nan)
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 0)

        # If one branch is NaN, assume we will avoid it (similar to assuming
        # that None != 0 when we have 1/None)
        expr = m.x[2] * pyo.Expr_if(m.x[1], nan, 2)
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[2])

        expr = m.x[2] * pyo.Expr_if(m.x[1], 2, nan)
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[2])

        # If either branch is None, no cancellation occurs
        expr = m.x[2] * pyo.Expr_if(m.x[1], 3, None) - 3 * m.x[2]
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[2])

        # IMO incidence graph generation with Expr_if that branches on an
        # uninitialized parameter is only well-defined if both branches
        # contain the same variables. We haven't implemented this, so
        # we just raise an error for uninitialized parameters with
        # non-constant branches.
        expr = pyo.Expr_if(m.x[1], m.x[2], m.x[2] ** 2)
        msg = "Cannot generate incident variables for Expr_if"
        with self.assertRaisesRegex(ValueError, msg):
            variables = get_incident_variables(expr)

        expr = pyo.Expr_if(m.x[1], m.x[2], -m.x[2] + m.x[3])
        msg = "Cannot generate incident variables for Expr_if"
        with self.assertRaisesRegex(ValueError, msg):
            variables = get_incident_variables(expr)

    def test_product_two_constants(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.x[1].fix()
        expr = m.x[1] * m.x[2] * m.x[3]
        m.x[1].fix()
        m.x[2].fix(0)
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 0)

        m.x[1].fix(0)
        m.x[2].fix(None)
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 0)

        nan = float("nan")
        m.x[1].fix(nan)
        expr = (m.x[1] * m.x[2] + 1) * m.x[3]
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 0)

    def test_negation_fixed_var(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        # Why does this get treated as negation(monomial) but -2*x[1] does not?
        expr = -(2 * m.x[1] * m.x[2]) + 2 * m.x[1] * m.x[2]
        m.x[1].fix()
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[2])

    def test_division_by_fixed_var(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = m.x[1] / m.x[2]
        m.x[2].fix()
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[1])

        expr = (m.x[1] + m.x[3]) / m.x[2] - m.x[1]
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet([m.x[1], m.x[3]]))

        # I am trying to test an `if arg1[1].mult == 0` branch, but it is very
        # non-obvious how I could ever get AMPLRepn.mult to be zero.
        m.x[3].fix(0)
        # This currently tests the constant NaN/None branch
        expr = ((m.x[1] + m.x[1] ** 2) / m.x[3]) / m.x[2]
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 0)

        m.x[3].fix(1)
        expr = m.x[3] / m.x[2] * m.x[1]
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[1])

        m.x[3].fix(0)
        expr = m.x[3] / m.x[2] * m.x[1]
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 0)

        m.x[3].fix(0)
        expr = (m.x[2] / m.x[3] + 1) * m.x[1]
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 0)

        m.x[3].fix(3)
        expr = m.x[2] / m.x[3] * m.x[1]
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[1])

    def test_fixed_var_divided_by(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = m.x[2] / m.x[3]
        m.x[2].fix()
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[3])

    def test_external_function(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])

        def fcn(*args):
            return args[0] + 0.5 * args[1] ** 2

        m.ef = pyo.ExternalFunction(fcn)
        expr = m.x[1] * m.ef(m.x[2], m.x[3])
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(len(variables), 3)
        self.assertEqual(var_set, ComponentSet(m.x[:]))

        m.x[2].fix()
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(len(variables), 2)
        self.assertEqual(var_set, ComponentSet([m.x[1], m.x[3]]))

        m.x[3].fix()
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(len(variables), 1)
        self.assertEqual(var_set, ComponentSet([m.x[1]]))


class TestInitialized(unittest.TestCase):
    def test_nonlinear(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=1)
        expr = m.x[1] * m.x[2] * m.x[3]
        m.x[2].fix()
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(len(var_set), 2)
        self.assertIn(m.x[1], var_set)
        self.assertIn(m.x[3], var_set)

    def test_nonlinear_with_const_mult(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=1)
        expr = 5 * m.x[1] + (-1) * m.x[2] ** 3 * m.x[3] ** 2
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet(m.x[:]))

        # There is a (amplrepn.mult != -1) branch that we don't
        # cover if we fix either of m.x[2] or m.x[3]
        expr = 5 * m.x[1] + 3 * m.x[2] ** 3 * m.x[3] ** 2
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet([m.x[1], m.x[2], m.x[3]]))

    def test_combine_like_linear(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=2)
        expr = (m.x[1] + m.x[1] ** 2) + (m.x[2] * m.x[1] + 2 * m.x[1] ** 3)
        m.x[2].fix()
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[1])

    def test_pow_fixed_var(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=2)
        expr = m.x[1] ** 2 * m.x[2]
        m.x[1].fix()
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[2])

    def test_var_pow_zero(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=2)
        expr = m.x[1] ** 0 * m.x[2]
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[2])

        expr = m.x[1] ** 0 * m.x[2] - m.x[2]
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 0)

    def test_var_pow_one(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=2)
        m.x[1].fix(1)
        # x**1 appears to get simplified during expression generation, so
        # using a fixed var here is necessary to cover the x**1 simplification
        # in the visitor.
        expr = m.x[2] ** m.x[1]
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[2])

        expr = m.x[2] ** m.x[1] - m.x[2]
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 0)

    def test_abs_fixed_var(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=-1)
        m.x[1].fix()
        expr = abs(m.x[1]) * m.x[2]
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet([m.x[2]]))

        expr = abs(m.x[1]) * m.x[2] - m.x[2]
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 0)

    def test_exprif(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=1)
        expr = pyo.Expr_if(m.x[1], m.x[2], m.x[3])
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet(m.x[:]))

        m.x[1].fix(0)
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet([m.x[3]]))

        m.x[1].fix(2)
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet([m.x[2]]))

    def test_equality(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=1)
        expr = m.x[1] + m.x[2] == m.x[2]
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        # Note that cancellations don't occur with equality expressions.
        # Should relational expressions be supported?
        self.assertEqual(var_set, ComponentSet([m.x[1], m.x[2]]))

        m.con = pyo.Constraint(expr=m.x[1] + m.x[2] == m.x[2])
        variables = get_incident_variables(m.con.expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet([m.x[1], m.x[2]]))

        # I believe body is guaranteed to contain all variables.
        variables = get_incident_variables(m.con.body)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet([m.x[1]]))

        m.x[:].fix()
        variables = get_incident_variables(m.con.expr)
        self.assertEqual(len(variables), 0)

    def test_inequality(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=1)
        expr = m.x[1] + m.x[2] <= m.x[2]
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        # Note that cancellations don't occur with inequality expressions.
        # Should relational expressions be supported?
        self.assertEqual(var_set, ComponentSet([m.x[1], m.x[2]]))

        m.con = pyo.Constraint(expr=m.x[1] + m.x[2] <= m.x[2])
        variables = get_incident_variables(m.con.expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet([m.x[1], m.x[2]]))

        # I believe body is guaranteed to contain all variables.
        variables = get_incident_variables(m.con.body)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet([m.x[1]]))

        m.x[:].fix()
        variables = get_incident_variables(m.con.expr)
        self.assertEqual(len(variables), 0)

    def test_product_two_constants(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=1)
        expr = m.x[1] * m.x[2] * m.x[3]
        m.x[1].fix(2)
        m.x[2].fix(0)
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 0)

        nan = float('nan')
        m.x[1].fix(nan)
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 0)

        # NOTE: This test relies on deprecated behavior and should be
        # updated once 0*nan -> nan (correctly) rather than 0
        expr = (m.x[1] * m.x[2] + 1) * m.x[3]
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[3])

        expr = m.x[1] * m.x[2] * m.x[3]
        m.x[1].fix(0)
        m.x[2].fix(2)
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 0)

        # NOTE: This test relies on deprecated behavior and should be
        # updated once 0*nan -> nan (correctly) rather than 0
        expr = (m.x[1] * m.x[2] + 1) * m.x[3]
        m.x[2].fix(nan)
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 1)
        self.assertEqual(variables[0], m.x[3])

        expr = m.x[1] * m.x[2] * m.x[3] - 12 * m.x[3]
        m.x[1].fix(3)
        m.x[2].fix(4)
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 0)

    def test_product_const_general(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=1)
        nan = float('nan')
        expr = nan * (m.x[1] + m.x[2])
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 0)

    # def test_monomial_zero_coef(self):
    #    # How do I generate a monomial with zero coef?
    #    # 0*x gets replaced by _CONSTANT
    #    m = pyo.ConcreteModel()
    #    m.x = pyo.Var([1, 2, 3], initialize=1)
    #    #expr = (m.x[1] + 3)**2 * (2*m.x[2] - 2*m.x[2]) + m.x[3]

    #    # The cancellation works here
    #    #expr = (m.x[1]**2) * 0
    #    # but not here
    #    expr = (m.x[1] ** 2) * (m.x[2] - m.x[2])

    #    variables = get_incident_variables(expr)
    #    var_set = ComponentSet(variables)
    #    self.assertEqual(len(variables), 0)
    #    #self.assertIn(m.x[3], var_set)
    #    self.assertNotIn(m.x[1], var_set)
    #    #self.assertNotIn(m.x[2], var_set)

    def test_negation_fixed_var(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=1)
        # Why does this get treated as negation(monomial)
        # but -2*m.x[1] does not?
        expr = -(2 * m.x[1] * m.x[2]) + 2 * m.x[1] * m.x[2]
        m.x[1].fix()
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 0)

    def test_division_by_fixed_var(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=2)
        expr = m.x[1] / m.x[2] - m.x[1] / 2
        m.x[2].fix()
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 0)

        m.x[2].fix(0)
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[1])

        m.x[2].fix(2)
        expr = (m.x[1] + m.x[3]) / m.x[2] - m.x[1] / 2
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[3])

        m.x[2].fix(0)
        expr = (m.x[1] + m.x[3]) / m.x[2] - m.x[1] / 2
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[1])

    def test_fixed_var_divided_by(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=1)
        expr = m.x[2] / m.x[3]
        m.x[2].fix(0)
        variables = get_incident_variables(expr)
        self.assertEqual(len(variables), 0)

    def test_external_function(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=1)

        def fcn(*args):
            return args[0] + 0.5 * args[1] ** 2

        m.ef = pyo.ExternalFunction(fcn)
        expr = m.x[1] * m.ef(m.x[2], m.x[3])
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(len(variables), 3)
        self.assertEqual(var_set, ComponentSet(m.x[:]))

        m.x[2].fix()
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(len(variables), 2)
        self.assertEqual(var_set, ComponentSet([m.x[1], m.x[3]]))

        m.x[3].fix()
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(len(variables), 1)
        self.assertEqual(var_set, ComponentSet([m.x[1]]))


if __name__ == "__main__":
    unittest.main()
