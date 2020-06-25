# -*- coding: utf-8 -*-
"""
Testing for the logical expression system
"""
from __future__ import division
import operator
import platform
import sys
from itertools import product

import pyutilib.th as unittest
import six

from pyomo.core.expr.cnf_walker import to_cnf
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import (
    land, atleast, atmost, BooleanConstant, BooleanVarList, ComponentMap, equivalent, exactly, implies, lor, RangeSet,
    value,
    ConcreteModel, BooleanVar,
    lnot, xor, )


def _generate_possible_truth_inputs(nargs):
    return product([True, False], repeat=nargs)


def _check_equivalent(assert_handle, expr_1, expr_2):
    expr_1_vars = list(identify_variables(expr_1, include_fixed=False))
    expr_2_vars = list(identify_variables(expr_2, include_fixed=False))
    assert_handle.assertEquals(len(expr_1_vars), len(expr_2_vars))
    for truth_combination in _generate_possible_truth_inputs(len(expr_1_vars)):
        for var, truth_value in zip(expr_1_vars, truth_combination):
            var.value = truth_value
        assert_handle.assertEquals(value(expr_1), value(expr_2))


class TestLogicalClasses(unittest.TestCase):

    def test_BooleanVar(self):
        """
        Simple construction and value setting
        """
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()

        self.assertIsNone(m.Y1.value)
        m.Y1.set_value(False)
        self.assertFalse(m.Y1.value)
        m.Y1.set_value(True)
        self.assertTrue(m.Y1.value)

    def test_unary_not(self):
        m = ConcreteModel()
        m.Y = BooleanVar()
        op_static = lnot(m.Y)
        op_operator = ~m.Y
        for truth_combination in _generate_possible_truth_inputs(1):
            m.Y.set_value(truth_combination[0])
            correct_value = not truth_combination[0]
            self.assertEquals(value(op_static), correct_value)
            self.assertEquals(value(op_operator), correct_value)

    def test_binary_equiv(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        op_static = equivalent(m.Y1, m.Y2)
        op_class = m.Y1.equivalent_to(m.Y2)
        # op_operator = m.Y1 == m.Y2
        for truth_combination in _generate_possible_truth_inputs(2):
            m.Y1.value, m.Y2.value = truth_combination[0], truth_combination[1]
            correct_value = operator.eq(*truth_combination)
            self.assertEquals(value(op_static), correct_value)
            self.assertEquals(value(op_class), correct_value)
            # self.assertEquals(value(op_operator), correct_value)

    def test_binary_xor(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        op_static = xor(m.Y1, m.Y2)
        op_class = m.Y1.xor(m.Y2)
        # op_operator = m.Y1 ^ m.Y2
        for truth_combination in _generate_possible_truth_inputs(2):
            m.Y1.value, m.Y2.value = truth_combination[0], truth_combination[1]
            correct_value = operator.xor(*truth_combination)
            self.assertEquals(value(op_static), correct_value)
            self.assertEquals(value(op_class), correct_value)
            # self.assertEquals(value(op_operator), correct_value)

    def test_binary_implies(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        op_static = implies(m.Y1, m.Y2)
        op_class = m.Y1.implies(m.Y2)
        # op_loperator = m.Y2 << m.Y1
        # op_roperator = m.Y1 >> m.Y2
        for truth_combination in _generate_possible_truth_inputs(2):
            m.Y1.value, m.Y2.value = truth_combination[0], truth_combination[1]
            correct_value = (not truth_combination[0]) or truth_combination[1]
            self.assertEquals(value(op_static), correct_value)
            self.assertEquals(value(op_class), correct_value)
            # self.assertEquals(value(op_loperator), correct_value)
            # self.assertEquals(value(op_roperator), correct_value)
            nnf = lnot(m.Y1).lor(m.Y2)
            self.assertEquals(value(op_static), value(nnf))

    def test_binary_and(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        op_static = land(m.Y1, m.Y2)
        op_class = m.Y1.land(m.Y2)
        # op_operator = m.Y1 & m.Y2
        for truth_combination in _generate_possible_truth_inputs(2):
            m.Y1.value, m.Y2.value = truth_combination[0], truth_combination[1]
            correct_value = all(truth_combination)
            self.assertEquals(value(op_static), correct_value)
            self.assertEquals(value(op_class), correct_value)
            # self.assertEquals(value(op_operator), correct_value)

    def test_binary_or(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        op_static = lor(m.Y1, m.Y2)
        op_class = m.Y1.lor(m.Y2)
        # op_operator = m.Y1 | m.Y2
        for truth_combination in _generate_possible_truth_inputs(2):
            m.Y1.value, m.Y2.value = truth_combination[0], truth_combination[1]
            correct_value = any(truth_combination)
            self.assertEquals(value(op_static), correct_value)
            self.assertEquals(value(op_class), correct_value)
            # self.assertEquals(value(op_operator), correct_value)

    def test_nary_and(self):
        nargs = 3
        m = ConcreteModel()
        m.s = RangeSet(nargs)
        m.Y = BooleanVar(m.s)
        op_static = land(*(m.Y[i] for i in m.s))
        op_class = BooleanConstant(True)
        # op_operator = True
        for y in m.Y.values():
            op_class = op_class.land(y)
            # op_operator &= y
        for truth_combination in _generate_possible_truth_inputs(nargs):
            m.Y.set_values(dict(enumerate(truth_combination, 1)))
            correct_value = all(truth_combination)
            self.assertEquals(value(op_static), correct_value)
            self.assertEquals(value(op_class), correct_value)
            # self.assertEquals(value(op_operator), correct_value)

    def test_nary_or(self):
        nargs = 3
        m = ConcreteModel()
        m.s = RangeSet(nargs)
        m.Y = BooleanVar(m.s)
        op_static = lor(*(m.Y[i] for i in m.s))
        op_class = BooleanConstant(False)
        # op_operator = False
        for y in m.Y.values():
            op_class = op_class.lor(y)
            # op_operator |= y
        for truth_combination in _generate_possible_truth_inputs(nargs):
            m.Y.set_values(dict(enumerate(truth_combination, 1)))
            correct_value = any(truth_combination)
            self.assertEquals(value(op_static), correct_value)
            self.assertEquals(value(op_class), correct_value)
            # self.assertEquals(value(op_operator), correct_value)

    def test_nary_exactly(self):
        nargs = 5
        m = ConcreteModel()
        m.s = RangeSet(nargs)
        m.Y = BooleanVar(m.s)
        for truth_combination in _generate_possible_truth_inputs(nargs):
            for ntrue in range(nargs + 1):
                m.Y.set_values(dict(enumerate(truth_combination, 1)))
                correct_value = sum(truth_combination) == ntrue
                self.assertEquals(value(exactly(ntrue, *(m.Y[i] for i in m.s))), correct_value)
                self.assertEquals(value(exactly(ntrue, m.Y)), correct_value)

    def test_nary_atmost(self):
        nargs = 5
        m = ConcreteModel()
        m.s = RangeSet(nargs)
        m.Y = BooleanVar(m.s)
        for truth_combination in _generate_possible_truth_inputs(nargs):
            for ntrue in range(nargs + 1):
                m.Y.set_values(dict(enumerate(truth_combination, 1)))
                correct_value = sum(truth_combination) <= ntrue
                self.assertEquals(value(atmost(ntrue, *(m.Y[i] for i in m.s))), correct_value)
                self.assertEquals(value(atmost(ntrue, m.Y)), correct_value)

    def test_nary_atleast(self):
        nargs = 5
        m = ConcreteModel()
        m.s = RangeSet(nargs)
        m.Y = BooleanVar(m.s)
        for truth_combination in _generate_possible_truth_inputs(nargs):
            for ntrue in range(nargs + 1):
                m.Y.set_values(dict(enumerate(truth_combination, 1)))
                correct_value = sum(truth_combination) >= ntrue
                self.assertEquals(value(atleast(ntrue, *(m.Y[i] for i in m.s))), correct_value)
                self.assertEquals(value(atleast(ntrue, m.Y)), correct_value)

    def test_to_string(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        m.Y3 = BooleanVar()

        self.assertEqual(str(land(m.Y1, m.Y2, m.Y3)), "Y1 ∧ Y2 ∧ Y3")
        self.assertEqual(str(lor(m.Y1, m.Y2, m.Y3)), "Y1 ∨ Y2 ∨ Y3")
        self.assertEqual(str(equivalent(m.Y1, m.Y2)), "Y1 iff Y2")
        self.assertEqual(str(implies(m.Y1, m.Y2)), "Y1 --> Y2")
        self.assertEqual(str(xor(m.Y1, m.Y2)), "Y1 ⊻ Y2")
        self.assertEqual(str(atleast(1, m.Y1, m.Y2)), "atleast(1: [Y1, Y2])")
        self.assertEqual(str(atmost(1, m.Y1, m.Y2)), "atmost(1: [Y1, Y2])")
        self.assertEqual(str(exactly(1, m.Y1, m.Y2)), "exactly(1: [Y1, Y2])")

        # Precedence check
        self.assertEquals(str(m.Y1.implies(m.Y2).lor(m.Y3)), "(Y1 --> Y2) ∨ Y3")

    def test_node_types(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        m.Y3 = BooleanVar()

        self.assertFalse(m.Y1.is_expression_type())
        self.assertTrue(lnot(m.Y1).is_expression_type())
        self.assertTrue(equivalent(m.Y1, m.Y2).is_expression_type())
        self.assertTrue(atmost(1, [m.Y1, m.Y2, m.Y3]).is_expression_type())

    def test_numeric_invalid(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        m.Y3 = BooleanVar()

        def invalid_expression_generator():
            yield lambda: m.Y1 + m.Y2
            yield lambda: m.Y1 - m.Y2
            yield lambda: m.Y1 * m.Y2
            yield lambda: m.Y1 / m.Y2
            yield lambda: m.Y1**m.Y2
            yield lambda: 0 + m.Y2
            yield lambda: 0 - m.Y2
            yield lambda: 0 * m.Y2
            yield lambda: 0 / m.Y2
            yield lambda: 0**m.Y2

        numeric_error_msg = "(?:(?:unsupported operand type)|(?:operands do not support))"
        for invalid_expr_fcn in invalid_expression_generator():
            with self.assertRaisesRegex(TypeError, numeric_error_msg):
                _ = invalid_expr_fcn()

        def invalid_unary_expression_generator():
            yield lambda: -m.Y1
            yield lambda: +m.Y1

        for invalid_expr_fcn in invalid_unary_expression_generator():
            with self.assertRaisesRegex(TypeError, "(?:(?:bad operand type for unary)"
                                                   "|(?:unsupported operand type for unary))"):
                _ = invalid_expr_fcn()

        def invalid_comparison_generator():
            yield lambda: m.Y1 >= 0
            yield lambda: m.Y1 <= 0
            yield lambda: m.Y1 > 0
            yield lambda: m.Y1 < 0

        # These errors differ between python versions, regrettably
        comparison_error_msg = "(?:(?:unorderable types)|(?:not supported between instances of))"
        if six.PY3:
            for invalid_expr_fcn in invalid_comparison_generator():
                with self.assertRaisesRegex(TypeError, comparison_error_msg):
                    _ = invalid_expr_fcn()
        else:  # Python 2
            pass
            # Note: Python 2 behavior is weird, returning native type bool:
            # m.Y1 >= 0  -> True
            # m.Y1 <= 0  -> False
            # m.Y1 > 0   -> True
            # m.Y1 < 0   -> False

    def test_invalid_conversion(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()

        with self.assertRaisesRegex(
                TypeError, "argument must be a string or a number"):
            float(m.Y1)

        with self.assertRaisesRegex(
                TypeError, "argument must be a string"
                           "(?:, a bytes-like object)? or a number"):
            int(m.Y1)


@unittest.skipUnless(sympy_available, "sympy not available")
class TestCNF(unittest.TestCase):
    def test_cnf(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()

        implication = implies(m.Y1, m.Y2)
        x = to_cnf(implication)[0]
        _check_equivalent(self, implication, x)

        atleast_expr = atleast(1, m.Y1, m.Y2)
        x = to_cnf(atleast_expr)[0]
        self.assertIs(atleast_expr, x)  # should be no change

        nestedatleast = implies(m.Y1, atleast_expr)
        m.extraY = BooleanVarList()
        indicator_map = ComponentMap()
        x = to_cnf(nestedatleast, m.extraY, indicator_map)
        self.assertEquals(str(x[0]), "extraY[1] ∨ ~Y1")
        self.assertIs(indicator_map[m.extraY[1]], atleast_expr)

    # TODO need to test other combinations as well


if __name__ == "__main__":
    unittest.main()
