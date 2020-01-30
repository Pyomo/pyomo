import operator
from itertools import product

import pyutilib.th as unittest

from pyomo.core.expr.cnf_walker import to_cnf
from pyomo.core.expr.logical_expr import (Not, Equivalent,
                                          Or, Implies, And, Exactly, AtMost, AtLeast, Xor,
                                          )
from pyomo.core.expr.logicalvalue import LogicalConstant
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import *

'''
First testing file.
'''


def create_model1(y):
    # model 1 with 5 checkpoints, 11 literals
    c1 = (Implies(y[1], y[2])).equals(Xor(y[3], y[4]))
    c2 = y[0] & Not(c1)
    c3 = Not(Implies(y[5], y[6]))
    c4 = Or(y[7], y[8], y[9])
    c5 = And((c3 & c4), y[10])
    root_node = c2 | Not(c5)
    return root_node


def create_model2(y):
    # model 2 with 3 checkpoints, 11 literals
    c1 = Not(y[1]).implies(y[2] ^ y[3])
    c2 = And(Or(y[4], y[5], y[6]))
    c3 = Equivalent(Xor(y[8], y[9]), y[10])
    root_node = Or(y[0], And(c1, c2), Not(y[7]), c3)
    return root_node


def create_model3(y):
    # model3
    a1 = And(y[0].implies(y[1]), Not(y[2]))
    a3 = And(Xor(y[4], y[5]))
    root_node = a1 | And(y[3]) | a3
    return root_node


def create_model4(y):
    # model4
    a0 = y[0].implies(y[1])
    a1 = Not(y[2])
    a2 = a0 & a1
    a3 = a2 | y[3]
    return a3
    # return (y[0].implies(y[1]) and Not(y[2])) or y[3]


def create_model5(y):
    # model 5, for cnf walker
    a2 = And(y[0], y[1])
    o1 = Or(a2, y[3])
    a1 = And(o1, y[4])
    return a1


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
        Test 1
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
        op_static = Not(m.Y)
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
        op_static = Equivalent(m.Y1, m.Y2)
        op_class = m.Y1.equivalent_to(m.Y2)
        op_operator = m.Y1 == m.Y2
        for truth_combination in _generate_possible_truth_inputs(2):
            m.Y1.value, m.Y2.value = truth_combination[0], truth_combination[1]
            correct_value = operator.eq(*truth_combination)
            self.assertEquals(value(op_static), correct_value)
            self.assertEquals(value(op_class), correct_value)
            self.assertEquals(value(op_operator), correct_value)

    def test_binary_xor(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        op_static = Xor(m.Y1, m.Y2)
        op_class = m.Y1.xor(m.Y2)
        op_operator = m.Y1 ^ m.Y2
        for truth_combination in _generate_possible_truth_inputs(2):
            m.Y1.value, m.Y2.value = truth_combination[0], truth_combination[1]
            correct_value = operator.xor(*truth_combination)
            self.assertEquals(value(op_static), correct_value)
            self.assertEquals(value(op_class), correct_value)
            self.assertEquals(value(op_operator), correct_value)

    def test_binary_implies(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        op_static = Implies(m.Y1, m.Y2)
        op_class = m.Y1.implies(m.Y2)
        op_loperator = m.Y2 << m.Y1
        op_roperator = m.Y1 >> m.Y2
        for truth_combination in _generate_possible_truth_inputs(2):
            m.Y1.value, m.Y2.value = truth_combination[0], truth_combination[1]
            correct_value = (not truth_combination[0]) or truth_combination[1]
            self.assertEquals(value(op_static), correct_value)
            self.assertEquals(value(op_class), correct_value)
            self.assertEquals(value(op_loperator), correct_value)
            self.assertEquals(value(op_roperator), correct_value)
            nnf = Not(m.Y1) | m.Y2
            self.assertEquals(value(op_static), value(nnf))

    def test_binary_and(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        op_static = And(m.Y1, m.Y2)
        op_class = m.Y1.and_(m.Y2)
        op_operator = m.Y1 & m.Y2
        for truth_combination in _generate_possible_truth_inputs(2):
            m.Y1.value, m.Y2.value = truth_combination[0], truth_combination[1]
            correct_value = all(truth_combination)
            self.assertEquals(value(op_static), correct_value)
            self.assertEquals(value(op_class), correct_value)
            self.assertEquals(value(op_operator), correct_value)

    def test_binary_or(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        op_static = Or(m.Y1, m.Y2)
        op_class = m.Y1.or_(m.Y2)
        op_operator = m.Y1 | m.Y2
        for truth_combination in _generate_possible_truth_inputs(2):
            m.Y1.value, m.Y2.value = truth_combination[0], truth_combination[1]
            correct_value = any(truth_combination)
            self.assertEquals(value(op_static), correct_value)
            self.assertEquals(value(op_class), correct_value)
            self.assertEquals(value(op_operator), correct_value)

    def test_nary_and(self):
        nargs = 3
        m = ConcreteModel()
        m.s = RangeSet(nargs)
        m.Y = BooleanVar(m.s)
        op_static = And(*(m.Y[i] for i in m.s))
        op_class = LogicalConstant(True)
        op_operator = True
        for y in m.Y.values():
            op_class = op_class.and_(y)
            op_operator &= y
        for truth_combination in _generate_possible_truth_inputs(nargs):
            m.Y.set_values(dict(enumerate(truth_combination, 1)))
            correct_value = all(truth_combination)
            self.assertEquals(value(op_static), correct_value)
            self.assertEquals(value(op_class), correct_value)
            self.assertEquals(value(op_operator), correct_value)

    def test_nary_or(self):
        nargs = 3
        m = ConcreteModel()
        m.s = RangeSet(nargs)
        m.Y = BooleanVar(m.s)
        op_static = Or(*(m.Y[i] for i in m.s))
        op_class = LogicalConstant(False)
        op_operator = False
        for y in m.Y.values():
            op_class = op_class.or_(y)
            op_operator |= y
        for truth_combination in _generate_possible_truth_inputs(nargs):
            m.Y.set_values(dict(enumerate(truth_combination, 1)))
            correct_value = any(truth_combination)
            self.assertEquals(value(op_static), correct_value)
            self.assertEquals(value(op_class), correct_value)
            self.assertEquals(value(op_operator), correct_value)

    def test_nary_exactly(self):
        nargs = 5
        m = ConcreteModel()
        m.s = RangeSet(nargs)
        m.Y = BooleanVar(m.s)
        for truth_combination in _generate_possible_truth_inputs(nargs):
            for ntrue in range(nargs + 1):
                m.Y.set_values(dict(enumerate(truth_combination, 1)))
                correct_value = sum(truth_combination) == ntrue
                self.assertEquals(value(Exactly(ntrue, *(m.Y[i] for i in m.s))), correct_value)
                self.assertEquals(value(Exactly(ntrue, m.Y)), correct_value)

    def test_nary_atmost(self):
        nargs = 5
        m = ConcreteModel()
        m.s = RangeSet(nargs)
        m.Y = BooleanVar(m.s)
        for truth_combination in _generate_possible_truth_inputs(nargs):
            for ntrue in range(nargs + 1):
                m.Y.set_values(dict(enumerate(truth_combination, 1)))
                correct_value = sum(truth_combination) <= ntrue
                self.assertEquals(value(AtMost(ntrue, *(m.Y[i] for i in m.s))), correct_value)
                self.assertEquals(value(AtMost(ntrue, m.Y)), correct_value)

    def test_nary_atleast(self):
        nargs = 5
        m = ConcreteModel()
        m.s = RangeSet(nargs)
        m.Y = BooleanVar(m.s)
        for truth_combination in _generate_possible_truth_inputs(nargs):
            for ntrue in range(nargs + 1):
                m.Y.set_values(dict(enumerate(truth_combination, 1)))
                correct_value = sum(truth_combination) >= ntrue
                self.assertEquals(value(AtLeast(ntrue, *(m.Y[i] for i in m.s))), correct_value)
                self.assertEquals(value(AtLeast(ntrue, m.Y)), correct_value)

    def test_to_string(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        m.Y3 = BooleanVar()

        self.assertEqual(str(And(m.Y1, m.Y2, m.Y3)), "Y1 & Y2 & Y3")
        # TODO need to test other combinations as well

    def test_node_types(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        m.Y3 = BooleanVar()

        self.assertFalse(m.Y1.is_expression_type())
        self.assertTrue(Not(m.Y1).is_expression_type())
        self.assertTrue(Equivalent(m.Y1, m.Y2).is_expression_type())
        self.assertTrue(AtMost(1, [m.Y1, m.Y2, m.Y3]).is_expression_type())

    @unittest.skipUnless(sympy_available, "sympy not available")
    def test_cnf(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()

        implication = Implies(m.Y1, m.Y2)
        x = to_cnf(implication)[0]
        _check_equivalent(self, implication, x)

        atleast = AtLeast(1, m.Y1, m.Y2)
        x = to_cnf(atleast)[0]
        self.assertIs(atleast, x)  # should be no change

        nestedatleast = Implies(m.Y1, atleast)
        m.extraY = BooleanVarList()
        indicator_map = ComponentMap()
        x = to_cnf(nestedatleast, m.extraY, indicator_map)
        self.assertEquals(str(x[0]), "extraY[1] | (~Y1)")
        self.assertIs(indicator_map[m.extraY[1]], atleast)

        # TODO need to test other combinations as well


if __name__ == "__main__":
    unittest.main()
