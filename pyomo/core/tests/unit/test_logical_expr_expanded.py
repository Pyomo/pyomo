import pyutilib.th as unittest

import pyomo.core.expr.CNF as cnf
from pyomo.core.expr.logical_expr import (Not, Equivalence,
                                          LogicalOr, Implies, LogicalAnd, Exactly, AtMost, AtLeast, LogicalXor,
                                          )
from pyomo.environ import *

'''
First testing file.
'''


def create_model1(y):
    # model 1 with 5 checkpoints, 11 literals
    c1 = (Implies(y[1], y[2])).equals(LogicalXor(y[3], y[4]))
    c2 = y[0] & Not(c1)
    c3 = Not(Implies(y[5], y[6]))
    c4 = LogicalOr(y[7], y[8], y[9])
    c5 = LogicalAnd((c3 & c4), y[10])
    root_node = c2 | Not(c5)
    return root_node


def create_model2(y):
    # model 2 with 3 checkpoints, 11 literals
    c1 = Not(y[1]).implies(y[2] ^ y[3])
    c2 = LogicalAnd(LogicalOr(y[4], y[5], y[6]))
    c3 = Equivalence(LogicalXor(y[8], y[9]), y[10])
    root_node = LogicalOr(y[0], LogicalAnd(c1, c2), Not(y[7]), c3)
    return root_node


def create_model3(y):
    # model3
    a1 = LogicalAnd(y[0].implies(y[1]), Not(y[2]))
    a3 = LogicalAnd(LogicalXor(y[4], y[5]))
    root_node = a1 | LogicalAnd(y[3]) | a3
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
    a2 = LogicalAnd(y[0], y[1])
    o1 = LogicalOr(a2, y[3])
    a1 = LogicalAnd(o1, y[4])
    return a1


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

        m.Y1.value, m.Y2.value = True, True
        self.assertTrue(value(LogicalAnd(m.Y1, m.Y2)))
        self.assertTrue(value(LogicalOr(m.Y1, m.Y2)))
        self.assertTrue(value(Implies(m.Y1, m.Y2)))
        self.assertFalse(value(LogicalXor(m.Y1, m.Y2)))

        m.Y1.value, m.Y2.value = False, True
        self.assertFalse(value(LogicalAnd(m.Y1, m.Y2)))
        self.assertTrue(value(LogicalOr(m.Y1, m.Y2)))
        self.assertTrue(value(Implies(m.Y1, m.Y2)))
        self.assertTrue(value(LogicalXor(m.Y1, m.Y2)))

        m.Y1.value, m.Y2.value = True, False
        self.assertFalse(value(LogicalAnd(m.Y1, m.Y2)))
        self.assertTrue(value(LogicalOr(m.Y1, m.Y2)))
        self.assertFalse(value(Implies(m.Y1, m.Y2)))
        self.assertTrue(value(LogicalXor(m.Y1, m.Y2)))

        m.Y1.value, m.Y2.value = False, False
        self.assertFalse(value(LogicalAnd(m.Y1, m.Y2)))
        self.assertFalse(value(LogicalOr(m.Y1, m.Y2)))
        self.assertTrue(value(Implies(m.Y1, m.Y2)))
        self.assertFalse(value(LogicalXor(m.Y1, m.Y2)))

    def test_And_Or_nodes(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        m.Y3 = BooleanVar()
        m.Y4 = BooleanVar()

        m.Y1.value = True
        m.Y2.value = True
        m.Y3.value = True
        # try alterbate way to set value
        # m.Y1.set_value(True)
        # m.Y2.set_value(True)
        And_static = LogicalAnd(m.Y1, m.Y2, m.Y3)
        Or_static = LogicalOr(m.Y1, m.Y2, m.Y3)
        And_operator = m.Y1 & m.Y2 & m.Y3
        Or_operator = m.Y1 | m.Y2 | m.Y3
        self.assertTrue(value(And_static))
        self.assertTrue(value(Or_static))
        self.assertTrue(value(And_operator.implies(Or_operator)))
        self.assertTrue(value(Implies(And_operator, Or_operator)))

        m.Y1.value, m.Y2.value, m.Y3.value, = False, True, True
        And_static = LogicalAnd(m.Y1, m.Y2, m.Y3)
        Or_static = LogicalOr(m.Y1, m.Y2, m.Y3)
        And_operator = m.Y1 & m.Y2 & m.Y3
        Or_operator = m.Y1 | m.Y2 | m.Y3
        self.assertFalse(value(And_static))
        self.assertTrue(value(Or_static))
        self.assertTrue(value(And_operator.implies(Or_operator)))
        self.assertTrue(value(Implies(And_operator, Or_operator)))

    def test_binary_equiv(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        op_static = Equivalence(m.Y1, m.Y2)
        op_class = m.Y1.equals(m.Y2)
        op_operator = m.Y1 == m.Y2
        truth_table = [
            [True, True, True],
            [True, False, False],
            [False, True, False],
            [False, False, True]
        ]
        for line in truth_table:
            m.Y1.value, m.Y2.value = line[0], line[1]
            self.assertEquals(value(op_static), line[2])
            self.assertEquals(value(op_class), line[2])
            self.assertEquals(value(op_operator), line[2])

    def test_binary_xor(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        op_static = LogicalXor(m.Y1, m.Y2)
        op_class = m.Y1.xor(m.Y2)
        op_operator = m.Y1 ^ m.Y2
        truth_table = [
            [True, True, False],
            [True, False, True],
            [False, True, True],
            [False, False, False]
        ]
        for line in truth_table:
            m.Y1.value, m.Y2.value = line[0], line[1]
            self.assertEquals(value(op_static), line[2])
            self.assertEquals(value(op_class), line[2])
            self.assertEquals(value(op_operator), line[2])

    def test_binary_implies(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        op_static = Implies(m.Y1, m.Y2)
        op_class = m.Y1.implies(m.Y2)
        truth_table = [
            [True, True, True],
            [True, False, False],
            [False, True, True],
            [False, False, True]
        ]
        for line in truth_table:
            m.Y1.value, m.Y2.value = line[0], line[1]
            self.assertEquals(value(op_static), line[2])
            self.assertEquals(value(op_class), line[2])
            nnf = Not(m.Y1) | m.Y2
            self.assertEquals(value(op_static), value(nnf))

    def test_MultiArgsExpression(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        m.Y3 = BooleanVar()
        m.Y4 = BooleanVar()

        ###########----------------------###########

        """
        Test for AndExpression and OrExpression
        """

        m.Y1.value, m.Y2.value, m.Y3.value, m.Y4.value = True, True, True, True
        And_static = LogicalAnd(m.Y1, m.Y2, m.Y3, m.Y4)
        And_operator = m.Y1 & m.Y2 & m.Y3 & m.Y4
        self.assertTrue(value(And_static))
        self.assertTrue(value(And_operator))

        Or_static = LogicalOr(m.Y1, m.Y2, m.Y3, m.Y4)
        Or_operator = m.Y1 | m.Y2 | m.Y3 | m.Y4
        self.assertTrue(value(Or_static))
        self.assertTrue(value(Or_operator))

        m.Y1.value, m.Y2.value, m.Y3.value, m.Y4.value = True, True, True, False
        And_static = LogicalAnd(m.Y1, m.Y2, m.Y3, m.Y4)
        And_operator = m.Y1 & m.Y2 & m.Y3 & m.Y4
        self.assertFalse(value(And_static))
        self.assertFalse(value(And_operator))

        Or_static = LogicalOr(m.Y1, m.Y2, m.Y3, m.Y4)
        Or_operator = m.Y1 | m.Y2 | m.Y3 | m.Y4
        self.assertTrue(value(Or_static))
        self.assertTrue(value(Or_operator))

        m.Y1.value, m.Y2.value, m.Y3.value, m.Y4.value = False, False, False, False

        And_static = LogicalAnd(m.Y1, m.Y2, m.Y3, m.Y4)
        And_operator = m.Y1 & m.Y2 & m.Y3 & m.Y4
        self.assertFalse(value(And_static))
        self.assertFalse(value(And_operator))
        Or_static = LogicalOr(m.Y1, m.Y2, m.Y3, m.Y4)
        Or_operator = m.Y1 | m.Y2 | m.Y3 | m.Y4
        self.assertFalse(value(Or_static))
        self.assertFalse(value(Or_operator))

        ###########----------------------###########

        """
        Test for Exactly, AtMost and AtLeast
        """

        m.Y1.value, m.Y2.value, m.Y3.value, m.Y4.value = True, True, True, False
        Exactly_0 = Exactly(0, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        Exactly_1 = Exactly(1, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        Exactly_2 = Exactly(2, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        Exactly_3 = Exactly(3, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        Exactly_4 = Exactly(4, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        self.assertFalse(value(Exactly_0))
        self.assertFalse(value(Exactly_1))
        self.assertFalse(value(Exactly_2))
        self.assertTrue(value(Exactly_3))
        self.assertFalse(value(Exactly_4))

        AtMost_0 = AtMost(0, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        AtMost_1 = AtMost(1, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        AtMost_2 = AtMost(2, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        AtMost_3 = AtMost(3, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        AtMost_4 = AtMost(4, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        self.assertTrue(value(AtMost_0))
        self.assertTrue(value(AtMost_1))
        self.assertTrue(value(AtMost_2))
        self.assertTrue(value(AtMost_3))
        self.assertFalse(value(AtMost_4))

        AtLeast_0 = AtLeast(0, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        AtLeast_1 = AtLeast(1, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        AtLeast_2 = AtLeast(2, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        AtLeast_3 = AtLeast(3, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        AtLeast_4 = AtLeast(4, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        self.assertFalse(value(AtLeast_0))
        self.assertFalse(value(AtLeast_1))
        self.assertFalse(value(AtLeast_2))
        self.assertTrue(value(AtLeast_3))
        self.assertTrue(value(AtLeast_4))

        m.Y1.value, m.Y2.value, m.Y3.value, m.Y4.value = True, False, False, False

        Exactly_0 = Exactly(0, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        Exactly_1 = Exactly(1, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        Exactly_2 = Exactly(2, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        Exactly_3 = Exactly(3, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        Exactly_4 = Exactly(4, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        self.assertFalse(value(Exactly_0))
        self.assertTrue(value(Exactly_1))
        self.assertFalse(value(Exactly_2))
        self.assertFalse(value(Exactly_3))
        self.assertFalse(value(Exactly_4))

        AtMost_0 = AtMost(0, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        AtMost_1 = AtMost(1, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        AtMost_2 = AtMost(2, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        AtMost_3 = AtMost(3, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        AtMost_4 = AtMost(4, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        self.assertTrue(value(AtMost_0))
        self.assertTrue(value(AtMost_1))
        self.assertFalse(value(AtMost_2))
        self.assertFalse(value(AtMost_3))
        self.assertFalse(value(AtMost_4))

        AtLeast_0 = AtLeast(0, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        AtLeast_1 = AtLeast(1, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        AtLeast_2 = AtLeast(2, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        AtLeast_3 = AtLeast(3, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        AtLeast_4 = AtLeast(4, list([m.Y1, m.Y2, m.Y3, m.Y4]))
        self.assertFalse(value(AtLeast_0))
        self.assertTrue(value(AtLeast_1))
        self.assertTrue(value(AtLeast_2))
        self.assertTrue(value(AtLeast_3))
        self.assertTrue(value(AtLeast_4))

    def test_to_string(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        m.Y3 = BooleanVar()
        """
        remember to add more about this
        """

        # and_str = str()
        self.assertEqual(LogicalAnd(m.Y1, m.Y2, m.Y3), "Y1 AND Y3 AND Y2")

    def test_node_level(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        m.Y3 = BooleanVar()
        m.Y1.value, m.Y2.value, m.Y3.value = True, False, True

        self.assertFalse(m.Y1.is_expression_type())
        self.assertFalse(m.Y2.is_expression_type())
        self.assertFalse(m.Y3.is_expression_type())
        self.assertTrue(Not(m.Y1).is_expression_type())
        self.assertTrue(Equivalence(m.Y1, m.Y2).is_expression_type())
        self.assertTrue(AtMost(1, [m.Y1, m.Y2, m.Y3]).is_expression_type())

    def test_is_CNF(self):
        m = ConcreteModel()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        m.Y3 = BooleanVar()
        m.Y4 = BooleanVar()

        """
        A single literal test for if_CNF
        """

        m.Y1.value = True
        self.assertTrue(cnf.is_CNF(m.Y1))

        """
        A single literal with a NotExpression test for i_CNF
        """
        self.assertTrue(cnf.is_CNF(Not(m.Y1)))

        """
        A single non-nested OrExpression test for is_CNF
        """
        m.Y2.value = True
        self.assertTrue(cnf.is_CNF(m.Y1 | m.Y2))

        """
        A simple test for if_CNF with only and, or and not expression with no
        nested node. Expect True
        """
        m.Y3.value, m.Y4.value = True, False
        Or_node_1 = LogicalOr(m.Y1, m.Y2)
        Or_node_2 = LogicalOr(m.Y3, m.Y4)
        root_node_and = LogicalAnd(Or_node_1, Or_node_2)
        self.assertTrue(cnf.is_CNF(root_node_and))

        """
        A simple test for if_CNF with only and, or and not expression with no
        nested node. Expect False
        """

        m.Y1.value, m.Y2.value, m.Y3.value, m.Y4.value = True, True, True, False
        And_node = m.Y1 & m.Y2
        Not_node = ~m.Y3
        Leaf_node = m.Y4
        root_node_or = And_node | Not_node | Leaf_node
        self.assertFalse(cnf.is_CNF(root_node_or))
        # self.assertTrue(type(root_node_and) is MultiArgsExpression)

    # def test_reduce_not(self):
    '''
    def test_make_column(self):
        m = ConcreteModel()
        m.Y0 = BooleanVar()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        m.Y3 = BooleanVar()
        rn = LogicalOr(m.Y0,(m.Y2 and m.Y3), m.Y3)
        x = cnf.make_columns(rn._args_)
        print(list(x))
    '''

    def test_prepare_to_distribute(self):
        m = ConcreteModel()
        m.Y0 = BooleanVar()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        m.Y3 = BooleanVar()
        m.Y4 = BooleanVar()
        m.Y5 = BooleanVar()
        m.Y6 = BooleanVar()
        m.Y7 = BooleanVar()
        m.Y8 = BooleanVar()
        m.Y9 = BooleanVar()
        m.Y10 = BooleanVar()
        Y = list([m.Y0, m.Y1, m.Y2, m.Y3, m.Y4, m.Y5, m.Y6, m.Y7, m.Y8, m.Y9, m.Y10])
        rn = create_model2(Y)  # create a root node of model 2
        cnf.prepare_to_distribute(rn)
        """
        print(rn._args_[0])
        print(rn._args_[1])
        print(rn._args_[2])
        print(rn._args_[3]) 
        print(rn._args_[4])
        """
        # print(rn)
        # cnf.column

    def test_distribute(self):
        m = ConcreteModel()
        m.Y0 = BooleanVar()
        m.Y1 = BooleanVar()
        m.Y2 = BooleanVar()
        m.Y3 = BooleanVar()
        m.Y4 = BooleanVar()
        m.Y5 = BooleanVar()
        Y = list([m.Y0, m.Y1, m.Y2, m.Y3, m.Y4, m.Y5])
        rn = create_model3(Y)
        tup = cnf.make_columns(rn)
        LogicalOr(list(tup))
        # cnf.distribute_and_in_or

    def test_cnf_walker(self):
        m = ConcreteModel()
        m.Y = BooleanVar(RangeSet(0, 4))
        rn = create_model4(m.Y)
        print(rn)


if __name__ == "__main__":
    unittest.main()
