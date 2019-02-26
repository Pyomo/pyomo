import pyutilib.th as unittest
from itertools import product
from copy import deepcopy
from pyomo.contrib.logical_expression_system.nodes import \
    (NotNode, LeafNode, OrNode, AndNode, ImplicationNode, XOrNode,
     EquivalenceNode, isNotNode, isOrNode)
from pyomo.contrib.logical_expression_system.util import \
    (bring_to_conjunctive_normal_form,
     is_conjunctive_normal_form,
     is_leaf_not_node)


class TestLogicalExpressionSystem(unittest.TestCase):

    @staticmethod
    def generate_value_dicts(var_names):
        bool_combinations = product([False, True], repeat=len(var_names))
        value_dicts = [dict(zip(var_names, bool_vals))
                       for bool_vals in bool_combinations]
        return value_dicts

    @staticmethod
    def assert_model_equality(m1, m2, var_names):
        value_dicts = TestLogicalExpressionSystem.generate_value_dicts(var_names)
        for d in value_dicts:
            assert (m1.evaluate(d) == m2.evaluate(d))

    def test_simple_equality(self):
        y = dict((i, LeafNode('y' + str(i))) for i in range(1, 7))
        names = [n.child for n in y.values()]
        N1 = OrNode((y[1], y[2], y[3]))
        N2 = OrNode((y[4], y[5]))
        N3 = AndNode((N1, N2, y[6]))
        N3_compare = deepcopy(N3)

        self.assert_model_equality(N3, N3_compare, names)

    def test_atomic_nodes(self):
        l1 = LeafNode('y1')
        l2 = LeafNode('y2')
        self.assertTrue(
            all(l1.evaluate(lhs) is rhs for (lhs, rhs)
                in [({'y1': True}, True),
                    ({'y1': False}, False)]))

        not_node = NotNode(l1)
        self.assertTrue(all(not_node.evaluate(lhs) is rhs for (lhs, rhs)
                            in [({'y1': True}, False),
                                ({'y1': False}, True)]))

        and_node = AndNode([l1, l2])
        self.assertTrue(all(and_node.evaluate(lhs) is rhs for (lhs, rhs)
                            in [({'y1': False, 'y2': False}, False),
                                ({'y1': False, 'y2': True}, False),
                                ({'y1': True, 'y2': False}, False),
                                ({'y1': True, 'y2': True}, True)]))

        or_node = OrNode([l1, l2])
        self.assertTrue(all(or_node.evaluate(lhs) is rhs for (lhs, rhs)
                            in [({'y1': False, 'y2': False}, False),
                                ({'y1': False, 'y2': True}, True),
                                ({'y1': True, 'y2': False}, True),
                                ({'y1': True, 'y2': True}, True)]))

        xor_node = XOrNode([l1, l2])
        self.assertTrue(all(xor_node.evaluate(lhs) is rhs for (lhs, rhs)
                            in [({'y1': False, 'y2': False}, False),
                                ({'y1': False, 'y2': True}, True),
                                ({'y1': True, 'y2': False}, True),
                                ({'y1': True, 'y2': True}, False)]))

        equivalence_node = EquivalenceNode(l1, l2)
        self.assertTrue(all(equivalence_node.evaluate(lhs) is rhs for (lhs, rhs)
                            in [({'y1': False, 'y2': False}, True),
                                ({'y1': False, 'y2': True}, False),
                                ({'y1': True, 'y2': False}, False),
                                ({'y1': True, 'y2': True}, True)]))

        if_node = ImplicationNode(l1, l2)
        self.assertTrue(all(if_node.evaluate(lhs) is rhs for (lhs, rhs)
                            in [({'y1': False, 'y2': False}, True),
                                ({'y1': False, 'y2': True}, True),
                                ({'y1': True, 'y2': False}, False),
                                ({'y1': True, 'y2': True}, True)]))

    def test_become_other_node(self):
        l1 = LeafNode('y1')
        l2 = LeafNode('y2')
        l3 = LeafNode('y3')

        n1 = NotNode(l1)
        n1.becomeOtherNode(AndNode([l2, l3]))
        self.assertTrue(len(n1.children) == 2)
        self.assertTrue(sum([1 for l in n1.children if l.var() in ['y2', 'y3']]) == 2)

    def double_not_node_test(self):
        l1 = LeafNode('y1')
        n1 = NotNode(l1)
        n2 = NotNode(n1)
        n2.notNodeIntoOtherNode()
        self.assertTrue(isinstance(n2, LeafNode))
        self.assertTrue(n2.child == l1.child)

    def simple_not_and_test(self):
        l1 = LeafNode('y1')
        l2 = LeafNode('y2')
        n1 = AndNode([l1, l2])
        n2 = NotNode(n1)
        n2.notNodeIntoOtherNode()
        self.assertTrue(isinstance(n2, OrNode))
        self.assertTrue(all([isinstance(c, NotNode) for c in n2.children]))
        self.assertTrue(all([n.child.var() in ['y1', 'y2'] for n in n2.children]))

    def distributivity_equality_test(self):
        y = dict((i, LeafNode('y' + str(i))) for i in range(1, 7))
        names = [n.child for n in y.values()]
        N1 = OrNode((y[1], y[2], y[3]))
        N2 = OrNode((y[4], y[5]))
        N3 = AndNode((N1, N2, y[6]))
        N3_compare = deepcopy(N3)
        N3.distributivity_and_in_or()
        N3.distributivity_or_in_and()

        self.assert_model_equality(N3, N3_compare, names)

    def purge_single_child_test(self):
        l1 = LeafNode('y1')
        l2 = LeafNode('y2')
        n1 = AndNode([l1, l2])
        n2 = OrNode([n1])
        n2.tryPurgingWithSingleChild()
        self.assertTrue(isinstance(n2, AndNode))
        self.assertTrue(len(n2.children) == 2)
        self.assertTrue(all(isinstance(n, LeafNode) for n in n2.children))

    def purge_same_type_children_test(self):
        l1 = LeafNode('y1')
        l2 = LeafNode('y2')
        l3 = LeafNode('y3')
        n1 = AndNode([l1, l2])
        n2 = AndNode([n1, l3])
        n2.tryPurgingSameTypeChildren()
        self.assertTrue(all(isinstance(n, LeafNode) for n in n2.children))

    def or_in_and_test(self):
        y = dict([(i, 'y' + str(i)) for i in range(1, 6)])
        n = dict([(i, LeafNode(y[i])) for i in y.keys()])
        p1 = AndNode([n[4], n[5]])
        p2 = OrNode([n[3], p1])
        p3 = AndNode([n[2], p2])
        p4 = OrNode([n[1], p3])
        p4.distributivity_or_in_and()
        self.assertTrue(isinstance(p4, AndNode))
        for n in p4.children:
            self.assertTrue(isinstance(n, OrNode) or isinstance(n, NotNode)
                            or isinstance(n, LeafNode))
            for leaf_or_not_node in filter(isOrNode, n.children):
                self.assertTrue(isinstance(leaf_or_not_node, NotNode)
                                or isinstance(leaf_or_not_node, LeafNode))
                for l in filter(isNotNode, leaf_or_not_node):
                    self.assertTrue(isinstance(l, LeafNode))

    def simple_if_test(self):
        y = dict([(i, 'y' + str(i)) for i in range(1, 3)])
        l = dict([(i, LeafNode(y[i])) for i in y.keys()])
        n1 = ImplicationNode(l[1], l[2])
        n1_ref = deepcopy(n1)
        n1.ifToOr()
        self.assertTrue(isinstance(n1, OrNode))
        self.assertTrue(all(isinstance(n, NotNode) or isinstance(n, LeafNode)
                            for n in n1.children))
        self.assert_model_equality(n1, n1_ref, y.values())

    def is_leaf_not_node_test(self):
        y = dict([(i, 'y' + str(i)) for i in range(1, 5)])
        l = dict([(i, LeafNode(y[i])) for i in y.keys()])
        n1 = NotNode(l[1])
        n2 = AndNode([n1, l[2], l[3]])
        self.assertTrue(is_leaf_not_node(n2.children))
        n3 = OrNode([n2, l[4]])
        self.assertTrue(not is_leaf_not_node(n3.children))

    def is_conjunctive_normal_form_test(self):
        y = dict([(i, 'y' + str(i)) for i in range(1, 6)])
        l = dict([(i, LeafNode(y[i])) for i in y.keys()])
        n1 = OrNode([l[1], NotNode(l[2])])
        n2 = OrNode([NotNode(l[3])])
        n3 = OrNode([l[4], l[5]])
        n4 = AndNode([n1, n2, n3])
        self.assertTrue(is_conjunctive_normal_form(n4))

    def bring_if_to_conjunctive_normal_form_test(self):
        y = dict([(i, 'y' + str(i)) for i in range(1, 6)])
        l = dict([(i, LeafNode(y[i])) for i in y.keys()])
        n1 = AndNode([NotNode(l[1]), l[2]])
        n2 = NotNode(OrNode([l[3], l[4]]))
        n3 = ImplicationNode(n1, n2)
        n3_ref = deepcopy(n3)
        bring_to_conjunctive_normal_form(n3)
        self.assertTrue(is_conjunctive_normal_form(n3))
        self.assert_model_equality(n3, n3_ref, y.values())

    def bring_equivalence_to_conjunctive_normal_form_test(self):
        y = dict([(i, 'y' + str(i)) for i in range(1, 6)])
        l = dict([(i, LeafNode(y[i])) for i in y.keys()])
        n1 = AndNode([NotNode(l[1]), l[2]])
        n2 = NotNode(OrNode([l[3], l[4]]))
        n3 = EquivalenceNode(n1, n2)
        n3_ref = deepcopy(n3)
        bring_to_conjunctive_normal_form(n3)
        self.assertTrue(is_conjunctive_normal_form(n3))
        self.assert_model_equality(n3, n3_ref, y.values())

    def bring_leaf_to_conjunctive_normal_form_test(self):
        y1 = 'y1'
        l = LeafNode(y1)
        l_ref = deepcopy(l)
        bring_to_conjunctive_normal_form(l)
        self.assertTrue(is_conjunctive_normal_form(l))
        self.assert_model_equality(l, l_ref, [y1])

    def bring_and_to_conjunctive_normal_form_test(self):
        n = AndNode([LeafNode('y1'), LeafNode('y2')])
        n_ref = deepcopy(n)
        bring_to_conjunctive_normal_form(n)
        self.assertTrue(is_conjunctive_normal_form(n))
        self.assert_model_equality(n, n_ref, ['y1', 'y2'])

    def bring_not_to_conjunctive_normal_form_test(self):
        n = NotNode(LeafNode('y1'))
        n_ref = deepcopy(n)
        bring_to_conjunctive_normal_form(n)
        self.assertTrue(is_conjunctive_normal_form(n))
        self.assert_model_equality(n, n_ref, ['y1'])
