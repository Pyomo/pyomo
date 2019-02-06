from itertools import product
from copy import deepcopy
from expression_system import \
    (NotNode, LeafNode, OrNode, AndNode, IfNode,
     bring_to_conjunctive_normal_form, is_conjunctive_normal_form,
     is_leaf_not_node, isNotNode, isOrNode)


def generate_value_dicts(var_names):
    bool_combinations = product([False, True], repeat=len(var_names))
    value_dicts = [dict(zip(var_names, bool_vals))
                   for bool_vals in bool_combinations]
    return value_dicts


def assert_model_equality(m1, m2, var_names):
    value_dicts = generate_value_dicts(var_names)
    for d in value_dicts:
        assert(m1.evaluate(d) == m2.evaluate(d))


def simple_equality_test():
    y = dict((i, LeafNode('y'+str(i))) for i in range(1, 7))
    names = [n.child for n in y.values()]
    N1 = OrNode((y[1], y[2], y[3]))
    N2 = OrNode((y[4], y[5]))
    N3 = AndNode((N1, N2, y[6]))
    N3_compare = deepcopy(N3)

    assert_model_equality(N3, N3_compare, names)


def become_other_node_test():
    l1 = LeafNode('y1')
    l2 = LeafNode('y2')
    l3 = LeafNode('y3')

    n1 = NotNode(l1)
    n1.becomeOtherNode(AndNode([l2, l3]))
    assert(len(n1.children) == 2)
    assert(sum([1 for l in n1.children if l.var() in ['y2', 'y3']]) == 2)


def double_not_node_test():
    l1 = LeafNode('y1')
    n1 = NotNode(l1)
    n2 = NotNode(n1)
    n2.notNodeIntoOtherNode()
    assert(isinstance(n2, LeafNode))
    assert(n2.child == l1.child)


def simple_not_and_test():
    l1 = LeafNode('y1')
    l2 = LeafNode('y2')
    n1 = AndNode([l1, l2])
    n2 = NotNode(n1)
    n2.notNodeIntoOtherNode()
    assert(isinstance(n2, OrNode))
    assert(all([isinstance(c, NotNode) for c in n2.children]))
    assert(all([n.child.var() in ['y1', 'y2'] for n in n2.children]))


def distributivity_equality_test():
    y = dict((i, LeafNode('y'+str(i))) for i in range(1, 7))
    names = [n.child for n in y.values()]
    N1 = OrNode((y[1], y[2], y[3]))
    N2 = OrNode((y[4], y[5]))
    N3 = AndNode((N1, N2, y[6]))
    N3_compare = deepcopy(N3)
    N3.distributivity_and_in_or()
    N3.distributivity_or_in_and()

    assert_model_equality(N3, N3_compare, names)


def purge_single_child_test():
    l1 = LeafNode('y1')
    l2 = LeafNode('y2')
    n1 = AndNode([l1, l2])
    n2 = OrNode([n1])
    n2.tryPurgingWithSingleChild()
    assert(isinstance(n2, AndNode))
    assert(len(n2.children) == 2)
    assert(all(isinstance(n, LeafNode) for n in n2.children))


def purge_same_type_children_test():
    l1 = LeafNode('y1')
    l2 = LeafNode('y2')
    l3 = LeafNode('y3')
    n1 = AndNode([l1, l2])
    n2 = AndNode([n1, l3])
    n2.tryPurgingSameTypeChildren()
    assert(all(isinstance(n, LeafNode) for n in n2.children))


def or_in_and_test():
    y = dict([(i, 'y'+str(i)) for i in range(1, 6)])
    n = dict([(i, LeafNode(y[i])) for i in y.keys()])
    p1 = AndNode([n[4], n[5]])
    p2 = OrNode([n[3], p1])
    p3 = AndNode([n[2], p2])
    p4 = OrNode([n[1], p3])
    p4.distributivity_or_in_and()
    assert(isinstance(p4, AndNode))
    for n in p4.children:
        assert(isinstance(n, OrNode) or isinstance(n, NotNode)
               or isinstance(n, LeafNode))
        for leaf_or_not_node in filter(isOrNode, n.children):
            assert(isinstance(leaf_or_not_node, NotNode)
                   or isinstance(leaf_or_not_node, LeafNode))
            for l in filter(isNotNode, leaf_or_not_node):
                assert(isinstance(l, LeafNode))


def simple_if_test():
    y = dict([(i, 'y'+str(i)) for i in range(1, 3)])
    l = dict([(i, LeafNode(y[i])) for i in y.keys()])
    n1 = IfNode(l[1], l[2])
    n1_ref = deepcopy(n1)
    n1.ifToOr()
    assert(isinstance(n1, OrNode))
    assert(all(isinstance(n, NotNode) or isinstance(n, LeafNode)
               for n in n1.children))
    assert_model_equality(n1, n1_ref, y.values())


def is_leaf_not_node_test():
    y = dict([(i, 'y'+str(i)) for i in range(1, 5)])
    l = dict([(i, LeafNode(y[i])) for i in y.keys()])
    n1 = NotNode(l[1])
    n2 = AndNode([n1, l[2], l[3]])
    assert(is_leaf_not_node(n2.children))
    n3 = OrNode([n2, l[4]])
    assert(not is_leaf_not_node(n3.children))


def is_conjunctive_normal_form_test():
    y = dict([(i, 'y'+str(i)) for i in range(1, 6)])
    l = dict([(i, LeafNode(y[i])) for i in y.keys()])
    n1 = OrNode([l[1], NotNode(l[2])])
    n2 = NotNode(l[3])
    n3 = AndNode([n1, n2, l[4], l[5]])
    assert(is_conjunctive_normal_form(n3))


def bring_to_conjunctive_normal_form_test():
    y = dict([(i, 'y'+str(i)) for i in range(1, 6)])
    l = dict([(i, LeafNode(y[i])) for i in y.keys()])
    n1 = AndNode([NotNode(l[1]), l[2]])
    n2 = NotNode(OrNode([l[3], l[4]]))
    n3 = IfNode(n1, n2)
    n3_ref = deepcopy(n3)
    bring_to_conjunctive_normal_form(n3)
    assert(is_conjunctive_normal_form(n3))
    assert_model_equality(n3, n3_ref, y.values())
