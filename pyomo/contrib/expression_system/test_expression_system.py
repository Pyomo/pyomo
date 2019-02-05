from itertools import product
from copy import deepcopy
from expression_system import *
from pyomo.environ import (ConcreteModel, Var, Constraint)

def generate_value_dicts(var_names):
    bool_combinations = product([False,True], repeat=len(var_names))
    value_dicts = [dict(zip(var_names, bool_vals)) for bool_vals in bool_combinations]
    return value_dicts

def assert_model_equality(m1, m2, var_names):
    value_dicts = generate_value_dicts(var_names)
    for d in value_dicts:
        assert(m1.evaluate(d) == m2.evaluate(d))

def simple_equality_test():
    y = dict((i,LeafNode('y'+str(i))) for i in range(1,7))
    names = [n.child for n in y.values()]
    N1 = OrNode((y[1], y[2], y[3]))
    N2 = OrNode((y[4], y[5]))
    N3 = AndNode((N1,N2,y[6]))
    N3_compare = deepcopy(N3)
    
    assert_model_equality(N3, N3_compare, names)


def distributivity_test():
    y = dict((i,LeafNode('y'+str(i))) for i in range(1,7))
    names = [n.child for n in y.values()]
    N1 = OrNode((y[1], y[2], y[3]))
    N2 = OrNode((y[4], y[5]))
    N3 = AndNode((N1,N2,y[6]))
    N3_compare = deepcopy(N3)
    N3.distributivity_and_in_or()
    N3.distributivity_or_in_and()

    assert_model_equality(N3, N3_compare, names)

def or_in_and_test():
    y = dict([(i,'y'+str(i)) for i in range(1,6)])
    n = dict([(i,LeafNode(y[i])) for i in y.keys()])
    p1 = AndNode([n[4],n[5]])
    p2 = OrNode([n[3],p1])
    p3 = AndNode([n[2],p2])
    p4 = OrNode([n[1],p3])
    p4_ref = deepcopy(p4)
    p4.distributivity_or_in_and()
    assert(isinstance(p4, AndNode))
    for n in p4.children:
        assert(isinstance(n, OrNode) or isinstance(n, NotNode) or isinstance(n, LeafNode))
        for leaf_or_not_node in filter(isOrNode, n.children):
            assert(isinstance(leaf_or_not_node, NotNode) or isinstance(leaf_or_not_node, LeafNode))
            for l in filter(isNotNode, leaf_or_not_node):
                assert(isinstance(l, LeafNode))
