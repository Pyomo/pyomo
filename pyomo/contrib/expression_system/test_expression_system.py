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
