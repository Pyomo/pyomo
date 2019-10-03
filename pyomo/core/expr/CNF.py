from .logical_expr import (LogicalExpressionBase, NotExpression, AndExpression, 
    OrExpression, Implication, EquivalenceExpression, XorExpression, 
    ExactlyExpression, AtMostExpression, AtLeastExpression, Not, Equivalence, 
    LogicalOr, Implies, LogicalAnd, Exactly, AtMost, AtLeast, LogicalXor
    )

import itertools as it


def is_elementary_operation(node):
    if (type(node) is ExactlyExpression) and (not is_nested(node)):
        return True
    if (type(node) is AtMostExpression) and (not is_nested(node)):
        return True
    if (type(node) is AtLeastExpression) and (not is_nested(node)):
        return True
    return False 

def is_literal(node):
    if not node.is_expression_type():
        return True
    if (type(node) is NotExpression) and (not (node._args_[0]).is_expression_type()):
        return True
    if is_elementary_operation(node):
        return True
    return False 

def is_nested(node):
    for i in range(len(node._args_)):
        if not is_literal(node._args_[i]):
            return True
    return False

def is_CNF_child(node):
    if is_literal(node):
        return True
    if (type(node) is OrExpression) and (not is_nested(node)):
        return True
    return False

def is_CNF_root(node):
    if type(node) is AndExpression:
        return True
    if (type(node) is OrExpression) and (not is_nested(node)):
        return True
    return False


def is_CNF(node):
    if is_literal(node):
        return True
    #The node is not a leaf node it gets here
    if not is_CNF_root(node):
        return False
    for i in range(len(node._args_)):
        if not is_CNF_child(node._args_[i]):
            return False
    return True

def is_binary_expression(node):
    if type(node) is XorExpression:
        return True
    if type(node) is EquivalenceExpression:
        return True
    if type(node) is Implication:
        return True
    return False

def Binary2or(node, parent = None, index = -1):
    if type(node) is XorExpression:
        larg = node._args_[0] & Not(node._args_[1])
        rarg = node._args_[1] & Not(node._args_[0])
        new_node = larg | rarg
    elif type(node) is EquivalenceExpression:
        larg = node._args_[0] & node._args_[1]
        rarg = Not(node._args_[1]) & Not(node._args_[0])
        new_node = larg | rarg
    elif type(node) is Implication:
        new_node = Not(node._args_[0]) | node._args_[1]
    if parent is not None:
        assert node is parent[index]
        parent[index] = new_node
    return 

"""
def bring_to_CNF(node):
    #checking changes is needed 
    if is_CNF(node):
        return node
    #distribute and reduce binary expressions
    if is_binary_expression(node):
        node = Binary2or(node)

    #reduce not

if is_binary_expression(node._args_[i]):
                Binary2or(node._args_[i])
            if (type(node._args_[i]) is NotExpression) and (is_nested(node._args_[i])):
                reduce_NotExpession(node._args_[i])
"""    


def reduce_not(node, parent = None, index = -1):
    if is_literal(node):
        return
    if type(node._args_[0]) is AndExpression:
        new_node = Not((node)._args_[0]._args_[0]) 
        if len(node._args_[0]._args_) > 1:
            for i in range(len(node._args_[0]) - 1):
                new_node = new_node or Not(node._args_[0]._args_[i+1])
        assert node is parent[index]
        parent[index] = new_node
        return

    if is_binary_expression(node._args_[0]):
        Binary2or(node._args_[0], node._args_, 0) 

    if type(node._args_[0]) is OrExpression:
        new_node = Not((node)._args_[0]._args_[0]) 
        if len(node._args_[0]._args_) > 1:
            for i in range(len(node._args_[0]) - 1):
                new_node = new_node and Not(node._args_[0]._args_[i+1])
        parent[index] = new_node
        return
"""
def merge_same_type_nodes(node):
    length  = len(node._args_)
    i = 0
    if type(node) is AndExpression:
        while (i < length):
            if type(node._args_[i]) is AndExpression:
                node._add(node._args_[i])
                node._args_.pop(i)
                length -= 1
            else:
                i += 1
    elif type(node) is OrExpression:
        while (i < length):
            if type(node._args_[i]) is OrExpression:
                node._add(node._args_[i])
                node._args_.pop(i)
                length -= 1  
    return          
"""               

def prepare_to_distribute(node):
    """
    This function does the following things:
        1. iterate through all the children
        2. convert binary child to OrExpression
        3. reduce nested NotExpression
    """
    i = 0
    l = len(node._args_)
    if type(node) is AndExpression:
        while i < l:
            if is_binary_expression(node._args_[i]):
                Binary2or(node._args_[i], node._args_, i) 
            if type(node._args_[i]) is NotExpression:
                reduce_not(node._args_[i], node._args_, i)
            if type(node._args_[i]) is AndExpression:
                tmp = node._args_.pop(i)
                for j in range(len(tmp._args_)):
                    node._add(tmp._args_[j])
                    l = l + 1
                l = l - 1 
                i = i - 1 
            if is_literal(node._args_[i]):
                nodes._args_[i] = OrExpression(nodes._args_[i])
            assert type(node._args_[i]) is OrExpression #delete later
            i = i + 1

    elif type(node) is OrExpression:
        while i < l:
            if is_binary_expression(node._args_[i]):
                Binary2or(node._args_[i], node._args_, i)
            if type(node._args_[i]) is NotExpression:
                reduce_not(node._args_[i], node._args_, i)
            if type(node._args_[i]) is OrExpression:
                tmp = node._args_.pop(i)
                for j in range(len(tmp._args_)):
                    node._add(tmp._args_[j])
                    l = l + 1
                l = l - 1 
                i = i - 1 
            if is_literal(node._args_[i]):
                node._args_[i] = LogicalAnd(node._args_[i])
            if(type(node._args_[i]) is not AndExpression):
                assert 1 == 0
                #delete later
            i = i + 1
    return 


def make_columns(arr):
    #make columns from prepared And/Or node
    return tuple(arr)

def distribute_and_in_or(node): 
    prepare_to_distribute(node)
    tups = make_columns(node._args_)
    res_list = list(it.product(tups))



