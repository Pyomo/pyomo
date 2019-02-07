from pyomo.core import (Constraint, RangeSet)
from pyomo.contrib.logical_expression_system.nodes import \
    (AndNode, OrNode, LeafNode, NotNode,
     isNode, isNotNode, isLeafNode, isOrNode, isAndNode)


def bring_to_conjunctive_normal_form(root_node):
    if is_conjunctive_normal_form(root_node):
        return
    else:
        root_node.tryPurgingWithSingleChild(recursive=True)
        root_node.tryPurgingSameTypeChildren(recursive=True)
        root_node.equivalentToAnd(recursive=True)
        root_node.ifToOr(recursive=True)
        root_node.xorToOr(recursive=True)
        root_node.notNodeIntoOtherNode(recursive=True)
        root_node.distributivity_or_in_and(recursive=True)
        if isinstance(root_node, OrNode):
            target_node = root_node
            root_node.becomeOtherNode(AndNode([OrNode(target_node.children)]))
        elif isLeafNode(root_node):
            leaf_node = root_node
            root_node.becomeOtherNode(
                AndNode([OrNode([LeafNode(leaf_node.child)])]))
        elif isNotNode(root_node):
            not_node = root_node
            root_node.becomeOtherNode(
                AndNode([OrNode([NotNode(not_node.child)])]))
        elif any(not isOrNode(n) for n in root_node.children):
            children = set(n for n in root_node.children if not isOrNode(n))
            for n in children:
                root_node.children.remove(n)
                root_node.children.add(OrNode([n]))


def is_leaf_not_node(nodes):
    if isNotNode(nodes) or isLeafNode(nodes):
        return True
    elif isNode(nodes):
        return False
    leaf_or_not_node = all(isLeafNode(n) or isNotNode(n) for n in nodes)
    if not leaf_or_not_node:
        return False
    not_children_are_leaf = all(
        isLeafNode(n.child) for n in filter(isNotNode, nodes))
    return leaf_or_not_node and not_children_are_leaf


def is_conjunctive_normal_form(root_node):
    root_is_and_node = isAndNode(root_node)
    if not root_is_and_node:
        return False

    and_children_are_or_nodes = all(isOrNode(n) for n in root_node.children)
    if not and_children_are_or_nodes:
        return False

    or_children_are_leaf_not_nodes = all(
        is_leaf_not_node(n.children)
        for n in filter(isOrNode, root_node.children))

    return root_is_and_node and and_children_are_or_nodes \
        and or_children_are_leaf_not_nodes


def CNF_to_linear_constraints(model, node_in_CNF):
    assert(is_conjunctive_normal_form(node_in_CNF))

    model.logical_constr_idx = RangeSet(len(node_in_CNF.children))
    model.logical_constr = Constraint(model.logical_constr_idx)
    for (i, n_or) in zip(model.logical_constr_idx, node_in_CNF.children):
        model.logical_constr[i] = sum(
            n.var() if isinstance(n, LeafNode) else (1 - n.child.var())
            for n in n_or.children) >= 1
