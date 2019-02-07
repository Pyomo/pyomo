from pyomo.core.base.component import _ComponentBase


class Node:
    def __init__(self):
        raise RuntimeError("Tried to initialize abstract class 'Node'")

    def evaluate(self, value_dict):
        raise NotImplementedError()

    def print(self):
        raise NotImplementedError()

    def becomeOtherNode(self, target_node):
        if isMultiNode(target_node):
            children = target_node.children
            self.__class__ = type(target_node)
            self.__init__(children)
        elif isUnaryNode(target_node):
            child = target_node.child
            self.__class__ = type(target_node)
            self.__init__(child)
        elif isBinaryNode(target_node):
            child_l, child_r = target_node.child_l, target_node.child_r
            self.__class__ = type(target_node)
            self.__init__(child_l, child_r)

    def ifToOr(self, recursive=False):
        if recursive:
            for n in self._children_as_list():
                n.ifToOr(recursive=True)

    def equivalentToAnd(self, recursive=False):
        if recursive:
            for n in self._children_as_list():
                n.equivalentToAnd(recursive=True)

    def xorToOr(self, recursive=False):
        if recursive:
            for n in self._children_as_list():
                n.xorToOr(recursive=True)

    def notNodeIntoOtherNode(self, recursive=False):
        if recursive:
            for n in self._children_as_list():
                n.notNodeIntoOtherNode(recursive=recursive)

    def tryPurgingWithSingleChild(self, recursive=False):
        if recursive:
            for n in self._children_as_list():
                n.tryPurgingWithSingleChild(recursive=True)

    def tryPurgingSameTypeChildren(self, recursive=False):
        if recursive:
            for n in self._children_as_list():
                n.tryPurgingSameTypeChildren(recursive=True)

    def distributivity_and_in_or(self, recursive=False):
        if recursive:
            for n in self._children_as_list():
                n.distributivity_and_in_or(recursive=True)

    def distributivity_or_in_and(self, recursive=False):
        if recursive:
            for n in self._children_as_list():
                n.distributivity_or_in_and(recursive=True)


class BinaryNode(Node):
    def __init__(self, child_l, child_r):
        self.child_l = child_l
        self.child_r = child_r

    def _children_as_list(self):
        return [self.child_l, self.child_r]


class IfNode(BinaryNode):
    def print(self):
        lhs = self.child_l.print()
        rhs = self.child_r.print()
        return lhs + ' => ' + rhs

    def evaluate(self, value_dict):
        lhs = self.child_l.evaluate(value_dict)
        rhs = self.child_r.evaluate(value_dict)
        return not lhs or rhs

    def ifToOr(self, recursive=False):
        if recursive:
            self.child_l.ifToOr(recursive=True)
            self.child_r.ifToOr(recursive=True)

        child_l = self.child_l
        child_r = self.child_r
        self.becomeOtherNode(OrNode([NotNode(child_l), child_r]))


class EquivalenceNode(BinaryNode):
    def print(self):
        lhs = self.child_l.print()
        rhs = self.child_r.print()
        return lhs + ' <=> ' + rhs

    def evaluate(self, value_dict):
        lhs = self.child_l.evaluate(value_dict)
        rhs = self.child_r.evaluate(value_dict)
        return lhs == rhs

    def equivalentToAnd(self, recursive=False):
        if recursive:
            self.child_l.equivalentToAnd(recursive=True)
            self.child_r.equivalentToAnd(recursive=True)
        child_l = self.child_l
        child_r = self.child_r
        self.becomeOtherNode(
            AndNode([IfNode(child_l, child_r), IfNode(child_r, child_l)]))


class UnaryNode(Node):
    def __init__(self):
        self.child = None

    def _children_as_list(self):
        return [self.child]


class LeafNode(UnaryNode):
    def __init__(self, single_node):
        self.child = single_node

    def print(self):
        if isinstance(self.child, str):
            return self.child
        elif isinstance(self.child, _ComponentBase):
            return self.child.name

    def evaluate(self, value_dict):
        return value_dict[self.child]

    def ifToOr(self, recursive=False):
        pass

    def equivalentToAnd(self, recursive=False):
        pass

    def distributivity_and_in_or(self, recursive=False):
        pass

    def distributivity_or_in_and(self, recursive=False):
        pass

    def var(self):
        return self.child

    def notNodeIntoOtherNode(self, recursive=False):
        pass

    def tryPurgingWithSingleChild(self, recursive=False):
        pass

    def tryPurgingSameTypeChildren(self, recursive=False):
        pass


class NotNode(UnaryNode):
    def __init__(self, single_node):
        self.child = single_node

    def print(self):
        output = self.child.print()
        return '!'+output

    def evaluate(self, value_dict):
        return not self.child.evaluate(value_dict)

    def notNodeIntoOtherNode(self, recursive=False):
        if isMultiNode(self.child):
            if isAndNode(self.child):
                new_child = OrNode([])
            elif isOrNode(self.child):
                new_child = AndNode([])
            new_child.children = set([NotNode(n) for n in self.child.children])
            self.becomeOtherNode(new_child)
        elif isNotNode(self.child):
            self.becomeOtherNode(self.child.child)

        if recursive:
            if isMultiNode(self):
                for n in self.children:
                    n.notNodeIntoOtherNode(recursive=True)
                self.tryPurgingSameTypeChildren()
            elif isBinaryNode(self):
                for n in [self.child_l, self.child_r]:
                    n.notNodeIntoOtherNode(recursive=True)
            elif isLeafNode(self):
                pass
            elif isUnaryNode(self):
                self.child.notNodeIntoOtherNode(recursive=True)


class MultiNode(Node):
    def __init__(self, var_list):
        self.children = set([v for v in var_list])

    def _children_as_list(self):
        return self.children

    def tryPurgingWithSingleChild(self, recursive=False):
        if recursive:
            for n in filter(isMultiNode, self.children):
                n.tryPurgingWithSingleChild(recursive=True)

        if len(self.children) == 1:
            child = self.children.pop()
            self.__class__ = type(child)
            self.__init__(child.children)

    def tryPurgingSameTypeChildren(self, recursive=False):
        if recursive:
            for n in filter(isMultiNode, self.children):
                n.tryPurgingSameTypeChildren(recursive=True)

        def isSameType(n): return isinstance(n, type(self))
        for n in filter(isSameType, self.children.copy()):
            self.children.update(n.children)
            self.children.remove(n)

    def notNodeIntoOtherNode(self, recursive=False):
        for n in self.children:
            n.notNodeIntoOtherNode(recursive=recursive)
        self.tryPurgingSameTypeChildren()


class AndNode(MultiNode):
    def print(self):
        output = " ^ ".join([c.print() for c in self.children])
        return '{'+output+'}'

    def evaluate(self, value_dict):
        return all(n.evaluate(value_dict) for n in self.children)

    def distributivity_or_in_and(self, recursive=True):
        for n in filter(isMultiNode, self.children):
            n.distributivity_or_in_and(recursive=recursive)
        self.tryPurgingWithSingleChild()
        self.tryPurgingSameTypeChildren()
        self.tryPurgingWithSingleChild()

    def distributivity_and_in_or(self, recursive=True):
        while any(isinstance(n, OrNode) for n in self.children) \
                and len(self.children) > 1:
            or_node = next(n for n in self.children if isinstance(n, OrNode))
            other_nodes = set([n for n in self.children if n is not or_node])
            new_or_node = OrNode(
                AndNode(set([or_el]) | other_nodes)
                for or_el in or_node.children)
            self.children -= set([or_node]) | other_nodes
            self.children |= set([new_or_node])
        self.tryPurgingSameTypeChildren()
        for n in filter(isAndNode, self.children):
            n.distributivity_and_in_or(recursive=recursive)
        self.tryPurgingSameTypeChildren()


class XOrNode(MultiNode):
    def print(self):
        output = " x ".join([c.print() for c in self.children])
        return '['+output+']'

    def evaluate(self, value_dict):
        return sum(n.evaluate(value_dict) for n in self.children) == 1

    def xorToOr(self, recursive=False):
        if recursive:
            for n in self.children:
                n.xorToOr(recursive=True)

        new_or_node = OrNode(
            [AndNode([selected_node, *[NotNode(other_nodes)
                      for other_nodes in self.children.difference(n)]])
             for selected_node in self.children])
        self.becomeOtherNode(OrNode(new_or_node.children))


class OrNode(MultiNode):
    def print(self):
        output = " v ".join([c.print() for c in self.children])
        return '('+output+')'

    def evaluate(self, value_dict):
        return any([n.evaluate(value_dict) for n in self.children])

    def distributivity_or_in_and(self, recursive=True):
        for n in filter(isMultiNode, self.children):
            n.distributivity_or_in_and(recursive=recursive)
        self.tryPurgingSameTypeChildren()
        while any(isinstance(n, AndNode) for n in self.children) \
                and len(self.children) > 1:
            and_node = next(n for n in self.children if isinstance(n, AndNode))
            other_nodes = set([n for n in self.children if n is not and_node])
            new_and_node = AndNode([OrNode(set([and_el]) | other_nodes)
                                    for and_el in and_node.children])
            self.children -= set([and_node]) | other_nodes
            self.children |= set([new_and_node])
        self.tryPurgingWithSingleChild()
        for n in filter(isMultiNode, self.children):
            n.tryPurgingSameTypeChildren()

    def distributivity_and_in_or(self, recursive=True):
        self.tryPurgingSameTypeChildren()
        for n in filter(isAndNode, self.children):
            n.distributivity_and_in_or(recursive=recursive)
        self.tryPurgingSameTypeChildren()


def isNode(n): return isinstance(n, Node)
def isUnaryNode(n): return isinstance(n, UnaryNode)
def isNotNode(n): return isinstance(n, NotNode)
def isLeafNode(n): return isinstance(n, LeafNode)
def isBinaryNode(n): return isinstance(n, BinaryNode)
def isIfNode(n): return isinstance(n, IfNode)
def isMultiNode(n): return isinstance(n, MultiNode)
def isAndNode(n): return isinstance(n, AndNode)
def isOrNode(n): return isinstance(n, OrNode)
