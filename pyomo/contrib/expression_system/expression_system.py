# coding: utf-8
from pyomo.environ import (ConcreteModel, Var, Constraint)
from pyomo.core.base.component import _ComponentBase

class Node:
    def __init__(self):
        raise RuntimeError("Tried to initialize abstract class 'Node'")

    def evaluate(self, value_dict):
        raise NotImplementedError()

    def print(self):
        raise NotImplementedError()

    def ifToOr(self, recursive=False):
        if recursive:
            self.child_l.ifToOr(recursive=True)
            self.child_r.ifToOr(recursive=True)

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
        
class BinaryNode(Node):
    def __init__(self):
        self.commutative = False
        self.child_l = None
        self.child_r = None
    def notNodeIntoOtherNode(self, recursive=False):
        for n in [self.child_l, self.child_r]:
            n.notNodeIntoOtherNode(recursive=recursive)

class IfNode(BinaryNode):
    def __init__(self, child_l, child_r):
        self.child_l = child_l
        self.child_r = child_r
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
        self.__class__ = OrNode
        self.__init__([NotNode(child_l), child_r])
        
class UnaryNode(Node):
    def __init__(self):
        self.commutative=False
        self.child = None
    def notNodeIntoOtherNode(self, recursive=False):
        self.child.notNodeIntoOtherNode(recursive=recursive)

class LeafNode(UnaryNode):
    def __init__(self, single_node):
        self.child = single_node
    def print(self):
        if isinstance(self.child, str):
            return self.child
        elif isinstance(self.child, _ComponentBase):
            return self.child.name
    def evaluate(self,value_dict):
        return value_dict[self.child]
    def ifToOr(self, recursive=False):
        pass
    def var(self):
        return self.child
    def notNodeIntoOtherNode(self, recursive=False):
        pass

class NotNode(UnaryNode):
    def __init__(self, single_node):
        self.child = single_node
    def print(self):
        output = self.child.print()
        return '!'+output
    def evaluate(self,value_dict):
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
            elif isUnaryNode(self):
                self.child.notNodeIntoOtherNode(recursive=True)

class MultiNode(Node):
    def __init__(self):
        self.commutative = True
        self.children = []
    def tryPurgingWithSingleChild(self, recursive=False):
        if recursive:
            for n in filter(isMultiNode, self.children):
                n.tryPurgingWithSingleChild(recursive=True)

        if len(self.children) == 1:
            child = self.children.pop()
            self.__class__ = type(child)
            self.__init__(child.children)
            print('purged, now Im ' + str(type(child)))

    def tryPurgingSameTypeChildren(self, recursive=False):
        if recursive:
            for n in filter(isMultiNode, self.children):
                n.tryPurgingSameTypeChildren(recursive=True)

        isSameType = lambda n: isinstance(n, type(self))
        for n in filter(isSameType, self.children.copy()):
            self.children.update(n.children)
            self.children.remove(n)
            print('combined children in class' + str(type(self)))
        #if all([isinstance(n, type(self)) for n in self.children]):
        #    self.children = set.union(*[c.children for c in self.children])
    def notNodeIntoOtherNode(self, recursive=False):
        for n in self.children:
            n.notNodeIntoOtherNode(recursive=recursive)
        self.tryPurgingSameTypeChildren()

        
class AndNode(MultiNode):
    def __init__(self, var_list):
        self.children = set([v for v in var_list])

    def print(self):
        output = " ^ ".join([c.print() for c in self.children])
        return '{'+output+'}'

    def evaluate(self, value_dict):
        return all(n.evaluate(value_dict) for n in self.children)

    def distributivity_or_in_and(self):
        #self.tryPurgingWithSingleChild()
        #self.tryPurgingSameTypeChildren()
        for n in filter(isMultiNode, self.children):
            n.distributivity_or_in_and()
        self.tryPurgingWithSingleChild()
        self.tryPurgingSameTypeChildren()
        self.tryPurgingWithSingleChild()

    def distributivity_and_in_or(self):
        while any([isinstance(n, OrNode) for n in self.children]) and len(self.children) > 1:
            or_node = next(n for n in self.children if isinstance(n, OrNode))
            other_nodes = set([n for n in self.children if not n is or_node])
            new_or_node = OrNode([AndNode(set([or_el]) | other_nodes) for or_el in or_node.children])
            self.children -= set([or_node]) | other_nodes
            self.children |= set([new_or_node])
        #self.tryPurgingWithSingleChild()
        self.tryPurgingSameTypeChildren()
        for n in filter(isAndNode, self.children):
            n.distributivity_and_in_or()
        #self.tryPurgingWithSingleChild()
        self.tryPurgingSameTypeChildren()
        
class OrNode(MultiNode):
    def __init__(self,var_list):
        self.children = set([v for v in var_list])

    def print(self):
        output = " v ".join([c.print() for c in self.children])
        return '('+output+')'

    def evaluate(self, value_dict):
        return any([n.evaluate(value_dict) for n in self.children])

    def distributivity_or_in_and(self):
        for n in filter(isMultiNode, self.children):
            n.distributivity_or_in_and()
        #self.tryPurgingWithSingleChild()
        self.tryPurgingSameTypeChildren()
        while any([isinstance(n, AndNode) for n in self.children]) and len(self.children) > 1:
            and_node = next(n for n in self.children if isinstance(n, AndNode))
            other_nodes = set([n for n in self.children if not n is and_node])
            new_and_node = AndNode([OrNode(set([and_el]) | other_nodes) for and_el in and_node.children])
            self.children -= set([and_node]) | other_nodes
            self.children |= set([new_and_node])
        self.tryPurgingWithSingleChild()
        for n in filter(isMultiNode, self.children):
            n.tryPurgingSameTypeChildren()
        #self.tryPurgingSameTypeChildren()

        #self.tryPurgingWithSingleChild()

    def distributivity_and_in_or(self):
        #self.tryPurgingWithSingleChild()
        self.tryPurgingSameTypeChildren()
        for n in filter(isAndNode, self.children):
            n.distributivity_and_in_or()
        #self.tryPurgingWithSingleChild()
        self.tryPurgingSameTypeChildren()

isOrNode = lambda n: isinstance(n, OrNode)
#isNotOrNode = lambda n: not isinstance(n, OrNode)
isAndNode = lambda n: isinstance(n, AndNode)
isMultiNode = lambda n: isinstance(n, MultiNode)
isUnaryNode = lambda n: isinstance(n, UnaryNode)
isBinaryNode = lambda n: isinstance(n, BinaryNode)
isNotNode = lambda n: isinstance(n, NotNode)
#isNotAndNode = lambda n: not isinstance(n, AndNode)
isNode = lambda n: isinstance(n, Node)
isLeafNode = lambda n: isinstance(n, LeafNode)
