import CNF as cnf 
from pyomo.core.expr.logical_expr import (LogicalExpressionBase, UnaryExpression,
    NotExpression, BinaryExpression, MultiArgsExpression,
    AndExpression, OrExpression, Implication, EquivalenceExpression, XorExpression, 
    ExactlyExpression, AtMostExpression, AtLeastExpression, Not, Equivalence, 
    LogicalOr, Implies, LogicalAnd, Exactly, AtMost, AtLeast, LogicalXor, 
    )


# def conversion(N1):
# 	reduce_not(N1)
# 	return
#
# def beforechild(self, N1, N2):
# 	#non-descend case
# 	if N2 is literal:
# 		descend = False
# 		child_result N2
# 	#descend case
# 	else:
# 		descend = True
# 		child_result = None
# 	return descned, child_result
#
#
# def distribute(node):
#     mark = cnf.prepare_to_distribute(node)
#     tups = list(make_columns(node._args_))
#    	if mark:
#    		node = LogicalOr(list(tups))
#    	else:
#    		node = LogicalAnd(list(tups))
#    	return node
#
#
# def exitNode(self, N1, data = None):
# 	if cnf.is_literal(N1):
# 		return N1
# 	return distribute(N1)
#
#
# def node2dat(N1):
# 	if is_literal(N1):
# 		return None
# 	return N1._args_[0:-1] = node2dat(N1)
#
#
# def enterNode(self, N1):
# 	conversion(N1)	#convert every node into and/or/literal
# 	N1._args_[0:-1] = node2dat(N1)
# 	for N2 in N1 N1._args_[0:-1]:
# 		des,  cr = beforeChild(N1, N2)
# 		if des:
# 			enterNode(N2)
# 			exitNode(N2, node2dat(N2))
# 		acceptChildResult(N1, data, cr)
# 		afterChild(N1, N2)
# 	return exitNode(N1)
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor


class ToCNFWalker(StreamBasedExpressionVisitor):
	"""This expression walker converts a logical proposition into conjunctive normal form."""
	def enterNode(self, node):
		if type(node) == Implication:
			return (Not(node))
		return None, []

	def exitNode(self, node, data):
		return None

	def acceptChildResult(self, node, data, child_result):
		return None
