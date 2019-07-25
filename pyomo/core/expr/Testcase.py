

#from pyomo.core.expr.logicalvalue import LogicalValue, LogicalConstant, native_logical_types, as_logical

from pyomo.core.expr.logical_expr import (AndExpression, LogicalExpressionBase, NotExpression,
	 UnaryExpression, BinaryExpression, Not, Equivalence, Xor, And, Or)

#NotExpression Safety Test:

b = UnaryExpression(0)
a = BinaryExpression(1,2)
c = NotExpression(3)

print(type(And(a,b)))
print(And([a,b]).nargs()) # wrong length here
tmp = And(a,b)
print(len(tmp._args_)) # wrong length here.
print(type(tmp))
print(tmp)

print(type(Or(a,b)))
print(len(Or(a,b)._args_))
tmp = Or(a,b)
print(len(tmp._args_)) # wrong length here.
print(type(tmp))
print(tmp)

x = []
print(type(x))

#Conjunct(list([a,b,c]))
