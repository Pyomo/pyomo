

#from pyomo.core.expr.logicalvalue import LogicalValue, LogicalConstant, native_logical_types, as_logical

from pyomo.core.expr.logical_expr import (AndExpression, LogicalExpressionBase, NotExpression,
	 UnaryExpression, BinaryExpression, Not, Equivalence, Xor, Conjunct, And, Disjunct, Or)

#NotExpression Safety Test:

b = UnaryExpression(0)
a = BinaryExpression(1,2)
c = NotExpression(3)

print(type(Conjunct(list([a,b]))))
print(len(Conjunct(list([a,b]))._arg_))
tmp = And(a,b)
print(len(tmp._arg_)) # wrong length here.
print(type(tmp))
print(tmp)

print(type(Disjunct(list([a,b]))))
print(len(Disjunct(list([a,b]))._arg_))
tmp = Or(a,b)
print(len(tmp._arg_)) # wrong length here.
print(type(tmp))
print(tmp)


#Conjunct(list([a,b,c]))
