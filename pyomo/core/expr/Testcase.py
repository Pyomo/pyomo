

#from pyomo.core.expr.logicalvalue import LogicalValue, LogicalConstant, native_logical_types, as_logical

from pyomo.core.expr.logical_expr import (AndExpression, LogicalExpressionBase, NotExpression,
<<<<<<< HEAD
	 UnaryExpression, BinaryExpression, Not, Equivalence, Xor, And, Or)
=======
	 UnaryExpression, BinaryExpression, Not, Equivalence, Xor, Conjunct, And, Disjunct, Or)
>>>>>>> e5dcbbcdad3506d83513b0a7c8b31f8002f5d999

#NotExpression Safety Test:

b = UnaryExpression(0)
a = BinaryExpression(1,2)
c = NotExpression(3)

<<<<<<< HEAD
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
=======
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

>>>>>>> e5dcbbcdad3506d83513b0a7c8b31f8002f5d999

#Conjunct(list([a,b,c]))
