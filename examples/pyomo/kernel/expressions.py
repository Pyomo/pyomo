import pyomo.core.kernel as pk

v = pk.variable(value=2)

#
# Expressions
#

e = pk.expression()
assert e() == None
assert e.expr == None

e = pk.expression(expr= v**2 + 1)
assert e() == 5
assert pk.value(e) == 5
assert pk.value(e.expr) == 5

e = pk.expression()
e.expr = v - 1
assert pk.value(e) == 1

esub = pk.expression(expr= v + 1)
e = pk.expression(expr= esub + 1)
assert pk.value(esub) == 3
assert pk.value(e) == 4

esub.expr = v - 1
assert pk.value(esub) == 1
assert pk.value(e) == 2

c = pk.constraint()
c.body = e + 1
assert pk.value(c.body) == 3

e.expr = 3
assert pk.value(c.body) == 4

#
# Data expressions (can be used in constraint bounds)
#

e = pk.data_expression()
c = pk.constraint()
c.lb = e + 1

e.expr = -1
assert pk.value(c.lb) == 0

# the following will result in an error
#e = pk.expression()
#c = pk.constraint()
#c.lb = e
