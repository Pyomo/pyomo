import pyomo.kernel as pmo

v = pmo.variable(value=2)

#
# Expressions
#

e = pmo.expression()
assert e() == None
assert e.expr == None

e = pmo.expression(expr= v**2 + 1)
assert e() == 5
assert pmo.value(e) == 5
assert pmo.value(e.expr) == 5

e = pmo.expression()
e.expr = v - 1
assert pmo.value(e) == 1

esub = pmo.expression(expr= v + 1)
e = pmo.expression(expr= esub + 1)
assert pmo.value(esub) == 3
assert pmo.value(e) == 4

esub.expr = v - 1
assert pmo.value(esub) == 1
assert pmo.value(e) == 2

c = pmo.constraint()
c.body = e + 1
assert pmo.value(c.body) == 3

e.expr = 3
assert pmo.value(c.body) == 4

#
# Data expressions (can be used in constraint bounds)
#

e = pmo.data_expression()
c = pmo.constraint()
c.lb = e + 1

e.expr = -1
assert pmo.value(c.lb) == 0

# the following will result in an error
#e = pmo.expression()
#c = pmo.constraint()
#c.lb = e
