import pyomo.core.kernel as pk

v = pk.variable(value=2)

#
# Objectives
#

o = pk.objective()
assert o() == None
assert o.expr == None

o = pk.objective(expr= v**2 + 1)
assert o() == 5
assert pk.value(o) == 5
assert pk.value(o.expr) == 5

o = pk.objective()
o.expr = v - 1
assert pk.value(o) == 1

osub = pk.objective(expr= v + 1)
o = pk.objective(expr= osub + 1)
assert pk.value(osub) == 3
assert pk.value(o) == 4

osub.expr = v - 1
assert pk.value(osub) == 1
assert pk.value(o) == 2
