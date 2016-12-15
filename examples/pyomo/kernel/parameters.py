import pyomo.core.kernel as pk

#
# Mutable parameters
#

p = pk.parameter()
assert p.value == None

p = pk.parameter(value=2)
assert p.value == 2

p.value = 4
assert p.value == 4
assert pk.value(p**2) == 16
assert pk.value(p - 1) == 3

v = pk.variable()
c = pk.constraint(p-1 <= v <= p+1)
assert pk.value(c.lb) == 3
assert pk.value(c.ub) == 5

p.value = -1
assert pk.value(c.lb) == -2
assert pk.value(c.ub) == 0
