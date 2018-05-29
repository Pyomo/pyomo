import pyomo.kernel as pmo

#
# Mutable parameters
#

p = pmo.parameter()
assert p.value == None

p = pmo.parameter(value=2)
assert p.value == 2

p.value = 4
assert p.value == 4
assert pmo.value(p**2) == 16
assert pmo.value(p - 1) == 3

v = pmo.variable()
c = pmo.constraint((p-1, v, p+1))
assert pmo.value(c.lb) == 3
assert pmo.value(c.ub) == 5

p.value = -1
assert pmo.value(c.lb) == -2
assert pmo.value(c.ub) == 0
