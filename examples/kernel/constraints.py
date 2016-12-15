import pyomo.core.kernel as pk

v = pk.variable()

#
# Equality constraints
#

c = pk.constraint(v == 1)

c = pk.constraint(expr= v == 1)

c = pk.constraint(body=v, rhs=1)

c = pk.constraint()
c.body = v
c.rhs = 1

#
# Single-sided inequality constraints
#

c = pk.constraint(v <= 1)

c = pk.constraint(expr= v <= 1)

c = pk.constraint(body=v, ub=1)

c = pk.constraint()
c.body = v
c.ub = 1

c = pk.constraint(v >= 1)

c = pk.constraint(expr= v >= 1)

c = pk.constraint(body=v, lb=1)

c = pk.constraint()
c.body = v
c.lb = 1

#
# Range constraints
#

c = pk.constraint(0 <= v <= 1)

c = pk.constraint(expr= 0 <= v <= 1)

c = pk.constraint(lb=0, body=v, ub=1)

c = pk.constraint()
c.lb = 0
c.body = v
c.ub = 1

#
# Usage
#

v.value = 2

# initialize a range constraint
r = pk.constraint(lb=0, body=v**2, ub=5)
assert pk.value(r.lb) == 0
assert pk.value(r.ub) == 5
assert pk.value(r.body) == 4
assert r.equality == False
assert r.lslack == 4
assert r.uslack == 1
assert r.slack == 1

# change the lb
r.lb = -1
assert r.lslack == 5
assert r.uslack == 1
assert r.slack == 1

# setting rhs turns this into an equality constraint
r.rhs = 2
assert pk.value(r.lb) == 2
assert pk.value(r.ub) == 2
assert pk.value(r.rhs) == 2
assert pk.value(r.body) == 4
assert r.equality == True
assert r.lslack == 2
assert r.uslack == -2
assert r.slack == -2
