import pyomo.core.kernel as pk

#
# Continuous variables
#

v = pk.variable()

v = pk.variable(domain=pk.Reals)

v = pk.variable(domain=pk.NonNegativeReals,
                ub=10)

v = pk.variable(domain_type=pk.RealSet,
                lb=1)

# error (because domain lower bound is finite)
#v = pk.variable(domain=pk.NonNegativeReals,
#                lb=1)

#
# Discrete variables
#

v = pk.variable(domain=pk.Binary)

v = pk.variable(domain=pk.Integers)

v = pk.variable(domain=pk.NonNegativeIntegers,
                ub=10)

v = pk.variable(domain_type=pk.IntegerSet,
                lb=1)

# error (because domain upper bound is finite)
#v = pk.variable(domain=pk.NegativeIntegers,
#                ub=10)

#
# Usage
#

v = pk.variable()
assert v.value == None
assert v.lb == None
assert v.ub == None
assert v.fixed == False
assert v.domain_type == pk.RealSet

# set the value
v.value = 2
assert v.value == 2
assert pk.value(v**2) == 4

# set the bounds
v.lb = 10
v.ub = 20
assert v.lb == 10
assert v.ub == 20

# set the domain (always overwrites bounds, even if infinite)
v.domain = pk.Reals
assert v.lb == None
assert v.ub == None
assert v.domain_type == pk.RealSet
v.domain = pk.Binary
assert v.lb == 0
assert v.ub == 1
assert v.domain_type == pk.IntegerSet

# set the domain_type (never overwrites bounds)
v.domain_type = pk.RealSet
assert v.lb == 0
assert v.ub == 1
assert v.domain_type == pk.RealSet

# fix the variable to its current value
v.fix()
assert v.value == 2
assert v.fixed == True

# fix the variable to a new value
v.fix(1)
assert v.value == 1
assert v.fixed == True

# free the variable
v.free()
assert v.value == 1
assert v.fixed == False
