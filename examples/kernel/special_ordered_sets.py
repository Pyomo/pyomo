import pyomo.core.kernel as pk

v1 = pk.variable()
v2 = pk.variable()
v3 = pk.variable()

#
# Special Ordered Sets (Type 1)
#

s = pk.sos([v1,v2])
assert s.level == 1
assert s.weights == (1,2)
assert len(s.variables) == 2
assert v1 in s
assert v2 in s

s = pk.sos([v1,v2], level=1)
assert s.level == 1
assert s.weights == (1,2)
assert len(s.variables) == 2
assert v1 in s
assert v2 in s

s = pk.sos1([v1,v2])
assert s.level == 1
assert s.weights == (1,2)
assert len(s.variables) == 2
assert v1 in s
assert v2 in s

#
# Special Ordered Sets (Type 2)
#

s = pk.sos([v1,v2], level=2)
assert s.level == 2
assert s.weights == (1,2)
assert len(s.variables) == 2
assert v1 in s
assert v2 in s

s = pk.sos2([v1,v2])
assert s.level == 2
assert s.weights == (1,2)
assert len(s.variables) == 2
assert v1 in s
assert v2 in s

#
# Special Ordered Sets (Type n)
#

s = pk.sos([v1,v2,v3], level=3)
assert s.level == 3
assert s.weights == (1,2,3)
assert len(s.variables) == 3
assert v1 in s
assert v2 in s
assert v3 in s

#
# Specifying weights
#

# using known values
s = pk.sos([v1,v2], weights=[1.2,2.5])
assert s.weights == (1.2,2.5)

# using paramters
p = pk.parameter_list(
    pk.parameter() for i in range(2))
s = pk.sos([v1,v2], weights=[p[0]**2, p[1]**2])
assert len(s.weights) == 2
p[0].value = 1
p[1].value = 2
assert tuple(pk.value(w) for w in s.weights) == (1, 4)

# using data expressions
d = pk.expression_list(
    pk.data_expression() for i in range(2))
s = pk.sos([v1,v2], weights=d)
assert len(s.weights) == 2
d[0].expr = p[0] + 1
d[1].expr = p[0] + p[1]
assert tuple(pk.value(w) for w in s.weights) == (2, 3)

#
# Example model (piecewise linear objective)
#
import pyomo.environ

breakpoints = [-1.0, 0.0, 1.0, 2.0]
function_points = [2.0, -2.5, 3.0, 1.0]

m = pk.block()

m.z = pk.variable_list(
    pk.variable(domain=pk.NonNegativeReals)
    for i in range(4))
m.x = pk.variable()
m.y = pk.variable()

m.o = pk.objective(m.y)

m.c1 = pk.constraint(
    m.y == sum(f*z for f,z in zip(m.z, function_points)))
m.c2 = pk.constraint(
    m.x == sum(w*z for w,z in zip(m.z, breakpoints)))
m.c3 = pk.constraint(sum(m.z) == 1)
m.s = pk.sos2(m.z)

opt = pk.SolverFactory("cplex")
status = opt.solve(m)
assert status.solver.status == \
    pk.SolverStatus.ok
assert status.solver.termination_condition == \
    pk.TerminationCondition.optimal
