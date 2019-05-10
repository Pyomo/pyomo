import pyomo.kernel as pmo

#
# Specialized Conic Constraints
#

c = pmo.conic.quadratic(
    x=[pmo.variable(), pmo.variable()],
    r=pmo.variable(lb=0))
assert not c.has_lb()
assert c.has_ub() and (c.ub == 0)
assert c.check_convexity_conditions()
print(c.body)

c = pmo.conic.rotated_quadratic(
    x=[pmo.variable(), pmo.variable()],
    r1=pmo.variable(lb=0),
    r2=pmo.variable(lb=0))
assert not c.has_lb()
assert c.has_ub() and (c.ub == 0)
assert c.check_convexity_conditions()
print(c.body)

c = pmo.conic.primal_exponential(
    x1=pmo.variable(lb=0),
    x2=pmo.variable(),
    r=pmo.variable(lb=0))
assert not c.has_lb()
assert c.has_ub() and (c.ub == 0)
assert c.check_convexity_conditions()
print(c.body)

c = pmo.conic.primal_power(
    x=[pmo.variable(), pmo.variable()],
    r1=pmo.variable(lb=0),
    r2=pmo.variable(lb=0),
    alpha=0.5)
assert not c.has_lb()
assert c.has_ub() and (c.ub == 0)
assert c.check_convexity_conditions()
print(c.body)

c = pmo.conic.dual_exponential(
    x1=pmo.variable(),
    x2=pmo.variable(ub=0),
    r=pmo.variable(lb=0))
assert not c.has_lb()
assert c.has_ub() and (c.ub == 0)
assert c.check_convexity_conditions()
print(c.body)

c = pmo.conic.dual_power(
    x=[pmo.variable(), pmo.variable()],
    r1=pmo.variable(lb=0),
    r2=pmo.variable(lb=0),
    alpha=0.5)
assert not c.has_lb()
assert c.has_ub() and (c.ub == 0)
assert c.check_convexity_conditions()
print(c.body)

#
# Alternative interface available for each, where inputs can
# be variables, constants, or linear expressions. Return
# type is a block with the core conic constraint linked to
# the inputs via auxiliary variables and constraints.
#

b = pmo.conic.quadratic.as_domain(
    x=[pmo.variable() + 1, 1.5],
    r=pmo.variable(lb=0) / 2)
assert type(b.q) is pmo.conic.quadratic
assert type(b.c) is pmo.constraint_tuple
assert type(b.x) is pmo.variable_tuple
assert type(b.r) is pmo.variable
