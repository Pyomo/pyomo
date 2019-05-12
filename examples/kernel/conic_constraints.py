import pyomo.kernel as pmo

#
# Specialized Conic Constraints
#

c = pmo.conic.quadratic(
    r=pmo.variable(lb=0),
    x=[pmo.variable(), pmo.variable()])
assert not c.has_lb()
assert c.has_ub() and (c.ub == 0)
assert c.check_convexity_conditions()
print(c.body)

c = pmo.conic.rotated_quadratic(
    r1=pmo.variable(lb=0),
    r2=pmo.variable(lb=0),
    x=[pmo.variable(), pmo.variable()])
assert not c.has_lb()
assert c.has_ub() and (c.ub == 0)
assert c.check_convexity_conditions()
print(c.body)

c = pmo.conic.primal_exponential(
    r=pmo.variable(lb=0),
    x1=pmo.variable(lb=0),
    x2=pmo.variable())
assert not c.has_lb()
assert c.has_ub() and (c.ub == 0)
assert c.check_convexity_conditions()
print(c.body)

c = pmo.conic.primal_power(
    r1=pmo.variable(lb=0),
    r2=pmo.variable(lb=0),
    x=[pmo.variable(), pmo.variable()],
    alpha=0.5)
assert not c.has_lb()
assert c.has_ub() and (c.ub == 0)
assert c.check_convexity_conditions()
print(c.body)

c = pmo.conic.dual_exponential(
    r=pmo.variable(lb=0),
    x1=pmo.variable(),
    x2=pmo.variable(ub=0))
assert not c.has_lb()
assert c.has_ub() and (c.ub == 0)
assert c.check_convexity_conditions()
print(c.body)

c = pmo.conic.dual_power(
    r1=pmo.variable(lb=0),
    r2=pmo.variable(lb=0),
    x=[pmo.variable(), pmo.variable()],
    alpha=0.5)
assert not c.has_lb()
assert c.has_ub() and (c.ub == 0)
assert c.check_convexity_conditions()
print(c.body)

#
# Alternative interface available for each, where inputs can
# be variables, constants, linear expressions, or None. Return
# type is a block with the core conic constraint linked to
# the inputs via auxiliary variables and constraints.
#

b = pmo.conic.quadratic.as_domain(
    r=0.5*pmo.variable(lb=0),
    x=[pmo.variable() + 1, 1.5, None, None])
assert type(b.q) is pmo.conic.quadratic
assert type(b.c) is pmo.constraint_tuple
assert len(b.c) == 3
assert type(b.r) is pmo.variable
assert type(b.x) is pmo.variable_tuple
assert len(b.x) == 4
