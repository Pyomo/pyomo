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
