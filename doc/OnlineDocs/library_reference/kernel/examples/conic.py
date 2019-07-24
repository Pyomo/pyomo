# @Class
import pyomo.kernel as pmo
m = pmo.block()
m.x1 = pmo.variable(lb=0)
m.x2 = pmo.variable()
m.r = pmo.variable(lb=0)
m.q = pmo.conic.primal_exponential(
    x1=m.x1,
    x2=m.x2,
    r=m.r)
# @Class
del m

# @Domain
import pyomo.kernel as pmo
import math
m = pmo.block()
m.x = pmo.variable(lb=0)
m.y = pmo.variable(lb=0)
m.b = pmo.conic.primal_exponential.as_domain(
    x1=math.sqrt(2)*m.x,
    x2=2.0,
    r=2*(m.x + m.y))
# @Domain
