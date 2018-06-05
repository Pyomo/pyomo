from pyomo.environ import *
from pyomo.dae import *
from matplotlib import pyplot as plt

m = ConcreteModel()
m.t = ContinuousSet(bounds=(0,10))
m.p = Param(initialize=5, mutable=True)
m.v1 = Var(m.t)
m.dv1 = DerivativeVar(m.v1)
m.v2 = Var(m.t)
m.dv2 = DerivativeVar(m.v2)
m.v3 = Var(m.t)
m.dv3 = DerivativeVar(m.v3)

def _con1(m, t):
    return m.dv1[t] == 10 + m.p
m.con1 = Constraint(m.t, rule=_con1)

def _con2(m, t):
    return m.dv2[t] == -m.v1[t]
m.con2 = Constraint(m.t, rule=_con2)

def _con3(m, t):
    return m.v3[t] == -m.v2[t]
m.con3 = Constraint(m.t, rule=_con3)

m.v1[0] = 0
m.v2[0] = 0
m.v3[0] = 0

# discretizer = TransformationFactory('dae.finite_difference')
# discretizer.apply_to(m)



mysim = Simulator(m, package='casadi')
tsim, profiles = mysim.simulate(numpoints=100)

plt.plot(tsim, profiles)
plt.show()
