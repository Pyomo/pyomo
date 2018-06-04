from pyomo.environ import *
from pyomo.dae import *
from matplotlib import pyplot as plt

m = ConcreteModel()
m.t = ContinuousSet(bounds=(0,10))
m.p = Param(initialize=5, mutable=True)
m.v1 = Var(m.t)
m.dv1 = DerivativeVar(m.v1)

def _con1(m, t):
    return m.dv1[t] == 10 + m.p
m.con1 = Constraint(m.t, rule=_con1)

m.v1[0] = 0

discretizer = TransformationFactory('dae.finite_difference')
discretizer.apply_to(m)



mysim = Simulator(m, package='scipy')
tsim, profiles = mysim.simulate(numpoints=100)

plt.plot(tsim, profiles)
plt.show()
