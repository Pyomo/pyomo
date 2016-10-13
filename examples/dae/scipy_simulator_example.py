from pyomo.environ import *
from pyomo.dae import *
from pyomo.dae.simulator import Simulator

#from integrator import scipy_integrator

m = ConcreteModel()

m.t = ContinuousSet(bounds=(0.0,10.0))

m.b = Param(initialize=0.25)
m.c = Param(initialize=5.0)

m.omega = Var(m.t)
m.theta = Var(m.t)

m.domegadt = DerivativeVar(m.omega,wrt=m.t)
m.dthetadt = DerivativeVar(m.theta,wrt=m.t)

# Setting the initial conditions
m.omega[0] = 0.0
m.theta[0] = 3.14-0.1

def _diffeq1(m,t):
    return m.domegadt[t] == -m.b*m.omega[t] - m.c*sin(m.theta[t])
m.diffeq1 = Constraint(m.t,rule=_diffeq1)


def _diffeq2(m,t):
    return m.dthetadt[t] == m.omega[t]
m.diffeq2 = Constraint(m.t,rule=_diffeq2)

sim = Simulator(m)
tsim, profiles = sim.simulate()
varorder = sim.get_variable_order()

import matplotlib.pyplot as plt

for idx,v in enumerate(varorder):
    plt.plot(tsim,profiles[:,idx],label=v)
plt.xlabel('t')
plt.legend(loc='best')
plt.show()
