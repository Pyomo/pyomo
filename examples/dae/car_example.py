#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# Ampl Car Example
#
# Shows how to convert a minimize final time optimal control problem
# to a format pyomo.dae can handle by removing the time scaling from
# the ContinuousSet.
#
# min tf
# dxdt = 0
# dvdt = a-R*v^2
# x(0)=0; x(tf)=L
# v(0)=0; v(tf)=0
# -3<=a<=1

from pyomo.environ import *
from pyomo.dae import *

m = ConcreteModel()

m.R = Param(initialize=0.001)  #  Friction factor
m.L = Param(initialize=100.0)  #  Final position

m.tau = ContinuousSet(bounds=(0, 1))  # Unscaled time
m.time = Var(m.tau)  # Scaled time
m.tf = Var()
m.x = Var(m.tau, bounds=(0, m.L + 50))
m.v = Var(m.tau, bounds=(0, None))
m.a = Var(m.tau, bounds=(-3.0, 1.0), initialize=0)

m.dtime = DerivativeVar(m.time)
m.dx = DerivativeVar(m.x)
m.dv = DerivativeVar(m.v)

m.obj = Objective(expr=m.tf)


def _ode1(m, i):
    if i == 0:
        return Constraint.Skip
    return m.dx[i] == m.tf * m.v[i]


m.ode1 = Constraint(m.tau, rule=_ode1)


def _ode2(m, i):
    if i == 0:
        return Constraint.Skip
    return m.dv[i] == m.tf * (m.a[i] - m.R * m.v[i] ** 2)


m.ode2 = Constraint(m.tau, rule=_ode2)


def _ode3(m, i):
    if i == 0:
        return Constraint.Skip
    return m.dtime[i] == m.tf


m.ode3 = Constraint(m.tau, rule=_ode3)


def _init(m):
    yield m.x[0] == 0
    yield m.x[1] == m.L
    yield m.v[0] == 0
    yield m.v[1] == 0
    yield m.time[0] == 0


m.initcon = ConstraintList(rule=_init)

discretizer = TransformationFactory('dae.finite_difference')
discretizer.apply_to(m, nfe=15, scheme='BACKWARD')

solver = SolverFactory('ipopt')
solver.solve(m, tee=True)

print("final time = %6.2f" % (value(m.tf)))

x = []
v = []
a = []
time = []

for i in m.tau:
    time.append(value(m.time[i]))
    x.append(value(m.x[i]))
    v.append(value(m.v[i]))
    a.append(value(m.a[i]))

import matplotlib.pyplot as plt

plt.subplot(131)
plt.plot(time, x, label='x')
plt.title('location')
plt.xlabel('time')

plt.subplot(132)
plt.plot(time, v, label='v')
plt.xlabel('time')
plt.title('velocity')

plt.subplot(133)
plt.plot(time, a, label='a')
plt.xlabel('time')
plt.title('acceleration')

plt.show()
