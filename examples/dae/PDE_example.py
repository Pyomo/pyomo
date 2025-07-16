#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# Example 1 from http://www.mathworks.com/help/matlab/ref/pdepe.html

import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar

m = pyo.ConcreteModel()
m.pi = pyo.Param(initialize=3.1416)
m.t = ContinuousSet(bounds=(0, 2))
m.x = ContinuousSet(bounds=(0, 1))
m.u = pyo.Var(m.x, m.t)

m.dudx = DerivativeVar(m.u, wrt=m.x)
m.dudx2 = DerivativeVar(m.u, wrt=(m.x, m.x))
m.dudt = DerivativeVar(m.u, wrt=m.t)


def _pde(m, i, j):
    if i == 0 or i == 1 or j == 0:
        return pyo.Constraint.Skip
    return m.pi**2 * m.dudt[i, j] == m.dudx2[i, j]


m.pde = pyo.Constraint(m.x, m.t, rule=_pde)


def _initcon(m, i):
    if i == 0 or i == 1:
        return pyo.Constraint.Skip
    return m.u[i, 0] == pyo.sin(m.pi * i)


m.initcon = pyo.Constraint(m.x, rule=_initcon)


def _lowerbound(m, j):
    return m.u[0, j] == 0


m.lowerbound = pyo.Constraint(m.t, rule=_lowerbound)


def _upperbound(m, j):
    return m.pi * pyo.exp(-j) + m.dudx[1, j] == 0


m.upperbound = pyo.Constraint(m.t, rule=_upperbound)

m.obj = pyo.Objective(expr=1)

# Discretize using Orthogonal Collocation
# discretizer = pyo.TransformationFactory('dae.collocation')
# discretizer.apply_to(m,nfe=10,ncp=3,wrt=m.x)
# discretizer.apply_to(m,nfe=20,ncp=3,wrt=m.t)

# Discretize using Finite Difference and Collocation
discretizer = pyo.TransformationFactory('dae.finite_difference')
discretizer2 = pyo.TransformationFactory('dae.collocation')
discretizer.apply_to(m, nfe=25, wrt=m.x, scheme='BACKWARD')
discretizer2.apply_to(m, nfe=20, ncp=3, wrt=m.t)

# Discretize using Finite Difference Method
# discretizer = pyo.TransformationFactory('dae.finite_difference')
# discretizer.apply_to(m,nfe=25,wrt=m.x,scheme='BACKWARD')
# discretizer.apply_to(m,nfe=20,wrt=m.t,scheme='BACKWARD')

solver = pyo.SolverFactory('ipopt')
results = solver.solve(m, tee=True)

x = []
t = []
u = []

for i in sorted(m.x):
    temp = []
    tempx = []
    for j in sorted(m.t):
        tempx.append(i)
        temp.append(pyo.value(m.u[i, j]))
    x.append(tempx)
    t.append(sorted(m.t))
    u.append(temp)


import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlabel('Distance x')
ax.set_ylabel('Time t')
p = ax.plot_wireframe(x, t, u, rstride=1, cstride=1)
fig.show()
