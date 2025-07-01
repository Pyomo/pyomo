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

import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar

m = pyo.ConcreteModel()
m.x = ContinuousSet(bounds=(0, 1))
m.y = ContinuousSet(bounds=(0, 1))
m.u = pyo.Var(m.x, m.y)

m.dudx = DerivativeVar(m.u, wrt=(m.x, m.x))
m.dudy = DerivativeVar(m.u, wrt=(m.y, m.y))


def _lowerY(m, i):
    if i == 0 or i == 1:
        return pyo.Constraint.Skip
    return m.u[i, 0] == 1


m.lowerY = pyo.Constraint(m.x, rule=_lowerY)


def _upperY(m, i):
    if i == 0 or i == 1:
        return pyo.Constraint.Skip
    return m.u[i, 1] == 2


m.upperY = pyo.Constraint(m.x, rule=_upperY)


def _lowerX(m, j):
    if j == 0 or j == 1:
        return pyo.Constraint.Skip
    return m.u[0, j] == 1


m.lowerX = pyo.Constraint(m.y, rule=_lowerX)


def _upperX(m, j):
    if j == 0 or j == 1:
        return pyo.Constraint.Skip
    return m.u[1, j] == 2


m.upperX = pyo.Constraint(m.y, rule=_upperX)


def _laplace(m, i, j):
    if i == 0 or i == 1:
        return pyo.Constraint.Skip
    if j == 0 or j == 1:
        return pyo.Constraint.Skip

    return m.dudx[i, j] + m.dudy[i, j] == 0


m.laplace = pyo.Constraint(m.x, m.y, rule=_laplace)


def _dummy(m):
    return 1.0


m.obj = pyo.Objective(rule=_dummy)

discretizer = pyo.TransformationFactory('dae.finite_difference')
discretizer.apply_to(m, nfe=20, wrt=m.y, scheme='FORWARD')
discretizer.apply_to(m, nfe=20, wrt=m.x, scheme='CENTRAL')

solver = pyo.SolverFactory('ipopt')

results = solver.solve(m, tee=True)

# disc.u.pprint()

x = []
y = []
u = []

for i in sorted(m.x):
    temp = []
    tempx = []
    for j in sorted(m.y):
        tempx.append(i)
        temp.append(pyo.value(m.u[i, j]))
    x.append(tempx)
    y.append(sorted(m.y))
    u.append(temp)


import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
p = ax.plot_wireframe(x, y, u, rstride=1, cstride=1)
fig.show()
