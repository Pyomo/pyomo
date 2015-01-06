#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core import *
from pyomo.dae import *
from pyomo.opt import SolverFactory
from pyomo.dae.plugins.finitedifference import Finite_Difference_Transformation

m = ConcreteModel()
m.x = ContinuousSet(bounds=(0,1))
m.y = ContinuousSet(bounds=(0,1))
m.u = StateVar(m.x,m.y)

m.dudx = DerivativeVar(m.u,wrt=(m.x,m.x))
m.dudy = DerivativeVar(m.u,wrt=(m.y,m.y))

def _lowerY(m,i):
    if i == 0 or i == 1:
        return Constraint.Skip
    return m.u[i,0] == 1
m.lowerY = Constraint(m.x,rule=_lowerY)

def _upperY(m,i):
    if i == 0 or i == 1:
        return Constraint.Skip
    return m.u[i,1] == 2
m.upperY = Constraint(m.x,rule=_upperY)

def _lowerX(m,j):
    if j == 0 or j == 1:
        return Constraint.Skip
    return m.u[0,j] == 1
m.lowerX = Constraint(m.y,rule=_lowerX)

def _upperX(m,j):
    if j == 0 or j == 1:
        return Constraint.Skip
    return m.u[1,j] == 2
m.upperX = Constraint(m.y,rule=_upperX)

def _laplace(m,i,j):
    if i == 0 or i == 1:
        return Constraint.Skip
    if j == 0 or j == 1:
        return Constraint.Skip

    return m.dudx[i,j] + m.dudy[i,j] == 0
m.laplace = Constraint(m.x,m.y,rule=_laplace)

def _dummy(m):
    return 1.0
m.obj = Objective(rule=_dummy)

discretize = Finite_Difference_Transformation()
disc = discretize.apply(m,nfe=20,wrt=m.y,scheme='FORWARD')
disc = discretize.apply(disc,nfe=20,wrt=m.x,scheme='CENTRAL',clonemodel=False)

solver='ipopt'
opt=SolverFactory(solver)

results = opt.solve(disc,tee=True)
disc.load(results)

#disc.u.pprint()

x = []
y = []
u = []

for i in sorted(disc.x):
    temp=[]
    tempx = []
    for j in sorted(disc.y):
        tempx.append(i)
        temp.append(value(disc.u[i,j]))
    x.append(tempx)
    y.append(sorted(disc.y))
    u.append(temp)


import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection='3d')
p = ax.plot_wireframe(x,y,u,rstride=1,cstride=1)
fig.show()
