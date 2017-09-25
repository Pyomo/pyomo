#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.environ import *
from pyomo.dae import *

m = ConcreteModel()
m.x = ContinuousSet(bounds=(0,1))
m.y = ContinuousSet(bounds=(0,1))
m.u = Var(m.x,m.y)

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

discretizer = TransformationFactory('dae.finite_difference')
discretizer.apply_to(m,nfe=20,wrt=m.y,scheme='FORWARD')
discretizer.apply_to(m,nfe=20,wrt=m.x,scheme='CENTRAL')

solver=SolverFactory('ipopt')

results = solver.solve(m,tee=True)

#disc.u.pprint()

x = []
y = []
u = []

for i in sorted(m.x):
    temp=[]
    tempx = []
    for j in sorted(m.y):
        tempx.append(i)
        temp.append(value(m.u[i,j]))
    x.append(tempx)
    y.append(sorted(m.y))
    u.append(temp)


import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection='3d')
p = ax.plot_wireframe(x,y,u,rstride=1,cstride=1)
fig.show()
