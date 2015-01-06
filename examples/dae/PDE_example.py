#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

# Example 1 from http://www.mathworks.com/help/matlab/ref/pdepe.html

from pyomo.core import *
from pyomo.dae import *
from pyomo.opt import SolverFactory
from pyomo.dae.plugins.finitedifference import Finite_Difference_Transformation
from pyomo.dae.plugins.colloc import Collocation_Discretization_Transformation
import math

m = ConcreteModel()
m.t = ContinuousSet(bounds=(0,2))
m.x = ContinuousSet(bounds=(0,1))
m.u = StateVar(m.x,m.t)

m.dudx = DerivativeVar(m.u,wrt=m.x)
m.dudx2 = DerivativeVar(m.u,wrt=(m.x,m.x))
m.dudt = DerivativeVar(m.u,wrt=m.t)

def _pde(m,i,j):
    if i == 0 or i == 1 or j == 0 :
        return Constraint.Skip
    return math.pi**2*m.dudt[i,j] == m.dudx2[i,j]
m.pde = Constraint(m.x,m.t,rule=_pde)

def _initcon(m,i):
    if i == 0 or i == 1:
        return Constraint.Skip
    return m.u[i,0] == sin(math.pi*i)
m.initcon = Constraint(m.x,rule=_initcon)

def _lowerbound(m,j):
    return m.u[0,j] == 0
m.lowerbound = Constraint(m.t,rule=_lowerbound)

def _upperbound(m,j):
    return math.pi*exp(-j)+m.dudx[1,j] == 0
m.upperbound = Constraint(m.t,rule=_upperbound)

m.obj = Objective(expr=1)

# Discretize using Finite Difference Method
discretize = Finite_Difference_Transformation()
disc = discretize.apply(m,nfe=25,wrt=m.x,scheme='BACKWARD')
disc = discretize.apply(disc,nfe=20,wrt=m.t,scheme='BACKWARD',clonemodel=False)

# Discretize using Orthogonal Collocation
#discretize2 = Collocation_Discretization_Transformation()
#disc = discretize2.apply(disc,nfe=10,ncp=3,wrt=m.x,clonemodel=False)
#disc = discretize2.apply(disc,nfe=20,ncp=3,wrt=m.t,clonemodel=False)


solver='ipopt'
opt=SolverFactory(solver)

results = opt.solve(disc,tee=True)
disc.load(results)

#disc.u.pprint()

x = []
t = []
u = []

for i in sorted(disc.x):
    temp=[]
    tempx = []
    for j in sorted(disc.t):
        tempx.append(i)
        temp.append(value(disc.u[i,j]))
    x.append(tempx)
    t.append(sorted(disc.t))
    u.append(temp)


import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection='3d')
ax.set_xlabel('Distance x')
ax.set_ylabel('Time t')
ax.set_title('Numerical Solution Using Backward Difference Method')
p = ax.plot_wireframe(x,t,u,rstride=1,cstride=1)
fig.show()
