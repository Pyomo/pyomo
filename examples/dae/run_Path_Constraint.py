#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.environ import *
from pyomo.dae import *
from Path_Constraint import m

# Discretize model using Finite Difference Method
# discretizer = TransformationFactory('dae.finite_difference')
# discretizer.apply_to(m,nfe=20,scheme='BACKWARD')

# Discretize model using Orthogonal Collocation
discretizer = TransformationFactory('dae.collocation')
discretizer.apply_to(m,nfe=7,ncp=6,scheme='LAGRANGE-RADAU')
discretizer.reduce_collocation_points(m,var=m.u,ncp=1,contset=m.t)

solver=SolverFactory('ipopt')

results = solver.solve(m, tee=True)

x1 = []
x2 = []
u = []
t=[]

for i in sorted(m.t):
    t.append(i)
    x1.append(value(m.x1[i]))
    x2.append(value(m.x2[i]))
    u.append(value(m.u[i]))

import matplotlib.pyplot as plt
plt.subplot(121)
plt.plot(t,x1)
plt.plot(t,x2,'r')
plt.legend(('x1','x2'))
plt.xlabel('t')
plt.subplot(122)
plt.plot(t,u)
plt.plot(t,u,'o')
plt.xlabel('t')
plt.ylabel('u')
plt.show()
