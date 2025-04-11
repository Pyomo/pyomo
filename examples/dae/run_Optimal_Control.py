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
from Optimal_Control import m

# Discretize model using Backward Finite Difference method
# discretizer = pyo.TransformationFactory('dae.finite_difference')
# discretizer.apply_to(m,nfe=20,scheme='BACKWARD')

# Discretize model using Orthogonal Collocation
discretizer = pyo.TransformationFactory('dae.collocation')
discretizer.apply_to(m, nfe=20, ncp=3, scheme='LAGRANGE-RADAU')
discretizer.reduce_collocation_points(m, var=m.u, ncp=1, contset=m.t)

solver = pyo.SolverFactory('ipopt')

results = solver.solve(m, tee=True)

x1 = []
x2 = []
u = []
t = []

print(sorted(m.t))

for i in sorted(m.t):
    t.append(i)
    x1.append(pyo.value(m.x1[i]))
    x2.append(pyo.value(m.x2[i]))
    u.append(pyo.value(m.u[i]))

import matplotlib.pyplot as plt

plt.plot(t, x1)
plt.plot(t, x2)
plt.show()

plt.plot(t, u)
plt.show()
