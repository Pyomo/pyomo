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

from pyomo.environ import *
from pyomo.dae import *
from distill_DAE import model

instance = model.create_instance('distill.dat')

# Discretize using Finite Difference Approach
discretizer = TransformationFactory('dae.finite_difference')
discretizer.apply_to(instance, nfe=50, scheme='BACKWARD')

# Discretize using Orthogonal Collocation
# discretizer = TransformationFactory('dae.collocation')
# discretizer.apply_to(instance,nfe=50,ncp=3)

# The objective function in the manually discretized pyomo model
# iterated over all finite elements and all collocation points.  Since
# the objective function is not explicitly indexed by a ContinuousSet
# we add the objective function to the model after it has been
# discretized to ensure that we include all the discretization points
# when we take the sum.


def obj_rule(m):
    return m.alpha * sum(
        (m.y[1, i] - m.y1_ref) ** 2 for i in m.t if i != 1
    ) + m.rho * sum((m.u1[i] - m.u1_ref) ** 2 for i in m.t if i != 1)


instance.OBJ = Objective(rule=obj_rule)

solver = SolverFactory('ipopt')

results = solver.solve(instance, tee=True)

# If you have matplotlib you can use the following code to plot the
# results
t = []
x5 = []
x20 = []

for i in sorted(instance.t):
    x5.append(value(instance.x[5, i]))
    x20.append(value(instance.x[20, i]))
    t.append(i)

import matplotlib.pyplot as plt

plt.plot(t, x5)
plt.plot(t, x20)
plt.show()
