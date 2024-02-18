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
from Parameter_Estimation import model

instance = model.create_instance('data_set2.dat')
instance.t.pprint()

# Discretize model using Backward Finite Difference method
# discretizer = TransformationFactory('dae.finite_difference')
# discretizer.apply_to(instance,nfe=20,scheme='BACKWARD')

# Discretize model using Orthogonal Collocation
discretizer = TransformationFactory('dae.collocation')
discretizer.apply_to(instance, nfe=8, ncp=5)

solver = SolverFactory('ipopt')

results = solver.solve(instance, tee=True)

x1 = []
x1_meas = []
t = []
t_meas = []

print(sorted(instance.t))

for i in sorted(instance.MEAS_t):
    t_meas.append(i)
    x1_meas.append(value(instance.x1_meas[i]))

for i in sorted(instance.t):
    t.append(i)
    x1.append(value(instance.x1[i]))

import matplotlib.pyplot as plt

plt.plot(t, x1)
plt.plot(t_meas, x1_meas, 'o')
plt.xlabel('t')
plt.ylabel('x')
plt.title('Dynamic Parameter Estimation Using Collocation')
plt.show()
