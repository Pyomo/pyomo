from pyomo.environ import *
from pyomo.dae import *
from path_constraint import m

# Discretize model using Orthogonal Collocation
# @disc:
discretizer = TransformationFactory('dae.collocation')
discretizer.apply_to(m,nfe=7,ncp=6,scheme='LAGRANGE-RADAU')
# @:disc
# @reduce:
discretizer.reduce_collocation_points(m, var=m.u, ncp=1, contset=m.t)
# @:reduce

solver=SolverFactory('ipopt')
results = solver.solve(m, tee=True)

