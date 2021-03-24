import pyomo.environ as pyo
from pyomo.dae import *
from path_constraint import m

# Discretize model using Orthogonal Collocation
# @disc:
discretizer = pyo.TransformationFactory('dae.collocation')
discretizer.apply_to(m,nfe=7,ncp=6,scheme='LAGRANGE-RADAU')
# @:disc
# @reduce:
discretizer.reduce_collocation_points(m, var=m.u, ncp=1, contset=m.t)
# @:reduce

solver=pyo.SolverFactory('ipopt')
solver.solve(m, tee=True)

