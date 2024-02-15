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

import pyomo.environ as pyo
from pyomo.dae import *
from path_constraint import m

# Discretize model using Orthogonal Collocation
# @disc:
discretizer = pyo.TransformationFactory('dae.collocation')
discretizer.apply_to(m, nfe=7, ncp=6, scheme='LAGRANGE-RADAU')
# @:disc
# @reduce:
discretizer.reduce_collocation_points(m, var=m.u, ncp=1, contset=m.t)
# @:reduce

solver = pyo.SolverFactory('ipopt')
solver.solve(m, tee=True)
