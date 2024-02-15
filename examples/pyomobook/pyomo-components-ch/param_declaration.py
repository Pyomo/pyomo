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

model = pyo.ConcreteModel()

# @decl1:
model.Z = pyo.Param(initialize=32)
# @:decl1

# @decl3:
model.A = pyo.Set(initialize=[1, 2, 3])
model.B = pyo.Set(initialize=['A', 'B'])
model.U = pyo.Param(model.A, initialize={1: 10, 2: 20, 3: 30})
model.T = pyo.Param(
    model.A, model.B, initialize={(1, 'A'): 10, (2, 'B'): 20, (3, 'A'): 30}
)
# @:decl3

model.pprint()
