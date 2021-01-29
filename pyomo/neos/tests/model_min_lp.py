#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo

model = pyo.ConcreteModel()
model.y = pyo.Var(bounds=(-10,10), initialize=0.5)
model.x = pyo.Var(bounds=(-5,5), initialize=0.5)

@model.ConstraintList()
def c(m):
    yield m.y >= m.x - 2
    yield m.y >= - m.x
    yield m.y <= m.x
    yield m.y <= 2 - m.x

model.obj = pyo.Objective(expr=model.y, sense=pyo.minimize)
