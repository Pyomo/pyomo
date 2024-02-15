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
# @body:
model.x = pyo.Var()
model.y = pyo.Var(bounds=(-2, 4))
model.z = pyo.Var(initialize=1.0, within=pyo.NonNegativeReals)

model.obj = pyo.Objective(expr=model.x**2 + model.y + model.z)

model.eq_con = pyo.Constraint(expr=model.x + model.y + model.z == 1)
model.ineq_con = pyo.Constraint(expr=model.x + model.y <= 0)
# @:body

model.pprint()
