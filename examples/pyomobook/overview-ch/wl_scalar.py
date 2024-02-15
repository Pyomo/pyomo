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

# wl_scalar.py: snippets that show the warehouse location problem implemented as scalar quantities
import pyomo.environ as pyo

model = pyo.ConcreteModel()

# @vars:
model.x_Harlingen_NYC = pyo.Var(bounds=(0, 1))
model.x_Harlingen_LA = pyo.Var(bounds=(0, 1))
model.x_Harlingen_Chicago = pyo.Var(bounds=(0, 1))
model.x_Harlingen_Houston = pyo.Var(bounds=(0, 1))
model.x_Memphis_NYC = pyo.Var(bounds=(0, 1))
model.x_Memphis_LA = pyo.Var(bounds=(0, 1))
# ...
# @:vars
model.x_Memphis_Chicago = pyo.Var(bounds=(0, 1))
model.x_Memphis_Houston = pyo.Var(bounds=(0, 1))
model.x_Ashland_NYC = pyo.Var(bounds=(0, 1))
model.x_Ashland_LA = pyo.Var(bounds=(0, 1))
model.x_Ashland_Chicago = pyo.Var(bounds=(0, 1))
model.x_Ashland_Houston = pyo.Var(bounds=(0, 1))

model.y_Harlingen = pyo.Var(within=pyo.Binary)
model.y_Memphis = pyo.Var(within=pyo.Binary)
model.y_Ashland = pyo.Var(within=pyo.Binary)

P = 2

# @cons:
model.one_warehouse_for_NYC = pyo.Constraint(
    expr=model.x_Harlingen_NYC + model.x_Memphis_NYC + model.x_Ashland_NYC == 1
)

model.one_warehouse_for_LA = pyo.Constraint(
    expr=model.x_Harlingen_LA + model.x_Memphis_LA + model.x_Ashland_LA == 1
)
# ...
# @:cons

# @maxY:
model.maxY = pyo.Constraint(
    expr=model.y_Harlingen + model.y_Memphis + model.y_Ashland <= P
)
# @:maxY
model.pprint()
