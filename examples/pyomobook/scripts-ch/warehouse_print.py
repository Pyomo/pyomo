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

import json
import pyomo.environ as pyo
from warehouse_model import create_wl_model

# load the data from a json file
with open('warehouse_data.json', 'r') as fd:
    data = json.load(fd)

# call function to create model
model = create_wl_model(data, P=2)

# solve the model
solver = pyo.SolverFactory('glpk')
solver.solve(model)

# @printmistake:
print(model.y)
# @:printmistake

# @printindex:
print(pyo.value(model.y['Ashland']))
# @:printindex

# @printloop:
for i in model.y:
    print('{0} = {1}'.format(model.y[i], pyo.value(model.y[i])))
# @:printloop

# @printloopset:
for i in model.WH:
    print('{0} = {1}'.format(model.y[i], pyo.value(model.y[i])))
# @:printloopset

# @printslicing:
for v in model.x['Ashland', :]:
    print('{0} = {1}'.format(v, pyo.value(v)))
# @:printslicing

# @generalprintloop:
# loop over the Var objects on the model
for v in model.component_objects(ctype=pyo.Var):
    for index in v:
        print('{0} <= {1}'.format(v[index], pyo.value(v[index].ub)))

# or use the following to loop over the individual
# indices of each of the Var objects directly
for v in model.component_data_objects(ctype=pyo.Var):
    print('{0} <= {1}'.format(v, pyo.value(v.ub)))
# @:generalprintloop
