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

# @script:
import json
import pyomo.environ as pyo
from warehouse_model import create_wl_model

# load the data from a json file
with open('warehouse_data.json', 'r') as fd:
    data = json.load(fd)

# call function to create model
model = create_wl_model(data, P=2)

# create the solver
solver = pyo.SolverFactory('glpk')

# options can be set directly on the solver
solver.options['noscale'] = None
solver.options['log'] = 'warehouse.log'
solver.solve(model, tee=True)
model.y.pprint()

# options can also be passed via the solve command
myoptions = dict()
myoptions['noscale'] = None
myoptions['log'] = 'warehouse.log'
solver.solve(model, options=myoptions, tee=True)
model.y.pprint()
# @:script

import os

os.remove('warehouse.log')
