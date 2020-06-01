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
import numpy as np
import pandas as pd
from itertools import product
from rooney_biegler import rooney_biegler_model

from pyomo.contrib.interior_point.inverse_reduced_hessian import inv_reduced_hessian_barrier

# Data
data = pd.DataFrame(data=[[1,8.3],[2,10.3],[3,19.0],
                          [4,16.0],[5,15.6],[7,19.8]],
                    columns=['hour', 'y'])

model = rooney_biegler_model(data)

def residual_rule(m, i):
        expr = data.y[i] - m.response_function[data.hour[i]]
        return expr

model.residuals = pyo.Expression(data.index, rule = residual_rule)

# solver = pyo.SolverFactory('ipopt')
# solver.solve(model)

status, inv_red_hes = inv_reduced_hessian_barrier(model, 
                    independent_variables= [model.asymptote, model.rate_constant],
                    solver_options=None,
                    tee=True)

print("inverse of the reduced Hessian =\n",inv_red_hes)

obj = model.SSE()
# print(model.pprint())
print('asymptote = ', model.asymptote())
print('rate constant = ', model.rate_constant())
print('covariance\n=',2*obj/(len(data) - 2)*inv_red_hes)

