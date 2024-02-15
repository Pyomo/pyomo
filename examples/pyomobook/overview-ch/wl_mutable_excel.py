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

# wl_mutable_excel.py: solve problem with different values for P
import pandas
import pyomo.environ as pyo
from wl_mutable import create_warehouse_model

# read the data from Excel using Pandas
df = pandas.read_excel('wl_data.xlsx', 'Delivery Costs', header=0, index_col=0)

N = list(df.index.map(str))
M = list(df.columns.map(str))
d = {(r, c): df.at[r, c] for r in N for c in M}
P = 2

# create the Pyomo model
model = create_warehouse_model(N, M, d, P)

# create the solver interface
solver = pyo.SolverFactory('glpk')

# loop over values for mutable parameter P
for n in range(1, 10):
    model.P = n
    res = solver.solve(model)
    pyo.assert_optimal_termination(res)
    print('# warehouses:', n, 'delivery cost:', pyo.value(model.obj))
