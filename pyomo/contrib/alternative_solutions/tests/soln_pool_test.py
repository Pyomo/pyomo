# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:21:06 2023

@author: jlgearh
"""

import test_cases

model = test_cases.knapsack(10)

ast = '*'*10

# print(ast,'Start APPSI',ast)
# from pyomo.contrib import appsi
# opt = appsi.solvers.Gurobi()
# opt.config.stream_solver = True
# #opt.set_instance(model)
# opt.gurobi_options['PoolSolutions'] = 10
# opt.gurobi_options['PoolSearchMode'] = 2
# #opt.set_gurobi_param('PoolSolutions', 10)
# #opt.set_gurobi_param('PoolSearchMode', 2)
# results = opt.solve(model)
# print(ast,'END APPSI',ast)

# print(ast,'Start Solve Factory',ast)
# from pyomo.opt import SolverFactory
# opt2 = SolverFactory('gurobi')
# opt.gurobi_options['PoolSolutions'] = 10
# opt.gurobi_options['PoolSearchMode'] = 2
# opt2.solve(model, tee=True)
# print(ast,'End Solve Factory',ast)

print(ast,'Start Solve Factory',ast)
from pyomo.opt import SolverFactory
opt3 = SolverFactory('appsi_gurobi')
opt3.gurobi_options['PoolSolutions'] = 10
opt3.gurobi_options['PoolSearchMode'] = 2
opt3.solve(model, tee=True)
print(ast,'End Solve Factory',ast)