from warehouse_data import *
import pyomo.environ as pe
import warehouse_function as wf

for pp in [1,2,3]:
    # call function to create model
    model = wf.create_wl_model(N, M, d, pp)

    # solve the model
    solver = pe.SolverFactory('glpk')
    solver.solve(model)

    # look at the solution
    print('--- P = {0} ---'.format(pp))
    model.y.pprint()
