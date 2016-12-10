from warehouse_data import *
from pyomo.environ import *
import warehouse_function as wf

# call function to create model
model = wf.create_wl_model(N, M, d, P)
model.integer_cuts = ConstraintList()
done = False
while not done:
    # solve the model
    solver = SolverFactory('glpk')
    results = solver.solve(model)
    print(results)
    status = str(results['Solver'][0]['Status'])
    term_cond = str(results['Solver'][0]['Termination condition'])
    print()
    print('--- Solver Status: {0} ---'.format(term_cond))
   
    if status != 'ok' or term_cond != 'optimal':
        done = True
    else:
        # look at the solution
        print('Optimal Obj. Value = {0}'.format(value(model.obj)))
        model.y.pprint()

        # create new integer cut to exclude this solution
        N_True = [i for i in N if value(model.y[i]) > 0.5]
        N_False = [i for i in N if value(model.y[i]) < 0.5]
        expr1 = sum(model.y[i] for i in N_True)
        expr2 = sum(model.y[i] for i in N_False)
        model.integer_cuts.add( sum(model.y[i] for i in N_True) \
                                - sum(model.y[i] for i in N_False) \
                                <= len(N_True)-1 )


