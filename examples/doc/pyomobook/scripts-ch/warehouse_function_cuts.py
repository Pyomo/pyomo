import warnings
warnings.filterwarnings("ignore")
# The following import/use is needed to prevent matplotlib from using
# the X-backend on *nix platforms, which would fail when run in
# automated testing environments or when $DISPLAY is not set.
import matplotlib
matplotlib.use('agg')
# @all:
from warehouse_data import *
from pyomo.environ import *
from pyomo.opt import TerminationCondition
import warehouse_function as wf
import matplotlib.pyplot as plt

# call function to create model
model = wf.create_wl_model(N, M, d, P)
model.integer_cuts = ConstraintList()
objective_values = list()
done = False
while not done:
    # solve the model
    solver = SolverFactory('glpk')
    results = solver.solve(model)
    objective_values.append(value(model.obj))
    term_cond = results.solver.termination_condition
    print('')
    print('--- Solver Status: {0} ---'.format(term_cond))
   
    if term_cond != TerminationCondition.optimal:
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


x = range(len(objective_values))
plt.bar(x, objective_values, align='center')
plt.gca().set_xticks(x)
plt.xlabel('Solution Number')
plt.ylabel('Optimal Obj. Value')
plt.savefig('warehouse_function_cuts.pdf')
#plt.show()
# @:all
