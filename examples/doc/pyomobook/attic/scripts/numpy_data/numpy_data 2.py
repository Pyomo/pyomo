from pyomo.environ import *
from pyomo.opt import TerminationCondition
import numpy
import scipy

model = ConcreteModel()

# load previously defined cost and rhs array using numpy
model.c = numpy.loadtxt('numpy_datac.txt').tolist()
model.n = len(model.c)
model.b = numpy.loadtxt('numpy_datab.txt').tolist()
model.m = len(model.b)

# generate a random sparse constraint matrix in
# Compressed Sparse Row storage format using scipy
model.A = scipy.sparse.rand(
    model.m, model.n, density=0.5, format='csr')
model.A_data = model.A.data.tolist()
model.A_indices = model.A.indices.tolist()
model.A_indptr = model.A.indptr.tolist()

# define the Pyomo optimization objects
model.rows = RangeSet(0, model.m-1)
model.cols = RangeSet(0, model.n-1)
model.x = Var(model.cols, within=NonNegativeReals)

model.obj = Objective(expr= \
    sum(model.c[j]*model.x[j] for j in model.cols))

def _con_rule(model, i):
    if model.A_indptr[i+1] == model.A_indptr[i] + 1:
        return Constraint.Skip
    return sum((model.A_data[p] * \
                model.x[model.A_indices[p]])
               for p in range(model.A_indptr[i],
                              model.A_indptr[i+1])) \
           == model.b[i]
model.con = Constraint(model.rows, rule=_con_rule)

# solve the model
with SolverFactory("glpk") as opt:
    results = opt.solve(model)
    # check that the solver reports
    # solution optimality
    success = False
    if (results.solver.termination_condition ==
        TerminationCondition.optimal):
        success = True
    print("Pyomo Solve Result:")
    print("     solver status: %s"
          % (results.solver.status))
    print(" termination cond.: %s"
          % (results.solver.termination_condition))
    print("           success: %s" % (success))
    if success:
        print("         objective: %f" % (model.obj()))
    else:
        print("         objective: None")
