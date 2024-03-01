import pyomo.environ as pyo
m = pyo.ConcreteModel("Trivial Quad")
m.x = pyo.Var([1,2], bounds=(0,1))
m.y = pyo.Var(bounds=(0, 1))
m.c = pyo.Constraint(expr=m.x[1] * m.x[2] == -1)
m.d = pyo.Constraint(expr=m.x[1] + m.y >= 1)

from pyomo.contrib.mis.mis import compute_infeasibility_explanation
# if IDAES is installed, compute_infeasibility_explanation doesn't need to be passed a solver
# Note: this particular little problem is quadratic
# As of 18Feb DLW is not sure the explanation code works with solvers other than ipopt
ipopt = pyo.SolverFactory("ipopt")
compute_infeasibility_explanation(m, solver=ipopt)

