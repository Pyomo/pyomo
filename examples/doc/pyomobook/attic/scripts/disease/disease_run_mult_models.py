from pyomo.environ import *
from DiseaseEasy import model as easy_model
from DiseaseHard import model as hard_model

easy_instance = easy_model.create_instance('DiseaseEstimation.dat')
hard_instance = hard_model.create_instance('DiseaseEstimation.dat')

# solve the easier problem
with SolverFactory("ipopt") as solver:
    solver.solve(easy_instance)

# load the solution from the easy instance into the hard
# instance to provide a good initial guess for the solver
for t in easy_instance.S_SI:
    hard_instance.I[t].value = easy_instance.I[t].value
    hard_instance.S[t].value = easy_instance.S[t].value
    hard_instance.eps_I[t].value = easy_instance.eps_I[t].value
hard_instance.beta.value = easy_instance.beta.value

# solve the hard problem with the new initialization
with SolverFactory("ipopt") as solver:
    solver.solve(hard_instance)

print("Objective: %f" % (hard_instance.objective()))
print("Alpha: %f" % (hard_instance.alpha()))
print("Beta: %f" % (hard_instance.beta()))
