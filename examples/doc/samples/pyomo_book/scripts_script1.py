from pyomo.opt import SolverFactory
from DiseaseEstimation import model

# create the instance
instance = model.create_instance('DiseaseEstimation.dat')

# create the solver and solve
with SolverFactory("ipopt") as solver:
    solver.solve(instance)

# report results
instance.pprint()
