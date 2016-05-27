from pyomo.core import DataPortal
from pyomo.opt import SolverFactory
from DiseaseEstimation import model

# create the instance from multiple data files
data = DataPortal(model=model)
data.load(filename='DiseaseEstimation.dat')
data.load(filename='DiseasePop.dat')
instance = model.create_instance(data)

# create the solver and solve
with SolverFactory("ipopt") as solver:
    solver.solve(instance, tee=True)

# report results
instance.pprint()
