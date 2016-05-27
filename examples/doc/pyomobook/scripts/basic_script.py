from pyomo.environ import SolverFactory
from DiseaseEstimation import model

model.pprint()

instance = model.create('DiseaseEstimation.dat')
instance.pprint()

solver = SolverFactory("ipopt")
results = solver.solve(instance)

results.write()
instance.display()
