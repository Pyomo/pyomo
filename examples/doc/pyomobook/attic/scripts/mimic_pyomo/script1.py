from pyomo.opt import SolverFactory
from DiseaseEstimation import model

model.pprint()

instance = model.create('DiseaseEstimation.dat')
instance.pprint()

opt = SolverFactory("ipopt")
results = opt.solve(instance)

results.write()
