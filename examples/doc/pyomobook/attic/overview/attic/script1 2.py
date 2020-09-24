import pyomo.environ
from pyomo.opt import SolverFactory
from concrete1 import model

model.pprint()

instance = model
instance.pprint()

opt = SolverFactory("glpk")
results = opt.solve(instance)
instance.solutions.store_to(results)

results.write()
