import pyomo.environ
from pyomo.opt import SolverFactory
from abstract5 import model

model.pprint()

instance = model.create_instance('abstract5.dat')
instance.pprint()

opt = SolverFactory("glpk")
results = opt.solve(instance)
instance.solutions.store_to(results)

results.write()
