from pyomo.environ import SolverFactory
from abstract5 import model

model.pprint()

instance = model.create('abstract5.dat')
instance.pprint()

opt = SolverFactory("glpk")
results = opt.solve(instance)

results.write()
