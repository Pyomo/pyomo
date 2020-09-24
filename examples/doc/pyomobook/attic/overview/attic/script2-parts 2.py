import pyomo.environ
from pyomo.opt import SolverFactory
from abstract5 import model

model.pprint()

# @create:
instance = model.create_instance('abstract5.dat')
instance.pprint()
# @:create

opt = SolverFactory("glpk")
results = opt.solve(instance)

results.write()
