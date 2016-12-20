from pyomo.opt import SolverFactory
from concrete1 import model as instance

instance.pprint()

# @cmd:
solver = SolverFactory('ipopt')
# @:cmd
results = solver.solve(instance)

instance.pprint()
