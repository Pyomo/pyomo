from pyomo.opt import *

# @create:
results = SolverResults()
# @:create

# @attr:
print results.problem
# @:attr

# @getitem:
print results['problem']
# @:getitem
