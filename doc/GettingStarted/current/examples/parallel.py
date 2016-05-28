# parallel.py
from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.opt.parallel import SolverManagerFactory
import sys

action_handle_map = {} # maps action handles to instances

# Create a solver
optsolver = SolverFactory('cplex')

# create a solver manager
# 'pyro' could be replaced with 'serial'
solver_manager = SolverManagerFactory('pyro')
if solver_manager is None:
    print "Failed to create solver manager."
    sys.exit(1)

#
# A simple model with binary variables and
# an empty constraint list.
#
model = AbstractModel()
model.n = Param(default=4)
model.x = Var(RangeSet(model.n), within=Binary)
def o_rule(model):
    return summation(model.x)
model.o = Objective(rule=o_rule)
model.c = ConstraintList()

# Create two model instances
instance1 = model.create()

instance2 = model.create()
instance2.x[1] = 1
instance2.x[1].fixed = True

# send them to the solver(s)
action_handle = solver_manager.queue(instance1, opt=optsolver, warmstart=False, tee=True, verbose=False)
action_handle_map[action_handle] = "Original"
action_handle = solver_manager.queue(instance2, opt=optsolver, warmstart=False, tee=True, verbose=False)
action_handle_map[action_handle] = "One Var Fixed"

# retrieve the solutions
for i in range(2): # we know there are two instances
    this_action_handle = solver_manager.wait_any()
    solved_name = action_handle_map[this_action_handle]
    results = solver_manager.get_results(this_action_handle)
    print "Results for",solved_name
    print results

