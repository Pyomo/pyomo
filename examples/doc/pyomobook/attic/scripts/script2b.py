from pyomo.opt import SolverFactory
from concrete1 import model as instance

# @cmd:
solver = SolverFactory('asl:coliny', solver='sco:ps')
# @:cmd

solver.solve(instance)
