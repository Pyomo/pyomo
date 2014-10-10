
from coopr.pyomo import *
from sc import *

@pyomo_callback('solve-callback')
def solve_callback(solver, model):
    print "CB-Solve"

@pyomo_callback('cut-callback')
def cut_callback(solver, model):
    print "CB-Cut"

@pyomo_callback('node-callback')
def node_callback(solver, model):
    print "CB-Node"


