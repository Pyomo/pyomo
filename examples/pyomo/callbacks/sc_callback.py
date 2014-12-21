#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________


from pyomo.core import *
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


