#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# The parametric.mod example from sIPOPT
#
# Original implementation by Hans Pirany is found in 
#	pyomo/examples/pyomo/suffixes
#


import pyomo.environ
from pyomo.core import *
from pyomo.opt import SolverFactory
from sensitivity_toolbox import sipopt

def create_model():
    m = ConcreteModel()
    
    m.x1 = Var(initialize = 0.15, within=NonNegativeReals)
    m.x2 = Var(initialize = 0.15, within=NonNegativeReals)
    m.x3 = Var(initialize = 0.0, within=NonNegativeReals)
    
    m.eta1 = Param(initialize=4.5,mutable=True)
    m.eta2 = Param(initialize=1.0,mutable=True)
    
    m.const1 = Constraint(expr=6*m.x1+3*m.x2+2*m.x3-m.eta1 ==0)
    m.const2 = Constraint(expr=m.eta2*m.x1+m.x2-m.x3-1 ==0)
    m.cost = Objective(expr=m.x1**2+m.x2**2+m.x3**2)
    
    m.perturbed_eta1 = Param(initialize = 3.0)
    m.perturbed_eta2 = Param(initialize = 1.0)
    
    return m    
