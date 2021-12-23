#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# This model is adapted from Noriyuki Yoshio's model for his and Biegler's
# 2020 publication in AIChE

from pyomo.environ import (
    ConcreteModel, Var, Reals, ExternalFunction, sin, cos,
    sqrt, Constraint, Objective)
from pyomo.opt import SolverFactory

m = ConcreteModel()
m.name = 'Example 2: Yoshio'

m.x1 = Var(initialize=0)
m.x2 = Var(bounds=(-2.0, None), initialize=0)

def ext_fcn(x, y):
    return x**2 + y**2
def grad_ext_fcn(args, fixed):
    x, y = args[:2]
    return [ 2*x, 2*y ]

m.EF = ExternalFunction(ext_fcn, grad_ext_fcn)

@m.Constraint()
def con(m):
    return 2*m.x1 + m.x2 + 10.0 == m.EF(m.x1, m.x2)

m.obj = Objective(expr = (m.x1 - 1)**2 + (m.x2 - 3)**2 + m.EF(m.x1, m.x2)**2)

def basis_rule(component, ef_expr):
    x1 = ef_expr.arg(0)
    x2 = ef_expr.arg(1)
    return x1**2 - x2 # This is the low fidelity model

optTRF = SolverFactory('trustregion',
                       maximum_iterations=50,
                       verbose=True)
optTRF.solve(m, [m.x1], ext_fcn_surrogate_map_rule=basis_rule)
