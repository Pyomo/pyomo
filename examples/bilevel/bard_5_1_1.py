#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# Example 5.1.1 from
#
# Practical Bilevel Optimization: Algorithms and Applications
#   Jonathan Bard

from pyomo.environ import *
from pyomo.bilevel import *

def pyomo_create_model(options, model_options):
    M = ConcreteModel()
    M.x = Var(bounds=(0,None))
    M.y = Var(bounds=(0,None))
    M.o = Objective(expr=M.x - 4*M.y)

    M.sub = SubModel(fixed=M.x)
    M.sub.o = Objective(expr=M.y)
    M.sub.c1 = Constraint(expr=-  M.x -   M.y <= -3)
    M.sub.c2 = Constraint(expr=-2*M.x +   M.y <=  0)
    M.sub.c3 = Constraint(expr= 2*M.x +   M.y <= 12)
    M.sub.c4 = Constraint(expr=-3*M.x + 2*M.y <= -4)

    return M
