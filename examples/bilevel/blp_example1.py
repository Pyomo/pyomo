#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.environ import *
from pyomo.bilevel import *

def pyomo_create_model(options, model_options):
    M = ConcreteModel()
    M.x = Var(bounds=(0,None))
    M.y1 = Var(bounds=(1,None))
    M.y2 = Var(bounds=(None,2))
    M.y3 = Var(bounds=(None,None))
    M.y4 = Var(bounds=(3,4))
    M.o = Objective(expr=M.x - 4*M.y1)
    
    M.sub = SubModel(fixed=M.x)
    M.sub.o  = Objective( expr=                     + M.y2 +  9*M.y3      )
    M.sub.c1 = Constraint(expr=        M.x + 5*M.y1        + 10*M.y3 <= 19)
    M.sub.c2 = Constraint(expr=20 <= 2*M.x + 6*M.y1                       )
    M.sub.c3 = Constraint(expr=21 <= 3*M.x + 7*M.y1                  <= 21)
    M.sub.c4 = Constraint(expr=22 == 4*M.x + 8*M.y1                       )

    return M

instance = pyomo_create_model(None, None)
xfrm = TransformationFactory('bilevel.linear_mpec')
xfrm.apply_to(instance)
instance.pprint()
