#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.environ import *
from pyomo.bilevel import *

def pyomo_create_model(options, model_options):
    M = ConcreteModel()
    M.x1 = Var(bounds=(0,None))
    M.x2 = Var(within=Binary)
    M.y1 = Var(bounds=(1,None))
    M.y2 = Var(bounds=(-100,2))
    M.y3 = Var(bounds=(None,None))
    M.y4 = Var(bounds=(3,4))
    M.o = Objective(expr=M.x1 - 4*M.y1)
    
    M.sub = SubModel(fixed=(M.x1, M.x2))
    M.sub.o  = Objective( expr=              11*M.x2 +          12*M.x2*M.y1 +         M.y2 +       9*M.y3                      )
    M.sub.c1 = Constraint(expr=                 M.x1 + 13*M.x2*M.y1 + 5*M.y1                                               <= 19)
    M.sub.c2 = Constraint(expr=20 <=          2*M.x1 +                6*M.y1 + 14*M.x2*M.y2 +      10*M.y3                      )
    M.sub.c3 = Constraint(expr=32 ==          4*M.x1 +                8*M.y1                               + 15*M.x2*M.y4       )
    M.sub.c4 = Constraint(expr=inequality(22, 3*M.x1 +                7*M.y1                + 16*M.x2*M.y3                ,  28))

    return M

if False:
    instance = pyomo_create_model(None, None)
    instance.pprint()
    print("+"*80)

    xfrm = TransformationFactory('bilevel.linear_mpec')
    xfrm.apply_to(instance)
    instance.pprint()
    print("+"*80)

    xfrm = TransformationFactory('mpec.simple_disjunction')
    xfrm.apply_to(instance)
    instance.pprint()
    print("+"*80)

    xfrm = TransformationFactory('gdp.bigm')
    xfrm.apply_to(instance, bigM=999)
    instance.pprint()
    print("+"*80)

    xfrm = TransformationFactory('gdp.bilinear')
    xfrm.apply_to(instance)
    instance.pprint()
    print("+"*80)

    xfrm = TransformationFactory('gdp.bigm')
    xfrm.apply_to(instance, bigM=888)
    instance.pprint()
    print("+"*80)


