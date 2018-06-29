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

    model = ConcreteModel()
    model.z = Var(within=NonPositiveReals)
    model.x1 = Var(within=NonNegativeReals)
    model.x2 = Var(within=NonNegativeReals)
    model.x3 = Var(within=NonNegativeReals)
    model.o = Objective(expr=model.z*(6*model.x1 + 4*model.x2 + 2*model.x3), sense=maximize)

    # Create a submodel
    # The argument indicates the lower-level decision variables
    model.sub = SubModel(fixed=model.z)
    model.sub.o = Objective(expr=model.o.expr, sense=minimize)
    model.sub.c1 = Constraint(expr=4*model.x1 + 2*model.x2 + model.x3 >= 5)
    model.sub.c2 = Constraint(expr=model.x1 + model.x2 >= 3)
    model.sub.c3 = Constraint(expr=model.x2 + model.x3 >= 4)

    return model

