#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# A duality example adapted from
#    http://www.stanford.edu/~ashishg/msande111/notes/chapter4.pdf
#
from pyomo.environ import *


def pyomo_create_model(options, model_options):
    model = ConcreteModel()
    model.x1 = Var(within=NonNegativeReals)
    model.x2 = Var(within=NonNegativeReals)
    model.o = Objective(expr=3 * model.x1 + 2.5 * model.x2, sense=maximize)

    model.c1 = Constraint(expr=4.44 * model.x1 <= 100)
    model.c2 = Constraint(expr=6.67 * model.x2 <= 100)
    model.c3 = Constraint(expr=4 * model.x1 + 2.86 * model.x2 <= 100)
    model.c4 = Constraint(expr=3 * model.x1 + 6 * model.x2 <= 100)

    return model
