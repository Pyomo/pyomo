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

from pyomo.environ import *


def create_model(N):
    model = ConcreteModel()

    model.A = RangeSet(N)
    model.x = Var(model.A, bounds=(1, 2))

    expr = 0
    for i in model.A:
        if not (i + 1) in model.A:
            continue
        expr += i * (model.x[i] * model.x[i + 1] + 1)
    model.obj = Objective(expr=expr)

    return model


def pyomo_create_model(options=None, model_options=None):
    return create_model(100000)
