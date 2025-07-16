#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo


def create_model(N):
    model = pyo.ConcreteModel()

    model.A = pyo.RangeSet(N)
    model.x = pyo.Var(model.A)

    expr = sum(i * model.x[i] for i in model.A)
    model.obj = pyo.Objective(expr=expr)

    def c_rule(model, i):
        return (N - i + 1) * model.x[i] >= N

    model.c = pyo.Constraint(model.A)

    return model


def pyomo_create_model(options=None, model_options=None):
    return create_model(100)
