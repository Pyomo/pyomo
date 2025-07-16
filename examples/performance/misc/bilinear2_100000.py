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
    model.x = pyo.Var(model.A, bounds=(1, 2))

    with pyo.nonlinear_expression as expr:
        for i in model.A:
            if not (i + 1) in model.A:
                continue
            expr += i * (model.x[i] * model.x[i + 1] + 1)
    model.obj = pyo.Objective(expr=expr)

    return model


def pyomo_create_model(options=None, model_options=None):
    return create_model(100000)
