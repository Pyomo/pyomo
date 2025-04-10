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


def pyomo_create_model(options, model_options):
    model = pyo.ConcreteModel()
    model.x1 = pyo.Var(within=pyo.NonNegativeReals)
    model.x2 = pyo.Var(within=pyo.NonPositiveReals)
    model.x3 = pyo.Var(within=pyo.Reals)

    model.o = pyo.Objective(
        expr=model.x1 + 2 * model.x2 + 3 * model.x3, sense=pyo.maximize
    )

    model.c1 = pyo.Constraint(expr=-model.x1 + 3 * model.x2 == 5)
    model.c2 = pyo.Constraint(expr=2 * model.x1 - model.x2 + 3 * model.x3 >= 6)
    model.c3 = pyo.Constraint(expr=model.x3 <= 4)

    return model
