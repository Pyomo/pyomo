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
    model.x2 = pyo.Var(within=pyo.NonNegativeReals)
    model.x3 = pyo.Var(within=pyo.NonNegativeReals)

    model.o = pyo.Objective(
        expr=6 * model.x1 + 4 * model.x2 + 2 * model.x3, sense=pyo.minimize
    )

    model.c1 = pyo.Constraint(expr=4 * model.x1 + 2 * model.x2 + model.x3 >= 5)
    model.c2 = pyo.Constraint(expr=model.x1 + model.x2 >= 3)
    model.c3 = pyo.Constraint(expr=model.x2 + model.x3 >= 4)

    return model
