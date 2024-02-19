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


from pyomo.core import *


def pyomo_create_model(options=None, model_options=None):
    import random

    random.seed(1000)

    model = AbstractModel()

    model.N = Param(within=PositiveIntegers)

    model.Locations = RangeSet(1, model.N)

    model.P = Param(within=RangeSet(1, model.N))

    model.M = Param(within=PositiveIntegers)

    model.Customers = RangeSet(1, model.M)

    model.d = Param(
        model.Locations,
        model.Customers,
        initialize=lambda n, m, model: random.uniform(1.0, 2.0),
        within=Reals,
    )

    model.x = Var(model.Locations, model.Customers, bounds=(0.0, 1.0))

    model.y = Var(model.Locations, within=Binary)

    def rule(model):
        return sum(
            model.d[n, m] * model.x[n, m]
            for n in model.Locations
            for m in model.Customers
        )

    model.obj = Objective(rule=rule)

    def rule(model, m):
        return (sum(model.x[n, m] for n in model.Locations), 1.0)

    model.single_x = Constraint(model.Customers, rule=rule)

    def rule(model, n, m):
        return (None, model.x[n, m] - model.y[n], 0.0)

    model.bound_y = Constraint(model.Locations, model.Customers, rule=rule)

    def rule(model):
        return (sum(model.y[n] for n in model.Locations) - model.P, 0.0)

    model.num_facilities = Constraint(rule=rule)

    return model
