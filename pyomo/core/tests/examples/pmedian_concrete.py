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

import math
from pyomo.environ import (
    ConcreteModel,
    Param,
    RangeSet,
    Var,
    Reals,
    Binary,
    PositiveIntegers,
)


def _cost_rule(model, n, m):
    # We will assume costs are an arbitrary function of the indices
    return math.sin(n * 2.33333 + m * 7.99999)


def create_model(n=3, m=3, p=2):
    model = ConcreteModel(name="M1")

    model.N = Param(initialize=n, within=PositiveIntegers)
    model.M = Param(initialize=m, within=PositiveIntegers)
    model.P = Param(initialize=p, within=RangeSet(1, model.N), mutable=True)

    model.Locations = RangeSet(1, model.N)
    model.Customers = RangeSet(1, model.M)

    model.cost = Param(
        model.Locations, model.Customers, initialize=_cost_rule, within=Reals
    )
    model.serve_customer_from_location = Var(
        model.Locations, model.Customers, bounds=(0.0, 1.0)
    )
    model.select_location = Var(model.Locations, within=Binary)

    @model.Objective()
    def obj(model):
        return sum(
            model.cost[n, m] * model.serve_customer_from_location[n, m]
            for n in model.Locations
            for m in model.Customers
        )

    @model.Constraint(model.Customers)
    def single_x(model, m):
        return (
            sum(model.serve_customer_from_location[n, m] for n in model.Locations)
            == 1.0
        )

    @model.Constraint(model.Locations, model.Customers)
    def bound_y(model, n, m):
        return model.serve_customer_from_location[n, m] <= model.select_location[n]

    @model.Constraint()
    def num_facilities(model):
        return sum(model.select_location[n] for n in model.Locations) == model.P

    return model
