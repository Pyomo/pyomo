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

r"""
A step function:

       / 0      , 0 <= x <= 1
f(x) = | 1      , 1 <  x <= 2
       \ x-1    , 2 <  x <= 3
"""

from pyomo.core import (
    ConcreteModel,
    Var,
    Objective,
    Param,
    Piecewise,
    Constraint,
    maximize,
)

DOMAIN_PTS = [0, 1, 1, 2, 3]
F = [0, 0, 1, 1, 2]


def define_model(**kwds):
    model = ConcreteModel()

    model.x1 = Var(bounds=(0, 3))  # domain variable
    model.x2 = Var(bounds=(0, 3))  # domain variable
    model.x3 = Var(bounds=(0, 3))  # domain variable
    model.x4 = Var(bounds=(0, 3))  # domain variable

    model.Fx1 = Var()  # range variable
    model.Fx2 = Var()  # range variable
    model.Fx3 = Var()  # range variable
    model.Fx4 = Var()  # range variable
    model.p = Param(initialize=1.0)

    model.obj = Objective(
        expr=model.Fx1
        + model.Fx2
        + model.Fx3
        + model.Fx4
        + model.x1
        + model.x2
        + model.x3
        + model.x4,
        sense=kwds.pop('sense', maximize),
    )

    model.piecewise1 = Piecewise(
        model.Fx1, model.x1, pw_pts=DOMAIN_PTS, f_rule=F, **kwds
    )
    model.piecewise2 = Piecewise(
        model.Fx2, model.x2, pw_pts=DOMAIN_PTS, f_rule=F, **kwds
    )

    model.piecewise3 = Piecewise(
        model.Fx3, model.x3, pw_pts=DOMAIN_PTS, f_rule=F, **kwds
    )

    model.piecewise4 = Piecewise(
        model.Fx4, model.x4, pw_pts=DOMAIN_PTS, f_rule=F, **kwds
    )

    # Fix the answer for testing purposes
    model.set_answer_constraint1 = Constraint(
        expr=model.x1 == 0.5
    )  # Fx1 should solve to 0
    model.set_answer_constraint2 = Constraint(expr=model.x2 == 1.0)  #
    model.set_answer_constraint3 = Constraint(expr=model.Fx2 == 0.5)  #
    model.set_answer_constraint4 = Constraint(
        expr=model.x3 == 1.5
    )  # Fx3 should solve to 1
    model.set_answer_constraint5 = Constraint(
        expr=model.x4 == 2.5
    )  # Fx4 should solve to 1.5

    return model
