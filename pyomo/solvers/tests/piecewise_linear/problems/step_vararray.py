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
    Piecewise,
    Constraint,
    maximize,
    sum_product,
)

INDEX = [1, 2, 3, 4]
DOMAIN_PTS = [0, 1, 1, 2, 3]
F = [0, 0, 1, 1, 2]


def define_model(**kwds):
    model = ConcreteModel()

    model.x = Var(INDEX)  # domain variable

    model.Fx = Var(INDEX)  # range variable

    model.obj = Objective(
        expr=sum_product(model.Fx) + sum_product(model.x),
        sense=kwds.pop('sense', maximize),
    )

    model.piecewise = Piecewise(
        INDEX,
        model.Fx,
        model.x,
        pw_pts=DOMAIN_PTS,
        f_rule=F,
        unbounded_domain_var=True,
        **kwds
    )

    # Fix the answer for testing purposes
    model.set_answer_constraint1 = Constraint(
        expr=model.x[1] == 0.5
    )  # Fx1 should solve to 0
    model.set_answer_constraint2 = Constraint(expr=model.x[2] == 1.0)  #
    model.set_answer_constraint3 = Constraint(expr=model.Fx[2] == 0.5)  #
    model.set_answer_constraint4 = Constraint(
        expr=model.x[3] == 1.5
    )  # Fx3 should solve to 1
    model.set_answer_constraint5 = Constraint(
        expr=model.x[4] == 2.5
    )  # Fx4 should solve to 1.5

    return model
