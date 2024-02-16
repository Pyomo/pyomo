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
       / 2x          , 0 <= x <= 1
       | (1/2)x+(3/2), 1 <= x <= 3
f(x) = | -3x+12      , 3 <= x <= 5
       \ 2x-13       , 5 <= x <= 6
"""

from pyomo.core import (
    ConcreteModel,
    Var,
    Objective,
    Param,
    Piecewise,
    Constraint,
    maximize,
    sum_product,
)

INDEX_SET1 = ['1', '2', '3', '40']  # There will be two copies of this function
INDEX_SET2 = [(t1, t2) for t1 in range(1, 4) for t2 in range(1, 5)]
DOMAIN_PTS = dict(
    [
        ((t1, t2, t3), [float(i) for i in [0, 1, 3, 5, 6]])
        for t1 in INDEX_SET1
        for (t2, t3) in INDEX_SET2
    ]
)
RANGE_PTS = {0.0: 0.0, 1.0: 2.0, 3.0: 3.0, 5.0: -3.0, 6.0: -1.0}


def F(model, t1, t2, t3, x):
    return RANGE_PTS[x] * model.p[t1, t2, t3]


def define_model(**kwds):
    model = ConcreteModel()

    model.x = Var(INDEX_SET1, INDEX_SET2, bounds=(0, 6))  # domain variable
    model.Fx = Var(INDEX_SET1, INDEX_SET2)  # range variable
    model.p = Param(INDEX_SET1, INDEX_SET2, initialize=1.0)

    model.obj = Objective(expr=sum_product(model.Fx), sense=kwds.pop('sense', maximize))

    model.piecewise = Piecewise(
        INDEX_SET1, INDEX_SET2, model.Fx, model.x, pw_pts=DOMAIN_PTS, f_rule=F, **kwds
    )

    # Fix the answer for testing purposes
    model.set_answer_constraint1 = Constraint(
        INDEX_SET2, rule=lambda model, t2, t3: model.x['1', t2, t3] == 0.0
    )
    model.set_answer_constraint2 = Constraint(
        INDEX_SET2, rule=lambda model, t2, t3: model.x['2', t2, t3] == 3.0
    )
    model.set_answer_constraint3 = Constraint(
        INDEX_SET2, rule=lambda model, t2, t3: model.x['3', t2, t3] == 5.5
    )
    model.set_answer_constraint4 = Constraint(
        INDEX_SET2, rule=lambda model, t2, t3: model.x['40', t2, t3] == 6.0
    )

    return model
