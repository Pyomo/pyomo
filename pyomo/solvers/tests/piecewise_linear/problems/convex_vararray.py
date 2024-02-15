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
       / -9x-20, -5 <= x <= -4
       | -7x-12, -4 <= x <= -3
       | -5x-6 , -3 <= x <= -2
       | -3x-2 , -2 <= x <= -1
f(x) = | 1     , -1 <= x <=  1 
       | 3x-2  ,  1 <= x <=  2
       | 5x-6  ,  2 <= x <=  3
       \ 7x-12 ,  3 <= x <=  4

"""

from pyomo.environ import (
    ConcreteModel,
    Var,
    Param,
    Piecewise,
    Constraint,
    Objective,
    sum_product,
    maximize,
)

INDEX_SET = range(1, 8)  # There will be two copies of this function
DOMAIN_PTS = dict(
    [
        (t, [float(i) for i in (list(range(-5, 0)) + list(range(1, 5)))])
        for t in INDEX_SET
    ]
)


def F(model, t, x):
    return (x**2) * model.p[t]


def define_model(**kwds):
    model = ConcreteModel()

    model.x = Var(INDEX_SET, bounds=(-5, 4))  # domain variable
    model.Fx = Var(INDEX_SET)  # range variable
    model.p = Param(INDEX_SET, initialize=1.0)

    model.obj = Objective(expr=sum_product(model.Fx), sense=kwds.pop('sense', maximize))

    model.piecewise = Piecewise(
        INDEX_SET, model.Fx, model.x, pw_pts=DOMAIN_PTS, f_rule=F, **kwds
    )

    # Fix the answer for testing purposes
    model.set_answer_constraint1 = Constraint(expr=model.x[1] == -5.0)
    model.set_answer_constraint2 = Constraint(expr=model.x[2] == -3.0)
    model.set_answer_constraint3 = Constraint(expr=model.x[3] == -2.5)
    model.set_answer_constraint4 = Constraint(expr=model.x[4] == -1.5)
    model.set_answer_constraint5 = Constraint(expr=model.x[5] == 2.0)
    model.set_answer_constraint6 = Constraint(expr=model.x[6] == 3.5)
    model.set_answer_constraint7 = Constraint(expr=model.x[7] == 4.0)

    return model
