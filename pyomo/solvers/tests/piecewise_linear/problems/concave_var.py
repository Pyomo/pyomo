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
       / 9x+20 , -5 <= x <= -4
       | 7x+12 , -4 <= x <= -3
       | 5x+6  , -3 <= x <= -2
       | 3x+2  , -2 <= x <= -1
f(x) = | 1     , -1 <= x <=  1 
       | -3x+2 ,  1 <= x <=  2
       | -5x+6 ,  2 <= x <=  3
       \ -7x+12,  3 <= x <=  4
"""

from pyomo.core import ConcreteModel, Var, Objective, Piecewise, Constraint, maximize

DOMAIN_PTS = [float(i) for i in (list(range(-5, 0)) + list(range(1, 5)))]


def F(model, x):
    return -(x**2)


def define_model(**kwds):
    model = ConcreteModel()

    model.x1 = Var(bounds=(-5, 4))  # domain variable
    model.x2 = Var(bounds=(-5, 4))  # domain variable
    model.x3 = Var(bounds=(-5, 4))  # domain variable
    model.x4 = Var(bounds=(-5, 4))  # domain variable
    model.x5 = Var(bounds=(-5, 4))  # domain variable
    model.x6 = Var(bounds=(-5, 4))  # domain variable
    model.x7 = Var(bounds=(-5, 4))  # domain variable

    model.Fx1 = Var()  # range variable
    model.Fx2 = Var()  # range variable
    model.Fx3 = Var()  # range variable
    model.Fx4 = Var()  # range variable
    model.Fx5 = Var()  # range variable
    model.Fx6 = Var()  # range variable
    model.Fx7 = Var()  # range variable

    model.obj = Objective(
        expr=model.Fx1
        + model.Fx2
        + model.Fx3
        + model.Fx4
        + model.Fx5
        + model.Fx6
        + model.Fx7,
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
    model.piecewise5 = Piecewise(
        model.Fx5, model.x5, pw_pts=DOMAIN_PTS, f_rule=F, **kwds
    )
    model.piecewise6 = Piecewise(
        model.Fx6, model.x6, pw_pts=DOMAIN_PTS, f_rule=F, **kwds
    )
    model.piecewise7 = Piecewise(
        model.Fx7, model.x7, pw_pts=DOMAIN_PTS, f_rule=F, **kwds
    )

    # Fix the answer for testing purposes
    model.set_answer_constraint1 = Constraint(expr=model.x1 == -5.0)
    model.set_answer_constraint2 = Constraint(expr=model.x2 == -3.0)
    model.set_answer_constraint3 = Constraint(expr=model.x3 == -2.5)
    model.set_answer_constraint4 = Constraint(expr=model.x4 == -1.5)
    model.set_answer_constraint5 = Constraint(expr=model.x5 == 2.0)
    model.set_answer_constraint6 = Constraint(expr=model.x6 == 3.5)
    model.set_answer_constraint7 = Constraint(expr=model.x7 == 4.0)

    return model
