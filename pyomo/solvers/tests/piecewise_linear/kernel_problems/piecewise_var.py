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

from pyomo.kernel import (
    block,
    variable,
    variable_list,
    block_list,
    piecewise,
    objective,
    constraint,
    constraint_list,
)

breakpoints = [0, 1, 3, 5, 6]
values = [0, 2, 3, -3, -1]


def define_model(**kwds):
    sense = kwds.pop("sense")

    m = block()

    m.x = variable_list()
    m.Fx = variable_list()
    m.piecewise = block_list()
    for i in range(4):
        m.x.append(variable(lb=0, ub=6))
        m.Fx.append(variable())
        m.piecewise.append(
            piecewise(breakpoints, values, input=m.x[i], output=m.Fx[i], **kwds)
        )

    m.obj = objective(expr=sum(m.Fx), sense=sense)

    # fix the answer for testing purposes
    m.set_answer = constraint_list()
    m.set_answer.append(constraint(m.x[0] == 0.0))
    m.set_answer.append(constraint(m.x[1] == 3.0))
    m.set_answer.append(constraint(m.x[2] == 5.5))
    m.set_answer.append(constraint(m.x[3] == 6.0))

    return m
