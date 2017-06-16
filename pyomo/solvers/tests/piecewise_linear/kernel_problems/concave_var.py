#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.kernel as pmo

breakpoints = list(range(-5,0))+list(range(1,5))
values = [-x**2 for x in breakpoints]

def define_model(**kwds):

    sense = kwds.pop("sense")

    m = pmo.block()

    m.x = pmo.variable_list()
    m.Fx = pmo.variable_list()
    m.piecewise = pmo.block_list()
    for i in range(7):
        m.x.append(pmo.variable(lb=-5, ub=4))
        m.Fx.append(pmo.variable())
        m.piecewise.append(
            pmo.piecewise(breakpoints, values,
                          input=m.x[i],
                          output=m.Fx[i],
                          **kwds))

    m.obj = pmo.objective(expr=sum(m.Fx),
                          sense=sense)

    # fix the answer for testing purposes
    m.set_answer = pmo.constraint_list()
    m.set_answer.append(pmo.constraint(m.x[0] == -5.0))
    m.set_answer.append(pmo.constraint(m.x[1] == -3.0))
    m.set_answer.append(pmo.constraint(m.x[2] == -2.5))
    m.set_answer.append(pmo.constraint(m.x[3] == -1.5))
    m.set_answer.append(pmo.constraint(m.x[4] == 2.0))
    m.set_answer.append(pmo.constraint(m.x[5] == 3.5))
    m.set_answer.append(pmo.constraint(m.x[6] == 4.0))

    return m
