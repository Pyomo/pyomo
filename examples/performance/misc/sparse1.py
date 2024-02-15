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

#
# This is a performance test that we cannot easily execute right now
#
from pyomo.environ import *


def f(N):
    M = ConcreteModel()
    M.A = Set(initialize=range(N))
    M.x = Var()
    M.o = Objective(expr=M.x)

    def rule(m, i):
        if i == 3 or i == 5:
            return M.x >= i
        return Constraint.Skip

    M.c = Constraint(M.A, rule=rule)
    return M


#
# Generation of this model is slow because set M.A is big
#
model = f(1000000)
