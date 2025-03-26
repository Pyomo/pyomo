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

#
# This is a performance test that we cannot easily execute right now
#
import pyomo.environ as pyo


def f(N):
    M = pyo.ConcreteModel()
    M.A = pyo.Set(initialize=range(N))
    M.x = pyo.Var()
    M.o = pyo.Objective(expr=M.x)

    def rule(m, i):
        if i == 3 or i == 5:
            return M.x >= i
        return pyo.Constraint.Skip

    M.c = pyo.Constraint(M.A, rule=rule)
    return M


#
# Generation of this model is slow because set M.A is big
#
model = f(1000000)
