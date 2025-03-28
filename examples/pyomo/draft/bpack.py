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


import pyomo.environ as pyo

model = pyo.AbstractModel()
#
# Parameter N
#
model.N = pyo.Param(within=pyo.Integers)
#
# Set I
#
model.I = pyo.RangeSet(1, model.N)
#
# Variable b
#
model.b = pyo.Var(model.I, domain=pyo.Boolean)


#
# Objective zot
#
def costrule(model):
    ans = 0
    for i in model.I:
        #               ans += (-1 - .02*i)*model.b[i]
        ans += (1 + 0.02 * i) * model.b[i]
    return ans


# model.zot = pyo.Objective(rule=costrule)
model.zot = pyo.Objective(rule=costrule, sense=pyo.maximize)


#
# Set w_ind
#
def w_ind_rule(model):
    ans = set()
    i = 1
    N9 = model.N.value - 9
    while i <= N9:
        j = i
        i9 = i + 9
        while j <= i9:
            ans.add((i, j))
            j += 1
        i += 1
    return ans


model.w_ind = pyo.Set(initialize=w_ind_rule, dimen=2)
#
# Parameter w
#
model.w = pyo.Param(model.w_ind)
#
# Set rhs_ind
#
model.rhs_ind = pyo.RangeSet(1, model.N - 9)
#
# Parameter rhs
#
model.rhs = pyo.Param(model.rhs_ind)


#
# Constraint bletch
#
def bletch_rule(model, i):
    ans = 0
    j = i
    i9 = i + 9
    while j <= i9:
        ans += model.w[i, j] * model.b[j]
        j += 1
    ans = ans < model.rhs[i]
    return ans


model.bletch = pyo.Constraint(model.rhs_ind, rule=bletch_rule)
