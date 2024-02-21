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


from pyomo.core import *

model = AbstractModel()
#
# Parameter N
#
model.N = Param(within=Integers)
#
# Set I
#
model.I = RangeSet(1, model.N)
#
# Variable b
#
model.b = Var(model.I, domain=Boolean)


#
# Objective zot
#
def costrule(model):
    ans = 0
    for i in model.I:
        #               ans += (-1 - .02*i)*model.b[i]
        ans += (1 + 0.02 * i) * model.b[i]
    return ans


# model.zot = Objective(rule=costrule)
model.zot = Objective(rule=costrule, sense=maximize)


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


model.w_ind = Set(initialize=w_ind_rule, dimen=2)
#
# Parameter w
#
model.w = Param(model.w_ind)
#
# Set rhs_ind
#
model.rhs_ind = RangeSet(1, model.N - 9)
#
# Parameter rhs
#
model.rhs = Param(model.rhs_ind)


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


model.bletch = Constraint(model.rhs_ind, rule=bletch_rule)
