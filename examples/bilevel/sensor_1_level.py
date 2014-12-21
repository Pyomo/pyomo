#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________


from pyomo.core import *
import math
import sys

model = AbstractModel()

model.V = Param(within=PositiveIntegers)
model.S = Param(within=PositiveIntegers)
model.r0 = Param(within=PositiveIntegers)
model.M = Param(within=Reals)

model.VERTICES = RangeSet(1, model.V)
model.SCENARIOS = RangeSet(1, model.S)

model.SCENARIO_VERTICES = Set(within=model.SCENARIOS * model.VERTICES)
model.s = Param(model.SCENARIO_VERTICES)
model.v = Param(model.SCENARIO_VERTICES)
model.p = Param(model.SCENARIO_VERTICES)
model.theta = Param(model.SCENARIOS)
model.x_tilde = Param(model.VERTICES)
model.r = Param(model.VERTICES)


# define variables
model.d=Var(model.VERTICES, within=Binary)
model.alpha=Var(model.SCENARIOS, within=Reals)
model.beta=Var(model.SCENARIO_VERTICES, within=NonPositiveReals)
model.beta_hat=Var(model.SCENARIO_VERTICES, within=NonPositiveReals)
model.gamma=Var(model.SCENARIO_VERTICES, within=NonPositiveReals)

# define objective function
def obj_rule(model):
    return sum(model.alpha[j] for j in model.SCENARIOS) + \
           sum(model.x_tilde[sv[1]] * (model.beta[sv] - model.beta_hat[sv]) for sv in model.SCENARIO_VERTICES) +\
           sum(model.gamma[sv] for sv in model.SCENARIO_VERTICES)
model.obj = Objective(rule=obj_rule, sense=maximize)

# sensor interdiction budget constraint
def budget_rule(model):
    return sum(model.r[v] * model.d[v] for v in model.VERTICES) <= model.r0
model.budget = Constraint(rule=budget_rule)

# w_variable constraints
def w_rule(model, s,v):
    return model.alpha[s] + model.beta[s,v] + model.gamma[s,v] <= model.theta[s] * model.p[s,v]
model.w_const = Constraint(model.SCENARIO_VERTICES, rule=w_rule)

# linearization constraint set 1
def lin_rule1(model, s, v):
    return model.beta_hat[s,v] >= model.beta[s,v] - model.M * (1-model.d[v])
model.lin_const1 = Constraint(model.SCENARIO_VERTICES, rule=lin_rule1)

# linearization constraint set 2
def lin_rule2(model, s, v):
    return model.beta_hat[s,v] >= -model.M * model.d[v]
model.lin_const2 = Constraint(model.SCENARIO_VERTICES, rule=lin_rule2)

# linearization constraint set 3
def lin_rule3(model, s, v):
    return model.beta_hat[s,v] <= model.beta[s,v] + model.M * (1-model.d[v])
model.lin_const3 = Constraint(model.SCENARIO_VERTICES, rule=lin_rule3)

