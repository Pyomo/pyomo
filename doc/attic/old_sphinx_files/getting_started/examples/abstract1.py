# abstract1.py
from __future__ import division
from pyomo.environ import *

# @Declare_abstract_model
model = AbstractModel()
# @Declare_abstract_model

# @Declare_param_within
model.m = Param(within=NonNegativeIntegers)
model.n = Param(within=NonNegativeIntegers)
# @Declare_param_within

# @Define_indexsets
model.I = RangeSet(1, model.m)
model.J = RangeSet(1, model.n)
# @Define_indexsets

# @Define_indexed_parameters
model.a = Param(model.I, model.J)
model.b = Param(model.I)
model.c = Param(model.J)
# @Define_indexed_parameters

# @Define_variable
# the next line declares a variable indexed by the set J
model.x = Var(model.J, domain=NonNegativeReals)
# @Define_variable

# @Define_objective_expression
def obj_expression(model):
    return summation(model.c, model.x)
# @Define_objective_expression

# @Declare_objective_function
model.OBJ = Objective(rule=obj_expression)
# @Declare_objective_function

# @Define_constraints_expression
def ax_constraint_rule(model, i):
    # return the expression for the constraint for i
    return sum(model.a[i,j] * model.x[j] for j in model.J) >= model.b[i]
# @Define_constraints_expression

# @Declare_constraints
# the next line creates one constraint for each member of the set model.I
model.AxbConstraint = Constraint(model.I, rule=ax_constraint_rule)
# @Declare_constraints
