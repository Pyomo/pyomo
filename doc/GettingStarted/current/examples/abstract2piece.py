# abstract2piece.py
# Similar to abstract2.py, but the objective is now c times x to the fourth power

from pyomo.environ import *

model = AbstractModel()

model.I = Set()
model.J = Set()

Topx = 6.1 # range of x variables

model.a = Param(model.I, model.J)
model.b = Param(model.I)
model.c = Param(model.J)

# the next line declares a variable indexed by the set J
model.x = Var(model.J, domain=NonNegativeReals, bounds=(0, Topx))
model.y = Var(model.J, domain=NonNegativeReals)

# to avoid warnings, we set breakpoints at or beyond the bounds
PieceCnt = 100
bpts = []
for i in range(PieceCnt+2):
    bpts.append(float((i*Topx)/PieceCnt))

def f4(model, j, xp):
    # we not need j, but it is passed as the index for the constraint
    return xp**4

model.ComputeObj = Piecewise(model.J, model.y, model.x, pw_pts=bpts, pw_constr_type='EQ', f_rule=f4)

def obj_expression(model):
    return sum_product(model.c, model.y)

model.OBJ = Objective(rule=obj_expression)

def ax_constraint_rule(model, i):
    # return the expression for the constraint for i
    return sum(model.a[i,j] * model.x[j] for j in model.J) >= model.b[i]

# the next line creates one constraint for each member of the set model.I
model.AxbConstraint = Constraint(model.I, rule=ax_constraint_rule)
