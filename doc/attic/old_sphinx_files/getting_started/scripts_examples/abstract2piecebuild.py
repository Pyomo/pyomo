# abstract2piecebuild.py
# Similar to abstract2piece.py, but the breakpoints are created using a build action

from pyomo.environ import *

model = AbstractModel()

model.I = Set()
model.J = Set()

model.a = Param(model.I, model.J)
model.b = Param(model.I)
model.c = Param(model.J)

model.Topx = Param(default=6.1) # range of x variables
model.PieceCnt = Param(default=100)

# the next line declares a variable indexed by the set J
model.x = Var(model.J, domain=NonNegativeReals, bounds=(0,model.Topx))
model.y = Var(model.J, domain=NonNegativeReals)

# to avoid warnings, we set breakpoints beyond the bounds
# we are using a dictionary so that we can have different
# breakpoints for each index. But we won't.
model.bpts = {}
def bpts_build(model, j):
    model.bpts[j] = []
    for i in range(model.PieceCnt+2):
        model.bpts[j].append(float((i*model.Topx)/model.PieceCnt))
# The object model.BuildBpts is not refered to again; 
# the only goal is to trigger the action at build time
model.BuildBpts = BuildAction(model.J, rule=bpts_build)

def f4(model, j, xp):
    # we not need j in this example, but it is passed as the index for the constraint
    return xp**4

model.ComputePieces = Piecewise(model.J, model.y, model.x, pw_pts=model.bpts, pw_constr_type='EQ', f_rule=f4)

def obj_expression(model):
    return summation(model.c, model.y)

model.OBJ = Objective(rule=obj_expression)

def ax_constraint_rule(model, i):
    # return the expression for the constraint for i
    return sum(model.a[i,j] * model.x[j] for j in model.J) >= model.b[i]

# the next line creates one constraint for each member of the set model.I
model.AxbConstraint = Constraint(model.I, rule=ax_constraint_rule)
