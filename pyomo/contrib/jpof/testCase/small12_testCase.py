#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Author:  Gabe Hackebeil
# Purpose: For testing to ensure that the Pyomo NL writer properly
#          handles the Expr_if component.
#
#          This test model relies on the gjh_asl_json executable. It
#          will not solve if sent to a real optimizer.
#

from pyomo.environ import ConcreteModel, Var, Param, Objective, Constraint, inequality
from pyomo.core.expr.current import Expr_if


model = ConcreteModel()

model.vTrue  = Var(initialize=1)
model.vFalse = Var(initialize=-1)

model.pTrue  = Param(initialize=1)
model.pFalse = Param(initialize=-1)

model.vN1 = Var(initialize=-1)
model.vP1 = Var(initialize=1)
model.v0  = Var(initialize=0)
model.vN2 = Var(initialize=-2)
model.vP2 = Var(initialize=2)

model.obj = Objective(expr=10.0*Expr_if(IF=model.v0,
                                            THEN=model.vTrue,
                                            ELSE=model.vFalse))

# True/False
model.c1 = Constraint(expr= Expr_if(IF=(0), THEN=(model.vTrue), ELSE=(model.vFalse)) == model.pFalse)
model.c2 = Constraint(expr= Expr_if(IF=(1), THEN=(model.vTrue), ELSE=(model.vFalse)) == model.pTrue)

# x <= 0
model.c3 = Constraint(expr= Expr_if(IF=(model.vN1 <= 0), THEN=(model.vTrue), ELSE=(model.vFalse)) == model.pTrue)
model.c4 = Constraint(expr= Expr_if(IF=(model.v0  <= 0), THEN=(model.vTrue), ELSE=(model.vFalse)) == model.pTrue)
model.c5 = Constraint(expr= Expr_if(IF=(model.vP1 <= 0), THEN=(model.vTrue), ELSE=(model.vFalse)) == model.pFalse)

# x < 0
model.c6 = Constraint(expr= Expr_if(IF=(model.vN1 < 0), THEN=(model.vTrue), ELSE=(model.vFalse)) == model.pTrue)
model.c7 = Constraint(expr= Expr_if(IF=(model.v0  < 0), THEN=(model.vTrue), ELSE=(model.vFalse)) == model.pFalse)
model.c8 = Constraint(expr= Expr_if(IF=(model.vP1 < 0), THEN=(model.vTrue), ELSE=(model.vFalse)) == model.pFalse)

# x >= 0
model.c9 = Constraint(expr= Expr_if(IF=(model.vN1*10.0 >= 0), THEN=(model.vTrue), ELSE=(model.vFalse))  == model.pFalse)
model.c10 = Constraint(expr= Expr_if(IF=(model.v0*10.0  >= 0), THEN=(model.vTrue), ELSE=(model.vFalse)) == model.pTrue)
model.c11 = Constraint(expr= Expr_if(IF=(model.vP1*10.0 >= 0), THEN=(model.vTrue), ELSE=(model.vFalse)) == model.pTrue)

# x > 0
model.c12 = Constraint(expr= Expr_if(IF=(model.vN1*10.0 > 0), THEN=(model.vTrue), ELSE=(model.vFalse)) == model.pFalse)
model.c13 = Constraint(expr= Expr_if(IF=(model.v0*10.0  > 0), THEN=(model.vTrue), ELSE=(model.vFalse)) == model.pFalse)
model.c14 = Constraint(expr= Expr_if(IF=(model.vP1*10.0 > 0), THEN=(model.vTrue), ELSE=(model.vFalse)) == model.pTrue)

# -1 <= x <= 1
model.c15 = Constraint(expr= Expr_if(IF=inequality(-1,              model.vN2, 1), THEN=(model.vTrue), ELSE=(model.vFalse)) == model.pFalse)
model.c16 = Constraint(expr= Expr_if(IF=inequality(-1*model.vP1,    model.vN1, 1), THEN=(model.vTrue), ELSE=(model.vFalse)) == model.pTrue)
model.c17 = Constraint(expr= Expr_if(IF=inequality(-1*model.vP1**2, model.v0,  1), THEN=(model.vTrue), ELSE=(model.vFalse)) == model.pTrue)
model.c18 = Constraint(expr= Expr_if(IF=inequality(model.vN1,       model.vP1, 1), THEN=(model.vTrue), ELSE=(model.vFalse)) == model.pTrue)
model.c19 = Constraint(expr= Expr_if(IF=inequality(-1,              model.vP2, 1), THEN=(model.vTrue), ELSE=(model.vFalse)) == model.pFalse)

# -1 < x < 1
model.c20 = Constraint(expr= Expr_if(IF=inequality(-1, model.vN2, 1, strict=True), THEN=(model.vTrue), ELSE=(model.vFalse)) == model.pFalse)
model.c21 = Constraint(expr= Expr_if(IF=inequality(-1, model.vN1, 1*model.vP1, strict=True), THEN=(model.vTrue), ELSE=(model.vFalse)) == model.pFalse)
model.c22 = Constraint(expr= Expr_if(IF=inequality(-1, model.v0,  1*model.vP1**2, strict=True), THEN=(model.vTrue), ELSE=(model.vFalse)) == model.pTrue)
model.c23 = Constraint(expr= Expr_if(IF=inequality(-1, model.vP1, model.vP1, strict=True), THEN=(model.vTrue), ELSE=(model.vFalse)) == model.pFalse)
model.c24 = Constraint(expr= Expr_if(IF=inequality(-1, model.vP2, 1, strict=True) , THEN=(model.vTrue), ELSE=(model.vFalse)) == model.pFalse)
