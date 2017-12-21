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
# Purpose: For regression testing to ensure that the Pyomo
#          NL writer properly reports the values corresponding
#          to the nl file header line with the label
#          '# nonlinear vars in constraints, objectives, both'
#

from pyomo.environ import *

model = ConcreteModel()

model.x = Var()

model.OBJ = Objective(expr=model.x)

model.c_log     = Constraint(expr=log(model.x) == 0)
model.c_log10   = Constraint(expr=log10(model.x) == 0)

model.c_sin     = Constraint(expr=sin(model.x) == 0)
model.c_cos     = Constraint(expr=cos(model.x) == 0)
model.c_tan     = Constraint(expr=tan(model.x) == 0)

model.c_sinh    = Constraint(expr=sinh(model.x) == 0)
model.c_cosh    = Constraint(expr=cosh(model.x) == 0)
model.c_tanh    = Constraint(expr=tanh(model.x) == 0)

model.c_asin    = Constraint(expr=asin(model.x) == 0)
model.c_acos    = Constraint(expr=acos(model.x) == 0)
model.c_atan    = Constraint(expr=atan(model.x) == 0)

model.c_asinh   = Constraint(expr=asinh(model.x) == 0)
model.c_acosh   = Constraint(expr=acosh(model.x) == 0)
model.c_atanh   = Constraint(expr=atanh(model.x) == 0)

model.c_exp     = Constraint(expr=exp(model.x) == 0)
model.c_sqrt    = Constraint(expr=sqrt(model.x) == 0)
model.c_ceil    = Constraint(expr=ceil(model.x) == 0)
model.c_floor   = Constraint(expr=floor(model.x) == 0)
model.c_abs     = Constraint(expr=abs(model.x) == 0)

