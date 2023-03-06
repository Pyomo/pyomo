#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, log, log10, sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, asinh, acosh, atanh, exp, sqrt, ceil, floor
from math import e, pi

model = ConcreteModel()

# pick a value in the domain of all of these functions
model.ONE = Var(initialize=1)
model.ZERO = Var(initialize=0)


model.obj = Objective(expr=model.ONE+model.ZERO)

model.c_log     = Constraint(expr=log(model.ONE) == 0)
model.c_log10   = Constraint(expr=log10(model.ONE) == 0)

model.c_sin     = Constraint(expr=sin(model.ZERO) == 0)
model.c_cos     = Constraint(expr=cos(model.ZERO) == 1)
model.c_tan     = Constraint(expr=tan(model.ZERO) == 0)

model.c_sinh    = Constraint(expr=sinh(model.ZERO) == 0)
model.c_cosh    = Constraint(expr=cosh(model.ZERO) == 1)
model.c_tanh    = Constraint(expr=tanh(model.ZERO) == 0)

model.c_asin    = Constraint(expr=asin(model.ZERO) == 0)
model.c_acos    = Constraint(expr=acos(model.ZERO) == pi/2)
model.c_atan    = Constraint(expr=atan(model.ZERO) == 0)

model.c_asinh   = Constraint(expr=asinh(model.ZERO) == 0)
model.c_acosh   = Constraint(expr=acosh((e**2 + model.ONE)/(2*e)) == 0)
model.c_atanh   = Constraint(expr=atanh(model.ZERO) == 0)

model.c_exp     = Constraint(expr=exp(model.ZERO) == 1)
model.c_sqrt    = Constraint(expr=sqrt(model.ONE) == 1)
model.c_ceil    = Constraint(expr=ceil(model.ONE) == 1)
model.c_floor   = Constraint(expr=floor(model.ONE) == 1)
model.c_abs     = Constraint(expr=abs(model.ONE) == 1)
