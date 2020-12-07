#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, log, log10, exp, sqrt

model = ConcreteModel()

# pick a value in the domain of all of these functions
model.ONE = Var(initialize=1)
model.ZERO = Var(initialize=0)


model.obj = Objective(expr=model.ONE+model.ZERO)

model.c_log     = Constraint(expr=log(model.ONE) == 0)
model.c_log10   = Constraint(expr=log10(model.ONE) == 0)

model.c_exp     = Constraint(expr=exp(model.ZERO) == 1)
model.c_sqrt    = Constraint(expr=sqrt(model.ONE) == 1)
model.c_abs     = Constraint(expr=abs(model.ONE) == 1)
