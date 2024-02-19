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

# Sample Problem 2: Parameter Estimation
# (Ex 5 from Dynopt Guide)
#
# 	min sum((X1(ti)-X1_meas(ti))^2)
# 	s.t.	X1_dot = X2				X1(0) = p1
# 			X2_dot = 1-2*X2-X1		X2(0) = p2
# 			-1.5 <= p1,p2 <= 1.5
# 			tf = 6
#

from pyomo.environ import *
from pyomo.dae import *

model = AbstractModel()
model.t = ContinuousSet()
model.MEAS_t = Set(within=model.t)  # Measurement times, must be subset of t
model.x1_meas = Param(model.MEAS_t)

model.x1 = Var(model.t)
model.x2 = Var(model.t)

model.p1 = Var(bounds=(-1.5, 1.5))
model.p2 = Var(bounds=(-1.5, 1.5))

model.x1dot = DerivativeVar(model.x1, wrt=model.t)
model.x2dot = DerivativeVar(model.x2)


def _init_conditions(model):
    yield model.x1[0] == model.p1
    yield model.x2[0] == model.p2


model.init_conditions = ConstraintList(rule=_init_conditions)

# Alternate way to declare initial conditions
# def _initx1(model):
# 	return model.x1[0] == model.p1
# model.initx1 = Constraint(rule=_initx1)

# def _initx2(model):
# 	return model.x2[0] == model.p2
# model.initx2 = Constraint(rule=_initx2)


def _x1dot(model, i):
    return model.x1dot[i] == model.x2[i]


model.x1dotcon = Constraint(model.t, rule=_x1dot)


def _x2dot(model, i):
    return model.x2dot[i] == 1 - 2 * model.x2[i] - model.x1[i]


model.x2dotcon = Constraint(model.t, rule=_x2dot)


def _obj(model):
    return sum((model.x1[i] - model.x1_meas[i]) ** 2 for i in model.MEAS_t)


model.obj = Objective(rule=_obj)
