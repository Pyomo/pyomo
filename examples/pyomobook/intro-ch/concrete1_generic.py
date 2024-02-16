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

import pyomo.environ as pyo
import mydata

model = pyo.ConcreteModel()

model.x = pyo.Var(mydata.N, within=pyo.NonNegativeReals)


def obj_rule(model):
    return sum(mydata.c[i] * model.x[i] for i in mydata.N)


model.obj = pyo.Objective(rule=obj_rule)


def con_rule(model, m):
    return sum(mydata.a[m, i] * model.x[i] for i in mydata.N) >= mydata.b[m]


model.con = pyo.Constraint(mydata.M, rule=con_rule)
