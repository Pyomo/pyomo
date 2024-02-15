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

from pyomo.core import *

model = AbstractModel()

model.S_SI = Set(ordered=True)

model.P_REP_CASES = Param(model.S_SI)
model.P_POP = Param()

model.I = Var(model.S_SI, bounds=(0, model.P_POP), initialize=1)
model.S = Var(model.S_SI, bounds=(0, model.P_POP), initialize=300)
model.beta = Var(bounds=(0.05, 70))
model.alpha = Var(bounds=(0.5, 1.5))
model.eps_I = Var(model.S_SI, initialize=0.0)


def _objective(model):
    return sum((model.eps_I[i]) ** 2 for i in model.S_SI)


model.objective = Objective(rule=_objective, sense=minimize)


def _InfDynamics(model, i):
    if i != 1:
        return (
            model.I[i]
            == (model.beta * model.S[i - 1] * model.I[i - 1] ** model.alpha)
            / model.P_POP
        )


model.InfDynamics = Constraint(model.S_SI, rule=_InfDynamics)


def _SusDynamics(model, i):
    if i != 1:
        return model.S[i] == model.S[i - 1] - model.I[i]


model.SusDynamics = Constraint(model.S_SI, rule=_SusDynamics)


def _Data(model, i):
    return model.P_REP_CASES[i] == model.I[i] + model.eps_I[i]


model.Data = Constraint(model.S_SI, rule=_Data)
