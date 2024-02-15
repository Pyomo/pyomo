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

# disease_estimation.py
import pyomo.environ as pyo

model = pyo.AbstractModel()

model.S_SI = pyo.Set(ordered=True)

model.P_REP_CASES = pyo.Param(model.S_SI)
model.P_POP = pyo.Param()

model.I = pyo.Var(model.S_SI, bounds=(0, model.P_POP), initialize=1)
model.S = pyo.Var(model.S_SI, bounds=(0, model.P_POP), initialize=300)
model.beta = pyo.Var(bounds=(0.05, 70))
model.alpha = pyo.Var(bounds=(0.5, 1.5))
model.eps_I = pyo.Var(model.S_SI, initialize=0.0)


def _objective(model):
    return sum((model.eps_I[i]) ** 2 for i in model.S_SI)


model.objective = pyo.Objective(rule=_objective, sense=pyo.minimize)


def _InfDynamics(model, i):
    if i != 1:
        return (
            model.I[i]
            == (model.beta * model.S[i - 1] * model.I[i - 1] ** model.alpha)
            / model.P_POP
        )
    return pyo.Constraint.Skip


model.InfDynamics = pyo.Constraint(model.S_SI, rule=_InfDynamics)


def _SusDynamics(model, i):
    if i != 1:
        return model.S[i] == model.S[i - 1] - model.I[i]
    return pyo.Constraint.Skip


model.SusDynamics = pyo.Constraint(model.S_SI, rule=_SusDynamics)


def _Data(model, i):
    return model.P_REP_CASES[i] == model.I[i] + model.eps_I[i]


model.Data = pyo.Constraint(model.S_SI, rule=_Data)

# create the ConcreteModel
instance = model.create_instance('disease_estimation.dat')
status = pyo.SolverFactory('ipopt').solve(instance)
pyo.assert_optimal_termination(status)

print(' ***')
print(' *** Optimal beta Value: %.2f' % pyo.value(instance.beta))
print(' *** Optimal alpha Value: %.2f' % pyo.value(instance.alpha))
print(' ***')
