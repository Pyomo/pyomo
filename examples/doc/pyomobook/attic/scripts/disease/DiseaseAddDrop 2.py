from pyomo.environ import *
from pyomo.opt import SolverFactory

model = AbstractModel()

model.S_SI = Set(ordered=True)

model.P_REP_CASES = Param(model.S_SI)
model.P_POP = Param()

model.I = Var(model.S_SI, bounds=(0,model.P_POP), initialize=1)
model.S = Var(model.S_SI, bounds=(0,model.P_POP), initialize=300)
model.beta = Var(bounds=(0.05, 70))
model.alpha = Var(bounds=(0.5, 1.5))
model.eps_I = Var(model.S_SI, initialize=0.0)

def _objective(model):
    return sum((model.eps_I[i])**2 for i in model.S_SI)
model.objective = Objective(rule=_objective, sense=minimize)

def _InfDynamics(model, i):
    if i != 1:
        return model.I[i] == (model.beta * model.S[i-1] * model.I[i-1]**model.alpha)/model.P_POP
    return Constraint.Skip
model.InfDynamics = Constraint(model.S_SI, rule=_InfDynamics)

def _EasierInfDynamics(model, i):
    if i != 1:
        return model.I[i] == (model.beta * model.S[i-1] * model.P_REP_CASES[i-1])/model.P_POP
    return Constraint.Skip
model.EasierInfDynamics = Constraint(model.S_SI, rule=_EasierInfDynamics)

def _SusDynamics(model, i):
    if i != 1:
        return model.S[i] == model.S[i-1] - model.I[i]
    return Constraint.Skip
model.SusDynamics = Constraint(model.S_SI, rule=_SusDynamics)

def _Data(model, i):
    return model.P_REP_CASES[i] == model.I[i]+model.eps_I[i]
model.Data = Constraint(model.S_SI, rule=_Data)

instance = model.create_instance('DiseaseEstimation.dat');

# disable the hard constraints
instance.InfDynamics.deactivate()

# solve the problem with the easy constraints
print("*** Solving the \"easy\" problem")
with SolverFactory("ipopt") as solver:
    solver.solve(instance)

print("beta for easy problem: %s" % (instance.beta.value))
print("alpha for easy problem: %s" % (instance.alpha.value))
print("")

# enable the hard constraints and disable the easy
# constraints
instance.InfDynamics.activate()
instance.EasierInfDynamics.deactivate()

# alpha does not appear in the easy problem
# so we must initialize it here
instance.alpha = 1.0

# solve the problem with the hard constraints using the
# current solution as the initial starting point for the
# solver
print("*** Solving the \"hard\" problem")
with SolverFactory("ipopt") as solver:
    solver.solve(instance)
print("beta for hard problem: %s" % (instance.beta.value))
print("alpha for hard problem: %s" % (instance.alpha.value))
