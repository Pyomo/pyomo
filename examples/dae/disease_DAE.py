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

###
# SIR disease model using radau collocation
###

# Scaling in Continuous Set
from pyomo.environ import *
from pyomo.dae import *

years = 20
beta_py = 26
fepr = 1
fepy = beta_py * fepr
fe = fepy * years
step = 365.0 / fepy
model = AbstractModel()

# Define unindexed parameters
model.P_GAMMA = Param(default=1.0 / 14.0)
model.P_NUM_BETA = Param(default=beta_py)
model.P_FEPY = Param(default=fepy)
model.P_FE = Param(default=fe)
model.P_STEP = Param(default=step)
model.P_TRI = Param(default=beta_py * years)
model.P_FEPR = Param(default=fepr)

model.I_OBJ_WT = Param(default=0.995)
model.PHI_OBJ_WT = Param(default=0.005)

# Define sets
model.S_BETA = RangeSet(1, model.P_NUM_BETA)
model.S_FE = RangeSet(1, model.P_FE)


def _TIME_init(model):
    return (model.P_STEP * i for i in model.S_FE)


model.TIME = ContinuousSet(initialize=_TIME_init, bounds=(0, None))

model.S_TRI = RangeSet(1, model.P_TRI)

# Define indexed parameters
beta_ndx = {}
if (beta_py > 26) or (fepr > 1):
    for i in range(1, fe + 1):
        beta_ndx[i] = (((i + 1) / fepr) - 1) % beta_py + 1
else:
    for i in range(1, fe + 1):
        beta_ndx[i] = ((i - 1) % beta_py) + 1
model.P_BETA_NDX = Param(model.S_FE, initialize=beta_ndx, default=1.0)

model.P_POP = Param(default=1.0e6)
model.P_REP_FRAC = Param(model.S_TRI, default=0.4)
model.P_REP_CASES = Param(model.S_TRI, default=10.0)
model.P_BIRTHS = Param(model.S_FE, default=100.0)
model.P_DATA_WTS = Param(model.S_TRI, default=1.0)
model.P_ALL_CASES = Param(model.S_TRI, default=10.0)

# Define initialization parameters and rules
model.init_S_bar = Param(default=1.0e5)
model.init_beta_bar = Param(default=1.0)
model.init_I_init = Param(default=10.0)
model.init_S_init = Param(default=1000.0)
model.init_beta = Param(model.S_BETA, default=1.0)
model.init_beta_pos = Param(model.S_BETA, default=0.0)
model.init_beta_neg = Param(model.S_BETA, default=0.0)
model.init_eps_I = Param(model.S_FE, default=0.0)
model.init_eps_phi = Param(model.S_TRI, default=0.0)

model.init_S = Param(model.S_FE, default=100.0)
model.init_I = Param(model.S_FE, default=10.0)
model.init_phi = Param(model.S_FE, default=0.0)
model.init_Sdot = Param(model.S_FE, default=1.0)
model.init_Idot = Param(model.S_FE, default=1.0)
model.init_phidot = Param(model.S_FE, default=1.0)

model.init_beta_patt = Param(model.S_BETA, default=0.0)
model.init_beta_int = Param(default=1.0)


def _init_S_bar(model):
    return model.init_S_bar


def _init_beta_bar(model):
    return model.init_beta_bar


def _init_I_init(model):
    return model.init_I_init


def _init_S_init(model):
    return model.init_S_init


def _init_beta(model, i):
    return model.init_beta[i]


def _init_beta_pos(model, i):
    return model.init_beta_pos[i]


def _init_beta_neg(model, i):
    return model.init_beta_neg[i]


def _init_eps_I(model, i):
    return model.init_eps_I[i]


def _init_eps_phi(model, i):
    return model.init_eps_phi[i]


def _init_S(model, i):
    if i == 0:
        return model.init_S_init
    fe = model.TIME.get_upper_element_boundary(i)
    j = model.TIME._fe.index(fe)
    return model.init_S[j]


def _init_I(model, i):
    if i == 0:
        return model.init_I_init
    fe = model.TIME.get_upper_element_boundary(i)
    j = model.TIME._fe.index(fe)
    return model.init_I[j]


def _init_phi(model, i):
    if i == 0:
        return 0
    fe = model.TIME.get_upper_element_boundary(i)
    j = model.TIME._fe.index(fe)
    return model.init_phi[j]


def _init_Sdot(model, i):
    if i == 0:
        return 1
    fe = model.TIME.get_upper_element_boundary(i)
    j = model.TIME._fe.index(fe)
    return model.P_STEP * model.init_Sdot[j]


def _init_Idot(model, i):
    if i == 0:
        return 1
    fe = model.TIME.get_upper_element_boundary(i)
    j = model.TIME._fe.index(fe)
    return model.P_STEP * model.init_Idot[j]


def _init_phidot(model, i):
    if i == 0:
        return 1
    fe = model.TIME.get_upper_element_boundary(i)
    j = model.TIME._fe.index(fe)
    return model.P_STEP * model.init_phidot[j]


def _init_beta_patt(model, i):
    return model.init_beta_patt[i]


def _init_beta_int(model):
    return model.init_beta_int


def _people_bounds(model, i):
    return (0.0, model.P_POP)


def _init_people_bounds(model):
    return (0.0, model.P_POP)


# Define unindexed variables
model.S_bar = Var(initialize=_init_S_bar, bounds=(0, None))
model.beta_bar = Var(initialize=_init_beta_bar, bounds=(0.05, 5))
model.I_init = Var(initialize=_init_I_init, bounds=_init_people_bounds)
model.S_init = Var(initialize=_init_S_init, bounds=_init_people_bounds)
model.phi_init = Param(default=0.0)

# Define indexed variables
model.beta = Var(model.S_BETA, initialize=_init_beta, bounds=(0.01, 5))

model.beta_pos = Var(model.S_BETA, initialize=_init_beta_pos, bounds=(0, None))
model.beta_neg = Var(model.S_BETA, initialize=_init_beta_neg, bounds=(0, None))
model.beta_patt = Var(model.S_BETA, initialize=_init_beta_patt, bounds=(-5, 5))
model.beta_int = Var(initialize=_init_beta_int, bounds=(0.01, 5.0))
model.beta_c = Var(initialize=1.0)
model.alpha = Var(initialize=0.05, bounds=(-1.0, 1.0))

model.eps_I = Var(model.S_FE, initialize=_init_eps_I)
model.eps_phi = Var(model.S_TRI, initialize=_init_eps_phi)

model.S = Var(model.TIME, initialize=_init_S, bounds=_people_bounds)
model.I = Var(model.TIME, initialize=_init_I, bounds=_people_bounds)
model.phi = Var(model.TIME, initialize=_init_phi, bounds=(0, None))
model.Sdot = DerivativeVar(model.S, initialize=_init_Sdot)
model.Idot = DerivativeVar(model.I, initialize=_init_Idot)
model.phidot = DerivativeVar(model.phi, initialize=_init_phidot, bounds=(-10, None))


def _obj_rule(model):
    return model.I_OBJ_WT * sum(
        model.eps_I[i] ** 2 for i in model.S_FE
    ) + model.PHI_OBJ_WT * sum(
        model.P_DATA_WTS[i] * model.eps_phi[i] ** 2 for i in model.S_TRI
    )


model.obj = Objective(rule=_obj_rule)


########################
# Initial Conditions
########################
def _init_conditions(model):
    yield model.I[0] == model.I_init
    yield model.S[0] == model.S_init
    yield model.phi[0] == model.phi_init


model.init_conditions = ConstraintList(rule=_init_conditions)


def _reported_cases(model, i):
    if i == 1:
        if model.P_DATA_WTS[i] > 0.1:
            return (
                model.P_REP_CASES[i]
                == model.P_REP_FRAC[i]
                * (model.phi[model.TIME._fe[i * model.P_FEPR]] - model.phi_init)
                + model.eps_phi[i]
            )
        else:
            return Constraint.Skip
    else:
        if model.P_DATA_WTS[i] > 0.1:
            return (
                model.P_REP_CASES[i]
                == model.P_REP_FRAC[i]
                * (
                    model.phi[model.TIME._fe[i * model.P_FEPR]]
                    - model.phi[model.TIME._fe[(i - 1) * model.P_FEPR]]
                )
                + model.eps_phi[i]
            )
        else:
            return Constraint.Skip


model.con_reported_cases = Constraint(model.S_TRI, rule=_reported_cases)


def _beta_bar(model):
    return (
        model.beta_bar,
        sum(model.beta[i] for i in model.S_BETA) / len(model.S_BETA),
    )


model.con_beta_bar = Constraint(rule=_beta_bar)


def _phidot_eq(model, i):
    if i == 0:
        return Constraint.Skip
    fe = model.TIME.get_upper_element_boundary(i)
    j = model.TIME._fe.index(fe)
    return (
        model.phidot[i]
        == model.eps_I[j]
        + (model.beta[model.P_BETA_NDX[j]] * model.I[i] * model.S[i]) / model.P_POP
    )


model.phidot_eq = Constraint(model.TIME, rule=_phidot_eq)


def _Idot_eq(model, i):
    if i == 0:
        return Constraint.Skip
    return model.Idot[i] == model.phidot[i] - model.P_GAMMA * model.I[i]


model.Idot_eq = Constraint(model.TIME, rule=_Idot_eq)


def _Sdot_eq(model, i):
    if i == 0:
        return Constraint.Skip
    fe = model.TIME.get_upper_element_boundary(i)
    j = model.TIME._fe.index(fe)
    return model.Sdot[i] == -model.phidot[i] + model.P_BIRTHS[j]


model.Sdot_eq = Constraint(model.TIME, rule=_Sdot_eq)


def _scaled_beta(model, i):
    return (model.beta[i], model.beta_c * model.beta_patt[i])


model.con_city_varying_beta = Constraint(model.S_BETA, rule=_scaled_beta)


def _mean_patt(model):
    return (1.0, sum_product(model.beta_patt) / len(model.S_BETA))


model.con_mean_patt = Constraint(rule=_mean_patt)


def _beta_c(model):
    return (0.75, model.beta_c, 1.5)


model.con_beta_c = Constraint(rule=_beta_c)
