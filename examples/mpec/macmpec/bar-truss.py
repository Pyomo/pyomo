# bar-truss.py LQR2-MN-35-29
# Original Pyomo coding by William Hart
# Adapted from Original AMPL coding by Sven Leyffer

# From a GAMS model of a 3-bar truss min weight design problem
# by M.C. Ferris and F. Tin-Loi, "On the solution of a minimum
# weight elastoplastic problem involving displacement and
# complementarity constraints", Comp. Meth. in Appl. Mech & Engng,
# 174:107-120, 1999.

# Number of variables:   35
# Number of constraints: 28 + 1

import pyomo.environ
from pyomo.core import *
from pyomo.mpec import *


model = AbstractModel()
model.d = Set()             # No. of structure dof
model.m = Set()             # No. of members
model.y = Set()             # No. of yield functs per member

model.E = Param()           # Young's modulus
model.sigma = Param()       # Yield limit

model.L = Param(model.m)    # length of members
model.F = Param(model.d)
model.C = Param(model.m, model.d)
model.N = Param(model.m, model.y)

model.S = Var(model.m, initialize=1)
model.r = Var(model.m, model.y, initialize=1)
model.H = Var(model.m, model.y, model.y, initialize=1) # hardening parameters in tension  & compression
model.Q = Var(model.m)
model.u = Var(model.d, bounds=(-4,4))           # deflection
model.a = Var(model.m, bounds=(0,1))            # bar areas
model.z = Var(model.m, model.y, within=NonNegativeReals)
model.w = Var(model.m, model.y, bounds=(0,1))   # yield function

def volume_(model):
    return summation(model.L, model.a)
model.volume = Objective(rule=volume_)

def tech_(model, i):
    return model.a[i] == model.a['m1']
model.tech = Constraint(model.m, rule=tech_)

def stiff_(model, i):
    return model.S[i] - model.E*model.a[i] / model.L[i] == 0
model.stiff = Constraint(model.m, rule=stiff_)

def limit_(model, i, j):
    return model.r[i,j] - model.sigma* model.a[i] == 0
model.limit = Constraint(model.m, model.y, rule=limit_)

def hard_(model, i, j):
    return model.H[i,j,j] - 0.125*model.E*model.a[i]/model.L[i] == 0
model.hard = Constraint(model.m, model.y, rule=hard_)
  
def compat_(model, i):
    return - model.Q[i] + model.S[i]*sum(model.C[i,k] * model.u[k] for k in model.d) \
                        - model.S[i]*sum(model.N[i,j] * model.z[i,j] for j in model.y) == 0
model.compat = Constraint(model.m, rule=compat_)

def equil_(model, k):
    return sum(model.C[i,k] * model.Q[i] for i in model.m) - model.F[k] == 0
model.equil = Constraint(model.d, rule=equil_)

def yyield_(model, i, j):
    return - model.N[i,j]*model.Q[i] + sum(H[i,j,jj] * z[i,jj] for jj in model.y) \
                          + model.r[i,j] == model.w[i,j]
model.yyield = Constraint(model.m, model.y, rule=yyield_)

def compl_(model, i, j):
    return complements(0 <= model.w[i,j], model.z[i,j] >= 0)
model.compl = Complementarity(model.m, model.y, rule=compl_)

