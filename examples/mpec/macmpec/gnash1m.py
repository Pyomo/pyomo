# gnash1m.py   QQR2-MN-10-8
# Original Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, University of Dundee

# Formulation with mixed complementarity of ...
#
# An MPEC from F. Facchinei, H. Jiang and L. Qi, A smoothing method for
# mathematical programs with equilibrium constraints, Universita di Roma
# Technical report, 03.96. Problem number 8
#
# Arises from Gournot Nash equilibrium , 10 instances are available 
# (see gnash1i.dat, i=0,1,...,9)

# Number of variables:   10
# Number of constraints:  8

import pyomo.modeling
from pyomo.core import *
from pyomo.mpec import *


model = AbstractModel()

# ... parameters for each firm/company
model.c = Param(sequence(1,5))  # c_i
model.K = Param(sequence(1,5))  # K_i
model.b = Param(sequence(1,5))  # \beta_i

# ... parameters for each problem instance
model.L = Param()   # L
model.g = Param()   # \gamma

# ... computed constants
def gg_(model):
    return 5000**(1/model.g)
model.gg = Expression(expr=5000**(1/model.g))

model.x = Var(bounds=(0,model.L))
model.y = Var(sequence(1,4))
model.l = Var(sequence(1,4))    # Multipliers
def Q_(model):
    return model.x+model.y[1]+model.y[2]+model.y[3]+model.y[4]
model.Q = Expression(rule=Q_)   # defined variable Q

def f_(model):
    return model.c[1]*model.x + model.b[1]/(model.b[1]+1)*model.K[1]**(-1/model.b[1])*model.x**((1+model.b[1])/model.b[1]) - model.x*( model.gg*model.Q**(-1/model.g) )
model.f = Objective(rule=f_)

def F1_(model):
    return 0 == ( model.c[2] + model.K[2]**(-1/model.b[2])*model.y[1] ) - ( model.gg*model.Q**(-1/model.g) ) - model.y[1]*( -1/model.g*model.gg*model.Q**(-1-1/model.g) ) - model.l[1]
model.F1 = Constraint(rule=F1_)

def F2_(model):
    return 0 == ( model.c[3] + model.K[3]**(-1/model.b[3])*model.y[2] ) - ( model.gg*model.Q**(-1/model.g) ) - model.y[2]*( -1/model.g*model.gg*model.Q**(-1-1/model.g) ) - model.l[2]
model.F2 = Constraint(rule=F2_)

def F3_(model):
    return 0 == ( model.c[4] + model.K[4]**(-1/model.b[4])*model.y[3] ) - ( model.gg*model.Q**(-1/model.g) ) - model.y[3]*( -1/model.g*model.gg*model.Q**(-1-1/model.g) ) - model.l[3]
model.F3 = Constraint(rule=F3_)

def F4_(model):
    return 0 == ( model.c[5] + model.K[5]**(-1/model.b[5])*model.y[4] ) - ( model.gg*model.Q**(-1/model.g) ) - model.y[4]*( -1/model.g*model.gg*model.Q**(-1-1/model.g) ) - model.l[4]
model.F4 = Constraint(rule=F4_)

def g1_(model):
    return complements(0 <= model.y[1] <= model.L, model.l[1])
model.g1 = Complementarity(rule=g1_)

def g3_(model):
    return complements(0 <= model.y[2] <= model.L, model.l[2])
model.g3 = Complementarity(rule=g3_)

def g5_(model):
    return complements(0 <= model.y[3] <= model.L, model.l[3])
model.g5 = Complementarity(rule=g5_)

def g7_(model):
    return complements(0 <= model.y[4] <= model.L, model.l[4])
model.g7 = Complementarity(rule=g7_)

