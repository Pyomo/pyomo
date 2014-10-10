# gnash1.py    QQR2-MN-21-13
# Original Coopr coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, University of Dundee

# An MPEC from F. Facchinei, H. Jiang and L. Qi, A smoothing method for
# mathematical programs with equilibrium constraints, Universita di Roma
# Technical report, 03.96. Problem number 8
#
# Arises from Gournot Nash equilibrium , 10 instances are available 
# (see gnash1i.dat, i=0,1,...,9)

# Number of variables:   13
# Number of constraints: 13

import coopr.environ
from coopr.pyomo import *
from coopr.mpec import *


model = AbstractModel()

# ... parameters for each firm/company
model.c = Param(range(1,6))     # c_i
model.K = Param(range(1,6))     # K_i
model.b = Param(range(1,6))     # \beta_i

# ... parameters for each problem instance
model.L = Param()               # L
model.g = Param()               # \gamma

# ... computed constants
model.gg = Expression(expr=5000**(1/model.g))

model.x = Var(bounds=(0,model.L))
model.y = Var(range(1,5))
model.l = Var(sequence(1,8), within=NonNegativeReals)       # Multipliers
def Q_(model):
    return model.x+model.y[1]+model.y[2]+model.y[3]+model.y[4]
model.Q = Expression(rule=Q_)   # defined variable Q

def f_(model):
    return model.c[1]*model.x + model.b[1]/(model.b[1]+1)*model.K[1]**(-1/model.b[1])*model.x**((1+model.b[1])/model.b[1]) \
            - model.x*( model.gg*model.Q**(-1/model.g) )
model.f = Objective(rule=f_)

def F1_(model):
    return 0 == ( model.c[2] + model.K[2]**(-1/model.b[2])*model.y[1] ) - \
                                ( model.gg*model.Q**(-1/model.g) ) - \
                                model.y[1]*( -1/model.g*model.gg*model.Q**(-1-1/model.g) ) - (model.l[1] - model.l[2])
model.F1 = Constraint(rule=F1_)

def F2_(model):
    return 0 == ( model.c[3] + model.K[3]**(-1/model.b[3])*model.y[2] ) - \
                                ( model.gg*model.Q**(-1/model.g) ) - \
                                model.y[2]*( -1/model.g*model.gg*model.Q**(-1-1/model.g) ) - (model.l[3] - model.l[4])
model.F2 = Constraint(rule=F2_)

def F3_(model):
    return 0 == ( model.c[4] + model.K[4]**(-1/model.b[4])*model.y[3] ) - \
                                ( model.gg*model.Q**(-1/model.g) ) - \
                                model.y[3]*( -1/model.g*model.gg*model.Q**(-1-1/model.g) ) - (model.l[5] - model.l[6])
model.F3 = Constraint(rule=F3_)

def F4_(model):
    return 0 == ( model.c[5] + model.K[5]**(-1/model.b[5])*model.y[4] ) - \
                                ( model.gg*model.Q**(-1/model.g) ) - \
                                model.y[4]*( -1/model.g*model.gg*model.Q**(-1-1/model.g) ) - (model.l[7] - model.l[8])
model.F4 = Constraint(rule=F4_)

def g1_(model):
    return complements(0 <= model.y[1]          , model.l[1] >= 0)
model.g1 = Complementarity(rule=g1_)
def g2_(model):
    return complements(0 <= model.L - model.y[1], model.l[2] >= 0)
model.g2 = Complementarity(rule=g2_)
def g3_(model):
    return complements(0 <= model.y[2]          , model.l[3] >= 0)
model.g3 = Complementarity(rule=g3_)
def g4_(model):
    return complements(0 <= model.L - model.y[2], model.l[4] >= 0)
model.g4 = Complementarity(rule=g4_)
def g5_(model):
    return complements(0 <= y[3]                , model.l[5] >= 0)
model.g5 = Complementarity(rule=g5_)
def g6_(model):
    return complements(0 <= model.L - model.y[3], model.l[6] >= 0)
model.g6 = Complementarity(rule=g6_)
def g7_(model):
    return complements(0 <= model.y[4]          , model.l[7] >= 0)
model.g7 = Complementarity(rule=g7_)
def g8_(model):
    return complements(0 <= model.L - model.y[4], model.l[8] >= 0)
model.g8 = Complementarity(rule=g8_)

