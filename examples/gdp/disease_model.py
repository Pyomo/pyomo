# ============================================
# SIR disease model using a low/high transmission parameter
# This is formulated as a disjunctive program
#
# Daniel Word, November 1, 2010
# ============================================

# import packages
from coopr.pyomo import *
from coopr.gdp import *
import math

# import data
from data_set import *
#from new_data_set import *

# declare model name
model = AbstractModel()

# declare constants
bpy = 26      # biweeks per year
years = 15    # years of data
bigM = 50.0    # big M for disjunction constraints

# declare sets
model.S_meas = RangeSet(1,bpy*years)
model.S_meas_small = RangeSet(1,bpy*years-1)
model.S_beta = RangeSet(1,bpy)

# define variable bounds
def _gt_zero(m,i):
    return (0.0,1e7)
def _beta_bounds(m):
    return (None,5.0)

# define variables

# log of estimated cases
#model.logI = Var(model.S_meas, bounds=_gt_zero)
model.logI = Var(model.S_meas, bounds=(0.001,1e7))
# log of transmission parameter beta
#model.logbeta = Var(model.S_beta, bounds=_gt_zero)
model.logbeta = Var(model.S_beta, bounds=(0.0001,5))
# binary variable y over all betas
#model.y = Var(model.S_beta, within=Binary)
# low value of beta
#model.logbeta_low = Var(bounds=_beta_bounds)
model.logbeta_low = Var(bounds=(0.0001,5))
# high value of beta
#model.logbeta_high = Var(bounds=_beta_bounds)
model.logbeta_high = Var(bounds=(0.0001,5))
# dummy variables
model.p = Var(model.S_meas, bounds=_gt_zero)
model.n = Var(model.S_meas, bounds=_gt_zero)

# define indexed constants

# log of measured cases after adjusting for underreporting
logIstar = logIstar
# changes in susceptible population profile from susceptible reconstruction
deltaS = deltaS
# mean susceptibles
#meanS = 1.04e6
meanS = 8.65e5
# log of measured population
logN = pop
# define index for beta over all measurements
beta_set = beta_set

# define objective
def _obj_rule(m):
    expr = sum(m.p[i] + m.n[i] for i in m.S_meas)
    return expr
model.obj = Objective(rule=_obj_rule, sense=minimize)

# define constraints
def _logSIR(m,i):
    expr = m.logI[i+1] - ( m.logbeta[beta_set[i-1]] + m.logI[i] + math.log(deltaS[i-1] + meanS) - logN[i-1] )
    return (0.0, expr)
model.logSIR = Constraint(model.S_meas_small, rule=_logSIR)

# objective function constraint
def _p_n_const(m,i):
    expr = logIstar[i-1] - m.logI[i] - m.p[i] + m.n[i]
    return (0.0, expr)
model.p_n_const = Constraint(model.S_meas,rule=_p_n_const)

# disjuncts

model.y = RangeSet(0,1)
def _high_low(model, disjunct, i, y):
    disjunct.set_M(20)
    if y:
        disjunct.c = Constraint(expr=model.logbeta_high - model.logbeta[i]== 0.0)
    else:
        disjunct.c = Constraint(expr=model.logbeta[i] - model.logbeta_low == 0.0)
model.high_low = Disjunct(model.S_beta, model.y, rule=_high_low)

# disjunctions
def _disj(model, i):
    return [model.high_low[i,j] for j in model.y]
model.disj = Disjunction(model.S_beta, rule=_disj)


"""
# high beta disjuncts
def highbeta_L(m,i):
    expr = m.logbeta[i] - m.logbeta_high + bigM*(1-m.y[i])
    return (0.0, expr, None)
model.highbeta_L = Constraint(model.S_beta, rule=highbeta_L)

def highbeta_U(m,i):
    expr = m.logbeta[i] - m.logbeta_high
    return (None, expr, 0.0)
model.highbeta_U = Constraint(model.S_beta, rule=highbeta_U)

# low beta disjuncts
def lowbeta_U(m,i):
    expr = m.logbeta[i] - m.logbeta_low - bigM*(m.y[i])
    return (None, expr, 0.0)
model.lowbeta_U = Constraint(model.S_beta, rule=lowbeta_U)

def lowbeta_L(m,i):
    expr = m.logbeta[i] - m.logbeta_low
    return (0.0, expr, None)
model.lowbeta_L = Constraint(model.S_beta, rule=lowbeta_L)
"""
