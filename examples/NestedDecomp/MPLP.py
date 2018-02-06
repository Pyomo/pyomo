
################################################################################
##### Model taken from: Ind.Eng.Chem.Res (2003), 42, 3045, 3055 ################
# "Temporal Decomposition Scheme for Nonlinear Multisite Production Planning   #
# and Distribution Models", Jackson, J.R. & Grossmann, I.E., ###################
# Example 1: Linear Network ####################################################
################################################################################

from pyomo.environ import *
from MPdata import *

m = ConcreteModel()

#Sets
m.T = RangeSet(3) #time periods
m.I = Set(initialize=['A', 'B', 'C'], ordered=True) #products
m.M = RangeSet(3) #markets
m.S = RangeSet(3) #production sites

#Superset of iterations
m.iter = RangeSet(5)
#Set of iterations for which cuts are generated
m.K = Set(within=m.iter, dimen=1)

#Model Parameters
m.cap = Param(m.I, m.S, initialize=cap)
m.quota = Param(m.I, m.S, initialize=quota)
m.aa = Param(m.I, m.S, initialize=aa)
m.gg = Param(m.I, m.S, m.M, initialize=gg)
m.bb = Param(m.I, m.M, initialize=bb)
m.invi = Param(m.I, m.S, initialize=0)
m.fcast = Param(m.I, m.M, m.T, initialize=fcast)

#Decomposition Parameters
m.invpar=Param(m.I,m.S,m.T, default=0, initialize=0, mutable=True)
m.invpar_k=Param(m.I,m.S,m.T,m.iter, default=0, initialize=0, mutable=True)
m.profit=Param(m.T,m.iter, default=0, initialize=0, mutable=True)
m.multiplier=Param(m.I,m.S,m.T,m.iter, default=0, initialize=0, mutable=True)
m.profit_t=Param(m.T,m.iter, default=0, initialize=0, mutable=True)

#Create a block for a single time period
def planning_block_rule(b,t):
    #define variables
    b.p = Var(m.I, m.S, within=NonNegativeReals)
    b.f = Var(m.I, m.M, m.S, within=NonNegativeReals)
    b.sl = Var(m.I, m.M, within=NonNegativeReals)
    b.pen = Var(m.I, m.S, within=NonNegativeReals)
    b.tp = Var(m.I, m.S, within=NonNegativeReals)
    b.inv = Var(m.I, m.S, within=NonNegativeReals)
    b.inv_prev = Var(m.I, m.S, within=NonNegativeReals) #copy of linking var: inv
    b.alphafut = Var(within=Reals) #cost-to-go function

    #define constraints
    def c1(_b,i,s):
        return _b.pen[i,s] >=m.quota[i,s] - _b.inv[i,s]
    b.c1 = Constraint(m.I, m.S, rule=c1)

    def c2(_b,i,s):
        return _b.pen[i,s] >= _b.inv[i,s] - m.quota[i,s]
    b.c2 = Constraint(m.I, m.S, rule=c2)

    def c3(_b,i,mk):
        return _b.sl[i,mk] == sum(_b.f[i,mk,s] for s in m.S)
    b.c3 = Constraint(m.I, m.M, rule=c3)

    def c4(_b,i,mk):
        return _b.sl[i,mk] <= m.fcast[i,mk,t]
    b.c4 = Constraint(m.I, m.M, rule=c4)

    def c5(_b,i,s):
        if t == 1:
            return _b.p[i,s] + m.quota[i,s] == _b.inv[i,s] + sum(_b.f[i,mk,s] for mk in m.M)
        return _b.p[i,s] + _b.inv_prev[i,s] == _b.inv[i,s] + sum(_b.f[i,mk,s] for mk in m.M)
    b.c5 = Constraint(m.I, m.S, rule=c5)

    def c6(_b,i,s):
        return _b.p[i,s] <= m.cap[i,s]*_b.tp[i,s]
    b.c6 = Constraint(m.I, m.S, rule=c6)

    def c7(_b,s):
        return sum(_b.tp[i,s] for i in m.I) == 1
    b.c7 = Constraint(m.S, rule=c7)

    #linking equality
    def link_equal(_b,i,s):
        if t != 1:
            return _b.inv_prev[i,s] == m.invpar[i,s,t-1]
        return Constraint.Skip
    b.link_equal = Constraint(m.I,m.S, rule=link_equal)

    #Benders cut
    b.fut_cost = ConstraintList()

    #objective function
    def obj_rule(_b):
        return - sum(m.bb[i,mk]*_b.sl[i,mk] for i in m.I for mk in m.M) + \
                 sum(m.aa[i,s]*(_b.p[i,s] + 0.2*_b.pen[i,s]) \
                    for i in m.I for s in m.S) + \
                 sum(m.gg[i,s,mk]*_b.f[i,mk,s] \
                    for i in m.I for s in m.S for mk in m.M) \
                 + _b.alphafut
    b.obj = Objective(rule=obj_rule, sense=minimize)
m.Bl = Block(m.T, rule=planning_block_rule)
