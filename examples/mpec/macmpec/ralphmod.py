# -------------------------------------------------------
#
#   MPEC due to Danny Ralph, see also
#
#        G Maier, F Giannessi and A Nappi, Indirect
#        identification of yield limits by mathematical
#        programming, Engineering Structures 4, 1982,
#        86-98.
#
#   From a GAMS model by S.P. Dirkse & M.C. Ferris (MPECLIB),
#   (see http://www.gams.com/mpec/).
#
#   AMPL coding Sven Leyffer, University of Dundee, Jan. 2000
#
#   Coopr coding William Hart
#
#   -------------------------------------------------------


import coopr.environ
from coopr.pyomo import *
from coopr.mpec import *


model = AbstractModel()

model.vars = RangeSet(1, 104)
model.design = RangeSet(1, 4)
model.state = RangeSet(5, 104)
model.side = RangeSet(1, 8)

model.P = Param(model.vars, model.vars, default=0)
model.M = Param(model.state, model.vars, default=0)
model.q = Param(model.state, default=0)
model.c = Param(model.vars, default=0)

model.x = Var(model.design, bounds=(0, 1000))
model.y = Var(model.state, bounds=(0, None))

model.s = Var(model.state, bounds=(0, None))

def obj_(model):
    return 0.5*( sum{i in design} sum(x[i]*sum{j in design}P[i,j]*x[j] 
                    + 2.0*sum{i in design}x[i]*sum{j in state}P[i,j]*y[i]
                    + sum{ i in state}y[i]*sum{j in state}P[i,j]*y[j]     )
              + sum{i in design}c[i]*x[i] + sum{i in state}c[i]*y[i];
model.obj = Objective(rule=obj_)

def F_(model, i):
    return complements(0 <=  sum(model.M[i,k]*model.x[k] for k in model.design) \
                                + (model.M[i,j]*model.y[j] for j in model.state) + model.q[i]
                       model.y[i] >= 0)
model.F = Complementarity(model.state, rule=F_)

