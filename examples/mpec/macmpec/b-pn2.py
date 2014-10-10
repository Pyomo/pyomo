#   b-pn2.py QOR2-MN-v-v
#
#   MPEC with equilibrium constraint as explicit nonlinear equation
#
#   BEM Quasibrittle Fracture Identification (N, mm)
#   B_pn2.gms = Two-branch law, Penalty, 2-norm
#   Imposed displacement q
#
#   traction    : t = q.te + Z.w
#   load        : p = q.pe + r'.w
#   crack width : w(i) = lw(i,"y3")
#   yield       : phi = tc.v1 - tb.v2 - k.M1.lw - h.M2.lw - t.n
#   phi >= 0, (yield) \perp (lw >= 0)
#
#   Uses 1 data file: bem-milanc30.dat  ... contains Z etc.
#   Calculated p is for full beam, 1 mm thick
#   Beam in expt = 15 mm thick, u_expt in microns
#   p_expt = 15*p Newtons
#   u_expt = 1000*u microns
#
#   F. Tin-Loi : 24 Feb 99
#
#   Pyomo coding by William Hart, Sandia National Laboratories, 2014
#   from AMPL coding by Sven Leyffer, University of Dundee, Mar. 2000
#   from a GAMS file by F. Tin-Loi.
#
#   Data files: bem-milanc30-s.dat: set SS = {18,20,22,24,26,28,30,32};
#               bem-milanc30-l.dat: set SS = S;

import pyomo.environ
from pyomo.core import *
from pyomo.mpec import *


model = AbstractModel()

model.I = RangeSet(1, 61)           # ... Num bem nodes, size Z etc
model.Y = RangeSet(1, 3)            # ... Num yield modes per node
model.S = RangeSet(1, 48)           # ... Load situations

# ... change this set to select different points to be matched
model.SS = Set(within=model.S)

# ... data originally from bem_expt.dat
model.q = Param(model.S)            # ... parameter q(s)  (mm)
model.Qm = Param(model.S)           # ... parameter Qm(s) (N) total Q for unit thickness

# ... data originally from bem_milanc30.dat
model.Z = Param(model.I, model.I)   # ... table     Z(i,i) 
model.te = Param(model.I)           # ... parameter te(i)  
model.r = Param(model.I)            # ... parameter r(i)   
model.pe = Param()                  # ... scalar    pe     

# ... 2-branch parameter
model.v1 = Param(model.Y, initialize=1)

def v2_init(model, y):
    return 1.0 if y == 1 else 0.0
model.v2 = Param(model.Y, initialize=v2_init)

def M1_init(model, y_, y):
    if y == 1:
        return -1.0
    elif y == 2:
        return -1.0
    elif y == 3:
        return 1.0
    return 0.0
model.M1 = Param(model.Y, model.Y, initialize=M1_init)

def M2_init(model, y, yy):
    if (y == 1) and (yy == 1):
        return -1.0
    return 0.0
model.M2 = Param(model.Y, model.Y, initialize=M2_init)

def n_init(model, y):
    if y == 3:
        return 1.0
    return 0.0
model.n = Param(model.Y, initialize=n_init)

# ... problem variables
model.tc = Var(bounds=(1,30), initialize=10)                # ... Traction limit 1
model.tb = Var(bounds=(1,30), initialize=4)                 # ... Traction limit 2
model.k = Var(bounds=(10,500), initialize=200)              # ... Slope branch 1
model.h = Var(bounds=(10,500), initialize=200)              # ... Slope branch 1
model.errQ = Var(model.SS, initialize=1)                    # ... Qc - Qm
model.t = Var(model.SS, model.I)                            # ... Tractions
model.lw = Var(model.SS, model.I, model.Y, bounds=(0,1))    # ... Lambda-w vector
model.Qc = Var(model.S, initialize=model.Qm)                # ... Calculated load
model.phi = Var(model.SS, model.I, model.Y, bounds=(0,1))   # ... Yield functions


model.cost = Objective(expr=summation(model.errQ, model.errQ))

def compl_(model, s, i, y):
    return complements(0 <= model.phi[s,i,y], model.lw[s,i,y] >= 0)
model.compl = Complementarity(model.SS, model.I, model.Y, rule=compl_)

def traction_(model, s, i):
      model.t[s,i] == model.q[s]*model.te[i] + sum(model.Z[i,j]*model.lw[s,j,3] for j in model.I)
model.traction = Constraint(model.SS, model.I, rule=traction_)

def yyield_(model, s, i, y):
    return model.tc*model.v1[y] - model.tb*model.v2[y] - model.k*sum(model.M1[y,yy]*model.lw[s,i,yy] for yy in model.Y) \
      - model.h*sum(model.M2[y,yy]*model.lw[s,i,y] for yy in model.Y) - model.t[s,i]*model.n[y] \
      == model.phi[s,i,y]
model.yyield = Constraint(model.SS, model.I, model.Y, rule=yyield_)

def err_Q_(model, s):
    return model.errQ[s] == model.Qm[s] - model.Qc[s]
model.err_Q = Constraint(model.SS, rule=err_Q_)

def def_Qc_(model, s):
    return model.Qc[s] == model.q[s]*model.pe + sum(model.r[i]*model.lw[s,i,3] for i in model.I)
model.def_Qc = Constraint(model.SS, rule=def_Qc_)

model.tc_tb = Constraint(expr=model.tc >= model.tb)

