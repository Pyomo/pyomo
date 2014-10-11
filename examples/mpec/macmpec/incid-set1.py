# incid-set1.py LQR2-MN-v-v
# Original Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, University of Dundee
#
# An MPEC from Outrata, Kocvara & Zowe, Nonsmooth Approach to
# Optimization Problems with Equilibrium Constraints, Kluwer, 1998.
#
# The incidence set identification problem of Section 9.4:
# Aim is to bring the membrane of a surface into contact with 
# the obstacle in a prescribed region whose shape is also to 
# be determined.
#
# A 2D obstacle problem on [0,1] x [0,1].
#
# Several discretizations are available (see insid-set-i.dat, 
# i=8,16,32).
#
# Formulation uses FE discretization with similar data
# structure as provided by triangulization routines.

import pyomo.modeling
from pyomo.core import *
from pyomo.mpec import *


model = AbstractModel()

# ... discretization parameters (change for finer grid)
model.n = Param(within=Integers)    # n = 8, 16, 32, 64
model.h = Param(initialize=1/model.n)     # discretization size

# ... parameters
model.Nnd = Param(initialize=(model.n+1)*(model.n+1))   # Number of nodes
model.r = Param(initialize=33.0)                        # weight for objective
model.n1 = Param(initialize=model.n/4)
model.n2 = Param(initialize=3*model.n/4)

# ... sets of nodes / classes of nodes
model.nodes = RangeSet(1, model.Nnd)                                    # set of nodes
model.bnd_nodes = Set(within=model.nodes)                               # set of boundary nodes
model.int_nodes = model.nodes - model.bnd_nodes                         # set of internal nodes
model.elements = Set(within=model.nodes * model.nodes * model.nodes)   # set of elements

# ... Omega0 = region, where membrane is fixed to obstacle
model.Omega0 = Set(within=model.int_nodes)

# ... [d1,d2] = start/end of Omega0 in x-direction indices
model.d1 = Param()
model.d2 = Param()

# ... node positions on i-j grid of indices
model.i_ref = Param(model.nodes, within=Integers) #bounds=(0, model.n)
model.j_ref = Param(model.nodes, within=Integers) #bounds=(0, model.n)

# ... node positions (y fixed)
def y_(model, i):
    return model.j_ref[i]*model.h
model.y = Param(model.nodes, initialize=y_)

# ... boundary conditions
model.u0 = Param(model.bnd_nodes)

# ... parameterization of "moving" boundary (alpha)
model.a_index = RangeSet(0, model.n)
model.a = Var(model.a_index, bounds=(0.7, 1.2), initialize=1.0)

# ... parameterization of "moving" obstacle
model.w_index = RangeSet(model.n1, model.n2)
model.w1 = Var(model.w_index, bounds=(0.15, 0.65), initialize=0.25)
model.w2 = Var(model.w_index, bounds=(0.15, 0.65), initialize=0.50)

# ... node positions (x depends on a and w1, w2)
def x_(model, i):
    if ((model.y[i] < 0.25) or (model.y[i] > 0.75)):
        return model.i_ref[i] * model.a[j_ref[i]] / model.n
    else:
        if (model.i_ref[i] < model.d1):
            return model.i_ref[i] * model.w1[j_ref[i]] / model.d1
        elif (model.i_ref[i] <= model.d2):
            return model.w1[model.j_ref[i]] + (model.i_ref[i]-model.d1)*(model.w2[model.j_ref[i]]-model.w1[model.j_ref[i]])/(model.d2-model.d1)
        else:
            return model.w2[model.j_ref[i]] + (model.i_ref[i]-model.d2)*(model.a[model.j_ref[i]]-model.w2[model.j_ref[i]])/(model.n-model.d2);
model.x = Expression(model.nodes, rule=x_)

# ... determinant of Je transformation for element stiffness matrix
def detJe_(model, i, j, k):
    return (model.x[j]-model.x[i])*(model.y[k]-model.y[i]) - (model.y[j]-model.y[i])*(model.x[k]-model.x[i])
model.detJe = Expression(model.elements, rule=detJe_)

# ... obstacle 
model.xi = Param(model.nodes, initialize=- 0.03)

# ... unknown height of membrane
model.u = Var(model.nodes)

# ... slack variables to deal with complementarity
model.s1 = Var(model.int_nodes, within=NonNegativeReals)

# ... global load vector (load is f(x,y) = -1.0 constant)
def l_(model, i):
    return  - sum(detJe[i,j,k] for (i,j,k) in model.elements)/6.0 \
            - sum(detJe[j,i,k] for (j,i,k) in model.elements)/6.0 \
            - sum(detJe[k,j,i] for (k,j,i) in model.elements)/6.0
model.l = Expression(model.int_nodes, rule=l_)

# ... set up PDE equation at all internal nodes here
def Au_(model, i):
    return \
    sum(
      (  ( (model.y[k]-model.y[i])**2 + (model.x[k]-model.x[i])**2 + (model.x[j]-model.x[i])**2 
          - 2*(model.x[k]-model.x[i])*(model.x[j]-model.x[i])                                                        ) * model.u[i]
       + (-(model.y[k]-model.y[i])**2 - (model.x[k]-model.x[i])**2 + (model.x[k]-model.x[i])*(model.x[j]-model.x[i]) ) * model.u[j]
       + (-(model.x[j]-model.x[i])**2 + (model.x[k]-model.x[i])*(model.x[j]-model.x[i])                              ) * model.u[k]
      ) / ( 2*model.detJe[i,j,k] ) for (i,j,k) in model.elements)

    + sum(
      (  ( (model.y[k]-model.y[j])**2 + (model.x[k]-model.x[j])**2                                                   ) * model.u[i]
       + (-(model.y[k]-model.y[j])**2 - (model.x[k]-model.x[j])**2 + (model.x[k]-model.x[j])*(model.x[i]-model.x[j]) ) * model.u[j]
       + (-(model.x[k]-model.x[j])*(model.x[i]-model.x[j])                                                           ) * model.u[k]
      ) / ( 2*model.detJe[j,i,k] ) for (j,i,k) in model.elements)

    + sum(
      (  ( (model.x[j]-model.x[k])**2                                                    ) * model.u[i]
       + (-(model.x[i]-model.x[k])*(model.x[j]-model.x[k])                               ) * model.u[j]
       + (-(model.x[j]-model.x[k])**2 + (model.x[i]-model.x[k])*(model.x[j]-model.x[k])  ) * model.u[k]
      ) / ( 2*model.detJe[k,j,i] ) for (k,j,i) in model.elements)
model.Au = Expression(model.int_nodes, rule=Au_)

def Jr_(model):
    return sum(model.s1[i] for i in model.int_nodes - model.Omega0) \
           + model.r*model.h**2*sum(model.u[i] - model.xi[i] for i in model.Omega0)
model.Jr = Objective(rule=Jr_)

# ... constraint on slope of "moving" boundary
def slope_a_(model, i):
    return -3*model.h <= model.a[i-1] - model.a[i] <= 3*model.h
model.slope_a = Constraint(RangeSet(1, model.n), rule=slope_a_)

# ... constraint on slope of "moving" boundary of Omega0
def slope_w1_(model, i):
    return -2.5*model.h <= model.w1[i-1] - model.w1[i] <= 2.5*model.h
model.slope_w1 = Constraint(RangeSet(model.n1+1, model.n2), rule=slope_w1_)

# ... constraint on slope of "moving" boundary of Omega0
def slope_w2_(model, i):
    return -2.5*model.h <= model.w2[i-1] - model.w2[i] <= 2.5*model.h
model.slope_w2 = Constraint(RangeSet(model.n1+1, model.n2), rule=slope_w2_)

# ... restriction ensure w1 <= w2
def lin_(model, i):
    return model.w1[i] + 0.05 <= model.w2[i]
model.lin = Constraint(RangeSet(model.n1, model.n2), rule=lin_)

# ... boundary conditions
def bnd_cond_(model, i):
    return model.u[i] == model.u0[i]
model.bnd_cond = Constraint(model.bnd_nodes, rule=bnd_cond_)

# ... FE approx to Laplacian
def PDE_(model, i):
    return model.s1[i] == model.Au[i] - model.l[i]
model.PDE = Constraint(model.int_nodes, rule=PDE_)

# ... obstacle lower bound
def obst_(model, i):
    return complements(0 <= model.s1[i], model.u[i] - model.xi[i] >= 0)
model.obst = Complementarity(model.int_nodes, rule=obst_)

