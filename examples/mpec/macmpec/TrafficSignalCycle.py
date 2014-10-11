#################################################################################################
#  March/2012											                                        #
#												                                                #
#  This problem is described in "Optimal cycle for a signalized intersection using Global       #
#  Optimization and Complementarity, by Isabel M. Ribeiro and M. Lurdes Simoes in Sociedad de   #    
#  Estadistica e Investigacion Operativa 2010, DOI 10.1007/s11750-010-0167-3                    #
#												                                                #
#  AMPL Code by Teofilo Melo, Teresa Monteiro and Joao Matias                                   #   
#  Pyomo Code by William Hart                                                                   #
#################################################################################################	

import pyomo.modeling
from pyomo.core import *
from pyomo.mpec import *


model = AbstractModel()

# SETS
				
model.S = Set()         # traffic streams
model.K = Set()         # time instants 

# PARAMETERS

model.N = Param()			        # time periods  
model.xmax = Param() 				# maximum queue length in each traffic stream
model.gmax = Param() 				# maximum duration for green and red time
model.gmin = Param() 			    # minimum duration for green and red time
model.Lambda = Param(model.S)       # average arrival rate of vehicles in traffic each stream 
model.L0 = Param(model.S)           # queue length in traffic stream S at initial time t0
model.b1 = Param(model.S)           # vectors b1, b2, b3, b4, b5 and b6  
model.b2 = Param(model.S)           
model.b3 = Param(model.S)           
model.b4 = Param(model.S)           
model.b5 = Param(model.S)           
model.b6 = Param(model.S)           

model.Periods = RangeSet(1, model.N)

# VARIABLES 

model.L = Var(model.S, model.K)     # queue length in traffic stream S at instant time K
model.y = Var([1,2])                # time duration for red and green time 

# FUNCTION TO MINIMIZE: average waiting time over all queues

model.Obj = Objective(expr=\
    sum(1/model.Lambda[i]*((1/(2*(2*model.N+1))*model.L0[i])) + \
        sum((1/(2*model.N+1)* model.L[i,j]) + 1/(2*(2*model.N+1))*model.L[i,2*model.N+1] 
            for j in sequence(1,2*model.N)) 
        for i in model.S))

# CONSTRAINTS 

def c1_(model, i, j):
    return 0 <= model.L[i,j] <= model.xmax
model.c1 = Constraint(model.S, model.K, rule=c1_)

def c21_(model, i):
    return model.L[i,1] >= model.L0[i] + model.b1[i]*model.y[2] + model.b3[i]
model.c21 = Constraint(model.S, rule=c21_)

def c31_(model, i):
    return model.L[i,1] >= model.b5[i]
model.c31 = Constraint(model.S, rule=c31_)

def c41_(model, i):
    return complements(model.L[i,1] >= model.L0[i] + model.b1[i]*model.y[2] + model.b3[i], 
                       model.L[i,1] >= model.b5[i])
model.c41 = Complementarity(model.S, rule=c41_)
 
def c2_(model, i, j):
    return model.L[i, 2*j] >= model.L[i,2*j-1] + model.b1[i]*model.y[2] + model.b3[i]
model.c2 = Constraint(model.S, model.Periods, rule=c2_)

def c3_(model, i, j):
    return model.L[i, 2*j] >= model.b5[i]
model.c3 = Constraint(model.S, model.Periods, rule=c3_)

def c4_(model, i, j):
    return complements(model.L[i, 2*j] >= model.L[i,2*j-1] + model.b1[i]*model.y[2] + model.b3[i],
                       model.L[i, 2*j] >= model.b5[i])
model.c4 = Complementarity(model.S, model.Periods, rule=c4_)

def c5_(model, i, j):
    return model.L[i, 2*j+1] >= model.L[i,2*j] + model.b2[i]*model.y[1] + model.b4[i]
model.c5 = Constraint(model.S, model.Periods, rule=c5_)

def c6_(model, i, j):
    return model.L[i, 2*j+1] >= model.b6[i]
model.c6 = Constraint(model.S, model.Periods, rule=c6_)

def c7_(model, i, j):
    return complements(model.L[i, 2*j+1] >= model.L[i,2*j] + model.b2[i]*model.y[1] + model.b4[i],
                       model.L[i, 2*j+1] >= model.b6[i])
model.c7 = Complementarity(model.S, model.Periods, rule=c7_)

def c8_(model, i):
    return model.gmin <= model.y[i] <= model.gmax
model.c8 = Constraint([1,2], rule=c8_)

