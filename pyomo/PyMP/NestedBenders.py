#Nested Benders Decomposition 
__author__ = "Cristiana L. Lara"

from pyomo.environ import *
from MPLP import *

#Parameters to compute upper and lower bounds
m.profit_forward=Param(m.iter, default=0, initialize=0, mutable=True)
m.profit_backward=Param(m.iter, default=0, initialize=0, mutable=True)

#retrieve duals
for t in m.T:
    m.Bl[t].dual = Suffix(direction=Suffix.IMPORT)

#Nested Benders Decomposition algorithm
for iter_ in m.iter:

    #Forward Pass
    for t in m.T:
        #Fix alphafut=0 for iter 1
        if iter_ == 1:
            m.Bl[t].alphafut.fix(0)

        #Solve the model using CPLEX
        solver = SolverFactory('cplex')
        results = solver.solve(m.Bl[t]) #, tee=True)

        #Fix the linking variable as parameter for next t
        for (i,s) in m.I*m.S:
            m.invpar_k[i,s,t,iter_]=m.Bl[t].inv[i,s].value
            m.invpar[i,s,t]=m.Bl[t].inv[i,s].value
#        m.invpar.pprint()

        #Store obj value to compute UB
        m.profit_t[t,iter_]=m.Bl[t].obj() - m.Bl[t].alphafut.value

    m.profit_forward[iter_]=sum(m.profit_t[t,iter_] for t in m.T)
    m.profit_forward.pprint()
    m.Bl[t].alphafut.unfix()

    #Backward Pass
    m.K.add(iter_)

    for t in reversed(list(m.T)):
        if t == m.T.last():
            m.Bl[t].alphafut.fix(0)
        else:
            m.Bl[t].alphafut.unfix()

        #add Benders cut
        if t != m.T.last():
            for k in m.K:
                m.Bl[t].fut_cost.add(expr=(m.Bl[t].alphafut >= m.profit[t+1,k] \
                                     + sum(m.multiplier[i,s,t+1,k]* \
                                     (m.invpar_k[i,s,t,k] - m.Bl[t].inv[i,s]) \
                                     for i in m.I for s in m.S)))

        #Solve the model using CPLEX
        solver = SolverFactory('cplex')
        results = solver.solve(m.Bl[t]) #, tee=True)
#        m.Bl[t].alphafut.pprint()

        #Get Lagrange multiplier from linking equality
        if t != m.T.first():
            for (i,s) in m.I*m.S:
                m.multiplier[i,s,t,iter_]= - m.Bl[t].dual[m.Bl[t].link_equal[i,s]]
#        m.multiplier.pprint()

        #Get optimal value
        m.profit[t,iter_]=m.Bl[t].obj()
#        m.profit.pprint()

    m.profit_backward[iter_]=m.profit[1,iter_]
    m.profit_backward.pprint()

    print ("Upper Bound", m.profit_forward[iter_].value)
    print ("Lower Bound", m.profit_backward[iter_].value)
