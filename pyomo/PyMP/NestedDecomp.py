__author__ = "Cristiana L. Lara"

import time
start_time = time.time()
from GTEPblock import *

#Parameters to compute upper and lower bounds
m.cost_feasible=Param(m.iter, default=0, initialize=0, mutable=True)
m.cost_UB=Param(m.iter, default=0, initialize=0, mutable=True)
m.cost_LB=Param(m.iter, default=0, initialize=0, mutable=True)
m.gap=Param(m.iter, default=0, initialize=0, mutable=True)

#retrieve duals
for t in m.t:
    m.Bl[t].dual = Suffix(direction=Suffix.IMPORT)

#Nested Decomposition algorithm
for iter_ in m.iter:

    #Forward Pass##############################################################
    for t in m.t:
        #Fix alphafut=0 for iter 1
        if iter_ == 1:
            m.Bl[t].alphafut.fix(0)

        #Solve the model using CPLEX
        mipsolver = SolverFactory('cplex')
#        mipsolver.options['relax_integrality'] = 1
        mipsolver.options['mipgap']=0.0001
        mipsolver.options['timelimit']=30
        mipsolver.options['threads']=10
        results = mipsolver.solve(m.Bl[t])#, tee=True)

        #Fix the linking Paramiable as parameter for next t
        for (rn,r) in m.i_r:
            if rn in m.rn:
                m.ngo_rn_par_k[rn,r,t,iter_]=m.Bl[t].ngo_rn[rn,r].value
                m.ngo_rn_par[rn,r,t]=m.Bl[t].ngo_rn[rn,r].value
                m.ngb_rn_par_k[rn,r,t,iter_]=m.Bl[t].ngb_rn[rn,r].value
                m.ngb_rn_par[rn,r,t]=m.Bl[t].ngb_rn[rn,r].value
#        m.ngo_rn_par.display()

        for (th,r) in m.i_r:
            if th in m.th:
                m.ngo_th_par_k[th,r,t,iter_]=m.Bl[t].ngo_th[th,r].value
                m.ngo_th_par[th,r,t]=m.Bl[t].ngo_th[th,r].value
                m.ngb_th_par_k[th,r,t,iter_]=m.Bl[t].ngb_th[th,r].value
                m.ngb_th_par[th,r,t]=m.Bl[t].ngb_th[th,r].value
#        m.ngo_th_par.display()

        #Store feasible results
        for (i,r,ss,s) in m.i_r*m.ss*m.s:
            m.P_result[i,r,t,ss,s,iter_] = m.Bl[t].P[i,r,ss,s].value
        for (r,ss,s) in m.r*m.ss*m.s:
            m.cu_result[r,t,ss,s,iter_] =  m.Bl[t].cu[r,ss,s].value
        m.RES_def_result[t,iter_] = m.Bl[t].RES_def.value
        for (r,r_,ss,s) in m.r*m.r*m.ss*m.s:
            if r_ != r:
                m.P_flow_result[r,r_,t,ss,s,iter_] = m.Bl[t].P_flow[r,r_,ss,s].value
        for (th,r,ss,s) in m.i_r*m.ss*m.s:
                if th in m.th:
                    m.Q_spin_result[th,r,t,ss,s,iter_] = m.Bl[t].Q_spin[th,r,ss,s].value
                    m.Q_Qstart_result[th,r,t,ss,s,iter_] = m.Bl[t].Q_Qstart[th,r,ss,s].value
                    m.u_result[th,r,t,ss,s,iter_] = m.Bl[t].u[th,r,ss,s].value
                    m.su_result[th,r,t,ss,s,iter_] = m.Bl[t].su[th,r,ss,s].value
                    m.sd_result[th,r,t,ss,s,iter_] = m.Bl[t].sd[th,r,ss,s].value
        for (rn,r) in m.i_r:
            if rn in m.rn:
                m.ngo_rn_result[rn,r,t,iter_] = m.Bl[t].ngo_rn[rn,r].value
                m.ngb_rn_result[rn,r,t,iter_] = m.Bl[t].ngb_rn[rn,r].value
                m.ngr_rn_result[rn,r,t,iter_] = m.Bl[t].ngr_rn[rn,r].value
                m.nge_rn_result[rn,r,t,iter_] = m.Bl[t].nge_rn[rn,r].value
        for (th,r) in m.i_r:
            if th in m.th:
                m.ngo_th_result[th,r,t,iter_] = m.Bl[t].ngo_th[th,r].value
                m.ngb_th_result[th,r,t,iter_] = m.Bl[t].ngb_th[th,r].value
                m.ngr_th_result[th,r,t,iter_] = m.Bl[t].ngr_th[th,r].value
                m.nge_th_result[th,r,t,iter_] = m.Bl[t].nge_th[th,r].value

        #Store obj value to compute UB
        m.cost_t[t,iter_]=m.Bl[t].obj() - m.Bl[t].alphafut.value

    #Compute upper bound (feasible solution)
    m.cost_feasible[iter_]=sum(m.cost_t[t,iter_] for t in m.t)
    m.cost_UB[iter_]=min(value(m.cost_feasible[kk]) for kk in m.iter if kk <= iter_)
    m.cost_UB.pprint()
    m.Bl[t].alphafut.unfix()
    elapsed_time = time.time() - start_time
    print ("CPU Time (s)", elapsed_time)

    #Backward Pass############################################################
    m.k.add(iter_)

    for t in reversed(list(m.t)):
        if t == m.t.last():
            m.Bl[t].alphafut.fix(0)
        else:
            m.Bl[t].alphafut.unfix()

        #add Benders cut
        if t != m.t.last():

            for k in m.k:
                m.Bl[t].fut_cost.add(expr=(m.Bl[t].alphafut >= m.cost[t+1,k] \
                                     + sum(m.mltp_o_rn[rn,r,t+1,k]* \
                                     (m.ngo_rn_par_k[rn,r,t,k] - m.Bl[t].ngo_rn[rn,r]) \
                                     for rn,r in m.i_r if rn in m.rn)\
                                     + sum(m.mltp_o_th[th,r,t+1,k]* \
                                     (m.ngo_th_par_k[th,r,t,k] - m.Bl[t].ngo_th[th,r]) \
                                     for th,r in m.i_r if th in m.th)\
                                     + sum(m.mltp_b_rn[rn,r,t+m.LT[rn],k]* \
                                     (m.ngb_rn_par_k[rn,r,t,k] - m.Bl[t].ngb_rn[rn,r]) \
                                     for rn,r in m.i_r if rn in m.rn and (t+m.LT[rn] <= m.t.last()))\
                                     + sum(m.mltp_b_th[th,r,t+m.LT[th],k]* \
                                     (m.ngb_th_par_k[th,r,t,k] - m.Bl[t].ngb_th[th,r]) \
                                     for th,r in m.i_r if th in m.th and (t+m.LT[th] <= m.t.last()))
                                     ))

        #Solve the model using CPLEX
        opt = SolverFactory('cplex')
        opt.options['relax_integrality'] = 1
        opt.options['threads']=10
        results = opt.solve(m.Bl[t])#, tee=True)
#        m.Bl[t].alphafut.pprint()

        #Get Lagrange multiplier from linking equality
        if t != m.t.first():
            for (rn,r) in m.i_r:
                if rn in m.rn:
                    m.mltp_o_rn[rn,r,t,iter_]= - m.Bl[t].dual[m.Bl[t].link_equal1[rn,r]]
                    if t > m.LT[rn]:
                        m.mltp_b_rn[rn,r,t,iter_]= - m.Bl[t].dual[m.Bl[t].link_equal3[rn,r]]
                    else:
                        m.mltp_b_rn[rn,r,t,iter_]=0
        #m.mltp_o_rn.pprint()
        #m.mltp_b_rn.pprint()
        if t != m.t.first():
            for (th,r) in m.i_r:
                if th in m.th:
                    m.mltp_o_th[th,r,t,iter_]= - m.Bl[t].dual[m.Bl[t].link_equal2[th,r]]
                    if t > m.LT[th]:
                        m.mltp_b_th[th,r,t,iter_]= - m.Bl[t].dual[m.Bl[t].link_equal4[th,r]]
                    else:
                        m.mltp_b_th[th,r,t,iter_]=0
        #m.mltp_o_th.pprint()
        #m.mltp_b_th.pprint()

        #Get optimal value
        m.cost[t,iter_]=m.Bl[t].obj()
#        m.cost.pprint()

    #Compute lower bound
    m.cost_LB[iter_]=m.cost[1,iter_]
    m.cost_LB.pprint()
    #Compute optimality gap
    m.gap[iter_]=(m.cost_UB[iter_]-m.cost_LB[iter_])/m.cost_UB[iter_]*100
    print m.gap[iter_].value

    if value(m.gap[iter_]) <= 0.01:
        break

    elapsed_time = time.time() - start_time
    print ("CPU Time (s)", elapsed_time)

elapsed_time = time.time() - start_time

print ("Upper Bound", m.cost_UB[iter_].value)
print ("Lower Bound", m.cost_LB[iter_].value)
print ("Optimality gap (%)", m.gap[iter_].value)
print ("CPU Time (s)", elapsed_time)

post_process()
