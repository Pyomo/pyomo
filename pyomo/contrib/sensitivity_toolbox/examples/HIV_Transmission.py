#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Model Reference: "A Dynamic HIV-Transmission Model for Evaluating the Costs  
#	and Benefits of Vaccine Programs", D.M. Edwards, R.D. Shachter,
#	and D.K. Owen 1998, Interfaces
#

from __future__ import division
from pyomo.environ import (ConcreteModel, Param, Var, Objective,
                           Constraint, Set, Expression, Suffix,
                           value, exp, TransformationFactory)
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.simulator import Simulator
from pyomo.contrib.sensitivity_toolbox.sens import sipopt


def create_model():

    m = ConcreteModel()

    m.tf = Param(initialize=20)
    m.t = ContinuousSet(bounds=(0,m.tf))
    m.i = Set(initialize=[0,1,2,3,4,5],ordered=True)
    m.j = Set(initialize=[0,1],ordered=True)
    m.ij = Set(initialize=[(0,0),(0,1),(1,0),(1,1),(2,0),(2,1),(3,0),(4,0)],
                          ordered=True) 
    
    #Set Parameters
    m.eps = Param(initialize = 0.75, mutable=True)
    
    m.sig = Param(initialize = 0.15, mutable=True)
    m.xi  = Param(initialize = 0.983, mutable=True)
    m.omeg  = Param(initialize = 1.0/10, mutable=True)
    
    m.Y0   = Param(initialize = 55816.0, mutable=True)
    m.phi0 = Param(initialize = 0.493, mutable=True)
    
    d={}
    d[(0,0)] = 0.0222
    d[(0,1)] = 0.0222
    d[(1,0)] = 1/7.1
    d[(1,1)] = 1/7.1
    d[(2,0)] = 1/8.1
    d[(2,1)] = 1/(8.1+5)
    d[(3,0)] = 1/2.7
    d[(4,0)] = 1/2.1
    m.dd = Param(m.ij, initialize=d, mutable=True)
    
    d_inv={}
    d_inv[(1,0)] = 7.1
    d_inv[(2,0)] = 8.1
    d_inv[(3,0)] = 2.7
    d_inv[(4,0)] = 2.1
    m.dd_inv = Param(m.ij, initialize=d_inv, default=0, mutable=True)
    
    I={}
    I[(0,0)] =  0.9*m.dd[(0,0)]*m.Y0
    I[(1,0)] = 0.04*m.dd[(0,0)]*m.Y0
    I[(2,0)] = 0.04*m.dd[(0,0)]*m.Y0
    I[(3,0)] = 0.02*m.dd[(0,0)]*m.Y0
    m.II=Param(m.ij, initialize=I, default=0, mutable=True)
    
    p={}
    p[(4,0)] = 0.667
    m.pp = Param(m.ij, initialize=p, default=2, mutable=True)
    
    b={}
    b[(1,0)] = 0.066
    b[(1,1)] = 0.066
    b[(2,0)] = 0.066
    b[(2,1)] = 0.066*(1-0.25)
    b[(3,0)] = 0.147
    b[(4,0)] = 0.147
    m.bb = Param(m.ij,initialize=b, default=0, mutable=True)
    
    eta00={}
    eta00[(0,0)] = 0.505
    eta00[(0,1)] = 0.505
    eta00[(1,0)] = 0.505
    eta00[(1,1)] = 0.505
    eta00[(2,0)] = 0.307
    eta00[(2,1)] = 0.4803
    eta00[(3,0)] = 0.235
    eta00[(4,0)] = 0.235
    m.eta00 = Param(m.ij, initialize=eta00, mutable=True)
    
    eta01={}
    eta01[(0,0)] = 0.505
    eta01[(0,1)] = 0.6287
    eta01[(1,0)] = 0.505
    eta01[(1,1)] = 0.6287
    eta01[(2,0)] = 0.307
    eta01[(2,1)] = 0.4803
    eta01[(3,0)] = 0.235
    eta01[(4,0)] = 0.235
    m.eta01 = Param(m.ij, initialize=eta01, mutable=True)
    
    m.kp  = Param(initialize = 1000.0, mutable=True)
    m.kt  = Param(initialize = 1000.0, mutable=True)
    m.rr  = Param(initialize = 0.05, mutable=True)
    
    c={}
    c[(0,0)] = 3307
    c[(0,1)] = 3307
    c[(1,0)] = 5467
    c[(1,1)] = 5467
    c[(2,0)] = 5467
    c[(2,1)] = 5467
    c[(3,0)] = 12586
    c[(4,0)] = 35394
    m.cc = Param(m.ij, initialize=c, mutable=True)
    
    q={}
    q[(0,0)] = 1
    q[(0,1)] = 1
    q[(1,0)] = 1
    q[(1,1)] = 1
    q[(2,0)] = 0.83
    q[(2,1)] = 0.83
    q[(3,0)] = 0.42
    q[(4,0)] = 0.17
    m.qq = Param(m.ij, initialize=q, mutable=True)
    
    m.aa = Param(initialize = 0.0001, mutable=True)
    
    #Set Variables
    m.yy = Var(m.t,m.ij)
    m.L = Var(m.t)
    
    m.vp = Var(m.t, initialize=0.75, bounds=(0,0.75))
    m.vt = Var(m.t, initialize=0.75, bounds=(0,0.75))
    
    m.dyy = DerivativeVar(m.yy, wrt=m.t)
    m.dL = DerivativeVar(m.L, wrt=m.t)
    
    def CostFunc(m):
        return m.L[m.tf]
    m.cf = Objective(rule=CostFunc)
    
    
    def _initDistConst(m):
        return (m.phi0*m.Y0)/sum(m.dd_inv[kk] for kk in m.ij)
    m.idc = Expression(rule=_initDistConst)
    
    m.yy[0,(0,0)].fix(value((1-m.phi0)*m.Y0))
    m.yy[0,(0,1)].fix(0)
    m.yy[0,(1,0)].fix(value(m.dd_inv[(1,0)]*m.idc))
    m.yy[0,(1,1)].fix(0)
    m.yy[0,(2,0)].fix(value(m.dd_inv[(2,0)]*m.idc))
    m.yy[0,(2,1)].fix(0)
    m.yy[0,(3,0)].fix(value(m.dd_inv[(3,0)]*m.idc))
    m.yy[0,(4,0)].fix(value(m.dd_inv[(4,0)]*m.idc))
    m.L[0].fix(0)
    
    
    #ODEs
    def _yy00(m, t): 
        return sum(m.pp[kk]*m.yy[t,kk] for kk in m.ij)*m.dyy[t,(0,0)] == \
    	   	sum(m.pp[kk]*m.yy[t,kk] for kk in m.ij)*(m.II[(0,0)]-\
    		(m.vp[t]+m.dd[(0,0)])*m.yy[t,(0,0)]+m.omeg*m.yy[t,(0,1)])-\
               	m.pp[(0,0)]*sum(m.bb[kk]*m.eta00[kk]*m.pp[kk]*m.yy[t,kk] 
                                for kk in m.ij)*m.yy[t,(0,0)] 
    m.yy00DiffCon = Constraint(m.t, rule=_yy00)
    
    def _yy01(m, t):
        return sum(m.pp[kk]*m.yy[t,kk] for kk in m.ij)*m.dyy[t,(0,1)] == \
               	sum(m.pp[kk]*m.yy[t,kk] 
                    for kk in m.ij)*(m.vp[t]*m.yy[t,(0,0)]-
                                    (m.dd[(0,0)]+m.omeg)*m.yy[t,(0,1)])-\
                m.pp[(0,1)]*(1-m.eps)*sum(m.bb[kk]*m.eta01[kk]*
                                          m.pp[kk]*m.yy[t,kk] 
                                          for kk in m.ij)*m.yy[t,(0,1)]
    m.yy01DiffCon = Constraint(m.t, rule=_yy01)
    
    def _yy10(m, t):
        return sum(m.pp[kk]*m.yy[t,kk] for kk in m.ij)*m.dyy[t,(1,0)] == \
               	sum(m.pp[kk]*m.yy[t,kk] for kk in m.ij)*\
                (m.II[(1,0)]-((m.sig*m.xi)+m.vp[t]+m.dd[(1,0)]+m.dd[(0,0)])*
                  m.yy[t,(1,0)]+m.omeg*m.yy[t,(1,1)]
                )+m.pp[(0,0)]*sum(m.bb[kk]*m.eta00[kk]*
                                  m.pp[kk]*m.yy[t,kk] 
                                  for kk in m.ij)*m.yy[t,(0,0)]
    m.yy10DiffCon = Constraint(m.t, rule=_yy10)
    
    def _yy11(m, t):
        return sum(m.pp[kk]*m.yy[t,kk] for kk in m.ij)*m.dyy[t,(1,1)] == \
               	sum(m.pp[kk]*m.yy[t,kk] for kk in m.ij)*(m.vp[t]*m.yy[t,(1,0)]-\
                (m.omeg+(m.sig*m.xi)+m.dd[(1,1)]+m.dd[(0,0)])*m.yy[t,(1,1)])+\
    		m.pp[(0,1)]*(1-m.eps)*sum(m.bb[kk]*m.eta01[kk]*
                                          m.pp[kk]*m.yy[t,kk] 
                                          for kk in m.ij)*m.yy[t,(0,1)]
    m.yy11DiffCon = Constraint(m.t, rule=_yy11)
    
    def _yy20(m, t):
        return m.dyy[t,(2,0)] == \
    		m.II[(2,0)]+m.sig*m.xi*(m.yy[t,(1,0)]+m.yy[t,(1,1)])-\
    		(m.vt[t]+m.dd[(2,0)]+m.dd[(0,0)])*m.yy[t,(2,0)]
    m.yy20DiffCon = Constraint(m.t, rule=_yy20)
    
    def _yy21(m, t):
        return m.dyy[t,(2,1)] == \
    		m.vt[t]*m.yy[t,(2,0)]-(m.dd[(2,1)]+m.dd[(0,0)])*m.yy[t,(2,1)]
    m.yy21DiffCon = Constraint(m.t, rule=_yy21)
    
    def _yy30(m, t):
        return m.dyy[t,(3,0)] == \
    		m.II[(3,0)]+m.dd[(1,0)]*m.yy[t,(1,0)]+\
                m.dd[(1,1)]*m.yy[t,(1,1)]+m.dd[(2,0)]*m.yy[t,(2,0)]+\
                m.dd[(2,1)]*m.yy[t,(2,1)]-\
                (m.dd[(3,0)]+m.dd[(0,0)])*m.yy[t,(3,0)]
    m.yy30DiffCon = Constraint(m.t, rule=_yy30)
    
    def _yy40(m, t):
        return m.dyy[t, (4,0)] == \
    		m.dd[(3,0)]*m.yy[t,(3,0)]-(m.dd[(4,0)]+\
                m.dd[(0,0)])*m.yy[t,(4,0)]
    m.yy40DiffCon = Constraint(m.t, rule=_yy40)
    
    def _L(m, t):
        return m.dL[t] == \
            exp(-m.rr*t)*(m.aa*(m.kp*m.vp[t]*(m.yy[t,(0,0)]+m.yy[t,(1,0)]) \
            +(m.kt*m.vt[t]*m.yy[t,(2,0)])+sum(m.cc[kk]*m.yy[t,kk] 
                                              for kk in m.ij)) \
            -(1-m.aa)*sum(m.qq[kk]*m.yy[t,kk] for kk in m.ij))
    m.LDiffCon = Constraint(m.t, rule=_L)
   
    return m 
    

def initialize_model(m,n_sim,n_nfe,n_ncp):
    vp_profile = {0:0.75}
    vt_profile = {0:0.75}
    
    
    m.u_input = Suffix(direction=Suffix.LOCAL)
    m.u_input[m.vp] = vp_profile
    m.u_input[m.vt] = vt_profile
    
    sim = Simulator(m, package='scipy')
    tsim, profiles = sim.simulate(numpoints=n_sim, varying_inputs=m.u_input)
    
    
    discretizer = TransformationFactory('dae.collocation')
    discretizer.apply_to(m,nfe=n_nfe,ncp=n_ncp,scheme='LAGRANGE-RADAU')
    
    sim.initialize_model()
    


if __name__=='__main__':
    m = create_model()
    initialize_model(m,10,5,1)

    m.epsDelta = Param(initialize = 0.75001)
    
    q_del={}
    q_del[(0,0)] = 1.001
    q_del[(0,1)] = 1.002
    q_del[(1,0)] = 1.003
    q_del[(1,1)] = 1.004
    q_del[(2,0)] = 0.83001
    q_del[(2,1)] = 0.83002
    q_del[(3,0)] = 0.42001
    q_del[(4,0)] = 0.17001
    m.qqDelta = Param(m.ij, initialize=q_del)
    
    m.aaDelta = Param(initialize = .0001001)
   
    m_sipopt = sipopt(m,[m.eps,m.qq,m.aa],
                        [m.epsDelta,m.qqDelta,m.aaDelta],
                        streamSoln = True)
