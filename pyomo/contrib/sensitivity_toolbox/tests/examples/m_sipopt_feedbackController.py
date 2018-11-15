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
# Reference: "Optimal Control Theory: An Introduction", Donald E. Kirk, 
# 		(1970/1998)
#
# Example 5.2-1
#
# x'(t) = a*x(t)+u(t)
# 
# min J(u) = (1/2)*H*x^2(T)+\int_0^T ((1/4)*u^2(t)dt)

from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.simulator import Simulator
from pyomo.contrib.sensitivity_toolbox.sens import sipopt


def create_model():
    m = ConcreteModel()

    m.a = Param(initialize = -0.2, mutable=True)
    m.H = Param(initialize = 0.5, mutable=True)
    m.T = Param(initialize = 15)
    
    m.t = ContinuousSet(bounds=(0,m.T))
    
    m.x = Var(m.t)
    m.F = Var(m.t)
    m.u = Var(m.t,initialize=0, bounds=(-0.2,0))
    
    m.dx = DerivativeVar(m.x, wrt=m.t)
    m.df0 = DerivativeVar(m.F, wrt=m.t)
    
    m.x[0].fix(5)
    m.F[0].fix(0)
    
    def _x(m,t):
        return m.dx[t]==m.a*m.x[t]+m.u[t]
    m.x_dot = Constraint(m.t, rule=_x)
    
    def _f0(m,t):
        return m.df0[t]==0.25*m.u[t]**2
    m.FDiffCon = Constraint(m.t, rule=_f0)
    
    def _Cost(m):
        return 0.5*m.H*m.x[m.T]**2+m.F[m.T]
    m.J = Objective(rule=_Cost)
    
    
    ####################################
    u_profile = {0:-0.06}
    #u_profile = {0:0}
    
    m.u_input = Suffix(direction=Suffix.LOCAL)
    m.u_input[m.u]=u_profile
    
    sim = Simulator(m,package='scipy')
    tsim, profiles = sim.simulate(numpoints=100, varying_inputs=m.u_input)
    
    discretizer = TransformationFactory('dae.collocation')
    discretizer.apply_to(m, nfe=100, ncp=1, scheme='LAGRANGE-RADAU')
    
    sim.initialize_model()
    
    return m
