from pyomo.environ import *
from pyomo.dae import *
from pyomo.opt import SolverFactory
from sensitivity_toolbox import sipopt
from pyomo.dae.simulator import Simulator


m = ConcreteModel()

m.a = Param(initialize = -0.2, mutable=True)
m.H = Param(initialize = 5, mutable=True)
m.T = Param(initialize = 15, mutable=True)

m.t = ContinuousSet(bounds=(0,1))

m.x = Var(m.t)
m.F = Var(m.t)
m.u = Var(m.t,initialize=-0.06)

m.dx = DerivativeVar(m.x, wrt=m.t)
m.df0 = DerivativeVar(m.F, wrt=m.t)

m.x[0].fix(5)
m.F[0].fix(0)

def _x(m,t):
    return m.dx[t]==m.T*(m.a*m.x[t]+m.u[t])
m.x_dot = Constraint(m.t, rule=_x)

def _f0(m,t):
    return m.df0[t]==m.T*(0.25*m.u[t]**2)
m.FDiffCon = Constraint(m.t, rule=_f0)

def _Cost(m):
    return 0.5*m.H*m.x[1]**2+m.F[1]
m.J = Objective(rule=_Cost)


#####################################
#u_profile = {0:-0.06}
#
#m.u_input = Suffix(direction=Suffix.LOCAL)
#m.u_input[m.u]=u_profile
#
#sim = Simulator(m,package='scipy')
#tsim, profiles = sim.simulate(numpoints=100, varying_inputs=m.u_input)
#
#discretizer = TransformationFacotory('dae.collocation')
#discretizer.apply_to(m, nfe=10, ncp=3, scheme='LAGRANGE-RADAU')
#
#sim.initialize_model()

#discretizer.reduce_collocation_points(m,var=m.u,ncp=1,contset=m.t)

#`solver=SolverFactory('ipopt')
#`results = solver.solve(m, tee=True)
#`

#solver=SolverFactory('ipopt')
#results=solver.solve(m,tee=True)

#m_sipopt, results, z_L, z_U = sipopt(m,[m.eta1,m.eta2],[m.perturbed_eta1,m.perturbed_eta2],streamSoln=True)
