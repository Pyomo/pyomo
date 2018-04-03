
from pyomo.environ import *
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from TRF import TRF

m = ConcreteModel()
m.z = Var(range(3),domain=Reals)
for i in range(3):
   m.z[i] = 2.0
m.x = Var(range(2))
for i in range(2):
   m.x[i] = 2.0
m.x[1] = 1.0

def blackbox(a,b):
   return sin(a-b)

bb = ExternalFunction(blackbox)
eflist = [bb]
m.obj = Objective(expr=  (m.z[0]-1.0)**2 + (m.z[0]-m.z[1])**2 + (m.z[2]-1.0)**2 + (m.x[0]-1.0)**4 + (m.x[1]-1.0)**6)# + bb(m.x[0],m.x[1]) 
m.c1 = Constraint(expr=m.x[0] * m.z[0]**2 + bb(m.x[0],m.x[1]) == 2*sqrt(2.0))
m.c2 = Constraint(expr=m.z[2]**4 * m.z[1]**2 + m.z[1] == 8+sqrt(2.0))




# optTRF = SolverFactory('trustregion')
# optTRF.solve(m,eflist)


# m.pprint()
TRF(m,eflist)
