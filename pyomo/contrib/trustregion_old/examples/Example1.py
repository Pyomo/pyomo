#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.environ import ConcreteModel, Var, Reals, ExternalFunction, sin, sqrt, Constraint, Objective
from pyomo.opt import SolverFactory

m = ConcreteModel()
m.z = Var(range(3), domain=Reals, initialize=2.)
m.x = Var(range(2), initialize=2.)
m.x[1] = 1.0

def blackbox(a,b):
   return sin(a-b)
bb = ExternalFunction(blackbox)

m.obj = Objective(
   expr=(m.z[0]-1.0)**2 + (m.z[0]-m.z[1])**2 + (m.z[2]-1.0)**2 \
       + (m.x[0]-1.0)**4 + (m.x[1]-1.0)**6 # + m.bb(m.x[0],m.x[1])
)
m.c1 = Constraint(expr=m.x[0] * m.z[0]**2 + bb(m.x[0],m.x[1]) == 2*sqrt(2.0))
m.c2 = Constraint(expr=m.z[2]**4 * m.z[1]**2 + m.z[1] == 8+sqrt(2.0))

m.pprint()

optTRF = SolverFactory('trustregion')
optTRF.solve(m, [bb])

m.display()
