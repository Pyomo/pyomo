#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, exp
from pyomo.contrib.interior_point.interior_point import InteriorPointSolver
from pyomo.contrib.interior_point.interface import InteriorPointInterface
from pyomo.contrib.interior_point.linalg.mumps_interface import MumpsInterface
import logging


logging.basicConfig(level=logging.INFO)
# Supposedly this sets the root logger's level to INFO.
# But when linear_solver.logger logs with debug, 
# it gets propagated to a mysterious root logger with
# level NOTSET...

m =  ConcreteModel()
m.x =  Var()
m.y =  Var()
m.obj =  Objective(expr=m.x**2 + m.y**2)
m.c1 =  Constraint(expr=m.y ==  exp(m.x))
m.c2 =  Constraint(expr=m.y >= (m.x - 1)**2)
interface = InteriorPointInterface(m)
linear_solver = MumpsInterface(
#        log_filename='lin_sol.log',
        icntl_options={11: 1}, # Set error level to 1 (most detailed)
        )

ip_solver = InteriorPointSolver(linear_solver)
x, duals_eq, duals_ineq = ip_solver.solve(interface)
print(x, duals_eq, duals_ineq)
