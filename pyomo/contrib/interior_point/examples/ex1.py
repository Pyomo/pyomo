import pyomo.environ as pe
from pyomo.contrib.interior_point.interior_point import solve_interior_point
from pyomo.contrib.interior_point.interface import InteriorPointInterface
from pyomo.contrib.interior_point.linalg.mumps_interface import MumpsInterface
import logging


logging.basicConfig(level=logging.INFO)


m = pe.ConcreteModel()
m.x = pe.Var()
m.y = pe.Var()
m.obj = pe.Objective(expr=m.x**2 + m.y**2)
m.c1 = pe.Constraint(expr=m.y == pe.exp(m.x))
m.c2 = pe.Constraint(expr=m.y >= (m.x - 1)**2)
interface = InteriorPointInterface(m)
linear_solver = MumpsInterface()
x, duals_eq, duals_ineq = solve_interior_point(interface, linear_solver)
print(x, duals_eq, duals_ineq)
