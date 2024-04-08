# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:55:46 2023

@author: jlgearh
"""

import pyomo.contrib.alternative_solutions.tests.test_cases as tc
from pyomo.contrib.alternative_solutions import lp_enum
import pyomo.environ as pe

m = tc.get_3d_polyhedron_problem()
m.o.deactivate()
m.obj = pe.Objective(expr=m.x[0] + m.x[1] + m.x[2])
sols = lp_enum.enumerate_linear_solutions(m, solver="gurobi")


n = tc.get_pentagonal_pyramid_mip()
n.o.sense = pe.minimize
n.x.domain = pe.Reals
n.y.domain = pe.Reals
sols = lp_enum.enumerate_linear_solutions(n, solver="gurobi")
n.pprint()
