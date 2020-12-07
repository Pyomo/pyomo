#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.utils import (build_bounds_mask,
                                                     build_compression_matrix,
                                                     full_to_compressed)
import pyomo.environ as pyo
import numpy as np


def create_basic_model():

    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3], domain=pyo.Reals)
    for i in range(1, 4):
        m.x[i].value = i
    m.c1 = pyo.Constraint(expr=m.x[1] ** 2 - m.x[2] - 1 == 0)
    m.c2 = pyo.Constraint(expr=m.x[1] - m.x[3] - 0.5 == 0)
    m.d1 = pyo.Constraint(expr=m.x[1] + m.x[2] <= 100.0)
    m.d2 = pyo.Constraint(expr=m.x[2] + m.x[3] >= -100.0)
    m.d3 = pyo.Constraint(expr=m.x[2] + m.x[3] + m.x[1] >= -500.0)
    m.x[2].setlb(0.0)
    m.x[3].setlb(0.0)
    m.x[2].setub(100.0)
    m.obj = pyo.Objective(expr=m.x[2]**2)
    return m

model = create_basic_model()
solver = pyo.SolverFactory('ipopt')
solver.solve(model, tee=True)

# build nlp initialized at the solution
nlp = PyomoNLP(model)

# get initial point
print(nlp.variable_names())
x0 = nlp.init_primals()

# vectors of lower and upper bounds
xl = nlp.primals_lb()
xu = nlp.primals_ub()

# demonstrate use of compression from full set of bounds
# to only finite bounds using masks
xlb_mask = build_bounds_mask(xl)
xub_mask = build_bounds_mask(xu)
# get the compressed vector
compressed_xl = full_to_compressed(xl, xlb_mask)
compressed_xu = full_to_compressed(xu, xub_mask)
# we can also build compression matrices
Cx_xl = build_compression_matrix(xlb_mask)
Cx_xu = build_compression_matrix(xub_mask)

# lower and upper bounds residual
res_xl = Cx_xl * x0 - compressed_xl
res_xu = compressed_xu - Cx_xu * x0
print("Residuals lower bounds x-xl:", res_xl)
print("Residuals upper bounds xu-x:", res_xu)

# set the value of the primals (we can skip the duals)
# here we set them to the initial values, but we could
# set them to anything
nlp.set_primals(x0)

# evaluate residual of equality constraints
print(nlp.constraint_names())
res_eq = nlp.evaluate_eq_constraints()
print("Residuals of equality constraints:", res_eq)

# evaluate residual of inequality constraints
res_ineq = nlp.evaluate_ineq_constraints()

# demonstrate the use of compression from full set of
# lower and upper bounds on the inequality constraints
# to only the finite values using masks
ineqlb_mask = build_bounds_mask(nlp.ineq_lb())
inequb_mask = build_bounds_mask(nlp.ineq_ub())
# get the compressed vector
compressed_ineq_lb = full_to_compressed(nlp.ineq_lb(), ineqlb_mask)
compressed_ineq_ub = full_to_compressed(nlp.ineq_ub(), inequb_mask)
# we can also build compression matrices
Cineq_ineqlb = build_compression_matrix(ineqlb_mask)
Cineq_inequb = build_compression_matrix(inequb_mask)

# lower and upper inequalities residual
res_ineq_lb = Cineq_ineqlb * res_ineq - compressed_ineq_lb
res_ineq_ub = compressed_ineq_ub - Cineq_inequb*res_ineq
print("Residuals of inequality constraints lower bounds:", res_ineq_lb)
print("Residuals of inequality constraints upper bounds:", res_ineq_ub)

feasible = False
if np.all(res_xl >= 0) and np.all(res_xu >= 0) \
    and np.all(res_ineq_lb >= 0) and np.all(res_ineq_ub >= 0) and \
    np.allclose(res_eq, np.zeros(nlp.n_eq_constraints()), atol=1e-5):
    feasible = True

print("Is x0 feasible:", feasible)
