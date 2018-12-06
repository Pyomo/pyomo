#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from pyomo.contrib.pynumero.interfaces import PyomoNLP
import matplotlib.pylab as plt
import pyomo.environ as aml
import numpy as np


def create_basic_model():

    m = aml.ConcreteModel()
    m.x = aml.Var([1, 2, 3], domain=aml.Reals)
    for i in range(1, 4):
        m.x[i].value = i
    m.c1 = aml.Constraint(expr=m.x[1] ** 2 - m.x[2] - 1 == 0)
    m.c2 = aml.Constraint(expr=m.x[1] - m.x[3] - 0.5 == 0)
    m.d1 = aml.Constraint(expr=m.x[1] + m.x[2] <= 100.0)
    m.d2 = aml.Constraint(expr=m.x[2] + m.x[3] >= -100.0)
    m.d3 = aml.Constraint(expr=m.x[2] + m.x[3] + m.x[1] >= -500.0)
    m.x[2].setlb(0.0)
    m.x[3].setlb(0.0)
    m.x[2].setub(100.0)
    m.obj = aml.Objective(expr=m.x[2]**2)
    return m

model = create_basic_model()
solver = aml.SolverFactory('ipopt')
solver.solve(model, tee=True)

# build nlp initialized at the solution
nlp = PyomoNLP(model)

# get initial point
print(nlp.variable_order())
x0 = nlp.x_init()

# vectors of finite lower and upper bounds
xl = nlp.xl(condensed=True)
xu = nlp.xu(condensed=True)

# build expansion matrices
Pxl = nlp.expansion_matrix_xl()
Pxu = nlp.expansion_matrix_xu()

# lower and upper bounds residual
res_xl = Pxl.transpose() * x0 - xl
res_xu = xu - Pxu.transpose() * x0
print("Residuals lower bounds x-xl:", res_xl)
print("Residuals upper bounds xu-x:", res_xu)

# evaluate residual of equality constraints
print(nlp.constraint_order())
res_c = nlp.evaluate_c(x0)
print("Residuals equality constraints c(x):", res_c)
# evaluate residual of inequality constraints
d = nlp.evaluate_d(x0)

# compression matrices
Pdl = nlp.expansion_matrix_dl()
Pdu = nlp.expansion_matrix_du()

# vectors of finite lower and upper bounds
dl = nlp.dl(condensed=True)
du = nlp.du(condensed=True)

# lower and upper inequalities residual
res_dl = Pdl.transpose() * d - dl
res_du = du - Pdu.transpose() * d
print("Residuals lower bounds d-dl:", res_dl)
print("Residuals upper bounds du-d:", res_du)

feasible = False
if np.all(res_xl >= 0) and np.all(res_xu >= 0) \
    and np.all(res_dl >= 0) and np.all(res_du >= 0) and \
    np.allclose(res_c, np.zeros(nlp.nc), atol=1e-5):
    feasible = True

print("Is x0 feasible:", feasible)

