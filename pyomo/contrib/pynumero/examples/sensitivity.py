#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
import pyomo.environ as aml
from pyomo.contrib.pynumero.interfaces import PyomoNLP
from pyomo.contrib.pynumero.sparse import BlockSymMatrix, BlockMatrix
from pyomo.contrib.pynumero.interfaces.utils import compute_init_lam
from scipy.sparse.linalg import spsolve
import numpy as np


def create_model(eta1, eta2):
    model = aml.ConcreteModel()
    # variables
    model.x1 = aml.Var(initialize=0.15)
    model.x2 = aml.Var(initialize=0.15)
    model.x3 = aml.Var(initialize=0.0)
    # parameters
    model.eta1 = aml.Var()
    model.eta2 = aml.Var()

    model.nominal_eta1 = aml.Param(initialize=eta1, mutable=True)
    model.nominal_eta2 = aml.Param(initialize=eta2, mutable=True)

    # constraints + objective
    model.const1 = aml.Constraint(expr=6*model.x1+3*model.x2+2*model.x3 - model.eta1 == 0)
    model.const2 = aml.Constraint(expr=model.eta2*model.x1+model.x2-model.x3-1 == 0)
    model.cost = aml.Objective(expr=model.x1**2 + model.x2**2 + model.x3**2)
    model.consteta1 = aml.Constraint(expr=model.eta1 == model.nominal_eta1)
    model.consteta2 = aml.Constraint(expr=model.eta2 == model.nominal_eta2)

    return model

#################################################################
m = create_model(4.5, 1.0)
opt = aml.SolverFactory('ipopt')
results = opt.solve(m, tee=True)

#################################################################
nlp = PyomoNLP(m)
x = nlp.x_init()
y = compute_init_lam(nlp, x=x)

J = nlp.jacobian_g(x)
H = nlp.hessian_lag(x, y)

M = BlockSymMatrix(2)
M[0, 0] = H
M[1, 0] = J

Np = BlockMatrix(2, 1)
Np[0, 0] = nlp.hessian_lag(x, y, subset_variables_col=[m.eta1, m.eta2])
Np[1, 0] = nlp.jacobian_g(x, subset_variables=[m.eta1, m.eta2])

ds = spsolve(M.tocsc(), Np.tocsc())
print(nlp.variable_order())

#################################################################

p0 = np.array([aml.value(m.nominal_eta1), aml.value(m.nominal_eta2)])
p = np.array([4.45, 1.05])
dp = p - p0
dx = ds.dot(dp)[0:nlp.nx]
new_x = x + dx
print(new_x)

#################################################################
m = create_model(4.45, 1.05)
opt = aml.SolverFactory('ipopt')
results = opt.solve(m, tee=True)
nlp = PyomoNLP(m)
print(nlp.x_init())
