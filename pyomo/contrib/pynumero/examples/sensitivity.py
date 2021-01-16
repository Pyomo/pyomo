#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
import numpy as np


def create_model(eta1, eta2):
    model = pyo.ConcreteModel()
    # variables
    model.x1 = pyo.Var(initialize=0.15)
    model.x2 = pyo.Var(initialize=0.15)
    model.x3 = pyo.Var(initialize=0.0)
    # parameters
    model.eta1 = pyo.Var()
    model.eta2 = pyo.Var()

    model.nominal_eta1 = pyo.Param(initialize=eta1, mutable=True)
    model.nominal_eta2 = pyo.Param(initialize=eta2, mutable=True)

    # constraints + objective
    model.const1 = pyo.Constraint(expr=6*model.x1+3*model.x2+2*model.x3 - model.eta1 == 0)
    model.const2 = pyo.Constraint(expr=model.eta2*model.x1+model.x2-model.x3-1 == 0)
    model.cost = pyo.Objective(expr=model.x1**2 + model.x2**2 + model.x3**2)
    model.consteta1 = pyo.Constraint(expr=model.eta1 == model.nominal_eta1)
    model.consteta2 = pyo.Constraint(expr=model.eta2 == model.nominal_eta2)

    return model

def compute_init_lam(nlp, x=None, lam_max=1e3):
    if x is None:
        x = nlp.init_primals()
    else:
        assert x.size == nlp.n_primals()
    nlp.set_primals(x)

    assert nlp.n_ineq_constraints() == 0, "only supported for equality constrained nlps for now"

    nx = nlp.n_primals()
    nc = nlp.n_constraints()

    # create Jacobian
    jac = nlp.evaluate_jacobian()

    # create gradient of objective
    df = nlp.evaluate_grad_objective()

    # create KKT system
    kkt = BlockMatrix(2,2)
    kkt.set_block(0, 0, identity(nx))
    kkt.set_block(1, 0, jac)
    kkt.set_block(0, 1, jac.transpose())

    zeros = np.zeros(nc)
    rhs = BlockVector(2)
    rhs.set_block(0, -df)
    rhs.set_block(1, zeros)

    flat_kkt = kkt.tocoo().tocsc()
    flat_rhs = rhs.flatten()

    sol = spsolve(flat_kkt, flat_rhs)
    return sol[nlp.n_primals() : nlp.n_primals() + nlp.n_constraints()]

#################################################################
m = create_model(4.5, 1.0)
opt = pyo.SolverFactory('ipopt')
results = opt.solve(m, tee=True)

#################################################################
nlp = PyomoNLP(m)
x = nlp.init_primals()
y = compute_init_lam(nlp, x=x)
nlp.set_primals(x)
nlp.set_duals(y)

J = nlp.extract_submatrix_jacobian(pyomo_variables=[m.x1, m.x2, m.x3], pyomo_constraints=[m.const1, m.const2])
H = nlp.extract_submatrix_hessian_lag(pyomo_variables_rows=[m.x1, m.x2, m.x3], pyomo_variables_cols=[m.x1, m.x2, m.x3])

M = BlockMatrix(2,2)
M.set_block(0, 0, H)
M.set_block(1, 0, J)
M.set_block(0, 1, J.transpose())

Np = BlockMatrix(2, 1)
Np.set_block(0, 0, nlp.extract_submatrix_hessian_lag(pyomo_variables_rows=[m.x1, m.x2, m.x3], pyomo_variables_cols=[m.eta1, m.eta2]))
Np.set_block(1, 0, nlp.extract_submatrix_jacobian(pyomo_variables=[m.eta1, m.eta2], pyomo_constraints=[m.const1, m.const2]))

ds = spsolve(M.tocsc(), -Np.tocsc())

print("ds:\n", ds.todense())
#################################################################

p0 = np.array([pyo.value(m.nominal_eta1), pyo.value(m.nominal_eta2)])
p = np.array([4.45, 1.05])
dp = p - p0
dx = ds.dot(dp)[0:3]
x_indices = nlp.get_primal_indices([m.x1, m.x2, m.x3])
x_names = np.array(nlp.variable_names())
new_x = x[x_indices] + dx
print("dp:", dp)
print("dx:", dx)
print("Variable names: \n", x_names[x_indices])
print("Sensitivity based x:\n", new_x)

#################################################################
m = create_model(4.45, 1.05)
opt = pyo.SolverFactory('ipopt')
results = opt.solve(m, tee=False)
nlp = PyomoNLP(m)
new_x = nlp.init_primals()
print("NLP based x:\n", new_x[nlp.get_primal_indices([m.x1, m.x2, m.x3])])

