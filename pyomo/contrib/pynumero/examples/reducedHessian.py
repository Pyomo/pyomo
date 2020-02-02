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
from pyomo.contrib.pynumero.sparse import BlockSymMatrix, BlockMatrix, BlockVector
from pyomo.contrib.pynumero.interfaces.utils import extract_submatrix
from scipy.sparse import identity
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, inv
import numpy as np

"""
This example demonstrates two different ways to compute the reduced hessian for a nonlinear equality constrained optimization problem. The first approach relies on the calculation of the null space matrix Z. The second approach uses the KKT matrix to compute the reduced matrix by performing a sequence of inexpensive backsolves. 

The sample can be found in section 3.2 of V.Zavala PhD thesis https://pdfs.semanticscholar.org/469b/ecbd5b2413b115cfe6afd6255986e5c651dc.pdf
"""

def create_model():
    model = pyo.ConcreteModel()
    # variables
    model.x1 = pyo.Var(initialize=1.0)
    model.x2 = pyo.Var(initialize=1.0)
    model.x3 = pyo.Var(initialize=1.0)
    
    # constraints + objective
    model.const1 = pyo.Constraint(expr=model.x1+2*model.x2+3*model.x3 == 0)
    model.obj = pyo.Objective(expr=(model.x1-1)**2+(model.x2-2)**2+(model.x3-3)**2)
    return model

m = create_model()
nlp = PyomoNLP(m)
x = nlp.init_primals()
y = nlp.init_duals()

nlp.set_primals(x)
nlp.set_duals(y)

J = nlp.evaluate_jacobian()
H = nlp.evaluate_hessian_lag()

kkt = BlockSymMatrix(2)
kkt[0, 0] = H
kkt[1, 0] = J

d_vars = [m.x2, m.x3]
nd = len(d_vars)
Ad = nlp.extract_submatrix_jacobian(pyomo_variables=d_vars,
                                    pyomo_constraints=[m.const1])
xd_indices = nlp.get_primal_indices(d_vars)
b_vars = [m.x1]
nb= len(b_vars)
Ab = nlp.extract_submatrix_jacobian(pyomo_variables=b_vars,
                                    pyomo_constraints=[m.const1])
xb_indices = nlp.get_primal_indices(b_vars)

# null space matrix
Z = BlockMatrix(2,1)
Z[0,0] = spsolve(-Ab.tocsc(), Ad.tocsc())
Z[1,0] = identity(nd)
Z_sparse = Z.tocsr()
print("Null space matrix:\n",Z.toarray())

# computing reduced hessian with null space matriz
reduced_hessian = Z_sparse.T * H * Z_sparse
print("Reduced hessian matrix:\n",reduced_hessian.toarray())
print("Inverse reduced hessian:\n", inv(reduced_hessian).toarray())

# computing the reduced hessian with back solves
kkt_matrix = kkt.tocsc()
nvars = kkt_matrix.shape[1]
row = xd_indices
col = np.arange(nd)
data = np.ones(nd)

rhs = coo_matrix((data, (row, col)), shape=(nvars, nd))
backsolves = spsolve(kkt_matrix, rhs.tocsc())
reduced_hessian = extract_submatrix(backsolves, xd_indices, np.arange(nd))
print("Inverse reduced hessian matrix (kkt backsolves):\n",reduced_hessian.toarray())

    
