#from pyomo.contrib.pynumero.linalg.solvers import ma27_solver
from pyomo.contrib.pynumero.sparse import (BlockVector,
                             BlockSymMatrix,
                             COOSymMatrix)
from scipy.sparse.linalg import spsolve
import numpy as np


def compute_init_lam(nlp, x=None, lam_max=1e3):

    if x is None:
        x = nlp.x_init
    else:
        assert x.size == nlp.nx

    assert nlp.nd == 0, "only supported for equality constrained nlps for now"

    nx = nlp.nx
    nc = nlp.nc

    # create Jacobian
    jac_c = nlp.jacobian_g(x)

    # create gradient of objective
    df = nlp.grad_objective(x)

    diag_ones = np.ones(nx)
    irows = np.arange(nx)
    jcols = np.arange(nx)
    eye = COOSymMatrix((diag_ones, (irows, jcols)), shape=(nx, nx))

    # create KKT system
    kkt = BlockSymMatrix(2)
    kkt[0, 0] = eye
    kkt[1, 0] = jac_c

    zeros = np.zeros(nc)
    rhs = BlockVector([-df, zeros])


    flat_kkt = kkt.tofullmatrix().tocsc()
    flat_rhs = rhs.flatten()

    sol = spsolve(flat_kkt, flat_rhs)
    return sol[nlp.nx: nlp.nx + nlp.ng]
    """
    lsolver = ma27_solver.MA27LinearSolver()
    lsolver.do_symbolic_factorization(kkt, include_diagonal=False)
    status = lsolver.do_numeric_factorization(kkt)
    if status != 0:
        raise RuntimeError("Cannot initialize lambdas. singular jacobian")

    sol = lsolver.do_back_solve(rhs)

    if np.linalg.norm(sol[1], ord=np.inf) > lam_max:
        print("larger than lam_max. returning zeros")
        return np.zeros(nc)
    return sol[1]
    """


def grad_x_lagrangian(grad_objective, jacobian, lam):
    return grad_objective + jacobian.transpose()*lam