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
from pyomo.contrib.pynumero.sparse import BlockSymMatrix, BlockVector
from pyomo.contrib.pynumero.examples.algorithms.kkt_solver import FullKKTSolver
import pyomo.contrib.pynumero as pn
import pyomo.environ as aml
import numpy as np

if not pn.mumps_available:
    raise ImportError('Need MUMPS to run pynumero interior-point')

class BasicSQPSOlver(object):

    def solve(self, nlp, max_iter=1000, tol=1e-8):
        # get initial guesses for primal and dual variables
        x = nlp.x_init()
        y = nlp.y_init()

        # create linear solver
        lsolver = FullKKTSolver('mumps')

        # create block matrix/vector for kkt and rhs
        kkt = BlockSymMatrix(2)
        rhs = BlockVector(2)
        xy = BlockVector(2)
        xy[0] = x
        xy[1] = y
        obj = None

        for k in range(max_iter):
            # check convergence
            obj = nlp.objective(x)
            grad_lag = nlp.grad_objective(x) + nlp.jacobian_c(x).T.dot(y)
            resid = nlp.evaluate_c(x)

            if np.linalg.norm(resid, ord=np.inf) <= tol and \
               np.linalg.norm(grad_lag, ord=np.inf) <= tol:
                break

            # compute step direction
            kkt[0, 0] = nlp.hessian_lag(x, y)
            kkt[1, 0] = nlp.jacobian_c(x)
            rhs[0] = grad_lag
            rhs[1] = resid
            
            dxy, info = lsolver.solve(kkt, -rhs, nlp)
            xy += dxy

        info = {'objective': obj, 'iterations': k, 'y':y}

        return x, info


if __name__ == "__main__":
    m = aml.ConcreteModel()
    m.x = aml.Var([1, 2], initialize=1.0)
    m.c1 = aml.Constraint(expr=4*m.x[1]**2 + m.x[2]**2 - 8 == 0)
    m.obj = aml.Objective(expr=-2*m.x[1] + m.x[2])
    model = m

    nlp = PyomoNLP(m)

    # solver = aml.SolverFactory('ipopt')
    # solver.solve(m, tee=True)

    solver = BasicSQPSOlver()
    x, info = solver.solve(nlp, tee=True)
    print(x, info)
