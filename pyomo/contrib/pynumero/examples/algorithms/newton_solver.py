from scipy.sparse.linalg import spsolve, MatrixRankWarning
from pyomo.contrib.pynumero.interfaces import PyomoNLP
import pyomo.environ as aml
import numpy as np
import warnings
import logging


class NewtonSolver(object):

    """
    Nonlinear system of equations solver

    Attributes
    ----------

    max_iter : int
        Maximum number of Newton-Raphson iterations (default=5000)

    tol : float
        Convergence tolerance (default=1e-6)

    wls : bool
        Flag to enable line search (default=True)

    red_factor : float
        Reduction factor for backtracking line search (default=0.5)

    ls_max_iter: int
        Maximum number of backtracking iterations in line search (default=40)
    """

    def __init__(self, options=None):
        """

        Parameters
        ----------
        options : dict, optional
            dictionary with solver options
        """

        self.max_iter = 5000
        self.tol = 1e-6
        self.wls = True
        self.ls_max_iter = 40
        self.red_factor = 0.5

        if options is not None:
            if not isinstance(options, dict):
                raise RuntimeError("options must be a dictionary")

            for k, v in options.items():
                if not hasattr(self, k):
                    raise RuntimeError("{} is not a parameter of NewtonSolver".format(k))
                else:
                    self.__setattr__(k, v)

    def solve(self, jacobian, residual, x0, tee=False):

        """
        Solves nonlinear system of equations

        Parameters
        ----------
        jacobian: callable
            function that returns sparse matrix of Jacobian of constraints f(x,out=None)

        residual: callable
            function that returns residual of functions f(x,out=None)

        x0: ndarray
            initial guess

        tee: bool, optional
            if true, iterations are printed out

        Returns
        -------

        """
        x = x0
        d = np.zeros(len(x))
        jac = jacobian(x)
        rhs = residual(x)
        alpha = 1.0
        ls = 0
        for i in range(self.max_iter):

            norm_infeasibility = np.linalg.norm(rhs, ord=np.inf)
            norm_d = np.linalg.norm(d, ord=np.inf)

            if i % 10 == 0 and tee:
                print("  iter    inf_pr     ||d||   alpha    ls")
            if tee:
                _str = "{:>4d}     {:>7.2e}  {:>7.2e} {:>8.2e} {:>3d}".format(i,
                                                                         norm_infeasibility,
                                                                         norm_d,
                                                                         alpha,
                                                                         ls)
                print(_str)
            if norm_infeasibility < self.tol:
                if tee:
                    print('Feasible Solution Found')
                return [x, i, 0, 'Solved Successfully']

            # build system of equations
            rhs = residual(x, out=rhs)
            jac = jacobian(x, out=jac)
            mat = jac.tocsr()

            # Call Linear solver
            try:
                d = -spsolve(mat, rhs, use_umfpack=False)
            except MatrixRankWarning:
                if tee:
                    print('Singularity detected')
                return [x, i, 1, 'Singular Jacobian at iteration ' + str(i)]

            # line search
            alpha = 1.0
            if self.wls:
                ls = 0
                for j in range(self.ls_max_iter):
                    x_ = x + alpha * d
                    rhs = residual(x_, out=rhs)
                    if np.linalg.norm(rhs, ord=np.inf) < (1.0 - 1e-4 * alpha) * norm_infeasibility:
                        x = x_
                        ls += 1
                        break
                    else:
                        alpha *= self.red_factor
                    ls += 1

                if ls >= self.ls_max_iter:
                    return [x, i, 1, 'Line search failed at iteration ' + str(i)]

            else:
                x += d

if __name__ == "__main__":

    model = aml.ConcreteModel()
    model.x = aml.Var(initialize=1.0)
    model.y = aml.Var(initialize=1.0)
    model.c1 = aml.Constraint(expr=2.0*model.x**2+model.y**2 == 24.0)
    model.c2 = aml.Constraint(expr=model.x**2-model.y**2 == -12.0)

    nlp = PyomoNLP(model)
    solver = NewtonSolver()
    sol = solver.solve(nlp.jacobian_g,
                       nlp.evaluate_g,
                       nlp.x_init(),
                       tee=True)
