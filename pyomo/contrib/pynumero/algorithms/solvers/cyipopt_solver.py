#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

try:
    import ipopt
except ImportError:
    raise ImportError('ipopt solver relies on cyipopt. Install cyipopt'+
                      ' https://github.com/matthias-k/cyipopt.git')
import numpy as np
from pyomo.contrib.pynumero.interfaces import PyomoNLP
import pyomo.environ as aml
import sys
import os


def redirect_stdout():
    sys.stdout.flush() # <--- important when redirecting to files

    # Duplicate stdout (file descriptor 1)
    # to a different file descriptor number
    newstdout = os.dup(1)

    # /dev/null is used just to discard what is being printed
    devnull = os.open('/dev/null', os.O_WRONLY)

    # Duplicate the file descriptor for /dev/null
    # and overwrite the value for stdout (file descriptor 1)
    os.dup2(devnull, 1)

    # Close devnull after duplication (no longer needed)
    os.close(devnull)

    # Use the original stdout to still be able
    # to print to stdout within python
    sys.stdout = os.fdopen(newstdout, 'w')
    return newstdout


class _CyIpoptProblem(object):

    def __init__(self, nlp):
        self._nlp = nlp
        self._is_composite = False
        if hasattr(nlp, 'nblocks'):
            self._is_composite = True

        x = nlp.x_init()
        y = nlp.y_init()
        y.fill(1.0)

        # get structures
        self._df = nlp.grad_objective(x)
        self._g = nlp.evaluate_g(x)
        if not self._is_composite:
            self._jac_g = nlp.jacobian_g(x)
            self._hess_lag = nlp.hessian_lag(x, y)
            self._hess_lower_mask = self._hess_lag.row >= self._hess_lag.col
        else:
            self._jac_g = nlp.jacobian_g(x)
            expanded = self._jac_g.tocoo()
            self._jac_row = expanded.row
            self._jac_col = expanded.col

            self._hess_lag = nlp.hessian_lag(x, y)
            expanded = self._hess_lag.tocoo()
            self._hess_lower_mask = expanded.row >= expanded.col
            self._hess_row = np.compress(self._hess_lower_mask, expanded.row)
            self._hess_col = np.compress(self._hess_lower_mask, expanded.col)

    def objective(self, x):
        return self._nlp.objective(x)

    def gradient(self, x):
        self._nlp.grad_objective(x, out=self._df)
        if not self._is_composite:
            return self._df
        return self._df.flatten()

    def constraints(self, x):
        self._nlp.evaluate_g(x, out=self._g)
        if not self._is_composite:
            return self._g
        return self._g.flatten()

    def jacobian(self, x):
        self._nlp.jacobian_g(x, out=self._jac_g)
        if not self._is_composite:
            return self._jac_g.data
        return self._jac_g.coo_data()

    def hessianstructure(self):
        if not self._is_composite:
            row = np.compress(self._hess_lower_mask, self._hess_lag.row)
            col = np.compress(self._hess_lower_mask, self._hess_lag.col)
            return row, col
        return self._hess_row, self._hess_col

    def jacobianstructure(self):
        if not self._is_composite:
            return self._jac_g.row, self._jac_g.col
        return self._jac_row, self._jac_col

    def hessian(self, x, lagrange, obj_factor):
        self._nlp.hessian_lag(x,
                              lagrange,
                              out=self._hess_lag,
                              eval_f_c=False,
                              obj_factor=obj_factor)
        if not self._is_composite:
            data = np.compress(self._hess_lower_mask, self._hess_lag.data)
            return data
        data = np.compress(self._hess_lower_mask, self._hess_lag.coo_data())
        return data

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
    ):
        pass


class CyIpoptSolver(object):

    def __init__(self, nlp, options=None):

        self._nlp = nlp

        self._is_composite = False
        if hasattr(nlp, 'nblocks'):
            self._is_composite = True

        self._problem = _CyIpoptProblem(nlp)

        self._options = options
        if options is not None:
            assert isinstance(options, dict)
        else:
            self._options = dict()

    def solve(self, x0=None, tee=False):

        if not self._is_composite:
            cyipopt_solver = ipopt.problem(n=self._nlp.nx,
                                           m=self._nlp.ng,
                                           problem_obj=self._problem,
                                           lb=self._nlp.xl(),
                                           ub=self._nlp.xu(),
                                           cl=self._nlp.gl(),
                                           cu=self._nlp.gu()
                                           )
        else:
            xl = self._nlp.xl()
            xu = self._nlp.xu()
            gl = self._nlp.gl()
            gu = self._nlp.gu()
            nx = int(self._nlp.nx)
            ng = int(self._nlp.ng)
            cyipopt_solver = ipopt.problem(n=nx,
                                           m=ng,
                                           problem_obj=self._problem,
                                           lb=xl.flatten(),
                                           ub=xu.flatten(),
                                           cl=gl.flatten(),
                                           cu=gu.flatten()
                                           )
        if x0 is None:
            xstart = self._nlp.x_init()
            if self._is_composite:
                xstart = xstart.flatten()
        else:
            assert isinstance(x0, np.ndarray)
            assert x0.size == self._nlp.nx
            xstart = x0

        # this is needed until NLP hessian takes obj_factor as an input
        if not self._nlp._future_libraries:
            cyipopt_solver.addOption('nlp_scaling_method', 'none')

        # add options
        for k, v in self._options.items():
            cyipopt_solver.addOption(k, v)

        if tee:
            x, info = cyipopt_solver.solve(xstart)
        else:
            newstdout = redirect_stdout()
            x, info = cyipopt_solver.solve(xstart)
            os.dup2(newstdout, 1)

        return x, info




