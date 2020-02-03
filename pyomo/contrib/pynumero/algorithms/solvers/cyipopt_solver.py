#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
"""
The cyipopt_solver module includes the python interface to the 
Cythonized ipopt solver cyipopt (see more: 
https://github.com/matthias-k/cyipopt.git). To use the solver, 
you can create a derived implementation from the abstract base class
CyIpoptProblemInterface that provides the necessary methods.

Note: This module also includes a default implementation CyIpopt
that works with problems derived from AslNLP as long as those 
classes return numpy ndarray objects for the vectors and coo_matrix
objects for the matrices (e.g., AmplNLP and PyomoNLP)
"""
try:
    import ipopt
except ImportError:
    raise ImportError('ipopt solver relies on cyipopt. Install cyipopt'
                      ' https://github.com/matthias-k/cyipopt.git')
import numpy as np
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import six
import sys
import os
import abc


@six.add_metaclass(abc.ABCMeta)
class CyIpoptProblemInterface(abc.ABC):
    @abc.abstractmethod
    def x_init(self):
        """Return the initial values for x as a numpy ndarray
        """
        pass
    
    @abc.abstractmethod
    def x_lb(self):
        """Return the lower bounds on x as a numpy ndarray
        """
        pass

    @abc.abstractmethod
    def x_ub(self):
        """Return the upper bounds on x as a numpy ndarray
        """
        pass

    @abc.abstractmethod
    def g_lb(self):
        """Return the lower bounds on the constraints as a numpy ndarray
        """
        pass

    @abc.abstractmethod
    def g_ub(self):
        """Return the upper bounds on the constraints as a numpy ndarray
        """
        pass

    @abc.abstractmethod
    def objective(self, x):
        """Return the value of the objective
        function evaluated at x
        """
        pass

    @abc.abstractmethod
    def gradient(self, x):
        """Return the gradient of the objective
        function evaluated at x as a numpy ndarray
        """
        pass

    @abc.abstractmethod
    def constraints(self, x):
        """Return the residuals of the constraints
        evaluated at x as a numpy ndarray
        """
        pass

    @abc.abstractmethod
    def jacobianstructure(self):
        """Return the structure of the jacobian
        in coordinate format. That is, return (rows,cols)
        where rows and cols are both numpy ndarray 
        objects that contain the row and column indices
        for each of the nonzeros in the jacobian.
        """
        pass

    @abc.abstractmethod
    def jacobian(self, x):
        """Return the values for the jacobian evaluated at x
        as a numpy ndarray of nonzero values corresponding
        to the rows and columns specified in the jacobianstructure
        """
        pass

    @abc.abstractmethod
    def hessianstructure(self):
        """Return the structure of the hessian
        in coordinate format. That is, return (rows,cols)
        where rows and cols are both numpy ndarray 
        objects that contain the row and column indices
        for each of the nonzeros in the hessian.
        Note: return ONLY the lower diagonal of this symmetric matrix.
        """
        pass

    @abc.abstractmethod
    def hessian(self, x, y, obj_factor):
        """Return the values for the hessian evaluated at x
        as a numpy ndarray of nonzero values corresponding
        to the rows and columns specified in the
        hessianstructure method.
        Note: return ONLY the lower diagonal of this symmetric matrix.
        """
        pass
    
    def intermediate(self, alg_mod, iter_count, obj_value,
            inf_pr, inf_du, mu, d_norm, regularization_size,
            alpha_du, alpha_pr, ls_trials):
        """Callback that can be used to examine or report intermediate
        results. This method is called each iteration
        """
        # TODO: Document the arguments
        pass
    

class CyIpoptNLP(CyIpoptProblemInterface):
    def __init__(self, nlp):
        """This class provides a CyIpoptProblemInterface for use
        with the CyIpoptSolver class that can take in an NLP 
        as long as it provides vectors as numpy ndarrays and 
        matrices as scipy.sparse.coo_matrix objects. This class
        provides the interface between AmplNLP or PyomoNLP objects
        and the CyIpoptSolver
        """
        self._nlp = nlp
        x = nlp.init_primals()
        y = nlp.init_duals()
        if np.any(np.isnan(y)):
            # did not get initial values for y, use this default
            y.fill(1.0)

        self._cached_x = x.copy()
        self._cached_y = y.copy()
        self._cached_obj_factor = 1.0

        nlp.set_primals(self._cached_x)
        nlp.set_duals(self._cached_y)

        # get jacobian and hessian structures
        self._jac_g = nlp.evaluate_jacobian()
        self._hess_lag = nlp.evaluate_hessian_lag()
        self._hess_lower_mask = self._hess_lag.row >= self._hess_lag.col

    def _set_primals_if_necessary(self, x):
        if not np.array_equal(x, self._cached_x):
            self._nlp.set_primals(x)
            self._cached_x = x.copy()

    def _set_duals_if_necessary(self, y):
        if not np.array_equal(y, self._cached_y):
            self._nlp.set_duals(y)
            self._cached_y = y.copy()

    def _set_obj_factor_if_necessary(self, obj_factor):
        if obj_factor != self._cached_obj_factor:
            self._nlp.set_obj_factor(obj_factor)
            self._cached_obj_factor = obj_factor

    def x_init(self):
        return self._nlp.init_primals()

    def x_lb(self):
        return self._nlp.primals_lb()
    
    def x_ub(self):
        return self._nlp.primals_ub()

    def g_lb(self):
        return self._nlp.constraints_lb()

    def g_ub(self):
        return self._nlp.constraints_ub()
    
    def objective(self, x):
        self._set_primals_if_necessary(x)
        return self._nlp.evaluate_objective()

    def gradient(self, x):
        self._set_primals_if_necessary(x)
        return self._nlp.evaluate_grad_objective()

    def constraints(self, x):
        self._set_primals_if_necessary(x)
        return self._nlp.evaluate_constraints()

    def jacobianstructure(self):
        return self._jac_g.row, self._jac_g.col
        
    def jacobian(self, x):
        self._set_primals_if_necessary(x)
        self._nlp.evaluate_jacobian(out=self._jac_g)
        return self._jac_g.data

    def hessianstructure(self):
        row = np.compress(self._hess_lower_mask, self._hess_lag.row)
        col = np.compress(self._hess_lower_mask, self._hess_lag.col)
        return row, col


    def hessian(self, x, y, obj_factor):
        self._set_primals_if_necessary(x)
        self._set_duals_if_necessary(y)
        self._set_obj_factor_if_necessary(obj_factor)
        self._nlp.evaluate_hessian_lag(out=self._hess_lag)
        data = np.compress(self._hess_lower_mask, self._hess_lag.data)
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

class CyIpoptSolver(object):

    def __init__(self, problem_interface, options=None):
        """Create an instance of the CyIpoptSolver. You must
        provide a problem_interface that corresponds to 
        the abstract class CyIpoptProblemInterface

        options can be provided as a dictionary of key value
        pairs
        """
        self._problem = problem_interface

        self._options = options
        if options is not None:
            assert isinstance(options, dict)
        else:
            self._options = dict()

    def solve(self, x0=None, tee=False):
        xl = self._problem.x_lb()
        xu = self._problem.x_ub()
        gl = self._problem.g_lb()
        gu = self._problem.g_ub()

        if x0 is None:
            x0 = self._problem.x_init()
        xstart = x0
        
        nx = len(xstart)
        ng = len(gl)

        cyipopt_solver = ipopt.problem(n=nx,
                                       m=ng,
                                       problem_obj=self._problem,
                                       lb=xl,
                                       ub=xu,
                                       cl=gl,
                                       cu=gu
        )

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
