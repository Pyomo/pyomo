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

import six
import sys
import os
import abc

from pyomo.common.dependencies import (
    attempt_import,
    numpy as np, numpy_available,
)
cyipopt, cyipopt_available = attempt_import(
    'ipopt',
    'cyipopt solver relies on cyipopt. Install cyipopt '
    'https://github.com/matthias-k/cyipopt.git'
)
# Because pynumero.interfaces requires numpy, we will leverage deferred
# imports here so that the solver can be registered even when numpy is
# not available.
pyomo_nlp = attempt_import('pyomo.contrib.pynumero.interfaces.pyomo_nlp')[0]
egb = attempt_import('pyomo.contrib.pynumero.interfaces.external_grey_box')[0]

from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.core.base import Block

@six.add_metaclass(abc.ABCMeta)
class CyIpoptProblemInterface(object):
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
    def scaling_factors(self):
        """Return the values for scaling factors as a tuple
        (objective_scaling, x_scaling, g_scaling). Return None
        if the scaling factors are to be ignored
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
        try:
            self._hess_lag = nlp.evaluate_hessian_lag()
            self._hess_lower_mask = self._hess_lag.row >= self._hess_lag.col
            self._hessian_available = True
        except NotImplementedError:
            self._hessian_available = False
            self._hess_lag = None
            self._hess_lower_mask = None

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
    
    def scaling_factors(self):
        obj_scaling = self._nlp.get_obj_scaling()
        x_scaling = self._nlp.get_primals_scaling()
        g_scaling = self._nlp.get_constraints_scaling()
        return obj_scaling, x_scaling, g_scaling

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
        if not self._hessian_available:
            return np.zeros(0), np.zeros(0)

        row = np.compress(self._hess_lower_mask, self._hess_lag.row)
        col = np.compress(self._hess_lower_mask, self._hess_lag.col)
        return row, col


    def hessian(self, x, y, obj_factor):
        if not self._hessian_available:
            raise ValueError("Hessian requested, but not supported by the NLP")
        
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


def _redirect_stdout():
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


def _numpy_vector(val):
    ans = np.array(val, np.float64)
    if len(ans.shape) != 1:
        raise ValueError("expected a vector, but recieved a matrix "
                         "with shape %s" % (ans.shape,))
    return ans


class CyIpoptSolver(object):

    CONFIG = ConfigBlock("cyipopt")
    CONFIG.declare("tee", ConfigValue(
        default=False,
        domain=bool,
        description="Stream solver output to console",
    ))
    CONFIG.declare("x0", ConfigValue(
        default=None,
        domain=_numpy_vector,
        description="Stream solver output to console",
    ))
    CONFIG.declare("options", ConfigBlock(implicit=True))


    def __init__(self, model, options=None, **kwds):
        """Create an instance of the CyIpoptSolver. You must
        provide a problem_interface that corresponds to 
        the abstract class CyIpoptProblemInterface

        options can be provided as a dictionary of key value
        pairs
        """
        if isinstance(model, CyIpoptProblemInterface):
            # If model is already a CyIpoptNLP, then there is nothing to do
            self._problem = model
        elif isinstance(model, Block):
            # If this is a Pyomo model / block, then we need to create
            # the appropriate PyomoNLP, then wrap it in a CyIpoptNLP
            grey_box_blocks = list(model.component_data_objects(
                egb.ExternalGreyBoxBlock, active=True))
            if grey_box_blocks:
                nlp = pyomo_nlp.PyomoGreyBoxNLP(model)
            else:
                nlp = pyomo_nlp.PyomoNLP(model)
            self._problem = CyIpoptNLP(nlp)
        else:
            # Assume that model is some form of NLP that can be passed
            # to CyIpoptNLP
            self._problem = CyIpoptNLP(model)

        # Backwards compatibility: previous versions allowed specifying
        # cyipopt options as a positional argument
        if options is not None:
            if 'options' in kwds:
                kwds['options'].update(options)
            else:
                kwds['options'] = options
        self.config = self.CONFIG(kwds)


    def available(self, exception_flag=False):
        return numpy_available and cyipopt_available


    def version(self):
        return tuple(int(_) for _ in cyipopt.__version__.split('.'))


    def solve(self, **kwds):
        config = self.config(kwds, preserve_implicit=True)

        xl = self._problem.x_lb()
        xu = self._problem.x_ub()
        gl = self._problem.g_lb()
        gu = self._problem.g_ub()

        if config.x0 is None:
            config.x0 = self._problem.x_init()
        
        nx = len(config.x0)
        ng = len(gl)

        cyipopt_solver = cyipopt.problem(
            n=nx,
            m=ng,
            problem_obj=self._problem,
            lb=xl,
            ub=xu,
            cl=gl,
            cu=gu
        )

        # check if we need scaling
        obj_scaling, x_scaling, g_scaling = self._problem.scaling_factors()
        if any((obj_scaling, x_scaling, g_scaling)):
            # need to set scaling factors
            if obj_scaling is None:
                obj_scaling = 1.0
            if x_scaling is None:
                x_scaling = np.ones(nx)
            if g_scaling is None:
                g_scaling = np.ones(ng)

            cyipopt_solver.setProblemScaling(obj_scaling, x_scaling, g_scaling)

        # add options
        for k, v in config.options.items():
            cyipopt_solver.addOption(k, v)

        if config.tee:
            x, info = cyipopt_solver.solve(config.x0)
        else:
            newstdout = _redirect_stdout()
            x, info = cyipopt_solver.solve(config.x0)
            os.dup2(newstdout, 1)

        return x, info
