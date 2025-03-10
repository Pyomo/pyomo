#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
"""
The cyipopt_interface module includes the python interface to the
Cythonized ipopt solver cyipopt (see more:
https://github.com/mechmotum/cyipopt.git). To use the interface,
you can create a derived implementation from the abstract base class
CyIpoptProblemInterface that provides the necessary methods.

Note: This module also includes a default implementation CyIpopt
that works with problems derived from AslNLP as long as those
classes return numpy ndarray objects for the vectors and coo_matrix
objects for the matrices (e.g., AmplNLP and PyomoNLP)
"""
import abc
import inspect

from pyomo.common.dependencies import attempt_import, numpy as np, numpy_available
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError


def _cyipopt_importer():
    import cyipopt

    # cyipopt before version 1.0.3 called the problem class "Problem"
    if not hasattr(cyipopt, "Problem"):
        cyipopt.Problem = cyipopt.problem
    # cyipopt before version 1.0.3 put the __version__ flag in the ipopt
    # module (which was deprecated starting in 1.0.3)
    if not hasattr(cyipopt, "__version__"):
        import ipopt

        cyipopt.__version__ = ipopt.__version__
    # Beginning in 1.0.3, STATUS_MESSAGES is in a separate
    # ipopt_wrapper module
    if not hasattr(cyipopt, "STATUS_MESSAGES"):
        import ipopt_wrapper

        cyipopt.STATUS_MESSAGES = ipopt_wrapper.STATUS_MESSAGES
    return cyipopt


cyipopt, cyipopt_available = attempt_import(
    "ipopt",
    error_message="cyipopt solver relies on the ipopt module from cyipopt. "
    "See https://github.com/mechmotum/cyipopt.git for cyipopt "
    "installation instructions.",
    importer=_cyipopt_importer,
)

# If cyipopt is not available, we will use object as our base class for
# CyIpoptProblemInterface so we don't require cyipopt to import from
# this file.
# Note that this *does* trigger the import attempt and therefore is
# moderately time-consuming.
cyipopt_Problem = cyipopt.Problem if cyipopt_available else object


class CyIpoptProblemInterface(cyipopt_Problem, metaclass=abc.ABCMeta):
    """Abstract subclass of ``cyipopt.Problem`` defining an object that can be
    used as an interface to CyIpopt. Subclasses must define all methods necessary
    for the CyIpopt solve and must call this class's ``__init__`` method to
    initialize Ipopt's data structures.

    Note that, if "output_file" is provided as an Ipopt option, the log file
    is open until this object (and thus the underlying Ipopt NLP object) is
    deallocated. To force this deallocation, call the ``close()`` method, which
    is defined by ``cyipopt.Problem``.

    """

    # Flag used to determine whether the underlying IpoptProblem struct
    # has been initialized. This is used to prevent segfaults when calling
    # cyipopt.Problem's solve method if cyipopt.Problem.__init__ hasn't been
    # called.
    _problem_initialized = False

    def __init__(self):
        """Initialize the problem interface

        This method calls ``cyipopt.Problem.__init__``, and *must* be called
        by any subclass's ``__init__`` method. If not, we will segfault when
        we call ``cyipopt.Problem.solve`` from this object.

        """
        if not cyipopt_available:
            raise RuntimeError(
                "cyipopt is required to instantiate CyIpoptProblemInterface"
            )

        # Call cyipopt.Problem.__init__
        xl = self.x_lb()
        xu = self.x_ub()
        gl = self.g_lb()
        gu = self.g_ub()
        nx = len(xl)
        ng = len(gl)
        super(CyIpoptProblemInterface, self).__init__(
            n=nx, m=ng, lb=xl, ub=xu, cl=gl, cu=gu
        )
        # Set a flag to indicate that the IpoptProblem struct has been
        # initialized
        self._problem_initialized = True

    def solve(self, x, lagrange=None, zl=None, zu=None):
        """Solve a CyIpopt Problem

        Checks whether __init__ has been called before calling
        cyipopt.Problem.solve

        """
        lagrange = [] if lagrange is None else lagrange
        zl = [] if zl is None else zl
        zu = [] if zu is None else zu
        # Check a flag to make sure __init__ has been called. This is to prevent
        # segfaults if we try to call solve from a subclass that has not called
        # super().__init__
        #
        # Note that we can still segfault if a user overrides solve and does not
        # call cyipopt.Problem.__init__, but in this case we assume they know what
        # they are doing.
        if not self._problem_initialized:
            raise RuntimeError(
                "Attempting to call cyipopt.Problem.solve when"
                " cyipopt.Problem.__init__ has not been called. This can happen"
                " if a subclass of CyIpoptProblemInterface overrides __init__"
                " without calling CyIpoptProblemInterface.__init__ or setting"
                " the CyIpoptProblemInterface._problem_initialized flag."
            )
        return super(CyIpoptProblemInterface, self).solve(
            x, lagrange=lagrange, zl=zl, zu=zu
        )

    @abc.abstractmethod
    def x_init(self):
        """Return the initial values for x as a numpy ndarray"""
        pass

    @abc.abstractmethod
    def x_lb(self):
        """Return the lower bounds on x as a numpy ndarray"""
        pass

    @abc.abstractmethod
    def x_ub(self):
        """Return the upper bounds on x as a numpy ndarray"""
        pass

    @abc.abstractmethod
    def g_lb(self):
        """Return the lower bounds on the constraints as a numpy ndarray"""
        pass

    @abc.abstractmethod
    def g_ub(self):
        """Return the upper bounds on the constraints as a numpy ndarray"""
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
        ls_trials,
    ):
        """Callback that can be used to examine or report intermediate
        results. This method is called each iteration
        """
        # TODO: Document the arguments
        pass


class CyIpoptNLP(CyIpoptProblemInterface):
    def __init__(self, nlp, intermediate_callback=None, halt_on_evaluation_error=None):
        """This class provides a CyIpoptProblemInterface for use
        with the CyIpoptSolver class that can take in an NLP
        as long as it provides vectors as numpy ndarrays and
        matrices as scipy.sparse.coo_matrix objects. This class
        provides the interface between AmplNLP or PyomoNLP objects
        and the CyIpoptSolver
        """
        self._nlp = nlp
        self._intermediate_callback = intermediate_callback

        cyipopt_has_eval_error = cyipopt_available and hasattr(
            cyipopt, "CyIpoptEvaluationError"
        )
        if halt_on_evaluation_error is None:
            # If using cyipopt >= 1.3, the default is to continue.
            # Otherwise, the default is to halt (because we are forced to).
            #
            # If CyIpopt is not available, we "halt" (re-raise the original
            # exception).
            self._halt_on_evaluation_error = not cyipopt_has_eval_error
        elif not halt_on_evaluation_error and not cyipopt_has_eval_error:
            raise ValueError(
                "halt_on_evaluation_error=False is only supported for cyipopt >= 1.3.0"
            )
        else:
            self._halt_on_evaluation_error = halt_on_evaluation_error

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
        except (AttributeError, NotImplementedError):
            self._hessian_available = False
            self._hess_lag = None
            self._hess_lower_mask = None

        # Call CyIpoptProblemInterface.__init__, which calls
        # cyipopt.Problem.__init__
        super(CyIpoptNLP, self).__init__()

        # Pre-Pyomo 6.8.0, we had no way to pass the cyipopt.Problem object
        # to the user in an intermediate callback. This prevented them from calling
        # the useful get_current_iterate and get_current_violations methods. Now,
        # we support this by adding the Problem object to the args we pass to a user's
        # callback. To preserve backwards compatibility, we inspect the user's
        # callback to infer whether they want this argument. To preserve backwards
        # compatibility if the user asked for variable-length *args, we do not pass
        # the Problem object as an argument in this case.
        # A more maintainable solution may be to force users to accept **kwds if they
        # want "extra info." If we find ourselves continuing to augment this callback,
        # this may be worth considering. -RBP
        self._use_13arg_callback = None
        if self._intermediate_callback is not None:
            signature = inspect.signature(self._intermediate_callback)
            positional_kinds = {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
            }
            positional = [
                param
                for param in signature.parameters.values()
                if param.kind in positional_kinds
            ]
            has_var_args = any(
                p.kind is inspect.Parameter.VAR_POSITIONAL
                for p in signature.parameters.values()
            )

            if len(positional) == 13 and not has_var_args:
                # If *args is expected, we do not use the new callback
                # signature.
                self._use_13arg_callback = True
            elif len(positional) == 12 or has_var_args:
                # If *args is expected, we use the old callback signature
                # for backwards compatibility.
                self._use_13arg_callback = False
            else:
                raise ValueError(
                    "Invalid intermediate callback. A function with either 12 or 13"
                    " positional arguments, or a variable number of arguments, is"
                    " expected."
                )

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
        try:
            self._set_primals_if_necessary(x)
            return self._nlp.evaluate_objective()
        except PyNumeroEvaluationError:
            if self._halt_on_evaluation_error:
                raise
            else:
                raise cyipopt.CyIpoptEvaluationError(
                    "Error in objective function evaluation"
                )

    def gradient(self, x):
        try:
            self._set_primals_if_necessary(x)
            return self._nlp.evaluate_grad_objective()
        except PyNumeroEvaluationError:
            if self._halt_on_evaluation_error:
                raise
            else:
                raise cyipopt.CyIpoptEvaluationError(
                    "Error in objective gradient evaluation"
                )

    def constraints(self, x):
        try:
            self._set_primals_if_necessary(x)
            return self._nlp.evaluate_constraints()
        except PyNumeroEvaluationError:
            if self._halt_on_evaluation_error:
                raise
            else:
                raise cyipopt.CyIpoptEvaluationError("Error in constraint evaluation")

    def jacobianstructure(self):
        return self._jac_g.row, self._jac_g.col

    def jacobian(self, x):
        try:
            self._set_primals_if_necessary(x)
            self._nlp.evaluate_jacobian(out=self._jac_g)
            return self._jac_g.data
        except PyNumeroEvaluationError:
            if self._halt_on_evaluation_error:
                raise
            else:
                raise cyipopt.CyIpoptEvaluationError(
                    "Error in constraint Jacobian evaluation"
                )

    def hessianstructure(self):
        if not self._hessian_available:
            return np.zeros(0), np.zeros(0)

        row = np.compress(self._hess_lower_mask, self._hess_lag.row)
        col = np.compress(self._hess_lower_mask, self._hess_lag.col)
        return row, col

    def hessian(self, x, y, obj_factor):
        if not self._hessian_available:
            raise ValueError("Hessian requested, but not supported by the NLP")

        try:
            self._set_primals_if_necessary(x)
            self._set_duals_if_necessary(y)
            self._set_obj_factor_if_necessary(obj_factor)
            self._nlp.evaluate_hessian_lag(out=self._hess_lag)
            data = np.compress(self._hess_lower_mask, self._hess_lag.data)
            return data
        except PyNumeroEvaluationError:
            if self._halt_on_evaluation_error:
                raise
            else:
                raise cyipopt.CyIpoptEvaluationError(
                    "Error in Lagrangian Hessian evaluation"
                )

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
        ls_trials,
    ):
        """Calls user's intermediate callback

        This method has the call signature expected by CyIpopt. We then extend
        this call signature to provide users of this interface class additional
        functionality. Additional arguments are:

        - The ``NLP`` object that was used to construct this class instance.
          This is useful for querying the variables, constraints, and
          derivatives during the callback.
        - The class instance itself. This is useful for calling the
          ``get_current_iterate`` and ``get_current_violations`` methods, which
          query Ipopt's internal data structures to provide this information.

        """
        if self._intermediate_callback is not None:
            if self._use_13arg_callback:
                # This is the callback signature expected as of Pyomo 6.8.0
                return self._intermediate_callback(
                    self._nlp,
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
                    ls_trials,
                )
            else:
                # This is the callback signature expected pre-Pyomo 6.8.0 and
                # is supported for backwards compatibility.
                return self._intermediate_callback(
                    self._nlp,
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
                    ls_trials,
                )
        return True
