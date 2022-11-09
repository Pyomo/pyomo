#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from collections import namedtuple
from pyomo.core.base.objective import Objective
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.modeling import unique_component_name
from pyomo.common.config import ConfigBlock, ConfigValue, In
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
    DenseSquareNlpSolver,
)
from pyomo.opt import (
    SolverStatus,
    SolverResults,
    TerminationCondition,
    ProblemSense,
)
from pyomo.common.dependencies import (
    attempt_import,
    numpy as np, numpy_available,
    scipy as sp, scipy_available,
)
# Use attempt_import here so that we can register the solver even if SciPy is
# not available.
pyomo_nlp, _ = attempt_import("pyomo.contrib.pynumero.interfaces.pyomo_nlp")


class FsolveNlpSolver(DenseSquareNlpSolver):

    OPTIONS = ConfigBlock(
        description="Options for SciPy fsolve wrapper",
    )
    OPTIONS.declare("xtol", ConfigValue(
        default=1e-8,
        domain=float,
        description="Tolerance for convergence of variable vector",
    ))
    OPTIONS.declare("maxfev", ConfigValue(
        default=100,
        domain=int,
        description="Maximum number of function evaluations per solve",
    ))
    OPTIONS.declare("tol", ConfigValue(
        default=None,
        domain=float,
        description="Tolerance for convergence of function residual",
    ))
    OPTIONS.declare("full_output", ConfigValue(
        default=True,
        domain=bool
    ))

    def solve(self, x0=None):
        if x0 is None:
            x0 = self._nlp.get_primals()

        res = sp.optimize.fsolve(
            self.evaluate_function,
            x0,
            fprime=self.evaluate_jacobian,
            full_output=self.options.full_output,
            xtol=self.options.xtol,
            maxfev=self.options.maxfev,
        )
        if self.options.full_output:
            x, info, ier, msg = res
        else:
            x, ier, msg = res

        #
        # fsolve converges with a tolerance specified on the variable
        # vector x. We may also want to enforce a tolerance on function
        # value, which we check here.
        #
        if self.options.tol is not None:
            if self.options.full_output:
                fcn_val = info["fvec"]
            else:
                fcn_val = self.evaluate_function(x)
            if not np.all(np.abs(fcn_val) <= self.options.tol):
                raise RuntimeError(
                    "fsolve converged to a solution that does not satisfy the"
                    " function tolerance 'tol' of %s."
                    " You may need to relax the 'tol' option or tighten the"
                    " 'xtol' option (currently 'xtol' is %s)."
                    % (self.options.tol, self.options.xtol)
                )

        return res


class RootNlpSolver(DenseSquareNlpSolver):

    OPTIONS = ConfigBlock(
        description="Options for SciPy fsolve wrapper",
    )
    OPTIONS.declare("tol", ConfigValue(
        default=1e-8,
        domain=float,
        description="Convergence tolerance",
    ))
    OPTIONS.declare("method", ConfigValue(
        default="hybr",
        # NOTE: Only supporting Powell hybrid method and Levenberg-Marquardt
        # methods (both from MINPACK) for now.
        domain=In({"hybr", "lm"}),
        #domain=str,
        description="Method used to solve for the function root",
    ))

    def solve(self, x0=None):
        if x0 is None:
            x0 = self._nlp.get_primals()

        results = sp.optimize.root(
            self.evaluate_function,
            x0,
            jac=self.evaluate_jacobian,
        )
        return results


class NewtonNlpSolver(DenseSquareNlpSolver):

    OPTIONS = ConfigBlock(
        description="Options for SciPy newton wrapper",
    )
    OPTIONS.declare("tol", ConfigValue(
        default=1e-8,
        domain=float,
        description="Convergence tolerance",
    ))
    OPTIONS.declare("secant", ConfigValue(
        default=False,
        domain=bool,
        description="Whether to use SciPy's secant method",
    ))
    OPTIONS.declare("full_output", ConfigValue(
        default=True,
        domain=bool,
        description="Whether underlying solver should return its full output",
    ))
    OPTIONS.declare("maxiter", ConfigValue(
        default=50,
        domain=int,
        description="Maximum number of function evaluations per solve",
    ))

    def __init__(self, nlp, timer=None, options=None):
        super().__init__(nlp, timer=timer, options=options)
        if nlp.n_primals() != 1:
            raise RuntimeError(
                "Cannot use the scipy.optimize.newton solver on an NLP with"
                " more than one variable and equality constraint. Got %s"
                " primals. Please use RootNlpSolver or FsolveNlpSolver instead."
            )

    def solve(self, x0=None):
        self._timer.start("solve")
        if x0 is None:
            x0 = self._nlp.get_primals()

        if self.options.secant:
            fprime = None
        else:
            fprime = lambda x: self.evaluate_jacobian(np.array([x]))[0, 0]
        results = sp.optimize.newton(
            lambda x: self.evaluate_function(np.array([x]))[0],
            x0[0],
            fprime=fprime,
            tol=self.options.tol,
            full_output=self.options.full_output,
            maxiter=self.options.maxiter,
        )
        self._timer.stop("solve")
        return results


class SecantNewtonNlpSolver(NewtonNlpSolver):

    OPTIONS = ConfigBlock(
        description="Options for SciPy newton wrapper",
    )
    OPTIONS.declare("tol", ConfigValue(
        default=1e-8,
        domain=float,
        description="Convergence tolerance",
    ))
    OPTIONS.declare("secant_iter", ConfigValue(
        default=2,
        domain=int,
        description=(
            "Number of secant iterations to perform before switching"
            " to Newton's method."
        ),
    ))

    # TODO: Check that NLP is one-dimensional
    def solve(self, x0=None):
        self._timer.start("solve")
        if x0 is None:
            x0 = self._nlp.get_primals()

        try:
            results = sp.optimize.newton(
                lambda x: self.evaluate_function(np.array([x]))[0],
                x0[0],
                fprime=None,
                tol=self.options.tol,
                maxiter=self.options.secant_iter,
            )
        except RuntimeError:
            x0 = self._nlp.get_primals()
            results = sp.optimize.newton(
                lambda x: self.evaluate_function(np.array([x]))[0],
                x0[0],
                fprime=lambda x: self.evaluate_jacobian(np.array([x]))[0, 0],
                tol=self.options.tol,
            )
        self._timer.stop("solve")
        return results


class PyomoScipySolver(object):

    def __init__(self, options=None):
        if options is None:
            options = {}
        self._nlp = None
        self._nlp_solver = None
        self._full_output = None
        self.options = options

    def available(self, exception_flag=False):
        return bool(numpy_available and scipy_available)

    def license_is_valid(self):
        return True

    def version(self):
        return tuple(int(_) for _ in sp.__version__.split('.'))

    def set_options(self, options):
        self.options = options

    def solve(self, model, timer=None):
        if timer is None:
            timer = HierarchicalTimer()
        self._timer = timer
        self._timer.start("solve")
        active_objs = list(model.component_data_objects(Objective, active=True))
        if len(active_objs) == 0:
            obj_name = unique_component_name(model, "_obj")
            obj = Objective(expr=0.0)
            model.add_component(obj_name, obj)

        nlp = pyomo_nlp.PyomoNLP(model)
        self._nlp = nlp

        if len(active_objs) == 0:
            model.del_component(obj_name)

        # Call to solve(nlp)
        self._nlp_solver = self.create_nlp_solver(options=self.options)
        x0 = nlp.get_primals()
        results = self._nlp_solver.solve(x0=x0)

        # Transfer values back to Pyomo model
        for var, val in zip(nlp.get_pyomo_variables(), nlp.get_primals()):
            var.set_value(val)

        self._timer.stop("solve")

        # Translate results into a Pyomo-compatible results structure
        pyomo_results = self.get_pyomo_results(model, results)

        return pyomo_results

    def get_nlp(self):
        return self._nlp

    def create_nlp_solver(self, **kwds):
        raise NotImplementedError(
            "%s has not implemented the create_nlp_solver method"
            % self.__class__
        )

    def get_pyomo_results(self, model, scipy_results):
        raise NotImplementedError(
            "%s has not implemented the get_results method"
            % self.__class__
        )

    #
    # Support "with" statements.
    #
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass


class PyomoFsolveSolver(PyomoScipySolver):

    _term_cond = {
        1: TerminationCondition.feasible,
    }

    def create_nlp_solver(self, **kwds):
        nlp = self.get_nlp()
        solver = FsolveNlpSolver(nlp, **kwds)
        return solver

    def get_pyomo_results(self, model, scipy_results):
        nlp = self.get_nlp()
        if self._nlp_solver.options.full_output:
            x, info, ier, msg = scipy_results
        else:
            x, ier, msg = scipy_results
        results = SolverResults()

        # Record problem data
        results.problem.name = model.name
        results.problem.number_of_constraints = nlp.n_eq_constraints()
        results.problem.number_of_variables = nlp.n_primals()
        results.problem.number_of_binary_variables = 0
        results.problem.number_of_integer_variables = 0
        results.problem.number_of_continuous_variables = nlp.n_primals()

        # Record solver data
        results.solver.name = "fsolve"
        results.solver.return_code = ier
        results.solver.message = msg
        results.solver.wallclock_time = self._timer.timers["solve"].total_time
        results.solver.termination_condition = self._term_cond.get(
            ier, TerminationCondition.error
        )
        results.solver.status = TerminationCondition.to_solver_status(
            results.solver.termination_condition
        )
        if self._nlp_solver.options.full_output:
            results.solver.number_of_function_evaluations = info["nfev"]
            results.solver.number_of_gradient_evaluations = info["njev"]
        return results


class PyomoRootSolver(PyomoScipySolver):

    _term_cond = {
        1: TerminationCondition.feasible,
    }

    def create_nlp_solver(self, **kwds):
        nlp = self.get_nlp()
        solver = RootNlpSolver(nlp, **kwds)
        return solver

    def get_pyomo_results(self, model, scipy_results):
        nlp = self.get_nlp()
        results = SolverResults()

        # Record problem data
        results.problem.name = model.name
        results.problem.number_of_constraints = nlp.n_eq_constraints()
        results.problem.number_of_variables = nlp.n_primals()
        results.problem.number_of_binary_variables = 0
        results.problem.number_of_integer_variables = 0
        results.problem.number_of_continuous_variables = nlp.n_primals()

        # Record solver data
        results.solver.name = "root"
        results.solver.return_code = scipy_results.status
        results.solver.message = scipy_results.message
        results.solver.wallclock_time = self._timer.timers["solve"].total_time
        results.solver.termination_condition = self._term_cond.get(
            scipy_results.status, TerminationCondition.error
        )
        results.solver.status = TerminationCondition.to_solver_status(
            results.solver.termination_condition
        )
        # This attribute is in the SciPy documentation but appears not to
        # be implemented for "hybr" or "lm" solvers...
        #results.solver.number_of_iterations = scipy_results.nit
        results.solver.number_of_function_evaluations = scipy_results.nfev
        results.solver.number_of_gradient_evaluations = scipy_results.njev

        return results


class PyomoNewtonSolver(PyomoScipySolver):

    def create_nlp_solver(self, **kwds):
        nlp = self.get_nlp()
        solver = NewtonNlpSolver(nlp, **kwds)
        return solver

    def get_pyomo_results(self, model, scipy_results):
        nlp = self.get_nlp()
        results = SolverResults()

        if self._nlp_solver.options.full_output:
            root, res = scipy_results
        else:
            root = scipy_results

        # Record problem data
        results.problem.name = model.name
        results.problem.number_of_constraints = nlp.n_eq_constraints()
        results.problem.number_of_variables = nlp.n_primals()
        results.problem.number_of_binary_variables = 0
        results.problem.number_of_integer_variables = 0
        results.problem.number_of_continuous_variables = nlp.n_primals()

        # Record solver data
        results.solver.name = "scipy.newton"

        results.solver.wallclock_time = self._timer.timers["solve"].total_time

        if self._nlp_solver.options.full_output:
            # We only have access to any of this information if the solver was
            # requested to return its full output.

            # For this solver, res.flag is a string.
            # If successful, it is 'converged'
            results.solver.message = res.flag

            if res.converged:
                term_cond = TerminationCondition.feasible
            else:
                term_cond = TerminationCondition.Error
            results.solver.termination_condition = term_cond
            results.solver.status = TerminationCondition.to_solver_status(
                results.solver.termination_condition
            )

            results.solver.number_of_function_evaluations = res.function_calls
        return results
