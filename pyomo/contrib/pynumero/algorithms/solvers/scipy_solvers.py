from collections import namedtuple
from pyomo.core.base.objective import Objective
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.modeling import unique_component_name
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
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
    numpy as np, numpy_available,
    scipy as sp, scipy_available,
)


TimeBins = namedtuple("TimeBins", ["get_primals", "solve"])
timebins = TimeBins("get_primals", "solve")


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
            self._timer.start(timebins.get_primals)
            x0 = self._nlp.get_primals()
            self._timer.stop(timebins.get_primals)

        self._timer.start(timebins.solve)
        res = sp.optimize.fsolve(
            self.evaluate_function,
            x0,
            fprime=self.evaluate_jacobian,
            full_output=self.options.full_output,
            xtol=self.options.xtol,
            maxfev=self.options.maxfev,
        )
        self._timer.stop(timebins.solve)
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

    def solve(self, x0=None):
        if x0 is None:
            self._timer.start(timebins.get_primals)
            x0 = self._nlp.get_primals()
            self._timer.stop(timebins.get_primals)

        self._timer.start(timebins.solve)
        results = sp.optimize.root(
            self.evaluate_function,
            x0,
            jac=self.evaluate_jacobian,
        )
        self._timer.stop(timebins.solve)
        return results


class PyomoScipySolver(object):

    def __init__(self, options=None):
        if options is None:
            options = {}
        self._nlp = None
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

        nlp = PyomoNLP(model)
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

    def create_nlp_solver(self):
        nlp = self.get_nlp()
        solver = RootNlpSolver(nlp)
        return solver

    def get_pyomo_results(self, results):
        return results
