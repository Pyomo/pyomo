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
import numpy as np
import scipy as sp


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
    OPTIONS.declare("maxiter", ConfigValue(
        default=100,
        domain=int,
        description="Maximum number of function evaluations per solve",
    ))
    OPTIONS.declare("tol", ConfigValue(
        default=None,
        domain=float,
        description="Tolerance for convergence of function residual",
    ))

    def solve(self, x0=None):
        if x0 is None:
            self._timer.start(timebins.get_primals)
            x0 = self._nlp.get_primals()
            self._timer.stop(timebins.get_primals)

        self._timer.start(timebins.solve)
        x, info, ier, msg = sp.optimize.fsolve(
            self.evaluate_function,
            x0,
            fprime=self.evaluate_jacobian,
            full_output=True,
            xtol=self.options.xtol,
            maxfev=self.options.maxiter,
        )
        self._timer.stop(timebins.solve)

        #
        # fsolve converges with a tolerance specified on the variable
        # vector x. We may also want to enforce a tolerance on function
        # value, which we check here.
        #
        if self.options.tol is not None:
            fcn_val = self.evaluate_function(x)
            if not np.all(np.abs(fcn_val) <= self.options.tol):
                raise RuntimeError(
                    "fsolve converged to a solution that does not satisfy the"
                    " function tolerance 'tol' of %s."
                    " You may need to relax the 'tol' option or tighten the"
                    " 'xtol' option (currently 'xtol' is %s)."
                    % (self.options.tol, self.options.xtol)
                )

        return x, info, ier, msg


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


class PyomoScipySquareSolver(object):

    def solve(self, model):

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
        nlp_solver = self.create_nlp_solver()
        x0 = nlp.get_primals()
        results = nlp_solver.solve(x0=x0)

        # Transfer values back to Pyomo model
        for var, val in zip(nlp.get_pyomo_variables(), nlp.get_primals()):
            var.set_value(val)

        # Translate results into a Pyomo-compatible results structure
        pyomo_results = self.get_pyomo_results(results)

        return pyomo_results

    def get_nlp(self):
        return self._nlp

    def create_nlp_solver(self):
        raise NotImplementedError(
            "%s has not implemented the create_nlp_solver method"
            % self.__class__
        )

    def get_pyomo_results(self):
        raise NotImplementedError(
            "%s has not implemented the get_results method"
            % self.__class__
        )


class PyomoFsolveSolver(PyomoScipySquareSolver):

    def create_nlp_solver(self):
        nlp = self.get_nlp()
        solver = FsolveNlpSolver(nlp)
        return solver

    def get_pyomo_results(self, results):
        return results


class PyomoRootSolver(PyomoScipySquareSolver):

    def create_nlp_solver(self):
        nlp = self.get_nlp()
        solver = RootNlpSolver(nlp)
        return solver

    def get_pyomo_results(self, results):
        return results
