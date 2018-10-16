#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


from __future__ import division

import logging

from pyomo.common.config import (
    ConfigBlock, ConfigValue, In, add_docstring_list
)
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.multistart.high_conf_stop import should_stop
from pyomo.contrib.multistart.reinit import reinitialize_variables
from pyomo.core import Objective, Var, minimize, value
from pyomo.opt import SolverFactory, SolverStatus
from pyomo.opt import TerminationCondition as tc

logger = logging.getLogger('pyomo.contrib.multistart')


@SolverFactory.register('multistart',
                        doc='MultiStart solver for NLPs')
class MultiStart(object):
    """Solver wrapper that initializes at multiple starting points.

    # TODO: also return appropriate duals

    For theoretical underpinning, see
    https://www.semanticscholar.org/paper/How-many-random-restarts-are-enough-Dick-Wong/55b248b398a03dc1ac9a65437f88b835554329e0

    Keyword arguments below are specified for the ``solve`` function.

    """

    CONFIG = ConfigBlock("MultiStart")
    CONFIG.declare("strategy", ConfigValue(
        default="rand", domain=In([
            "rand", "midpoint_guess_and_bound",
            "rand_guess_and_bound", "rand_distributed"]),
        description="Specify the restart strategy. Defaults to rand.",
        doc="""Specify the restart strategy.

        - "rand": random choice between variable bounds
        - "midpoint_guess_and_bound": midpoint between current value and farthest bound
        - "rand_guess_and_bound": random choice between current value and farthest bound
        - "rand_distributed": random choice among evenly distributed values
        """
    ))
    CONFIG.declare("solver", ConfigValue(
        default="ipopt",
        description="solver to use, defaults to ipopt"
    ))
    CONFIG.declare("solver_args", ConfigValue(
        default={},
        description="Dictionary of keyword arguments to pass to the solver."
    ))
    CONFIG.declare("iterations", ConfigValue(
        default=10,
        description="Specify the number of iterations, defaults to 10. "
        "If -1 is specified, the high confidence stopping rule will be used"
    ))
    CONFIG.declare("stopping_mass", ConfigValue(
        default=0.5,
        description="Maximum allowable estimated missing mass of optima.",
        doc="""Maximum allowable estimated missing mass of optima for the
        high confidence stopping rule, only used with the random strategy.
        The lower the parameter, the stricter the rule.
        Value bounded in (0, 1]."""
    ))
    CONFIG.declare("stopping_delta", ConfigValue(
        default=0.5,
        description="1 minus the confidence level required for the stopping rule.",
        doc="""1 minus the confidence level required for the stopping rule for the
        high confidence stopping rule, only used with the random strategy.
        The lower the parameter, the stricter the rule.
        Value bounded in (0, 1]."""
    ))
    CONFIG.declare("suppress_unbounded_warning", ConfigValue(
        default=False, domain=bool,
        description="True to suppress warning for skipping unbounded variables."
    ))
    CONFIG.declare("HCS_max_iterations", ConfigValue(
        default=1000,
        description="Maximum number of iterations before interrupting the high confidence stopping rule."
    ))
    CONFIG.declare("HCS_tolerance", ConfigValue(
        default=0,
        description="Tolerance on HCS objective value equality. Defaults to Python float equality precision."
    ))

    __doc__ = add_docstring_list(__doc__, CONFIG)

    def available(self, exception_flag=True):
        """Check if solver is available.

        TODO: For now, it is always available. However, sub-solvers may not
        always be available, and so this should reflect that possibility.

        """
        return True

    def solve(self, model, **kwds):
        # initialize keyword args
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)

        # initialize the solver
        solver = SolverFactory(config.solver)

        # Model sense
        objectives = model.component_data_objects(Objective, active=True)
        obj = next(objectives, None)
        if next(objectives, None) is not None:
            raise RuntimeError(
                "Multistart solver is unable to handle model with multiple active objectives.")
        if obj is None:
            raise RuntimeError(
                "Multistart solver is unable to handle model with no active objective.")

        # store objective values and objective/result information for best
        # solution obtained
        objectives = []
        obj_sign = 1 if obj.sense == minimize else -1
        best_objective = float('inf') * obj_sign
        best_model = model
        best_result = None

        try:
            # create temporary variable list for value transfer
            tmp_var_list_name = unique_component_name(model, "_vars_list")
            setattr(model, tmp_var_list_name,
                    list(model.component_data_objects(
                        ctype=Var, descend_into=True)))

            best_result = result = solver.solve(model, **config.solver_args)
            if (result.solver.status is SolverStatus.ok and
                    result.solver.termination_condition is tc.optimal):
                obj_val = value(model.obj.expr)
                best_objective = obj_val
                objectives.append(obj_val)
            num_iter = 0
            max_iter = config.iterations
            # if HCS rule is specified, reinitialize completely randomly until
            # rule specifies stopping
            using_HCS = config.iterations == -1
            HCS_completed = False
            if using_HCS:
                assert config.strategy == "rand", \
                    "High confidence stopping rule requires rand strategy."
                max_iter = config.HCS_max_iterations

            while num_iter < max_iter:
                if using_HCS and should_stop(
                        objectives, config.stopping_mass,
                        config.stopping_delta, config.HCS_tolerance):
                    HCS_completed = True
                    break
                num_iter += 1
                # at first iteration, solve the originally passed model
                m = model.clone() if num_iter > 1 else model
                reinitialize_variables(m, config)
                result = solver.solve(m, **config.solver_args)
                if (result.solver.status is SolverStatus.ok and
                        result.solver.termination_condition is tc.optimal):
                    obj_val = value(m.obj.expr)
                    objectives.append(obj_val)
                    if obj_val * obj_sign < obj_sign * best_objective:
                        # objective has improved
                        best_objective = obj_val
                        best_model = m
                        best_result = result
                if num_iter == 1:
                    # if it's the first iteration, set the best_model and
                    # best_result regardless of solution status in case the
                    # model is infeasible.
                    best_model = m
                    best_result = result

            if using_HCS and not HCS_completed:
                logger.warning(
                    "High confidence stopping rule was unable to complete "
                    "after %s iterations. To increase this limit, change the "
                    "HCS_max_iterations flag." % num_iter)

            # if no better result was found than initial solve, then return
            # that without needing to copy variables.
            if best_model is model:
                return best_result

            # reassign the given models vars to the new models vars
            orig_var_list = getattr(model, tmp_var_list_name)
            best_soln_var_list = getattr(best_model, tmp_var_list_name)
            for orig_var, new_var in zip(orig_var_list, best_soln_var_list):
                if not orig_var.is_fixed():
                    orig_var.value = new_var.value

            return best_result
        finally:
            # Remove temporary variable list
            delattr(model, tmp_var_list_name)

    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass
