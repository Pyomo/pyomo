# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________


import logging

from pyomo.common.config import (
    ConfigBlock,
    ConfigDict,
    ConfigValue,
    In,
    document_kwargs_from_configdict,
    document_class_CONFIG,
    document_configdict,
    ADVANCED_OPTION,
)

from typing import Any, Optional
import datetime

import pyomo.environ as pyo
from pyomo.common.timing import HierarchicalTimer, default_timer
from pyomo.common.modeling import unique_component_name
from pyomo.common.dependencies import numpy as np
from pyomo.contrib.multistart.high_conf_stop import should_stop
from pyomo.contrib.multistart.reinit import reinitialize_variables, strategies
from pyomo.core import Objective, Constraint, Var, minimize, value
from pyomo.contrib.solver.common.base import SolverBase
from pyomo.contrib.solver.common.config import SolverConfig
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.contrib.solver.common.results import (
    Results,
    TerminationCondition,
    SolutionStatus,
)

from pyomo.contrib.solver.common.util import NoOptimalSolutionError, NoSolutionError
from pyomo.util.vars_from_expressions import get_vars_from_components

from pyomo.common.dependencies.scipy import stats
from pyomo.core.staleflag import StaleFlagManager

logger = logging.getLogger('pyomo.contrib.multistart')


@document_configdict()
class MultistartConfig(SolverConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super().__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.strategy = self.declare(
            "strategy",
            ConfigValue(
                default="rand",
                domain=In(strategies.keys()),
                description="Specify the restart strategy. Defaults to rand.",
                doc="""Specify the restart strategy.

            - "rand": random choice between variable bounds
            - "rand_vector": random choice, vectorized approach with sampler
            - "midpoint_guess_and_bound": midpoint between current value and farthest bound
            - "rand_guess_and_bound": random choice between current value and farthest bound
            - "rand_distributed": random choice among evenly distributed values
            - "midpoint": exact midpoint between the bounds. If using this option, multiple iterations are useless.
            """,
            ),
        )
        self.subsolver = self.declare(
            "subsolver",
            ConfigValue(
                default="ipopt",
                description="solver to use, defaults to ipopt"
                "Accepts solver names as string or solver objects.",
            ),
        )
        self.subsolver_args = self.declare(
            "subsolver_args",
            ConfigValue(
                default={},
                description="Dictionary of keyword arguments to pass to the solver.",
            ),
        )
        self.iterations = self.declare(
            "iterations",
            ConfigValue(
                default=10,
                description="Specify the number of iterations, defaults to 10. "
                "If -1 is specified, the high confidence stopping rule will be used",
            ),
        )
        self.stopping_mass = self.declare(
            "stopping_mass",
            ConfigValue(
                default=0.5,
                description="Maximum allowable estimated missing mass of optima.",
                doc="""Maximum allowable estimated missing mass of optima for the
            high confidence stopping rule, only used with the random strategy.
            The lower the parameter, the stricter the rule.
            Value bounded in (0, 1].""",
            ),
        )
        self.stopping_delta = self.declare(
            "stopping_delta",
            ConfigValue(
                default=0.5,
                description="1 minus the confidence level required for the stopping rule.",
                doc="""1 minus the confidence level required for the stopping rule for the
            high confidence stopping rule, only used with the random strategy.
            The lower the parameter, the stricter the rule.               visibility=DEVELOPER_OPTION,
            Value bounded in (0, 1].""",
            ),
        )
        self.suppress_unbounded_warning = self.declare(
            "suppress_unbounded_warning",
            ConfigValue(
                default=False,
                domain=bool,
                description="True to suppress warning for skipping unbounded variables.",
            ),
        )
        self.HCS_max_iterations = self.declare(
            "HCS_max_iterations",
            ConfigValue(
                default=1000,
                description="Maximum number of iterations before interrupting the high confidence stopping rule.",
            ),
        )
        self.HCS_tolerance = self.declare(
            "HCS_tolerance",
            ConfigValue(
                default=0,
                description="Tolerance on HCS objective value equality. Defaults to Python float equality precision.",
            ),
        )
        self.break_on_solution = self.declare(
            "break_on_solution",
            ConfigValue(
                default=False,
                description="Condition to break if a feasible or optimal solution is found. Defaults to False.",
            ),
        )
        self.sampling_method = self.declare(
            "sampling_method",
            ConfigValue(
                default="random_uniform",
                description="Method for sampling random starting points for reinitialization step. "
                "Supported options are 'random_uniform', 'latin_hypercube', and 'sobol_sampling'. ",
            ),
        )
        self.seed = self.declare(
            "seed",
            ConfigValue(
                default=None,
                description="Seed for reproducibility in random sampling methods.",
            ),
        )
        self.rng = self.declare(
            "rng",
            ConfigValue(
                default=None,
                description="Random number generator for reproducibility in random sampling methods. \
                    Preferred over seed.",
            ),
        )


class MultiStartResults(Results):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super().__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )
        self.feasible_solution_list: Optional[list] = self.declare(
            'feasible_solution_list',
            ConfigValue(
                description="Object for loading the solution back into the model."
            ),
        )
        self.feasible_iter_list: Optional[list] = self.declare(
            'feasible_iter_list',
            ConfigValue(
                description="Object for loading the solution back into the model."
            ),
        )


@SolverFactory.register('multistart', doc='MultiStart solver for NLPs')
@document_class_CONFIG(methods=['solve'])
class MultiStart(SolverBase):
    """Solver wrapper that initializes at multiple starting points.

    # TODO: also return appropriate duals

    For theoretical underpinning, see
    https://www.semanticscholar.org/paper/How-many-random-restarts-are-enough-Dick-Wong/55b248b398a03dc1ac9a65437f88b835554329e0

    Keyword arguments below are specified for the ``solve`` function.

    """

    CONFIG = MultistartConfig()

    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)

        #: Instance configuration;
        self.config = self.config

    def available(self, exception_flag=True):
        """Check if solver is available.

        The multistart solver wrapper should always be available,
        The subsolver availability will be checked after it is assigned in solve()
        """
        return True

    def version(self):
        """Get solver version."""
        """
            Original implementation: 0.1.0,
            Current implementation: 0.2.0,
        """
        current = (0, 2, 0)
        return current

    def license_is_valid(self):
        return True

    def solve(self, model, **kwds):
        start_time = default_timer()

        # initialize keyword args
        config = self.config(kwds.pop('options', {}))
        config.set_value(kwds)

        timer = config.timer
        if timer is None:
            timer = config.timer = HierarchicalTimer()

        # Create centralized sampler once
        sampler = SamplingManager(
            method=config.sampling_method, rng=config.rng, seed=config.seed
        )

        # Allocate the results object so we can populate it as we go
        results = MultiStartResults()
        results.solver_name = self.name
        results.solver_version = self.version()
        results.timing_info.start_timestamp = datetime.datetime.now(
            datetime.timezone.utc
        )

        # Define solver using either string input or provided solver object
        if isinstance(config.subsolver, str):
            self.subsolver = SolverFactory(config.subsolver)

        else:
            self.subsolver = config.subsolver

        if not self.subsolver.available():
            raise RuntimeError(
                f"Selected subsolver '{config.subsolver}' is not available."
            )

        subsolver_args = dict(config.subsolver_args)
        self.remaining_time_limit = config.time_limit

        if self.remaining_time_limit is not None:
            self._timing_lim = True
            if self.remaining_time_limit == 0:
                results.termination_condition = TerminationCondition.maxTimeLimit
                results.solution_status = SolutionStatus.noSolution
                logger.warning(
                    "Time limit set to 0 seconds. Multistart solver did not run."
                )
                return results

            if self.subsolver.config.time_limit is None:
                if "time_limit" not in subsolver_args:
                    subsolver_args["time_limit"] = self.remaining_time_limit
                else:
                    subsolver_args["time_limit"] = min(
                        subsolver_args["time_limit"], self.remaining_time_limit
                    )
            else:
                subsolver_args["time_limit"] = min(
                    self.subsolver.config.time_limit, self.remaining_time_limit
                )
        else:
            self._timing_lim = False

        subsolver_args["load_solutions"] = False
        subsolver_args["raise_exception_on_nonoptimal_result"] = False

        # Model sense
        objectives = list(model.component_data_objects(Objective, active=True))
        # Check length
        if len(objectives) > 1:
            raise RuntimeError(
                "Multistart solver is unable to handle model with multiple active objectives."
            )
        elif len(objectives) == 1:
            obj = objectives[0]
            obj.sign = 1 if obj.sense == minimize else -1
            obj_sign = obj.sign

        else:
            obj = None
            obj_sign = 1
            config.break_on_solution = True
            logger.warning(
                "No objective found in the provided model. The solver will "
                "stop if it finds a feasible solution before completing "
                "the total number of iterations."
            )

        # store objective values and objective/result information for best
        # solution obtained
        objectives = []
        best_result = None
        best_objective = float('inf') * obj_sign
        results.feasible_solution_list = []
        results.feasible_iter_list = []

        # As we are about to run a solver, update the stale flag
        StaleFlagManager.mark_all_as_stale()

        # create temporary variable list for value transfer
        tmp_var_list_name = unique_component_name(model, "_vars_list")
        setattr(
            model,
            tmp_var_list_name,
            list(model.component_data_objects(Var, descend_into=True)),
        )
        # If the list has nothing in it, check components
        if len(model._vars_list) == 0:
            setattr(
                model,
                tmp_var_list_name,
                list(
                    get_vars_from_components(
                        model, ctype=(Constraint, Objective), active=True
                    )
                ),
            )

        num_iter = 0
        timer.start(f"timer_iter_{num_iter}")
        logger.info(f"Running initial solve. Iteration: {num_iter}\n")
        best_result = result = self.subsolver.solve(model, **subsolver_args)
        logger.info(
            f'solved NLP: {result.solution_status}, {result.termination_condition}'
        )

        # Check the solution status before loading variables into the model.
        if result.solution_status in {SolutionStatus.feasible, SolutionStatus.optimal}:
            results.feasible_solution_list.append(result)
            results.feasible_iter_list.append(num_iter)

        if result.solution_status is SolutionStatus.optimal:
            if obj is not None:
                obj_val = result.incumbent_objective
                best_objective = obj_val
                objectives.append(obj_val)
        timer.start(f"timer_iter_{num_iter}")

        max_iter = config.iterations

        if self._timing_lim:
            self._update_solver_timelimit(num_iter, timer, subsolver_args)
            if self.remaining_time_limit == 0:
                logger.warning(
                    f"Time limit reached after {num_iter} iterations. Exiting."
                )
                max_iter = 0

        # if HCS rule is specified, reinitialize completely randomly until
        # rule specifies stopping
        using_HCS = config.iterations == -1
        HCS_completed = False
        if using_HCS and num_iter != max_iter:
            assert (
                config.strategy == "rand"
            ), "High confidence stopping rule requires rand strategy."
            max_iter = config.HCS_max_iterations

        timer.start('iterative_solves')
        while num_iter < max_iter:
            num_iter += 1
            timer.start(f"timer_iter_{num_iter}")
            if using_HCS and should_stop(
                objectives,
                config.stopping_mass,
                config.stopping_delta,
                config.HCS_tolerance,
            ):
                HCS_completed = True
                timer.stop(f"timer_iter_{num_iter}")
                break

            logger.info(f"Iteration: {num_iter}\n")

            # at first iteration, solve the originally passed model
            m = model
            reinitialize_variables(m, config, sampler)
            result = self.subsolver.solve(m, **subsolver_args)
            logger.info(
                f'solved NLP: {result.solution_status}, {result.termination_condition}'
            )
            # Check the solution status before loading variables into the model.
            if result.solution_status in {
                SolutionStatus.feasible,
                SolutionStatus.optimal,
            }:
                results.feasible_solution_list.append(result)
                results.feasible_iter_list.append(num_iter)
                # If we are looking for the first feasible solution, then return immediately
                if config.break_on_solution:
                    best_result = result
                    timer.stop(f"timer_iter_{num_iter}")
                    break

            if result.solution_status is SolutionStatus.optimal:
                if obj is not None:
                    obj_val = result.incumbent_objective
                    objectives.append(obj_val)
                    if obj_val * obj_sign < obj_sign * best_objective:
                        # objective has improved
                        best_objective = obj_val
                        best_result = result
            timer.stop(f"timer_iter_{num_iter}")

            if self._timing_lim:
                self._update_solver_timelimit(num_iter, timer, subsolver_args)
                if self.remaining_time_limit == 0:
                    logger.warning(
                        f"Time limit reached after {num_iter} iterations. Exiting"
                    )
                    break

        timer.stop('iterative_solves')
        delattr(model, tmp_var_list_name)

        if using_HCS:
            if not HCS_completed:
                logger.warning(
                    "High confidence stopping rule was unable to complete "
                    "after %s iterations. To increase this limit, change the "
                    "HCS_max_iterations flag." % num_iter
                )

        if (
            config.raise_exception_on_nonoptimal_result
            and best_result.solution_status != SolutionStatus.optimal
        ):
            raise NoOptimalSolutionError()

        # Check termination condition for ipopt-specific outputs
        if str(
            best_result.solver_name.lower()
        ) == "ipopt" and best_result.termination_condition in {
            TerminationCondition.locallyInfeasible,
            TerminationCondition.unbounded,
            TerminationCondition.provenInfeasible,
        }:
            results.termination_condition = TerminationCondition.infeasibleOrUnbounded
        elif self.remaining_time_limit == 0:
            results.termination_condition = TerminationCondition.maxTimeLimit
        else:
            results.termination_condition = best_result.termination_condition

        results.solution_loader = best_result.solution_loader
        results.solution_status = best_result.solution_status
        results.incumbent_objective = best_result.incumbent_objective
        results.solver_log = best_result.solver_log

        if config.load_solutions:
            if results.solution_status == SolutionStatus.noSolution:
                raise NoSolutionError()

            results.solution_loader.load_solution()

        results.solver_config = config
        results.timing_info.timer = timer
        results.timing_info.wall_time = default_timer() - start_time
        return results

    def _update_solver_timelimit(self, iteration, timer, subsolver_args):
        # Get elapsed time from last timer
        last_timer = timer._get_timer(f"timer_iter_{iteration}")
        elapsed_time = last_timer.total_time

        # Take elapsed time off of time_limit for subsolver
        new_time_limit = self.remaining_time_limit - elapsed_time
        # Set new timelimits
        self.remaining_time_limit = max(new_time_limit, 0)
        subsolver_args["time_limit"] = max(
            1e-6, min(self.remaining_time_limit, subsolver_args["time_limit"])
        )

    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass


# Sampling class to organize and configure random samplers
class SamplingManager:
    def __init__(self, method, rng=None, seed=None):

        self.method = method
        self._check_method()

        self.seed = seed

        # Define or create a random number generator
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng(seed)

        self.qmc_sampler = None

    def _check_method(self):
        # Define accepted method names.
        aliases = {
            "random_uniform": "uniform",
            "uniform": "uniform",
            "latin_hypercube": "lhs",
            "lhs": "lhs",
            "sobol_sampling": "sobol",
            "sobol": "sobol",
        }
        key = self.method.lower()
        if key in aliases:
            self.method = aliases[key]
        else:
            raise ValueError(
                f"Unknown sampling method '{self.method}'."
                "Supported methods: random_uniform, latin_hypercube, "
                "or sobol_sampling."
            )

    def _ensure_qmc(self, dim):
        if self.qmc_sampler is not None:
            return
        if self.method == "lhs":
            self.qmc_sampler = stats.qmc.LatinHypercube(d=dim, rng=self.rng)
        elif self.method == "sobol":
            self.qmc_sampler = stats.qmc.Sobol(d=dim, scramble=True, seed=self.rng)

    def sample_scalar(self, lower, upper):
        if self.method == "uniform":
            return self.rng.uniform(lower, upper)

        if self.method in ("lhs", "sobol"):
            self._ensure_qmc(dim=1)
            x = self.qmc_sampler.random(n=1)  # shape (1, d)
            return stats.qmc.scale(x, lower, upper).item()

    def sample_vector(self, lower, upper):
        """Vector sample for uniform/lhs/sobol over all vars at once."""
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        if self.method == "uniform":
            return self.rng.uniform(lower, upper)

        if self.method in ("lhs", "sobol"):
            self._ensure_qmc(dim=len(lower))
            x = self.qmc_sampler.random(n=1)  # shape (1, d)
            return stats.qmc.scale(x, lower, upper)[0]
