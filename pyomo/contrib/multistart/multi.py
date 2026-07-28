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

from typing import Any
from pyomo.common.timing import HierarchicalTimer, default_timer
from pyomo.common.modeling import unique_component_name
from pyomo.common.dependencies import numpy as np
from pyomo.contrib.multistart.high_conf_stop import should_stop
from pyomo.contrib.multistart.reinit import reinitialize_variables, strategies
from pyomo.core import Objective, Constraint, Var, minimize, value
from pyomo.contrib.solver.common.base import SolverBase
from pyomo.contrib.solver.common.config import SolverConfig
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.contrib.solver.common.results import SolutionStatus
from pyomo.contrib.solver.common.util import (
    NoOptimalSolutionError,
    NoSolutionError,
    NoFeasibleSolutionError,
)
from pyomo.util.vars_from_expressions import get_vars_from_components

from pyomo.common.dependencies.scipy import stats
from pyomo.common.dependencies import numpy as np

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
        self.solver = self.declare(
            "solver",
            ConfigValue(
                default="ipopt",
                description="solver to use, defaults to ipopt"
                "Should also be able to accept solver objects. In progress",
            ),
        )
        self.solver_args = self.declare(
            "solver_args",
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
            The lower the parameter, the stricter the rule.
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
                default="latin_hypercube",
                description="Method for sampling random starting points for reinitialization step. "
                "Supported options are 'random_uniform', 'latin_hypercube', and 'sobol_sampling'. "
                "Only utilized when config.strategy is 'rand_vector'.",
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
        but it is not guaranteed the subsolvers will be.
        Check if the selected subsolver is available, which by default is ipopt.
        """

        subsolver = SolverFactory(self.config.solver)
        return subsolver.available()

    def version(self):
        """Get solver version."""
        """
            Original implementation: 0.1.0,
            Current implementation: 0.2.0,
        """
        return (0, 2, 0)

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

        # Set sub-solver options so infeasible solve does not interrupt runs
        config.solver_args["load_solutions"] = False
        config.solver_args["raise_exception_on_nonoptimal_result"] = False

        # config.raise_exception_on_nonoptimal_result = False

        solver = SolverFactory(config.solver)

        # Model sense
        objectives = list(model.component_data_objects(Objective, active=True))
        # Check length
        print(len(objectives))
        if len(objectives) > 1:
            raise RuntimeError(
                "Multistart solver is unable to handle model with multiple active objectives."
            )
        elif len(objectives) == 1:
            obj = objectives[0]
            obj.sign = 1 if obj.sense == minimize else -1
            obj_sign = obj.sign

        else:
            obj_sign = 1

        best_objective = float('inf') * obj_sign

        # store objective values and objective/result information for best
        # solution obtained
        objectives = []
        best_model = model
        best_result = None

        try:
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
            best_result = result = solver.solve(model, **config.solver_args)
            # Check the solution status before loading variables into the model.
            if result.solution_status in {
                SolutionStatus.feasible,
                SolutionStatus.optimal,
            }:
                result.solution_loader.load_vars()
                logger.info(
                    f'solved NLP: {result.solution_status}, {result.termination_condition}'
                )

            if best_result.solution_status is SolutionStatus.optimal:
                obj_val = value(obj.expr)
                best_objective = obj_val
                objectives.append(obj_val)

            num_iter = 0
            max_iter = config.iterations
            # if HCS rule is specified, reinitialize completely randomly until
            # rule specifies stopping
            using_HCS = config.iterations == -1
            HCS_completed = False
            if using_HCS:
                assert (
                    config.strategy == "rand"
                ), "High confidence stopping rule requires rand strategy."
                max_iter = config.HCS_max_iterations

            while num_iter < max_iter:
                if using_HCS and should_stop(
                    objectives,
                    config.stopping_mass,
                    config.stopping_delta,
                    config.HCS_tolerance,
                ):
                    HCS_completed = True
                    break
                logger.info(f"num_iter: {num_iter}\n")
                num_iter += 1
                # at first iteration, solve the originally passed model
                m = model.clone() if num_iter > 1 else model
                reinitialize_variables(m, config, sampler)
                result = solver.solve(m, **config.solver_args)

                # Check the solution status before loading variables into the model.
                if result.solution_status in {
                    SolutionStatus.feasible,
                    SolutionStatus.optimal,
                }:
                    result.solution_loader.load_vars()
                    logger.info(
                        f'solved NLP: {result.solution_status}, {result.termination_condition}'
                    )
                    # If we are looking for the first feasible solution, then return immediately
                    if config.break_on_solution:
                        return best_result

                if best_result.solution_status is SolutionStatus.optimal:
                    model_objectives = m.component_data_objects(Objective, active=True)
                    print("Model objs", model_objectives)
                    mobj = next(model_objectives)
                    obj_val = value(mobj.expr)
                    objectives.append(obj_val)
                    if obj_val * obj_sign < obj_sign * best_objective:
                        # objective has improved
                        best_objective = obj_val
                        best_model = m
                        best_result = result

            if using_HCS:
                if not HCS_completed:
                    logger.warning(
                        "High confidence stopping rule was unable to complete "
                        "after %s iterations. To increase this limit, change the "
                        "HCS_max_iterations flag." % num_iter
                    )

            else:
                if config.raise_exception_on_nonoptimal_result and not using_HCS:
                    if best_result.solution_status != SolutionStatus.optimal:
                        raise NoOptimalSolutionError

            # if no better result was found than initial solve, then return
            # that without needing to copy variables.
            if best_model is model:
                return best_result

            # reassign the given models vars to the new models vars
            orig_var_list = getattr(model, tmp_var_list_name)
            best_soln_var_list = getattr(best_model, tmp_var_list_name)
            for orig_var, new_var in zip(orig_var_list, best_soln_var_list):
                if not orig_var.is_fixed():
                    orig_var.set_value(new_var.value, skip_validation=True)

            return best_result
        finally:
            # Remove temporary variable list
            delattr(model, tmp_var_list_name)

    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass


# Sampling class to organize and configure random samplers
class SamplingManager:
    def __init__(self, method="lhs", rng=None, seed=None):
        aliases = {
            "random_uniform": "uniform",
            "uniform": "uniform",
            "latin_hypercube": "lhs",
            "lhs": "lhs",
            "sobol_sampling": "sobol",
            "sobol": "sobol",
        }
        self.method = aliases[method.lower()]

        self.seed = seed

        # Define or create a random number generator
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng(seed)

        self.qmc_sampler = None

    def _ensure_qmc(self, dim):
        if self.qmc_sampler is not None:
            return

        if self.method == "lhs":
            self.qmc_sampler = stats.qmc.LatinHypercube(d=dim, seed=self.seed)
        elif self.method == "sobol":
            self.qmc_sampler = stats.qmc.Sobol(d=dim, scramble=True, seed=self.seed)
        else:
            raise ValueError(f"QMC sampler not valid for method '{self.method}'")

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

        raise ValueError(f"Unknown sampling method '{self.method}'")
