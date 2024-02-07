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

# pyros.py: Generalized Robust Cutting-Set Algorithm for Pyomo
import logging
from pyomo.common.config import document_kwargs_from_configdict
from pyomo.common.collections import Bunch, ComponentSet
from pyomo.core.base.block import Block
from pyomo.core.expr import value
from pyomo.core.base.var import Var
from pyomo.core.base.objective import Objective
from pyomo.contrib.pyros.util import time_code
from pyomo.common.modeling import unique_component_name
from pyomo.opt import SolverFactory
from pyomo.contrib.pyros.config import pyros_config
from pyomo.contrib.pyros.util import (
    model_is_valid,
    recast_to_min_obj,
    add_decision_rule_constraints,
    add_decision_rule_variables,
    load_final_solution,
    pyrosTerminationCondition,
    ObjectiveType,
    validate_uncertainty_set,
    identify_objective_functions,
    validate_kwarg_inputs,
    transform_to_standard_form,
    turn_bounds_to_constraints,
    replace_uncertain_bounds_with_constraints,
    IterationLogRecord,
    setup_pyros_logger,
    TimingData,
)
from pyomo.contrib.pyros.solve_data import ROSolveResults
from pyomo.contrib.pyros.pyros_algorithm_methods import ROSolver_iterative_solve
from pyomo.core.base import Constraint

from datetime import datetime


__version__ = "1.2.9"


default_pyros_solver_logger = setup_pyros_logger()


def _get_pyomo_version_info():
    """
    Get Pyomo version information.
    """
    import os
    import subprocess
    from pyomo.version import version

    pyomo_version = version
    commit_hash = "unknown"

    pyros_dir = os.path.join(*os.path.split(__file__)[:-1])
    commit_hash_command_args = [
        "git",
        "-C",
        f"{pyros_dir}",
        "rev-parse",
        "--short",
        "HEAD",
    ]
    try:
        commit_hash = (
            subprocess.check_output(commit_hash_command_args).decode("ascii").strip()
        )
    except subprocess.CalledProcessError:
        commit_hash = "unknown"

    return {"Pyomo version": pyomo_version, "Commit hash": commit_hash}


@SolverFactory.register(
    "pyros",
    doc="Robust optimization (RO) solver implementing "
    "the generalized robust cutting-set algorithm (GRCS)",
)
class PyROS(object):
    '''
    PyROS (Pyomo Robust Optimization Solver) implementing a
    generalized robust cutting-set algorithm (GRCS)
    to solve two-stage NLP optimization models under uncertainty.
    '''

    CONFIG = pyros_config()
    _LOG_LINE_LENGTH = 78

    def available(self, exception_flag=True):
        """Check if solver is available."""
        return True

    def version(self):
        """Return a 3-tuple describing the solver version."""
        return __version__

    def license_is_valid(self):
        '''License for using PyROS'''
        return True

    # The Pyomo solver API expects that solvers support the context
    # manager API
    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        pass

    def _log_intro(self, logger, **log_kwargs):
        """
        Log PyROS solver introductory messages.

        Parameters
        ----------
        logger : logging.Logger
            Logger through which to emit messages.
        **log_kwargs : dict, optional
            Keyword arguments to ``logger.log()`` callable.
            Should not include `msg`.
        """
        logger.log(msg="=" * self._LOG_LINE_LENGTH, **log_kwargs)
        logger.log(
            msg=f"PyROS: The Pyomo Robust Optimization Solver, v{self.version()}.",
            **log_kwargs,
        )

        # git_info_str = ", ".join(
        #     f"{field}: {val}" for field, val in _get_pyomo_git_info().items()
        # )
        version_info = _get_pyomo_version_info()
        version_info_str = ' ' * len("PyROS: ") + ("\n" + ' ' * len("PyROS: ")).join(
            f"{key}: {val}" for key, val in version_info.items()
        )
        logger.log(msg=version_info_str, **log_kwargs)
        logger.log(
            msg=(
                f"{' ' * len('PyROS:')} "
                f"Invoked at UTC {datetime.utcnow().isoformat()}"
            ),
            **log_kwargs,
        )
        logger.log(msg="", **log_kwargs)
        logger.log(
            msg=("Developed by: Natalie M. Isenberg (1), Jason A. F. Sherman (1),"),
            **log_kwargs,
        )
        logger.log(
            msg=(
                f"{' ' * len('Developed by:')} "
                "John D. Siirola (2), Chrysanthos E. Gounaris (1)"
            ),
            **log_kwargs,
        )
        logger.log(
            msg=(
                "(1) Carnegie Mellon University, " "Department of Chemical Engineering"
            ),
            **log_kwargs,
        )
        logger.log(
            msg="(2) Sandia National Laboratories, Center for Computing Research",
            **log_kwargs,
        )
        logger.log(msg="", **log_kwargs)
        logger.log(
            msg=(
                "The developers gratefully acknowledge support "
                "from the U.S. Department"
            ),
            **log_kwargs,
        )
        logger.log(
            msg=(
                "of Energy's "
                "Institute for the Design of Advanced Energy Systems (IDAES)."
            ),
            **log_kwargs,
        )
        logger.log(msg="=" * self._LOG_LINE_LENGTH, **log_kwargs)

    def _log_disclaimer(self, logger, **log_kwargs):
        """
        Log PyROS solver disclaimer messages.

        Parameters
        ----------
        logger : logging.Logger
            Logger through which to emit messages.
        **log_kwargs : dict, optional
            Keyword arguments to ``logger.log()`` callable.
            Should not include `msg`.
        """
        disclaimer_header = " DISCLAIMER ".center(self._LOG_LINE_LENGTH, "=")

        logger.log(msg=disclaimer_header, **log_kwargs)
        logger.log(msg="PyROS is still under development. ", **log_kwargs)
        logger.log(
            msg=(
                "Please provide feedback and/or report any issues by creating "
                "a ticket at"
            ),
            **log_kwargs,
        )
        logger.log(msg="https://github.com/Pyomo/pyomo/issues/new/choose", **log_kwargs)
        logger.log(msg="=" * self._LOG_LINE_LENGTH, **log_kwargs)

    def _log_config(self, logger, config, exclude_options=None, **log_kwargs):
        """
        Log PyROS solver options.

        Parameters
        ----------
        logger : logging.Logger
            Logger for the solver options.
        config : ConfigDict
            PyROS solver options.
        exclude_options : None or iterable of str, optional
            Options (keys of the ConfigDict) to exclude from
            logging. If `None` passed, then the names of the
            required arguments to ``self.solve()`` are skipped.
        **log_kwargs : dict, optional
            Keyword arguments to each statement of ``logger.log()``.
        """
        # log solver options
        if exclude_options is None:
            exclude_options = [
                "first_stage_variables",
                "second_stage_variables",
                "uncertain_params",
                "uncertainty_set",
                "local_solver",
                "global_solver",
            ]

        logger.log(msg="Solver options:", **log_kwargs)
        for key, val in config.items():
            if key not in exclude_options:
                logger.log(msg=f" {key}={val!r}", **log_kwargs)
        logger.log(msg="-" * self._LOG_LINE_LENGTH, **log_kwargs)

    @document_kwargs_from_configdict(
        config=CONFIG,
        section="Keyword Arguments",
        indent_spacing=4,
        width=72,
        visibility=0,
    )
    def solve(
        self,
        model,
        first_stage_variables,
        second_stage_variables,
        uncertain_params,
        uncertainty_set,
        local_solver,
        global_solver,
        **kwds,
    ):
        """Solve a model.

        Parameters
        ----------
        model: ConcreteModel
            The deterministic model.
        first_stage_variables: list of Var
            First-stage model variables (or design variables).
        second_stage_variables: list of Var
            Second-stage model variables (or control variables).
        uncertain_params: list of Param
            Uncertain model parameters.
            The `mutable` attribute for every uncertain parameter
            objects must be set to True.
        uncertainty_set: UncertaintySet
            Uncertainty set against which the solution(s) returned
            will be confirmed to be robust.
        local_solver: Solver
            Subordinate local NLP solver.
        global_solver: Solver
            Subordinate global NLP solver.

        Returns
        -------
        return_soln : ROSolveResults
            Summary of PyROS termination outcome.

        """

        # === Add the explicit arguments to the config
        config = self.CONFIG(kwds.pop('options', {}))
        config.first_stage_variables = first_stage_variables
        config.second_stage_variables = second_stage_variables
        config.uncertain_params = uncertain_params
        config.uncertainty_set = uncertainty_set
        config.local_solver = local_solver
        config.global_solver = global_solver

        dev_options = kwds.pop('dev_options', {})
        config.set_value(kwds)
        config.set_value(dev_options)

        model = model

        # === Validate kwarg inputs
        validate_kwarg_inputs(model, config)

        # === Validate ability of grcs RO solver to handle this model
        if not model_is_valid(model):
            raise AttributeError(
                "This model structure is not currently handled by the ROSolver."
            )

        # === Define nominal point if not specified
        if len(config.nominal_uncertain_param_vals) == 0:
            config.nominal_uncertain_param_vals = list(
                p.value for p in config.uncertain_params
            )
        elif len(config.nominal_uncertain_param_vals) != len(config.uncertain_params):
            raise AttributeError(
                "The nominal_uncertain_param_vals list must be the same length"
                "as the uncertain_params list"
            )

        # === Create data containers
        model_data = ROSolveResults()
        model_data.timing = Bunch()

        # === Start timer, run the algorithm
        model_data.timing = TimingData()
        with time_code(
            timing_data_obj=model_data.timing,
            code_block_name="main",
            is_main_timer=True,
        ):
            # output intro and disclaimer
            self._log_intro(logger=config.progress_logger, level=logging.INFO)
            self._log_disclaimer(logger=config.progress_logger, level=logging.INFO)
            self._log_config(
                logger=config.progress_logger,
                config=config,
                exclude_options=None,
                level=logging.INFO,
            )

            # begin preprocessing
            config.progress_logger.info("Preprocessing...")
            model_data.timing.start_timer("main.preprocessing")

            # === A block to hold list-type data to make cloning easy
            util = Block(concrete=True)
            util.first_stage_variables = config.first_stage_variables
            util.second_stage_variables = config.second_stage_variables
            util.uncertain_params = config.uncertain_params

            model_data.util_block = unique_component_name(model, 'util')
            model.add_component(model_data.util_block, util)
            # Note:  model.component(model_data.util_block) is util

            # === Validate uncertainty set happens here, requires util block for Cardinality and FactorModel sets
            validate_uncertainty_set(config=config)

            # === Leads to a logger warning here for inactive obj when cloning
            model_data.original_model = model
            # === For keeping track of variables after cloning
            cname = unique_component_name(model_data.original_model, 'tmp_var_list')
            src_vars = list(model_data.original_model.component_data_objects(Var))
            setattr(model_data.original_model, cname, src_vars)
            model_data.working_model = model_data.original_model.clone()

            # identify active objective function
            # (there should only be one at this point)
            # recast to minimization if necessary
            active_objs = list(
                model_data.working_model.component_data_objects(
                    Objective, active=True, descend_into=True
                )
            )
            assert len(active_objs) == 1
            active_obj = active_objs[0]
            active_obj_original_sense = active_obj.sense
            recast_to_min_obj(model_data.working_model, active_obj)

            # === Determine first and second-stage objectives
            identify_objective_functions(model_data.working_model, active_obj)
            active_obj.deactivate()

            # === Put model in standard form
            transform_to_standard_form(model_data.working_model)

            # === Replace variable bounds depending on uncertain params with
            #     explicit inequality constraints
            replace_uncertain_bounds_with_constraints(
                model_data.working_model, model_data.working_model.util.uncertain_params
            )

            # === Add decision rule information
            add_decision_rule_variables(model_data, config)
            add_decision_rule_constraints(model_data, config)

            # === Move bounds on control variables to explicit ineq constraints
            wm_util = model_data.working_model

            # === Every non-fixed variable that is neither first-stage
            #     nor second-stage is taken to be a state variable
            fsv = ComponentSet(model_data.working_model.util.first_stage_variables)
            ssv = ComponentSet(model_data.working_model.util.second_stage_variables)
            sv = ComponentSet()
            model_data.working_model.util.state_vars = []
            for v in model_data.working_model.component_data_objects(Var):
                if not v.fixed and v not in fsv | ssv | sv:
                    model_data.working_model.util.state_vars.append(v)
                    sv.add(v)

            # Bounds on second stage variables and state variables are separation objectives,
            #  they are brought in this was as explicit constraints
            for c in model_data.working_model.util.second_stage_variables:
                turn_bounds_to_constraints(c, wm_util, config)

            for c in model_data.working_model.util.state_vars:
                turn_bounds_to_constraints(c, wm_util, config)

            # === Make control_variable_bounds array
            wm_util.ssv_bounds = []
            for c in model_data.working_model.component_data_objects(
                Constraint, descend_into=True
            ):
                if "bound_con" in c.name:
                    wm_util.ssv_bounds.append(c)

            model_data.timing.stop_timer("main.preprocessing")
            preprocessing_time = model_data.timing.get_total_time("main.preprocessing")
            config.progress_logger.info(
                f"Done preprocessing; required wall time of "
                f"{preprocessing_time:.3f}s."
            )

            # === Solve and load solution into model
            pyros_soln, final_iter_separation_solns = ROSolver_iterative_solve(
                model_data, config
            )
            IterationLogRecord.log_header_rule(config.progress_logger.info)

            return_soln = ROSolveResults()
            if pyros_soln is not None and final_iter_separation_solns is not None:
                if config.load_solution and (
                    pyros_soln.pyros_termination_condition
                    is pyrosTerminationCondition.robust_optimal
                    or pyros_soln.pyros_termination_condition
                    is pyrosTerminationCondition.robust_feasible
                ):
                    load_final_solution(model_data, pyros_soln.master_soln, config)

                # account for sense of the original model objective
                # when reporting the final PyROS (master) objective,
                # since maximization objective is changed to
                # minimization objective during preprocessing
                if config.objective_focus == ObjectiveType.nominal:
                    return_soln.final_objective_value = (
                        active_obj_original_sense
                        * value(pyros_soln.master_soln.master_model.obj)
                    )
                elif config.objective_focus == ObjectiveType.worst_case:
                    return_soln.final_objective_value = (
                        active_obj_original_sense
                        * value(pyros_soln.master_soln.master_model.zeta)
                    )
                return_soln.pyros_termination_condition = (
                    pyros_soln.pyros_termination_condition
                )
                return_soln.iterations = pyros_soln.total_iters + 1

                # === Remove util block
                model.del_component(model_data.util_block)

                del pyros_soln.util_block
                del pyros_soln.working_model
            else:
                return_soln.final_objective_value = None
                return_soln.pyros_termination_condition = (
                    pyrosTerminationCondition.robust_infeasible
                )
                return_soln.iterations = 0

        return_soln.config = config
        return_soln.time = model_data.timing.get_total_time("main")

        # log termination-related messages
        config.progress_logger.info(return_soln.pyros_termination_condition.message)
        config.progress_logger.info("-" * self._LOG_LINE_LENGTH)
        config.progress_logger.info(f"Timing breakdown:\n\n{model_data.timing}")
        config.progress_logger.info("-" * self._LOG_LINE_LENGTH)
        config.progress_logger.info(return_soln)
        config.progress_logger.info("-" * self._LOG_LINE_LENGTH)
        config.progress_logger.info("All done. Exiting PyROS.")
        config.progress_logger.info("=" * self._LOG_LINE_LENGTH)

        return return_soln
