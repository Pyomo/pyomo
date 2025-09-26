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

# pyros.py: Generalized Robust Cutting-Set Algorithm for Pyomo
from datetime import datetime, timezone
import logging

from pyomo.common.config import document_kwargs_from_configdict
from pyomo.core.expr import value
from pyomo.opt import SolverFactory

from pyomo.contrib.pyros.config import pyros_config, logger_domain
from pyomo.contrib.pyros.pyros_algorithm_methods import ROSolver_iterative_solve
from pyomo.contrib.pyros.solve_data import ROSolveResults
from pyomo.contrib.pyros.util import (
    load_final_solution,
    pyrosTerminationCondition,
    validate_pyros_inputs,
    log_model_statistics,
    IterationLogRecord,
    setup_pyros_logger,
    time_code,
    TimingData,
    ModelData,
)


__version__ = "1.3.9"


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
            subprocess.check_output(
                commit_hash_command_args,
                # suppress git error if Pyomo installation
                # is not a git repo
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
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
                "Invoked at UTC "
                f"{datetime.now(timezone.utc).isoformat()}"
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

    def _resolve_and_validate_pyros_args(self, model, **kwds):
        """
        Resolve and validate arguments to ``self.solve()``.

        Parameters
        ----------
        model : ConcreteModel
            Deterministic model object passed to ``self.solve()``.
        **kwds : dict
            All other arguments to ``self.solve()``.

        Returns
        -------
        config : ConfigDict
            Standardized arguments.
        user_var_partitioning : util.VarPartitioning
            User-based partitioning of the in-scope model variables.

        Note
        ----
        This method can be broken down into three steps:

        1. Cast arguments to ConfigDict. Argument-wise
           validation is performed automatically.
           Note that arguments specified directly take
           precedence over arguments specified indirectly
           through direct argument 'options'.
        2. Inter-argument validation.
        """
        config = self.CONFIG(kwds.pop("options", {}))
        config = config(kwds)
        user_var_partitioning = validate_pyros_inputs(model, config)

        return config, user_var_partitioning

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
        first_stage_variables: VarData, Var, or iterable of VarData/Var
            First-stage model variables (or design variables).
        second_stage_variables: VarData, Var, or iterable of VarData/Var
            Second-stage model variables (or control variables).
        uncertain_params: (iterable of) Param, Var, ParamData, or VarData
            Uncertain model parameters.
            Of every constituent `Param` object,
            the `mutable` attribute must be set to True.
            All constituent `Var`/`VarData` objects should be
            fixed.
        uncertainty_set: UncertaintySet
            Uncertainty set against which the solution(s) returned
            will be confirmed to be robust.
        local_solver: str or solver type
            Subordinate local NLP solver.
            If a `str` is passed, then the `str` is cast to
            ``SolverFactory(local_solver)``.
        global_solver: str or solver type
            Subordinate global NLP solver.
            If a `str` is passed, then the `str` is cast to
            ``SolverFactory(global_solver)``.

        Returns
        -------
        return_soln : ROSolveResults
            Summary of PyROS termination outcome.

        """
        model_data = ModelData(original_model=model, timing=TimingData(), config=None)
        with time_code(
            timing_data_obj=model_data.timing,
            code_block_name="main",
            is_main_timer=True,
        ):
            kwds.update(
                dict(
                    first_stage_variables=first_stage_variables,
                    second_stage_variables=second_stage_variables,
                    uncertain_params=uncertain_params,
                    uncertainty_set=uncertainty_set,
                    local_solver=local_solver,
                    global_solver=global_solver,
                )
            )

            # we want to log the intro and disclaimer in
            # advance of assembling the config.
            # this helps clarify to the user that any
            # messages logged during assembly of the config
            # were, in fact, logged after PyROS was initiated
            progress_logger = logger_domain(
                kwds.get(
                    "progress_logger",
                    kwds.get("options", dict()).get(
                        "progress_logger", default_pyros_solver_logger
                    ),
                )
            )
            self._log_intro(logger=progress_logger, level=logging.INFO)
            self._log_disclaimer(logger=progress_logger, level=logging.INFO)

            config, user_var_partitioning = self._resolve_and_validate_pyros_args(
                model, **kwds
            )
            self._log_config(
                logger=config.progress_logger,
                config=config,
                exclude_options=None,
                level=logging.INFO,
            )
            model_data.config = config

            config.progress_logger.info("Preprocessing...")
            model_data.timing.start_timer("main.preprocessing")
            robust_infeasible = model_data.preprocess(user_var_partitioning)
            model_data.timing.stop_timer("main.preprocessing")
            preprocessing_time = model_data.timing.get_total_time("main.preprocessing")
            config.progress_logger.info(
                f"Done preprocessing; required wall time of "
                f"{preprocessing_time:.3f}s."
            )

            log_model_statistics(model_data)

            # === Solve and load solution into model
            return_soln = ROSolveResults()
            if not robust_infeasible:
                pyros_soln = ROSolver_iterative_solve(model_data)
                IterationLogRecord.log_header_rule(config.progress_logger.info)

                termination_acceptable = pyros_soln.pyros_termination_condition in {
                    pyrosTerminationCondition.robust_optimal,
                    pyrosTerminationCondition.robust_feasible,
                }
                if termination_acceptable:
                    load_final_solution(
                        model_data=model_data,
                        master_soln=pyros_soln.master_results,
                        original_user_var_partitioning=user_var_partitioning,
                    )

                # get the most recent master objective, if available
                return_soln.final_objective_value = None
                master_epigraph_obj_value = value(
                    pyros_soln.master_results.master_model.epigraph_obj, exception=False
                )
                if master_epigraph_obj_value is not None:
                    # account for sense of the original model objective
                    # when reporting the final PyROS (master) objective,
                    # since maximization objective is changed to
                    # minimization objective during preprocessing
                    return_soln.final_objective_value = (
                        model_data.active_obj_original_sense * master_epigraph_obj_value
                    )

                return_soln.pyros_termination_condition = (
                    pyros_soln.pyros_termination_condition
                )
                return_soln.iterations = pyros_soln.iterations
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
