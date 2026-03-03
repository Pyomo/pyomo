# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
#
# Pyomo.DoE was produced under the Department of Energy Carbon Capture Simulation
# Initiative (CCSI), and is copyright (c) 2022 by the software owners:
# TRIAD National Security, LLC., Lawrence Livermore National Security, LLC.,
# Lawrence Berkeley National Laboratory, Pacific Northwest National Laboratory,
# Battelle Memorial Institute, University of Notre Dame,
# The University of Pittsburgh, The University of Texas at Austin,
# University of Toledo, West Virginia University, et al. All rights reserved.
#
# NOTICE. This Software was developed under funding from the
# U.S. Department of Energy and the U.S. Government consequently retains
# certain rights. As such, the U.S. Government has been granted for itself
# and others acting on its behalf a paid-up, nonexclusive, irrevocable,
# worldwide license in the Software to reproduce, distribute copies to the
# public, prepare derivative works, and perform publicly and display
# publicly, and to permit other to do so.
# ____________________________________________________________________________________

from enum import Enum
from itertools import (
    permutations,
    product,
    combinations as _combinations,
    islice as _islice,
)
import concurrent.futures as _cf
import json
import logging
import math
import os
import threading
import time
import warnings

from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    pandas as pd,
    pathlib,
    matplotlib as plt,
    scipy_available,
)

from pyomo.common.errors import DeveloperError
from pyomo.common.timing import TicTocTimer

from pyomo.contrib.sensitivity_toolbox.sens import get_dsdp

if numpy_available and scipy_available:
    from pyomo.contrib.doe.grey_box_utilities import FIMExternalGreyBox

    from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
    from scipy.stats.qmc import LatinHypercube

import pyomo.environ as pyo
from pyomo.contrib.doe.utils import (
    check_FIM,
    compute_FIM_metrics,
    _SMALL_TOLERANCE_DEFINITENESS,
)
from pyomo.contrib.parmest.utils.model_utils import update_model_from_suffix

from pyomo.opt import SolverStatus


class ObjectiveLib(Enum):
    determinant = "determinant"  # det(FIM), D-optimality
    trace = "trace"  # trace(inv(FIM)), A-optimality
    pseudo_trace = "pseudo_trace"  # trace(FIM), pseudo-A-optimality
    minimum_eigenvalue = "minimum_eigenvalue"  # min(eig(FIM)), E-optimality
    condition_number = "condition_number"  # cond(FIM), ME-optimality
    zero = "zero"  # Constant zero objective, useful for initialization and debugging


class FiniteDifferenceStep(Enum):
    forward = "forward"
    central = "central"
    backward = "backward"


class InitializationMethod(Enum):
    lhs = "lhs"


class _DoEResultsJSONEncoder(json.JSONEncoder):
    """JSON encoder for DoE result payloads with numpy/Pyomo objects."""

    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Enum):
            return str(obj)
        return super().default(obj)


class DesignOfExperiments:
    # Objective options whose scalar score is compared with "larger is better"
    # in initialization and diagnostics paths.
    _MAXIMIZE_OBJECTIVES = frozenset(
        {
            ObjectiveLib.determinant,
            ObjectiveLib.pseudo_trace,
            ObjectiveLib.minimum_eigenvalue,
        }
    )

    def __init__(
        self,
        experiment=None,
        experiment_list=None,
        fd_formula="central",
        step=1e-3,
        objective_option="determinant",
        use_grey_box_objective=False,
        scale_constant_value=1.0,
        scale_nominal_param_value=False,
        prior_FIM=None,
        jac_initial=None,
        fim_initial=None,
        L_diagonal_lower_bound=1e-7,
        solver=None,
        grey_box_solver=None,
        tee=False,
        grey_box_tee=False,
        get_labeled_model_args=None,
        logger_level=logging.WARNING,
        improve_cholesky_roundoff_error=False,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
    ):
        """This package enables model-based design of experiments analysis
        with Pyomo.  Both direct optimization and enumeration modes are
        supported.

        The package has been refactored from its original form as of August 24. See
        the documentation for more information.

        Parameters
        ----------
        experiment_list:
            Experiment object(s) that hold the model and labels all the components.
            Can be a single Experiment object or a list of Experiment objects.
            For single experiments, you can pass the object directly: experiment_list=experiment
            or as a list: experiment_list=[experiment].
            Each object should have a ``get_labeled_model`` method that returns a model with
            the following labeled Pyomo Suffixes:

              - ``unknown_parameters``,
              - ``experimental_inputs``,
              - ``experimental_outputs``
              - ``measurement_error``.

        experiment:
            **DEPRECATED** - Use 'experiment_list' instead. This parameter will be removed
            in a future version. When provided, a DeprecationWarning is issued.

        fd_formula:
            Finite difference formula for computing the sensitivity matrix. Must be
            one of [``central``, ``forward``, ``backward``], default: ``central``
        step:
            Relative step size for the finite difference formula.
            default: 1e-3
        objective_option:
            String representation of the objective option. Current available options
            are:
            - ``determinant`` (for determinant, or D-optimality),
            - ``trace`` (for trace of covariance matrix, or A-optimality),
            - ``pseudo_trace`` (for trace of Fisher Information Matrix(FIM), or pseudo A-optimality),
            - ``minimum_eigenvalue``, (for E-optimality), or
            - ``condition_number`` (for ME-optimality)
            Note: E-optimality and ME-optimality are only supported when using the
            grey box objective (i.e., ``grey_box_solver`` is True)
            default: ``determinant``
        use_grey_box_objective:
            Boolean of whether or not to use the grey-box version of the objective
            function. True to use grey box, False to use standard.
            Default: False (do not use grey box)
        scale_constant_value:
            Constant scaling for the sensitivity matrix. Every element will be
            multiplied by this scaling factor.
            default: 1
        scale_nominal_param_value:
            Boolean for whether or not to scale the sensitivity matrix by the
            nominal parameter values. Every column of the sensitivity matrix
            will be divided by the respective nominal parameter value.
            default: False
        prior_FIM:
            2D numpy array representing information from prior experiments. If
            no value is given, the assumed prior will be a matrix of zeros. This
            matrix will be assumed to be scaled as the user has specified (i.e.,
            if scale_nominal_param_value is true, we will assume the FIM provided
            here has been scaled by the parameter values)
        jac_initial:
            2D numpy array as the initial values for the sensitivity matrix.
        fim_initial:
            2D numpy array as the initial values for the FIM.
        L_diagonal_lower_bound:
            Lower bound for the values of the lower triangular Cholesky factorization
            matrix.
            default: 1e-7
        solver:
            A ``solver`` object specified by the user, default=None.
            If not specified, default solver is set to IPOPT with MA57.
        tee:
            Solver option to be passed for verbose output.
        get_labeled_model_args:
            Additional arguments for the ``get_labeled_model`` function on the
            Experiment object.
        improve_cholesky_roundoff_error:
            Boolean value of whether or not to improve round-off error. If True, it will
            apply M[i,i] >= L[i,j]^2. Where, M is the FIM and L is the lower triangular matrix
            from Cholesky factorization. If the round-off error is not significant, this
            option can be turned off to improve performance by skipping this constraint.
        _Cholesky_option:
            Boolean value of whether or not to use the cholesky factorization to
            compute the determinant for the D-optimality criteria. This parameter
            should not be changed unless the user intends to make performance worse
            (i.e., compare an existing tool that uses the full FIM to this algorithm)
        _only_compute_fim_lower:
            If True, only the lower triangle of the FIM is computed. This parameter
            should not be changed unless the user intends to make performance worse
            (i.e., compare an existing tool that uses the full FIM to this algorithm)
        logger_level:
            Specify the level of the logger. Change to logging.DEBUG for all messages.

        """
        # Handle backward compatibility - experiment -> experiment_list
        if experiment is not None:
            warnings.warn(
                "The 'experiment' parameter in DesignOfExperiments is deprecated and "
                "will be removed in a future version. "
                "Please use 'experiment_list' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if experiment_list is None:
                experiment_list = experiment

        # Validate experiment_list is provided
        if experiment_list is None:
            raise ValueError(
                "The 'experiment_list' parameter is required. "
                "Pass a single Experiment object or a list of Experiment objects."
            )

        # Auto-convert single experiment to list
        if not isinstance(experiment_list, list):
            experiment_list = [experiment_list]

        # Validate list is not empty
        if len(experiment_list) == 0:
            raise ValueError("The 'experiment_list' cannot be empty.")

        # Check each experiment has get_labeled_model method
        for idx, exp in enumerate(experiment_list):
            if not hasattr(exp, "get_labeled_model"):
                raise ValueError(
                    f"Experiment at index {idx} in 'experiment_list' must have a "
                    f"'get_labeled_model' method"
                )

        # Store experiment_list
        self.experiment_list = experiment_list

        # Set the finite difference and subsequent step size
        self.fd_formula = FiniteDifferenceStep(fd_formula)
        self.step = step

        # Set the objective type and scaling options:
        self.objective_option = ObjectiveLib(objective_option)
        self.use_grey_box = use_grey_box_objective

        self.scale_constant_value = scale_constant_value
        self.scale_nominal_param_value = scale_nominal_param_value

        # Set the prior FIM (will be checked upon model construction)
        self.prior_FIM = prior_FIM

        # Set the initial values for the jacobian, fim, and L matrices
        self.jac_initial = jac_initial
        self.fim_initial = fim_initial

        # Set the lower bound on the Cholesky lower triangular matrix
        self.L_diagonal_lower_bound = L_diagonal_lower_bound

        # check if user-defined solver is given
        if solver:
            self.solver = solver
        # if not given, use default solver
        else:
            solver = pyo.SolverFactory("ipopt")
            solver.options["linear_solver"] = "ma57"
            solver.options["halt_on_ampl_error"] = "yes"
            solver.options["max_iter"] = 3000
            self.solver = solver

        self.tee = tee
        self.grey_box_tee = grey_box_tee

        if grey_box_solver:
            self.grey_box_solver = grey_box_solver
        else:
            grey_box_solver = pyo.SolverFactory("cyipopt")
            grey_box_solver.config.options["linear_solver"] = "ma57"
            grey_box_solver.config.options['tol'] = 1e-4
            grey_box_solver.config.options['mu_strategy'] = "monotone"

            self.grey_box_solver = grey_box_solver

        # Set get_labeled_model_args as an empty dict if no arguments are passed
        if get_labeled_model_args is None:
            get_labeled_model_args = {}
        self.get_labeled_model_args = get_labeled_model_args

        # Revtrieve logger and set logging level
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logger_level)

        # Set the private options if passed (only developers should pass these)
        self.Cholesky_option = _Cholesky_option
        self.only_compute_fim_lower = _only_compute_fim_lower

        # To improve round-off error in Cholesky-based objectives
        self.improve_cholesky_roundoff_error = improve_cholesky_roundoff_error

        # model attribute to avoid rebuilding models
        self.model = pyo.ConcreteModel()  # Build empty model

        # Empty results object
        self.results = {}

        # May need this attribute for more complicated structures?
        # (i.e., no model rebuilding for large models with sequential)
        self._built_scenarios = False

    # Perform doe
    def run_doe(self, model=None, results_file=None):
        """
        Runs DoE for a single experiment estimation. Can save results in
        a file based on the flag.

        Parameters
        ----------
        model: model to run the DoE, default: None (self.model)
        results_file: string name of the file path to save the results
                      to in the form of a .json file
                      default: None --> don't save

        """
        # Check results file name
        if results_file is not None:
            if not isinstance(results_file, (pathlib.Path, str)):
                raise ValueError(
                    "``results_file`` must be either a Path object or a string."
                )
        # Start timer
        sp_timer = TicTocTimer()
        sp_timer.tic(msg=None)
        self.logger.info("Beginning experimental optimization.")

        # Model is none, set it to self.model
        if model is None:
            model = self.model
        else:
            # TODO: Add safe naming when a model is passed by the user.
            # doe_block = pyo.Block()
            # doe_block_name = unique_component_name(model,
            #                                        "design_of_experiments_block")
            # model.add_component(doe_block_name, doe_block)
            pass

        # TODO: potentially work with this for more complicated models
        # Create the full DoE model (build scenarios for F.D. scheme)
        if not self._built_scenarios:
            self.create_doe_model(model=model)

        # Add the objective function to the model
        if self.use_grey_box:
            self.create_grey_box_objective_function(model=model)
        else:
            self.create_objective_function(model=model)

        # Track time required to build the DoE model
        build_time = sp_timer.toc(msg=None)
        self.logger.info(
            "Successfully built the DoE model.\nBuild time: %0.1f seconds" % build_time
        )

        # Solve the square problem first to
        # initialize the fim and
        # sensitivity constraints. First, we
        # Deactivate objective expression and
        # objective constraints (on a block),
        # and fix the design variables.
        model.objective.deactivate()
        model.obj_cons.deactivate()
        for comp in model.fd_scenario_blocks[0].experiment_inputs:
            comp.fix()

        # TODO: safeguard solver call to see if solver terminated successfully
        # see below commented code:
        # res = self.solver.solve(model, tee=self.tee, load_solutions=False)
        # if pyo.check_optimal_termination(res):
        #     model.load_solution(res)
        # else:
        #     # The solver was unsuccessful, might want to warn the user
        #     # or terminate gracefully, etc.
        model.dummy_obj = pyo.Objective(expr=0, sense=pyo.minimize)
        self.solver.solve(model, tee=self.tee)

        # Track time to initialize the DoE model
        initialization_time = sp_timer.toc(msg=None)
        self.logger.info(
            (
                "Successfully initialized the DoE model."
                "\nInitialization time: %0.1f seconds" % initialization_time
            )
        )

        model.dummy_obj.deactivate()

        # Reactivate objective and unfix experimental design decisions
        for comp in model.fd_scenario_blocks[0].experiment_inputs:
            comp.unfix()
        model.objective.activate()
        model.obj_cons.activate()

        if self.use_grey_box:
            # Initialize grey box inputs to be fim values currently
            for i in model.parameter_names:
                for j in model.parameter_names:
                    if list(model.parameter_names).index(i) >= list(
                        model.parameter_names
                    ).index(j):
                        model.obj_cons.egb_fim_block.inputs[(j, i)].set_value(
                            pyo.value(model.fim[(i, j)])
                        )
            # Set objective value
            if self.objective_option == ObjectiveLib.trace:
                trace_val = np.trace(np.linalg.pinv(self.get_FIM()))
                model.obj_cons.egb_fim_block.outputs["A-opt"].set_value(trace_val)
            elif self.objective_option == ObjectiveLib.determinant:
                det_val = np.linalg.det(np.array(self.get_FIM()))
                model.obj_cons.egb_fim_block.outputs["log-D-opt"].set_value(
                    np.log(det_val)
                )
            elif self.objective_option == ObjectiveLib.minimum_eigenvalue:
                eig, _ = np.linalg.eig(np.array(self.get_FIM()))
                model.obj_cons.egb_fim_block.outputs["E-opt"].set_value(np.min(eig))
            elif self.objective_option == ObjectiveLib.condition_number:
                eig, _ = np.linalg.eig(np.array(self.get_FIM()))
                cond_number = np.log(np.abs(np.max(eig) / np.min(eig)))
                model.obj_cons.egb_fim_block.outputs["ME-opt"].set_value(cond_number)

        # If the model has L, initialize it with the solved FIM
        if hasattr(model, "L"):
            # Get the FIM values
            fim_vals = [
                pyo.value(model.fim[i, j])
                for i in model.parameter_names
                for j in model.parameter_names
            ]
            fim_np = np.array(fim_vals).reshape(
                (len(model.parameter_names), len(model.parameter_names))
            )

            # Need to compute the full FIM before
            # initializing the Cholesky factorization
            if self.only_compute_fim_lower:
                fim_np = fim_np + fim_np.T - np.diag(np.diag(fim_np))

            # Check if the FIM is positive definite
            # If not, add jitter to the diagonal
            # to ensure positive definiteness
            min_eig = np.min(np.real(np.linalg.eigvals(fim_np)))

            if min_eig < _SMALL_TOLERANCE_DEFINITENESS:
                # Raise the minimum eigenvalue to at
                # least _SMALL_TOLERANCE_DEFINITENESS
                jitter = np.min(
                    [
                        -min_eig + _SMALL_TOLERANCE_DEFINITENESS,
                        _SMALL_TOLERANCE_DEFINITENESS,
                    ]
                )
            else:
                # No jitter needed
                jitter = 0

            # Add jitter to the diagonal to ensure positive definiteness
            L_vals_sq = np.linalg.cholesky(
                fim_np + jitter * np.eye(len(model.parameter_names))
            )
            for i, c in enumerate(model.parameter_names):
                for j, d in enumerate(model.parameter_names):
                    model.L[c, d].value = L_vals_sq[i, j]

            # Initialize the inverse of L if it exists
            if hasattr(model, "L_inv"):
                L_inv_vals = np.linalg.inv(L_vals_sq)

                for i, c in enumerate(model.parameter_names):
                    for j, d in enumerate(model.parameter_names):
                        if i >= j:
                            model.L_inv[c, d].value = L_inv_vals[i, j]
                        else:
                            model.L_inv[c, d].value = 0.0
                # Initialize the cov_trace if it exists
                if hasattr(model, "cov_trace"):
                    initial_cov_trace = np.sum(L_inv_vals**2)
                    model.cov_trace.value = initial_cov_trace

        if hasattr(model, "determinant"):
            model.determinant.value = np.linalg.det(np.array(self.get_FIM()))

        # Solve the full model, which has now been initialized with the square solve
        if self.use_grey_box:
            res = self.grey_box_solver.solve(model, tee=self.grey_box_tee)
        else:
            res = self.solver.solve(model, tee=self.tee)

        # Track time used to solve the DoE model
        solve_time = sp_timer.toc(msg=None)

        self.logger.info(
            (
                "Successfully optimized experiment."
                "\nSolve time: %0.1f seconds" % solve_time
            )
        )
        self.logger.info(
            "Total time for build, initialization, and solve: %0.1f seconds"
            % (build_time + initialization_time + solve_time)
        )

        # Avoid accidental carry-over of FIM information
        fim_local = self.get_FIM()

        # Make sure stale results don't follow the DoE object instance
        self.results = {}

        self.results["Solver Status"] = res.solver.status
        self.results["Termination Condition"] = res.solver.termination_condition
        if type(res.solver.message) is str:
            results_message = res.solver.message
        elif type(res.solver.message) is bytes:
            results_message = res.solver.message.decode("utf-8")
        self.results["Termination Message"] = results_message

        # Important quantities for optimal design
        self.results["FIM"] = fim_local
        self.results["Sensitivity Matrix"] = self.get_sensitivity_matrix()
        self.results["Experiment Design"] = self.get_experiment_input_values()
        self.results["Experiment Design Names"] = [
            str(pyo.ComponentUID(k, context=model.fd_scenario_blocks[0]))
            for k in model.fd_scenario_blocks[0].experiment_inputs
        ]
        self.results["Experiment Outputs"] = self.get_experiment_output_values()
        self.results["Experiment Output Names"] = [
            str(pyo.ComponentUID(k, context=model.fd_scenario_blocks[0]))
            for k in model.fd_scenario_blocks[0].experiment_outputs
        ]
        self.results["Unknown Parameters"] = self.get_unknown_parameter_values()
        self.results["Unknown Parameter Names"] = [
            str(pyo.ComponentUID(k, context=model.fd_scenario_blocks[0]))
            for k in model.fd_scenario_blocks[0].unknown_parameters
        ]
        self.results["Measurement Error"] = self.get_measurement_error_values()
        self.results["Measurement Error Names"] = [
            str(pyo.ComponentUID(k, context=model.fd_scenario_blocks[0]))
            for k in model.fd_scenario_blocks[0].measurement_error
        ]

        self.results["Prior FIM"] = [list(row) for row in list(self.prior_FIM)]

        # Saving some stats on the FIM for convenience
        self.results["Objective expression"] = str(self.objective_option).split(".")[-1]
        self.results["log10 A-opt"] = np.log10(np.trace(np.linalg.inv(fim_local)))
        self.results["log10 pseudo A-opt"] = np.log10(np.trace(fim_local))
        self.results["log10 D-opt"] = np.log10(np.linalg.det(fim_local))
        self.results["log10 E-opt"] = np.log10(min(np.linalg.eig(fim_local)[0]))
        self.results["FIM Condition Number"] = np.linalg.cond(fim_local)

        # Solve timing stats
        self.results["Build Time"] = build_time
        self.results["Initialization Time"] = initialization_time
        self.results["Solve Time"] = solve_time
        self.results["Wall-clock Time"] = build_time + initialization_time + solve_time

        # Settings used to generate the optimal DoE
        self.results["Finite Difference Scheme"] = str(self.fd_formula).split(".")[-1]
        self.results["Finite Difference Step"] = self.step
        self.results["Nominal Parameter Scaling"] = self.scale_nominal_param_value

        # TODO: Add more useful fields to the results object?
        # TODO: Add MetaData from the user to the results object? Or leave to the user?

        # If the user specifies to save the file, do it here as a json
        if results_file is not None:
            with open(results_file, "w") as file:
                json.dump(self.results, file)

    def optimize_experiments(
        self,
        _parameter_scenarios=None,
        results_file=None,
        n_exp: int = None,
        initialization_method=None,
        init_n_samples: int = 5,
        init_seed: int = None,
        init_parallel: bool = False,
        init_combo_parallel: bool = False,
        init_n_workers: int = None,
        init_combo_chunk_size: int = 5000,
        init_combo_parallel_threshold: int = 20000,
        init_max_wall_clock_time: float = None,
    ):
        """
        Optimize single experiment or multiple experiments simultaneously for
        Design of Experiments.

        The number of experiments is determined by the length of the
        experiment_list provided when creating the DesignOfExperiments object.

        Parameters
        ----------
        _parameter_scenarios:
            `dataclass` of parameter scenarios to consider for the multi-experiment
            optimization. This is currently unsupported; passing anything other
            than ``None`` raises ``NotImplementedError``. It is a placeholder
            for future functionality to incorporate parametric uncertainty.
            Default: None

        results_file:
            string name of the file path to save the results to in the form
            of a .json file

        initialization_method:
            Method used to initialize the experiment design variables before
            optimization. Options are:

            - ``None`` (default): No special initialization; use the initial
              values from ``get_labeled_model()``. To provide a custom starting
              point, initialize the ``Experiment`` objects with the desired
              design values before passing them in ``experiment_list``.
            - ``"lhs"`` (or ``InitializationMethod.lhs``): Use Latin Hypercube Sampling (LHS) to find a good
              initial design. For each experiment-input dimension, ``init_n_samples``
              points are sampled independently using 1-D LHS, and their Cartesian
              product forms the set of candidate experiment designs. The FIM is
              evaluated at every candidate, and the combination of ``n_exp``
              candidates (without replacement) that best satisfies the chosen
              objective is selected as the starting point for the optimization.

        init_n_samples:
            Number of LHS samples per experiment-input dimension when
            ``initialization_method="lhs"``. The total number of candidate
            designs is ``init_n_samples ** n_exp_inputs``. A warning is issued
            when this exceeds 10,000. Default: 5.

        init_seed:
            Integer seed for the LHS random-number generator (for
            reproducibility). Used only when ``initialization_method="lhs"``.
            Default: ``None`` (non-deterministic).
        init_parallel:
            If True, evaluate candidate-point FIMs in parallel during LHS
            initialization. Default: False.
        init_combo_parallel:
            If True, the scoring of Latin hypercube candidate combinations
            (``C(n_candidates, n_exp)`` during ``initialization_method="lhs"``)
            is split across a thread pool.  Each worker computes the scalar
            objective derived from the FIM for its chunk of combinations.  The
            flag has no effect unless ``initialization_method="lhs"`` and the
            total number of combinations exceeds ``init_combo_parallel_threshold``.
            Default: False.
        init_n_workers:
            Number of worker threads for combination FIM metric when
            ``init_combo_parallel=True``. Default: ``None`` (auto-select).
        init_combo_chunk_size:
            Number of combinations scored per worker task. Default: 5000.
        init_combo_parallel_threshold:
            Parallel combo scoring is used only when number of combinations is
            at least this value. Default: 20000.
        init_max_wall_clock_time:
            Optional time budget (seconds) for LHS initialization. If exceeded
            during combination scoring, best-so-far is returned.

        Notes
        -----
        Number of Experiments:
            When ``len(experiment_list) == 1`` (template mode), pass ``n_exp``
            to specify how many experiments to optimize.  When
            ``len(experiment_list) > 1`` (user-initialized mode), the list
            length determines the number of experiments and ``n_exp`` must
            not be set.

        Symmetry Breaking (for multiple experiments):
            To prevent equivalent permutations of identical experiments, you must
            mark a "primary" design variable using a Pyomo Suffix in your experiment's
            `label_experiment()` method:

            Example::

                m.sym_break_cons = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.sym_break_cons[m.CA[0]] = None  # Mark CA[0] as primary variable

            This will add constraints: exp[k-1].primary_var <= exp[k].primary_var
            for k = 1, ..., n_exp-1, which breaks permutation symmetry and can
            significantly reduce solve times.

        LHS Initialization (initialization_method="lhs"):
            Each dimension of the experiment inputs is sampled independently
            using a 1-D Latin Hypercube, giving ``init_n_samples`` evenly-spaced
            stratified samples across the variable bounds. The joint candidate
            set is the Cartesian product of these per-dimension samples (i.e.,
            a ``init_n_samples^n_inputs`` grid with good marginal coverage). The
            FIM is evaluated sequentially at each candidate, then all
            ``C(n_candidates, n_exp)`` combinations are scored and the best one
            is used as the initial point for the NLP solver. This can
            significantly improve solution quality when the problem has multiple
            local optima.

        Solver options in LHS worker evaluations:
            When ``init_parallel=True``, worker threads construct solver instances
            using the same solver name and options as ``self.solver`` (when
            available). Therefore, per-solve limits (e.g., iteration/time limits)
            configured on the main DoE solver are propagated to candidate FIM
            evaluations.
        """
        # Check results file name
        if results_file is not None:
            if not isinstance(results_file, (pathlib.Path, str)):
                raise ValueError(
                    "``results_file`` must be either a Path object or a string."
                )
        parameter_scenarios = _parameter_scenarios

        # --- Resolve n_exp and determine operating mode ---
        n_list = len(self.experiment_list)
        if n_list > 1:
            # User-initialized mode: experiment_list already contains all
            # pre-initialized experiment objects.
            if n_exp is not None:
                raise ValueError(
                    "``n_exp`` must not be set when ``experiment_list`` contains "
                    f"more than one experiment (got {n_list} experiments in the "
                    "list).  Either pass a single template experiment and set "
                    "``n_exp``, or pass a fully-initialized list and omit ``n_exp``."
                )
            n_exp = n_list
            _template_mode = False
        else:
            # Template mode: single experiment object cloned n_exp times.
            if n_exp is None:
                n_exp = 1  # default: single-experiment optimization
            elif not isinstance(n_exp, int) or n_exp < 1:
                raise ValueError(
                    f"``n_exp`` must be a positive integer, got {n_exp!r}."
                )
            _template_mode = True
        # ---------------------------------------------------

        # --- Validate initialization arguments ---
        if initialization_method is None:
            resolved_initialization_method = None
        else:
            try:
                resolved_initialization_method = InitializationMethod(
                    initialization_method
                )
            except ValueError:
                valid = ", ".join(f"'{m.value}'" for m in InitializationMethod)
                raise ValueError(
                    "``initialization_method`` must be one of [None, "
                    + valid
                    + f"], got {initialization_method!r}."
                )

        if resolved_initialization_method == InitializationMethod.lhs:
            if not _template_mode:
                raise ValueError(
                    "``initialization_method='lhs'`` is currently supported only in "
                    "template mode (``len(experiment_list) == 1``)."
                )
            if not scipy_available:
                raise ImportError(
                    "LHS initialization requires scipy. "
                    "Please install scipy to use initialization_method='lhs'."
                )
            if not isinstance(init_n_samples, int) or init_n_samples < 1:
                raise ValueError(
                    "``init_n_samples`` must be a positive integer, "
                    f"got {init_n_samples!r}."
                )
            if init_seed is not None and not isinstance(init_seed, int):
                raise ValueError(
                    "``init_seed`` must be None or an integer, " f"got {init_seed!r}."
                )
            if not isinstance(init_parallel, bool):
                raise ValueError(
                    f"``init_parallel`` must be a bool, got {init_parallel!r}."
                )
            if not isinstance(init_combo_parallel, bool):
                raise ValueError(
                    "``init_combo_parallel`` must be a bool, "
                    f"got {init_combo_parallel!r}."
                )
            if init_n_workers is not None and (
                not isinstance(init_n_workers, int) or init_n_workers < 1
            ):
                raise ValueError(
                    "``init_n_workers`` must be None or a positive integer, "
                    f"got {init_n_workers!r}."
                )
            if not isinstance(init_combo_chunk_size, int) or init_combo_chunk_size < 1:
                raise ValueError(
                    "``init_combo_chunk_size`` must be a positive integer, "
                    f"got {init_combo_chunk_size!r}."
                )
            if (
                not isinstance(init_combo_parallel_threshold, int)
                or init_combo_parallel_threshold < 1
            ):
                raise ValueError(
                    "``init_combo_parallel_threshold`` must be a positive integer, "
                    f"got {init_combo_parallel_threshold!r}."
                )
            if init_max_wall_clock_time is not None and (
                not isinstance(init_max_wall_clock_time, (int, float))
                or init_max_wall_clock_time <= 0
            ):
                raise ValueError(
                    "``init_max_wall_clock_time`` must be None or a positive number, "
                    f"got {init_max_wall_clock_time!r}."
                )
        # -----------------------------------------
        # Start timer
        sp_timer = TicTocTimer()
        sp_timer.tic(msg=None)
        self.logger.info(
            f"Beginning multi-experiment optimization with {n_exp} experiments."
        )
        # Rebuild the multi-experiment model from a clean base each call.
        self.model = pyo.ConcreteModel()

        if parameter_scenarios is None:
            n_param_scenarios = 1  # number of parameter scenarios
            # Use an immutable tuple since weights are not intended to be modified
            self.scenario_weights = (1.0,)  # Single scenario, weight = 1
        else:
            # TODO: Add parameter scenarios when incorporating parametric uncertainty
            raise NotImplementedError(
                "Parameter scenarios for multi-experiment optimization "
                "not yet supported."
            )
        # Add parameter scenario blocks to the model
        self.model.param_scenario_blocks = pyo.Block(range(n_param_scenarios))
        symmetry_breaking_info = {
            "enabled": n_exp > 1,
            "variable": None,
            "source": None,
        }
        diagnostics_warnings = []

        # Add experiment(s) for each scenario
        # TODO: Add s_prev = 0 to handle parameter scenarios
        for s in range(n_param_scenarios):
            self.model.param_scenario_blocks[s].exp_blocks = pyo.Block(range(n_exp))
            for k in range(n_exp):
                # Generate FIM and Sensitivity expressions for each experiment.
                # In template mode all experiments share the single template
                # (experiment_index=0); in user-initialized mode each experiment
                # maps to its own entry in experiment_list (experiment_index=k).
                self.create_doe_model(
                    model=self.model.param_scenario_blocks[s].exp_blocks[k],
                    experiment_index=0 if _template_mode else k,
                    _for_multi_experiment=True,  # Skip creating L matrix per experiment
                )
                # TODO: Update the parameter scenarios for each experiment block
                # when using parametric uncertainty

        # Add symmetry breaking constraints to prevent equivalent permutations for
        # multiple experiments
        if n_exp > 1:
            # Check if user provided a symmetry breaking variable via Suffix
            # Use first scenario since variable names are the same across all scenarios
            first_exp_block = (
                self.model.param_scenario_blocks[0].exp_blocks[0].fd_scenario_blocks[0]
            )

            # Determine symmetry breaking variable
            if (
                hasattr(first_exp_block, 'sym_break_cons')
                and len(first_exp_block.sym_break_cons) > 0
            ):
                # User provided symmetry breaking variable(s)
                sym_break_var_list = list(first_exp_block.sym_break_cons.keys())

                if len(sym_break_var_list) > 1:
                    warning_msg = (
                        f"Multiple variables marked in sym_break_cons. "
                        f"Using {sym_break_var_list[0].local_name} for symmetry breaking."
                    )
                    self.logger.warning(warning_msg)
                    diagnostics_warnings.append(warning_msg)

                sym_break_var = sym_break_var_list[0]
                if not any(
                    sym_break_var is inp
                    for inp in first_exp_block.experiment_inputs.keys()
                ):
                    raise ValueError(
                        "Variable selected in ``sym_break_cons`` must also be an "
                        "experiment input variable. "
                        f"Got non-input variable '{sym_break_var.local_name}'."
                    )
                symmetry_breaking_info["variable"] = sym_break_var.local_name
                symmetry_breaking_info["source"] = "user"
                self.logger.info(
                    f"Using user-specified variable '{sym_break_var.local_name}' for symmetry breaking."
                )
            else:
                # Use first experiment input as default symmetry breaking variable
                sym_break_var = next(iter(first_exp_block.experiment_inputs))
                symmetry_breaking_info["variable"] = sym_break_var.local_name
                symmetry_breaking_info["source"] = "auto"
                self.logger.warning(
                    "No symmetry breaking variable specified. Automatically using the first "
                    f"experiment input '{sym_break_var.local_name}' for ordering constraints. "
                    "To specify a different variable, add: "
                    "m.sym_break_cons = pyo.Suffix(direction=pyo.Suffix.LOCAL); "
                    "m.sym_break_cons[m.your_variable] = None"
                )
                diagnostics_warnings.append(
                    f"No symmetry breaking variable specified. Automatically using "
                    f"'{sym_break_var.local_name}'."
                )

            # Add constraints for each scenario
            for s in range(n_param_scenarios):
                for k in range(1, n_exp):
                    # Get the variable from experiment k-1
                    var_prev = pyo.ComponentUID(
                        sym_break_var, context=first_exp_block
                    ).find_component_on(
                        self.model.param_scenario_blocks[s]
                        .exp_blocks[k - 1]
                        .fd_scenario_blocks[0]
                    )

                    # Get the variable from experiment k
                    var_curr = pyo.ComponentUID(
                        sym_break_var, context=first_exp_block
                    ).find_component_on(
                        self.model.param_scenario_blocks[s]
                        .exp_blocks[k]
                        .fd_scenario_blocks[0]
                    )
                    if var_prev is None or var_curr is None:
                        raise RuntimeError(
                            "Failed to map symmetry breaking variable "
                            f"'{sym_break_var.local_name}' onto scenario {s}, "
                            f"experiment pair ({k - 1}, {k}). Ensure the variable "
                            "exists on all experiment blocks with compatible labels."
                        )

                    # Add symmetry breaking constraint
                    con_name = f"symmetry_breaking_s{s}_exp{k}"
                    self.model.param_scenario_blocks[s].add_component(
                        con_name, pyo.Constraint(expr=var_prev <= var_curr)
                    )

                self.logger.info(
                    f"Added {n_exp - 1} symmetry breaking constraints for scenario {s} "
                    f"using variable: {sym_break_var.local_name}"
                )

        # Create aggregated objective for multi-experiment optimization
        self.create_multi_experiment_objective_function(self.model)

        # Track time required to build the DoE model
        build_time = sp_timer.toc(msg=None)
        self.logger.info(
            "Successfully built the multi-experiment DoE model.\nBuild time: %0.1f seconds"
            % build_time
        )

        # --- Apply experiment initialization (if requested) ---
        # This must be done AFTER the model is built but BEFORE the square solve
        # so that the solver uses the correct starting design.
        lhs_init_diagnostics = None
        lhs_initialization_time = 0.0
        if resolved_initialization_method == InitializationMethod.lhs:
            lhs_timer = TicTocTimer()
            lhs_timer.tic(msg=None)
            self.logger.info(
                f"Applying LHS initialization with {init_n_samples} samples per "
                f"experiment-input dimension..."
            )
            best_initial_points, lhs_init_diagnostics = (
                self._lhs_initialize_experiments(
                    lhs_n_samples=init_n_samples,
                    lhs_seed=init_seed,
                    n_exp=n_exp,
                    lhs_parallel=init_parallel,
                    lhs_combo_parallel=init_combo_parallel,
                    lhs_n_workers=init_n_workers,
                    lhs_combo_chunk_size=init_combo_chunk_size,
                    lhs_combo_parallel_threshold=init_combo_parallel_threshold,
                    lhs_max_wall_clock_time=init_max_wall_clock_time,
                )
            )
            self.logger.info(
                "Setting LHS best-found initial design in the optimization model..."
            )
            for s in range(n_param_scenarios):
                for k in range(n_exp):
                    exp_input_vars = self._get_experiment_input_vars(
                        self.model.param_scenario_blocks[s].exp_blocks[k]
                    )
                    for var, val in zip(exp_input_vars, best_initial_points[k]):
                        var.set_value(val)
            lhs_initialization_time = lhs_timer.toc(msg=None)

        # ------------------------------------------------------
        # Reset delta timing so initialization_time measures only square solve.
        sp_timer.tic(msg=None)

        # Solve the square problem first to initialize the FIM and sensitivity constraints
        # Deactivate objective expression and objective constraints
        self.model.objective.deactivate()
        # Deactivate obj_cons for each scenario (holds Cholesky/determinant/trace constraints)
        for s in range(n_param_scenarios):
            if hasattr(self.model.param_scenario_blocks[s], "obj_cons"):
                self.model.param_scenario_blocks[s].obj_cons.deactivate()

        # Fix the design variables across all scenarios and experiments
        for s in range(n_param_scenarios):
            for k in range(n_exp):
                for comp in (
                    self.model.param_scenario_blocks[s]
                    .exp_blocks[k]
                    .fd_scenario_blocks[0]
                    .experiment_inputs
                ):
                    comp.fix()

        # Create and solve dummy objective for initialization
        self.model.dummy_obj = pyo.Objective(expr=0, sense=pyo.minimize)
        self.solver.solve(self.model, tee=self.tee)

        # Track time to initialize the DoE model
        initialization_time = sp_timer.toc(msg=None)
        self.logger.info(
            (
                "Successfully initialized the multi-experiment DoE model."
                "\nInitialization time: %0.1f seconds" % initialization_time
            )
        )

        # Deactivate dummy objective
        self.model.dummy_obj.deactivate()
        self.model.del_component("dummy_obj")

        # Reactivate objective, obj_cons, and unfix experimental design decisions
        for s in range(n_param_scenarios):
            for k in range(n_exp):
                for comp in (
                    self.model.param_scenario_blocks[s]
                    .exp_blocks[k]
                    .fd_scenario_blocks[0]
                    .experiment_inputs
                ):
                    comp.unfix()
        self.model.objective.activate()
        for s in range(n_param_scenarios):
            if hasattr(self.model.param_scenario_blocks[s], "obj_cons"):
                self.model.param_scenario_blocks[s].obj_cons.activate()

        # Initialize scenario-level variables (L, determinant, pseudo_trace) based on
        # the solved FIM values from the square solve
        parameter_names = (
            self.model.param_scenario_blocks[0].exp_blocks[0].parameter_names
        )

        for s in range(n_param_scenarios):
            scenario = self.model.param_scenario_blocks[s]
            # Update total_fim values from solved individual experiment FIMs
            # Individual experiment FIMs don't include prior_FIM in multi-experiment mode,
            # so we add it once here to the aggregated total
            for i, p in enumerate(parameter_names):
                for j, q in enumerate(parameter_names):
                    # When only_compute_fim_lower=True, only the lower triangle is computed
                    # Upper triangle elements are fixed to 0, so only aggregate lower triangle
                    if self.only_compute_fim_lower and i < j:
                        continue

                    fim_sum = sum(
                        pyo.value(scenario.exp_blocks[k].fim[p, q]) or 0
                        for k in range(n_exp)
                    )
                    scenario.total_fim[p, q].set_value(fim_sum + self.prior_FIM[i, j])

            # Initialize scenario-level variables based on total_fim
            if hasattr(scenario.obj_cons, "L"):
                # Compute Cholesky factorization
                total_fim_vals = [
                    pyo.value(scenario.total_fim[p, q])
                    for p in parameter_names
                    for q in parameter_names
                ]
                total_fim_np = np.array(total_fim_vals).reshape(
                    (len(parameter_names), len(parameter_names))
                )

                # Complete FIM if only computing lower triangle
                if self.only_compute_fim_lower:
                    total_fim_np = self._symmetrize_lower_tri(total_fim_np)

                # Check positive definiteness and add jitter if needed
                min_eig = np.min(np.real(np.linalg.eigvals(total_fim_np)))
                if min_eig < _SMALL_TOLERANCE_DEFINITENESS:
                    jitter = np.min(
                        [
                            -min_eig + _SMALL_TOLERANCE_DEFINITENESS,
                            _SMALL_TOLERANCE_DEFINITENESS,
                        ]
                    )
                else:
                    jitter = 0

                # Compute Cholesky decomposition
                L_vals = np.linalg.cholesky(
                    total_fim_np + jitter * np.eye(len(parameter_names))
                )

                # Initialize L values
                for i, p in enumerate(parameter_names):
                    for j, q in enumerate(parameter_names):
                        scenario.obj_cons.L[p, q].set_value(L_vals[i, j])

                # If trace objective, also initialize L_inv, fim_inv, and cov_trace
                if hasattr(scenario.obj_cons, "L_inv"):
                    L_inv_vals = np.linalg.inv(L_vals)

                    for i, p in enumerate(parameter_names):
                        for j, q in enumerate(parameter_names):
                            if i >= j:
                                scenario.obj_cons.L_inv[p, q].set_value(
                                    L_inv_vals[i, j]
                                )
                            else:
                                scenario.obj_cons.L_inv[p, q].set_value(0.0)

                    # Initialize fim_inv
                    if hasattr(scenario.obj_cons, "fim_inv"):
                        fim_inv_np = np.linalg.inv(
                            total_fim_np + jitter * np.eye(len(parameter_names))
                        )
                        for i, p in enumerate(parameter_names):
                            for j, q in enumerate(parameter_names):
                                scenario.obj_cons.fim_inv[p, q].set_value(
                                    fim_inv_np[i, j]
                                )

                    # Initialize cov_trace
                    if hasattr(scenario.obj_cons, "cov_trace"):
                        initial_cov_trace = np.sum(L_inv_vals**2)
                        scenario.obj_cons.cov_trace.set_value(initial_cov_trace)

            if hasattr(scenario.obj_cons, "determinant"):
                # Initialize determinant
                total_fim_vals = [
                    pyo.value(scenario.total_fim[p, q])
                    for p in parameter_names
                    for q in parameter_names
                ]
                total_fim_np = np.array(total_fim_vals).reshape(
                    (len(parameter_names), len(parameter_names))
                )
                scenario.obj_cons.determinant.set_value(np.linalg.det(total_fim_np))

            if hasattr(scenario.obj_cons, "pseudo_trace"):
                # Initialize pseudo_trace
                pseudo_trace_val = sum(
                    pyo.value(scenario.total_fim[j, j]) for j in parameter_names
                )
                scenario.obj_cons.pseudo_trace.set_value(pseudo_trace_val)

        # Solve the full model
        res = self.solver.solve(self.model, tee=self.tee)

        # Track time used to solve the DoE model
        solve_time = sp_timer.toc(msg=None)

        self.logger.info(
            (
                "Successfully optimized multi-experiment design."
                "\nSolve time: %0.1f seconds" % solve_time
            )
        )
        self.logger.info(
            "Total time for build, initialization, and solve: %0.1f seconds"
            % (build_time + initialization_time + solve_time)
        )

        # Collect results
        self.results = {}
        self.results["Solver Status"] = res.solver.status
        self.results["Termination Condition"] = res.solver.termination_condition
        if type(res.solver.message) is str:
            results_message = res.solver.message
        elif type(res.solver.message) is bytes:
            results_message = res.solver.message.decode("utf-8")
        else:
            results_message = (
                str(res.solver.message) if res.solver.message is not None else ""
            )
        self.results["Termination Message"] = results_message

        def _safe_metric(metric_name, compute_fn, scenario_index):
            try:
                val = float(compute_fn())
                return val if np.isfinite(val) else float("nan")
            except Exception as exc:
                self.logger.warning(
                    f"Scenario {scenario_index}: failed to compute {metric_name}: {exc}. "
                    "Setting metric to NaN."
                )
                return float("nan")

        # Store results for each scenario
        self.results["Scenarios"] = []
        scenarios_structured = []
        for s in range(n_param_scenarios):
            scenario = self.model.param_scenario_blocks[s]
            scenario_results = {}

            # Get aggregated FIM for this scenario
            total_fim_vals = [
                pyo.value(scenario.total_fim[p, q])
                for p in parameter_names
                for q in parameter_names
            ]
            total_fim_np = np.array(total_fim_vals).reshape(
                (len(parameter_names), len(parameter_names))
            )

            # Complete FIM if only computing lower triangle
            if self.only_compute_fim_lower:
                total_fim_np = self._symmetrize_lower_tri(total_fim_np)

            # Store the completed (symmetric) FIM
            scenario_results["Total FIM"] = total_fim_np.tolist()

            # Statistics on the aggregated FIM (consistent with run_doe), guarded
            # against singular/indefinite matrices and numerical failures.
            scenario_results["log10 A-opt"] = _safe_metric(
                "log10 A-opt",
                lambda: np.log10(np.trace(np.linalg.inv(total_fim_np))),
                s,
            )
            scenario_results["log10 pseudo A-opt"] = _safe_metric(
                "log10 pseudo A-opt", lambda: np.log10(np.trace(total_fim_np)), s
            )
            scenario_results["log10 D-opt"] = _safe_metric(
                "log10 D-opt", lambda: np.log10(np.linalg.det(total_fim_np)), s
            )
            scenario_results["log10 E-opt"] = _safe_metric(
                "log10 E-opt",
                lambda: np.log10(min(np.linalg.eigvalsh(total_fim_np))),
                s,
            )
            scenario_results["log10 ME-opt"] = _safe_metric(
                "log10 ME-opt", lambda: np.log10(np.linalg.cond(total_fim_np)), s
            )

            # Store unknown parameter values at scenario level (same for all experiments)
            # Use first experiment to get the values
            scenario_results["Unknown Parameters"] = self.get_unknown_parameter_values(
                model=scenario.exp_blocks[0]
            )

            # Store results for each experiment in this scenario
            scenario_results["Experiments"] = []
            scenario_structured = {
                "id": s,
                "total_fim": total_fim_np.tolist(),
                "metrics": {
                    "log10_a_opt": scenario_results["log10 A-opt"],
                    "log10_pseudo_a_opt": scenario_results["log10 pseudo A-opt"],
                    "log10_d_opt": scenario_results["log10 D-opt"],
                    "log10_e_opt": scenario_results["log10 E-opt"],
                    "log10_me_opt": scenario_results["log10 ME-opt"],
                },
                "unknown_parameters": None,
                "experiments": [],
            }
            for k in range(n_exp):
                exp_block = scenario.exp_blocks[k]
                exp_results = {}

                # Use helper functions for consistent extraction (same as run_doe)
                # Store only the VALUES for each experiment (names are at top level)
                exp_results["Experiment Design"] = self.get_experiment_input_values(
                    model=exp_block
                )
                exp_results["Experiment Outputs"] = self.get_experiment_output_values(
                    model=exp_block
                )
                exp_results["Measurement Error"] = self.get_measurement_error_values(
                    model=exp_block
                )

                # Individual experiment FIM (get_FIM handles symmetry completion)
                exp_results["FIM"] = self.get_FIM(model=exp_block)

                # Sensitivity matrix for this experiment
                if hasattr(exp_block, "sensitivity_jacobian"):
                    exp_results["Sensitivity Matrix"] = self.get_sensitivity_matrix(
                        model=exp_block
                    )

                scenario_results["Experiments"].append(exp_results)
                experiment_structured = {
                    "id": k,
                    "design": exp_results["Experiment Design"],
                    "outputs": exp_results["Experiment Outputs"],
                    "measurement_error": exp_results["Measurement Error"],
                    "fim": exp_results["FIM"],
                }
                if "Sensitivity Matrix" in exp_results:
                    experiment_structured["sensitivity"] = exp_results[
                        "Sensitivity Matrix"
                    ]
                scenario_structured["experiments"].append(experiment_structured)

            self.results["Scenarios"].append(scenario_results)
            scenario_structured["unknown_parameters"] = scenario_results[
                "Unknown Parameters"
            ]
            scenarios_structured.append(scenario_structured)

        # Store variable names once (structural properties, same across all scenarios/experiments)
        # Use first scenario's first experiment to get the structure
        first_exp_block_fd = (
            self.model.param_scenario_blocks[0].exp_blocks[0].fd_scenario_blocks[0]
        )
        self.results["Experiment Design Names"] = [
            str(pyo.ComponentUID(comp, context=first_exp_block_fd))
            for comp in first_exp_block_fd.experiment_inputs
        ]
        self.results["Experiment Output Names"] = [
            str(pyo.ComponentUID(comp, context=first_exp_block_fd))
            for comp in first_exp_block_fd.experiment_outputs
        ]
        self.results["Unknown Parameter Names"] = [
            str(pyo.ComponentUID(comp, context=first_exp_block_fd))
            for comp in first_exp_block_fd.unknown_parameters
        ]
        self.results["Measurement Error Names"] = [
            str(pyo.ComponentUID(comp, context=first_exp_block_fd))
            for comp in first_exp_block_fd.measurement_error
        ]

        # Store general settings and info
        self.results["Number of Scenarios"] = n_param_scenarios
        self.results["Number of Experiments per Scenario"] = n_exp
        self.results["Prior FIM"] = [list(row) for row in list(self.prior_FIM)]
        self.results["Objective expression"] = str(self.objective_option).split(".")[-1]
        self.results["Finite Difference Scheme"] = str(self.fd_formula).split(".")[-1]
        self.results["Finite Difference Step"] = self.step
        self.results["Nominal Parameter Scaling"] = self.scale_nominal_param_value

        # Initialization info
        self.results["Initialization Method"] = (
            resolved_initialization_method.value
            if resolved_initialization_method is not None
            else "none"
        )
        if resolved_initialization_method == InitializationMethod.lhs:
            self.results["LHS Samples Per Dimension"] = init_n_samples
            self.results["LHS Seed"] = init_seed
            self.results["LHS Best Initial Points"] = best_initial_points

        # Timing statistics
        self.results["Build Time"] = build_time
        self.results["Initialization Time"] = initialization_time
        self.results["LHS Initialization Time"] = lhs_initialization_time
        self.results["Solve Time"] = solve_time
        self.results["Wall-clock Time"] = (
            build_time + lhs_initialization_time + initialization_time + solve_time
        )

        # Structured result payload
        objective_sense = (
            "maximize"
            if self.objective_option in self._MAXIMIZE_OBJECTIVES
            else "minimize"
        )

        self.results["run_info"] = {
            "api": "DesignOfExperiments.optimize_experiments",
            "solver": {
                "name": getattr(self.solver, "name", str(self.solver)),
                "status": self.results["Solver Status"],
                "termination_condition": self.results["Termination Condition"],
                "message": self.results["Termination Message"],
            },
        }
        self.results["settings"] = {
            "objective": {
                "name": self.results["Objective expression"],
                "sense": objective_sense,
            },
            "finite_difference": {
                "scheme": self.results["Finite Difference Scheme"],
                "step": self.results["Finite Difference Step"],
            },
            "scaling": {
                "nominal_parameter_scaling": self.results["Nominal Parameter Scaling"]
            },
            "initialization": {
                "method": self.results["Initialization Method"],
                "lhs_n_samples": self.results.get("LHS Samples Per Dimension"),
                "lhs_seed": self.results.get("LHS Seed"),
                "best_points": self.results.get("LHS Best Initial Points"),
                "lhs_parallel": (
                    init_parallel
                    if resolved_initialization_method == InitializationMethod.lhs
                    else None
                ),
                "lhs_combo_parallel": (
                    init_combo_parallel
                    if resolved_initialization_method == InitializationMethod.lhs
                    else None
                ),
                "lhs_n_workers": (
                    init_n_workers
                    if resolved_initialization_method == InitializationMethod.lhs
                    else None
                ),
                "lhs_combo_chunk_size": (
                    init_combo_chunk_size
                    if resolved_initialization_method == InitializationMethod.lhs
                    else None
                ),
                "lhs_combo_parallel_threshold": (
                    init_combo_parallel_threshold
                    if resolved_initialization_method == InitializationMethod.lhs
                    else None
                ),
                "lhs_max_wall_clock_time": (
                    init_max_wall_clock_time
                    if resolved_initialization_method == InitializationMethod.lhs
                    else None
                ),
            },
            "modeling": {
                "n_scenarios": self.results["Number of Scenarios"],
                "n_experiments_per_scenario": self.results[
                    "Number of Experiments per Scenario"
                ],
                "template_mode": _template_mode,
            },
            "prior_fim": self.results["Prior FIM"],
        }
        self.results["timing"] = {
            "build_s": self.results["Build Time"],
            "lhs_initialization_s": self.results["LHS Initialization Time"],
            "initialization_s": self.results["Initialization Time"],
            "solve_s": self.results["Solve Time"],
            "total_s": self.results["Wall-clock Time"],
        }
        self.results["names"] = {
            "experiment_design": self.results["Experiment Design Names"],
            "experiment_output": self.results["Experiment Output Names"],
            "unknown_parameter": self.results["Unknown Parameter Names"],
            "measurement_error": self.results["Measurement Error Names"],
        }
        self.results["diagnostics"] = {
            "symmetry_breaking": symmetry_breaking_info,
            "warnings": diagnostics_warnings,
            "lhs_initialization": lhs_init_diagnostics,
        }
        self.results["scenarios"] = scenarios_structured

        # Save results to file if requested
        if results_file is not None:
            with open(results_file, "w") as file:
                json.dump(self.results, file, indent=2, cls=_DoEResultsJSONEncoder)

    # LHS-initialization helpers
    def _get_experiment_input_vars(self, exp_block):
        """
        Return the experiment-input Pyomo variable objects for an experiment
        block, abstracting over the specific sensitivity-computation structure
        (FD, AD, etc.).

        When the block exposes ``experiment_inputs`` directly (e.g. in a future
        automatic-differentiation path), those are used.  Otherwise the method
        falls back to the FD structure (``exp_block.fd_scenario_blocks[0]``).

        Parameters
        ----------
        exp_block : Pyomo Block
            An ``exp_blocks[k]`` sub-block of the multi-experiment model.

        Returns
        -------
        list
            Ordered list of Pyomo :class:`Var` objects corresponding to the
            experiment inputs.
        """
        if hasattr(exp_block, "experiment_inputs"):
            return list(exp_block.experiment_inputs.keys())
        # FD structure: inputs live on the base finite-difference scenario block
        return list(exp_block.fd_scenario_blocks[0].experiment_inputs.keys())

    @staticmethod
    def _evaluate_objective_for_option(fim_matrix, objective_option):
        _bad = (
            -np.inf
            if objective_option in DesignOfExperiments._MAXIMIZE_OBJECTIVES
            else np.inf
        )

        try:
            if objective_option == ObjectiveLib.determinant:
                return float(np.linalg.det(fim_matrix))
            elif objective_option == ObjectiveLib.pseudo_trace:
                return float(np.trace(fim_matrix))
            elif objective_option == ObjectiveLib.trace:
                return float(np.trace(np.linalg.inv(fim_matrix)))
            else:  # minimum_eigenvalue, condition_number, zero, or unknown
                return 0.0
        except (np.linalg.LinAlgError, ValueError):
            return _bad

    def _evaluate_objective_from_fim(self, fim_matrix):
        """
        Compute the scalar DoE objective from a numpy FIM matrix.

        Parameters
        ----------
        fim_matrix : np.ndarray
            Square FIM to score.

        Returns
        -------
        float
            Objective value.  For maximisation objectives (``determinant``,
            ``pseudo_trace``) a larger value is better.  For minimisation
            objectives (``trace`` / A-optimality) a smaller value is better.
            LHS initialization is not supported for ``minimum_eigenvalue`` or
            ``condition_number``; those return 0.0.
        """
        return self._evaluate_objective_for_option(fim_matrix, self.objective_option)

    @staticmethod
    def _symmetrize_lower_tri(mat):
        """Mirror lower-triangle FIM entries to the upper triangle."""
        return mat + mat.T - np.diag(np.diag(mat))

    @staticmethod
    def _make_cholesky_rule(fim_expr, L_expr, parameter_names):
        """
        Return a constraint rule that enforces ``fim_expr = L_expr * L_expr^T``
        on the lower-triangular portion defined by ``parameter_names``.

        The produced rule follows the Pyomo signature ``rule(block, p, q)``
        but does **not** actually use `block` in its body; the two matrix
        expressions are captured from the enclosing scope.

        Parameters
        ----------
        fim_expr : Var-like
            Indexed by ``(p, q)``; usually ``model.fim`` or
            ``scenario.total_fim``.
        L_expr : Var-like
            Indexed by ``(p, q)``; the corresponding lower-triangular
            Cholesky factors.
        parameter_names : Set
            Pyomo Set listing the parameter indices in order.
        """

        def rule(_b, p, q):
            p_idx = list(parameter_names).index(p)
            q_idx = list(parameter_names).index(q)
            if p_idx >= q_idx:
                return fim_expr[p, q] == sum(
                    L_expr[p, parameter_names.at(k + 1)]
                    * L_expr[q, parameter_names.at(k + 1)]
                    for k in range(q_idx + 1)
                )
            else:
                return pyo.Constraint.Skip

        return rule

    @staticmethod
    def _make_cholesky_inv_rule(fim_inv_expr, L_inv_expr, parameter_names):
        """
        Return a rule that enforces ``fim_inv_expr = L_inv_expr^T * L_inv_expr``
        over the lower-triangular index region.
        """

        def rule(_b, p, q):
            p_idx = list(parameter_names).index(p)
            q_idx = list(parameter_names).index(q)
            if p_idx >= q_idx:
                return fim_inv_expr[p, q] == sum(
                    L_inv_expr[parameter_names.at(k + 1), p]
                    * L_inv_expr[parameter_names.at(k + 1), q]
                    for k in range(p_idx, len(parameter_names))
                )
            return pyo.Constraint.Skip

        return rule

    @staticmethod
    def _make_cholesky_LLinv_rule(L_expr, L_inv_expr, parameter_names):
        """
        Return a rule that enforces ``L_expr * L_inv_expr = I`` over the
        lower-triangular index region.
        """

        def rule(_b, p, q):
            p_idx = list(parameter_names).index(p)
            q_idx = list(parameter_names).index(q)
            if p_idx < q_idx:
                return pyo.Constraint.Skip
            target = 1 if p_idx == q_idx else 0
            return (
                sum(
                    L_expr[p, parameter_names.at(k + 1)]
                    * L_inv_expr[parameter_names.at(k + 1), q]
                    for k in range(len(parameter_names))
                )
                == target
            )

        return rule

    def _compute_fim_at_point_no_prior(self, experiment_index, input_values):
        """
        Compute the FIM (without the prior FIM contribution) for the given
        experiment at the specified experiment-input values using the
        sequential finite-difference method.

        Parameters
        ----------
        experiment_index : int
            Index of the experiment in ``self.experiment_list`` to evaluate.
        input_values : list
            Numeric values for each experiment input variable (same order as
            ``model.experiment_inputs``).

        Returns
        -------
        np.ndarray
            ``(n_params, n_params)`` FIM matrix, **excluding** the prior.
            A zero matrix is returned on solver failure (with a warning).
        """
        # Get a fresh labeled model for this experiment
        model = (
            self.experiment_list[experiment_index]
            .get_labeled_model(**self.get_labeled_model_args)
            .clone()
        )
        self.check_model_labels(model=model)
        n_params = len(model.unknown_parameters)

        # Override experiment input values
        update_model_from_suffix(
            suffix_obj=model.experiment_inputs, values=input_values
        )

        # Temporarily zero the prior so that seq_FIM = Q^T * 𝚺^{-1} * Q only
        saved_prior = self.prior_FIM
        self.prior_FIM = np.zeros((n_params, n_params))

        try:
            self._sequential_FIM(model=model)
            fim = self.seq_FIM.copy()
        except Exception as exc:
            self.logger.warning(
                f"FIM evaluation failed at point {input_values}: {exc}. "
                "Using zero matrix as fallback."
            )
            fim = np.zeros((n_params, n_params))
        finally:
            self.prior_FIM = saved_prior

        return fim

    def _lhs_initialize_experiments(
        self,
        lhs_n_samples,
        lhs_seed,
        n_exp,
        lhs_parallel=False,
        lhs_combo_parallel=False,
        lhs_n_workers=None,
        lhs_combo_chunk_size=5000,
        lhs_combo_parallel_threshold=20000,
        lhs_max_wall_clock_time=None,
    ):
        """
        Use per-dimension Latin Hypercube Sampling to identify a good initial
        experiment design for ``optimize_experiments``.
        """
        # Use a monotonic wall-clock for progress estimates and deadline checks
        # in both serial and threaded LHS evaluation loops.
        t_start = time.perf_counter()

        # 1.  Get experiment-input bounds from the already-built model
        first_exp_block = self.model.param_scenario_blocks[0].exp_blocks[0]
        exp_input_vars = self._get_experiment_input_vars(first_exp_block)
        n_inputs = len(exp_input_vars)

        missing = [v.name for v in exp_input_vars if v.lb is None or v.ub is None]
        if missing:
            raise ValueError(
                "LHS initialization requires explicit lower and upper bounds on "
                "all experiment input variables. The following variables are "
                f"missing bounds: {missing}. "
                "Set bounds in your experiment input variables before "
                "calling ``optimize_experiments`` with "
                "``initialization_method='lhs'``."
            )

        lb_vals = np.array([v.lb for v in exp_input_vars])
        ub_vals = np.array([v.ub for v in exp_input_vars])

        # 2.  Generate per-dimension 1-D LHS samples and take Cartesian product
        # Split the user seed into per-dimension seeds so each 1-D LHS draw
        # is independent while remaining reproducible for a fixed lhs_seed.
        rng = np.random.default_rng(lhs_seed)
        per_dim_samples = []
        for i in range(n_inputs):
            dim_seed = int(rng.integers(0, 2**31))
            sampler = LatinHypercube(d=1, seed=dim_seed)
            s_unit = sampler.random(n=lhs_n_samples).flatten()
            s_scaled = lb_vals[i] + s_unit * (ub_vals[i] - lb_vals[i])
            per_dim_samples.append(s_scaled.tolist())

        candidate_points = tuple(product(*per_dim_samples))
        t_after_sampling = time.perf_counter()
        n_candidates = len(candidate_points)

        if n_candidates > 10_000:
            warnings.warn(
                f"LHS initialization generated {n_candidates:,} candidate "
                f"experiment designs (lhs_n_samples={lhs_n_samples}, "
                f"n_inputs={n_inputs}). Evaluating FIM at all candidates may "
                "take a long time. Consider reducing ``lhs_n_samples``.",
                UserWarning,
                stacklevel=2,
            )

        if hasattr(first_exp_block, "fd_scenario_blocks"):
            n_scenarios_per_candidate = len(list(first_exp_block.fd_scenario_blocks))
        else:
            n_scenarios_per_candidate = 1
        self.logger.info(
            f"LHS: evaluating FIM at {n_candidates} candidate designs "
            f"({n_candidates * n_scenarios_per_candidate} sequential solver calls expected)."
        )
        # Change the following block if we add support for LHS initialization with
        # non-FD structures (e.g. AD)
        if (
            not hasattr(first_exp_block, "fd_scenario_blocks")
            or len(first_exp_block.fd_scenario_blocks) == 0
        ):
            raise RuntimeError(
                "_lhs_initialize_experiments requires finite-difference scenario "
                "blocks on the experiment model. Ensure optimize_experiments is "
                "using the sequential FIM path."
            )

        resolved_workers = (
            lhs_n_workers
            if lhs_n_workers is not None
            else max(1, min(os.cpu_count() or 1, 8))
        )

        # Track worker DoE construction failures to avoid repeated logging of the same issue
        _solver_fallback_lock = threading.Lock()
        _solver_fallback_logged = False

        def _make_worker_solver():
            nonlocal _solver_fallback_logged
            solver_name = getattr(self.solver, "name", None)
            if solver_name is None:
                with _solver_fallback_lock:
                    if not _solver_fallback_logged:
                        self.logger.debug(
                            "LHS parallel: solver has no 'name' attribute; worker DoE "
                            "will use default solver settings."
                        )
                        _solver_fallback_logged = True
                return None
            worker_solver = pyo.SolverFactory(solver_name)
            if worker_solver is None:
                with _solver_fallback_lock:
                    if not _solver_fallback_logged:
                        self.logger.debug(
                            f"LHS parallel: could not construct solver '{solver_name}'; "
                            "worker DoE will use default solver settings."
                        )
                        _solver_fallback_logged = True
                return None
            try:
                if hasattr(self.solver, "options") and hasattr(
                    worker_solver, "options"
                ):
                    worker_solver.options.update(self.solver.options)
            except Exception:
                pass
            return worker_solver

        thread_state = threading.local()
        n_params = len(first_exp_block.fd_scenario_blocks[0].unknown_parameters)

        def _compute_candidate_fim(idx_pt):
            idx, pt = idx_pt
            try:
                worker_doe = getattr(thread_state, "doe", None)
                if worker_doe is None:
                    # Retry worker DoE construction on subsequent points even if a
                    # previous point failed; transient failures should not disable
                    # the rest of this thread's candidate evaluations.
                    worker_solver = _make_worker_solver()
                    worker_doe = DesignOfExperiments(
                        experiment_list=self.experiment_list,
                        fd_formula=self.fd_formula.value,
                        step=self.step,
                        objective_option=self.objective_option.value,
                        use_grey_box_objective=self.use_grey_box,
                        scale_constant_value=self.scale_constant_value,
                        scale_nominal_param_value=self.scale_nominal_param_value,
                        prior_FIM=self.prior_FIM,
                        jac_initial=self.jac_initial,
                        fim_initial=self.fim_initial,
                        L_diagonal_lower_bound=self.L_diagonal_lower_bound,
                        solver=worker_solver,
                        grey_box_solver=self.grey_box_solver,
                        tee=False,
                        grey_box_tee=False,
                        get_labeled_model_args=self.get_labeled_model_args,
                        logger_level=logging.ERROR,
                        improve_cholesky_roundoff_error=self.improve_cholesky_roundoff_error,
                        _Cholesky_option=self.Cholesky_option,
                        _only_compute_fim_lower=self.only_compute_fim_lower,
                    )
                    thread_state.doe = worker_doe
                # LHS initialization evaluates candidate points against the
                # canonical experiment template (experiment_list[0]).
                fim = worker_doe._compute_fim_at_point_no_prior(
                    experiment_index=0, input_values=list(pt)
                )
                return idx, fim
            except Exception as exc:
                if not getattr(thread_state, "construction_failed", False):
                    thread_state.construction_failed = True
                    self.logger.error(
                        f"LHS: worker DoE construction/evaluation failed on thread "
                        f"{threading.current_thread().name}: {exc}. "
                        "Using zero FIM for this candidate and continuing."
                    )
                return idx, np.zeros((n_params, n_params))

        # 3.  Evaluate FIM at every candidate design
        candidate_fims = [None] * n_candidates
        use_parallel_fim = lhs_parallel and resolved_workers > 1
        timed_out = False
        deadline = (
            None
            if lhs_max_wall_clock_time is None
            else (t_start + lhs_max_wall_clock_time)
        )
        if use_parallel_fim:
            self.logger.info(
                f"LHS: using parallel candidate FIM evaluation with {resolved_workers} workers."
            )
            idx_iter = iter(enumerate(candidate_points))
            max_pending = max(1, 2 * resolved_workers)
            with _cf.ThreadPoolExecutor(max_workers=resolved_workers) as ex:
                pending = set()
                n_done = 0
                while True:
                    while len(pending) < max_pending:
                        if deadline is not None and time.perf_counter() > deadline:
                            timed_out = True
                            break
                        try:
                            idx_pt = next(idx_iter)
                        except StopIteration:
                            break
                        pending.add(ex.submit(_compute_candidate_fim, idx_pt))

                    if not pending:
                        break

                    done_now, pending = _cf.wait(
                        pending, timeout=0.1, return_when=_cf.FIRST_COMPLETED
                    )
                    for fut in done_now:
                        pt_idx, fim = fut.result()
                        candidate_fims[pt_idx] = fim
                        n_done += 1
                        if n_done % max(1, n_candidates // 10) == 0:
                            elapsed = time.perf_counter() - t_start
                            frac_done = n_done / n_candidates
                            est_total = elapsed / frac_done if frac_done > 0 else 0
                            self.logger.info(
                                f"  LHS FIM eval: {n_done}/{n_candidates} "
                                f"({elapsed:.1f}s elapsed, ~{est_total:.1f}s total)"
                            )

                    if timed_out:
                        for fut in pending:
                            fut.cancel()
                        break
        else:
            for pt_idx, pt in enumerate(candidate_points):
                if deadline is not None and time.perf_counter() > deadline:
                    timed_out = True
                    break
                fim = self._compute_fim_at_point_no_prior(
                    experiment_index=0, input_values=list(pt)
                )
                candidate_fims[pt_idx] = fim
                if (pt_idx + 1) % max(1, n_candidates // 10) == 0:
                    elapsed = time.perf_counter() - t_start
                    frac_done = (pt_idx + 1) / n_candidates
                    est_total = elapsed / frac_done if frac_done > 0 else 0
                    self.logger.info(
                        f"  LHS FIM eval: {pt_idx + 1}/{n_candidates} "
                        f"({elapsed:.1f}s elapsed, ~{est_total:.1f}s total)"
                    )

        computed_pairs = [
            (pt, fim)
            for pt, fim in zip(candidate_points, candidate_fims)
            if fim is not None
        ]
        if not computed_pairs:
            timed_out = True
            computed_pairs = [(candidate_points[0], np.zeros((n_params, n_params)))]

        # If timeout stops FIM evaluation early, retain only scored candidates so
        # downstream combination scoring does not use missing entries.
        if len(computed_pairs) < n_candidates:
            self.logger.warning(
                "LHS candidate FIM evaluation reached time budget "
                f"({lhs_max_wall_clock_time}s). Scored {len(computed_pairs)}/{n_candidates} "
                "candidates; continuing with best available subset."
            )
            if len(computed_pairs) < n_exp:
                first_pt, first_fim = computed_pairs[0]
                computed_pairs.extend(
                    (first_pt, first_fim.copy())
                    for _ in range(n_exp - len(computed_pairs))
                )
            candidate_points = tuple(pt for pt, _ in computed_pairs)
            candidate_fims = [fim for _, fim in computed_pairs]
            n_candidates = len(candidate_points)

        t_after_fim = time.perf_counter()
        fim_eval_time = t_after_fim - t_after_sampling
        self.logger.info(f"LHS: completed FIM evaluations in {fim_eval_time:.1f}s.")

        # 4.  Enumerate combinations and score
        n_combinations = math.comb(n_candidates, n_exp)
        self.logger.info(
            f"LHS: scoring {n_combinations:,} combinations of {n_exp} experiments..."
        )
        if n_combinations > 100_000:
            self.logger.warning(
                f"LHS: {n_combinations:,} combinations to evaluate. "
                "This may be slow. Consider reducing ``lhs_n_samples``."
            )

        prior = self.prior_FIM.copy()
        _obj_option = self.objective_option
        is_maximize = _obj_option in self._MAXIMIZE_OBJECTIVES
        best_obj = -np.inf if is_maximize else np.inf
        best_combo = None
        _score_obj = DesignOfExperiments._evaluate_objective_for_option

        def _score_chunk(combo_chunk, deadline_ts):
            local_best_obj = -np.inf if is_maximize else np.inf
            local_best_combo = None
            local_timed_out = False
            for combo in combo_chunk:
                if deadline_ts is not None and time.perf_counter() > deadline_ts:
                    local_timed_out = True
                    break
                if n_exp == 2:
                    i, j = combo
                    fim_total = prior + candidate_fims[i] + candidate_fims[j]
                else:
                    fim_total = prior.copy()
                    for idx in combo:
                        fim_total = fim_total + candidate_fims[idx]
                obj_val = _score_obj(fim_total, _obj_option)
                if is_maximize:
                    if obj_val > local_best_obj:
                        local_best_obj = obj_val
                        local_best_combo = combo
                else:
                    if obj_val < local_best_obj:
                        local_best_obj = obj_val
                        local_best_combo = combo
            return local_best_obj, local_best_combo, local_timed_out

        use_parallel_combo = (
            lhs_combo_parallel
            and n_combinations >= lhs_combo_parallel_threshold
            and resolved_workers > 1
        )
        if lhs_combo_parallel and not use_parallel_combo:
            reasons = []
            if n_combinations < lhs_combo_parallel_threshold:
                reasons.append(
                    f"n_combinations={n_combinations} < "
                    f"lhs_combo_parallel_threshold={lhs_combo_parallel_threshold}"
                )
            if resolved_workers <= 1:
                reasons.append(f"resolved_workers={resolved_workers} <= 1")
            reason_txt = (
                "; ".join(reasons) if reasons else "parallel preconditions not met"
            )
            self.logger.warning(
                "LHS combination scoring requested with "
                "``lhs_combo_parallel=True``, but running serially: "
                f"{reason_txt}."
            )

        if use_parallel_combo:
            self.logger.info(
                f"LHS: using parallel combination scoring with {resolved_workers} workers "
                f"(chunk_size={lhs_combo_chunk_size})."
            )
            combo_iter = _combinations(range(n_candidates), n_exp)
            max_pending = max(1, 2 * resolved_workers)
            with _cf.ThreadPoolExecutor(max_workers=resolved_workers) as ex:
                pending = set()
                while True:
                    while len(pending) < max_pending:
                        if deadline is not None and time.perf_counter() > deadline:
                            timed_out = True
                            break
                        chunk = list(_islice(combo_iter, lhs_combo_chunk_size))
                        if not chunk:
                            break
                        pending.add(ex.submit(_score_chunk, chunk, deadline))
                    if not pending:
                        break

                    done, pending = _cf.wait(pending, return_when=_cf.FIRST_COMPLETED)
                    for fut in done:
                        local_obj, local_combo, local_timed_out = fut.result()
                        timed_out = timed_out or local_timed_out
                        if local_combo is None:
                            continue
                        if is_maximize:
                            if local_obj > best_obj:
                                best_obj = local_obj
                                best_combo = local_combo
                        else:
                            if local_obj < best_obj:
                                best_obj = local_obj
                                best_combo = local_combo

                    if deadline is not None and time.perf_counter() > deadline:
                        timed_out = True
                        for fut in pending:
                            fut.cancel()
                        break
        else:
            for combo in _combinations(range(n_candidates), n_exp):
                if deadline is not None and time.perf_counter() > deadline:
                    timed_out = True
                    break
                if n_exp == 2:
                    i, j = combo
                    fim_total = prior + candidate_fims[i] + candidate_fims[j]
                else:
                    fim_total = prior.copy()
                    for idx in combo:
                        fim_total = fim_total + candidate_fims[idx]
                obj_val = _score_obj(fim_total, _obj_option)
                if is_maximize:
                    if obj_val > best_obj:
                        best_obj = obj_val
                        best_combo = combo
                else:
                    if obj_val < best_obj:
                        best_obj = obj_val
                        best_combo = combo

        if timed_out:
            self.logger.warning(
                f"LHS combination scoring reached time budget "
                f"({lhs_max_wall_clock_time}s). Returning best-so-far."
            )

        t_after_combo = time.perf_counter()
        combo_time = t_after_combo - t_after_fim

        if best_combo is None:
            self.logger.warning(
                "LHS combination scoring ended before any combination was scored. "
                "Falling back to the first n_exp candidate points."
            )
            best_combo = tuple(range(n_exp))
            if n_exp == 2:
                i, j = best_combo
                fim_total = prior + candidate_fims[i] + candidate_fims[j]
            else:
                fim_total = prior.copy()
                for idx in best_combo:
                    fim_total = fim_total + candidate_fims[idx]
            best_obj = float(_score_obj(fim_total, _obj_option))

        best_obj_log10 = (
            float(np.log10(best_obj))
            if np.isfinite(best_obj) and best_obj > 0
            else None
        )
        self.logger.info(
            f"LHS: best {self.objective_option.value} objective = {best_obj:.6g}  "
            f"(combo scoring took {combo_time:.1f}s)."
        )

        best_initial_points = [list(candidate_points[i]) for i in best_combo]
        self.logger.info(
            f"LHS initial design: "
            + ", ".join(f"exp[{k}]={best_initial_points[k]}" for k in range(n_exp))
        )

        lhs_diagnostics = {
            "candidate_fim_mode": "thread" if use_parallel_fim else "serial",
            "combo_mode": "thread" if use_parallel_combo else "serial",
            "n_workers": resolved_workers,
            "n_candidates": n_candidates,
            "n_combinations": n_combinations,
            "elapsed_sampling_s": t_after_sampling - t_start,
            "elapsed_fim_eval_s": fim_eval_time,
            "elapsed_combo_scoring_s": combo_time,
            "elapsed_total_s": t_after_combo - t_start,
            "timed_out": timed_out,
            "time_budget_s": lhs_max_wall_clock_time,
            "best_obj": best_obj,
            "best_obj_log10": best_obj_log10,
        }
        return best_initial_points, lhs_diagnostics

    # Perform multi-experiment doe (sequential, or ``greedy`` approach)
    def run_multi_doe_sequential(self, N_exp=1):
        raise NotImplementedError("Multiple experiment optimization not yet supported.")

    # Perform multi-experiment doe (simultaneous, optimal approach)
    def run_multi_doe_simultaneous(self, N_exp=1):
        raise NotImplementedError("Multiple experiment optimization not yet supported.")

    # Compute FIM for the DoE object
    def compute_FIM(self, model=None, method="sequential"):
        """
        Computes the FIM for the experimental design that is
        initialized from the experiment`s ``get_labeled_model()``
        function.

        Parameters
        ----------
        model: model to compute FIM, default: None, (self.compute_FIM_model)
        method: string to specify which method should be used
                options are ``kaug`` and ``sequential``

        Returns
        -------
        computed FIM: 2D numpy array of the FIM
        """
        if model is None:
            self.compute_FIM_model = (
                self.experiment_list[0]
                .get_labeled_model(**self.get_labeled_model_args)
                .clone()
            )
            model = self.compute_FIM_model
        else:
            # TODO: Add safe naming when a model is passed by the user.
            # doe_block = pyo.Block()
            # doe_block_name = unique_component_name(model,
            #                                        "design_of_experiments_block")
            # model.add_component(doe_block_name, doe_block)
            # self.compute_FIM_model = model
            pass

        self.check_model_labels(model=model)

        # Set length values for the model features
        self.n_parameters = len(model.unknown_parameters)
        self.n_measurement_error = len(model.measurement_error)
        self.n_experiment_inputs = len(model.experiment_inputs)
        self.n_experiment_outputs = len(model.experiment_outputs)

        # Check FIM input, if it exists. Otherwise, set the prior_FIM attribute
        if self.prior_FIM is None:
            self.prior_FIM = np.zeros(
                (len(model.unknown_parameters), len(model.unknown_parameters))
            )
        else:
            self.check_model_FIM(FIM=self.prior_FIM)

        # TODO: Add a check to see if the model has an objective and deactivate it.
        #       This solve should only be a square solve without any obj function.

        if method == "sequential":
            self._sequential_FIM(model=model)
            self._computed_FIM = self.seq_FIM
        elif method == "kaug":
            self._kaug_FIM(model=model)
            self._computed_FIM = self.kaug_FIM
        else:
            raise ValueError(
                (
                    "The method provided, {}, must be either `sequential` "
                    "or `kaug`".format(method)
                )
            )

        return self._computed_FIM

    # Use a sequential method to get the FIM
    def _sequential_FIM(self, model=None):
        """
        Used to compute the FIM using a sequential approach,
        solving the model consecutively under each of the
        finite difference scenarios to build the sensitivity
        matrix to subsequently compute the FIM.

        """
        # Build a single model instance
        if model is None:
            self.compute_FIM_model = (
                self.experiment_list[0]
                .get_labeled_model(**self.get_labeled_model_args)
                .clone()
            )
            model = self.compute_FIM_model

        # Create suffix to keep track of parameter scenarios
        if hasattr(model, "parameter_scenarios"):
            model.del_component(model.parameter_scenarios)
        model.parameter_scenarios = pyo.Suffix(direction=pyo.Suffix.LOCAL)

        # Populate parameter scenarios, and scenario
        # inds based on finite difference scheme
        if self.fd_formula == FiniteDifferenceStep.central:
            model.parameter_scenarios.update(
                (2 * ind, k) for ind, k in enumerate(model.unknown_parameters.keys())
            )
            model.parameter_scenarios.update(
                (2 * ind + 1, k)
                for ind, k in enumerate(model.unknown_parameters.keys())
            )
            model.scenarios = range(len(model.unknown_parameters) * 2)
        elif self.fd_formula in [
            FiniteDifferenceStep.forward,
            FiniteDifferenceStep.backward,
        ]:
            model.parameter_scenarios.update(
                (ind + 1, k) for ind, k in enumerate(model.unknown_parameters.keys())
            )
            model.scenarios = range(len(model.unknown_parameters) + 1)
        else:
            raise DeveloperError(
                "Finite difference option not recognized. Please "
                "contact the developers as you should not see this error."
            )

        # Fix design variables
        for comp in model.experiment_inputs:
            comp.fix()

        measurement_vals = []
        # In a loop.....
        # Calculate measurement values for each scenario
        for s in model.scenarios:
            # Perturbation to be (1 + diff) * param_value
            if self.fd_formula == FiniteDifferenceStep.central:
                diff = self.step * (
                    (-1) ** s
                )  # Positive perturbation, even; negative, odd
            elif self.fd_formula == FiniteDifferenceStep.backward:
                diff = (
                    self.step * -1 * (s != 0)
                )  # Backward always negative perturbation; 0 at s = 0
            elif self.fd_formula == FiniteDifferenceStep.forward:
                diff = self.step * (s != 0)  # Forward always positive; 0 at s = 0

            # If we are doing forward/backward, no change for s=0
            skip_param_update = (
                self.fd_formula
                in [FiniteDifferenceStep.forward, FiniteDifferenceStep.backward]
            ) and (s == 0)
            if not skip_param_update:
                param = model.parameter_scenarios[s]
                # Update parameter values for the given finite difference scenario
                param.set_value(model.unknown_parameters[param] * (1 + diff))
            else:
                continue

            # Simulate the model
            try:
                res = self.solver.solve(model, tee=self.tee)
                pyo.assert_optimal_termination(res)
            except:
                # TODO: Make error message more verbose,
                #       (i.e., add unknown parameter values so the user
                #       can try to solve the model instance outside of
                #       the pyomo.DoE framework)
                raise RuntimeError(
                    "Model from experiment did not solve appropriately."
                    " Make sure the model is well-posed."
                )

            # Reset value of parameter to default value
            # before computing finite difference perturbation
            param.set_value(model.unknown_parameters[param])

            # Extract the measurement values for the scenario and append
            measurement_vals.append(
                [pyo.value(k) for k, v in model.experiment_outputs.items()]
            )

        # Use the measurement outputs to make the Q matrix
        measurement_vals_np = np.array(measurement_vals).T

        self.seq_jac = np.zeros(
            (
                len(model.experiment_outputs.items()),
                len(model.unknown_parameters.items()),
            )
        )

        # Counting variable for loop
        i = 0

        # Loop over parameter values and grab correct
        # columns for finite difference calculation

        for k, v in model.unknown_parameters.items():
            curr_step = v * self.step

            if self.fd_formula == FiniteDifferenceStep.central:
                col_1 = 2 * i
                col_2 = 2 * i + 1
                curr_step *= 2
            elif self.fd_formula == FiniteDifferenceStep.forward:
                col_1 = i
                col_2 = 0
            elif self.fd_formula == FiniteDifferenceStep.backward:
                col_1 = 0
                col_2 = i

            # If scale_nominal_param_value is active, scale
            # by nominal parameter value (v)
            scale_factor = (1.0 / curr_step) * self.scale_constant_value
            if self.scale_nominal_param_value:
                scale_factor *= v

            # Calculate the column of the sensitivity matrix
            self.seq_jac[:, i] = (
                measurement_vals_np[:, col_1] - measurement_vals_np[:, col_2]
            ) * scale_factor

            # Increment the count
            i += 1

        # TODO: As more complex measurement error schemes
        #       are put in place, this needs to change
        # Add independent (non-correlated) measurement
        # error for FIM calculation
        cov_y = np.zeros((len(model.measurement_error), len(model.measurement_error)))
        count = 0
        for k, v in model.measurement_error.items():
            cov_y[count, count] = 1 / v**2
            count += 1

        # Compute and record FIM
        self.seq_FIM = self.seq_jac.T @ cov_y @ self.seq_jac + self.prior_FIM

    # Use kaug to get FIM
    def _kaug_FIM(self, model=None):
        """
        Used to compute the FIM using kaug, a sensitivity-based
        approach that directly computes the FIM.

        Parameters
        ----------
        model: model to compute FIM, default: None, (self.compute_FIM_model)

        """
        # Remake compute_FIM_model if model is None.
        # compute_FIM_model needs to be the right version for function to work.
        if model is None:
            self.compute_FIM_model = (
                self.experiment_list[0]
                .get_labeled_model(**self.get_labeled_model_args)
                .clone()
            )
            model = self.compute_FIM_model

        # add zero (dummy/placeholder) objective function
        if not hasattr(model, "objective"):
            model.objective = pyo.Objective(expr=0, sense=pyo.minimize)

        # Fix design variables to make the problem square
        for comp in model.experiment_inputs:
            comp.fix()

        self.solver.solve(model, tee=self.tee)

        # Probe the solved model for dsdp results (sensitivities s.t. parameters)
        params_dict = {k.name: v for k, v in model.unknown_parameters.items()}
        params_names = list(params_dict.keys())

        dsdp_re, col = get_dsdp(model, params_names, params_dict, tee=self.tee)

        # analyze result
        dsdp_array = dsdp_re.toarray().T

        # store dsdp returned
        dsdp_extract = []
        # get right lines from results
        measurement_index = []

        # loop over measurement variables and their time points
        for k, v in model.experiment_outputs.items():
            name = k.name
            try:
                kaug_no = col.index(name)
                measurement_index.append(kaug_no)
                # get right line of dsdp
                dsdp_extract.append(dsdp_array[kaug_no])
            except:
                # k_aug does not provide value for fixed variables
                self.logger.debug("The variable is fixed:  %s", name)
                # produce the sensitivity for fixed variables
                zero_sens = np.zeros(len(params_names))
                # for fixed variables, the sensitivity are a zero vector
                dsdp_extract.append(zero_sens)

        # Extract and calculate sensitivity if scaled by constants or parameters.
        jac = [[] for k in params_names]

        for d in range(len(dsdp_extract)):
            for k, v in model.unknown_parameters.items():
                p = params_names.index(k.name)  # Index of parameter in np array
                # if scaled by parameter value or constant value
                sensi = dsdp_extract[d][p] * self.scale_constant_value
                if self.scale_nominal_param_value:
                    sensi *= v
                jac[p].append(sensi)

        # record kaug jacobian
        self.kaug_jac = np.array(jac).T

        # Compute FIM
        if self.prior_FIM is None:
            self.prior_FIM = np.zeros((len(params_names), len(params_names)))
        else:
            self.check_model_FIM(FIM=self.prior_FIM)

        # Constructing the Covariance of the measurements for the FIM calculation
        # The following assumes independent measurement error.
        cov_y = np.zeros((len(model.measurement_error), len(model.measurement_error)))
        count = 0
        for k, v in model.measurement_error.items():
            cov_y[count, count] = 1 / v**2
            count += 1

        # TODO: need to add a covariance matrix for measurements (sigma inverse)
        # i.e., cov_y = self.cov_y or model.cov_y
        # Still deciding where this would be best.

        self.kaug_FIM = self.kaug_jac.T @ cov_y @ self.kaug_jac + self.prior_FIM

    # Create the DoE model (with ``scenarios`` from finite differencing scheme)
    def create_doe_model(
        self, model=None, experiment_index=0, _for_multi_experiment=False
    ):
        """
        Add equations to compute sensitivities, FIM, and objective.
        Builds the DoE model. Adds the scenarios, the sensitivity matrix
        Q, the FIM, as well as the objective function to the model.

        The function alters the ``model`` input.

        In the single experiment case, ``model`` will be self.model. In the
        multi-experiment case, ``model`` will be one experiment to be enumerated.

        Parameters
        ----------
        model: model to add finite difference scenarios
        experiment_index: index of experiment in experiment_list to use for this model (default: 0)
        _for_multi_experiment: if True, skip creating L matrix and other objective-related
                               variables that will be created at the aggregated level (default: False)

        """
        if model is None:
            model = self.model
        else:
            # TODO: Add safe naming when a model is passed by the user.
            # doe_block = pyo.Block()
            # doe_block_name = unique_component_name(model,
            #                                        "design_of_experiments_block")
            # model.add_component(doe_block_name, doe_block)
            pass

        # Developer recommendation: use the Cholesky
        # decomposition for D-optimality. The explicit
        # formula is available for benchmarking purposes
        # and is NOT recommended.
        if (
            self.only_compute_fim_lower
            and self.objective_option == ObjectiveLib.determinant
            and not self.Cholesky_option
        ):
            raise ValueError(
                "Cannot compute determinant with explicit formula "
                "if only_compute_fim_lower is True."
            )

        # Generate scenarios for finite difference formulae
        self._generate_fd_scenario_blocks(
            model=model, experiment_index=experiment_index
        )

        # Set names for indexing sensitivity matrix (jacobian) and FIM
        scen_block_ind = min(
            [
                k.name.split(".").index("fd_scenario_blocks[0]")
                for k in model.fd_scenario_blocks[0].unknown_parameters.keys()
            ]
        )
        model.parameter_names = pyo.Set(
            initialize=[
                ".".join(k.name.split(".")[(scen_block_ind + 1) :])
                for k in model.fd_scenario_blocks[0].unknown_parameters.keys()
            ]
        )
        model.output_names = pyo.Set(
            initialize=[
                ".".join(k.name.split(".")[(scen_block_ind + 1) :])
                for k in model.fd_scenario_blocks[0].experiment_outputs.keys()
            ]
        )

        def identity_matrix(m, i, j):
            if i == j:
                return 1
            else:
                return 0

        ### Initialize the Jacobian if provided by the user

        # If the user provides an initial Jacobian, convert it to a dictionary
        if self.jac_initial is not None:
            dict_jac_initialize = {}
            for i, bu in enumerate(model.output_names):
                for j, un in enumerate(model.parameter_names):
                    # Jacobian is a numpy array, rows are experimental
                    # outputs, columns are unknown parameters
                    dict_jac_initialize[(bu, un)] = self.jac_initial[i][j]

        # Initialize the Jacobian matrix
        def initialize_jac(m, i, j):
            # If provided by the user, use the values now stored in the dictionary
            if self.jac_initial is not None:
                return dict_jac_initialize[(i, j)]
            # Otherwise initialize to 0.1 (which is an arbitrary non-zero value)
            else:
                raise DeveloperError(
                    "Jacobian being initialized when the jac_initial attribute "
                    "is None. Please contact the developers as you should not "
                    "see this error."
                )

        model.sensitivity_jacobian = pyo.Var(
            model.output_names, model.parameter_names, initialize=initialize_jac
        )

        # Initialize the FIM
        if self.fim_initial is not None:
            dict_fim_initialize = {
                (bu, un): self.fim_initial[i, j]
                for i, bu in enumerate(model.parameter_names)
                for j, un in enumerate(model.parameter_names)
            }
            if self.objective_option == ObjectiveLib.trace:
                fim_initial_inv = np.linalg.pinv(self.fim_initial)
                dict_fim_inv_initialize = {
                    (bu, un): fim_initial_inv[i, j]
                    for i, bu in enumerate(model.parameter_names)
                    for j, un in enumerate(model.parameter_names)
                }

        def initialize_fim(m, j, d):
            return dict_fim_initialize[(j, d)]

        def initialize_fim_inv(m, j, d):
            return dict_fim_inv_initialize[(j, d)]

        if self.fim_initial is not None:
            model.fim = pyo.Var(
                model.parameter_names, model.parameter_names, initialize=initialize_fim
            )
            if self.objective_option == ObjectiveLib.trace:
                model.fim_inv = pyo.Var(
                    model.parameter_names,
                    model.parameter_names,
                    initialize=initialize_fim_inv,
                )
        else:
            model.fim = pyo.Var(
                model.parameter_names, model.parameter_names, initialize=identity_matrix
            )
            if self.objective_option == ObjectiveLib.trace:
                model.fim_inv = pyo.Var(
                    model.parameter_names,
                    model.parameter_names,
                    initialize=identity_matrix,
                )

        # To-Do: Look into this functionality.....
        # if cholesky, define L elements as variables
        if (
            not _for_multi_experiment
            and self.Cholesky_option
            and self.objective_option in (ObjectiveLib.determinant, ObjectiveLib.trace)
        ):
            model.L = pyo.Var(
                model.parameter_names, model.parameter_names, initialize=identity_matrix
            )

            # If trace objective, also need L inverse
            if self.objective_option == ObjectiveLib.trace:
                model.L_inv = pyo.Var(
                    model.parameter_names,
                    model.parameter_names,
                    initialize=identity_matrix,
                )

            # loop over parameter name
            for i, c in enumerate(model.parameter_names):
                for j, d in enumerate(model.parameter_names):
                    # fix the 0 half of L matrix to be 0.0
                    if i < j:
                        model.L[c, d].fix(0.0)
                        if self.objective_option == ObjectiveLib.trace:
                            model.L_inv[c, d].fix(0.0)
                    # Give LB to the diagonal entries
                    if self.L_diagonal_lower_bound:
                        if c == d:
                            model.L[c, d].setlb(self.L_diagonal_lower_bound)
                            if self.objective_option == ObjectiveLib.trace:
                                model.L_inv[c, d].setlb(self.L_diagonal_lower_bound)

        # jacobian rule
        def jacobian_rule(m, n, p):
            """
            m: Pyomo model
            n: experimental output
            p: unknown parameter
            """
            fd_step_mult = 1
            cuid = pyo.ComponentUID(n)
            param_ind = m.parameter_names.data().index(p)

            # Different FD schemes lead to different scenarios for the computation
            if self.fd_formula == FiniteDifferenceStep.central:
                s1 = param_ind * 2
                s2 = param_ind * 2 + 1
                fd_step_mult = 2
            elif self.fd_formula == FiniteDifferenceStep.forward:
                s1 = param_ind + 1
                s2 = 0
            elif self.fd_formula == FiniteDifferenceStep.backward:
                s1 = 0
                s2 = param_ind + 1

            var_up = cuid.find_component_on(m.fd_scenario_blocks[s1])
            var_lo = cuid.find_component_on(m.fd_scenario_blocks[s2])

            param = m.parameter_scenarios[max(s1, s2)]
            param_loc = pyo.ComponentUID(param).find_component_on(
                m.fd_scenario_blocks[0]
            )
            param_val = m.fd_scenario_blocks[0].unknown_parameters[param_loc]
            param_diff = param_val * fd_step_mult * self.step

            if self.scale_nominal_param_value:
                return (
                    m.sensitivity_jacobian[n, p]
                    == (var_up - var_lo)
                    / param_diff
                    * param_val
                    * self.scale_constant_value
                )
            else:
                return (
                    m.sensitivity_jacobian[n, p]
                    == (var_up - var_lo) / param_diff * self.scale_constant_value
                )

        # A constraint to calculate elements in Hessian matrix
        # transfer prior FIM to be Expressions
        fim_initial_dict = {
            (bu, un): self.prior_FIM[i][j]
            for i, bu in enumerate(model.parameter_names)
            for j, un in enumerate(model.parameter_names)
        }

        def read_prior(m, i, j):
            return fim_initial_dict[(i, j)]

        model.prior_FIM = pyo.Expression(
            model.parameter_names, model.parameter_names, rule=read_prior
        )

        # Off-diagonal elements are symmetric, so only
        # half of the off-diagonal elements need to be
        # specified.
        def fim_rule(m, p, q):
            """
            m: Pyomo model
            p: unknown parameter
            q: unknown parameter
            """
            p_ind = list(m.parameter_names).index(p)
            q_ind = list(m.parameter_names).index(q)

            # If the row is less than the column, skip the constraint
            # This logic is consistent with making the FIM a lower
            # triangular matrix (as is done later in this function)
            if p_ind < q_ind:
                if self.only_compute_fim_lower:
                    return pyo.Constraint.Skip
                else:
                    return m.fim[p, q] == m.fim[q, p]
            else:
                # For multi-experiment optimization, prior_FIM is added to the
                # aggregated total_fim, not to each individual experiment's FIM
                if _for_multi_experiment:
                    return m.fim[p, q] == sum(
                        1
                        / m.fd_scenario_blocks[0].measurement_error[
                            pyo.ComponentUID(n).find_component_on(
                                m.fd_scenario_blocks[0]
                            )
                        ]
                        ** 2
                        * m.sensitivity_jacobian[n, p]
                        * m.sensitivity_jacobian[n, q]
                        for n in m.output_names
                    )
                else:
                    return (
                        m.fim[p, q]
                        == sum(
                            1
                            / m.fd_scenario_blocks[0].measurement_error[
                                pyo.ComponentUID(n).find_component_on(
                                    m.fd_scenario_blocks[0]
                                )
                            ]
                            ** 2
                            * m.sensitivity_jacobian[n, p]
                            * m.sensitivity_jacobian[n, q]
                            for n in m.output_names
                        )
                        + m.prior_FIM[p, q]
                    )

        model.jacobian_constraint = pyo.Constraint(
            model.output_names, model.parameter_names, rule=jacobian_rule
        )
        model.fim_constraint = pyo.Constraint(
            model.parameter_names, model.parameter_names, rule=fim_rule
        )

        if self.only_compute_fim_lower:
            # Fix the upper half of the FIM matrix elements to be 0.0.
            # This eliminates extra variables and ensures the expected number of
            # degrees of freedom in the optimization problem.
            for ind_p, p in enumerate(model.parameter_names):
                for ind_q, q in enumerate(model.parameter_names):
                    if ind_p < ind_q:
                        model.fim[p, q].fix(0.0)
                        if self.objective_option == ObjectiveLib.trace:
                            model.fim_inv[p, q].fix(0.0)

    # Create scenario block structure
    def _generate_fd_scenario_blocks(self, model=None, experiment_index=0):
        """
        Generates the modeling blocks corresponding to the scenarios for
        the finite differencing scheme to compute the sensitivity jacobian
        to compute the FIM.

        The function alters the ``model`` input.

        In the single experiment case, ``model`` will be self.model. In the
        multi-experiment case, ``model`` will be one experiment to be enumerated.

        Parameters
        ----------
        model: model to add finite difference scenarios
        experiment_index: index of experiment in experiment_list to use for this model (default: 0)
        """
        # If model is none, assume it is self.model
        if model is None:
            model = self.model

        # Generate initial scenario to populate unknown parameter values
        model.base_model = (
            self.experiment_list[experiment_index]
            .get_labeled_model(**self.get_labeled_model_args)
            .clone()
        )

        # Check the model that labels are correct
        self.check_model_labels(model=model.base_model)

        # Gather lengths of label structures for later use in the model build process
        self.n_parameters = len(model.base_model.unknown_parameters)
        self.n_measurement_error = len(model.base_model.measurement_error)
        self.n_experiment_inputs = len(model.base_model.experiment_inputs)
        self.n_experiment_outputs = len(model.base_model.experiment_outputs)

        if self.n_measurement_error != self.n_experiment_outputs:
            raise ValueError(
                "Number of experiment outputs, {}, and length of measurement error, "
                "{}, do not match. Please check model labeling.".format(
                    self.n_experiment_outputs, self.n_measurement_error
                )
            )

        self.logger.info("Experiment output and measurement error lengths match.")

        # Check that the user input FIM and Jacobian are the correct dimension
        if self.prior_FIM is not None:
            self.check_model_FIM(FIM=self.prior_FIM)
        else:
            self.prior_FIM = np.zeros((self.n_parameters, self.n_parameters))
        if self.fim_initial is not None:
            self.check_model_FIM(FIM=self.fim_initial)
        else:
            self.fim_initial = np.eye(self.n_parameters) + self.prior_FIM
        if self.jac_initial is not None:
            self.check_model_jac(self.jac_initial)
        else:
            self.jac_initial = np.eye(self.n_experiment_outputs, self.n_parameters)

        # Make a new Suffix to hold which scenarios
        # are associated with parameters
        model.parameter_scenarios = pyo.Suffix(direction=pyo.Suffix.LOCAL)

        # Populate parameter scenarios, and scenario
        # inds based on finite difference scheme
        if self.fd_formula == FiniteDifferenceStep.central:
            model.parameter_scenarios.update(
                (2 * ind, k)
                for ind, k in enumerate(model.base_model.unknown_parameters.keys())
            )
            model.parameter_scenarios.update(
                (2 * ind + 1, k)
                for ind, k in enumerate(model.base_model.unknown_parameters.keys())
            )
            model.scenarios = range(len(model.base_model.unknown_parameters) * 2)
        elif self.fd_formula in [
            FiniteDifferenceStep.forward,
            FiniteDifferenceStep.backward,
        ]:
            model.parameter_scenarios.update(
                (ind + 1, k)
                for ind, k in enumerate(model.base_model.unknown_parameters.keys())
            )
            model.scenarios = range(len(model.base_model.unknown_parameters) + 1)
        else:
            raise DeveloperError(
                "Finite difference option not recognized. Please contact "
                "the developers as you should not see this error."
            )

        # Run base model to get initialized model and check model function
        for comp in model.base_model.experiment_inputs:
            comp.fix()

        try:
            res = self.solver.solve(model.base_model, tee=self.tee)
            assert res.solver.termination_condition == "optimal"
            self.logger.info("Model from experiment solved.")
        except:
            raise RuntimeError(
                "Model from experiment did not solve appropriately. "
                "Make sure the model is well-posed."
            )

        for comp in model.base_model.experiment_inputs:
            comp.unfix()

        # Generate blocks for finite difference scenarios
        def build_block_scenarios(b, s):
            # Generate model for the finite difference scenario
            # Get the parent block that contains base_model
            parent_block = b.parent_block()
            b.transfer_attributes_from(parent_block.base_model.clone())

            # Forward/Backward difference have a stationary
            # case (s == 0), no parameter to perturb
            if self.fd_formula in [
                FiniteDifferenceStep.forward,
                FiniteDifferenceStep.backward,
            ]:
                if s == 0:
                    return

            param = parent_block.parameter_scenarios[s]

            # Perturbation to be (1 + diff) * param_value
            if self.fd_formula == FiniteDifferenceStep.central:
                diff = self.step * (
                    (-1) ** s
                )  # Positive perturbation, even; negative, odd
            elif self.fd_formula == FiniteDifferenceStep.backward:
                diff = self.step * -1  # Backward always negative perturbation
            elif self.fd_formula == FiniteDifferenceStep.forward:
                diff = self.step  # Forward always positive
            else:
                # TODO: add an error message for this as not being implemented yet
                diff = 0
                pass

            # Update parameter values for the given finite difference scenario
            pyo.ComponentUID(param, context=parent_block.base_model).find_component_on(
                b
            ).set_value(parent_block.base_model.unknown_parameters[param] * (1 + diff))

            # Fix experiment inputs before solve (enforce square solve)
            for comp in b.experiment_inputs:
                comp.fix()

            res = self.solver.solve(b, tee=self.tee)

            # Unfix experiment inputs after square solve
            for comp in b.experiment_inputs:
                comp.unfix()

        model.fd_scenario_blocks = pyo.Block(
            model.scenarios, rule=build_block_scenarios
        )

        # TODO: this might have to change if experiment inputs have
        #       a different value in the Suffix (currently it is the CUID)
        design_vars = [
            k for k, v in model.fd_scenario_blocks[0].experiment_inputs.items()
        ]

        # Add constraints to equate block design with global design:
        for ind, d in enumerate(design_vars):
            con_name = "global_design_eq_con_" + str(ind)

            # Constraint rule for global design constraints
            def global_design_fixing(m, s):
                if s == 0:
                    return pyo.Constraint.Skip
                block_design_var = pyo.ComponentUID(
                    d, context=m.fd_scenario_blocks[0]
                ).find_component_on(m.fd_scenario_blocks[s])
                return d == block_design_var

            model.add_component(
                con_name, pyo.Constraint(model.scenarios, rule=global_design_fixing)
            )

        # Clean up the base model used to generate the scenarios
        model.del_component(model.base_model)

        # TODO: consider this logic? Multi-block systems need something more fancy
        self._built_scenarios = True

    # Create objective function
    def create_objective_function(self, model=None):
        """
        Generates the objective function as an expression and as a
        Pyomo Objective object

        The function alters the ``model`` input.

        In the single experiment case, ``model`` will be self.model. In the
        multi-experiment case, ``model`` will be one experiment to be enumerated.

        Parameters
        ----------
        model: model to add finite difference scenarios
        """
        if model is None:
            model = self.model

        if self.objective_option not in [
            ObjectiveLib.determinant,
            ObjectiveLib.trace,
            ObjectiveLib.pseudo_trace,
            ObjectiveLib.zero,
        ]:
            raise DeveloperError(
                "Objective option not recognized. Please contact the "
                "developers as you should not see this error."
            )

        if not hasattr(model, "fim"):
            raise RuntimeError(
                "Model provided does not have variable `fim`. Please make "
                "sure the model is built properly before creating the objective."
            )

        small_number = 1e-10

        # Make objective block for constraints connected to objective
        model.obj_cons = pyo.Block()

        # Assemble the FIM matrix. This is helpful for initialization!
        # collect current FIM values in row-major order
        fim_vals = [
            model.fim[bu, un].value
            for bu in model.parameter_names
            for un in model.parameter_names
        ]
        fim = np.array(fim_vals).reshape(
            len(model.parameter_names), len(model.parameter_names)
        )

        ### Initialize the Cholesky decomposition matrix
        if self.Cholesky_option and self.objective_option in (
            ObjectiveLib.determinant,
            ObjectiveLib.trace,
        ):
            # Calculate the eigenvalues of the FIM matrix
            eig = np.linalg.eigvals(fim)

            # If the smallest eigenvalue is (practically) negative,
            # add a diagonal matrix to make it positive definite
            if min(eig) < small_number:
                fim = fim + np.eye(len(model.parameter_names)) * (
                    small_number - min(eig)
                )

            # Compute the Cholesky decomposition of the FIM matrix
            L = np.linalg.cholesky(fim)

            # Initialize the Cholesky matrix
            for i, c in enumerate(model.parameter_names):
                for j, d in enumerate(model.parameter_names):
                    model.L[c, d].value = L[i, j]

            # Compute L inverse for trace objective and initialize
            if self.objective_option == ObjectiveLib.trace:
                L_inv = np.linalg.inv(L)
                for i, c in enumerate(model.parameter_names):
                    for j, d in enumerate(model.parameter_names):
                        model.L_inv[c, d].value = L_inv[i, j]

        if self.objective_option == ObjectiveLib.trace and not self.Cholesky_option:
            raise ValueError(
                "objective_option='trace' currently only implemented with ``_Cholesky option=True``."
            )

        # If trace objective, need L inverse constraints
        if self.Cholesky_option and self.objective_option == ObjectiveLib.trace:

            def cholesky_inv_imp(b, c, d):
                """
                Calculate Cholesky L inverse matrix using algebraic constraints
                """
                # If the row is greater than or equal to the column, we are in the
                # lower triangle region of the L_inv matrix.
                # This region is where our equations are well-defined.
                m = b.model()
                if list(m.parameter_names).index(c) >= list(m.parameter_names).index(d):
                    return m.fim_inv[c, d] == sum(
                        m.L_inv[m.parameter_names.at(k + 1), c]
                        * m.L_inv[m.parameter_names.at(k + 1), d]
                        for k in range(
                            list(m.parameter_names).index(c), len(m.parameter_names)
                        )
                    )
                else:
                    # This is the empty half of L_inv above the diagonal
                    return pyo.Constraint.Skip

            # If trace objective, need L * L^-1 = Identity matrix constraints
            def cholesky_LLinv_imp(b, c, d):
                """
                Calculate Cholesky L * L inverse matrix using algebraic constraints
                """
                # If the row is greater than or equal to the column, we are in the
                # lower triangle region of the L and L_inv matrices.
                # This region is where our equations are well-defined.
                m = b.model()
                param_list = list(m.parameter_names)
                idx_c = param_list.index(c)
                idx_d = param_list.index(d)
                # Do not need to calculate upper triangle entries
                if idx_c < idx_d:
                    return pyo.Constraint.Skip

                target_value = 1 if idx_c == idx_d else 0
                return (
                    sum(
                        m.L[c, m.parameter_names.at(k + 1)]
                        * m.L_inv[m.parameter_names.at(k + 1), d]
                        for k in range(len(m.parameter_names))
                    )
                    == target_value
                )

            # To improve round off error in Cholesky decomposition
            if self.improve_cholesky_roundoff_error:

                def cholesky_fim_diag(b, c, d):
                    """
                    M[c,c] >= L[c,d]^2 to improve round off error
                    """
                    m = b.model()
                    return m.fim[c, c] >= m.L[c, d] ** 2

                def cholesky_fim_inv_diag(b, c, d):
                    """
                    M_inv[c,c] >= L_inv[c,d]^2 to improve round off error
                    """
                    m = b.model()
                    return m.fim_inv[c, c] >= m.L_inv[c, d] ** 2

            def cov_trace_calc(b):
                """
                Calculate trace of covariance matrix (inverse of FIM).
                Can scale each element with 1000 for performance
                """
                m = b.model()
                return m.cov_trace == sum(m.fim_inv[j, j] for j in m.parameter_names)

        def trace_calc(b):
            """
            Calculate FIM elements. Can scale each element with 1000 for performance
            """
            m = b.model()
            return m.fim_trace == sum(m.fim[j, j] for j in m.parameter_names)

        def determinant_general(b):
            r"""Calculate determinant. Can be applied to FIM of any size.
            det(A) = \sum_{\sigma in \S_n} (sgn(\sigma) * \Prod_{i=1}^n a_{i,\sigma_i})
            Use permutation() to get permutations, sgn() to get signature
            """
            m = b.model()
            r_list = list(range(len(m.parameter_names)))
            # get all permutations
            object_p = permutations(r_list)
            list_p = list(object_p)

            # generate a name_order to iterate \sigma_i
            # NOTE: Not used in calculation. Delete?
            det_perm = 0
            for i in range(len(list_p)):
                name_order = []
                x_order = list_p[i]
                # sigma_i is the value in the i-th
                # position after the reordering \sigma
                for x in range(len(x_order)):
                    for y, element in enumerate(m.parameter_names):
                        if x_order[x] == y:
                            name_order.append(element)
            # det(A) = sum_{\sigma \in \S_n} (sgn(\sigma) *
            #          \Prod_{i=1}^n a_{i,\sigma_i})
            det_perm = sum(
                self._sgn(list_p[d])
                * math.prod(
                    m.fim[m.parameter_names.at(val + 1), m.parameter_names.at(ind + 1)]
                    for ind, val in enumerate(list_p[d])
                )
                for d in range(len(list_p))
            )
            return m.determinant == det_perm

        if self.Cholesky_option and self.objective_option == ObjectiveLib.determinant:
            model.obj_cons.cholesky_cons = pyo.Constraint(
                model.parameter_names,
                model.parameter_names,
                rule=self._make_cholesky_rule(
                    model.fim, model.L, model.parameter_names
                ),
            )
            model.objective = pyo.Objective(
                expr=2 * sum(pyo.log10(model.L[j, j]) for j in model.parameter_names),
                sense=pyo.maximize,
            )

        elif self.objective_option == ObjectiveLib.determinant:
            # if not Cholesky but determinant, calculating
            # det and evaluate the OBJ with det
            model.determinant = pyo.Var(
                initialize=np.linalg.det(fim), bounds=(small_number, None)
            )
            model.obj_cons.determinant_rule = pyo.Constraint(rule=determinant_general)
            model.objective = pyo.Objective(
                expr=pyo.log10(model.determinant + 1e-6), sense=pyo.maximize
            )

        elif self.objective_option == ObjectiveLib.trace:
            # if Cholesky and trace, calculating
            # the OBJ with trace
            model.cov_trace = pyo.Var(
                initialize=np.trace(np.linalg.inv(fim)), bounds=(small_number, None)
            )
            model.obj_cons.cholesky_cons = pyo.Constraint(
                model.parameter_names,
                model.parameter_names,
                rule=self._make_cholesky_rule(
                    model.fim, model.L, model.parameter_names
                ),
            )
            model.obj_cons.cholesky_inv_cons = pyo.Constraint(
                model.parameter_names, model.parameter_names, rule=cholesky_inv_imp
            )
            model.obj_cons.cholesky_LLinv_cons = pyo.Constraint(
                model.parameter_names, model.parameter_names, rule=cholesky_LLinv_imp
            )
            if self.improve_cholesky_roundoff_error:
                model.obj_cons.cholesky_fim_diag_cons = pyo.Constraint(
                    model.parameter_names, model.parameter_names, rule=cholesky_fim_diag
                )
                model.obj_cons.cholesky_fim_inv_diag_cons = pyo.Constraint(
                    model.parameter_names,
                    model.parameter_names,
                    rule=cholesky_fim_inv_diag,
                )
            model.obj_cons.cov_trace_rule = pyo.Constraint(rule=cov_trace_calc)
            model.objective = pyo.Objective(expr=model.cov_trace, sense=pyo.minimize)

        elif self.objective_option == ObjectiveLib.pseudo_trace:
            # if not determinant or Cholesky, calculating
            # the OBJ with trace
            model.fim_trace = pyo.Var(
                initialize=np.trace(fim), bounds=(small_number, None)
            )
            model.obj_cons.trace_rule = pyo.Constraint(rule=trace_calc)
            model.objective = pyo.Objective(
                expr=pyo.log10(model.fim_trace), sense=pyo.maximize
            )

        # TODO: Add warning (should be unreachable) if the user calls
        #       the grey box objectives with the standard model
        elif self.objective_option == ObjectiveLib.zero:
            # add dummy objective function
            model.objective = pyo.Objective(expr=0)

    def create_multi_experiment_objective_function(self, model):
        """
        Create objective for multi-experiment optimization.

        For each scenario s:
          1. Creates total_fim[s] = sum of exp_blocks[k].fim + prior_FIM
          2. Creates Cholesky/determinant/trace variables and constraints per scenario
          3. Creates single top-level objective summing across scenarios

        Parameters
        ----------
        model: model with param_scenario_blocks structure
        """
        # Validate objective option for multi-experiment
        if self.objective_option not in [
            ObjectiveLib.determinant,
            ObjectiveLib.trace,
            ObjectiveLib.pseudo_trace,
            ObjectiveLib.zero,
        ]:
            raise DeveloperError(
                "Objective option not recognized for multi-experiment optimization. "
                "Please contact the developers as you should not see this error."
            )

        # Validate trace objective requires Cholesky option
        if self.objective_option == ObjectiveLib.trace and not self.Cholesky_option:
            raise ValueError(
                "objective_option='trace' currently only implemented with "
                "``_Cholesky_option=True``."
            )

        small_number = 1e-10
        n_scenarios = len(model.param_scenario_blocks)

        # Get weights from instance attribute (set in optimize_experiments) and
        # default to uniform weights if not provided
        # retrieve weights; default to uniform tuple of appropriate length
        default_weights = tuple([1.0 / n_scenarios] * n_scenarios)
        scenario_weights = getattr(self, 'scenario_weights', default_weights)

        # Get parameter names from first experiment (same across all)
        parameter_names = model.param_scenario_blocks[0].exp_blocks[0].parameter_names
        n_exp = len(model.param_scenario_blocks[0].exp_blocks)

        # For each scenario: create aggregated FIM and constraints
        for s in range(n_scenarios):
            scenario = model.param_scenario_blocks[s]

            # 1. Create aggregated FIM variable for each scenario:
            # total_fim = sum of all exp FIMs + prior_FIM
            scenario.total_fim = pyo.Var(parameter_names, parameter_names)

            # 2. Constraint: total_fim[p,q] = sum_k (exp_blocks[k].fim[p,q])
            # + prior_FIM[p,q]
            def total_fim_rule(b, p, q):
                p_idx = list(parameter_names).index(p)
                q_idx = list(parameter_names).index(q)

                # When only_compute_fim_lower=True, only compute lower triangle
                # Upper triangle elements will be handled through symmetry
                if self.only_compute_fim_lower and p_idx < q_idx:
                    return pyo.Constraint.Skip

                return b.total_fim[p, q] == (
                    sum(b.exp_blocks[k].fim[p, q] for k in range(n_exp))
                    + self.prior_FIM[p_idx, q_idx]
                )

            scenario.total_fim_cons = pyo.Constraint(
                parameter_names, parameter_names, rule=total_fim_rule
            )

            # 3. Fix upper triangle elements to 0 if only_compute_fim_lower=True
            # Initialize lower triangle from sum of individual FIMs
            for i, p in enumerate(parameter_names):
                for j, q in enumerate(parameter_names):
                    if self.only_compute_fim_lower and i < j:
                        # Fix upper triangle to 0
                        scenario.total_fim[p, q].fix(0.0)
                    else:
                        # Initialize lower triangle and diagonal
                        fim_sum = sum(
                            pyo.value(scenario.exp_blocks[k].fim[p, q]) or 0
                            for k in range(n_exp)
                        )
                        scenario.total_fim[p, q].value = fim_sum + self.prior_FIM[i, j]

            # 4. Create obj_cons block to hold objective-related constraints
            # (similar to single-experiment case in create_objective_function)
            scenario.obj_cons = pyo.Block()
            # 5. Create variables and constraints (initialization will happen after square solve)
            if (
                self.Cholesky_option
                and self.objective_option == ObjectiveLib.determinant
            ):
                # Add lower triangular Cholesky variables per scenario
                scenario.obj_cons.L = pyo.Var(parameter_names, parameter_names)

                # Fix upper triangle to 0 and set lower bound on diagonal
                for i, p in enumerate(parameter_names):
                    for j, q in enumerate(parameter_names):
                        # Fix upper triangle to 0
                        if i < j:
                            scenario.obj_cons.L[p, q].fix(0.0)
                        # Lower bound on diagonal
                        elif i == j and self.L_diagonal_lower_bound:
                            scenario.obj_cons.L[p, q].setlb(self.L_diagonal_lower_bound)

                # reuse shared helper to create the constraint rule
                cholesky_rule = self._make_cholesky_rule(
                    scenario.total_fim, scenario.obj_cons.L, parameter_names
                )
                scenario.obj_cons.cholesky_cons = pyo.Constraint(
                    parameter_names, parameter_names, rule=cholesky_rule
                )

            elif self.Cholesky_option and self.objective_option == ObjectiveLib.trace:
                # Add lower triangular Cholesky variables per scenario
                scenario.obj_cons.L = pyo.Var(parameter_names, parameter_names)
                scenario.obj_cons.L_inv = pyo.Var(parameter_names, parameter_names)
                scenario.obj_cons.fim_inv = pyo.Var(parameter_names, parameter_names)
                scenario.obj_cons.cov_trace = pyo.Var(bounds=(small_number, None))

                # Fix upper triangle of L and L_inv to 0 and set lower bound on diagonal
                for i, p in enumerate(parameter_names):
                    for j, q in enumerate(parameter_names):
                        # Fix upper triangle to 0
                        if i < j:
                            scenario.obj_cons.L[p, q].fix(0.0)
                            scenario.obj_cons.L_inv[p, q].fix(0.0)
                        # Lower bound on diagonal
                        elif i == j and self.L_diagonal_lower_bound:
                            scenario.obj_cons.L[p, q].setlb(self.L_diagonal_lower_bound)

                # reuse shared helper to create the constraint rule
                cholesky_rule = self._make_cholesky_rule(
                    scenario.total_fim, scenario.obj_cons.L, parameter_names
                )
                scenario.obj_cons.cholesky_cons = pyo.Constraint(
                    parameter_names, parameter_names, rule=cholesky_rule
                )

                # reuse helpers for the inverse and identity rules instead of
                # re-implementing the logic in-place
                cholesky_inv_rule = self._make_cholesky_inv_rule(
                    scenario.obj_cons.fim_inv, scenario.obj_cons.L_inv, parameter_names
                )

                cholesky_LLinv_rule = self._make_cholesky_LLinv_rule(
                    scenario.obj_cons.L, scenario.obj_cons.L_inv, parameter_names
                )

                # Covariance trace calculation
                def cov_trace_rule(b):
                    return b.cov_trace == sum(b.fim_inv[j, j] for j in parameter_names)

                # Add all constraints
                scenario.obj_cons.cholesky_inv_cons = pyo.Constraint(
                    parameter_names, parameter_names, rule=cholesky_inv_rule
                )
                scenario.obj_cons.cholesky_LLinv_cons = pyo.Constraint(
                    parameter_names, parameter_names, rule=cholesky_LLinv_rule
                )
                scenario.obj_cons.cov_trace_cons = pyo.Constraint(rule=cov_trace_rule)

                # Optional: improve Cholesky roundoff error
                if self.improve_cholesky_roundoff_error:

                    def cholesky_fim_diag(b, p, q):
                        return scenario.total_fim[p, p] >= b.L[p, q] ** 2

                    def cholesky_fim_inv_diag(b, p, q):
                        return b.fim_inv[p, p] >= b.L_inv[p, q] ** 2

                    scenario.obj_cons.cholesky_fim_diag_cons = pyo.Constraint(
                        parameter_names, parameter_names, rule=cholesky_fim_diag
                    )
                    scenario.obj_cons.cholesky_fim_inv_diag_cons = pyo.Constraint(
                        parameter_names, parameter_names, rule=cholesky_fim_inv_diag
                    )

            elif self.objective_option == ObjectiveLib.determinant:
                # Non-Cholesky determinant: create determinant var per scenario
                scenario.obj_cons.determinant = pyo.Var(bounds=(small_number, None))

                # Determinant constraint (explicit formula)
                def determinant_general(b):
                    r_list = list(range(len(parameter_names)))
                    object_p = permutations(r_list)
                    list_p = list(object_p)

                    det_perm = sum(
                        self._sgn(list_p[d])
                        * math.prod(
                            scenario.total_fim[
                                parameter_names.at(val + 1), parameter_names.at(ind + 1)
                            ]
                            for ind, val in enumerate(list_p[d])
                        )
                        for d in range(len(list_p))
                    )
                    return b.determinant == det_perm

                scenario.obj_cons.determinant_cons = pyo.Constraint(
                    rule=determinant_general
                )

            elif self.objective_option == ObjectiveLib.pseudo_trace:
                # Pseudo trace objective (Trace of FIM)
                scenario.obj_cons.pseudo_trace = pyo.Var(bounds=(small_number, None))

                # Pseudo trace constraint
                def pseudo_trace_rule(b):
                    return b.pseudo_trace == sum(
                        scenario.total_fim[j, j] for j in parameter_names
                    )

                scenario.obj_cons.pseudo_trace_cons = pyo.Constraint(
                    rule=pseudo_trace_rule
                )

        # 5. Create single top-level objective summing across scenarios
        if self.Cholesky_option and self.objective_option == ObjectiveLib.determinant:
            model.objective = pyo.Objective(
                expr=sum(
                    scenario_weights[s]
                    * 2
                    * sum(
                        pyo.log10(model.param_scenario_blocks[s].obj_cons.L[j, j])
                        for j in parameter_names
                    )
                    for s in range(n_scenarios)
                ),
                sense=pyo.maximize,
            )

        elif self.objective_option == ObjectiveLib.determinant:
            model.objective = pyo.Objective(
                expr=sum(
                    scenario_weights[s]
                    * pyo.log10(
                        model.param_scenario_blocks[s].obj_cons.determinant
                        + _SMALL_TOLERANCE_DEFINITENESS  # to avoid log(0)
                    )
                    for s in range(n_scenarios)
                ),
                sense=pyo.maximize,
            )

        elif self.Cholesky_option and self.objective_option == ObjectiveLib.trace:
            model.objective = pyo.Objective(
                expr=sum(
                    scenario_weights[s]
                    * model.param_scenario_blocks[s].obj_cons.cov_trace
                    for s in range(n_scenarios)
                ),
                sense=pyo.minimize,
            )

        elif self.objective_option == ObjectiveLib.pseudo_trace:
            model.objective = pyo.Objective(
                expr=sum(
                    scenario_weights[s]
                    * pyo.log10(
                        model.param_scenario_blocks[s].obj_cons.pseudo_trace
                        + _SMALL_TOLERANCE_DEFINITENESS  # to avoid log(0)
                    )
                    for s in range(n_scenarios)
                ),
                sense=pyo.maximize,
            )

        elif self.objective_option == ObjectiveLib.zero:
            # Dummy objective
            model.objective = pyo.Objective(expr=0)

    def create_grey_box_objective_function(self, model=None):
        # Add external grey box block to a block named ``obj_cons`` to
        # reuse material for initializing the objective-free square model
        if model is None:
            model = model = self.model

        # TODO: Make this naming convention robust
        model.obj_cons = pyo.Block()

        # Create FIM External Grey Box object
        grey_box_FIM = FIMExternalGreyBox(
            doe_object=self,
            objective_option=self.objective_option,
            logger_level=self.logger.getEffectiveLevel(),
        )

        # Attach External Grey Box Model
        # to the model as an External
        # Grey Box Block
        model.obj_cons.egb_fim_block = ExternalGreyBoxBlock(external_model=grey_box_FIM)

        # Adding constraints to for all grey box input values to equate to fim values
        def FIM_egb_cons(m, p1, p2):
            """

            m: Pyomo model
            p1: parameter 1
            p2: parameter 2

            """
            # Using upper triangular FIM to
            # make numerics better.
            if list(model.parameter_names).index(p1) >= list(
                model.parameter_names
            ).index(p2):
                return model.fim[(p1, p2)] == m.egb_fim_block.inputs[(p2, p1)]
            else:
                return pyo.Constraint.Skip

        # Add the FIM and External Grey
        # Box inputs constraints
        model.obj_cons.FIM_equalities = pyo.Constraint(
            model.parameter_names, model.parameter_names, rule=FIM_egb_cons
        )

        # Add objective based on user provided
        # type within ObjectiveLib
        if self.objective_option == ObjectiveLib.trace:
            model.objective = pyo.Objective(
                expr=model.obj_cons.egb_fim_block.outputs["A-opt"], sense=pyo.minimize
            )
        elif self.objective_option == ObjectiveLib.determinant:
            model.objective = pyo.Objective(
                expr=model.obj_cons.egb_fim_block.outputs["log-D-opt"],
                sense=pyo.maximize,
            )
        elif self.objective_option == ObjectiveLib.minimum_eigenvalue:
            model.objective = pyo.Objective(
                expr=model.obj_cons.egb_fim_block.outputs["E-opt"], sense=pyo.maximize
            )
        elif self.objective_option == ObjectiveLib.condition_number:
            model.objective = pyo.Objective(
                expr=model.obj_cons.egb_fim_block.outputs["ME-opt"], sense=pyo.minimize
            )
        # Else error not needed for spurious objective
        # options as the error will always appear
        # when creating the FIMExternalGreyBox block

    # Check to see if the model has all the required suffixes
    def check_model_labels(self, model=None):
        """
        Checks if the model contains the necessary suffixes for the
        DoE model to be constructed automatically.

        Parameters
        ----------
        model: model for suffix checking

        """
        # Check that experimental outputs exist
        try:
            outputs = [k.name for k, v in model.experiment_outputs.items()]
        except:
            raise RuntimeError(
                "Experiment model does not have suffix " + '"experiment_outputs".'
            )

        # Check that experiment_outputs is not empty
        if len(outputs) == 0:
            raise ValueError(
                "No experiment outputs found. Design of Experiments requires at least "
                "one experiment output (measurement) to optimize. Please add an "
                "'experiment_outputs' Suffix to your model with at least one variable."
            )

        # Check that experimental inputs exist
        try:
            inputs = [k.name for k, v in model.experiment_inputs.items()]
        except:
            raise RuntimeError(
                "Experiment model does not have suffix " + '"experiment_inputs".'
            )

        # Check that experiment_inputs is not empty
        if len(inputs) == 0:
            raise ValueError(
                "No experiment inputs found. Design of Experiments requires at least "
                "one experiment input (design variable) to optimize. Please add an "
                "'experiment_inputs' Suffix to your model with at least one variable."
            )

        # Check that unknown parameters exist
        try:
            parameters = [k.name for k, v in model.unknown_parameters.items()]
        except:
            raise RuntimeError(
                "Experiment model does not have suffix " + '"unknown_parameters".'
            )

        # Check that unknown_parameters is not empty
        if len(parameters) == 0:
            raise ValueError(
                "No unknown parameters found. Design of Experiments requires at least "
                "one unknown parameter to estimate. Please add an "
                "'unknown_parameters' Suffix to your model with at least one variable."
            )

        # Check that measurement errors exist
        try:
            errors = [k.name for k, v in model.measurement_error.items()]
        except:
            raise RuntimeError(
                "Experiment model does not have suffix " + '"measurement_error".'
            )

        # Check that measurement_error is not empty
        if len(errors) == 0:
            raise ValueError(
                "No measurement errors found. Design of Experiments requires at least "
                "one measurement error specification. Please add a "
                "'measurement_error' Suffix to your model with at least one variable."
            )

        # Check that measurement_error and experiment_outputs have the same length
        if len(errors) != len(outputs):
            raise ValueError(
                "Number of experiment outputs, {}, and length of measurement error, "
                "{}, do not match. Please check model labeling.".format(
                    len(outputs), len(errors)
                )
            )

        self.logger.info("Model has expected labels.")

    # Check the FIM shape against what is expected from the model.
    def check_model_FIM(self, model=None, FIM=None):
        """
        Checks if the specified matrix, FIM, matches the shape expected
        from the model. This method should only be called after the
        model has been probed for the length of the unknown parameter,
        experiment input, experiment output, and measurement error
        has been stored to the object.

        Parameters
        ----------
        model: model for suffix checking, Default: None, (self.model)
        FIM: FIM value to check on the model
        """
        if model is None:
            model = self.model

        if FIM.shape != (self.n_parameters, self.n_parameters):
            raise ValueError(
                "Shape of FIM provided should be n parameters by n parameters, "
                "or {} by {}, FIM provided has shape {} by {}".format(
                    self.n_parameters, self.n_parameters, FIM.shape[0], FIM.shape[1]
                )
            )

        check_FIM(FIM)

        self.logger.info(
            "FIM provided matches expected dimensions from model "
            "and is approximately positive (semi) definite."
        )

    # Check the jacobian shape against what is expected from the model.
    def check_model_jac(self, jac=None):
        if jac.shape != (self.n_experiment_outputs, self.n_parameters):
            raise ValueError(
                "Shape of Jacobian provided should be n experiment outputs "
                "by n parameters, or {} by {}, Jacobian provided has "
                "shape {} by {}".format(
                    self.n_experiment_outputs,
                    self.n_parameters,
                    jac.shape[0],
                    jac.shape[1],
                )
            )

        self.logger.info("Jacobian provided matches expected dimensions from model.")

    # Update the FIM for the specified model
    def update_FIM_prior(self, model=None, FIM=None):
        """
        Updates the prior FIM on the model object. This may be useful when
        running a loop and the user doesn't want to rebuild the model
        because it is expensive to build/initialize.

        Parameters
        ----------
        model: model where FIM prior is to be updated, Default: None, (self.model)
        FIM: 2D np array to be the new FIM prior, Default: None
        """
        if model is None:
            model = self.model

        # Check FIM input
        if FIM is None:
            raise ValueError(
                "FIM input for update_FIM_prior must be a 2D, square numpy array."
            )

        if not hasattr(model, "fim"):
            raise RuntimeError(
                "``fim`` is not defined on the model provided. "
                "Please build the model first."
            )

        self.check_model_FIM(model=model, FIM=FIM)

        # Update FIM prior
        for ind1, p1 in enumerate(model.parameter_names):
            for ind2, p2 in enumerate(model.parameter_names):
                model.prior_FIM[p1, p2].set_value(FIM[ind1, ind2])

        self.logger.info("FIM prior has been updated.")

    # TODO: Add an update function for the parameter values?
    #       Closed loop parameter estimation?
    def update_unknown_parameter_values(self, model=None, param_vals=None):
        raise NotImplementedError(
            "Updating unknown parameter values not yet supported."
        )

    # Evaluates FIM and statistics for a
    # full factorial space (same as run_grid_search)
    def compute_FIM_full_factorial(
        self, model=None, design_ranges=None, method="sequential"
    ):
        """
        Will run a simulation-based full factorial exploration of
        the experimental input space (i.e., a ``grid search`` or
        ``parameter sweep``) to understand how the FIM metrics
        change as a function of the experimental design space.

        Parameters
        ----------
        model: DoE model, optional
            model to perform the full factorial exploration on
        design_ranges: dict
            dictionary of lists, of the form {<var_name>: [start, stop, numsteps]}
        method: str, optional
            to specify which method should be used.
            Options are ``kaug`` and ``sequential``

        Returns
        -------
        fim_factorial_results: dict
            A dictionary of the results with the following keys and their corresponding
            values as a list:

            - keys of model's experiment_inputs
            - "log10 D-opt": list of log10(D-optimality)
            - "log10 A-opt": list of log10(A-optimality)
            - "log10 pseudo A-opt": list of log10(trace(FIM))
            - "log10 E-opt": list of log10(E-optimality)
            - "log10 ME-opt": list of log10(ME-optimality)
            - "eigval_min": list of minimum eigenvalues
            - "eigval_max": list of maximum eigenvalues
            - "det_FIM": list of determinants
            - "trace_cov": list of traces of covariance matrix
            - "trace_FIM": list of traces of FIM
            - "solve_time": list of solve times

        Raises
        ------
        ValueError
            If the design_ranges' keys do not match the model's experiment_inputs' keys.
        """

        # Start timer
        sp_timer = TicTocTimer()
        sp_timer.tic(msg=None)
        self.logger.info("Beginning Full Factorial Design.")

        # Make new model for factorial design
        self.factorial_model = (
            self.experiment_list[0]
            .get_labeled_model(**self.get_labeled_model_args)
            .clone()
        )
        model = self.factorial_model

        # Permute the inputs to be aligned with the experiment input indices
        design_ranges_enum = {k: np.linspace(*v) for k, v in design_ranges.items()}
        design_map = {
            ind: (k[0].name, k[0])
            for ind, k in enumerate(model.experiment_inputs.items())
        }

        # Make the full space
        try:
            valid_inputs = 0
            des_ranges = []
            for k, v in design_map.items():
                if v[0] in design_ranges_enum.keys():
                    des_ranges.append(design_ranges_enum[v[0]])
                    valid_inputs += 1
            assert valid_inputs > 0

            factorial_points = product(*des_ranges)
        except:
            raise ValueError(
                "Design ranges keys must be a subset of experimental design names."
            )

        # TODO: Add more objective types? i.e., modified-E; G-opt; V-opt; etc?
        # TODO: Also, make this a result object, or more user friendly.
        fim_factorial_results = {k.name: [] for k, v in model.experiment_inputs.items()}
        fim_factorial_results.update(
            {
                "log10 D-opt": [],
                "log10 A-opt": [],
                "log10 pseudo A-opt": [],
                "log10 E-opt": [],
                "log10 ME-opt": [],
                "eigval_min": [],
                "eigval_max": [],
                "det_FIM": [],
                "trace_cov": [],
                "trace_FIM": [],
                "solve_time": [],
            }
        )

        successes = 0
        failures = 0
        total_points = np.prod(
            np.array([len(v) for k, v in design_ranges_enum.items()])
        )
        time_set = []
        curr_point = 1  # Initial current point
        for design_point in factorial_points:
            # Fix design variables at fixed experimental design point
            for i in range(len(design_point)):
                design_map[i][1].fix(design_point[i])

            # Timing and logging objects
            self.logger.info("=======Iteration Number: %s =====", curr_point)
            iter_timer = TicTocTimer()
            iter_timer.tic(msg=None)

            # Compute FIM with given options
            try:
                curr_point = successes + failures + 1

                # Logging information for each run
                self.logger.info("This is run %s out of %s.", curr_point, total_points)

                # Attempt the FIM computation
                self.compute_FIM(model=model, method=method)
                successes += 1

                # iteration time
                iter_t = iter_timer.toc(msg=None)
                time_set.append(iter_t)

                # More logging
                self.logger.info(
                    "The code has run for %s seconds.", round(sum(time_set), 2)
                )
                self.logger.info(
                    "Estimated remaining time:  %s seconds",
                    round(
                        sum(time_set) / (curr_point) * (total_points - curr_point + 1),
                        2,
                    ),
                )
            except:
                self.logger.warning(
                    ":::::::::::Warning: Cannot converge this run.::::::::::::"
                )
                failures += 1
                self.logger.warning("failed count:", failures)

                self._computed_FIM = np.zeros(self.prior_FIM.shape)

                iter_t = iter_timer.toc(msg=None)
                time_set.append(iter_t)

            FIM = self._computed_FIM

            (
                det_FIM,
                trace_cov,
                trace_FIM,
                E_vals,
                E_vecs,
                D_opt,
                A_opt,
                pseudo_A_opt,
                E_opt,
                ME_opt,
            ) = compute_FIM_metrics(FIM)

            # Append the values for each of the experiment inputs
            for k, v in model.experiment_inputs.items():
                fim_factorial_results[k.name].append(pyo.value(k))

            fim_factorial_results["log10 D-opt"].append(D_opt)
            fim_factorial_results["log10 A-opt"].append(A_opt)
            fim_factorial_results["log10 pseudo A-opt"].append(pseudo_A_opt)
            fim_factorial_results["log10 E-opt"].append(E_opt)
            fim_factorial_results["log10 ME-opt"].append(ME_opt)
            fim_factorial_results["eigval_min"].append(E_vals.min())
            fim_factorial_results["eigval_max"].append(E_vals.max())
            fim_factorial_results["det_FIM"].append(det_FIM)
            fim_factorial_results["trace_cov"].append(trace_cov)
            fim_factorial_results["trace_FIM"].append(trace_FIM)
            fim_factorial_results["solve_time"].append(time_set[-1])

        self.fim_factorial_results = fim_factorial_results

        return self.fim_factorial_results

    # TODO: Overhaul plotting functions to not use strings
    # TODO: Make the plotting functionalities work for >2 design features
    def draw_factorial_figure(
        self,
        results=None,
        sensitivity_design_variables=None,
        fixed_design_variables=None,
        full_design_variable_names=None,
        title_text="",
        xlabel_text="",
        ylabel_text="",
        figure_file_name=None,
        font_axes=16,
        font_tick=14,
        log_scale=True,
    ):
        """
        Extract results needed for drawing figures from
        the results dictionary provided by the
        ``compute_FIM_full_factorial`` function.

        Draw either the 1D sensitivity curve or 2D heatmap.

        Parameters
        ----------
        results: dictionary, results dictionary from ``compute_FIM_full_factorial``
                 default: None (self.fim_factorial_results)
        sensitivity_design_variables: a list, design variable names to draw sensitivity
        fixed_design_variables: a dictionary, keys are the design variable names to be
                                fixed, values are the value of it to be fixed.
        full_design_variable_names: a list, all the design variables in the problem.
        title_text: a string, name for the figure
        xlabel_text: a string, label for the x-axis of the figure
                    default: last design variable name
            In a 1D sensitivity curve, it should be design variable by
            which the curve is drawn
            In a 2D heatmap, it should be the second design variable
            in the design_ranges
        ylabel_text: a string, label for the y-axis of the figure
                    default: None (1D); first design variable name (2D)
            A 1D sensitivity curve does not need it.
            In a 2D heatmap, it should be the first
            design variable in the dv_ranges
        figure_file_name: string or Path, path to save the figure as
        font_axes: axes label font size
        font_tick: tick label font size
        log_scale: if True, the result matrix will be scaled by log10

        """
        if results is None:
            if not hasattr(self, "fim_factorial_results"):
                raise RuntimeError(
                    "Results must be provided or the "
                    "compute_FIM_full_factorial function must be run."
                )
            results = self.fim_factorial_results
            full_design_variable_names = [
                k.name for k, v in self.factorial_model.experiment_inputs.items()
            ]
        else:
            if full_design_variable_names is None:
                raise ValueError(
                    "If results object is provided, you must "
                    "include all the design variable names."
                )

        des_names = full_design_variable_names

        # Inputs must exist for the function to do anything
        # TODO: Put in a default value function?????
        if sensitivity_design_variables is None:
            raise ValueError("``sensitivity_design_variables`` must be included.")

        if fixed_design_variables is None:
            raise ValueError("``fixed_design_variables`` must be included.")

        # Check that the provided design variables are within the results object
        check_des_vars = True
        for k, v in fixed_design_variables.items():
            check_des_vars *= k in ([k2 for k2, v2 in results.items()])
        check_sens_vars = True
        for k in sensitivity_design_variables:
            check_sens_vars *= k in [k2 for k2, v2 in results.items()]

        if not check_des_vars:
            raise ValueError(
                "Fixed design variables do not all appear "
                "in the results object keys."
            )
        if not check_sens_vars:
            raise ValueError(
                "Sensitivity design variables do not all appear "
                "in the results object keys."
            )

        # TODO: Make it possible to plot pair-wise sensitivities for all variables
        #       e.g. a curve like low-dimensional posterior distributions
        if len(sensitivity_design_variables) > 2:
            raise NotImplementedError(
                "Currently, only 1D and 2D sensitivity plotting is supported."
            )

        if len(fixed_design_variables.keys()) + len(
            sensitivity_design_variables
        ) != len(des_names):
            raise ValueError(
                "Error: All design variables that are not used to "
                "generate sensitivity plots must be fixed."
            )

        if type(results) is dict:
            results_pd = pd.DataFrame(results)
        else:
            results_pd = results

        # generate a combination of logic to
        # filter the results of the DOF needed.
        # an example filter: (self.store_all_results_dataframe["CA0"]==5).
        if len(fixed_design_variables.keys()) != 0:
            filter = ""
            i = 0
            for k, v in fixed_design_variables.items():
                filter += "(results_pd['"
                filter += str(k)
                filter += "']=="
                filter += str(v)
                filter += ")"
                if i < (len(fixed_design_variables.keys()) - 1):
                    filter += "&"
                i += 1
            # extract results with other dimensions fixed
            figure_result_data = results_pd.loc[eval(filter)]

        # if there is no other fixed dimensions
        else:
            figure_result_data = results_pd

        # Add attributes for drawing figures in later functions
        self.figure_result_data = figure_result_data
        self.figure_sens_des_vars = sensitivity_design_variables
        self.figure_fixed_des_vars = fixed_design_variables

        # if one design variable name is given as DOF, draw 1D sensitivity curve
        if len(self.figure_sens_des_vars) == 1:
            self._curve1D(
                title_text,
                xlabel_text,
                font_axes=font_axes,
                font_tick=font_tick,
                log_scale=log_scale,
                figure_file_name=figure_file_name,
            )
        # if two design variable names are given as DOF, draw 2D heatmaps
        elif len(self.figure_sens_des_vars) == 2:
            self._heatmap(
                title_text,
                xlabel_text,
                ylabel_text,
                font_axes=font_axes,
                font_tick=font_tick,
                log_scale=log_scale,
                figure_file_name=figure_file_name,
            )
        # TODO: Add the multidimensional plotting
        else:
            pass

    def _curve1D(
        self,
        title_text,
        xlabel_text,
        font_axes=16,
        font_tick=14,
        figure_file_name=None,
        log_scale=True,
    ):
        """
        Draw 1D sensitivity curves for all design criteria

        Parameters
        ----------
        title_text: name of the figure, a string
        xlabel_text: x label title, a string.
            In a 1D sensitivity curve, it is the design
            variable by which the curve is drawn.
        font_axes: axes label font size
        font_tick: tick label font size
        figure_file_name: string or Path, path to save the figure as
        log_scale: if True, the result matrix will be scaled by log10

        Returns
        --------
        5 Figures of 1D sensitivity curves for each criterion
        """
        if figure_file_name is not None:
            show_fig = False
        else:
            show_fig = True

        # extract the range of the DOF design variable
        x_range = self.figure_result_data[self.figure_sens_des_vars[0]].values.tolist()

        # decide if the results are log scaled
        if log_scale:
            y_range_A = np.log10(self.figure_result_data["log10 A-opt"].values.tolist())
            y_range_pseudo_A = np.log10(
                self.figure_result_data["log10 pseudo A-opt"].values.tolist()
            )
            y_range_D = np.log10(self.figure_result_data["log10 D-opt"].values.tolist())
            y_range_E = np.log10(self.figure_result_data["log10 E-opt"].values.tolist())
            y_range_ME = np.log10(
                self.figure_result_data["log10 ME-opt"].values.tolist()
            )
        else:
            y_range_A = self.figure_result_data["log10 A-opt"].values.tolist()
            y_range_pseudo_A = self.figure_result_data[
                "log10 pseudo A-opt"
            ].values.tolist()
            y_range_D = self.figure_result_data["log10 D-opt"].values.tolist()
            y_range_E = self.figure_result_data["log10 E-opt"].values.tolist()
            y_range_ME = self.figure_result_data["log10 ME-opt"].values.tolist()

        # Draw A-optimality
        fig = plt.pyplot.figure()
        plt.pyplot.rc("axes", titlesize=font_axes)
        plt.pyplot.rc("axes", labelsize=font_axes)
        plt.pyplot.rc("xtick", labelsize=font_tick)
        plt.pyplot.rc("ytick", labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {"mathtext.default": "regular"}
        # plt.rcParams.update(params)
        ax.plot(x_range, y_range_A)
        ax.scatter(x_range, y_range_A)
        ax.set_ylabel("$log_{10}$(Trace of Covariance)")
        ax.set_xlabel(xlabel_text)
        plt.pyplot.title(title_text + ": A-optimality")
        if show_fig:
            plt.pyplot.show()
        else:
            plt.pyplot.savefig(
                pathlib.Path(figure_file_name + "_A_opt.png"), format="png", dpi=450
            )
        # Draw pseudo A-optimality
        fig = plt.pyplot.figure()
        plt.pyplot.rc("axes", titlesize=font_axes)
        plt.pyplot.rc("axes", labelsize=font_axes)
        plt.pyplot.rc("xtick", labelsize=font_tick)
        plt.pyplot.rc("ytick", labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {"mathtext.default": "regular"}
        # plt.rcParams.update(params)
        ax.plot(x_range, y_range_pseudo_A)
        ax.scatter(x_range, y_range_pseudo_A)
        ax.set_ylabel("$log_{10}$(Trace of FIM)")
        ax.set_xlabel(xlabel_text)
        plt.pyplot.title(title_text + ": pseudo A-optimality")
        if show_fig:
            plt.pyplot.show()
        else:
            plt.pyplot.savefig(
                pathlib.Path(figure_file_name + "_pseudo_A_opt.png"),
                format="png",
                dpi=450,
            )

        # Draw D-optimality
        fig = plt.pyplot.figure()
        plt.pyplot.rc("axes", titlesize=font_axes)
        plt.pyplot.rc("axes", labelsize=font_axes)
        plt.pyplot.rc("xtick", labelsize=font_tick)
        plt.pyplot.rc("ytick", labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {"mathtext.default": "regular"}
        # plt.rcParams.update(params)
        ax.plot(x_range, y_range_D)
        ax.scatter(x_range, y_range_D)
        ax.set_ylabel("$log_{10}$(Determinant)")
        ax.set_xlabel(xlabel_text)
        plt.pyplot.title(title_text + ": D-optimality")
        if show_fig:
            plt.pyplot.show()
        else:
            plt.pyplot.savefig(
                pathlib.Path(figure_file_name + "_D_opt.png"), format="png", dpi=450
            )

        # Draw E-optimality
        fig = plt.pyplot.figure()
        plt.pyplot.rc("axes", titlesize=font_axes)
        plt.pyplot.rc("axes", labelsize=font_axes)
        plt.pyplot.rc("xtick", labelsize=font_tick)
        plt.pyplot.rc("ytick", labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {"mathtext.default": "regular"}
        # plt.rcParams.update(params)
        ax.plot(x_range, y_range_E)
        ax.scatter(x_range, y_range_E)
        ax.set_ylabel("$log_{10}$ (Minimum eigenvalue)")
        ax.set_xlabel(xlabel_text)
        plt.pyplot.title(title_text + ": E-optimality")
        if show_fig:
            plt.pyplot.show()
        else:
            plt.pyplot.savefig(
                pathlib.Path(figure_file_name + "_E_opt.png"), format="png", dpi=450
            )

        # Draw Modified E-optimality
        fig = plt.pyplot.figure()
        plt.pyplot.rc("axes", titlesize=font_axes)
        plt.pyplot.rc("axes", labelsize=font_axes)
        plt.pyplot.rc("xtick", labelsize=font_tick)
        plt.pyplot.rc("ytick", labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {"mathtext.default": "regular"}
        # plt.rcParams.update(params)
        ax.plot(x_range, y_range_ME)
        ax.scatter(x_range, y_range_ME)
        ax.set_ylabel("$log_{10}$(Condition number)")
        ax.set_xlabel(xlabel_text)
        plt.pyplot.title(title_text + ": Modified E-optimality")
        if show_fig:
            plt.pyplot.show()
        else:
            plt.pyplot.savefig(
                pathlib.Path(figure_file_name + "_ME_opt.png"), format="png", dpi=450
            )

    def _heatmap(
        self,
        title_text,
        xlabel_text,
        ylabel_text,
        font_axes=16,
        font_tick=14,
        figure_file_name=None,
        log_scale=True,
    ):
        """
        Draw 2D heatmaps for all design criteria

        Parameters
        ----------
        title_text: name of the figure, a string
        xlabel_text: x label title, a string.
            In a 2D heatmap, it should be the second
            design variable in the design_ranges
        ylabel_text: y label title, a string.
            In a 2D heatmap, it should be the first
            design variable in the dv_ranges
        font_axes: axes label font size
        font_tick: tick label font size
        figure_file_name: string or Path, path to save the figure as
        log_scale: if True, the result matrix will be scaled by log10

        Returns
        --------
        4 Figures of 2D heatmap for each criterion
        """
        if figure_file_name is not None:
            show_fig = False
        else:
            show_fig = True

        des_names = [k for k, v in self.figure_fixed_des_vars.items()]
        sens_ranges = {}
        for i in self.figure_sens_des_vars:
            sens_ranges[i] = list(self.figure_result_data[i].unique())

        x_range = sens_ranges[self.figure_sens_des_vars[0]]
        y_range = sens_ranges[self.figure_sens_des_vars[1]]

        # extract the design criteria values
        A_range = self.figure_result_data["log10 A-opt"].values.tolist()
        pseudo_A_range = self.figure_result_data["log10 pseudo A-opt"].values.tolist()
        D_range = self.figure_result_data["log10 D-opt"].values.tolist()
        E_range = self.figure_result_data["log10 E-opt"].values.tolist()
        ME_range = self.figure_result_data["log10 ME-opt"].values.tolist()

        # reshape the design criteria values for heatmaps
        cri_a = np.asarray(A_range).reshape(len(x_range), len(y_range))
        cri_pseudo_a = np.asarray(pseudo_A_range).reshape(len(x_range), len(y_range))
        cri_d = np.asarray(D_range).reshape(len(x_range), len(y_range))
        cri_e = np.asarray(E_range).reshape(len(x_range), len(y_range))
        cri_e_cond = np.asarray(ME_range).reshape(len(x_range), len(y_range))

        self.cri_a = cri_a
        self.cri_pseudo_a = cri_pseudo_a
        self.cri_d = cri_d
        self.cri_e = cri_e
        self.cri_e_cond = cri_e_cond

        # decide if log scaled
        if log_scale:
            hes_a = np.log10(self.cri_a)
            hes_pseudo_a = np.log10(self.cri_pseudo_a)
            hes_e = np.log10(self.cri_e)
            hes_d = np.log10(self.cri_d)
            hes_e2 = np.log10(self.cri_e_cond)
        else:
            hes_a = self.cri_a
            hes_pseudo_a = self.cri_pseudo_a
            hes_e = self.cri_e
            hes_d = self.cri_d
            hes_e2 = self.cri_e_cond

        # set heatmap x,y ranges
        xLabel = x_range
        yLabel = y_range

        # A-optimality
        fig = plt.pyplot.figure()
        plt.pyplot.rc("axes", titlesize=font_axes)
        plt.pyplot.rc("axes", labelsize=font_axes)
        plt.pyplot.rc("xtick", labelsize=font_tick)
        plt.pyplot.rc("ytick", labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {"mathtext.default": "regular"}
        plt.pyplot.rcParams.update(params)
        ax.set_yticks(range(len(yLabel)))
        ax.set_yticklabels(yLabel)
        ax.set_ylabel(ylabel_text)
        ax.set_xticks(range(len(xLabel)))
        ax.set_xticklabels(xLabel)
        ax.set_xlabel(xlabel_text)
        im = ax.imshow(hes_a.T, cmap=plt.pyplot.cm.hot_r)
        ba = plt.pyplot.colorbar(im)
        ba.set_label("log10(trace(cov))")
        plt.pyplot.title(title_text + ": A-optimality")
        if show_fig:
            plt.pyplot.show()
        else:
            plt.pyplot.savefig(
                pathlib.Path(figure_file_name + "_A_opt.png"), format="png", dpi=450
            )

        # pseudo A-optimality
        fig = plt.pyplot.figure()
        plt.pyplot.rc("axes", titlesize=font_axes)
        plt.pyplot.rc("axes", labelsize=font_axes)
        plt.pyplot.rc("xtick", labelsize=font_tick)
        plt.pyplot.rc("ytick", labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {"mathtext.default": "regular"}
        plt.pyplot.rcParams.update(params)
        ax.set_yticks(range(len(yLabel)))
        ax.set_yticklabels(yLabel)
        ax.set_ylabel(ylabel_text)
        ax.set_xticks(range(len(xLabel)))
        ax.set_xticklabels(xLabel)
        ax.set_xlabel(xlabel_text)
        im = ax.imshow(hes_pseudo_a.T, cmap=plt.pyplot.cm.hot_r)
        ba = plt.pyplot.colorbar(im)
        ba.set_label("log10(trace(FIM))")
        plt.pyplot.title(title_text + ": pseudo A-optimality")
        if show_fig:
            plt.pyplot.show()
        else:
            plt.pyplot.savefig(
                pathlib.Path(figure_file_name + "_pseudo_A_opt.png"),
                format="png",
                dpi=450,
            )

        # D-optimality
        fig = plt.pyplot.figure()
        plt.pyplot.rc("axes", titlesize=font_axes)
        plt.pyplot.rc("axes", labelsize=font_axes)
        plt.pyplot.rc("xtick", labelsize=font_tick)
        plt.pyplot.rc("ytick", labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {"mathtext.default": "regular"}
        plt.pyplot.rcParams.update(params)
        ax.set_yticks(range(len(yLabel)))
        ax.set_yticklabels(yLabel)
        ax.set_ylabel(ylabel_text)
        ax.set_xticks(range(len(xLabel)))
        ax.set_xticklabels(xLabel)
        ax.set_xlabel(xlabel_text)
        im = ax.imshow(hes_d.T, cmap=plt.pyplot.cm.hot_r)
        ba = plt.pyplot.colorbar(im)
        ba.set_label("log10(det(FIM))")
        plt.pyplot.title(title_text + ": D-optimality")
        if show_fig:
            plt.pyplot.show()
        else:
            plt.pyplot.savefig(
                pathlib.Path(figure_file_name + "_D_opt.png"), format="png", dpi=450
            )

        # E-optimality
        fig = plt.pyplot.figure()
        plt.pyplot.rc("axes", titlesize=font_axes)
        plt.pyplot.rc("axes", labelsize=font_axes)
        plt.pyplot.rc("xtick", labelsize=font_tick)
        plt.pyplot.rc("ytick", labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {"mathtext.default": "regular"}
        plt.pyplot.rcParams.update(params)
        ax.set_yticks(range(len(yLabel)))
        ax.set_yticklabels(yLabel)
        ax.set_ylabel(ylabel_text)
        ax.set_xticks(range(len(xLabel)))
        ax.set_xticklabels(xLabel)
        ax.set_xlabel(xlabel_text)
        im = ax.imshow(hes_e.T, cmap=plt.pyplot.cm.hot_r)
        ba = plt.pyplot.colorbar(im)
        ba.set_label("log10(minimal eig(FIM))")
        plt.pyplot.title(title_text + ": E-optimality")
        if show_fig:
            plt.pyplot.show()
        else:
            plt.pyplot.savefig(
                pathlib.Path(figure_file_name + "_E_opt.png"), format="png", dpi=450
            )

        # Modified E-optimality
        fig = plt.pyplot.figure()
        plt.pyplot.rc("axes", titlesize=font_axes)
        plt.pyplot.rc("axes", labelsize=font_axes)
        plt.pyplot.rc("xtick", labelsize=font_tick)
        plt.pyplot.rc("ytick", labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {"mathtext.default": "regular"}
        plt.pyplot.rcParams.update(params)
        ax.set_yticks(range(len(yLabel)))
        ax.set_yticklabels(yLabel)
        ax.set_ylabel(ylabel_text)
        ax.set_xticks(range(len(xLabel)))
        ax.set_xticklabels(xLabel)
        ax.set_xlabel(xlabel_text)
        im = ax.imshow(hes_e2.T, cmap=plt.pyplot.cm.hot_r)
        ba = plt.pyplot.colorbar(im)
        ba.set_label("log10(cond(FIM))")
        plt.pyplot.title(title_text + ": Modified E-optimality")
        if show_fig:
            plt.pyplot.show()
        else:
            plt.pyplot.savefig(
                pathlib.Path(figure_file_name + "_ME_opt.png"), format="png", dpi=450
            )

    # Gets the FIM from an existing model
    def get_FIM(self, model=None):
        """
        Gets the FIM values from the model specified

        Parameters
        ----------
        model: model to grab FIM from, Default: None, (self.model)

        Returns
        -------
        FIM: 2D list representation of the FIM (can be cast to numpy)

        """
        if model is None:
            model = self.model

        if not hasattr(model, "fim"):
            raise RuntimeError(
                "Model provided does not have variable `fim`. Please make sure "
                "the model is built properly before calling `get_FIM`"
            )

        fim_vals = [
            pyo.value(model.fim[i, j])
            for i in model.parameter_names
            for j in model.parameter_names
        ]
        fim_np = np.array(fim_vals).reshape(
            (len(model.parameter_names), len(model.parameter_names))
        )

        # FIM is a lower triangular matrix for the optimal DoE problem.
        # Exploit symmetry to fill in the zeros.
        for i in range(len(model.parameter_names)):
            for j in range(len(model.parameter_names)):
                if j < i:
                    fim_np[j, i] = fim_np[i, j]

        return [list(row) for row in list(fim_np)]

    # Gets the sensitivity matrix from an existing model
    def get_sensitivity_matrix(self, model=None):
        """
        Gets the sensitivity matrix (Q) values from the model specified.

        Parameters
        ----------
        model: model to grab Q from, Default: None, (self.model)

        Returns
        -------
        Q: 2D list representation of the sensitivity matrix (can be cast to numpy)

        """
        if model is None:
            model = self.model

        if not hasattr(model, "sensitivity_jacobian"):
            raise RuntimeError(
                "Model provided does not have variable `sensitivity_jacobian`. "
                "Please make sure the model is built properly before calling "
                "`get_sensitivity_matrix`"
            )

        Q_vals = [
            pyo.value(model.sensitivity_jacobian[i, j])
            for i in model.output_names
            for j in model.parameter_names
        ]
        Q_np = np.array(Q_vals).reshape(
            (len(model.output_names), len(model.parameter_names))
        )

        return [list(row) for row in list(Q_np)]

    # Gets the experiment input values from an existing model
    def get_experiment_input_values(self, model=None):
        """
        Gets the experiment input values (experimental design)
        from the model specified.

        Parameters
        ----------
        model: model to grab the experimental design from,
             default: None, (self.model)

        Returns
        -------
        d: 1D list of experiment input values (optimal or specified design)

        """
        if model is None:
            model = self.model

        if not hasattr(model, "experiment_inputs"):
            if not hasattr(model, "fd_scenario_blocks"):
                raise RuntimeError(
                    "Model provided does not have expected structure. "
                    "Please make sure model is built properly before "
                    "calling `get_experiment_input_values`"
                )

            d_vals = [
                pyo.value(k)
                for k, v in model.fd_scenario_blocks[0].experiment_inputs.items()
            ]
        else:
            d_vals = [pyo.value(k) for k, v in model.experiment_inputs.items()]

        return d_vals

    # Gets the unknown parameter values from an existing model
    def get_unknown_parameter_values(self, model=None):
        """
        Gets the unknown parameter values (theta)
        from the model specified.

        Parameters
        ----------
        model: model to grab theta from,
             default: None, (self.model)

        Returns
        -------
        theta: 1D list of unknown parameter values at which
               this experiment was designed

        """
        if model is None:
            model = self.model

        if not hasattr(model, "unknown_parameters"):
            if not hasattr(model, "fd_scenario_blocks"):
                raise RuntimeError(
                    "Model provided does not have expected structure. Please make "
                    "sure model is built properly before calling "
                    "`get_unknown_parameter_values`"
                )

            theta_vals = [
                pyo.value(k)
                for k, v in model.fd_scenario_blocks[0].unknown_parameters.items()
            ]
        else:
            theta_vals = [pyo.value(k) for k, v in model.unknown_parameters.items()]

        return theta_vals

    # Gets the experiment output values from an existing model
    def get_experiment_output_values(self, model=None):
        """
        Gets the experiment output values (y hat)
        from the model specified.

        Parameters
        ----------
        model: model to grab y hat from,
             default: None, (self.model)

        Returns
        -------
        y_hat: 1D list of experiment output values from the design experiment

        """
        if model is None:
            model = self.model

        if not hasattr(model, "experiment_outputs"):
            if not hasattr(model, "fd_scenario_blocks"):
                raise RuntimeError(
                    "Model provided does not have expected structure. Please make "
                    "sure model is built properly before calling "
                    "`get_experiment_output_values`"
                )

            y_hat_vals = [
                pyo.value(k)
                for k, v in model.fd_scenario_blocks[0].experiment_outputs.items()
            ]
        else:
            y_hat_vals = [pyo.value(k) for k, v in model.experiment_outputs.items()]

        return y_hat_vals

    # TODO: For more complicated error structures, this should become
    #       get cov_y, or so, and this method will be deprecated
    # Gets the measurement error values from an existing model
    def get_measurement_error_values(self, model=None):
        """
        Gets the experiment output values (sigma)
        from the model specified.

        Parameters
        ----------
        model: model to grab sigma values from,
             default: None, (self.model)

        Returns
        -------
        sigma_diag: 1D list of measurement errors used to design the experiment

        """
        if model is None:
            model = self.model

        if not hasattr(model, "measurement_error"):
            if not hasattr(model, "fd_scenario_blocks"):
                raise RuntimeError(
                    "Model provided does not have expected structure. Please make "
                    "sure model is built properly before calling "
                    "`get_measurement_error_values`"
                )

            sigma_vals = [
                pyo.value(k)
                for k, v in model.fd_scenario_blocks[0].measurement_error.items()
            ]
        else:
            sigma_vals = [pyo.value(k) for k, v in model.measurement_error.items()]

        return sigma_vals

    # Helper function for determinant calculation
    def _sgn(self, p):
        """
        This is a helper function for when constructing the determinant formula
        without the Cholesky factorization.

        Parameters
        -----------
        p: the permutation (a list)

        Returns
        -------
        1 if the number of exchange is an even number
        -1 if the number is an odd number
        """

        if len(p) == 1:
            return 1

        trans = 0

        for i in range(0, len(p)):
            j = i + 1

            for j in range(j, len(p)):
                if p[i] > p[j]:
                    trans = trans + 1

        if (trans % 2) == 0:
            return 1
        else:
            return -1
