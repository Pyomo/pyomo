#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#
#  Pyomo.DoE was produced under the Department of Energy Carbon Capture Simulation
#  Initiative (CCSI), and is copyright (c) 2022 by the software owners:
#  TRIAD National Security, LLC., Lawrence Livermore National Security, LLC.,
#  Lawrence Berkeley National Laboratory, Pacific Northwest National Laboratory,
#  Battelle Memorial Institute, University of Notre Dame,
#  The University of Pittsburgh, The University of Texas at Austin,
#  University of Toledo, West Virginia University, et al. All rights reserved.
#
#  NOTICE. This Software was developed under funding from the
#  U.S. Department of Energy and the U.S. Government consequently retains
#  certain rights. As such, the U.S. Government has been granted for itself
#  and others acting on its behalf a paid-up, nonexclusive, irrevocable,
#  worldwide license in the Software to reproduce, distribute copies to the
#  public, prepare derivative works, and perform publicly and display
#  publicly, and to permit other to do so.
#  ___________________________________________________________________________

from enum import Enum
from itertools import permutations, product

import json
import logging
import math

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

import pyomo.environ as pyo
from pyomo.contrib.doe.utils import (
    check_FIM,
    compute_FIM_metrics,
    _SMALL_TOLERANCE_DEFINITENESS,
)

from pyomo.opt import SolverStatus


class ObjectiveLib(Enum):
    determinant = "determinant"
    trace = "trace"
    minimum_eigenvalue = "minimum_eigenvalue"
    condition_number = "condition_number"
    zero = "zero"


class FiniteDifferenceStep(Enum):
    forward = "forward"
    central = "central"
    backward = "backward"


class DesignOfExperiments:
    def __init__(
        self,
        experiment=None,
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
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
    ):
        """
        This package enables model-based design of experiments analysis with Pyomo.
        Both direct optimization and enumeration modes are supported.

        The package has been refactored from its original form as of August 24. See
        the documentation for more information.

        Parameters
        ----------
        experiment:
            Experiment object that holds the model and labels all the components. The
            object should have a ``get_labeled_model`` where a model is returned with
            the following labeled sets: ``unknown_parameters``,
                                        ``experimental_inputs``,
                                        ``experimental_outputs``
        fd_formula:
            Finite difference formula for computing the sensitivity matrix. Must be
            one of [``central``, ``forward``, ``backward``], default: ``central``
        step:
            Relative step size for the finite difference formula.
            default: 1e-3
        objective_option:
            String representation of the objective option. Current available options
            are: ``determinant`` (for determinant, or D-optimality),
            ``trace`` (for trace, or A-optimality), ``minimum_eigenvalue``, (for
            E-optimality), or ``condition_number`` (for ME-optimality)
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
        if experiment is None:
            raise ValueError("Experiment object must be provided to perform DoE.")

        # Check if the Experiment object has callable ``get_labeled_model`` function
        if not hasattr(experiment, "get_labeled_model"):
            raise ValueError(
                "The experiment object must have a ``get_labeled_model`` function"
            )

        # Set the experiment object from the user
        self.experiment = experiment

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
            if type(results_file) not in [pathlib.Path, str]:
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
        for comp in model.scenario_blocks[0].experiment_inputs:
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
        for comp in model.scenario_blocks[0].experiment_inputs:
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
                # Do safe inverse here?
                trace_val = 1 / np.trace(np.array(self.get_FIM()))
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
                cond_number = np.linalg.cond(np.array(self.get_FIM()))
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
            min_eig = np.min(np.linalg.eigvals(fim_np))

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
            str(pyo.ComponentUID(k, context=model.scenario_blocks[0]))
            for k in model.scenario_blocks[0].experiment_inputs
        ]
        self.results["Experiment Outputs"] = self.get_experiment_output_values()
        self.results["Experiment Output Names"] = [
            str(pyo.ComponentUID(k, context=model.scenario_blocks[0]))
            for k in model.scenario_blocks[0].experiment_outputs
        ]
        self.results["Unknown Parameters"] = self.get_unknown_parameter_values()
        self.results["Unknown Parameter Names"] = [
            str(pyo.ComponentUID(k, context=model.scenario_blocks[0]))
            for k in model.scenario_blocks[0].unknown_parameters
        ]
        self.results["Measurement Error"] = self.get_measurement_error_values()
        self.results["Measurement Error Names"] = [
            str(pyo.ComponentUID(k, context=model.scenario_blocks[0]))
            for k in model.scenario_blocks[0].measurement_error
        ]

        self.results["Prior FIM"] = [list(row) for row in list(self.prior_FIM)]

        # Saving some stats on the FIM for convenience
        self.results["Objective expression"] = str(self.objective_option).split(".")[-1]
        self.results["log10 A-opt"] = np.log10(np.trace(fim_local))
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
            self.compute_FIM_model = self.experiment.get_labeled_model(
                **self.get_labeled_model_args
            ).clone()
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
            self.compute_FIM_model = self.experiment.get_labeled_model(
                **self.get_labeled_model_args
            ).clone()
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
            cov_y[count, count] = 1 / v
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
            self.compute_FIM_model = self.experiment.get_labeled_model(
                **self.get_labeled_model_args
            ).clone()
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
            cov_y[count, count] = 1 / v
            count += 1

        # TODO: need to add a covariance matrix for measurements (sigma inverse)
        # i.e., cov_y = self.cov_y or model.cov_y
        # Still deciding where this would be best.

        self.kaug_FIM = self.kaug_jac.T @ cov_y @ self.kaug_jac + self.prior_FIM

    # Create the DoE model (with ``scenarios`` from finite differencing scheme)
    def create_doe_model(self, model=None):
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
        self._generate_scenario_blocks(model=model)

        # Set names for indexing sensitivity matrix (jacobian) and FIM
        scen_block_ind = min(
            [
                k.name.split(".").index("scenario_blocks[0]")
                for k in model.scenario_blocks[0].unknown_parameters.keys()
            ]
        )
        model.parameter_names = pyo.Set(
            initialize=[
                ".".join(k.name.split(".")[(scen_block_ind + 1) :])
                for k in model.scenario_blocks[0].unknown_parameters.keys()
            ]
        )
        model.output_names = pyo.Set(
            initialize=[
                ".".join(k.name.split(".")[(scen_block_ind + 1) :])
                for k in model.scenario_blocks[0].experiment_outputs.keys()
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
                (bu, un): self.fim_initial[i][j]
                for i, bu in enumerate(model.parameter_names)
                for j, un in enumerate(model.parameter_names)
            }

        def initialize_fim(m, j, d):
            return dict_fim_initialize[(j, d)]

        if self.fim_initial is not None:
            model.fim = pyo.Var(
                model.parameter_names, model.parameter_names, initialize=initialize_fim
            )
        else:
            model.fim = pyo.Var(
                model.parameter_names, model.parameter_names, initialize=identity_matrix
            )

        # To-Do: Look into this functionality.....
        # if cholesky, define L elements as variables
        if self.Cholesky_option and self.objective_option == ObjectiveLib.determinant:
            model.L = pyo.Var(
                model.parameter_names, model.parameter_names, initialize=identity_matrix
            )

            # loop over parameter name
            for i, c in enumerate(model.parameter_names):
                for j, d in enumerate(model.parameter_names):
                    # fix the 0 half of L matrix to be 0.0
                    if i < j:
                        model.L[c, d].fix(0.0)
                    # Give LB to the diagonal entries
                    if self.L_diagonal_lower_bound:
                        if c == d:
                            model.L[c, d].setlb(self.L_diagonal_lower_bound)

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

            var_up = cuid.find_component_on(m.scenario_blocks[s1])
            var_lo = cuid.find_component_on(m.scenario_blocks[s2])

            param = m.parameter_scenarios[max(s1, s2)]
            param_loc = pyo.ComponentUID(param).find_component_on(m.scenario_blocks[0])
            param_val = m.scenario_blocks[0].unknown_parameters[param_loc]
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
                return (
                    m.fim[p, q]
                    == sum(
                        1
                        / m.scenario_blocks[0].measurement_error[
                            pyo.ComponentUID(n).find_component_on(m.scenario_blocks[0])
                        ]
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

    # Create scenario block structure
    def _generate_scenario_blocks(self, model=None):
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
        """
        # If model is none, assume it is self.model
        if model is None:
            model = self.model

        # Generate initial scenario to populate unknown parameter values
        model.base_model = self.experiment.get_labeled_model(
            **self.get_labeled_model_args
        ).clone()

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
            m = b.model()
            b.transfer_attributes_from(m.base_model.clone())

            # Forward/Backward difference have a stationary
            # case (s == 0), no parameter to perturb
            if self.fd_formula in [
                FiniteDifferenceStep.forward,
                FiniteDifferenceStep.backward,
            ]:
                if s == 0:
                    return

            param = m.parameter_scenarios[s]

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
            pyo.ComponentUID(param, context=m.base_model).find_component_on(
                b
            ).set_value(m.base_model.unknown_parameters[param] * (1 + diff))

            # Fix experiment inputs before solve (enforce square solve)
            for comp in b.experiment_inputs:
                comp.fix()

            res = self.solver.solve(b, tee=self.tee)

            # Unfix experiment inputs after square solve
            for comp in b.experiment_inputs:
                comp.unfix()

        model.scenario_blocks = pyo.Block(model.scenarios, rule=build_block_scenarios)

        # TODO: this might have to change if experiment inputs have
        #       a different value in the Suffix (currently it is the CUID)
        design_vars = [k for k, v in model.scenario_blocks[0].experiment_inputs.items()]

        # Add constraints to equate block design with global design:
        for ind, d in enumerate(design_vars):
            con_name = "global_design_eq_con_" + str(ind)

            # Constraint rule for global design constraints
            def global_design_fixing(m, s):
                if s == 0:
                    return pyo.Constraint.Skip
                block_design_var = pyo.ComponentUID(
                    d, context=m.scenario_blocks[0]
                ).find_component_on(m.scenario_blocks[s])
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
        fim_vals = [
            model.fim[bu, un].value
            for i, bu in enumerate(model.parameter_names)
            for j, un in enumerate(model.parameter_names)
        ]
        fim = np.array(fim_vals).reshape(
            len(model.parameter_names), len(model.parameter_names)
        )

        ### Initialize the Cholesky decomposition matrix
        if self.Cholesky_option and self.objective_option == ObjectiveLib.determinant:
            # Calculate the eigenvalues of the FIM matrix
            eig = np.linalg.eigvals(fim)

            # If the smallest eigenvalue is (practically) negative,
            # add a diagonal matrix to make it positive definite
            small_number = 1e-10
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

        def cholesky_imp(b, c, d):
            """
            Calculate Cholesky L matrix using algebraic constraints
            """
            # If the row is greater than or equal to the column, we are in the
            # lower triangle region of the L and FIM matrices.
            # This region is where our equations are well-defined.
            m = b.model()
            if list(m.parameter_names).index(c) >= list(m.parameter_names).index(d):
                return m.fim[c, d] == sum(
                    m.L[c, m.parameter_names.at(k + 1)]
                    * m.L[d, m.parameter_names.at(k + 1)]
                    for k in range(list(m.parameter_names).index(d) + 1)
                )
            else:
                # This is the empty half of L above the diagonal
                return pyo.Constraint.Skip

        def trace_calc(b):
            """
            Calculate FIM elements. Can scale each element with 1000 for performance
            """
            m = b.model()
            return m.trace == sum(m.fim[j, j] for j in m.parameter_names)

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
                model.parameter_names, model.parameter_names, rule=cholesky_imp
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
            # if not determinant or Cholesky, calculating
            # the OBJ with trace
            model.trace = pyo.Var(initialize=np.trace(fim), bounds=(small_number, None))
            model.obj_cons.trace_rule = pyo.Constraint(rule=trace_calc)
            model.objective = pyo.Objective(
                expr=pyo.log10(model.trace), sense=pyo.maximize
            )

        # TODO: Add warning (should be unreachable) if the user calls
        #       the grey box objectives with the standard model
        elif self.objective_option == ObjectiveLib.zero:
            # add dummy objective function
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

        # Check that experimental inputs exist
        try:
            outputs = [k.name for k, v in model.experiment_inputs.items()]
        except:
            raise RuntimeError(
                "Experiment model does not have suffix " + '"experiment_inputs".'
            )

        # Check that unknown parameters exist
        try:
            outputs = [k.name for k, v in model.unknown_parameters.items()]
        except:
            raise RuntimeError(
                "Experiment model does not have suffix " + '"unknown_parameters".'
            )

        # Check that measurement errors exist
        try:
            outputs = [k.name for k, v in model.measurement_error.items()]
        except:
            raise RuntimeError(
                "Experiment model does not have suffix " + '"measurement_error".'
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
            - "log10 E-opt": list of log10(E-optimality)
            - "log10 ME-opt": list of log10(ME-optimality)
            - "eigval_min": list of minimum eigenvalues
            - "eigval_max": list of maximum eigenvalues
            - "det_FIM": list of determinants
            - "trace_FIM": list of traces
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
        self.factorial_model = self.experiment.get_labeled_model(
            **self.get_labeled_model_args
        ).clone()
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
                "log10 E-opt": [],
                "log10 ME-opt": [],
                "eigval_min": [],
                "eigval_max": [],
                "det_FIM": [],
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

            det_FIM, trace_FIM, E_vals, E_vecs, D_opt, A_opt, E_opt, ME_opt = (
                compute_FIM_metrics(FIM)
            )

            # Append the values for each of the experiment inputs
            for k, v in model.experiment_inputs.items():
                fim_factorial_results[k.name].append(pyo.value(k))

            fim_factorial_results["log10 D-opt"].append(D_opt)
            fim_factorial_results["log10 A-opt"].append(A_opt)
            fim_factorial_results["log10 E-opt"].append(E_opt)
            fim_factorial_results["log10 ME-opt"].append(ME_opt)
            fim_factorial_results["eigval_min"].append(E_vals.min())
            fim_factorial_results["eigval_max"].append(E_vals.max())
            fim_factorial_results["det_FIM"].append(det_FIM)
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
        4 Figures of 1D sensitivity curves for each criterion
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
            y_range_D = np.log10(self.figure_result_data["log10 D-opt"].values.tolist())
            y_range_E = np.log10(self.figure_result_data["log10 E-opt"].values.tolist())
            y_range_ME = np.log10(
                self.figure_result_data["log10 ME-opt"].values.tolist()
            )
        else:
            y_range_A = self.figure_result_data["log10 A-opt"].values.tolist()
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
        ax.set_ylabel("$log_{10}$ Trace")
        ax.set_xlabel(xlabel_text)
        plt.pyplot.title(title_text + ": A-optimality")
        if show_fig:
            plt.pyplot.show()
        else:
            plt.pyplot.savefig(
                pathlib.Path(figure_file_name + "_A_opt.png"), format="png", dpi=450
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
        ax.set_ylabel("$log_{10}$ Determinant")
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
        ax.set_ylabel("$log_{10}$ Minimal eigenvalue")
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
        ax.set_ylabel("$log_{10}$ Condition number")
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
        D_range = self.figure_result_data["log10 D-opt"].values.tolist()
        E_range = self.figure_result_data["log10 E-opt"].values.tolist()
        ME_range = self.figure_result_data["log10 ME-opt"].values.tolist()

        # reshape the design criteria values for heatmaps
        cri_a = np.asarray(A_range).reshape(len(x_range), len(y_range))
        cri_d = np.asarray(D_range).reshape(len(x_range), len(y_range))
        cri_e = np.asarray(E_range).reshape(len(x_range), len(y_range))
        cri_e_cond = np.asarray(ME_range).reshape(len(x_range), len(y_range))

        self.cri_a = cri_a
        self.cri_d = cri_d
        self.cri_e = cri_e
        self.cri_e_cond = cri_e_cond

        # decide if log scaled
        if log_scale:
            hes_a = np.log10(self.cri_a)
            hes_e = np.log10(self.cri_e)
            hes_d = np.log10(self.cri_d)
            hes_e2 = np.log10(self.cri_e_cond)
        else:
            hes_a = self.cri_a
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
        ba.set_label("log10(trace(FIM))")
        plt.pyplot.title(title_text + ": A-optimality")
        if show_fig:
            plt.pyplot.show()
        else:
            plt.pyplot.savefig(
                pathlib.Path(figure_file_name + "_A_opt.png"), format="png", dpi=450
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
            if not hasattr(model, "scenario_blocks"):
                raise RuntimeError(
                    "Model provided does not have expected structure. "
                    "Please make sure model is built properly before "
                    "calling `get_experiment_input_values`"
                )

            d_vals = [
                pyo.value(k)
                for k, v in model.scenario_blocks[0].experiment_inputs.items()
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
            if not hasattr(model, "scenario_blocks"):
                raise RuntimeError(
                    "Model provided does not have expected structure. Please make "
                    "sure model is built properly before calling "
                    "`get_unknown_parameter_values`"
                )

            theta_vals = [
                pyo.value(k)
                for k, v in model.scenario_blocks[0].unknown_parameters.items()
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
            if not hasattr(model, "scenario_blocks"):
                raise RuntimeError(
                    "Model provided does not have expected structure. Please make "
                    "sure model is built properly before calling "
                    "`get_experiment_output_values`"
                )

            y_hat_vals = [
                pyo.value(k)
                for k, v in model.scenario_blocks[0].experiment_outputs.items()
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
            if not hasattr(model, "scenario_blocks"):
                raise RuntimeError(
                    "Model provided does not have expected structure. Please make "
                    "sure model is built properly before calling "
                    "`get_measurement_error_values`"
                )

            sigma_vals = [
                pyo.value(k)
                for k, v in model.scenario_blocks[0].measurement_error.items()
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
