#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
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


from pyomo.common.dependencies import numpy as np, numpy_available

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pickle
from itertools import permutations, product
import logging
from enum import Enum
from pyomo.common.timing import TicTocTimer
from pyomo.contrib.sensitivity_toolbox.sens import get_dsdp
from pyomo.contrib.doe.scenario import ScenarioGenerator, FiniteDifferenceStep
from pyomo.contrib.doe.result import FisherResults, GridSearchResult
import collections.abc

import inspect

from pyomo.common import DeveloperError


class CalculationMode(Enum):
    sequential_finite = "sequential_finite"
    direct_kaug = "direct_kaug"


class ObjectiveLib(Enum):
    det = "det"
    trace = "trace"
    zero = "zero"


class ModelOptionLib(Enum):
    parmest = "parmest"
    stage1 = "stage1"
    stage2 = "stage2"


# class FiniteDifferenceStep(Enum):
    # forward = "forward"
    # central = "central"
    # backward = "backward"


class DesignOfExperiments_:
    def __init__(
        self,
        experiment,
        fd_formula="central",
        step=1e-3,
        objective_option='det',
        scale_constant_value=1,
        scale_nominal_param_value=False,
        prior_FIM=None,
        jac_initial=None,
        fim_initial=None,
        L_initial=None,
        L_LB=1e-7,
        solver=None,
        tee=False,
        args=None,
        logger_level=logging.WARNING,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
    ):
        """
        This package enables model-based design of experiments analysis with Pyomo. 
        Both direct optimization and enumeration modes are supported.
        
        The package has been refactored from its original form as of ##/##/##24. See
        the documentation for more information.

        Parameters
        ----------
        experiment:
            Experiment object that holds the model and labels all the components. The object
            should have a ``get_labeled_model`` where a model is returned with the following
            labeled sets: ``unknown_parameters``, ``experimental_inputs``, ``experimental_outputs``
        fd_formula:
            Finite difference formula for computing the sensitivy matrix. Must be one of
            [``central``, ``forward``, ``backward``], default: ``central``
        step:
            Relative step size for the finite difference formula. 
            default: 1e-3
        objective_option:
            String representation of the objective option. Current available options are:
            ``det`` (for determinant, or D-optimality) and ``trace`` (for trace or
            A-optimality)
        scale_constant_value:
            Constant scaling for the sensitivty matrix. Every element will be multiplied by this
            scaling factor. 
            default: 1
        scale_nominal_param_value:
            Boolean for whether or not to scale the sensitivity matrix by the nominal parameter
            values. Every column of the sensitivity matrix will be divided by the respective
            nominal paramter value. 
            default: False
        prior_FIM:
            2D numpy array representing information from prior experiments. If no value is given,
            the assumed prior will be a matrix of zeros. This matrix will be assumed to be scaled
            as the user has specified (i.e., if scale_nominal_param_value is true, we will assume
            the FIM provided here has been scaled by the parameter values)
        jac_initial:
            2D numpy array as the initial values for the sensitivity matrix.
        fim_initial:
            2D numpy array as the initial values for the FIM.
        L_initial:
            2D numpy array as the initial values for the Cholesky matrix.
        L_LB:
            Lower bound for the values of the lower triangular Cholesky factorization matrix.
            default: 1e-7
        solver:
            A ``solver`` object specified by the user, default=None.
            If not specified, default solver is set to IPOPT with MA57.
        tee:
            Solver option to be passed for verbose output.
        args:
            Additional arguments for the ``get_labeled_model`` function on the Experiment object.
        _Cholesky_option:
            Boolean value of whether or not to use the choleskyn factorization to compute the
            determinant for the D-optimality criteria. This parameter should not be changed
            unless the user intends to make performance worse (i.e., compare an existing tool
            that uses the full FIM to this algorithm)
        _only_compute_fim_lower:
            If True, only the lower triangle of the FIM is computed. This parameter should not
            be changed unless the user intends to make performance worse (i.e., compare an
            existing tool that uses the full FIM to this algorithm)
        logger_level:
            Specify the level of the logger. Change to logging.DEBUG for all messages.
        """
        # Assert that the Experiment object has callable ``get_labeled_model`` function
        assert callable(getattr(experiment, 'get_labeled_model')), 'The experiment object must have a ``get_labeled_model`` function'
        
        # Set the experiment object from the user
        self.experiment = experiment
        
        # Set the finite difference and subsequent step size
        self.fd_formula = FiniteDifferenceStep(fd_formula)
        self.step = step

        # Set the objective type and scaling options:
        self.objective_option = ObjectiveLib(objective_option)
        
        self.scale_constant_value = scale_constant_value
        self.scale_nominal_param_value = scale_nominal_param_value
        
        # Set the prior FIM (will be checked upon model construction)
        self.prior_FIM = prior_FIM
        
        # Set the initial values for the jacobian, fim, and L matrices
        self.jac_initial = jac_initial
        self.fim_initial = fim_initial
        self.L_initial = L_initial
        
        # Set the lower bound on the Cholesky lower triangular matrix
        self.L_LB = L_LB

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
        
        # Set args as an empty dict if no arguments are passed
        if args is None:
            args = {}
        self.args = args

        # Revtrieve logger and set logging level
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logger_level)

        # Set the private options if passed (only developers should pass these)
        self.Cholesky_option = _Cholesky_option
        self.only_compute_fim_lower = _only_compute_fim_lower
        
        # model attribute to avoid rebuilding models
        self.model = pyo.ConcreteModel()  # Build empty model
        
        # May need this attribute for more complicated structures?
        # (i.e., no model rebuilding for large models with sequential)
        self._built_scenarios = False
    
    # Perform doe
    def run_doe(self, mod=None):
        # Start timer
        sp_timer = TicTocTimer()
        sp_timer.tic(msg=None)
        self.logger.info("Beginning experimental optimization.")

        # Model is none, set it to self.model
        if mod is None:
            mod = self.model
        
        # ToDo: potentially work with this for more complicated models
        # build the large DOE pyomo model
        if not self._built_scenarios:
            self.create_doe_model(mod=mod)

        # Add the objective function to the model
        self.create_objective_function(mod=mod)
        
        # Solve the square problem first to initialize the fim and
        # sensitivity constraints
        # Deactivate object and objective constraints, and fix design variables
        mod.Obj.deactivate()
        mod.obj_cons.deactivate()
        for comp, _ in mod.scenario_blocks[0].experiment_inputs.items():
            comp.fix()
        
        mod.dummy_obj = pyo.Objective(expr=0, sense=pyo.minimize)
        self.solver.solve(self.model, tee=self.tee)
        mod.dummy_obj.deactivate()
        
        # Reactivate objective and unfix experimental design decisions
        for comp, _ in mod.scenario_blocks[0].experiment_inputs.items():
            comp.unfix()
        mod.Obj.activate()
        mod.obj_cons.activate()
        
        # ToDo: add a ``get FIM from model`` function
        # If the model has L_ele, initialize it with the solved FIM
        if hasattr(mod, 'L_ele'):
            # Get the FIM values --> ToDo: add this as a function
            fim_vals = [pyo.value(mod.fim[i, j]) for i in mod.parameter_names for j in mod.parameter_names]
            fim_np = np.array(fim_vals).reshape((len(mod.parameter_names), len(mod.parameter_names)))
            
            L_vals_sq = np.linalg.cholesky(fim_np)
            for i, c in enumerate(mod.parameter_names):
                for j, d in enumerate(mod.parameter_names):
                    mod.L_ele[c, d].value = L_vals_sq[i, j]
        
        # Solve the full model, which has now been initialized with the square solve
        self.solver.solve(mod, tee=self.tee)
        
        # Finish timing
        dT = sp_timer.toc(msg=None)
        self.logger.info("Succesfully optimized experiment.\nElapsed time: %0.1f seconds" % dT)
    
    # Perform multi-experiment doe (sequential, or ``greedy`` approach)
    def run_multi_doe_sequential(self, N_exp=1):
        raise NotImplementedError(
            "Multipled experiment optimization not yet supported."
        )
    
    # Perform multi-experiment doe (simultaneous, optimal approach)
    def run_multi_doe_simultaneous(self, N_exp=1):
        raise NotImplementedError(
            "Multipled experiment optimization not yet supported."
        )
    
    # Compute FIM for the DoE object
    def compute_FIM(self, mod=None, method='sequential'):
        """
        Computes the FIM for the experimental design that is
        initialized from the experiment`s ``get_labeled_model()``
        function.
        
        Parameters
        ----------
        mod: model to compute FIM, default: None, (self.compute_FIM_model)
        method: string to specify which method should be used
                options are ``kaug`` and ``sequential``
        
        """
        if mod is None:
            self.compute_FIM_model = self.experiment.get_labeled_model(**self.args).clone()
            mod = self.compute_FIM_model
        
        self.check_model_labels(mod=mod)
        
        # Check FIM input, if it exists. Otherwise, set the prior_FIM attribute
        if self.prior_FIM is None:
            self.prior_FIM = np.zeros((len(mod.unknown_parameters), len(mod.unknown_parameters)))
        else:
            self.check_model_FIM(FIM=self.prior_FIM)
        
        # ToDo: Decide where the FIM should be saved.
        if method == 'sequential':
            self._sequential_FIM(mod=mod)
            self._computed_FIM = self.seq_FIM
        elif method == 'kaug':
            self._kaug_FIM(mod=mod)
            self._computed_FIM = self.kaug_FIM
        else:
            raise ValueError('The method provided, {}, must be either `sequential` or `kaug`'.format(method))

    # Use a sequential method to get the FIM
    def _sequential_FIM(self, mod=None):
        """
        Used to compute the FIM using a sequential approach,
        solving the model consecutively under each of the
        finite difference scenarios to build the sensitivity
        matrix to subsequently compute the FIM.

        """
        # Build a singular model instance
        if mod is None:
            self.compute_FIM_model = self.experiment.get_labeled_model(**self.args).clone()
            mod = self.compute_FIM_model
        
        # Create suffix to keep track of parameter scenarios
        mod.parameter_scenarios = pyo.Suffix(
            direction=pyo.Suffix.LOCAL,
        )
        
        # Populate parameter scenarios, and scenario inds based on finite difference scheme
        if self.fd_formula == FiniteDifferenceStep.central:
            mod.parameter_scenarios.update((2*ind, k) for ind, k in enumerate(mod.unknown_parameters.keys()))
            mod.parameter_scenarios.update((2*ind + 1, k) for ind, k in enumerate(mod.unknown_parameters.keys()))
            mod.scenarios = range(len(mod.unknown_parameters) * 2)
        elif self.fd_formula in [FiniteDifferenceStep.forward, FiniteDifferenceStep.backward]:
            mod.parameter_scenarios.update((ind + 1, k) for ind, k in enumerate(mod.unknown_parameters.keys()))
            mod.scenarios = range(len(mod.unknown_parameters) + 1)
        else:
            # To-Do: add an error message for this as not being implemented yet
            pass
        
        measurement_vals = []
        # In a loop.....
        # Calculate measurement values for each scenario
        for s in mod.scenarios:
            param = mod.parameter_scenarios[s]
            
            # Perturbation to be (1 + diff) * param_value
            if self.fd_formula == FiniteDifferenceStep.central:
                diff = self.step * ((-1) ** s)  # Positive perturbation, even; negative, odd
            elif self.fd_formula == FiniteDifferenceStep.backward:
                diff = self.step * -1 * (s != 0)  # Backward always negative perturbation; 0 at s = 0
            elif self.fd_formula == FiniteDifferenceStep.forward:
                diff = self.step * (s != 0) # Forward always positive; 0 at s = 0
            else:
                raise DeveloperError(
                "Finite difference option not recognized. Please contact the developers as you should not see this error."
            )
                diff = 0
                pass
            
            # Update parameter values for the given finite difference scenario
            param.set_value(mod.unknown_parameters[param] * (1 + diff))
            
            param.pprint()
            
            # Simulate the model
            self.solver.solve(mod)
            
            # Extract the measurement values for the scenario and append
            measurement_vals.append([pyo.value(k) for k, v in mod.experiment_outputs.items()])
        
        # Use the measurement outputs to make the Q matrix
        measurement_vals_np = np.array(measurement_vals).T
        
        self.seq_jac = np.zeros((len(mod.experiment_outputs.items()), len(mod.unknown_parameters.items())))
        
        # Counting variable for loop
        i = 0
        
        # Loop over parameter values and grab correct columns for finite difference calculation
        
        for k, v in mod.unknown_parameters.items():
            curr_step = v * self.step
            
            if self.fd_formula == FiniteDifferenceStep.central:
                col_1 = 2*i
                col_2 = 2*i + 1
                curr_step *= 2
            elif self.fd_formula == FiniteDifferenceStep.forward:
                col_1 = i
                col_2 = 0
            elif self.fd_formula == FiniteDifferenceStep.backward:
                col_1 = 0
                col_2 = i
            
            k.pprint()
            print(curr_step)
            
            # If scale_nominal_param_value is False, v ** 0 = 1 (not scaled with parameter value)
            scale_factor = (1 / curr_step) * self.scale_constant_value * (v ** self.scale_nominal_param_value)
            
            # Calculate the column of the sensitivity matrix
            self.seq_jac[:, i] = (measurement_vals_np[:, col_1] - measurement_vals_np[:, col_2]) * scale_factor
            
            # Increment the count
            i += 1
        
        # ToDo: As more complex measurement error schemes are put in place, this needs to change
        # Add independent (non-correlated) measurement error for FIM calculation
        cov_y = np.zeros((len(mod.measurement_error), len(mod.measurement_error)))
        count = 0
        for k, v in mod.measurement_error.items():
            cov_y[count, count] = 1 / v
            count += 1
        
        # Compute and record FIM
        self.seq_FIM = self.seq_jac.T @ cov_y @ self.seq_jac + self.prior_FIM

    # Use kaug to get FIM
    def _kaug_FIM(self, mod=None):
        """
        Used to compute the FIM using kaug, a sensitivity-based
        approach that directly computes the FIM.
        
        Parameters
        ----------
        mod: model to compute FIM, default: None, (self.compute_FIM_model)

        """
        # Remake compute_FIM_model if model is None.
        # compute_FIM_model needs to be the right version for function to work.
        if mod is None:
            self.compute_FIM_model = self.experiment.get_labeled_model(**self.args).clone()
            mod = self.compute_FIM_model
        
        # add zero (dummy/placeholder) objective function
        if not hasattr(mod, 'Obj'):
            mod.Obj = pyo.Objective(expr=0, sense=pyo.minimize)

        # call k_aug get_dsdp function
        # Solve the square problem
        # Deactivate object and fix experimental design decisions to make square
        for comp, _ in mod.experiment_inputs.items():
            comp.fix()
        
        self.solver.solve(mod, tee=self.tee)

        # Probe the solved model for dsdp results (sensitivities s.t. parameters)
        params_dict = {k.name: v for k, v in mod.unknown_parameters.items()}
        params_names = list(params_dict.keys())

        dsdp_re, col = get_dsdp(
            mod, params_names, params_dict, tee=self.tee
        )

        # analyze result
        dsdp_array = dsdp_re.toarray().T

        # store dsdp returned
        dsdp_extract = []
        # get right lines from results
        measurement_index = []

        # loop over measurement variables and their time points
        for k, v in mod.experiment_outputs.items():
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
            for k, v in mod.unknown_parameters.items():
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
        cov_y = np.zeros((len(mod.measurement_error), len(mod.measurement_error)))
        count = 0
        for k, v in mod.measurement_error.items():
            cov_y[count, count] = 1 / v
            count += 1
        
        # ToDo: need to add a covariance matrix for measurements (sigma inverse)
        # i.e., cov_y = self.cov_y or mod.cov_y
        # Still deciding where this would be best.
        
        self.kaug_FIM = self.kaug_jac.T @ cov_y @ self.kaug_jac + self.prior_FIM

    # Create the DoE model (with ``scenarios`` from finite differencing scheme)
    def create_doe_model(self, mod=None):
        """
        Add equations to compute sensitivities, FIM, and objective.
        Builds the DoE model. Adds the scenarios, the sensitivity matrix
        Q, the FIM, as well as the objective function to the model.
        
        The function alters the ``mod`` input.
        
        In the single experiment case, ``mod`` will be self.model. In the 
        multi-experiment case, ``mod`` will be one experiment to be enumerated.
        
        Parameters
        ----------
        mod: model to add finite difference scenarios

        """
        if mod is None:
            mod = self.model

        # Developer recommendation: use the Cholesky decomposition for D-optimality
        # The explicit formula is available for benchmarking purposes and is NOT recommended
        if (
            self.only_compute_fim_lower
            and self.objective_option == ObjectiveLib.det
            and not self.Cholesky_option
        ):
            raise ValueError(
                "Cannot compute determinant with explicit formula if only_compute_fim_lower is True."
            )
        
        # Generate scenarios for finite difference formulae
        self._generate_scenario_blocks(mod=mod)
        
        # Set names for indexing sensitivity matrix (jacobian) and FIM
        scen_block_ind = min([k.name.split('.').index('scenario_blocks[0]') for k in mod.scenario_blocks[0].unknown_parameters.keys()])
        mod.parameter_names = pyo.Set(initialize=[".".join(k.name.split('.')[(scen_block_ind + 1):]) for k in mod.scenario_blocks[0].unknown_parameters.keys()])
        mod.output_names = pyo.Set(initialize=[".".join(k.name.split('.')[(scen_block_ind + 1):]) for k in mod.scenario_blocks[0].experiment_outputs.keys()])

        def identity_matrix(m, i, j):
            if i == j:
                return 1
            else:
                return 0

        ### Initialize the Jacobian if provided by the user

        # If the user provides an initial Jacobian, convert it to a dictionary
        if self.jac_initial is not None:
            dict_jac_initialize = {}
            for i, bu in enumerate(mod.output_names):
                for j, un in enumerate(mod.parameter_names):
                    # Jacobian is a numpy array, rows are experimental outputs, columns are unknown parameters
                    dict_jac_initialize[(bu, un)] = self.jac_initial[i][j]

        # Initialize the Jacobian matrix
        def initialize_jac(m, i, j):
            # If provided by the user, use the values now stored in the dictionary
            if self.jac_initial is not None:
                return dict_jac_initialize[(i, j)]
            # Otherwise initialize to 0.1 (which is an arbitrary non-zero value)
            else:
                # Add flag as this should never be reached.
                return 0.1

        mod.sensitivity_jacobian = pyo.Var(
            mod.output_names,
            mod.parameter_names,
            initialize=initialize_jac,
        )

        # Initialize the FIM
        if self.fim_initial is not None:
            dict_fim_initialize = {
                (bu, un): self.fim_initial[i][j]
                for i, bu in enumerate(mod.parameter_names)
                for j, un in enumerate(mod.parameter_names)
            }

        def initialize_fim(m, j, d):
            return dict_fim_initialize[(j, d)]

        if self.fim_initial is not None:
            mod.fim = pyo.Var(
                mod.parameter_names,
                mod.parameter_names,
                initialize=initialize_fim,
            )
        else:
            mod.fim = pyo.Var(
                mod.parameter_names,
                mod.parameter_names,
                initialize=identity_matrix,
            )

        # To-Do: Look into this functionality.....
        # if cholesky, define L elements as variables
        if self.Cholesky_option and self.objective_option == ObjectiveLib.det:

            # move the L matrix initial point to a dictionary
            if self.L_initial is not None:
                dict_cho = {
                    (bu, un): self.L_initial[i][j]
                    for i, bu in enumerate(mod.parameter_names)
                    for j, un in enumerate(mod.parameter_names)
                }

            # use the L dictionary to initialize L matrix
            def init_cho(m, i, j):
                return dict_cho[(i, j)]

            # Define elements of Cholesky decomposition matrix as Pyomo variables and either
            # Initialize with L in L_initial
            if self.L_initial is not None:
                mod.L_ele = pyo.Var(
                    mod.parameter_names,
                    mod.parameter_names,
                    initialize=init_cho,
                )
            # or initialize with the identity matrix
            else:
                mod.L_ele = pyo.Var(
                    mod.parameter_names,
                    mod.parameter_names,
                    initialize=identity_matrix,
                )

            # loop over parameter name
            for i, c in enumerate(mod.parameter_names):
                for j, d in enumerate(mod.parameter_names):
                    # fix the 0 half of L matrix to be 0.0
                    if i < j:
                        mod.L_ele[c, d].fix(0.0)
                    # Give LB to the diagonal entries
                    if self.L_LB:
                        if c == d:
                            mod.L_ele[c, d].setlb(self.L_LB)

        # jacobian rule
        def jacobian_rule(m, n, p):
            """
            m: Pyomo model
            n: experimental output
            p: unknown parameter
            """
            fd_step_mult = 1
            cuid = pyo.ComponentUID(n)
            param_ind = mod.parameter_names.data().index(p)
            
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

            param = mod.parameter_scenarios[max(s1, s2)]
            param_loc = pyo.ComponentUID(param).find_component_on(mod.scenario_blocks[0])
            param_val = mod.scenario_blocks[0].unknown_parameters[param_loc]
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
            for i, bu in enumerate(mod.parameter_names)
            for j, un in enumerate(mod.parameter_names)
        }

        def read_prior(m, i, j):
            return fim_initial_dict[(i, j)]

        mod.priorFIM = pyo.Expression(
            mod.parameter_names, mod.parameter_names, rule=read_prior
        )

        # Off-diagonal elements are symmetric, so only half of the off-diagonal elements need to be specified.
        def fim_rule(m, p, q):
            """
            m: Pyomo model
            p: unknown parameter
            q: unknown parameter
            """
            p_ind = list(mod.parameter_names).index(p)
            q_ind = list(mod.parameter_names).index(q)
            
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
                        / mod.scenario_blocks[0].measurement_error[pyo.ComponentUID(n).find_component_on(mod.scenario_blocks[0])]
                        * m.sensitivity_jacobian[n, p]
                        * m.sensitivity_jacobian[n, q]
                        for n in mod.output_names
                    )
                    + m.priorFIM[p, q]
                )

        mod.jacobian_constraint = pyo.Constraint(
            mod.output_names, mod.parameter_names, rule=jacobian_rule
        )
        mod.fim_constraint = pyo.Constraint(
            mod.parameter_names, mod.parameter_names, rule=fim_rule
        )

        if self.only_compute_fim_lower:
            # Fix the upper half of the FIM matrix elements to be 0.0.
            # This eliminates extra variables and ensures the expected number of
            # degrees of freedom in the optimization problem.
            for ind_p, p in enumerate(mod.parameter_names):
                for ind_q, q in enumerate(mod.parameter_names):
                    if ind_p < ind_q:
                        mod.fim[p, q].fix(0.0)
    
    # Create scenario block structure
    def _generate_scenario_blocks(self, mod=None):
        """
        Generates the modeling blocks corresponding to the scenarios for 
        the finite differencing scheme to compute the sensitivity jacobian
        to compute the FIM.
        
        The function alters the ``mod`` input.
        
        In the single experiment case, ``mod`` will be self.model. In the 
        multi-experiment case, ``mod`` will be one experiment to be enumerated.
        
        Parameters
        ----------
        mod: model to add finite difference scenarios
        """
        # If model is none, assume it is self.model
        if mod is None:
            mod = self.model

        # Generate initial scenario to populate unknown parameter values
        mod.base_model = self.experiment.get_labeled_model(**self.args).clone()
        
        # Check the model that labels are correct
        self.check_model_labels(mod=mod)

        # Gather lengths of label structures for later use in the model build process
        self.n_parameters = len(mod.base_model.unknown_parameters)
        self.n_measurement_error = len(mod.base_model.measurement_error)
        self.n_experiment_inputs = len(mod.base_model.experiment_inputs)
        self.n_experiment_outputs = len(mod.base_model.experiment_outputs)

        assert (self.n_measurement_error == self.n_experiment_outputs), "Number of experiment outputs, {}, and length of measurement error, {}, do not match. Please check model labeling.".format(self.n_experiment_outputs, self.n_measurement_error)

        self.logger.info('Experiment output and measurement error lengths match.')

        # Check that the user input FIM and Jacobian are the correct dimension
        if self.prior_FIM is not None:
            self.check_model_FIM(self.prior_FIM)
        else:
            self.prior_FIM = np.zeros((self.n_parameters, self.n_parameters))
        if self.fim_initial is not None:
            self.check_model_FIM(self.fim_initial)
        else:
            self.fim_initial = np.eye(self.n_parameters) + self.prior_FIM
        if self.jac_initial is not None:
            self.check_model_jac(self.jac_initial)
        else:
            self.jac_initial = np.eye(self.n_experiment_outputs, self.n_parameters)
        
        # Make a new Suffix to hold which scenarios are associated with parameters 
        mod.parameter_scenarios = pyo.Suffix(
            direction=pyo.Suffix.LOCAL,
        )
        
        # Populate parameter scenarios, and scenario inds based on finite difference scheme
        if self.fd_formula == FiniteDifferenceStep.central:
            mod.parameter_scenarios.update((2*ind, k) for ind, k in enumerate(mod.base_model.unknown_parameters.keys()))
            mod.parameter_scenarios.update((2*ind + 1, k) for ind, k in enumerate(mod.base_model.unknown_parameters.keys()))
            mod.scenarios = range(len(mod.base_model.unknown_parameters) * 2)
        elif self.fd_formula in [FiniteDifferenceStep.forward, FiniteDifferenceStep.backward]:
            mod.parameter_scenarios.update((ind + 1, k) for ind, k in enumerate(mod.base_model.unknown_parameters.keys()))
            mod.scenarios = range(len(mod.base_model.unknown_parameters) + 1)
        else:
            raise DeveloperError(
                "Finite difference option not recognized. Please contact the developers as you should not see this error."
            )

        # To-Do: Fix parameter values if they are not Params?

        # Run base model to get initialized model and check model function
        for comp, _ in mod.base_model.experiment_inputs.items():
            comp.fix()
        
        try:
            self.solver.solve(mod.base_model, tee=self.tee)
            self.logger.info('Model from experiment solved.')
        except:
            raise RuntimeError('Model from experiment did not solve appropriately. Make sure the model is well-posed.')
            

        for comp, _ in mod.base_model.experiment_inputs.items():
            comp.unfix()


        # Generate blocks for finite difference scenarios
        def build_block_scenarios(b, s):
            # Generate model for the finite difference scenario
            b.transfer_attributes_from(mod.base_model.clone())
            
            # Forward/Backward difference have a stationary case (s == 0), no parameter to perturb
            if self.fd_formula in [FiniteDifferenceStep.forward, FiniteDifferenceStep.backward]:
                if s == 0:
                    return
            
            param = mod.parameter_scenarios[s]
            
            # Grabbing the index of the parameter without the "base_model" precursor
            base_model_ind = param.name.split('.').index('base_model')
            param_loc = ".".join(param.name.split('.')[(base_model_ind + 1):])

            # Perturbation to be (1 + diff) * param_value
            if self.fd_formula == FiniteDifferenceStep.central:
                diff = self.step * ((-1) ** s)  # Positive perturbation, even; negative, odd
            elif self.fd_formula == FiniteDifferenceStep.backward:
                diff = self.step * -1  # Backward always negative perturbation
            elif self.fd_formula == FiniteDifferenceStep.forward:
                diff = self.step # Forward always positive
            else:
                # To-Do: add an error message for this as not being implemented yet
                diff = 0
                pass
            
            # Update parameter values for the given finite difference scenario
            pyo.ComponentUID(param_loc).find_component_on(b).set_value(mod.base_model.unknown_parameters[param] * (1 + diff))
        mod.scenario_blocks = pyo.Block(mod.scenarios, rule=build_block_scenarios)
        
        # To-Do: this might have to change if experiment inputs have 
        # a different value in the Suffix (currently it is the CUID)
        design_vars = [k for k, v in mod.scenario_blocks[0].experiment_inputs.items()]
        
        # Add constraints to equate block design with global design:
        for ind, d in enumerate(design_vars):
            con_name = 'global_design_eq_con_' + str(ind)
            
            # Constraint rule for global design constraints
            def global_design_fixing(m, s):
                if s == 0:
                    return pyo.Constraint.Skip
                ref_design_var = mod.scenario_blocks[0].experiment_inputs[d]
                ref_design_var_loc = ".".join(ref_design_var.get_repr().split('.')[0:])
                block_design_var = pyo.ComponentUID(ref_design_var_loc).find_component_on(mod.scenario_blocks[s])
                return d == block_design_var
            setattr(mod, con_name, pyo.Constraint(mod.scenarios, rule=global_design_fixing))
        
        # Clean up the base model used to generate the scenarios
        mod.del_component(mod.base_model)
        
        # ToDo: consider this logic? Multi-block systems need something more fancy
        self._built_scenarios = True

    # Create objective function
    def create_objective_function(self, mod=None):
        """
        Generates the objective function as an expression and as a
        Pyomo Objective object
        
        The function alters the ``mod`` input.
        
        In the single experiment case, ``mod`` will be self.model. In the 
        multi-experiment case, ``mod`` will be one experiment to be enumerated.
        
        Parameters
        ----------
        mod: model to add finite difference scenarios
        """
        if mod is None:
            mod = self.model
        
        small_number = 1e-10
        
        # Make objective block for constraints connected to objective
        mod.obj_cons = pyo.Block()

        # Assemble the FIM matrix. This is helpful for initialization!
        fim_vals = [
            mod.fim[bu, un].value
            for i, bu in enumerate(mod.parameter_names)
            for j, un in enumerate(mod.parameter_names)
        ]
        fim = np.array(fim_vals).reshape(len(mod.parameter_names), len(mod.parameter_names))

        ### Initialize the Cholesky decomposition matrix
        if self.Cholesky_option and self.objective_option == ObjectiveLib.det:

            # Calculate the eigenvalues of the FIM matrix
            eig = np.linalg.eigvals(fim)

            # If the smallest eigenvalue is (practically) negative, add a diagonal matrix to make it positive definite
            small_number = 1e-10
            if min(eig) < small_number:
                fim = fim + np.eye(len(mod.parameter_names)) * (small_number - min(eig))

            # Compute the Cholesky decomposition of the FIM matrix
            L = np.linalg.cholesky(fim)

            # Initialize the Cholesky matrix
            for i, c in enumerate(mod.parameter_names):
                for j, d in enumerate(mod.parameter_names):
                    mod.L_ele[c, d].value = L[i, j]

        def cholesky_imp(m, c, d):
            """
            Calculate Cholesky L matrix using algebraic constraints
            """
            # If the row is greater than or equal to the column, we are in the
            # lower traingle region of the L and FIM matrices.
            # This region is where our equations are well-defined.
            if list(mod.parameter_names).index(c) >= list(mod.parameter_names).index(d):
                return mod.fim[c, d] == sum(
                    mod.L_ele[c, mod.parameter_names.at(k + 1)]
                    * mod.L_ele[d, mod.parameter_names.at(k + 1)]
                    for k in range(list(mod.parameter_names).index(d) + 1)
                )
            else:
                # This is the empty half of L above the diagonal
                return pyo.Constraint.Skip

        def trace_calc(m):
            """
            Calculate FIM elements. Can scale each element with 1000 for performance
            """
            return mod.trace == sum(mod.fim[j, j] for j in mod.parameter_names)

        def det_general(m):
            r"""Calculate determinant. Can be applied to FIM of any size.
            det(A) = \sum_{\sigma in \S_n} (sgn(\sigma) * \Prod_{i=1}^n a_{i,\sigma_i})
            Use permutation() to get permutations, sgn() to get signature
            """
            r_list = list(range(len(mod.parameter_names)))
            # get all permutations
            object_p = permutations(r_list)
            list_p = list(object_p)

            # generate a name_order to iterate \sigma_i
            det_perm = 0
            for i in range(len(list_p)):
                name_order = []
                x_order = list_p[i]
                # sigma_i is the value in the i-th position after the reordering \sigma
                for x in range(len(x_order)):
                    for y, element in enumerate(mod.parameter_names):
                        if x_order[x] == y:
                            name_order.append(element)

            # det(A) = sum_{\sigma \in \S_n} (sgn(\sigma) * \Prod_{i=1}^n a_{i,\sigma_i})
            det_perm = sum(
                self._sgn(list_p[d])
                * sum(
                    mod.fim[each, name_order[b]]
                    for b, each in enumerate(mod.parameter_names)
                )
                for d in range(len(list_p))
            )
            return mod.det == det_perm

        if self.Cholesky_option and self.objective_option == ObjectiveLib.det:
            mod.obj_cons.cholesky_cons = pyo.Constraint(
                mod.parameter_names, mod.parameter_names, rule=cholesky_imp
            )
            mod.Obj = pyo.Objective(
                expr=2 * sum(pyo.log10(mod.L_ele[j, j]) for j in mod.parameter_names),
                sense=pyo.maximize,
            )

        elif self.objective_option == ObjectiveLib.det:
            # if not cholesky but determinant, calculating det and evaluate the OBJ with det
            mod.det = pyo.Var(initialize=np.linalg.det(fim), bounds=(small_number, None))
            mod.obj_cons.det_rule = pyo.Constraint(rule=det_general)
            mod.Obj = pyo.Objective(expr=pyo.log10(mod.det), sense=pyo.maximize)

        elif self.objective_option == ObjectiveLib.trace:
            # if not determinant or cholesky, calculating the OBJ with trace
            mod.trace = pyo.Var(initialize=np.trace(fim), bounds=(small_number, None))
            mod.obj_cons.trace_rule = pyo.Constraint(rule=trace_calc)
            mod.Obj = pyo.Objective(expr=pyo.log10(mod.trace), sense=pyo.maximize)

        elif self.objective_option == ObjectiveLib.zero:
            # add dummy objective function
            mod.Obj = pyo.Objective(expr=0)
        else:
            # something went wrong!
            raise DeveloperError(
                "Objective option not recognized. Please contact the developers as you should not see this error."
            )

    # Check to see if the model has all the required suffixes
    def check_model_labels(self, mod=None):
        """
        Checks if the model contains the necessary suffixes for the
        DoE model to be constructed automatically.
        
        Parameters
        ----------
        mod: model for suffix checking, Default: None, (self.model)
        """
        if mod is None:
            mod = self.model.base_model
        
        # Check that experimental outputs exist
        try:
            outputs = [k.name for k, v in mod.experiment_outputs.items()]
        except:
            RuntimeError(
                'Experiment model does not have suffix ' + '"experiment_outputs".'
            )

        # Check that experimental inputs exist
        try:
            outputs = [k.name for k, v in mod.experiment_inputs.items()]
        except:
            RuntimeError(
                'Experiment model does not have suffix ' + '"experiment_inputs".'
            )

        # Check that unknown parameters exist
        try:
            outputs = [k.name for k, v in mod.unknown_parameters.items()]
        except:
            RuntimeError(
                'Experiment model does not have suffix ' + '"unknown_parameters".'
            )
    
        # Check that measurement errors exist
        try:
            outputs = [k.name for k, v in mod.measurement_error.items()]
        except:
            RuntimeError(
                'Experiment model does not have suffix ' + '"measurement_error".'
            )
        
        self.logger.info('Model has expected labels.')
           
    # Check the FIM shape against what is expected from the model.
    def check_model_FIM(self, FIM=None):
        """
        Checks if the specified matrix, FIM, matches the shape expected
        from the model. This method should only be called after the
        model has been probed for the length of the unknown parameter,
        experiment input, experiment output, and measurement error 
        has been stored to the object. 
        
        Parameters
        ----------
        mod: model for suffix checking, Default: None, (self.model)
        """
        assert FIM.shape == (self.n_parameters, self.n_parameters), "Shape of FIM provided should be n_parameters x n_parameters, or {}, FIM provided has shape: {}".format((self.n_parameters, self.n_parameters), FIM.shape)

        self.logger.info('FIM provided matches expected dimensions from model.')
    
    # Check the jacobian shape against what is expected from the model.
    def check_model_jac(self, jac=None):
        assert jac.shape == (self.n_experiment_outputs, self.n_parameters), "Shape of Jacobian provided should be n_experiment_outputs x n_parameters, or {}, Jacobian provided has shape: {}".format((self.n_experiment_outputs, self.n_parameters), jac.shape)

        self.logger.info('Jacobian provided matches expected dimensions from model.')

    # Update the FIM for the specified model
    def update_FIM_prior(self, mod=None, FIM=None):
        """
        Updates the prior FIM on the model object. This may be useful when
        running a loop and the user doesn't want to rebuild the model
        because it is expensive to build/initialize.
        
        Parameters
        ----------
        mod: model where FIM prior is to be updated, Default: None, (self.model)
        FIM: 2D np array to be the new FIM prior, Default: None
        """
        if mod is None:
            mod = self.model
        
        # Check FIM input
        if FIM is None:
            raise ValueError('FIM input for update_FIM_prior must be a 2D, square numpy array.')

        assert hasattr(mod, 'fim'), '``fim`` is not defined on the model provided. Please build the model first.'

        self.check_model_FIM(mod, FIM)

        # Update FIM prior
        for ind1, p1 in enumerate(mod.parameter_names):
            for ind2, p2 in enumerate(mod.parameter_names):
                mod.prior_FIM[p1, p2].set_value(FIM[ind1, ind2]) 

        self.logger.info('FIM prior has been updated.')
    
    # ToDo: Add an update function for the parameter values? --> closed loop parameter estimation?
    # Or leave this to the user?????
    def udpate_unknown_parameter_values(self, mod=None, param_vals=None):
        return

    # Rescale FIM (a scaling function to help rescale FIM from parameter values)
    def rescale_FIM(self, FIM, param_vals):
        """
        Rescales the FIM based on the input and parameter vals.
        It is assumed the parameter vals align with the FIM
        dimensions such that (1, i) corresponds to the i-th
        column or row of the FIM.
        
        Parameters
        ----------
        FIM: 2D numpy array to be scaled
        param_vals: scaling factors for the parameters

        """
        if isinstance(param_vals, list):
            param_vals = np.array([param_vals, ])
        elif isinstance(param_vals, np.ndarray):
            if len(param_vals.shape) > 2 or ((len(param_vals.shape) == 2) and (param_vals.shape[0] != 1)):
                raise ValueError('param_vals should be a vector of dimensions (1, n_params). The shape you provided is {}.'.format(param_vals.shape))
            if len(param_vals.shape) == 1:
                param_vals = np.array([param_vals, ])
        scaling_mat = (1 / param_vals).transpose().dot((1 / param_vals))
        scaled_FIM = np.multiply(FIM, scaling_mat)
        return scaled_FIM

    # Evaluates FIM and statistics for a full factorial space (same as run_grid_search)
    def compute_FIM_full_factorial(self, design_ranges=None, method='sequential'):
        """
        Will run a simulation-based full factorial exploration of
        the experimental input space (i.e., a ``grid search`` or
        ``parameter sweep``) to understand how the FIM metrics
        change as a function of the experimental design space.
        
        Parameters
        ----------
        mod: model to perform the full factorial exploration on
        design_ranges: dict of lists, of the form {<var_name>: [start, stop, numsteps]}
        method: string to specify which method should be used
                options are ``kaug`` and ``sequential``

        """
        # Start timer
        sp_timer = TicTocTimer()
        sp_timer.tic(msg=None)
        self.logger.info("Beginning Full Factorial Design.")

        # Make new model for factorial design
        self.factorial_model = self.experiment.get_labeled_model(**self.args).clone()
        mod = self.factorial_model
        
        # Permute the inputs to be aligned with the experiment input indicies
        design_ranges_enum = {k: np.linspace(*v) for k, v in design_ranges.items()}
        design_map = {ind: (k[0].name, k[0]) for ind, k in enumerate(mod.experiment_inputs.items())}
        
        # Make the full space
        try:
            valid_inputs = 0
            des_ranges = []
            for k,v in design_map.items():
                if v[0] in design_ranges_enum.keys():
                    des_ranges.append(design_ranges_enum[v[0]])
                    valid_inputs += 1
            assert (valid_inputs > 0)
            
            factorial_points = product(*des_ranges)
        except:
            raise ValueError('Design ranges keys must be a subset of experimental design names.')
        
        # ToDo: Add more objetive types? i.e., modified-E; G-opt; V-opt; etc?
        # ToDo: Also, make this a result object, or more user friendly.
        fim_factorial_results = {k.name: [] for k, v in mod.experiment_inputs.items()}
        fim_factorial_results.update({'log D-opt': [], 'log A-opt': [], 'log E-opt': [], 'solve_time': [], })
        
        succeses = 0
        failures = 0
        total_points = np.prod(np.array([len(v) for k, v in design_ranges_enum.items()]))
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
                curr_point = succeses + failures + 1

                # Logging information for each run
                self.logger.info("This is run %s out of %s.", curr_point, total_points)
                
                # Attempt the FIM computation
                self.compute_FIM(mod=mod, method=method)
                succeses += 1
                
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
                        sum(time_set) / (curr_point) * (total_points - curr_point + 1), 2
                    ),
                )
            except:
                self.logger.warning(
                    ":::::::::::Warning: Cannot converge this run.::::::::::::"
                )
                failures += 1
                self.logger.warning("failed count:", failures)
                
                self._computed_FIM = np.zeros(self.prior_FIM.shape)
                iter_timer.tic(msg=None)
            
            FIM = self._computed_FIM
            
            # Compute and record metrics on FIM
            D_opt = np.log10(np.linalg.det(FIM))
            A_opt = np.log10(np.trace(FIM))
            E_opt = np.log10(min(np.linalg.eig(FIM)[0]))
            
            # Append the values for each of the experiment inputs
            for k, v in mod.experiment_inputs.items():
                fim_factorial_results[k.name].append(pyo.value(k))
            
            fim_factorial_results['log D-opt'].append(D_opt)
            fim_factorial_results['log A-opt'].append(A_opt)
            fim_factorial_results['log E-opt'].append(E_opt)
            fim_factorial_results['solve_time'].append(time_set[-1])
        
        self.fim_factorial_results = fim_factorial_results
        
        return self.fim_factorial_results
        # ToDo: add automated figure drawing as it was before (perhaps reuse the code)
        


##############################
#  Below is deprecated code  #
##############################

class DesignOfExperiments:
    def __init__(
        self,
        param_init,
        design_vars,
        measurement_vars,
        create_model,
        solver=None,
        prior_FIM=None,
        discretize_model=None,
        args=None,
        logger_level=logging.INFO,
        only_compute_fim_lower=True,
    ):
        """
        This package enables model-based design of experiments analysis with Pyomo.
        Both direct optimization and enumeration modes are supported.
        NLP sensitivity tools, e.g.,  sipopt and k_aug, are supported to accelerate analysis via enumeration.
        It can be applied to dynamic models, where design variables are controlled throughout the experiment.

        Parameters
        ----------
        param_init:
            A  ``dictionary`` of parameter names and values.
            If they defined as indexed Pyomo variable, put the variable name and index, such as 'theta["A1"]'.
        design_vars:
            A ``DesignVariables`` which contains the Pyomo variable names and their corresponding indices
            and bounds for experiment degrees of freedom
        measurement_vars:
            A ``MeasurementVariables`` which contains the Pyomo variable names and their corresponding indices and
            bounds for experimental measurements
        create_model:
            A Python ``function`` that returns a Concrete Pyomo model, similar to the interface for ``parmest``
        solver:
            A ``solver`` object that User specified, default=None.
            If not specified, default solver is IPOPT MA57.
        prior_FIM:
            A 2D numpy array containing Fisher information matrix (FIM) for prior experiments.
            The default None means there is no prior information.
        discretize_model:
            A user-specified ``function`` that discretizes the model. Only use with Pyomo.DAE, default=None
        args:
            Additional arguments for the create_model function.
        logger_level:
            Specify the level of the logger. Change to logging.DEBUG for all messages.
        only_compute_fim_lower:
            If True, only the lower triangle of the FIM is computed. Default is True.
        """

        # parameters
        if not isinstance(param_init, collections.abc.Mapping):
            raise ValueError("param_init should be a dictionary.")
        self.param = param_init
        # design variable name
        self.design_name = design_vars.variable_names
        self.design_vars = design_vars
        self.create_model = create_model

        # check if create model function conforms to the original
        # Pyomo.DoE interface
        model_option_arg = (
            "model_option" in inspect.getfullargspec(self.create_model).args
        )
        mod_arg = "mod" in inspect.getfullargspec(self.create_model).args
        if model_option_arg and mod_arg:
            self._original_create_model_interface = True
        else:
            self._original_create_model_interface = False

        if args is None:
            args = {}
        self.args = args

        # create the measurement information object
        self.measurement_vars = measurement_vars
        self.measure_name = self.measurement_vars.variable_names

        if (
            self.measurement_vars.variable_names is None
            or not self.measurement_vars.variable_names
        ):
            raise ValueError(
                "There are no measurement variables. Check for a modeling mistake."
            )

        # check if user-defined solver is given
        if solver:
            self.solver = solver
        # if not given, use default solver
        else:
            self.solver = self._get_default_ipopt_solver()

        # check if discretization is needed
        self.discretize_model = discretize_model

        # check if there is prior info
        if prior_FIM is None:
            self.prior_FIM = np.zeros((len(self.param), len(self.param)))
        else:
            self.prior_FIM = prior_FIM
        self._check_inputs()

        # if print statements
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logger_level)

        self.only_compute_fim_lower = only_compute_fim_lower

    def _check_inputs(self):
        """
        Check if the prior FIM is N*N matrix, where N is the number of parameter
        """
        if self.prior_FIM is not None:
            if np.shape(self.prior_FIM)[0] != np.shape(self.prior_FIM)[1]:
                raise ValueError("Found wrong prior information matrix shape.")
            elif np.shape(self.prior_FIM)[0] != len(self.param):
                raise ValueError("Found wrong prior information matrix shape.")

    def stochastic_program(
        self,
        if_optimize=True,
        objective_option="det",
        scale_nominal_param_value=False,
        scale_constant_value=1,
        optimize_opt=None,
        if_Cholesky=False,
        L_LB=1e-7,
        L_initial=None,
        jac_initial=None,
        fim_initial=None,
        formula="central",
        step=0.001,
        tee_opt=True,
    ):
        """
        Optimize DOE problem with design variables being the decisions.
        The DOE model is formed invasively and all scenarios are computed simultaneously.
        The function will first run a square problem with design variable being fixed at
        the given initial points (Objective function being 0), then a square problem with
        design variables being fixed at the given initial points (Objective function being Design optimality),
        and then unfix the design variable and do the optimization.

        Parameters
        ----------
        if_optimize:
            if true, continue to do optimization. else, just run square problem with given design variable values
        objective_option:
            choose from the ObjectiveLib enum,
            "det": maximizing the determinant with ObjectiveLib.det,
            "trace": or the trace of the FIM with ObjectiveLib.trace
        scale_nominal_param_value:
            if True, the parameters are scaled by its own nominal value in param_init
        scale_constant_value:
            scale all elements in Jacobian matrix, default is 1.
        optimize_opt:
            A dictionary, keys are design variables, values are True or False deciding if this design variable will be optimized as DOF or not
        if_Cholesky:
            if True, Cholesky decomposition is used for Objective function for D-optimality.
        L_LB:
            L is the Cholesky decomposition matrix for FIM, i.e. FIM = L*L.T.
            L_LB is the lower bound for every element in L.
            if FIM is positive definite, the diagonal element should be positive, so we can set a LB like 1E-10
        L_initial:
            initialize the L
        jac_initial:
            a matrix used to initialize jacobian matrix
        fim_initial:
            a matrix used to initialize FIM matrix
        formula:
            choose from "central", "forward", "backward",
            which refers to the Enum FiniteDifferenceStep.central, .forward, or .backward
        step:
            Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001
        tee_opt:
            if True, IPOPT console output is printed

        Returns
        -------
        analysis_square: result summary of the square problem solved at the initial point
        analysis_optimize: result summary of the optimization problem solved

        """
        # store inputs in object
        self.design_values = self.design_vars.variable_names_value
        self.optimize = if_optimize
        self.objective_option = ObjectiveLib(objective_option)
        self.scale_nominal_param_value = scale_nominal_param_value
        self.scale_constant_value = scale_constant_value
        self.Cholesky_option = if_Cholesky
        self.L_LB = L_LB
        self.L_initial = L_initial
        self.jac_initial = jac_initial
        self.fim_initial = fim_initial
        self.formula = FiniteDifferenceStep(formula)
        self.step = step
        self.tee_opt = tee_opt

        # calculate how much the FIM element is scaled by a constant number
        # FIM = Jacobian.T@Jacobian, the FIM is scaled by squared value the Jacobian is scaled
        self.fim_scale_constant_value = self.scale_constant_value**2

        # Start timer
        sp_timer = TicTocTimer()
        sp_timer.tic(msg=None)

        # build the large DOE pyomo model
        m = self._create_doe_model(no_obj=True)

        # solve model, achieve results for square problem, and results for optimization problem
        m, analysis_square = self._compute_stochastic_program(m, optimize_opt)

        if self.optimize:
            # If set to optimize, solve the optimization problem (with degrees of freedom)
            analysis_optimize = self._optimize_stochastic_program(m)
            dT = sp_timer.toc(msg=None)
            self.logger.info("elapsed time: %0.1f seconds" % dT)
            # Return both square problem and optimization problem results
            return analysis_square, analysis_optimize

        else:
            dT = sp_timer.toc(msg=None)
            self.logger.info("elapsed time: %0.1f seconds" % dT)
            # Return only square problem results
            return analysis_square

    def _compute_stochastic_program(self, m, optimize_option):
        """
        Solve the stochastic program problem as a square problem.
        """

        # Solve square problem first
        # result_square: solver result
        result_square = self._solve_doe(m, fix=True, opt_option=optimize_option)

        # extract Jac
        jac_square = self._extract_jac(m)

        # create result object
        analysis_square = FisherResults(
            list(self.param.keys()),
            self.measurement_vars,
            jacobian_info=None,
            all_jacobian_info=jac_square,
            prior_FIM=self.prior_FIM,
            scale_constant_value=self.scale_constant_value,
        )
        # for simultaneous mode, FIM and Jacobian are extracted with extract_FIM()
        analysis_square.result_analysis(result=result_square)

        analysis_square.model = m

        self.analysis_square = analysis_square
        return m, analysis_square

    def _optimize_stochastic_program(self, m):
        """
        Solve the stochastic program problem as an optimization problem.
        """

        m = self._add_objective(m)

        result_doe = self._solve_doe(m, fix=False)

        # extract Jac
        jac_optimize = self._extract_jac(m)

        # create result object
        analysis_optimize = FisherResults(
            list(self.param.keys()),
            self.measurement_vars,
            jacobian_info=None,
            all_jacobian_info=jac_optimize,
            prior_FIM=self.prior_FIM,
        )
        # for simultaneous mode, FIM and Jacobian are extracted with extract_FIM()
        analysis_optimize.result_analysis(result=result_doe)
        analysis_optimize.model = m

        return analysis_optimize

    def compute_FIM(
        self,
        mode="direct_kaug",
        FIM_store_name=None,
        specified_prior=None,
        tee_opt=True,
        scale_nominal_param_value=False,
        scale_constant_value=1,
        store_output=None,
        read_output=None,
        extract_single_model=None,
        formula="central",
        step=0.001,
        only_compute_fim_lower=False,
    ):
        """
        This function calculates the Fisher information matrix (FIM) using sensitivity information obtained
        from two possible modes (defined by the CalculationMode Enum):

            1.  sequential_finite: sequentially solve square problems and use finite difference approximation
            2.  direct_kaug: solve a single square problem then extract derivatives using NLP sensitivity theory

        Parameters
        ----------
        mode:
            supports CalculationMode.sequential_finite or CalculationMode.direct_kaug
        FIM_store_name:
            if storing the FIM in a .csv or .txt, give the file name here as a string.
        specified_prior:
            a 2D numpy array providing alternate prior matrix, default is no prior.
        tee_opt:
            if True, IPOPT console output is printed
        scale_nominal_param_value:
            if True, the parameters are scaled by its own nominal value in param_init
        scale_constant_value:
            scale all elements in Jacobian matrix, default is 1.
        store_output:
            if storing the output (value stored in Var 'output_record') as a pickle file, give the file name here as a string.
        read_output:
            if reading the output (value for Var 'output_record') as a pickle file, give the file name here as a string.
        extract_single_model:
            if True, the solved model outputs for each scenario are all recorded as a .csv file.
            The output file uses the name AB.csv, where string A is store_output input, B is the index of scenario.
            scenario index is the number of the scenario outputs which is stored.
        formula:
            choose from the Enum FiniteDifferenceStep.central, .forward, or .backward.
            This option is only used for CalculationMode.sequential_finite mode.
        step:
            Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001

        Returns
        -------
        FIM_analysis: result summary object of this solve
        """

        # save inputs in object
        self.design_values = self.design_vars.variable_names_value
        self.scale_nominal_param_value = scale_nominal_param_value
        self.scale_constant_value = scale_constant_value
        self.formula = FiniteDifferenceStep(formula)
        self.mode = CalculationMode(mode)
        self.step = step

        # This method only solves square problem
        self.optimize = False
        # Set the Objective Function to 0 helps solve square problem quickly
        self.objective_option = ObjectiveLib.zero
        self.tee_opt = tee_opt

        self.FIM_store_name = FIM_store_name
        self.specified_prior = specified_prior

        # calculate how much the FIM element is scaled by a constant number
        # As FIM~Jacobian.T@Jacobian, FIM is scaled twice the number the Q is scaled
        self.fim_scale_constant_value = self.scale_constant_value**2

        square_timer = TicTocTimer()
        square_timer.tic(msg=None)
        if self.mode == CalculationMode.sequential_finite:
            FIM_analysis = self._sequential_finite(
                read_output, extract_single_model, store_output
            )

        elif self.mode == CalculationMode.direct_kaug:
            FIM_analysis = self._direct_kaug()

        dT = square_timer.toc(msg=None)
        self.logger.info("elapsed time: %0.1f seconds" % dT)

        return FIM_analysis

    def _sequential_finite(self, read_output, extract_single_model, store_output):
        """Sequential_finite mode uses Pyomo Block to evaluate the sensitivity information."""

        # if measurements are provided
        if read_output:
            with open(read_output, "rb") as f:
                output_record = pickle.load(f)
                f.close()
            jac = self._finite_calculation(output_record)

        # if measurements are not provided
        else:
            mod = self._create_block()

            # dict for storing model outputs
            output_record = {}

            # Deactivate any existing objective functions
            for obj in mod.component_objects(pyo.Objective):
                obj.deactivate()

            # add zero (dummy/placeholder) objective function
            mod.Obj = pyo.Objective(expr=0, sense=pyo.minimize)

            # solve model
            square_result = self._solve_doe(mod, fix=True)

            # save model from optional post processing function
            self._square_model_from_compute_FIM = mod

            if extract_single_model:
                mod_name = store_output + ".csv"
                dataframe = extract_single_model(mod, square_result)
                dataframe.to_csv(mod_name)

            # loop over blocks for results
            for s in range(len(self.scenario_list)):
                # loop over measurement item and time to store model measurements
                output_iter = []

                # extract variable values
                for r in self.measure_name:
                    cuid = pyo.ComponentUID(r)
                    try:
                        var_up = cuid.find_component_on(mod.block[s])
                    except:
                        raise ValueError(
                            f"measurement {r} cannot be found in the model."
                        )
                    output_iter.append(pyo.value(var_up))

                output_record[s] = output_iter

                output_record["design"] = self.design_values

                if store_output:
                    f = open(store_output, "wb")
                    pickle.dump(output_record, f)
                    f.close()

            # calculate jacobian
            jac = self._finite_calculation(output_record)

            # return all models formed
            self.model = mod

            # Store the Jacobian information for access by users, not necessarily call result object to achieve jacobian information
            # It is the overall set of Jacobian information,
            # while in the result object the jacobian can be cut to achieve part of the FIM information
            self.jac = jac

        # Assemble and analyze results
        if self.specified_prior is None:
            prior_in_use = self.prior_FIM
        else:
            prior_in_use = self.specified_prior

        FIM_analysis = FisherResults(
            list(self.param.keys()),
            self.measurement_vars,
            jacobian_info=None,
            all_jacobian_info=jac,
            prior_FIM=prior_in_use,
            store_FIM=self.FIM_store_name,
            scale_constant_value=self.scale_constant_value,
        )

        return FIM_analysis

    def _direct_kaug(self):
        # create model
        if self._original_create_model_interface:
            mod = self.create_model(model_option=ModelOptionLib.parmest, **self.args)
        else:
            mod = self.create_model(**self.args)

        # discretize if needed
        if self.discretize_model is not None:
            mod = self.discretize_model(mod, block=False)

        # Deactivate any existing objective functions
        for obj in mod.component_objects(pyo.Objective):
            obj.deactivate()

        # add zero (dummy/placeholder) objective function
        mod.Obj = pyo.Objective(expr=0, sense=pyo.minimize)

        # set ub and lb to parameters
        for par in self.param.keys():
            cuid = pyo.ComponentUID(par)
            var = cuid.find_component_on(mod)
            var.setlb(self.param[par])
            var.setub(self.param[par])

        # generate parameter name list and value dictionary with index
        var_name = list(self.param.keys())

        # call k_aug get_dsdp function
        square_result = self._solve_doe(mod, fix=True)

        # save model from optional post processing function
        self._square_model_from_compute_FIM = mod

        dsdp_re, col = get_dsdp(
            mod, list(self.param.keys()), self.param, tee=self.tee_opt
        )

        # analyze result
        dsdp_array = dsdp_re.toarray().T
        self.dsdp = dsdp_array
        self.dsdp = col
        # store dsdp returned
        dsdp_extract = []
        # get right lines from results
        measurement_index = []

        # loop over measurement variables and their time points
        for mname in self.measure_name:
            try:
                kaug_no = col.index(mname)
                measurement_index.append(kaug_no)
                # get right line of dsdp
                dsdp_extract.append(dsdp_array[kaug_no])
            except:
                # k_aug does not provide value for fixed variables
                self.logger.debug("The variable is fixed:  %s", mname)
                # produce the sensitivity for fixed variables
                zero_sens = np.zeros(len(self.param))
                # for fixed variables, the sensitivity are a zero vector
                dsdp_extract.append(zero_sens)

        # Extract and calculate sensitivity if scaled by constants or parameters.
        # Convert sensitivity to a dictionary
        jac = {}
        for par in self.param.keys():
            jac[par] = []

        for d in range(len(dsdp_extract)):
            for p, par in enumerate(self.param.keys()):
                # if scaled by parameter value or constant value
                sensi = dsdp_extract[d][p] * self.scale_constant_value
                if self.scale_nominal_param_value:
                    sensi *= self.param[par]
                jac[par].append(sensi)

        # check if another prior experiment FIM is provided other than the user-specified one
        if self.specified_prior is None:
            prior_in_use = self.prior_FIM
        else:
            prior_in_use = self.specified_prior

        # Assemble and analyze results
        FIM_analysis = FisherResults(
            list(self.param.keys()),
            self.measurement_vars,
            jacobian_info=None,
            all_jacobian_info=jac,
            prior_FIM=prior_in_use,
            store_FIM=self.FIM_store_name,
            scale_constant_value=self.scale_constant_value,
        )

        self.jac = jac
        self.mod = mod

        return FIM_analysis

    def _create_block(self):
        """
        Create a pyomo Concrete model and add blocks with different parameter perturbation scenarios.

        Returns
        -------
        mod: Concrete Pyomo model
        """

        # create scenario information for block scenarios
        scena_gen = ScenarioGenerator(
            parameter_dict=self.param, formula=self.formula, step=self.step
        )

        self.scenario_data = scena_gen.ScenarioData

        # a list of dictionary, each one is a parameter dictionary with perturbed parameter values
        self.scenario_list = self.scenario_data.scenario
        # dictionary, keys are parameter name, values are a list of scenario index where this parameter is perturbed.
        self.scenario_num = self.scenario_data.scena_num
        # dictionary, keys are parameter name, values are the perturbation step
        self.eps_abs = self.scenario_data.eps_abs
        self.scena_gen = scena_gen

        # Determine if create_model takes theta as an optional input
        pass_theta_to_initialize = (
            "theta" in inspect.getfullargspec(self.create_model).args
        )

        # Allow user to self-define complex design variables
        if self._original_create_model_interface:

            # Create a global model
            mod = pyo.ConcreteModel()

            if pass_theta_to_initialize:
                # Add model on block with theta values
                self.create_model(
                    mod=mod,
                    model_option=ModelOptionLib.stage1,
                    theta=self.param,
                    **self.args,
                )
            else:
                # Add model on block without theta values
                self.create_model(
                    mod=mod, model_option=ModelOptionLib.stage1, **self.args
                )

        else:
            # Create a global model
            mod = self.create_model(**self.args)

        # Set for block/scenarios
        mod.scenario = pyo.Set(initialize=self.scenario_data.scenario_indices)

        # Fix parameter values in the copy of the stage1 model (if they exist)
        for par in self.param:
            cuid = pyo.ComponentUID(par)
            var = cuid.find_component_on(mod)
            if var is not None:
                # Fix the parameter value
                # Otherwise, the parameter does not exist on the stage 1 model
                var.fix(self.param[par])

        def block_build(b, s):
            # create block scenarios
            # idea: check if create_model takes theta as an optional input, if so, pass parameter values to create_model

            if self._original_create_model_interface:
                if pass_theta_to_initialize:
                    # Grab the values of theta for this scenario/block
                    theta_initialize = self.scenario_data.scenario[s]
                    # Add model on block with theta values
                    self.create_model(
                        mod=b,
                        model_option=ModelOptionLib.stage2,
                        theta=theta_initialize,
                        **self.args,
                    )
                else:
                    # Otherwise add model on block without theta values
                    self.create_model(
                        mod=b, model_option=ModelOptionLib.stage2, **self.args
                    )

                # save block in a temporary variable
                mod_ = b
            else:
                # Add model on block
                if pass_theta_to_initialize:
                    # Grab the values of theta for this scenario/block
                    theta_initialize = self.scenario_data.scenario[s]
                    mod_ = self.create_model(theta=theta_initialize, **self.args)
                else:
                    mod_ = self.create_model(**self.args)

            # fix parameter values to perturbed values
            for par in self.param:
                cuid = pyo.ComponentUID(par)
                var = cuid.find_component_on(mod_)
                var.fix(self.scenario_data.scenario[s][par])

            if not self._original_create_model_interface:
                # for the "new"/"slim" interface, we need to add the block to the model
                return mod_

        mod.block = pyo.Block(mod.scenario, rule=block_build)

        # discretize the model
        if self.discretize_model is not None:
            mod = self.discretize_model(mod)

        # force design variables in blocks to be equal to global design values
        for name in self.design_name:

            def fix1(mod, s):
                cuid = pyo.ComponentUID(name)
                design_var_global = cuid.find_component_on(mod)
                design_var = cuid.find_component_on(mod.block[s])
                return design_var == design_var_global

            con_name = "con" + name
            mod.add_component(con_name, pyo.Constraint(mod.scenario, expr=fix1))

            # Add user-defined design variable bounds
            cuid = pyo.ComponentUID(name)
            design_var_global = cuid.find_component_on(mod)
            # Set the lower and upper bounds of the design variables
            design_var_global.setlb(self.design_vars.lower_bounds[name])
            design_var_global.setub(self.design_vars.upper_bounds[name])

        return mod

    def _finite_calculation(self, output_record):
        """
        Calculate Jacobian for sequential_finite mode

        Parameters
        ----------
        output_record: a dict of outputs, keys are scenario names, values are a list of measurements values
        scena_gen: an object generated by Scenario_creator class

        Returns
        -------
        jac: Jacobian matrix, a dictionary, keys are parameter names, values are a list of jacobian values with respect to this parameter
        """
        # dictionary form of jacobian
        jac = {}

        # After collecting outputs from all scenarios, calculate sensitivity
        for para in self.param.keys():
            # extract involved scenario No. for each parameter from scenario class
            involved_s = self.scenario_data.scena_num[para]

            # each parameter has two involved scenarios
            s1 = involved_s[0]  # positive perturbation
            s2 = involved_s[1]  # negative perturbation
            list_jac = []
            for i in range(len(output_record[s1])):
                sensi = (
                    (output_record[s1][i] - output_record[s2][i])
                    / self.scenario_data.eps_abs[para]
                    * self.scale_constant_value
                )
                if self.scale_nominal_param_value:
                    sensi *= self.param[para]
                list_jac.append(sensi)
            # get Jacobian dict, keys are parameter name, values are sensitivity info
            jac[para] = list_jac

        return jac

    def _extract_jac(self, m):
        """
        Extract jacobian from the stochastic program

        Parameters
        ----------
        m: solved stochastic program model

        Returns
        -------
        JAC: the overall jacobian as a dictionary
        """
        # dictionary form of jacobian
        jac = {}
        # loop over parameters
        for p in self.param.keys():
            jac_para = []
            for res in m.measured_variables:
                jac_para.append(pyo.value(m.sensitivity_jacobian[p, res]))
            jac[p] = jac_para
        return jac

    def run_grid_search(
        self,
        design_ranges,
        mode="sequential_finite",
        tee_option=False,
        scale_nominal_param_value=False,
        scale_constant_value=1,
        store_name=None,
        read_name=None,
        store_optimality_as_csv=None,
        formula="central",
        step=0.001,
        post_processing_function=None,
    ):
        """
        Enumerate through full grid search for any number of design variables;
        solve square problems sequentially to compute FIMs.
        It calculates FIM with sensitivity information from two modes:

            1. sequential_finite: Calculates a one scenario model multiple times for multiple scenarios.
               Sensitivity info estimated by finite difference
            2. direct_kaug: calculate sensitivity by k_aug with direct sensitivity

        Parameters
        ----------
        design_ranges:
            a ``dict``, keys are design variable names,
            values are a list of design variable values to go over
        mode:
            choose from CalculationMode.sequential_finite, .direct_kaug.
        tee_option:
            if solver console output is made
        scale_nominal_param_value:
            if True, the parameters are scaled by its own nominal value in param_init
        scale_constant_value:
            scale all elements in Jacobian matrix, default is 1.
        store_name:
            a string of file name. If not None, store results with this name.
            It is a pickle file containing all measurement information after solving the
            model with perturbations.
            Since there are multiple experiments, results are numbered with a scalar number,
            and the result for one grid is 'store_name(count).csv' (count is the number of count).
        read_name:
            a string of file name. If not None, read result files.
            It should be a pickle file previously generated by store_name option.
            Since there are multiple experiments, this string should be the common part of all files;
            Real name of the file is "read_name(count)", where count is the number of the experiment.
        store_optimality_as_csv:
            if True, the design criterion values of grid search results stored with this file name as a csv
        formula:
            choose from FiniteDifferenceStep.central, .forward, or .backward.
            This option is only used for CalculationMode.sequential_finite.
        step:
            Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001
        post_processing_function:
            An optional function that executes after each solve of the grid search.
            The function should take one input: the Pyomo model. This could be a plotting function.
            Default is None.

        Returns
        -------
        figure_draw_object: a combined result object of class Grid_search_result
        """
        # Set the Objective Function to 0 helps solve square problem quickly
        self.objective_option = ObjectiveLib.zero
        self.store_optimality_as_csv = store_optimality_as_csv

        # calculate how much the FIM element is scaled
        self.fim_scale_constant_value = scale_constant_value**2

        # to store all FIM results
        result_combine = {}

        # all lists of values of each design variable to go over
        design_ranges_list = list(design_ranges.values())
        # design variable names to go over
        design_dimension_names = list(design_ranges.keys())

        # iteration 0
        count = 0
        failed_count = 0
        # how many sets of design variables will be run
        total_count = 1
        for rng in design_ranges_list:
            total_count *= len(rng)

        time_set = []  # record time for every iteration

        # generate combinations of design variable values to go over
        search_design_set = product(*design_ranges_list)

        # loop over design value combinations
        for design_set_iter in search_design_set:
            # generate the design variable dictionary needed for running compute_FIM
            # first copy value from design_values
            design_iter = self.design_vars.variable_names_value.copy()

            # convert to a list and cache
            list_design_set_iter = list(design_set_iter)

            # update the controlled value of certain time points for certain design variables
            for i, names in enumerate(design_dimension_names):
                if isinstance(names, str):
                    # if 'names' is simply a string, copy the new value
                    design_iter[names] = list_design_set_iter[i]
                elif isinstance(names, collections.abc.Sequence):
                    # if the element is a list, all design variables in this list share the same values
                    for n in names:
                        design_iter[n] = list_design_set_iter[i]
                else:
                    # otherwise just copy the value
                    # design_iter[names] = list(design_set_iter)[i]
                    raise NotImplementedError(
                        "You should not see this error message. Please report it to the Pyomo.DoE developers."
                    )

            self.design_vars.variable_names_value = design_iter
            iter_timer = TicTocTimer()
            self.logger.info("=======Iteration Number: %s =====", count + 1)
            self.logger.debug(
                "Design variable values of this iteration: %s", design_iter
            )
            iter_timer.tic(msg=None)
            # generate store name
            if store_name is None:
                store_output_name = None
            else:
                store_output_name = store_name + str(count)

            if read_name is not None:
                read_input_name = read_name + str(count)
            else:
                read_input_name = None

            # call compute_FIM to get FIM
            try:
                result_iter = self.compute_FIM(
                    mode=mode,
                    tee_opt=tee_option,
                    scale_nominal_param_value=scale_nominal_param_value,
                    scale_constant_value=scale_constant_value,
                    store_output=store_output_name,
                    read_output=read_input_name,
                    formula=formula,
                    step=step,
                )

                count += 1

                result_iter.result_analysis()

                # iteration time
                iter_t = iter_timer.toc(msg=None)
                time_set.append(iter_t)

                # give run information at each iteration
                self.logger.info("This is run %s out of %s.", count, total_count)
                self.logger.info(
                    "The code has run  %s seconds.", round(sum(time_set), 2)
                )
                self.logger.info(
                    "Estimated remaining time:  %s seconds",
                    round(
                        sum(time_set) / (count) * (total_count - count), 2
                    ),  # need to check this math... it gives a negative number for the final count
                )

                if post_processing_function is not None:
                    # Call the post processing function
                    post_processing_function(self._square_model_from_compute_FIM)

                # the combined result object are organized as a dictionary, keys are a tuple of the design variable values, values are a result object
                result_combine[tuple(design_set_iter)] = result_iter

            except:
                self.logger.warning(
                    ":::::::::::Warning: Cannot converge this run.::::::::::::"
                )
                count += 1
                failed_count += 1
                self.logger.warning("failed count:", failed_count)
                result_combine[tuple(design_set_iter)] = None

        # For user's access
        self.all_fim = result_combine

        # Create figure drawing object
        figure_draw_object = GridSearchResult(
            design_ranges_list,
            design_dimension_names,
            result_combine,
            store_optimality_name=store_optimality_as_csv,
        )

        self.logger.info("Overall wall clock time [s]:  %s", sum(time_set))

        return figure_draw_object

    def _create_doe_model(self, no_obj=True):
        """
        Add equations to compute sensitivities, FIM, and objective.

        Parameters
        -----------
        no_obj: if True, objective function is 0.

        Return
        -------
        model: the DOE model
        """

        # Developer recommendation: use the Cholesky decomposition for D-optimality
        # The explicit formula is available for benchmarking purposes and is NOT recommended
        if (
            self.only_compute_fim_lower
            and self.objective_option == ObjectiveLib.det
            and not self.Cholesky_option
        ):
            raise ValueError(
                "Cannot compute determinant with explicit formula if only_compute_fim_lower is True."
            )

        model = self._create_block()

        # variables for jacobian and FIM
        model.regression_parameters = pyo.Set(initialize=list(self.param.keys()))
        model.measured_variables = pyo.Set(initialize=self.measure_name)

        def identity_matrix(m, i, j):
            if i == j:
                return 1
            else:
                return 0

        ### Initialize the Jacobian if provided by the user

        # If the user provides an initial Jacobian, convert it to a dictionary
        if self.jac_initial is not None:
            dict_jac_initialize = {}
            for i, bu in enumerate(model.regression_parameters):
                for j, un in enumerate(model.measured_variables):
                    if isinstance(self.jac_initial, dict):
                        # Jacobian is a dictionary of arrays or lists where the key is the regression parameter name
                        dict_jac_initialize[(bu, un)] = self.jac_initial[bu][j]
                    elif isinstance(self.jac_initial, np.ndarray):
                        # Jacobian is a numpy array, rows are regression parameters, columns are measured variables
                        dict_jac_initialize[(bu, un)] = self.jac_initial[i][j]

        # Initialize the Jacobian matrix
        def initialize_jac(m, i, j):
            # If provided by the user, use the values now stored in the dictionary
            if self.jac_initial is not None:
                return dict_jac_initialize[(i, j)]
            # Otherwise initialize to 0.1 (which is an arbitrary non-zero value)
            else:
                return 0.1

        model.sensitivity_jacobian = pyo.Var(
            model.regression_parameters,
            model.measured_variables,
            initialize=initialize_jac,
        )

        if self.fim_initial is not None:
            dict_fim_initialize = {
                (bu, un): self.fim_initial[i][j]
                for i, bu in enumerate(model.regression_parameters)
                for j, un in enumerate(model.regression_parameters)
            }

        def initialize_fim(m, j, d):
            return dict_fim_initialize[(j, d)]

        if self.fim_initial is not None:
            model.fim = pyo.Var(
                model.regression_parameters,
                model.regression_parameters,
                initialize=initialize_fim,
            )
        else:
            model.fim = pyo.Var(
                model.regression_parameters,
                model.regression_parameters,
                initialize=identity_matrix,
            )

        # if cholesky, define L elements as variables
        if self.Cholesky_option and self.objective_option == ObjectiveLib.det:

            # move the L matrix initial point to a dictionary
            if self.L_initial is not None:
                dict_cho = {
                    (bu, un): self.L_initial[i][j]
                    for i, bu in enumerate(model.regression_parameters)
                    for j, un in enumerate(model.regression_parameters)
                }

            # use the L dictionary to initialize L matrix
            def init_cho(m, i, j):
                return dict_cho[(i, j)]

            # Define elements of Cholesky decomposition matrix as Pyomo variables and either
            # Initialize with L in L_initial
            if self.L_initial is not None:
                model.L_ele = pyo.Var(
                    model.regression_parameters,
                    model.regression_parameters,
                    initialize=init_cho,
                )
            # or initialize with the identity matrix
            else:
                model.L_ele = pyo.Var(
                    model.regression_parameters,
                    model.regression_parameters,
                    initialize=identity_matrix,
                )

            # loop over parameter name
            for i, c in enumerate(model.regression_parameters):
                for j, d in enumerate(model.regression_parameters):
                    # fix the 0 half of L matrix to be 0.0
                    if i < j:
                        model.L_ele[c, d].fix(0.0)
                    # Give LB to the diagonal entries
                    if self.L_LB:
                        if c == d:
                            model.L_ele[c, d].setlb(self.L_LB)

        # jacobian rule
        def jacobian_rule(m, p, n):
            """
            m: Pyomo model
            p: parameter
            n: response
            """
            cuid = pyo.ComponentUID(n)
            var_up = cuid.find_component_on(m.block[self.scenario_num[p][0]])
            var_lo = cuid.find_component_on(m.block[self.scenario_num[p][1]])
            if self.scale_nominal_param_value:
                return (
                    m.sensitivity_jacobian[p, n]
                    == (var_up - var_lo)
                    / self.eps_abs[p]
                    * self.param[p]
                    * self.scale_constant_value
                )
            else:
                return (
                    m.sensitivity_jacobian[p, n]
                    == (var_up - var_lo) / self.eps_abs[p] * self.scale_constant_value
                )

        # A constraint to calculate elements in Hessian matrix
        # transfer prior FIM to be Expressions
        fim_initial_dict = {
            (bu, un): self.prior_FIM[i][j]
            for i, bu in enumerate(model.regression_parameters)
            for j, un in enumerate(model.regression_parameters)
        }

        def read_prior(m, i, j):
            return fim_initial_dict[(i, j)]

        model.priorFIM = pyo.Expression(
            model.regression_parameters, model.regression_parameters, rule=read_prior
        )

        # The off-diagonal elements are symmetric, thus only half of the elements need to be calculated
        def fim_rule(m, p, q):
            """
            m: Pyomo model
            p: parameter
            q: parameter
            """

            if p > q:
                if self.only_compute_fim_lower:
                    return pyo.Constraint.Skip
                else:
                    return m.fim[p, q] == m.fim[q, p]
            else:
                return (
                    m.fim[p, q]
                    == sum(
                        1
                        / self.measurement_vars.variance[n]
                        * m.sensitivity_jacobian[p, n]
                        * m.sensitivity_jacobian[q, n]
                        for n in model.measured_variables
                    )
                    + m.priorFIM[p, q] * self.fim_scale_constant_value
                )

        model.jacobian_constraint = pyo.Constraint(
            model.regression_parameters, model.measured_variables, rule=jacobian_rule
        )
        model.fim_constraint = pyo.Constraint(
            model.regression_parameters, model.regression_parameters, rule=fim_rule
        )

        if self.only_compute_fim_lower:
            # Fix the upper half of the FIM matrix elements to be 0.0.
            # This eliminates extra variables and ensures the expected number of
            # degrees of freedom in the optimization problem.
            for p in model.regression_parameters:
                for q in model.regression_parameters:
                    if p > q:
                        model.fim[p, q].fix(0.0)

        return model

    def _add_objective(self, m):

        small_number = 1e-10

        # Assemble the FIM matrix. This is helpful for initialization!
        #
        # Suggestion from JS: "It might be more efficient to form the NP array in one shot
        # (from a list or using fromiter), and then reshaping to the 2-D matrix"
        #
        fim = np.zeros((len(self.param), len(self.param)))
        for i, bu in enumerate(m.regression_parameters):
            for j, un in enumerate(m.regression_parameters):
                # Copy value from Pyomo model into numpy array
                fim[i][j] = m.fim[bu, un].value

                # Set lower bound to ensure diagonal elements are (almost) non-negative
                # if i == j:
                #     m.fim[bu, un].setlb(-small_number)

        ### Initialize the Cholesky decomposition matrix
        if self.Cholesky_option and self.objective_option == ObjectiveLib.det:

            # Calculate the eigenvalues of the FIM matrix
            eig = np.linalg.eigvals(fim)

            # If the smallest eigenvalue is (practically) negative, add a diagonal matrix to make it positive definite
            small_number = 1e-10
            if min(eig) < small_number:
                fim = fim + np.eye(len(self.param)) * (small_number - min(eig))

            # Compute the Cholesky decomposition of the FIM matrix
            L = np.linalg.cholesky(fim)

            # Initialize the Cholesky matrix
            for i, c in enumerate(m.regression_parameters):
                for j, d in enumerate(m.regression_parameters):
                    m.L_ele[c, d].value = L[i, j]

        def cholesky_imp(m, c, d):
            """
            Calculate Cholesky L matrix using algebraic constraints
            """
            # If it is the left bottom half of L
            if list(self.param.keys()).index(c) >= list(self.param.keys()).index(d):
                return m.fim[c, d] == sum(
                    m.L_ele[c, list(self.param.keys())[k]]
                    * m.L_ele[d, list(self.param.keys())[k]]
                    for k in range(list(self.param.keys()).index(d) + 1)
                )
            else:
                # This is the empty half of L above the diagonal
                return pyo.Constraint.Skip

        def trace_calc(m):
            """
            Calculate FIM elements. Can scale each element with 1000 for performance
            """
            return m.trace == sum(m.fim[j, j] for j in m.regression_parameters)

        def det_general(m):
            r"""Calculate determinant. Can be applied to FIM of any size.
            det(A) = \sum_{\sigma in \S_n} (sgn(\sigma) * \Prod_{i=1}^n a_{i,\sigma_i})
            Use permutation() to get permutations, sgn() to get signature
            """
            r_list = list(range(len(m.regression_parameters)))
            # get all permutations
            object_p = permutations(r_list)
            list_p = list(object_p)

            # generate a name_order to iterate \sigma_i
            det_perm = 0
            for i in range(len(list_p)):
                name_order = []
                x_order = list_p[i]
                # sigma_i is the value in the i-th position after the reordering \sigma
                for x in range(len(x_order)):
                    for y, element in enumerate(m.regression_parameters):
                        if x_order[x] == y:
                            name_order.append(element)

            # det(A) = sum_{\sigma \in \S_n} (sgn(\sigma) * \Prod_{i=1}^n a_{i,\sigma_i})
            det_perm = sum(
                self._sgn(list_p[d])
                * sum(
                    m.fim[each, name_order[b]]
                    for b, each in enumerate(m.regression_parameters)
                )
                for d in range(len(list_p))
            )
            return m.det == det_perm

        if self.Cholesky_option and self.objective_option == ObjectiveLib.det:
            m.cholesky_cons = pyo.Constraint(
                m.regression_parameters, m.regression_parameters, rule=cholesky_imp
            )
            m.Obj = pyo.Objective(
                expr=2 * sum(pyo.log10(m.L_ele[j, j]) for j in m.regression_parameters),
                sense=pyo.maximize,
            )

        elif self.objective_option == ObjectiveLib.det:
            # if not cholesky but determinant, calculating det and evaluate the OBJ with det
            m.det = pyo.Var(initialize=np.linalg.det(fim), bounds=(small_number, None))
            m.det_rule = pyo.Constraint(rule=det_general)
            m.Obj = pyo.Objective(expr=pyo.log10(m.det), sense=pyo.maximize)

        elif self.objective_option == ObjectiveLib.trace:
            # if not determinant or cholesky, calculating the OBJ with trace
            m.trace = pyo.Var(initialize=np.trace(fim), bounds=(small_number, None))
            m.trace_rule = pyo.Constraint(rule=trace_calc)
            m.Obj = pyo.Objective(expr=pyo.log10(m.trace), sense=pyo.maximize)
            # m.Obj = pyo.Objective(expr=m.trace, sense=pyo.maximize)

        elif self.objective_option == ObjectiveLib.zero:
            # add dummy objective function
            m.Obj = pyo.Objective(expr=0)
        else:
            # something went wrong!
            raise DeveloperError(
                "Objective option not recognized. Please contact the developers as you should not see this error."
            )

        return m

    def _fix_design(self, m, design_val, fix_opt=True, optimize_option=None):
        """
        Fix design variable

        Parameters
        ----------
        m: model
        design_val: design variable values dict
        fix_opt: if True, fix. Else, unfix
        optimize: a dictionary, keys are design variable name, values are True or False, deciding if this design variable is optimized as DOF this time

        Returns
        -------
        m: model
        """
        for name in self.design_name:
            # Loop over design variables
            # Get Pyomo variable object
            cuid = pyo.ComponentUID(name)
            var = cuid.find_component_on(m)
            if fix_opt:
                # If fix_opt is True, fix the design variable
                var.fix(design_val[name])
            else:
                # Otherwise check optimize_option
                if optimize_option is None:
                    # If optimize_option is None, unfix all design variables
                    var.unfix()
                else:
                    # Otherwise, unfix only the design variables listed in optimize_option with value True
                    if optimize_option[name]:
                        var.unfix()
        return m

    def _get_default_ipopt_solver(self):
        """Default solver"""
        solver = SolverFactory("ipopt")
        solver.options["linear_solver"] = "ma57"
        solver.options["halt_on_ampl_error"] = "yes"
        solver.options["max_iter"] = 3000
        return solver

    def _solve_doe(self, m, fix=False, opt_option=None):
        """Solve DOE model.
        If it's a square problem, fix design variable and solve.
        Else, fix design variable and solve square problem first, then unfix them and solve the optimization problem

        Parameters
        ----------
        m:model
        fix: if true, solve two times (square first). Else, just solve the square problem
        opt_option: a dictionary, keys are design variable name, values are True or False,
            deciding if this design variable is optimized as DOF this time.
            If None, all design variables are optimized as DOF this time.

        Returns
        -------
        solver_results: solver results
        """
        # if fix = False, solve the optimization problem
        # if fix = True, solve the square problem

        # either fix or unfix the design variables
        mod = self._fix_design(
            m, self.design_values, fix_opt=fix, optimize_option=opt_option
        )

        # if user gives solver, use this solver. if not, use default IPOPT solver
        solver_result = self.solver.solve(mod, tee=self.tee_opt)

        return solver_result

    def _sgn(self, p):
        """
        This is a helper function for stochastic_program function to compute the determinant formula.
        Give the signature of a permutation

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
