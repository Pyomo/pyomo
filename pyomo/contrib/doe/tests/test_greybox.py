# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
import copy
import itertools
import json
import os.path
from unittest.mock import patch

from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    pandas as pd,
    pandas_available,
    scipy_available,
)

from pyomo.common.fileutils import this_file_dir
import pyomo.common.unittest as unittest

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pyomo.DoE needs scipy and numpy to run tests")

if scipy_available:
    from pyomo.contrib.doe import DesignOfExperiments, FIMExternalGreyBox
    from pyomo.contrib.doe.examples.reactor_example import (
        ReactorExperiment as FullReactorExperiment,
    )
    from pyomo.contrib.doe.tests.experiment_class_example_flags import (
        RooneyBieglerMultiExperiment,
    )
    from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
        RooneyBieglerExperiment,
    )

import pyomo.environ as pyo

from pyomo.opt import SolverFactory

ipopt_available = SolverFactory("ipopt").available()
cyipopt_available = SolverFactory("cyipopt").available()

currdir = this_file_dir()
file_path = os.path.join(currdir, "..", "examples", "result.json")

with open(file_path) as f:
    data_ex = json.load(f)

data_ex["control_points"] = {float(k): v for k, v in data_ex["control_points"].items()}

_FD_EPSILON_FIRST = 1e-6  # Epsilon for numerical comparison of derivatives
_FD_EPSILON_SECOND = 1e-4  # Epsilon for numerical comparison of derivatives

if numpy_available:
    # Randomly generated P.S.D. matrix
    # Matrix is 4x4 to match example
    # number of parameters.
    testing_matrix = np.array(
        [
            [5.13730123, 1.08084953, 1.6466824, 1.09943223],
            [1.08084953, 1.57183404, 1.50704403, 1.4969689],
            [1.6466824, 1.50704403, 2.54754738, 1.39902838],
            [1.09943223, 1.4969689, 1.39902838, 1.57406692],
        ]
    )

    masking_matrix = np.triu(np.ones_like(testing_matrix))


def get_numerical_derivative(grey_box_object=None):
    # Internal import to avoid circular imports
    from pyomo.contrib.doe import ObjectiveLib

    # Grab current FIM value
    current_FIM = grey_box_object._get_FIM()
    dim = current_FIM.shape[0]
    unperturbed_value = 0

    # Find the initial value of the function
    if grey_box_object.objective_option == ObjectiveLib.trace:
        unperturbed_value = np.trace(np.linalg.inv(current_FIM))
    elif grey_box_object.objective_option == ObjectiveLib.determinant:
        unperturbed_value = np.log(np.linalg.det(current_FIM))
    elif grey_box_object.objective_option == ObjectiveLib.minimum_eigenvalue:
        vals_init, vecs_init = np.linalg.eig(current_FIM)
        unperturbed_value = np.min(vals_init)
    elif grey_box_object.objective_option == ObjectiveLib.condition_number:
        vals_init, vecs_init = np.linalg.eig(current_FIM)
        unperturbed_value = np.log(np.abs(np.max(vals_init) / np.min(vals_init)))

    # Calculate the numerical derivative, using forward difference
    numerical_derivative = np.zeros_like(current_FIM)

    # perturb each direction
    for i in range(dim):
        for j in range(dim):
            FIM_perturbed = copy.deepcopy(current_FIM)
            FIM_perturbed[i, j] += _FD_EPSILON_FIRST

            new_value_ij = 0
            # Test which method is being used:
            if grey_box_object.objective_option == ObjectiveLib.trace:
                new_value_ij = np.trace(np.linalg.inv(FIM_perturbed))
            elif grey_box_object.objective_option == ObjectiveLib.determinant:
                new_value_ij = np.log(np.linalg.det(FIM_perturbed))
            elif grey_box_object.objective_option == ObjectiveLib.minimum_eigenvalue:
                vals, vecs = np.linalg.eig(FIM_perturbed)
                new_value_ij = np.min(vals)
            elif grey_box_object.objective_option == ObjectiveLib.condition_number:
                vals, vecs = np.linalg.eig(FIM_perturbed)
                new_value_ij = np.log(np.abs(np.max(vals) / np.min(vals)))

            # Calculate the derivative value from forward difference
            diff = (new_value_ij - unperturbed_value) / _FD_EPSILON_FIRST

            numerical_derivative[i, j] = diff

    return numerical_derivative


def get_numerical_second_derivative(grey_box_object=None, return_reduced=True):
    # Internal import to avoid circular imports
    from pyomo.contrib.doe import ObjectiveLib

    # Grab current FIM value
    current_FIM = grey_box_object._get_FIM()
    dim = current_FIM.shape[0]

    # Calculate the numerical derivative,
    # using second order formula
    numerical_derivative = np.zeros([dim, dim, dim, dim])

    # perturb each direction
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    FIM_perturbed_1 = copy.deepcopy(current_FIM)
                    FIM_perturbed_2 = copy.deepcopy(current_FIM)
                    FIM_perturbed_3 = copy.deepcopy(current_FIM)
                    FIM_perturbed_4 = copy.deepcopy(current_FIM)

                    # Need 4 perturbations to cover the
                    # formula H[i, j] = [(FIM + eps (both))
                    # + (FIM +/- eps one each)
                    # + (FIM -/+ eps one each)
                    # + (FIM - eps (both))] / (4*eps**2)
                    FIM_perturbed_1[i, j] += _FD_EPSILON_SECOND
                    FIM_perturbed_1[k, l] += _FD_EPSILON_SECOND

                    FIM_perturbed_2[i, j] += _FD_EPSILON_SECOND
                    FIM_perturbed_2[k, l] += -_FD_EPSILON_SECOND

                    FIM_perturbed_3[i, j] += -_FD_EPSILON_SECOND
                    FIM_perturbed_3[k, l] += _FD_EPSILON_SECOND

                    FIM_perturbed_4[i, j] += -_FD_EPSILON_SECOND
                    FIM_perturbed_4[k, l] += -_FD_EPSILON_SECOND

                    new_values = np.array([0.0, 0.0, 0.0, 0.0])
                    # Test which method is being used:
                    if grey_box_object.objective_option == ObjectiveLib.trace:
                        new_values[0] = np.trace(np.linalg.inv(FIM_perturbed_1))
                        new_values[1] = np.trace(np.linalg.inv(FIM_perturbed_2))
                        new_values[2] = np.trace(np.linalg.inv(FIM_perturbed_3))
                        new_values[3] = np.trace(np.linalg.inv(FIM_perturbed_4))
                    elif grey_box_object.objective_option == ObjectiveLib.determinant:
                        new_values[0] = np.log(np.linalg.det(FIM_perturbed_1))
                        new_values[1] = np.log(np.linalg.det(FIM_perturbed_2))
                        new_values[2] = np.log(np.linalg.det(FIM_perturbed_3))
                        new_values[3] = np.log(np.linalg.det(FIM_perturbed_4))
                    elif (
                        grey_box_object.objective_option
                        == ObjectiveLib.minimum_eigenvalue
                    ):
                        vals, vecs = np.linalg.eig(FIM_perturbed_1)
                        new_values[0] = np.min(vals)
                        vals, vecs = np.linalg.eig(FIM_perturbed_2)
                        new_values[1] = np.min(vals)
                        vals, vecs = np.linalg.eig(FIM_perturbed_3)
                        new_values[2] = np.min(vals)
                        vals, vecs = np.linalg.eig(FIM_perturbed_4)
                        new_values[3] = np.min(vals)
                    elif (
                        grey_box_object.objective_option
                        == ObjectiveLib.condition_number
                    ):
                        vals, vecs = np.linalg.eig(FIM_perturbed_1)
                        new_values[0] = np.log(np.abs(np.max(vals) / np.min(vals)))
                        vals, vecs = np.linalg.eig(FIM_perturbed_2)
                        new_values[1] = np.log(np.abs(np.max(vals) / np.min(vals)))
                        vals, vecs = np.linalg.eig(FIM_perturbed_3)
                        new_values[2] = np.log(np.abs(np.max(vals) / np.min(vals)))
                        vals, vecs = np.linalg.eig(FIM_perturbed_4)
                        new_values[3] = np.log(np.abs(np.max(vals) / np.min(vals)))

                    # Calculate the derivative value from second order difference formula
                    diff = (
                        new_values[0] - new_values[1] - new_values[2] + new_values[3]
                    ) / (4 * _FD_EPSILON_SECOND**2)

                    numerical_derivative[i, j, k, l] = diff

    if return_reduced:
        # This considers a 4-parameter system
        # which is what these tests are based
        # upon. This can be generalized but
        # requires checking the parameter length.
        #
        # Make ordered quads with no repeats
        # of the ordered pairs
        ordered_pairs = itertools.product(range(4), repeat=2)
        ordered_pairs_list = list(itertools.combinations_with_replacement(range(4), 2))
        ordered_quads = itertools.combinations_with_replacement(ordered_pairs, 2)

        numerical_derivative_reduced = np.zeros((10, 10))

        for curr_quad in ordered_quads:
            d1, d2 = curr_quad
            i, j, k, l = d1[0], d1[1], d2[0], d2[1]

            reordered_ijkl = grey_box_object._reorder_pairs(i, j, k, l)
            row = ordered_pairs_list.index((reordered_ijkl[2], reordered_ijkl[3]))
            col = ordered_pairs_list.index((reordered_ijkl[0], reordered_ijkl[1]))

            numerical_derivative_reduced[row, col] += numerical_derivative[i, j, k, l]

            # Duplicate check and addition
            if ((i != j) and (k != l)) and ((i == l) and (j == k)):
                numerical_derivative_reduced[row, col] += numerical_derivative[
                    i, j, k, l
                ]

        numerical_derivative_reduced += (
            numerical_derivative_reduced.transpose()
            - np.diag(np.diag(numerical_derivative_reduced))
        )
        return numerical_derivative_reduced

    # Otherwise return numerical derivative as normal
    return numerical_derivative


def get_standard_args(experiment, fd_method, obj_used):
    args = {}
    args['experiment'] = experiment
    args['fd_formula'] = fd_method
    args['step'] = 1e-3
    args['objective_option'] = obj_used
    args['scale_constant_value'] = 1
    args['scale_nominal_param_value'] = True
    args['prior_FIM'] = None
    args['jac_initial'] = None
    args['fim_initial'] = None
    args['L_diagonal_lower_bound'] = 1e-7
    solver = SolverFactory("ipopt")
    args['solver'] = solver
    # Make solver object with
    # good linear subroutines
    solver = SolverFactory("ipopt")
    solver.options["linear_solver"] = "ma57"
    solver.options["halt_on_ampl_error"] = "yes"
    solver.options["max_iter"] = 3000
    args['solver'] = solver
    # Make greybox solver object with
    # good linear subroutines
    grey_box_solver = SolverFactory("cyipopt")
    grey_box_solver.config.options["linear_solver"] = "ma57"
    grey_box_solver.config.options['tol'] = 1e-4
    grey_box_solver.config.options['mu_strategy'] = "monotone"
    args['grey_box_solver'] = grey_box_solver
    args['tee'] = False
    args['grey_box_tee'] = True
    args['get_labeled_model_args'] = None
    args['_Cholesky_option'] = True
    args['_only_compute_fim_lower'] = True
    return args


def make_greybox_and_doe_objects(objective_option):
    fd_method = "central"
    obj_used = objective_option

    experiment = FullReactorExperiment(data_ex, 10, 3)

    DoE_args = get_standard_args(experiment, fd_method, obj_used)
    DoE_args["use_grey_box_objective"] = True
    DoE_args["prior_FIM"] = testing_matrix

    doe_obj = DesignOfExperiments(**DoE_args)
    doe_obj.create_doe_model()

    grey_box_object = FIMExternalGreyBox(
        doe_object=doe_obj, objective_option=doe_obj.objective_option
    )

    return doe_obj, grey_box_object


def make_greybox_and_doe_objects_rooney_biegler(objective_option):
    fd_method = "central"
    obj_used = objective_option

    data = pd.DataFrame(data=[[2, 10.3]], columns=['hour', 'y'])
    theta = {'asymptote': 19.143, 'rate_constant': 0.5311}

    experiment = RooneyBieglerExperiment(
        data=data.loc[0, :], theta=theta, measure_error=1
    )

    DoE_args = get_standard_args(experiment, fd_method, obj_used)
    DoE_args["use_grey_box_objective"] = True

    # Make a custom grey box solver
    grey_box_solver = SolverFactory("cyipopt")
    grey_box_solver.config.options["linear_solver"] = "ma57"
    grey_box_solver.config.options['tol'] = 1e-4
    grey_box_solver.config.options['mu_strategy'] = "monotone"

    # Add the grey box solver to DoE_args
    DoE_args["grey_box_solver"] = grey_box_solver

    data = pd.DataFrame(data=[[1, 8.3], [7, 19.8]], columns=['hour', 'y'])
    theta = {'asymptote': 19.143, 'rate_constant': 0.5311}
    FIM_prior = np.zeros((2, 2))
    # Calculate prior using existing experiments
    for i in range(len(data)):
        prev_experiment = RooneyBieglerExperiment(
            data=data.loc[i, :], theta=theta, measure_error=1
        )
        doe_obj = DesignOfExperiments(
            **get_standard_args(prev_experiment, fd_method, obj_used)
        )

        FIM_prior += doe_obj.compute_FIM(method='sequential')
    DoE_args["prior_FIM"] = FIM_prior

    doe_obj = DesignOfExperiments(**DoE_args)
    doe_obj.create_doe_model()

    grey_box_object = FIMExternalGreyBox(
        doe_object=doe_obj, objective_option=doe_obj.objective_option
    )

    return doe_obj, grey_box_object


class _MockGreyBoxSolver:
    def __init__(self, name="mock-greybox"):
        self.name = name
        self.calls = []

    def solve(self, model, tee=False):
        self.calls.append({"model": model, "tee": tee})

        class _MockSolverInfo:
            status = "ok"
            termination_condition = "optimal"
            message = "mock-greybox-solve"

        class _MockResults:
            solver = _MockSolverInfo()

        return _MockResults()


def _make_ipopt_solver():
    solver = SolverFactory("ipopt")
    solver.options["linear_solver"] = "ma57"
    solver.options["halt_on_ampl_error"] = "yes"
    solver.options["max_iter"] = 3000
    return solver


def _make_cyipopt_solver(tol=1e-4):
    grey_box_solver = SolverFactory("cyipopt")
    grey_box_solver.config.options["linear_solver"] = "ma57"
    grey_box_solver.config.options['tol'] = tol
    grey_box_solver.config.options['mu_strategy'] = "monotone"
    return grey_box_solver


def _make_multiexperiment_greybox_doe(
    objective_option, prior_FIM=None, grey_box_solver=None
):
    if prior_FIM is None:
        prior_FIM = np.zeros((2, 2))
    return DesignOfExperiments(
        experiment=[RooneyBieglerMultiExperiment(hour=2.0, y=10.0)],
        objective_option=objective_option,
        use_grey_box_objective=True,
        step=1e-2,
        solver=_make_ipopt_solver(),
        grey_box_solver=(
            grey_box_solver if grey_box_solver is not None else _MockGreyBoxSolver()
        ),
        prior_FIM=prior_FIM,
    )


def _get_multiexperiment_scenario_data(doe_obj):
    scenario = doe_obj.model.param_scenario_blocks[0]
    total_fim = np.array(doe_obj.results["param_scenarios"][0]["total_fim"])
    parameter_names = list(scenario.exp_blocks[0].parameter_names)
    return scenario, total_fim, parameter_names


def _generate_lhs_candidate_points(doe_obj, lhs_n_samples, lhs_seed):
    from scipy.stats.qmc import LatinHypercube

    first_exp_block = doe_obj.model.param_scenario_blocks[0].exp_blocks[0]
    exp_input_vars = doe_obj._get_experiment_input_vars(first_exp_block)
    lb_vals = np.array([v.lb for v in exp_input_vars])
    ub_vals = np.array([v.ub for v in exp_input_vars])

    rng = np.random.default_rng(lhs_seed)
    per_dim_samples = []
    for i in range(len(exp_input_vars)):
        dim_seed = int(rng.integers(0, 2**31))
        sampler = LatinHypercube(d=1, seed=dim_seed)
        s_unit = sampler.random(n=lhs_n_samples).flatten()
        s_scaled = lb_vals[i] + s_unit * (ub_vals[i] - lb_vals[i])
        per_dim_samples.append(s_scaled.tolist())

    return list(itertools.product(*per_dim_samples))


def _expected_multiexperiment_greybox_output(objective_option, fim_np):
    if objective_option == "trace":
        return float(np.trace(np.linalg.pinv(fim_np)))
    if objective_option == "determinant":
        return float(np.log(np.linalg.det(fim_np)))
    if objective_option == "minimum_eigenvalue":
        return float(np.min(np.linalg.eigvalsh(fim_np)))
    if objective_option == "condition_number":
        eig = np.linalg.eigvalsh(fim_np)
        return float(np.log(np.abs(np.max(eig) / np.min(eig))))
    raise AssertionError(f"Unexpected greybox objective: {objective_option!r}")


def _reconstruct_fim_from_egb_inputs(egb_block, parameter_names):
    dim = len(parameter_names)
    fim = np.zeros((dim, dim))
    for i, p in enumerate(parameter_names):
        for j, q in enumerate(parameter_names):
            if i >= j:
                fim[i, j] = pyo.value(egb_block.inputs[(q, p)])
                fim[j, i] = fim[i, j]
    return fim


def _spd_hour_fim_oracle(experiment_index, input_values):
    hour = float(input_values[0])
    return np.array([[hour + 2.0, 0.2 * hour], [0.2 * hour, 14.0 - hour]])


def _diagonal_hour_fim_oracle(experiment_index, input_values):
    hour = float(input_values[0])
    return np.eye(2) * (hour + 1.0)


# Test whether or not cyipopt
# is appropriately calling the
# linear solvers.
bad_message = "Invalid option encountered."
cyipopt_call_working = True
if (
    numpy_available
    and scipy_available
    and ipopt_available
    and cyipopt_available
    and pandas_available
):
    try:
        objective_option = "determinant"
        doe_object, _ = make_greybox_and_doe_objects_rooney_biegler(
            objective_option=objective_option
        )

        # Use the grey box objective
        doe_object.use_grey_box = True

        # Run for 1 iteration to see if
        # cyipopt was called.
        doe_object.grey_box_solver.config.options["max_iter"] = 1

        doe_object.run_doe()

        cyipopt_call_working = not (
            bad_message in doe_object.results["Termination Message"]
        )
    except:
        cyipopt_call_working = False


@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
@unittest.skipIf(not numpy_available, "Numpy is not available")
@unittest.skipIf(not scipy_available, "scipy is not available")
@unittest.skipIf(not cyipopt_available, "'cyipopt' is not available")
class TestFIMExternalGreyBox(unittest.TestCase):
    # Test that we can properly
    # set the inputs for the
    # Grey Box object
    def test_set_inputs(self):
        objective_option = "trace"
        doe_obj, grey_box_object = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        # Set input values to the random testing matrix
        grey_box_object.set_input_values(testing_matrix[masking_matrix > 0])

        # Grab the values from get_FIM
        grey_box_FIM = grey_box_object._get_FIM()

        self.assertTrue(np.all(np.isclose(grey_box_FIM, testing_matrix)))

    # Testing that getting the
    # input names works properly
    def test_input_names(self):
        objective_option = "trace"
        doe_obj, grey_box_object = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        # Hard-coded names of the inputs, in the order we expect
        input_names = [
            ('A1', 'A1'),
            ('A1', 'A2'),
            ('A1', 'E1'),
            ('A1', 'E2'),
            ('A2', 'A2'),
            ('A2', 'E1'),
            ('A2', 'E2'),
            ('E1', 'E1'),
            ('E1', 'E2'),
            ('E2', 'E2'),
        ]

        # Grabbing input names from grey box object
        input_names_gb = grey_box_object.input_names()

        self.assertListEqual(input_names, input_names_gb)

    # Testing that getting the
    # output names works properly
    def test_output_names(self):
        # Need to test for each objective type
        # A-opt
        objective_option = "trace"
        doe_obj_A, grey_box_object_A = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        # D-opt
        objective_option = "determinant"
        doe_obj_D, grey_box_object_D = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        # E-opt
        objective_option = "minimum_eigenvalue"
        doe_obj_E, grey_box_object_E = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        # ME-opt
        objective_option = "condition_number"
        doe_obj_ME, grey_box_object_ME = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        # Hard-coded names of the outputs
        # There is one element per
        # objective type
        output_names = ['A-opt', 'log-D-opt', 'E-opt', 'ME-opt']

        # Grabbing input names from grey box object
        output_names_gb = []
        output_names_gb.extend(grey_box_object_A.output_names())
        output_names_gb.extend(grey_box_object_D.output_names())
        output_names_gb.extend(grey_box_object_E.output_names())
        output_names_gb.extend(grey_box_object_ME.output_names())

        self.assertListEqual(output_names, output_names_gb)

    # Testing output computation
    def test_outputs_A_opt(self):
        objective_option = "trace"
        doe_obj, grey_box_object = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        # Set input values to the random testing matrix
        grey_box_object.set_input_values(testing_matrix[masking_matrix > 0])

        grey_box_A_opt = grey_box_object.evaluate_outputs()

        A_opt = np.trace(np.linalg.inv(testing_matrix))

        self.assertTrue(np.isclose(grey_box_A_opt, A_opt))

    def test_outputs_D_opt(self):
        objective_option = "determinant"
        doe_obj, grey_box_object = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        # Set input values to the random testing matrix
        grey_box_object.set_input_values(testing_matrix[masking_matrix > 0])

        grey_box_D_opt = grey_box_object.evaluate_outputs()

        D_opt = np.log(np.linalg.det(testing_matrix))

        self.assertTrue(np.isclose(grey_box_D_opt, D_opt))

    def test_outputs_E_opt(self):
        objective_option = "minimum_eigenvalue"
        doe_obj, grey_box_object = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        # Set input values to the random testing matrix
        grey_box_object.set_input_values(testing_matrix[masking_matrix > 0])

        grey_box_E_opt = grey_box_object.evaluate_outputs()

        vals, vecs = np.linalg.eig(testing_matrix)
        E_opt = np.min(vals)

        self.assertTrue(np.isclose(grey_box_E_opt, E_opt))

    def test_outputs_ME_opt(self):
        objective_option = "condition_number"
        doe_obj, grey_box_object = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        # Set input values to the random testing matrix
        grey_box_object.set_input_values(testing_matrix[masking_matrix > 0])

        grey_box_ME_opt = grey_box_object.evaluate_outputs()

        vals, vecs = np.linalg.eig(testing_matrix)
        ME_opt = np.log(np.abs(np.max(vals) / np.min(vals)))

        self.assertTrue(np.isclose(grey_box_ME_opt, ME_opt))

    # Testing Jacobian computation
    def test_jacobian_A_opt(self):
        objective_option = "trace"
        doe_obj, grey_box_object = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        # Set input values to the random testing matrix
        grey_box_object.set_input_values(testing_matrix[masking_matrix > 0])

        # Grab the Jacobian values
        utri_vals_jac = grey_box_object.evaluate_jacobian_outputs().toarray()

        # Recover the Jacobian in Matrix Form
        jac = np.zeros_like(grey_box_object._get_FIM())
        jac[np.triu_indices_from(jac)] = utri_vals_jac
        jac += jac.transpose()
        jac = jac / 2

        # Get numerical derivative matrix
        jac_FD = get_numerical_derivative(grey_box_object)

        # assert that each component is close
        self.assertTrue(np.all(np.isclose(jac, jac_FD, rtol=1e-4, atol=1e-4)))

    def test_jacobian_D_opt(self):
        objective_option = "determinant"
        doe_obj, grey_box_object = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        # Set input values to the random testing matrix
        grey_box_object.set_input_values(testing_matrix[masking_matrix > 0])

        # Grab the Jacobian values
        utri_vals_jac = grey_box_object.evaluate_jacobian_outputs().toarray()

        # Recover the Jacobian in Matrix Form
        jac = np.zeros_like(grey_box_object._get_FIM())
        jac[np.triu_indices_from(jac)] = utri_vals_jac
        jac += jac.transpose()
        jac = jac / 2

        # Get numerical derivative matrix
        jac_FD = get_numerical_derivative(grey_box_object)

        # assert that each component is close
        self.assertTrue(np.all(np.isclose(jac, jac_FD, rtol=1e-4, atol=1e-4)))

    def test_jacobian_E_opt(self):
        objective_option = "minimum_eigenvalue"
        doe_obj, grey_box_object = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        # Set input values to the random testing matrix
        grey_box_object.set_input_values(testing_matrix[masking_matrix > 0])

        # Grab the Jacobian values
        utri_vals_jac = grey_box_object.evaluate_jacobian_outputs().toarray()

        # Recover the Jacobian in Matrix Form
        jac = np.zeros_like(grey_box_object._get_FIM())
        jac[np.triu_indices_from(jac)] = utri_vals_jac
        jac += jac.transpose()
        jac = jac / 2

        # Get numerical derivative matrix
        jac_FD = get_numerical_derivative(grey_box_object)

        # assert that each component is close
        self.assertTrue(np.all(np.isclose(jac, jac_FD, rtol=1e-4, atol=1e-4)))

    def test_jacobian_ME_opt(self):
        objective_option = "condition_number"
        doe_obj, grey_box_object = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        # Set input values to the random testing matrix
        grey_box_object.set_input_values(testing_matrix[masking_matrix > 0])

        # Grab the Jacobian values
        utri_vals_jac = grey_box_object.evaluate_jacobian_outputs().toarray()

        # Recover the Jacobian in Matrix Form
        jac = np.zeros_like(grey_box_object._get_FIM())
        jac[np.triu_indices_from(jac)] = utri_vals_jac
        jac += jac.transpose()
        jac = jac / 2

        # Get numerical derivative matrix
        jac_FD = get_numerical_derivative(grey_box_object)

        # assert that each component is close
        self.assertTrue(np.all(np.isclose(jac, jac_FD, rtol=1e-4, atol=1e-4)))

    # Testing Hessian Computation
    def test_hessian_A_opt(self):
        objective_option = "trace"
        doe_obj, grey_box_object = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        # Set input values to the random testing matrix
        grey_box_object.set_input_values(testing_matrix[masking_matrix > 0])

        # Grab the Hessian values
        hess_vals_from_gb = grey_box_object.evaluate_hessian_outputs().toarray()

        # Recover the Hessian in Matrix Form
        hess_gb = hess_vals_from_gb
        hess_gb += hess_gb.transpose() - np.diag(np.diag(hess_gb))

        # Get numerical derivative matrix
        hess_FD = get_numerical_second_derivative(grey_box_object)

        # assert that each component is close
        self.assertTrue(np.all(np.isclose(hess_gb, hess_FD, rtol=1e-4, atol=1e-4)))

    def test_hessian_D_opt(self):
        objective_option = "determinant"
        doe_obj, grey_box_object = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        # Set input values to the random testing matrix
        grey_box_object.set_input_values(testing_matrix[masking_matrix > 0])

        # Grab the Hessian values
        hess_vals_from_gb = grey_box_object.evaluate_hessian_outputs().toarray()

        # Recover the Hessian in Matrix Form
        hess_gb = hess_vals_from_gb
        hess_gb += hess_gb.transpose() - np.diag(np.diag(hess_gb))

        # Get numerical derivative matrix
        hess_FD = get_numerical_second_derivative(grey_box_object)

        # assert that each component is close
        self.assertTrue(np.all(np.isclose(hess_gb, hess_FD, rtol=1e-4, atol=1e-4)))

    def test_hessian_E_opt(self):
        objective_option = "minimum_eigenvalue"
        doe_obj, grey_box_object = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        # Set input values to the random testing matrix
        grey_box_object.set_input_values(testing_matrix[masking_matrix > 0])

        # Grab the Hessian values
        hess_vals_from_gb = grey_box_object.evaluate_hessian_outputs().toarray()

        # Recover the Hessian in Matrix Form
        hess_gb = hess_vals_from_gb
        hess_gb += hess_gb.transpose() - np.diag(np.diag(hess_gb))

        # Get numerical derivative matrix
        hess_FD = get_numerical_second_derivative(grey_box_object)

        # assert that each component is close
        self.assertTrue(np.all(np.isclose(hess_gb, hess_FD, rtol=1e-4, atol=1e-4)))

    def test_hessian_ME_opt(self):
        objective_option = "condition_number"
        doe_obj, grey_box_object = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        # Set input values to the random testing matrix
        grey_box_object.set_input_values(testing_matrix[masking_matrix > 0])

        # Grab the Hessian values
        hess_vals_from_gb = grey_box_object.evaluate_hessian_outputs().toarray()

        # Recover the Hessian in Matrix Form
        hess_gb = hess_vals_from_gb
        hess_gb += hess_gb.transpose() - np.diag(np.diag(hess_gb))

        # Get numerical derivative matrix
        hess_FD = get_numerical_second_derivative(grey_box_object)

        # assert that each component is close
        self.assertTrue(np.all(np.isclose(hess_gb, hess_FD, rtol=1e-4, atol=1e-4)))

    def test_equality_constraint_names(self):
        objective_option = "condition_number"
        doe_obj, grey_box_object = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        # Grab equality constraint names
        eq_con_names_gb = grey_box_object.equality_constraint_names()

        # Equality constraint names should be an
        # empty list.
        self.assertListEqual(eq_con_names_gb, [])

    def test_evaluate_equality_constraints(self):
        objective_option = "condition_number"
        doe_obj, grey_box_object = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        # Grab equality constraint names
        eq_con_vals_gb = grey_box_object.evaluate_equality_constraints()

        # Equality constraint values should be `None`
        # There are no equality constraints.
        self.assertIsNone(eq_con_vals_gb)

    def test_evaluate_jacobian_equality_constraints(self):
        objective_option = "condition_number"
        doe_obj, grey_box_object = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        # Grab equality constraint names
        jac_eq_con_vals_gb = grey_box_object.evaluate_jacobian_equality_constraints()

        # Jacobian of equality constraints
        # should be `None` as there are no
        # equality constraints
        self.assertIsNone(jac_eq_con_vals_gb)

    def test_evaluate_hessian_equality_constraints(self):
        objective_option = "condition_number"
        doe_obj, grey_box_object = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        # Grab equality constraint names
        hess_eq_con_vals_gb = grey_box_object.evaluate_hessian_equality_constraints()

        # Jacobian of equality constraints
        # should be `None` as there are no
        # equality constraints
        self.assertIsNone(hess_eq_con_vals_gb)

    # The following few tests will test whether
    # the DoE problem with grey box is built
    # properly.
    def test_A_opt_greybox_build(self):
        objective_option = "trace"
        doe_obj, _ = make_greybox_and_doe_objects(objective_option=objective_option)

        # Build the greybox objective block
        # on the DoE object
        doe_obj.create_grey_box_objective_function()

        # Check to see if each component exists
        all_exist = True

        # Check output and value
        # FIM Initial will be the prior FIM
        # added with the identity matrix.
        A_opt_val = np.trace(np.linalg.inv(testing_matrix + np.eye(4)))

        try:
            A_opt_val_gb = doe_obj.model.obj_cons.egb_fim_block.outputs["A-opt"].value
        except:
            A_opt_val_gb = -10.0  # Trace should never be negative
            all_exist = False

        # Intermediate check for output existence
        self.assertTrue(all_exist)
        self.assertAlmostEqual(A_opt_val, A_opt_val_gb)

        # Check inputs and values
        try:
            input_values = []
            for i in _.input_names():
                input_values.append(doe_obj.model.obj_cons.egb_fim_block.inputs[i]())
        except:
            input_values = np.zeros_like(testing_matrix)
            all_exist = False

        # Final check on existence of inputs
        self.assertTrue(all_exist)
        # Rebuild the current FIM from the input
        # values taken from the egb_fim_block
        current_FIM = np.zeros_like(testing_matrix)
        current_FIM[np.triu_indices_from(current_FIM)] = input_values
        current_FIM += current_FIM.transpose() - np.diag(np.diag(current_FIM))

        self.assertTrue(np.all(np.isclose(current_FIM, testing_matrix + np.eye(4))))

    def test_D_opt_greybox_build(self):
        objective_option = "determinant"
        doe_obj, _ = make_greybox_and_doe_objects(objective_option=objective_option)

        # Build the greybox objective block
        # on the DoE object
        doe_obj.create_grey_box_objective_function()

        # Check to see if each component exists
        all_exist = True

        # Check output and value
        # FIM Initial will be the prior FIM
        # added with the identity matrix.
        D_opt_val = np.log(np.linalg.det(testing_matrix + np.eye(4)))

        try:
            D_opt_val_gb = doe_obj.model.obj_cons.egb_fim_block.outputs[
                "log-D-opt"
            ].value
        except:
            D_opt_val_gb = -100.0  # Determinant should never be negative beyond -64
            all_exist = False

        # Intermediate check for output existence
        self.assertTrue(all_exist)
        self.assertAlmostEqual(D_opt_val, D_opt_val_gb)

        # Check inputs and values
        try:
            input_values = []
            for i in _.input_names():
                input_values.append(doe_obj.model.obj_cons.egb_fim_block.inputs[i]())
        except:
            input_values = np.zeros_like(testing_matrix)
            all_exist = False

        # Final check on existence of inputs
        self.assertTrue(all_exist)
        # Rebuild the current FIM from the input
        # values taken from the egb_fim_block
        current_FIM = np.zeros_like(testing_matrix)
        current_FIM[np.triu_indices_from(current_FIM)] = input_values
        current_FIM += current_FIM.transpose() - np.diag(np.diag(current_FIM))

        self.assertTrue(np.all(np.isclose(current_FIM, testing_matrix + np.eye(4))))

    def test_E_opt_greybox_build(self):
        objective_option = "minimum_eigenvalue"
        doe_obj, _ = make_greybox_and_doe_objects(objective_option=objective_option)

        # Build the greybox objective block
        # on the DoE object
        doe_obj.create_grey_box_objective_function()

        # Check to see if each component exists
        all_exist = True

        # Check output and value
        # FIM Initial will be the prior FIM
        # added with the identity matrix.
        vals, vecs = np.linalg.eig(testing_matrix + np.eye(4))
        E_opt_val = np.min(vals)

        try:
            E_opt_val_gb = doe_obj.model.obj_cons.egb_fim_block.outputs["E-opt"].value
        except:
            E_opt_val_gb = -10.0  # Determinant should never be negative
            all_exist = False

        # Intermediate check for output existence
        self.assertTrue(all_exist)
        self.assertAlmostEqual(E_opt_val, E_opt_val_gb)

        # Check inputs and values
        try:
            input_values = []
            for i in _.input_names():
                input_values.append(doe_obj.model.obj_cons.egb_fim_block.inputs[i]())
        except:
            input_values = np.zeros_like(testing_matrix)
            all_exist = False

        # Final check on existence of inputs
        self.assertTrue(all_exist)
        # Rebuild the current FIM from the input
        # values taken from the egb_fim_block
        current_FIM = np.zeros_like(testing_matrix)
        current_FIM[np.triu_indices_from(current_FIM)] = input_values
        current_FIM += current_FIM.transpose() - np.diag(np.diag(current_FIM))

        self.assertTrue(np.all(np.isclose(current_FIM, testing_matrix + np.eye(4))))

    def test_ME_opt_greybox_build(self):
        objective_option = "condition_number"
        doe_obj, _ = make_greybox_and_doe_objects(objective_option=objective_option)

        # Build the greybox objective block
        # on the DoE object
        doe_obj.create_grey_box_objective_function()

        # Check to see if each component exists
        all_exist = True

        # Check output and value
        # FIM Initial will be the prior FIM
        # added with the identity matrix.
        vals, vecs = np.linalg.eig(testing_matrix + np.eye(4))
        ME_opt_val = np.log(np.abs(np.max(vals) / np.min(vals)))

        try:
            ME_opt_val_gb = doe_obj.model.obj_cons.egb_fim_block.outputs["ME-opt"].value
        except:
            ME_opt_val_gb = -10.0  # Condition number should not be negative
            all_exist = False

        # Intermediate check for output existence
        self.assertTrue(all_exist)
        self.assertAlmostEqual(ME_opt_val, ME_opt_val_gb)

        # Check inputs and values
        try:
            input_values = []
            for i in _.input_names():
                input_values.append(doe_obj.model.obj_cons.egb_fim_block.inputs[i]())
        except:
            input_values = np.zeros_like(testing_matrix)
            all_exist = False

        # Final check on existence of inputs
        self.assertTrue(all_exist)
        # Rebuild the current FIM from the input
        # values taken from the egb_fim_block
        current_FIM = np.zeros_like(testing_matrix)
        current_FIM[np.triu_indices_from(current_FIM)] = input_values
        current_FIM += current_FIM.transpose() - np.diag(np.diag(current_FIM))

        self.assertTrue(np.all(np.isclose(current_FIM, testing_matrix + np.eye(4))))

    # Testing all the error messages
    def test_constructor_doe_object_error(self):
        with self.assertRaisesRegex(
            ValueError,
            "Either ``doe_object`` or both ``parameter_names`` and ``fim_initial`` "
            "must be provided to build the FIM grey box.",
        ):
            grey_box_object = FIMExternalGreyBox(doe_object=None)

    def test_constructor_objective_lib_error(self):
        objective_option = "trace"
        doe_object, grey_box_object = make_greybox_and_doe_objects(
            objective_option=objective_option
        )
        with self.assertRaisesRegex(
            ValueError, "'Bad Objective Option' is not a valid ObjectiveLib"
        ):
            bad_grey_box_object = FIMExternalGreyBox(
                doe_object=doe_object, objective_option="Bad Objective Option"
            )

    def test_output_names_obj_lib_error(self):
        objective_option = "trace"
        doe_object, grey_box_object = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        grey_box_object.objective_option = "Bad Objective Option"

        with self.assertRaisesRegex(
            ValueError, "'Bad Objective Option' is not a valid ObjectiveLib"
        ):
            grey_box_object.output_names()

    def test_evaluate_outputs_obj_lib_error(self):
        objective_option = "trace"
        doe_object, grey_box_object = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        grey_box_object.objective_option = "Bad Objective Option"

        with self.assertRaisesRegex(
            ValueError, "'Bad Objective Option' is not a valid ObjectiveLib"
        ):
            grey_box_object.evaluate_outputs()

    def test_evaluate_jacobian_outputs_obj_lib_error(self):
        objective_option = "trace"
        doe_object, grey_box_object = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        grey_box_object.objective_option = "Bad Objective Option"

        with self.assertRaisesRegex(
            ValueError, "'Bad Objective Option' is not a valid ObjectiveLib"
        ):
            grey_box_object.evaluate_jacobian_outputs()

    def test_evaluate_hessian_outputs_obj_lib_error(self):
        objective_option = "trace"
        doe_object, grey_box_object = make_greybox_and_doe_objects(
            objective_option=objective_option
        )

        grey_box_object.objective_option = "Bad Objective Option"

        with self.assertRaisesRegex(
            ValueError, "'Bad Objective Option' is not a valid ObjectiveLib"
        ):
            grey_box_object.evaluate_hessian_outputs()

    # Test all versions of solving
    # using grey box
    @unittest.skipIf(
        not cyipopt_call_working, "cyipopt is not properly accessing linear solvers"
    )
    def test_solve_D_optimality_log_determinant(self):
        # Two locally optimal design points exist
        # (time, optimal objective value)
        # Here, the objective value is
        # log-10(determinant) of the FIM
        optimal_experimental_designs = [
            np.array([1.782, 4.34]),
            np.array([10.00, 4.35]),
        ]
        objective_option = "determinant"
        doe_object, grey_box_object = make_greybox_and_doe_objects_rooney_biegler(
            objective_option=objective_option
        )

        # Set to use the grey box objective
        doe_object.use_grey_box = True

        # Solve the model
        doe_object.run_doe()

        optimal_time_val = doe_object.results["Experiment Design"][0]
        optimal_obj_val = np.log10(np.exp(pyo.value(doe_object.model.objective)))

        optimal_design_np_array = np.array([optimal_time_val, optimal_obj_val])

        self.assertTrue(
            np.all(
                np.isclose(
                    optimal_design_np_array, optimal_experimental_designs[0], 1e-1
                )
            )
            or np.all(
                np.isclose(
                    optimal_design_np_array, optimal_experimental_designs[1], 1e-1
                )
            )
        )

    @unittest.skipIf(
        not cyipopt_call_working, "cyipopt is not properly accessing linear solvers"
    )
    def test_solve_A_optimality_trace_of_inverse(self):
        # Two locally optimal design points exist
        # (time, optimal objective value)
        # Here, the objective value is
        # trace(inverse(FIM))
        optimal_experimental_designs = [
            np.array([1.33, 0.0277]),
            np.array([9.9, 0.0366]),
        ]
        objective_option = "trace"
        doe_object, grey_box_object = make_greybox_and_doe_objects_rooney_biegler(
            objective_option=objective_option
        )

        # Set to use the grey box objective
        doe_object.use_grey_box = True

        # Solve the model
        doe_object.run_doe()

        optimal_time_val = doe_object.results["Experiment Design"][0]
        optimal_obj_val = doe_object.model.objective()

        optimal_design_np_array = np.array([optimal_time_val, optimal_obj_val])

        self.assertTrue(
            np.all(
                np.isclose(
                    optimal_design_np_array, optimal_experimental_designs[0], 1e-1
                )
            )
            or np.all(
                np.isclose(
                    optimal_design_np_array, optimal_experimental_designs[1], 1e-1
                )
            )
        )

    @unittest.skipIf(
        not cyipopt_call_working, "cyipopt is not properly accessing linear solvers"
    )
    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_solve_E_optimality_minimum_eigenvalue(self):
        # Two locally optimal design points exist
        # (time, optimal objective value)
        # Here, the objective value is
        # minimum eigenvalue of the FIM
        optimal_experimental_designs = [
            np.array([1.30, 38.70]),
            np.array([10.00, 28.349]),
        ]
        objective_option = "minimum_eigenvalue"
        doe_object, grey_box_object = make_greybox_and_doe_objects_rooney_biegler(
            objective_option=objective_option
        )

        # Set to use the grey box objective
        doe_object.use_grey_box = True

        # Solve the model
        doe_object.run_doe()

        optimal_time_val = doe_object.results["Experiment Design"][0]
        optimal_obj_val = doe_object.model.objective()

        optimal_design_np_array = np.array([optimal_time_val, optimal_obj_val])

        self.assertTrue(
            np.all(
                np.isclose(
                    optimal_design_np_array, optimal_experimental_designs[0], 1e-1
                )
            )
            or np.all(
                np.isclose(
                    optimal_design_np_array, optimal_experimental_designs[1], 1e-1
                )
            )
        )

    @unittest.skipIf(
        not cyipopt_call_working, "cyipopt is not properly accessing linear solvers"
    )
    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_solve_ME_optimality_condition_number(self):
        # Two locally optimal design points exist
        # (time, optimal objective value)
        # Here, the objective value is
        # condition number of the FIM
        optimal_experimental_designs = [
            np.array([0.943, np.log(13.524)]),
            np.array([10.00, np.log(27.675)]),
        ]
        objective_option = "condition_number"
        doe_object, grey_box_object = make_greybox_and_doe_objects_rooney_biegler(
            objective_option=objective_option
        )

        # Set to use the grey box objective
        doe_object.use_grey_box = True

        # Solve the model
        doe_object.run_doe()

        optimal_time_val = doe_object.results["Experiment Design"][0]
        optimal_obj_val = doe_object.model.objective()

        optimal_design_np_array = np.array([optimal_time_val, optimal_obj_val])

        self.assertTrue(
            np.all(
                np.isclose(
                    optimal_design_np_array, optimal_experimental_designs[0], 1e-1
                )
            )
            or np.all(
                np.isclose(
                    optimal_design_np_array, optimal_experimental_designs[1], 1e-1
                )
            )
        )


@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
@unittest.skipIf(not numpy_available, "Numpy is not available")
@unittest.skipIf(not scipy_available, "scipy is not available")
@unittest.skipIf(not pandas_available, "pandas is not available")
class TestMultiexperimentBuild(unittest.TestCase):
    def test_optimize_experiments_greybox_uses_aggregated_fim(self):
        # Check that the multi-experiment greybox block is seeded from the
        # aggregated scenario FIM and that the final solve is routed through
        # the greybox solver interface.
        grey_box_solver = _MockGreyBoxSolver()
        doe_obj = _make_multiexperiment_greybox_doe(
            objective_option="minimum_eigenvalue", grey_box_solver=grey_box_solver
        )

        doe_obj.optimize_experiments(n_exp=2)

        self.assertEqual(len(grey_box_solver.calls), 1)
        self.assertEqual(
            doe_obj.results["run_info"]["solver"]["name"], grey_box_solver.name
        )

        scenario, total_fim, parameter_names = _get_multiexperiment_scenario_data(
            doe_obj
        )
        self.assertTrue(hasattr(scenario.obj_cons, "egb_fim_block"))
        self.assertFalse(hasattr(scenario.obj_cons, "L"))

        for i, p in enumerate(parameter_names):
            for j, q in enumerate(parameter_names):
                if i >= j:
                    self.assertAlmostEqual(
                        pyo.value(scenario.obj_cons.egb_fim_block.inputs[(q, p)]),
                        total_fim[i, j],
                        places=7,
                    )

        self.assertAlmostEqual(
            pyo.value(scenario.obj_cons.egb_fim_block.outputs["E-opt"]),
            np.min(np.linalg.eigvalsh(total_fim)),
            places=7,
        )

    def test_optimize_experiments_greybox_outputs_match_numpy_for_all_supported_objectives(
        self,
    ):
        # Validate the deterministic wiring: for each supported greybox metric,
        # the external block output should match direct NumPy on scenario.total_fim.
        prior_fim = np.array([[6.0, 0.75], [0.75, 4.0]])
        for objective_option in (
            "determinant",
            "trace",
            "minimum_eigenvalue",
            "condition_number",
        ):
            with self.subTest(objective=objective_option):
                doe_obj = _make_multiexperiment_greybox_doe(
                    objective_option=objective_option,
                    prior_FIM=prior_fim.copy(),
                    grey_box_solver=_MockGreyBoxSolver(name=f"mock-{objective_option}"),
                )

                doe_obj.optimize_experiments(n_exp=2)

                scenario, total_fim, parameter_names = (
                    _get_multiexperiment_scenario_data(doe_obj)
                )
                egb_block = scenario.obj_cons.egb_fim_block

                for i, p in enumerate(parameter_names):
                    for j, q in enumerate(parameter_names):
                        if i >= j:
                            self.assertAlmostEqual(
                                pyo.value(egb_block.inputs[(q, p)]),
                                total_fim[i, j],
                                places=7,
                            )

                self.assertAlmostEqual(
                    pyo.value(egb_block.outputs[doe_obj._grey_box_output_name()]),
                    _expected_multiexperiment_greybox_output(
                        objective_option, total_fim
                    ),
                    places=7,
                )

    def test_optimize_experiments_greybox_prior_fim_is_included_in_inputs_and_output(
        self,
    ):
        # Use a large prior so the external block must see
        # total_fim = sum(experiment_fim) + prior_FIM.
        prior_fim = np.array([[100.0, 12.0], [12.0, 80.0]])
        doe_obj = _make_multiexperiment_greybox_doe(
            objective_option="determinant",
            prior_FIM=prior_fim.copy(),
            grey_box_solver=_MockGreyBoxSolver(),
        )

        doe_obj.optimize_experiments(n_exp=2)

        scenario, total_fim, parameter_names = _get_multiexperiment_scenario_data(
            doe_obj
        )
        exp_fim_sum = sum(
            (
                np.array(exp_data["fim"])
                for exp_data in doe_obj.results["param_scenarios"][0]["experiments"]
            ),
            np.zeros_like(total_fim),
        )
        egb_block = scenario.obj_cons.egb_fim_block

        self.assertTrue(np.allclose(total_fim, exp_fim_sum + prior_fim, atol=1e-6))
        self.assertFalse(np.allclose(total_fim, exp_fim_sum, atol=1e-6))

        for i, p in enumerate(parameter_names):
            for j, q in enumerate(parameter_names):
                if i >= j:
                    self.assertAlmostEqual(
                        pyo.value(egb_block.inputs[(q, p)]), total_fim[i, j], places=7
                    )

        output_with_prior = pyo.value(egb_block.outputs["log-D-opt"])
        self.assertAlmostEqual(
            output_with_prior,
            _expected_multiexperiment_greybox_output("determinant", total_fim),
            places=7,
        )
        self.assertFalse(
            np.isclose(
                output_with_prior,
                _expected_multiexperiment_greybox_output("determinant", exp_fim_sum),
                rtol=1e-6,
                atol=1e-6,
            )
        )

    def test_optimize_experiments_greybox_uses_init_solver_for_square_solve_and_grey_box_solver_for_final_solve(
        self,
    ):
        # Initialization should use init_solver; the greybox solver should be
        # reserved for the final optimize_experiments NLP solve.
        main_solver = _make_ipopt_solver()
        init_solver = _make_ipopt_solver()
        init_solver.options["max_iter"] = 123
        grey_box_solver = _MockGreyBoxSolver()

        doe_obj = DesignOfExperiments(
            experiment=[RooneyBieglerMultiExperiment(hour=2.0)],
            objective_option="minimum_eigenvalue",
            use_grey_box_objective=True,
            step=1e-2,
            solver=main_solver,
            grey_box_solver=grey_box_solver,
        )

        init_calls = 0
        call_order = []
        original_init_solve = init_solver.solve
        original_grey_box_solve = grey_box_solver.solve

        def _init_solve(*args, **kwargs):
            nonlocal init_calls
            init_calls += 1
            call_order.append("init")
            return original_init_solve(*args, **kwargs)

        def _grey_box_solve(*args, **kwargs):
            call_order.append("greybox")
            return original_grey_box_solve(*args, **kwargs)

        with (
            patch.object(
                main_solver,
                "solve",
                side_effect=AssertionError(
                    "Primary solver should not be used in greybox optimize_experiments()."
                ),
            ),
            patch.object(init_solver, "solve", side_effect=_init_solve),
            patch.object(grey_box_solver, "solve", side_effect=_grey_box_solve),
        ):
            doe_obj.optimize_experiments(n_exp=2, init_solver=init_solver)

        self.assertGreaterEqual(init_calls, 1)
        self.assertEqual(len(grey_box_solver.calls), 1)
        self.assertEqual(call_order[-1], "greybox")
        self.assertTrue(all(tag == "init" for tag in call_order[:-1]))
        self.assertEqual(
            doe_obj.results["settings"]["initialization"]["solver_name"],
            getattr(init_solver, "name", str(init_solver)),
        )
        self.assertEqual(
            doe_obj.results["run_info"]["solver"]["name"], grey_box_solver.name
        )

    def test_optimize_experiments_greybox_is_reentrant_on_same_object(self):
        # Re-running the same greybox DoE object should rebuild a fresh
        # external block and reseed it from the current aggregated total FIM.
        grey_box_solver = _MockGreyBoxSolver()
        doe_obj = _make_multiexperiment_greybox_doe(
            objective_option="minimum_eigenvalue",
            prior_FIM=np.zeros((2, 2)),
            grey_box_solver=grey_box_solver,
        )

        doe_obj.optimize_experiments(n_exp=2)
        first_scenario, first_total_fim, _ = _get_multiexperiment_scenario_data(doe_obj)
        first_egb_block = first_scenario.obj_cons.egb_fim_block

        self.assertAlmostEqual(
            pyo.value(first_egb_block.outputs["E-opt"]),
            _expected_multiexperiment_greybox_output(
                "minimum_eigenvalue", first_total_fim
            ),
            places=7,
        )

        doe_obj.prior_FIM = np.array([[20.0, 2.0], [2.0, 15.0]])
        doe_obj.optimize_experiments(n_exp=2)

        second_scenario, second_total_fim, second_parameter_names = (
            _get_multiexperiment_scenario_data(doe_obj)
        )
        second_egb_block = second_scenario.obj_cons.egb_fim_block

        self.assertIsNot(first_egb_block, second_egb_block)
        self.assertEqual(len(list(doe_obj.model.param_scenario_blocks.keys())), 1)
        self.assertEqual(len(grey_box_solver.calls), 2)
        self.assertFalse(np.allclose(first_total_fim, second_total_fim, atol=1e-6))

        for i, p in enumerate(second_parameter_names):
            for j, q in enumerate(second_parameter_names):
                if i >= j:
                    self.assertAlmostEqual(
                        pyo.value(second_egb_block.inputs[(q, p)]),
                        second_total_fim[i, j],
                        places=7,
                    )

        self.assertAlmostEqual(
            pyo.value(second_egb_block.outputs["E-opt"]),
            _expected_multiexperiment_greybox_output(
                "minimum_eigenvalue", second_total_fim
            ),
            places=7,
        )

    def test_optimize_experiments_greybox_lhs_initialization_scores_e_opt_and_me_opt(
        self,
    ):
        # This checks the LHS candidate-combination scorer for the greybox-only
        # E-opt and ME-opt objectives. The patched oracle maps the single
        # Rooney-Biegler design input (hour) to a positive-definite 2x2 FIM so
        # the best combination can be computed independently and deterministically.
        lhs_n_samples = 4
        lhs_seed = 19

        for objective_option in ("minimum_eigenvalue", "condition_number"):
            with self.subTest(objective=objective_option):
                doe_obj = _make_multiexperiment_greybox_doe(
                    objective_option=objective_option,
                    prior_FIM=np.zeros((2, 2)),
                    grey_box_solver=_MockGreyBoxSolver(name=f"mock-{objective_option}"),
                )

                with patch.object(
                    doe_obj,
                    "_compute_fim_at_point_no_prior",
                    side_effect=_spd_hour_fim_oracle,
                ):
                    doe_obj.optimize_experiments(
                        n_exp=2,
                        init_method="lhs",
                        init_n_samples=lhs_n_samples,
                        init_seed=lhs_seed,
                    )

                lhs_diag = doe_obj.results["diagnostics"]["lhs_initialization"]
                actual_points = doe_obj.results["LHS Best Initial Points"]
                candidate_points = _generate_lhs_candidate_points(
                    doe_obj, lhs_n_samples=lhs_n_samples, lhs_seed=lhs_seed
                )
                candidate_norm = {
                    tuple(np.round(point, 8)) for point in candidate_points
                }

                self.assertEqual(doe_obj.results["Initialization Method"], "lhs")
                self.assertIsNotNone(lhs_diag)
                self.assertTrue(np.isfinite(lhs_diag["best_obj"]))
                self.assertGreater(lhs_diag["best_obj"], 0.0)

                for point in actual_points:
                    self.assertIn(tuple(np.round(point, 8)), candidate_norm)

                if doe_obj.objective_option in DesignOfExperiments._MAXIMIZE_OBJECTIVES:
                    best_obj = -np.inf
                    is_better = lambda new, best: new > best
                else:
                    best_obj = np.inf
                    is_better = lambda new, best: new < best

                for combo in itertools.combinations(range(len(candidate_points)), 2):
                    fim_total = sum(
                        (
                            _spd_hour_fim_oracle(0, candidate_points[idx])
                            for idx in combo
                        ),
                        np.zeros((2, 2)),
                    )
                    obj_val = doe_obj._evaluate_objective_from_fim(fim_total)
                    if is_better(obj_val, best_obj):
                        best_obj = obj_val

                actual_fim_total = sum(
                    (_spd_hour_fim_oracle(0, point) for point in actual_points),
                    np.zeros((2, 2)),
                )
                actual_obj = doe_obj._evaluate_objective_from_fim(actual_fim_total)

                self.assertAlmostEqual(actual_obj, best_obj, places=12)
                self.assertAlmostEqual(lhs_diag["best_obj"], best_obj, places=12)

    def test_optimize_experiments_greybox_initialization_refreshes_inputs_after_square_solve(
        self,
    ):
        # Build-time greybox inputs reflect the template design, but after LHS
        # changes the starting point the square-solve refresh should reseed the
        # block from the new aggregated FIM before the final solve.
        lhs_n_samples = 4
        lhs_seed = 29
        captured = {}
        doe_obj = _make_multiexperiment_greybox_doe(
            objective_option="minimum_eigenvalue",
            prior_FIM=np.zeros((2, 2)),
            grey_box_solver=_MockGreyBoxSolver(),
        )
        original_initialize = doe_obj._initialize_grey_box_block

        def _capture_initialize(egb_block, fim_np, parameter_names):
            captured["before"] = _reconstruct_fim_from_egb_inputs(
                egb_block, parameter_names
            )
            captured["refreshed"] = np.array(fim_np, copy=True)
            return original_initialize(egb_block, fim_np, parameter_names)

        with (
            patch.object(
                doe_obj,
                "_compute_fim_at_point_no_prior",
                side_effect=_diagonal_hour_fim_oracle,
            ),
            patch.object(
                doe_obj, "_initialize_grey_box_block", side_effect=_capture_initialize
            ),
        ):
            doe_obj.optimize_experiments(
                n_exp=1,
                init_method="lhs",
                init_n_samples=lhs_n_samples,
                init_seed=lhs_seed,
            )

        scenario, total_fim, parameter_names = _get_multiexperiment_scenario_data(
            doe_obj
        )
        final_fim = _reconstruct_fim_from_egb_inputs(
            scenario.obj_cons.egb_fim_block, parameter_names
        )

        self.assertIn("before", captured)
        self.assertIn("refreshed", captured)
        self.assertFalse(np.allclose(captured["before"], captured["refreshed"]))
        self.assertTrue(np.allclose(final_fim, captured["refreshed"], atol=1e-7))
        self.assertTrue(np.allclose(final_fim, total_fim, atol=1e-7))
        # The Rooney-Biegler template is built with hour=2.0, so this confirms
        # LHS moved away from the build-time design and made the refresh check meaningful.
        self.assertNotAlmostEqual(
            doe_obj.results["LHS Best Initial Points"][0][0], 2.0, places=6
        )

    def test_optimize_experiments_greybox_tee_flag_reaches_solver(self):
        # grey_box_tee is only meaningful if it propagates to the external
        # solver interface, so capture the mocked solve() call and verify tee.
        grey_box_solver = _MockGreyBoxSolver()
        doe_obj = DesignOfExperiments(
            experiment=[RooneyBieglerMultiExperiment(hour=2.0)],
            objective_option="minimum_eigenvalue",
            use_grey_box_objective=True,
            step=1e-2,
            solver=_make_ipopt_solver(),
            grey_box_solver=grey_box_solver,
            grey_box_tee=True,
        )

        doe_obj.optimize_experiments(n_exp=2)

        self.assertEqual(len(grey_box_solver.calls), 1)
        self.assertTrue(grey_box_solver.calls[0]["tee"])


@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
@unittest.skipIf(not numpy_available, "Numpy is not available")
@unittest.skipIf(not scipy_available, "scipy is not available")
@unittest.skipIf(not cyipopt_available, "'cyipopt' is not available")
@unittest.skipIf(
    not cyipopt_call_working, "cyipopt is not properly accessing linear solvers"
)
@unittest.skipIf(not pandas_available, "pandas is not available")
class TestMultiexperimentSolve(unittest.TestCase):
    def test_optimize_experiments_determinant_expected_values_greybox(self):
        # The scenario objective is order-invariant, so compare the chosen
        # experiment hours in sorted order.
        exp_list = [
            RooneyBieglerMultiExperiment(hour=1.0, y=8.3),
            RooneyBieglerMultiExperiment(hour=2.0, y=10.3),
        ]

        doe = DesignOfExperiments(
            experiment=exp_list,
            objective_option="determinant",
            step=1e-2,
            use_grey_box_objective=True,
            grey_box_solver=_make_cyipopt_solver(tol=1e-4),
            grey_box_tee=False,
        )
        doe.optimize_experiments()

        scenario = doe.results["param_scenarios"][0]
        got_hours = sorted(exp["design"][0] for exp in scenario["experiments"])
        self.assertStructuredAlmostEqual(
            got_hours, sorted([1.9321985035514362, 9.999999685577139]), abstol=1e-3
        )
        self.assertAlmostEqual(
            scenario["metrics"]["log10_d_opt"], 6.028152580313302, places=3
        )

    def test_optimize_experiments_trace_expected_values_greybox(self):
        # This is a regression test for the cyipopt-backed multi-experiment
        # greybox solve on A-opt: the chosen design and reported objective
        # should stay near the known Rooney-Biegler reference solution.
        exp_list = [
            RooneyBieglerMultiExperiment(hour=1.0, y=8.3),
            RooneyBieglerMultiExperiment(hour=2.0, y=10.3),
        ]

        doe = DesignOfExperiments(
            experiment=exp_list,
            objective_option="trace",
            step=1e-2,
            use_grey_box_objective=True,
            grey_box_solver=_make_cyipopt_solver(tol=1e-6),
            grey_box_tee=False,
        )
        doe.optimize_experiments()

        scenario = doe.results["param_scenarios"][0]
        got_hours = sorted(exp["design"][0] for exp in scenario["experiments"])
        self.assertStructuredAlmostEqual(got_hours, sorted([1.01, 10.0]), abstol=1e-3)
        self.assertAlmostEqual(scenario["metrics"]["log10_a_opt"], -1.9438, places=3)

    def test_optimize_experiments_min_eig_expected_values_greybox(self):
        # This checks the end-to-end greybox E-opt solve against a stable
        # reference solution so future greybox wiring changes do not silently
        # alter the chosen experiment pair or final metric.
        exp_list = [
            RooneyBieglerMultiExperiment(hour=1.0, y=8.3),
            RooneyBieglerMultiExperiment(hour=2.0, y=10.3),
        ]

        doe = DesignOfExperiments(
            experiment=exp_list,
            objective_option="minimum_eigenvalue",
            step=1e-2,
            solver=_make_ipopt_solver(),
            use_grey_box_objective=True,
            grey_box_solver=_make_cyipopt_solver(tol=1e-6),
            grey_box_tee=False,
        )
        doe.optimize_experiments()

        scenario = doe.results["param_scenarios"][0]
        got_hours = sorted(exp["design"][0] for exp in scenario["experiments"])
        self.assertStructuredAlmostEqual(got_hours, sorted([1.0, 10.0]), abstol=1e-2)
        self.assertAlmostEqual(scenario["metrics"]["log10_e_opt"], 1.9532, places=2)

    def test_optimize_experiments_me_opt_expected_values_greybox(self):
        # ME-opt is greybox-only in optimize_experiments(), so keep a dedicated
        # solve regression here to guard the condition-number objective path.
        exp_list = [
            RooneyBieglerMultiExperiment(hour=1.0, y=8.3),
            RooneyBieglerMultiExperiment(hour=2.0, y=10.3),
        ]

        doe = DesignOfExperiments(
            experiment=exp_list,
            objective_option="condition_number",
            step=1e-2,
            solver=_make_ipopt_solver(),
            use_grey_box_objective=True,
            grey_box_solver=_make_cyipopt_solver(tol=1e-6),
            grey_box_tee=False,
        )
        doe.optimize_experiments()

        scenario = doe.results["param_scenarios"][0]
        got_hours = sorted(exp["design"][0] for exp in scenario["experiments"])
        self.assertStructuredAlmostEqual(got_hours, sorted([7.13, 10.0]), abstol=1e-2)
        self.assertAlmostEqual(
            scenario["metrics"]["log10_me_opt"], 1.5289, places=2
        )


@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
@unittest.skipIf(not numpy_available, "Numpy is not available")
@unittest.skipIf(not scipy_available, "scipy is not available")
@unittest.skipIf(not cyipopt_available, "'cyipopt' is not available")
@unittest.skipIf(
    not cyipopt_call_working, "cyipopt is not properly accessing linear solvers"
)
@unittest.skipIf(not pandas_available, "pandas is not available")
class TestSingleExperimentSolve(unittest.TestCase):
    def test_optimize_experiments_single_experiment_greybox_path_works(self):
        # Even with n_exp=1, optimize_experiments() should take the template-mode
        # greybox path, solve with the grey_box_solver, and keep the greybox
        # block synchronized with the final scenario FIM.
        grey_box_solver = _make_cyipopt_solver(tol=1e-6)
        doe_obj = DesignOfExperiments(
            experiment=[RooneyBieglerMultiExperiment(hour=2.0, y=10.0)],
            objective_option="minimum_eigenvalue",
            step=1e-2,
            solver=_make_ipopt_solver(),
            use_grey_box_objective=True,
            grey_box_solver=grey_box_solver,
            grey_box_tee=False,
        )

        doe_obj.optimize_experiments(n_exp=1)

        scenario, total_fim, _ = _get_multiexperiment_scenario_data(doe_obj)
        scenario_results = doe_obj.results["param_scenarios"][0]
        design_hour = scenario_results["experiments"][0]["design"][0]

        self.assertEqual(doe_obj.results["Solver Status"], "ok")
        self.assertEqual(doe_obj.results["Number of Experiments per Scenario"], 1)
        self.assertTrue(doe_obj.results["settings"]["modeling"]["template_mode"])
        self.assertEqual(len(scenario_results["experiments"]), 1)
        self.assertEqual(
            doe_obj.results["run_info"]["solver"]["name"],
            getattr(grey_box_solver, "name", str(grey_box_solver)),
        )
        self.assertGreaterEqual(design_hour, 1.0)
        self.assertLessEqual(design_hour, 10.0)
        self.assertTrue(np.isfinite(scenario_results["metrics"]["log10_e_opt"]))
        self.assertAlmostEqual(
            pyo.value(scenario.obj_cons.egb_fim_block.outputs["E-opt"]),
            _expected_multiexperiment_greybox_output("minimum_eigenvalue", total_fim),
            places=7,
        )


@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
@unittest.skipIf(not numpy_available, "Numpy is not available")
@unittest.skipIf(not scipy_available, "scipy is not available")
@unittest.skipIf(not pandas_available, "pandas is not available")
class TestMultiexperimentError(unittest.TestCase):
    def test_optimize_experiments_greybox_unsupported_objectives_are_rejected(self):
        # These unsupported objectives share the same early-validation path, so
        # keep them in one table-driven test and verify none reaches the
        # external grey_box_solver interface.
        class _UnusedGreyBoxSolver:
            def solve(self, model, tee=False):
                raise AssertionError("grey_box_solver.solve should not be reached")

        for objective_option in ("pseudo_trace", "zero"):
            with self.subTest(objective=objective_option):
                doe_obj = DesignOfExperiments(
                    experiment=[RooneyBieglerMultiExperiment(hour=2.0, y=10.0)],
                    objective_option=objective_option,
                    use_grey_box_objective=True,
                    step=1e-2,
                    solver=_make_ipopt_solver(),
                    grey_box_solver=_UnusedGreyBoxSolver(),
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "Grey-box objective support in optimize_experiments\\(\\) is only "
                    "available",
                ):
                    doe_obj.optimize_experiments(n_exp=2)


if __name__ == "__main__":
    unittest.main()
