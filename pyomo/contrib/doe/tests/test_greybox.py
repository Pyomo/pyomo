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
import copy
import itertools
import json
import os.path

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
    from pyomo.contrib.doe.examples.rooney_biegler_example import (
        RooneyBieglerExperimentDoE,
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
        unperturbed_value = np.linalg.cond(current_FIM)

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
                new_value_ij = np.linalg.cond(FIM_perturbed)

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
                        new_values[0] = np.linalg.cond(FIM_perturbed_1)
                        new_values[1] = np.linalg.cond(FIM_perturbed_2)
                        new_values[2] = np.linalg.cond(FIM_perturbed_3)
                        new_values[3] = np.linalg.cond(FIM_perturbed_4)

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
        ordered_pairs = itertools.combinations_with_replacement(range(4), 2)
        ordered_pairs_list = list(itertools.combinations_with_replacement(range(4), 2))
        ordered_quads = itertools.combinations_with_replacement(ordered_pairs, 2)

        numerical_derivative_reduced = np.zeros((10, 10))

        for i in ordered_quads:
            row = ordered_pairs_list.index(i[0])
            col = ordered_pairs_list.index(i[1])
            numerical_derivative_reduced[row, col] = numerical_derivative[
                i[0][0], i[0][1], i[1][0], i[1][1]
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

    experiment = RooneyBieglerExperimentDoE(data={'hour': 2, 'y': 10.3})

    DoE_args = get_standard_args(experiment, fd_method, obj_used)
    DoE_args["use_grey_box_objective"] = True

    # Make a custom grey box solver
    grey_box_solver = SolverFactory("cyipopt")
    grey_box_solver.config.options["linear_solver"] = "ma57"
    grey_box_solver.config.options['tol'] = 1e-4
    grey_box_solver.config.options['mu_strategy'] = "monotone"

    # Add the grey box solver to DoE_args
    DoE_args["grey_box_solver"] = grey_box_solver

    data = [[1, 8.3], [7, 19.8]]
    FIM_prior = np.zeros((2, 2))
    # Calculate prior using existing experiments
    for i in range(len(data)):
        prev_experiment = RooneyBieglerExperimentDoE(
            data={'hour': data[i][0], 'y': data[i][1]}
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


# Test whether or not cyipopt
# is appropriately calling the
# linear solvers.
bad_message = "Invalid option encountered."
cyipopt_call_working = True
if numpy_available and scipy_available and ipopt_available and cyipopt_available:
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

        ME_opt = np.linalg.cond(testing_matrix)

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
        jac += jac.transpose() - np.diag(np.diag(jac))

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
        jac += jac.transpose() - np.diag(np.diag(jac))

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
        jac += jac.transpose() - np.diag(np.diag(jac))

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
        jac += jac.transpose() - np.diag(np.diag(jac))

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

        # Grab the Jacobian values
        hess_vals_from_gb = grey_box_object.evaluate_hessian_outputs().toarray()

        # Recover the Jacobian in Matrix Form
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

        # Grab the Jacobian values
        hess_vals_from_gb = grey_box_object.evaluate_hessian_outputs().toarray()

        # Recover the Jacobian in Matrix Form
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

        # Grab the Jacobian values
        hess_vals_from_gb = grey_box_object.evaluate_hessian_outputs().toarray()

        # Recover the Jacobian in Matrix Form
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
        ME_opt_val = np.linalg.cond(testing_matrix + np.eye(4))

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
            "DoE Object must be provided to build external grey box of the FIM.",
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
        optimal_experimental_designs = [np.array([2.24, 4.33]), np.array([10.00, 4.35])]
        objective_option = "determinant"
        doe_object, grey_box_object = make_greybox_and_doe_objects_rooney_biegler(
            objective_option=objective_option
        )

        # Set to use the grey box objective
        doe_object.use_grey_box = True

        # Solve the model
        doe_object.run_doe()

        print("Termination Message")
        print(doe_object.results["Termination Message"])
        print(cyipopt_call_working)
        print(bad_message in doe_object.results["Termination Message"])
        print("End Message")

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
            np.array([1.94, 0.0295]),
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
    def test_solve_E_optimality_minimum_eigenvalue(self):
        # Two locally optimal design points exist
        # (time, optimal objective value)
        # Here, the objective value is
        # minimum eigenvalue of the FIM
        optimal_experimental_designs = [
            np.array([1.92, 36.018]),
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
    def test_solve_ME_optimality_condition_number(self):
        # Two locally optimal design points exist
        # (time, optimal objective value)
        # Here, the objective value is
        # condition number of the FIM
        optimal_experimental_designs = [
            np.array([1.59, 15.22]),
            np.array([10.00, 27.675]),
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


if __name__ == "__main__":
    unittest.main()
