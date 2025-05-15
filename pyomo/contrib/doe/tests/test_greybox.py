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
import json
import os.path

from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    pandas as pd,
    pandas_available,
)
from pyomo.common.fileutils import this_file_dir
import pyomo.common.unittest as unittest

from pyomo.contrib.doe import DesignOfExperiments, FIMExternalGreyBox
from pyomo.contrib.doe.examples.reactor_example import (
    ReactorExperiment as FullReactorExperiment,
)

import pyomo.environ as pyo

from pyomo.opt import SolverFactory

ipopt_available = SolverFactory("ipopt").available()

currdir = this_file_dir()
file_path = os.path.join(currdir, "..", "examples", "result.json")

with open(file_path) as f:
    data_ex = json.load(f)

data_ex["control_points"] = {float(k): v for k, v in data_ex["control_points"].items()}

_FD_EPSILON = 1e-6  # Epsilon for numerical comparison of derivatives

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
            FIM_perturbed[i, j] += _FD_EPSILON

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
            diff = (new_value_ij - unperturbed_value) / _FD_EPSILON

            numerical_derivative[i, j] = diff

    return numerical_derivative


def get_numerical_second_derivative(grey_box_object=None):
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
                    FIM_perturbed_1[i, j] += _FD_EPSILON
                    FIM_perturbed_1[k, l] += _FD_EPSILON

                    FIM_perturbed_2[i, j] += _FD_EPSILON
                    FIM_perturbed_2[k, l] += -_FD_EPSILON

                    FIM_perturbed_3[i, j] += -_FD_EPSILON
                    FIM_perturbed_3[k, l] += _FD_EPSILON

                    FIM_perturbed_4[i, j] += -_FD_EPSILON
                    FIM_perturbed_4[k, l] += -_FD_EPSILON

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
                    ) / (4 * _FD_EPSILON**2)

                    numerical_derivative[i, j, k, l] = diff

    return numerical_derivative


def get_FIM_FIMPrior_Q_L(doe_obj=None):
    """
    Helper function to retrieve results to compare.

    """
    model = doe_obj.model

    n_param = doe_obj.n_parameters
    n_y = doe_obj.n_experiment_outputs

    FIM_vals = [
        pyo.value(model.fim[i, j])
        for i in model.parameter_names
        for j in model.parameter_names
    ]
    FIM_prior_vals = [
        pyo.value(model.prior_FIM[i, j])
        for i in model.parameter_names
        for j in model.parameter_names
    ]
    if hasattr(model, "L"):
        L_vals = [
            pyo.value(model.L[i, j])
            for i in model.parameter_names
            for j in model.parameter_names
        ]
    else:
        L_vals = [[0] * n_param] * n_param
    Q_vals = [
        pyo.value(model.sensitivity_jacobian[i, j])
        for i in model.output_names
        for j in model.parameter_names
    ]
    sigma_inv = [1 / v for k, v in model.scenario_blocks[0].measurement_error.items()]
    param_vals = np.array(
        [[v for k, v in model.scenario_blocks[0].unknown_parameters.items()]]
    )

    FIM_vals_np = np.array(FIM_vals).reshape((n_param, n_param))
    FIM_prior_vals_np = np.array(FIM_prior_vals).reshape((n_param, n_param))

    for i in range(n_param):
        for j in range(n_param):
            if j < i:
                FIM_vals_np[j, i] = FIM_vals_np[i, j]

    L_vals_np = np.array(L_vals).reshape((n_param, n_param))
    Q_vals_np = np.array(Q_vals).reshape((n_y, n_param))

    sigma_inv_np = np.zeros((n_y, n_y))

    for ind, v in enumerate(sigma_inv):
        sigma_inv_np[ind, ind] = v

    return FIM_vals_np, FIM_prior_vals_np, Q_vals_np, L_vals_np, sigma_inv_np


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
    # Change when we can access other solvers
    solver = SolverFactory("ipopt")
    solver.options["linear_solver"] = "MUMPS"
    args['solver'] = solver
    args['tee'] = False
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

    doe_obj = DesignOfExperiments(**DoE_args)
    doe_obj.create_doe_model()

    grey_box_object = FIMExternalGreyBox(
        doe_object=doe_obj, objective_option=doe_obj.objective_option
    )

    return doe_obj, grey_box_object


@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
@unittest.skipIf(not numpy_available, "Numpy is not available")
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
        self.assertTrue(np.all(np.isclose(jac, jac_FD)))

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
        self.assertTrue(np.all(np.isclose(jac, jac_FD)))

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
        self.assertTrue(np.all(np.isclose(jac, jac_FD)))

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
        self.assertTrue(np.all(np.isclose(jac, jac_FD)))

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


if __name__ == "__main__":
    unittest.main()
