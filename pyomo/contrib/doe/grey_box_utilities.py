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

from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    scipy,
    scipy_available,
)

from enum import Enum
import itertools
import logging

if scipy_available and numpy_available:
    from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel

import pyomo.environ as pyo


class FIMExternalGreyBox(
    ExternalGreyBoxModel if (scipy_available and numpy_available) else object
):
    def __init__(self, doe_object, objective_option="determinant", logger_level=None):
        """
        Grey box model for metrics on the FIM. This methodology reduces
        numerical complexity for the computation of FIM metrics related
        to eigenvalue decomposition.

        Parameters
        ----------
        doe_object:
           Design of Experiments object that contains a built model
           (with sensitivity matrix, Q, and fisher information matrix, FIM).
           The external grey box model will utilize elements of the
           `doe_object` model to build the FIM metric with consistent naming.
        obj_option:
           String representation of the objective option. Current available
           options are: ``determinant`` (D-optimality), ``trace`` (A-optimality),
           ``minimum_eigenvalue`` (E-optimality), ``condition_number``
           (modified E-optimality).
           default: ``determinant``
        logger_level:
           logging level to be specified if different from doe_object's logging level.
           default: None, or equivalently, use the logging level of doe_object.
                    NOTE: Use logging.DEBUG for all messages.
        """

        if doe_object is None:
            raise ValueError(
                "DoE Object must be provided to build external grey box of the FIM."
            )

        self.doe_object = doe_object

        # Grab parameter list from the doe_object model
        self._param_names = [i for i in self.doe_object.model.parameter_names]
        self._n_params = len(self._param_names)

        # Check if the doe_object has model components that are required
        # TODO: is this check necessary?
        from pyomo.contrib.doe import ObjectiveLib

        objective_option = ObjectiveLib(objective_option)
        self.objective_option = objective_option

        # Create logger for FIM egb object
        self.logger = logging.getLogger(__name__)

        # If logger level is None, use doe_object's logger level
        if logger_level is None:
            logger_level = doe_object.logger.level

        self.logger.setLevel(level=logger_level)

        # Set initial values for inputs
        # Need a mask structure
        self._masking_matrix = np.triu(np.ones_like(self.doe_object.fim_initial))
        self._input_values = np.asarray(
            self.doe_object.fim_initial[self._masking_matrix > 0], dtype=np.float64
        )
        self._n_inputs = len(self._input_values)

    def _get_FIM(self):
        # Grabs the current FIM subject
        # to the input values.
        # This function currently assumes
        # that we use a lower triangular
        # FIM.
        upt_FIM = self._input_values

        # Create FIM in the correct way
        current_FIM = np.zeros_like(self.doe_object.fim_initial)
        # Utilize upper triangular portion of FIM
        current_FIM[np.triu_indices_from(current_FIM)] = upt_FIM
        # Construct lower triangular using the
        # current upper triangle minus the diagonal.
        current_FIM += current_FIM.transpose() - np.diag(np.diag(current_FIM))

        return current_FIM

    def input_names(self):
        # Cartesian product gives us matrix indices flattened in row-first format
        # Can use itertools.combinations(self._param_names, 2) with added
        # diagonal elements, or do double for loops if we switch to upper triangular
        input_names_list = list(
            itertools.combinations_with_replacement(self._param_names, 2)
        )
        return input_names_list

    def equality_constraint_names(self):
        # TODO: Are there any objectives that will have constraints?
        return []

    def output_names(self):
        # TODO: add output name for the variable. This may have to be
        # an input from the user. Or it could depend on the usage of
        # the ObjectiveLib Enum object, which should have an associated
        # name for the objective function at all times.
        from pyomo.contrib.doe import ObjectiveLib

        if self.objective_option == ObjectiveLib.trace:
            obj_name = "A-opt"
        elif self.objective_option == ObjectiveLib.determinant:
            obj_name = "log-D-opt"
        elif self.objective_option == ObjectiveLib.minimum_eigenvalue:
            obj_name = "E-opt"
        elif self.objective_option == ObjectiveLib.condition_number:
            obj_name = "ME-opt"
        else:
            ObjectiveLib(self.objective_option)
        return [obj_name]

    def set_input_values(self, input_values):
        # Set initial values to be flattened initial FIM (aligns with input names)
        np.copyto(self._input_values, input_values)

    def evaluate_equality_constraints(self):
        # TODO: are there any objectives that will have constraints?
        return None

    def evaluate_outputs(self):
        # Evaluates the objective value for the specified
        # ObjectiveLib type.
        current_FIM = self._get_FIM()

        M = np.asarray(current_FIM, dtype=np.float64).reshape(
            self._n_params, self._n_params
        )

        # Change objective value based on ObjectiveLib type.
        from pyomo.contrib.doe import ObjectiveLib

        if self.objective_option == ObjectiveLib.trace:
            obj_value = np.trace(np.linalg.pinv(M))
        elif self.objective_option == ObjectiveLib.determinant:
            (sign, logdet) = np.linalg.slogdet(M)
            obj_value = logdet
        elif self.objective_option == ObjectiveLib.minimum_eigenvalue:
            eig, _ = np.linalg.eig(M)
            obj_value = np.min(eig)
        elif self.objective_option == ObjectiveLib.condition_number:
            eig, _ = np.linalg.eig(M)
            obj_value = np.max(eig) / np.min(eig)
        else:
            ObjectiveLib(self.objective_option)

        return np.asarray([obj_value], dtype=np.float64)

    def finalize_block_construction(self, pyomo_block):
        # Set bounds on the inputs/outputs
        # Set initial values of the inputs/outputs
        # This will depend on the objective used

        # Initialize grey box FIM values
        for ind, val in enumerate(self.input_names()):
            pyomo_block.inputs[val] = self.doe_object.fim_initial[
                self._masking_matrix > 0
            ][ind]

        # Initialize log_determinant value
        from pyomo.contrib.doe import ObjectiveLib

        # Calculate initial values for the output
        output_value = self.evaluate_outputs()[0]

        # Set the value of the output for the given
        # objective function.
        if self.objective_option == ObjectiveLib.trace:
            pyomo_block.outputs["A-opt"] = output_value
        elif self.objective_option == ObjectiveLib.determinant:
            pyomo_block.outputs["log-D-opt"] = output_value
        elif self.objective_option == ObjectiveLib.minimum_eigenvalue:
            pyomo_block.outputs["E-opt"] = output_value
        elif self.objective_option == ObjectiveLib.condition_number:
            pyomo_block.outputs["ME-opt"] = output_value

    def evaluate_jacobian_equality_constraints(self):
        # TODO: Do any objectives require constraints?

        # Returns coo_matrix of the correct shape
        return None

    def evaluate_jacobian_outputs(self):
        # Compute the jacobian of the objective function with
        # respect to the fisher information matrix. Then return
        # a coo_matrix that aligns with what IPOPT will expect.
        current_FIM = self._get_FIM()

        M = np.asarray(current_FIM, dtype=np.float64).reshape(
            self._n_params, self._n_params
        )

        # TODO: Add inertia correction for
        #       negative/small eigenvalues
        eig_vals, eig_vecs = np.linalg.eig(M)
        if min(eig_vals) <= 1e-3:
            pass

        from pyomo.contrib.doe import ObjectiveLib

        if self.objective_option == ObjectiveLib.trace:
            Minv = np.linalg.pinv(M)
            # Derivative formula of A-optimality
            # is -inv(FIM) @ inv(FIM). Add reference to
            # pyomo.DoE 2.0 manuscript S.I.
            jac_M = -Minv @ Minv
        elif self.objective_option == ObjectiveLib.determinant:
            Minv = np.linalg.pinv(M)
            # Derivative formula derived using tensor
            # calculus. Add reference to pyomo.DoE 2.0
            # manuscript S.I.
            jac_M = 0.5 * (Minv + Minv.transpose())
        elif self.objective_option == ObjectiveLib.minimum_eigenvalue:
            # Obtain minimum eigenvalue location
            min_eig_loc = np.argmin(eig_vals)

            # Grab eigenvector associated with
            # the minimum eigenvalue and make
            # it a matrix. This is so we can
            # use matrix operations later in
            # the code.
            min_eig_vec = np.array([eig_vecs[:, min_eig_loc]])

            # Calculate the derivative matrix.
            # This is the expansion product of
            # the eigenvector we grabbed in
            # the previous line of code.
            jac_M = min_eig_vec * np.transpose(min_eig_vec)
        elif self.objective_option == ObjectiveLib.condition_number:
            # Obtain minimum (and maximum) eigenvalue location(s)
            min_eig_loc = np.argmin(eig_vals)
            max_eig_loc = np.argmax(eig_vals)

            min_eig = np.min(eig_vals)
            max_eig = np.max(eig_vals)

            # Grab eigenvector associated with
            # the min (and max) eigenvalue and make
            # it a matrix. This is so we can
            # use matrix operations later in
            # the code.
            min_eig_vec = np.array([eig_vecs[:, min_eig_loc]])
            max_eig_vec = np.array([eig_vecs[:, max_eig_loc]])

            # Calculate the derivative matrix.
            # Similar to minimum eigenvalue,
            # this computation involves two
            # expansion products.
            min_eig_term = min_eig_vec * np.transpose(min_eig_vec)
            max_eig_term = max_eig_vec * np.transpose(max_eig_vec)

            min_eig_epsilon = 2e-16

            # Computing a (hopefully) nonsingular
            # condition number for the jacobian
            # expression.
            safe_cond_number = max_eig / (min_eig + np.sign(min_eig) * min_eig_epsilon)

            # Combining the expression
            jac_M = (
                1
                / (min_eig + np.sign(min_eig) * min_eig_epsilon)
                * (max_eig_term - safe_cond_number * min_eig_term)
            )
        else:
            ObjectiveLib(self.objective_option)

        # Filter jac_M using the
        # masking matrix
        jac_M = jac_M[self._masking_matrix > 0]
        M_rows = np.zeros((len(jac_M.flatten()), 1)).flatten()
        M_cols = np.arange(len(jac_M.flatten()))

        # Returns coo_matrix of the correct shape
        return scipy.sparse.coo_matrix(
            (jac_M.flatten(), (M_rows, M_cols)), shape=(1, len(jac_M.flatten()))
        )

    # Beyond here is for Hessian information
    def set_equality_constraint_multipliers(self, eq_con_multiplier_values):
        # TODO: Do any objectives require constraints?
        # Assert lengths match
        self._eq_con_mult_values = np.asarray(
            eq_con_multiplier_values, dtype=np.float64
        )

    def set_output_constraint_multipliers(self, output_con_multiplier_values):
        # TODO: Do any objectives require constraints?
        # Assert length matches
        self._output_con_mult_values = np.asarray(
            output_con_multiplier_values, dtype=np.float64
        )

    def evaluate_hessian_equality_constraints(self):
        # Returns coo_matrix of the correct shape
        # No constraints so this returns `None`
        return None

    def evaluate_hessian_outputs(self, FIM=None):
        # TODO: significant bookkeeping if the hessian's require vectorized
        # operations. Just need mapping that works well and we are good.
        current_FIM = self._get_FIM()

        M = np.asarray(current_FIM, dtype=np.float64).reshape(
            self._n_params, self._n_params
        )

        # Hessian with correct size for using only the
        # lower (upper) triangle of the FIM
        hess_vals = []
        hess_rows = []
        hess_cols = []

        # Need to iterate over the unique
        # differentials
        input_differentials_2D = itertools.combinations_with_replacement(
            self.input_names(), 2
        )

        from pyomo.contrib.doe import ObjectiveLib

        if self.objective_option == ObjectiveLib.trace:
            # Grab Inverse
            Minv = np.linalg.pinv(M)

            # Also grab inverse squared
            Minv_sq = Minv @ Minv

            for current_differential in input_differentials_2D:
                # Row will be the location of the
                # first ordered pair (d1) in input names
                #
                # Col will be the location of the
                # second ordered pair (d2) in input names
                d1, d2 = current_differential
                row = self.input_names().index(d2)
                col = self.input_names().index(d1)

                # Grabbing the ordered quadruple (i, j, k, l)
                # `location` here refers to the index in the
                # self._param_names list
                #
                # i is the location of the first element of d1
                # j is the location of the second element of d1
                # k is the location of the first element of d2
                # l is the location of the second element of d2
                i = self._param_names.index(d1[0])
                j = self._param_names.index(d1[1])
                k = self._param_names.index(d2[0])
                l = self._param_names.index(d2[1])

                # New Formula (tested with finite differencing)
                # Will be cited from the Pyomo.DoE 2.0 paper
                hess_vals.append(
                    (Minv[i, l] * Minv_sq[k, j]) + (Minv_sq[i, l] * Minv[k, j])
                )
                hess_rows.append(row)
                hess_cols.append(col)

        elif self.objective_option == ObjectiveLib.determinant:
            # Grab inverse
            Minv = np.linalg.pinv(M)

            for current_differential in input_differentials_2D:
                # Row, Col and i, j, k, l values are
                # obtained identically as in the trace
                # for loop above.
                d1, d2 = current_differential
                row = self.input_names().index(d2)
                col = self.input_names().index(d1)

                i = self._param_names.index(d1[0])
                j = self._param_names.index(d1[1])
                k = self._param_names.index(d2[0])
                l = self._param_names.index(d2[1])

                # New Formula (tested with finite differencing)
                # Will be cited from the Pyomo.DoE 2.0 paper
                hess_vals.append(-(Minv[i, l] * Minv[k, j]))
                hess_rows.append(row)
                hess_cols.append(col)
        elif self.objective_option == ObjectiveLib.minimum_eigenvalue:
            # Grab eigenvalues and eigenvectors
            # Also need the min location
            all_eig_vals, all_eig_vecs = np.linalg.eig(M)
            min_eig_loc = np.argmin(all_eig_vals)

            # Grabbing min eigenvalue and corresponding
            # eigenvector
            min_eig = all_eig_vals[min_eig_loc]
            min_eig_vec = np.array([all_eig_vecs[:, min_eig_loc]])

            for current_differential in input_differentials_2D:
                # Row, Col and i, j, k, l values are
                # obtained identically as in the trace
                # for loop above.
                d1, d2 = current_differential
                row = self.input_names().index(d2)
                col = self.input_names().index(d1)

                i = self._param_names.index(d1[0])
                j = self._param_names.index(d1[1])
                k = self._param_names.index(d2[0])
                l = self._param_names.index(d2[1])

                # For lop to iterate over all
                # eigenvalues/vectors
                curr_hess_val = 0
                for curr_eig in range(len(all_eig_vals)):
                    # Skip if we are at the minimum
                    # eigenvalue. Denominator is
                    # zero.
                    if curr_eig == min_eig_loc:
                        continue

                    # Formula derived in Pyomo.DoE Paper
                    curr_hess_val += (
                        1
                        * (
                            min_eig_vec[0, i]
                            * all_eig_vecs[j, curr_eig]
                            * min_eig_vec[0, l]
                            * all_eig_vecs[k, curr_eig]
                        )
                        / (min_eig - all_eig_vals[curr_eig])
                    )
                    curr_hess_val += (
                        1
                        * (
                            min_eig_vec[0, k]
                            * all_eig_vecs[i, curr_eig]
                            * min_eig_vec[0, j]
                            * all_eig_vecs[l, curr_eig]
                        )
                        / (min_eig - all_eig_vals[curr_eig])
                    )
                hess_vals.append(curr_hess_val)
                hess_rows.append(row)
                hess_cols.append(col)
        elif self.objective_option == ObjectiveLib.condition_number:
            pass
        else:
            ObjectiveLib(self.objective_option)

        # Returns coo_matrix of the correct shape
        return scipy.sparse.coo_matrix(
            (np.asarray(hess_vals), (hess_rows, hess_cols)),
            shape=(self._n_inputs, self._n_inputs),
        )
