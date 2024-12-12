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

from enum import Enum
import itertools
import logging
from scipy.sparse import coo_matrix

from pyomo.common.dependencies import (
    numpy as np,
)

from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel

import pyomo.environ as pyo


class FIMExternalGreyBox(ExternalGreyBoxModel):
    def __init__(
            self,
            doe_object,
            objective_option="determinant",
            logger_level=None,
    ):
        """
        Grey box model for metrics on the FIM. This methodology reduces numerical complexity for the
        computation of FIM metrics related to eigenvalue decomposition.

        Parameters
        ----------
        doe_object:
           Design of Experiments object that contains a built model (with sensitivity matrix, Q, and 
           fisher information matrix, FIM). The external grey box model will utilize elements of the
           doe_object's model to build the FIM metric with consistent naming. 
        obj_option:
           String representation of the objective option. Current available option is ``determinant``.
           Other options that are planned to be implemented soon are ``minimum_eig`` (E-optimality), 
           and ``condition_number`` (modified E-optimality). default option is ``determinant``
        logger_level:
           logging level to be specified if different from doe_object's logging level. default value
           is None, or equivalently, use the logging level of doe_object. Use logging.DEBUG for all
           messages.
        """

        if doe_object is None:
            raise ValueError("DoE Object must be provided to build external grey box of the FIM.")

        self.doe_object = doe_object

        # Grab parameter list from the doe_object model
        self._param_names = [i for i in self.doe_object.model.parameter_names]
        
        # Check if the doe_object has model components that are required
        # TODO: add checks for the model --> doe_object.model needs FIM; all other checks should
        #       have been satisfied before the FIM is created. Can add check for unknown_parameters...
        self.objective_option = objective_option.name  # Add failsafe to make sure this is ObjectiveLib object?
        # Will anyone ever call this without calling DoE? --> intended to be no; but maybe more utility?

        # Create logger for FIM egb object
        self.logger = logging.getLogger(__name__)

        # If logger level is None, use doe_object's logger level
        if logger_level is None:
            logger_level = doe_object.logger.getLevel()

        self.logger.setLevel(level=logger_level)

        # Set initial values for inputs
        self._input_values = np.asarray(self.doe_object.fim_initial.flatten(), dtype=np.float64)

    
    def input_names(self):
        # Cartesian product gives us matrix indicies flattened in row-first format
        input_names_list = list(itertools.product(self._param_names, self._param_names))
        return input_names_list

    def equality_constraint_names(self):
        # ToDo: Are there any objectives that will have constraints?
        return []

    def output_names(self):
        # ToDo: add output name for the variable. This may have to be
        # an input from the user. Or it could depend on the usage of
        # the ObjectiveLib Enum object, which should have an associated
        # name for the objective function at all times.
        return ["log_det", ]  # Change for hard-coded D-optimality

    def set_input_values(self, input_values):
        # Set initial values to be flattened initial FIM (aligns with input names)
        np.copyto(self._input_values, input_values)
        #self._input_values = list(self.doe_object.fim_initial.flatten())

    def evaluate_equality_constraints(self):
        # ToDo: are there any objectives that will have constraints?
        return None

    def evaluate_outputs(self):
        # ToDo: Take the objective function option and perform the
        # mathematical action to get the objective.

        # CALCULATE THE INVERSE VALUE
        # CHANGE HARD-CODED LOG DET
        current_FIM = self._input_values
        M = np.asarray(current_FIM, dtype=np.float64).reshape(len(self._param_names), len(self._param_names))

        # Trying symmetry calculation?
        #M = np.multiply(M, np.tril(np.ones((len(self._param_names), len(self._param_names)))))
        #M = M + M.transpose() - np.multiply(M, np.eye(len(self._param_names)))
        
        (sign, logdet) = np.linalg.slogdet(M)
        
        return np.asarray([logdet, ], dtype=np.float64)

    def finalize_block_construction(self, pyomo_block):
        # Set bounds on the inputs/outputs
        # Set initial values of the inputs/outputs
        # This will depend on the objective used

        # Initialize grey box FIM values
        for ind, val in enumerate(self.input_names()):
            pyomo_block.inputs[val] = self.doe_object.fim_initial.flatten()[ind]

        # Initialize log_determinant value
        pyomo_block.outputs["log_det"] = 0  # Remember to change hardcoded name

    def evaluate_jacobian_equality_constraints(self):
        # ToDo: Do any objectives require constraints?

        # Returns coo_matrix of the correct shape
        return None
        
    def evaluate_jacobian_outputs(self):
        # ToDo: compute the jacobian of the objective function with
        # respect to the fisher information matrix. Then return
        # a coo_matrix that aligns with what IPOPT will expect.
        #
        # ToDo: there will be significant bookkeeping for more
        # complicated objective functions and the Hessian        
        current_FIM = self._input_values
        M = np.asarray(current_FIM, dtype=np.float64).reshape(len(self._param_names), len(self._param_names))

        # Trying symmetry calculation?
        #M = np.multiply(M, np.tril(np.ones((len(self._param_names), len(self._param_names)))))
        #M = M + M.transpose() - np.multiply(M, np.eye(len(self._param_names)))
        
        Minv = np.linalg.pinv(M)
        eig, _ = np.linalg.eig(M)
        if min(eig) <= 1:
            print("Warning: {:0.6f}".format(min(eig)))

        # Since M is symmetric, the derivative of logdet(M) w.r.t M is
        # 2*inverse(M) - diagonal(inverse(M)) ADD SOURCE
        jac_M = 2*Minv - np.diagonal(Minv)

        # Rows are the integer division by number of columns
        M_rows = np.arange(len(jac_M.flatten())) // jac_M.shape[1]

        # Columns are the remaindar (mod) by number of rows
        M_cols = np.arange(len(jac_M.flatten())) % jac_M.shape[0]

        # Need to be flat?
        M_rows = np.zeros((len(jac_M.flatten()), 1)).flatten()

        # Need to be flat?
        M_cols = np.arange(len(jac_M.flatten()))
        
        # Returns coo_matrix of the correct shape
        #print(coo_matrix((jac_M.flatten(), (M_rows, M_cols)), shape=(1, len(jac_M.flatten()))))
        return coo_matrix((jac_M.flatten(), (M_rows, M_cols)), shape=(1, len(jac_M.flatten())))

    # Beyond here is for Hessian information
    def set_equality_constraint_multipliers(self, eq_con_multiplier_values):
        # ToDo: Do any objectives require constraints?
        # Assert lengths match
        self._eq_con_mult_values = np.asarray(eq_con_multiplier_values, dtype=np.float64)

    def set_output_constraint_multipliers(self, output_con_multiplier_values):
        # ToDo: Do any objectives require constraints?
        # Assert length matches
        self._output_con_mult_values = np.asarray(output_con_multiplier_values, dtype=np.float64)

#    def evaluate_hessian_equality_constraints(self):
        # ToDo: Do any objectives require constraints?

        # Returns coo_matrix of the correct shape
#        return None

#    def evaluate_hessian_outputs(self):
        # ToDo: Add for objectives where we can define the Hessian
        #
        # ToDo: significant bookkeeping if the hessian's require vectorized
        # operations. Just need mapping that works well and we are good.
        
        # Returns coo_matrix of the correct shape
#        return None
