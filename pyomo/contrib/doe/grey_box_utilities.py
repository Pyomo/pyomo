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
import logging
from scipy.sparse import coo_matrix

from pyomo.common.dependencies import (
    numpy as np,
)

from pyomo.contrib.doe import ObjectiveLib

from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock, ExternalGreyBoxModel

import pyomo.environ as pyo


class FIMExternalGreyBox(ExternalGreyBoxModel):
    def __init__(
            self,
            doe_object,
            obj_option="determinant",
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

        # Check if the doe_object has model components that are required
        # TODO: add checks for the model --> doe_object.model needs FIM; all other checks should
        #       have been satisfied before the FIM is created. Can add check for unknown_parameters...
        self.obj_option = ObjectiveLib(obj_option)

        # Create logger for FIM egb object
        self.logger = logging.getLogger(__name__)

        # If logger level is None, use doe_object's logger level
        if logger_level is None:
            logger_level = doe_object.logger.getLevel()

        self.logger.setLevel(level=logger_level)

    
    def input_names(self):
        # ToDo: add input names from the FIM coming in from
        # Question --> Should we avoid fragility and pass model components?
        #              Then we can grab names from the model components?
        #              Or is the fragility user-specified strings?
        return

    def equality_constraint_names(self):
        # ToDo: Are there any objectives that will have constraints?
        return

    def output_names(self):
        # ToDo: add output name for the variable. This may have to be
        # an input from the user. Or it could depend on the usage of
        # the ObjectiveLib Enum object, which should have an associated
        # name for the objective function at all times.
        return

    def set_input_values(self, input_values):
        # ToDo: update this to add checks and update if necessary
        # Assert that the names and inputs values have the same
        # length here.
        self._input_values = list(input_values)

    def evaluate_equality_constraints(self):
        # ToDo: are there any objectives that will have constraints?
        return

    def evaluate_outputs(self):
        # ToDo: Take the objective function option and perform the
        # mathematical action to get the objective.
        return np.asarray([], dtype=np.float64)

    def finalize_block_construction(self, pyomo_block):
        # Set bounds on the inputs/outputs
        # This will depend on the objective used
        
        # No return statement
        pass

    def evaluate_jacobian_equality_constraints(self):
        # ToDo: Do any objectives require constraints?

        # Returns coo_matrix of the correct shape
        return
        
    def evaluate_jacobian_outputs(self):
        # ToDo: compute the jacobian of the objective function with
        # respect to the fisher information matrix. Then return
        # a coo_matrix that aligns with what IPOPT will expect.
        #
        # ToDo: there will be significant bookkeeping for more
        # complicated objective functions and the Hessian

        # Returns coo_matrix of the correct shape
        return

    # Beyond here is for Hessian information
    def set_equality_constraint_multipliers(self, eq_con_multiplier_values):
        # ToDo: Do any objectives require constraints?
        # Assert lengths match
        self._eq_con_mult_values = np.asarray(eq_con_multiplier_values, dtype=np.float64)

    def set_output_constraint_multipliers(self, output_con_multiplier_values):
        # ToDo: Do any objectives require constraints?
        # Assert length matches
        self._output_con_mult_values = np.asarray(output_con_multiplier_values, dtype=np.float64)

    def evaluate_hessian_equality_constraints(self):
        # ToDo: Do any objectives require constraints?

        # Returns coo_matrix of the correct shape
        return

    def evaluate_hessian_outputs(self):
        # ToDo: Add for objectives where we can define the Hessian
        #
        # ToDo: significant bookkeeping if the hessian's require vectorized
        # operations. Just need mapping that works well and we are good.
        
        # Returns coo_matrix of the correct shape
        return
