from scipy.sparse import coo_matrix

from pyomo.common.dependencies import numpy as np

from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock

import pyomo.environ as pyo


class SumEGB(ExternalGreyBoxModel):
    def __init__(self, n_inputs):
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
        self._n_inputs = n_inputs
        
        self._input_names_list = ["x_" + str(i) for i in range(self._n_inputs)]
        
        self._n_inputs = len(self._input_values)
        # print(self._input_values)

    def input_names(self):
        # Cartesian product gives us matrix indices flattened in row-first format
        # Can use itertools.combinations(self._param_names, 2) with added
        # diagonal elements, or do double for loops if we switch to upper triangular
        # input_names_list = list(itertools.product(self._param_names, self._param_names))
        return self._inputs_names_list

    def equality_constraint_names(self):
        # ToDo: Are there any objectives that will have constraints?
        return []

    def output_names(self):
        # ToDo: add output name for the variable. This may have to be
        # an input from the user. Or it could depend on the usage of
        # the ObjectiveLib Enum object, which should have an associated
        # name for the objective function at all times.
        
        return ["obj", ]

    def set_input_values(self, input_values):
        # Set initial values to be flattened initial FIM (aligns with input names)
        np.copyto(self._input_values, input_values)
        # self._input_values = list(self.doe_object.fim_initial.flatten())

    def evaluate_equality_constraints(self):
        # ToDo: are there any objectives that will have constraints?
        return None

    def evaluate_outputs(self):
        # Evaluates the objective value for the specified
        # ObjectiveLib type.
        
        return np.asarray([obj_value], dtype=np.float64)

    def finalize_block_construction(self, pyomo_block):
        # Set bounds on the inputs/outputs
        # Set initial values of the inputs/outputs
        # This will depend on the objective used

        # Initialize grey box FIM values
        for ind, val in enumerate(self.input_names()):
            pyomo_block.inputs[val] = ind
            pyomo_block.inputs[val].setlb(0)
            pyomo_block.inputs[val].setub(20)
        
        pyomo_block.outputs["obj"] = sum(range(n_inputs))

    def evaluate_jacobian_equality_constraints(self):
        # ToDo: Do any objectives require constraints?

        # Returns coo_matrix of the correct shape
        return None

    def evaluate_jacobian_outputs(self):
        # Compute the jacobian of the objective function with
        # respect to the fisher information matrix. Then return
        # a coo_matrix that aligns with what IPOPT will expect.
        #
        # ToDo: there will be significant bookkeeping for more
        # complicated objective functions and the Hessian
        
        jac_M = np.eye(self._n_inputs)
        M_rows = np.zeros((len(jac_M.flatten()), 1)).flatten()
        M_cols = np.arange(len(jac_M.flatten()))
        
        return coo_matrix(
            (jac_M.flatten(), (M_rows, M_cols)), shape=(1, len(jac_M.flatten()))
        )

    # Beyond here is for Hessian information
    def set_equality_constraint_multipliers(self, eq_con_multiplier_values):
        # ToDo: Do any objectives require constraints?
        # Assert lengths match
        self._eq_con_mult_values = np.asarray(
            eq_con_multiplier_values, dtype=np.float64
        )

    def set_output_constraint_multipliers(self, output_con_multiplier_values):
        # ToDo: Do any objectives require constraints?
        # Assert length matches
        self._output_con_mult_values = np.asarray(
            output_con_multiplier_values, dtype=np.float64
        )


# Simple grey box problem to test determinism.
m = pyo.ConcreteModel()

grey_box = SumEGB(5)

m.egb_block = ExternalGreyBoxBlock(external_model=grey_box)

solver = pyo.SolverFactory("cyipopt")
solver.solve(m, tee=True)