import numpy as np
import pyomo.environ as pyo
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
    ExternalGreyBoxModel,
    ExternalGreyBoxBlock,
)

import inspect
from pathlib import Path


try:
    inspect.stack()[1]
    print(Path(inspect.stack()[1].filename).absolute)
except:
    print("No stack()[1]...")


class LogDetModel(ExternalGreyBoxModel):
    def __init__(
        self,
        n_parameters=2,
        initial_fim=None,
        use_exact_derivatives=True,
        print_level=0,
    ):
        """
        Greybox model to compute the log determinant of a sqaure symmetric matrix.

        Arguments
        ---------
        n_parameters: int
            Number of parameters in the model. The square symmetric matrix is of shape n_parameters*n_parameters
        initial_fim: dict
            key: tuple (i,j) where i, j are the row, column number of FIM. value: FIM[i,j]
            Initial value of the matrix. If None, the identity matrix is used.
        use_exact_derivatives: bool
            If True, the exact derivatives are used.
            If False, the finite difference approximation can be used, but not recommended/tested.
        print_level: integer
            0 (default): no extra output
            1: minimal info to indicate if initialized well
                print the following:
                - initial FIM received by the grey-box moduel
            2: intermediate info for debugging
                print all the level 1 print statements, plus:
                - the FIM output of the current iteration, both the output as the FIM matrix, and the flattened vector
            3: all details for debugging
                print all the level 2 print statements, plus:
                - the log determinant of the FIM output of the current iteration
                - the eigen values of the FIM output of the current iteration

        Return
        ------
        None
        """
        trash = input(str(inspect.stack()[0][3]))

        self._use_exact_derivatives = use_exact_derivatives
        self.print_level = print_level
        self.n_parameters = n_parameters
        # make sure it's integer since this is a number of inputs that shouldn't be fractional
        self.num_input = int(
            n_parameters + (n_parameters * n_parameters - n_parameters) / 2
        )
        self.initial_fim = initial_fim

        # variable to store the output value
        # Output constraint multiplier values. This is a 1-element vector because there is one output
        self._output_con_mult_values = np.zeros(1)

        if not use_exact_derivatives:
            raise NotImplementedError("use_exact_derivatives == False not supported")

    def input_names(self):
        """Return the names of the inputs.
        Define only the upper triangle of FIM because FIM is symmetric

        Return
        ------
        input_name_list: a list of the names of inputs
        """
        # trash = input(str(inspect.stack()[0][3]))

        # store the input names as a tuple
        input_name_list = []
        # loop over parameters
        for i in range(self.n_parameters):
            # loop over upper triangle
            for j in range(i, self.n_parameters):
                input_name_list.append((i, j))

        return input_name_list

    def equality_constraint_names(self):
        """Return the names of the equality constraints."""
        # no equality constraints
        # trash = input(str(inspect.stack()[0][3]))
        return []

    def output_names(self):
        """Return the names of the outputs."""
        # trash = input(str(inspect.stack()[0][3]))
        return ["log_det"]

    def set_output_constraint_multipliers(self, output_con_multiplier_values):
        """
        Set the values of the output constraint multipliers.

        Arguments
        ---------
        output_con_multiplier_values: a scalar number for the output constraint multipliers
        """
        # trash = input(str(inspect.stack()[0][3]))

        # because we only have one output constraint, the length is 1
        if len(output_con_multiplier_values) != 1:
            raise ValueError("Output should be a scalar value. ")

        np.copyto(self._output_con_mult_values, output_con_multiplier_values)

    def finalize_block_construction(self, pyomo_block):
        """
        Finalize the construction of the ExternalGreyBoxBlock.
        This function initializes the inputs with an initial value

        Arguments
        ---------
        pyomo_block: pass the created pyomo block here
        """
        # ele_to_order map the input position in FIM, like (a,b), to its flattend index
        # for e.g., ele_to_order[(0,0)] = 0
        # trash = input(str(inspect.stack()[0][3]))

        ele_to_order = {}
        count = 0

        if self.print_level >= 1:
            if self.initial_fim is not None:
                print("Grey-box initialize inputs with: ", self.initial_fim)
            else:
                print("Grey-box initialize inputs with an identity matrix.")

        # only generating upper triangular part
        # loop over parameters
        for i in range(self.n_parameters):
            # loop over parameters from current parameter to end
            for j in range(i, self.n_parameters):
                # flatten (i,j)
                ele_to_order[(i, j)] = count
                # this tuple is the position of this input in the FIM
                tuple_name = (i, j)

                # if an initial FIM is given, we can initialize with these values
                if self.initial_fim is not None:
                    pyomo_block.inputs[tuple_name].value = self.initial_fim[tuple_name]

                # if not given initial FIM, we initialize with an identity matrix
                else:
                    # identity matrix
                    if i == j:
                        pyomo_block.inputs[tuple_name].value = 1
                    else:
                        pyomo_block.inputs[tuple_name].value = 0

                count += 1

        self.ele_to_order = ele_to_order

    def set_input_values(self, input_values):
        """
        Set the values of the inputs.
        This function refers to the notebook:
        https://colab.research.google.com/drive/1VplaeOTes87oSznboZXoz-q5W6gKJ9zZ?usp=sharing

        Arguments
        ---------
        input_values: input initial values
        """
        # see the colab link in the doc string for why this should be a list
        self._input_values = list(input_values)

        # trash = input(str(inspect.stack()[0][3]))

    def evaluate_equality_constraints(self):
        """Evaluate the equality constraints.
        Return None because there are no equality constraints.
        """
        trash = input(str(inspect.stack()[0][3]))

        return None

    def evaluate_outputs(self):
        """
        Evaluate the output of the model.
        We call numpy here to compute the logdet of FIM. slogdet is used to avoid ill-conditioning issue
        This function refers to the notebook:
        https://colab.research.google.com/drive/1VplaeOTes87oSznboZXoz-q5W6gKJ9zZ?usp=sharing

        Return
        ------
        logdet: a one-element numpy array, containing the log det value as float
        """
        # trash = input(str(inspect.stack()[0][3]))

        # form matrix as a list of lists
        M = self._extract_and_assemble_fim()

        # compute log determinant
        (sign, logdet) = np.linalg.slogdet(M)

        if self.print_level >= 2:
            print("iteration")
            print("\n Consider M =\n", M)
            print("Solution: ", self._input_values)
            if self.print_level == 3:
                print("   logdet = ", logdet, "\n")
                print("Eigvals:", np.linalg.eigvals(M))

        # see the colab link in the doc string for why this should be a array with dtype as float64
        return np.asarray([logdet], dtype=np.float64)

    def evaluate_jacobian_equality_constraints(self):
        """Evaluate the Jacobian of the equality constraints."""
        # trash = input(str(inspect.stack()[0][3]))
        return None

    def _extract_and_assemble_fim(self):
        """
        This function make the flattened inputs back into the shape of an FIM

        Return
        ------
        M: a numpy array containing FIM.
        """
        # trash = input(str(inspect.stack()[0][3]))

        # FIM shape Np*Np
        M = np.zeros((self.n_parameters, self.n_parameters))
        # loop over parameters.
        # Expand here to be the full matrix.
        for i in range(self.n_parameters):
            for k in range(self.n_parameters):
                #  get symmetry part.
                # only have upper triangle, so the smaller index is the row number
                row_number, col_number = min(i, k), max(i, k)
                M[i, k] = self._input_values[
                    self.ele_to_order[(row_number, col_number)]
                ]

        return M

    def evaluate_jacobian_outputs(self):
        """
        Evaluate the Jacobian of the outputs.

        Return
        ------
        A sparse matrix, containing the first order gradient of the OBJ, in the shape [1,N_input]
        where N_input is the No. of off-diagonal elements//2 + Np
        """
        # trash = input(str(inspect.stack()[0][3]))

        if self._use_exact_derivatives:
            M = self._extract_and_assemble_fim()

            # compute pseudo inverse to be more numerically stable
            Minv = np.linalg.pinv(M)

            # compute gradient of log determinant
            row = np.zeros(self.num_input)  # to store row index
            col = np.zeros(self.num_input)  # to store column index
            data = np.zeros(self.num_input)  # to store data

            # construct gradients as a sparse matrix
            # loop over the upper triangular
            # loop over parameters
            for i in range(self.n_parameters):
                # loop over parameters from current parameter to end
                for j in range(i, self.n_parameters):
                    order = self.ele_to_order[(i, j)]
                    # diagonal elements. See Eq. 16 in paper for explanation
                    if i == j:
                        row[order], col[order], data[order] = (0, order, Minv[i, j])
                    # off-diagonal elements
                    else:  # factor = 2 since it is a symmetric matrix. See Eq. 16 in paper for explanation
                        row[order], col[order], data[order] = (0, order, 2 * Minv[i, j])
            # sparse matrix
            return coo_matrix((data, (row, col)), shape=(1, self.num_input))


# import idaes

m = pyo.ConcreteModel()
m.params = pyo.Set(initialize=[0, 1])
m.params_mat = m.params * m.params
m.M = pyo.Var(m.params_mat, bounds=(0, 50), initialize=1)

print('Made base model.')

ex_model = LogDetModel(
    n_parameters=2,
    initial_fim=None,
    # initial_fim=np.ones((2, 2)),
    print_level=1,
)

print('Added logdet model')
m.egb = ExternalGreyBoxBlock(external_model=ex_model)

print('Added as external grey box.')

# constraining outputs
m.M_con1 = pyo.Constraint(expr=(m.M[(0, 0)] == m.egb.inputs[(0, 0)]))
m.M_con2 = pyo.Constraint(expr=(m.M[(0, 1)] == m.egb.inputs[(0, 1)]))
m.M_con3 = pyo.Constraint(expr=(m.M[(1, 1)] == m.egb.inputs[(1, 1)]))
m.M_con4 = pyo.Constraint(expr=(m.M[(1, 0)] == m.M[(0, 1)]))

print('Added constraints on symmetry for FIM.')

m.logdet = pyo.Expression(rule=m.egb.outputs["log_det"])

m.obj = pyo.Objective(expr=m.logdet, sense=pyo.maximize)

print('Added objective function. Solve is next action.')

solver = pyo.SolverFactory("cyipopt")
solver.config.options['hessian_approximation'] = 'limited-memory'
# solver.config.options['mu_strategy'] = 'monotone'
# solver.config.options['linear_solver'] = 'ma27'

solver.solve(m, tee=True)

m.M.pprint()

# m.pprint()
