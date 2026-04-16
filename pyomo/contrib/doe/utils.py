# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
#
# Pyomo.DoE was produced under the Department of Energy Carbon Capture Simulation
# Initiative (CCSI), and is copyright (c) 2022 by the software owners:
# TRIAD National Security, LLC., Lawrence Livermore National Security, LLC.,
# Lawrence Berkeley National Laboratory, Pacific Northwest National Laboratory,
# Battelle Memorial Institute, University of Notre Dame,
# The University of Pittsburgh, The University of Texas at Austin,
# University of Toledo, West Virginia University, et al. All rights reserved.
#
# NOTICE. This Software was developed under funding from the
# U.S. Department of Energy and the U.S. Government consequently retains
# certain rights. As such, the U.S. Government has been granted for itself
# and others acting on its behalf a paid-up, nonexclusive, irrevocable,
# worldwide license in the Software to reproduce, distribute copies to the
# public, prepare derivative works, and perform publicly and display
# publicly, and to permit other to do so.
# ____________________________________________________________________________________

import pyomo.environ as pyo

from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.collections import ComponentSet

from pyomo.core.base.param import ParamData
from pyomo.core.base.var import VarData
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd, reverse_ad
from pyomo.core.expr.visitor import identify_variables

import logging

logger = logging.getLogger(__name__)

# This small and positive tolerance is used when checking
# if the prior is negative definite or approximately
# indefinite. It is defined as a tolerance here to ensure
# consistency between the code below and the tests. The
# user should not need to adjust it.
_SMALL_TOLERANCE_DEFINITENESS = 1e-6

# This small and positive tolerance is used to check
# the FIM is approximately symmetric. It is defined as
# a tolerance here to ensure consistency between the code
# below and the tests. The user should not need to adjust it.
_SMALL_TOLERANCE_SYMMETRY = 1e-6

# This small and positive tolerance is used to check
# if the imaginary part of the eigenvalues of the FIM is
# greater than a small tolerance. It is defined as a
# tolerance here to ensure consistency between the code
# below and the tests. The user should not need to adjust it.
_SMALL_TOLERANCE_IMG = 1e-6


# Rescale FIM (a scaling function to help rescale FIM from parameter values)
def rescale_FIM(FIM, param_vals):
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
        param_vals = np.array([param_vals])
    elif isinstance(param_vals, np.ndarray):
        if len(param_vals.shape) > 2 or (
            (len(param_vals.shape) == 2) and (param_vals.shape[0] != 1)
        ):
            raise ValueError(
                "param_vals should be a vector of dimensions: 1 by `n_params`. "
                + "The shape you provided is {}.".format(param_vals.shape)
            )
        if len(param_vals.shape) == 1:
            param_vals = np.array([param_vals])
    else:
        raise ValueError(
            "param_vals should be a list or numpy array of dimensions: 1 by `n_params`"
        )
    # Form the matrix with entries scaling_mat[i, j] = 1 / (theta_i theta_j), where theta_i
    # and theta_j are the i-th and j-th parameter values. The scaled FIM is then
    # computed elementwise as scaled_FIM[i, j] = FIM[i, j] / (theta_i theta_j).
    scaling_mat = (1 / param_vals).transpose().dot((1 / param_vals))
    scaled_FIM = np.multiply(FIM, scaling_mat)
    return scaled_FIM


def check_FIM(FIM):
    """
    Checks that the FIM is square, positive definite, and symmetric.

    Parameters
    ----------
    FIM: 2D numpy array representing the FIM

    Returns
    -------
    None, but will raise error messages as needed
    """
    # Check that the FIM is a square matrix
    if FIM.shape[0] != FIM.shape[1]:
        raise ValueError("FIM must be a square matrix")

    # Compute the eigenvalues of the FIM
    evals = np.linalg.eigvals(FIM)

    # Check if the FIM is positive definite
    if np.min(evals) < -_SMALL_TOLERANCE_DEFINITENESS:
        raise ValueError(
            "FIM provided is not positive definite. It has one or more negative "
            + "eigenvalue(s) less than -{:.1e}".format(_SMALL_TOLERANCE_DEFINITENESS)
        )

    # Check if the FIM is symmetric
    if not np.allclose(FIM, FIM.T, atol=_SMALL_TOLERANCE_SYMMETRY):
        raise ValueError(
            "FIM provided is not symmetric using absolute tolerance {}".format(
                _SMALL_TOLERANCE_SYMMETRY
            )
        )


# Functions to compute FIM metrics
def compute_FIM_metrics(FIM):
    """
    Parameters
    ----------
    FIM : numpy.ndarray
        2D array representing the Fisher Information Matrix (FIM).

    Returns
    -------
    Returns the following metrics as a tuple in the order shown below:

    det_FIM : float
        Determinant of the FIM.
    trace_cov : float
        Trace of the covariance matrix.
    trace_FIM : float
        Trace of the FIM.
    E_vals : numpy.ndarray
        1D array of eigenvalues of the FIM.
    E_vecs : numpy.ndarray
        2D array of eigenvectors of the FIM.
    D_opt : float
        log10(D-optimality) metric.
    A_opt : float
        log10(A-optimality) metric.
    pseudo_A_opt : float
        log10(trace(FIM)) metric.
    E_opt : float
        log10(E-optimality) metric.
    ME_opt : float
        log10(Modified E-optimality) metric.
    """

    # Check whether the FIM is square, positive definite, and symmetric
    check_FIM(FIM)
    # D-optimality uses det(FIM); larger determinant means a smaller parameter
    # confidence ellipsoid, i.e., tighter joint parameter uncertainty.

    det_FIM = np.linalg.det(FIM)
    D_opt = np.log10(det_FIM)

    # Trace of FIM is the pseudo A-optimality, not the proper definition of A-optimality,
    # The trace of covariance is the proper definition of A-optimality
    # trace(FIM) gives a convenient proxy for total information, while
    # trace(FIM^{-1}) gives the standard A-optimality metric based on
    # total parameter variance.
    # A-optimality geometrically minimizes the average
    # squared semi-axis length of the parameter confidence ellipsoid.

    trace_FIM = np.trace(FIM)
    pseudo_A_opt = np.log10(trace_FIM)
    trace_cov = np.trace(np.linalg.pinv(FIM))
    A_opt = np.log10(trace_cov)

    # E-optimality uses the smallest eigenvalue of the FIM, so it targets the
    # worst-identified parameter direction by minimizing the longest axis of the
    # confidence ellipsoid.

    E_vals, E_vecs = np.linalg.eig(FIM)
    E_ind = np.argmin(E_vals.real)  # index of smallest eigenvalue

    # Warn the user if there is a ``large`` imaginary component (should not be)
    if abs(E_vals.imag[E_ind]) > _SMALL_TOLERANCE_IMG:
        logger.warning(
            "Eigenvalue has imaginary component greater than "
            + f"{_SMALL_TOLERANCE_IMG}, contact the developers if this issue persists."
        )

    # If the real value is less than or equal to zero, set the E_opt value to nan
    if E_vals.real[E_ind] <= 0:
        E_opt = np.nan
    else:
        E_opt = np.log10(E_vals.real[E_ind])

    # Modified E-optimality is based on the FIM condition number and penalizes
    # confidence ellipsoids that are highly elongated in one direction.


    ME_opt = np.log10(np.linalg.cond(FIM))

    return (
        det_FIM,
        trace_cov,
        trace_FIM,
        E_vals,
        E_vecs,
        D_opt,
        A_opt,
        pseudo_A_opt,
        E_opt,
        ME_opt,
    )


# Standalone Function for user to calculate FIM metrics directly without using the class
def get_FIM_metrics(FIM):
    """This function calculates the FIM metrics and returns them as a dictionary.

    Parameters
    ----------
    FIM : numpy.ndarray
        2D numpy array of the FIM

    Returns
    -------
    A dictionary containing the following keys:

    "Determinant of FIM" : float
        determinant of the FIM
    "Trace of cov" : float
        trace of the covariance matrix
    "Trace of FIM" : float
        trace of the FIM
    "Eigenvalues" : numpy.ndarray
        eigenvalues of the FIM
    "Eigenvectors" : numpy.ndarray
        eigenvectors of the FIM
    "log10(D-Optimality)" : float
        log10(D-optimality) metric
    "log10(A-Optimality)" : float
        log10(A-optimality) metric
    "log10(Pseudo A-Optimality)" : float
        log10(trace(FIM)) metric
    "log10(E-Optimality)" : float
        log10(E-optimality) metric
    "log10(Modified E-Optimality)" : float
        log10(Modified E-optimality) metric
    """

    (
        det_FIM,
        trace_cov,
        trace_FIM,
        E_vals,
        E_vecs,
        D_opt,
        A_opt,
        pseudo_A_opt,
        E_opt,
        ME_opt,
    ) = compute_FIM_metrics(FIM)

    return {
        "Determinant of FIM": det_FIM,
        "Trace of cov": trace_cov,
        "Trace of FIM": trace_FIM,
        "Eigenvalues": E_vals,
        "Eigenvectors": E_vecs,
        "log10(D-Optimality)": D_opt,
        "log10(A-Optimality)": A_opt,
        "log10(Pseudo A-Optimality)": pseudo_A_opt,
        "log10(E-Optimality)": E_opt,
        "log10(Modified E-Optimality)": ME_opt,
    }


class ExperimentGradients:
    """Utilities for differentiating labeled experiment models.

    This helper implements the symbolic sensitivity path used in Pyomo.DoE.
    Instead of approximating sensitivities by finite-difference perturbations
    of the unknown parameters theta, it differentiates the 
    model F(x, u, theta) = 0 with respect to theta, with the design variables
    u fixed, and solves the resulting auxiliary sensitivity system

        dF/dx * dx/dtheta + dF/dtheta = 0

    to obtain the local output sensitivities dy/dtheta needed for Fisher
    information matrix calculations.
    """

    def __init__(self, experiment_model, symbolic=True, automatic=True, verbose=False):
        self.model = experiment_model
        self.verbose = verbose

        self._analyze_experiment_model()

        self.jac_dict_sd = None
        self.jac_dict_ad = None
        self.jac_measurements_wrt_param = None

        self._requested_symbolic = symbolic
        self._requested_automatic = automatic

        if symbolic or automatic:
            self._setup_differentiation()

    def _analyze_experiment_model(self):
        """Build index mappings for constraints, variables, parameters, and outputs.

        This inspects the labeled experiment model and records the ordered
        constraint and variable lists used later to assemble Jacobian blocks for
        sensitivity calculations for F(x, u, theta) = 0. It also tracks which
        indexed quantities correspond to unknown parameters theta and measured
        outputs y.
        """
        model = self.model

        # Fix the design variables u and unknown parameters theta so the
        # remaining active equations define the model F(x, u, theta) = 0
        # used for sensitivity calculations.
        for v in model.experiment_inputs.keys():
            v.fix()
        for v in model.unknown_parameters.keys():
            v.fix()

        param_set = ComponentSet(model.unknown_parameters.keys())
        output_set = ComponentSet(model.experiment_outputs.keys())
        con_set = ComponentSet()
        var_set = ComponentSet()

        for c in model.component_data_objects(
            pyo.Constraint, descend_into=True, active=True
        ):
            con_set.add(c)
            for v in identify_variables(c.body, include_fixed=False):
                var_set.add(v)

        # The parameters theta may not appear in identify_variables(...,
        # include_fixed=False) after being fixed, but they still need indexed
        # columns in the full Jacobian.
        for p in model.unknown_parameters.keys():
            if p not in var_set:
                var_set.add(p)

        measurement_mapping = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        parameter_mapping = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        measurement_error_included = pyo.Suffix(direction=pyo.Suffix.LOCAL)

        con_list = list(con_set)
        var_list = list(var_set)

        param_index = []
        model_var_index = []
        measurement_index = []

        # Partition the indexed quantities into parameter columns (theta),
        # model-variable columns (x), and measured-output rows (y) for later
        # Jacobian slicing.
        for i, v in enumerate(var_list):
            if v in param_set:
                param_index.append(i)
                parameter_mapping[v] = i
            else:
                model_var_index.append(i)
                if v in output_set:
                    measurement_index.append(i)
                    measurement_error_included[v] = model.measurement_error[v]
                    measurement_mapping[v] = i

        for o in model.experiment_outputs.keys():
            if o not in var_set:
                measurement_mapping[o] = None

        self.con_list = con_list
        self.var_list = var_list
        self.param_index = param_index
        self.model_var_index = model_var_index
        self.measurement_index = measurement_index
        self.measurement_error_included = measurement_error_included
        self.num_measurements = len(output_set)
        self.num_params = len(param_set)
        self.num_constraints = len(con_set)
        self.num_vars = len(var_set)
        self.var_set = var_set
        self.measurement_mapping = measurement_mapping
        self.parameter_mapping = parameter_mapping

    def _setup_differentiation(self):
        """Build symbolic and automatic Jacobian maps in a single pass."""
        if not self._requested_symbolic and not self._requested_automatic:
            raise ValueError("At least one differentiation method must be selected.")

        jac_dict_sd = {}
        jac_dict_ad = {}

        for i, c in enumerate(self.con_list):
            if not c.equality:
                raise ValueError(
                    "ExperimentGradients currently requires equality constraints."
                )

            # For each equation F_i(x, u, theta) = 0, compute the partial derivatives with
            # respect to the indexed variables and parameters. These derivatives form the
            # Jacobian rows used to assemble the blocks dF/dx and dF/dtheta in the
            # sensitivity system dF/dx * dx/dtheta + dF/dtheta = 0.

            der_map_sd = reverse_sd(c.body)
            der_map_ad = reverse_ad(c.body)

            for j, v in enumerate(self.var_list):
                # If a variable or parameter does not appear in F_i, its partial derivative in
                # that equation is zero, so absent derivative entries are filled in with 0.
                jac_dict_sd[(i, j)] = der_map_sd.get(v, 0)
                jac_dict_ad[(i, j)] = der_map_ad.get(v, 0)

        self.jac_dict_sd = jac_dict_sd
        self.jac_dict_ad = jac_dict_ad

    def compute_gradient_outputs_wrt_unknown_parameters(self):
        """Compute the output sensitivity matrix with respect to theta.

        This differentiates the model F(x, u, theta) = 0 with u fixed, solves
        for dx/dtheta, and then extracts the measured-output rows to return
        dy/dtheta.
        """
        if self.jac_dict_ad is None:
            self._setup_differentiation()

        # Assemble dF/dtheta, the Jacobian block of the model equations with
        # respect to the unknown parameters theta.
        jac_con_wrt_param = np.zeros((self.num_constraints, self.num_params))
        for i in range(self.num_constraints):
            for j, p in enumerate(self.param_index):
                jac_con_wrt_param[i, j] = self.jac_dict_ad[(i, p)]

        # Assemble dF/dx, the Jacobian block of the model equations with
        # respect to the model variables x.
        jac_con_wrt_vars = np.zeros((self.num_constraints, len(self.model_var_index)))
        for i in range(self.num_constraints):
            for j, v in enumerate(self.model_var_index):
                jac_con_wrt_vars[i, j] = self.jac_dict_ad[(i, v)]

        # With the design variables u fixed, differentiate F(x, u, theta) = 0
        # to obtain dF/dx * dx/dtheta + dF/dtheta = 0, then solve for
        # dx/dtheta = -(dF/dx)^{-1}(dF/dtheta).
        jac_vars_wrt_param = np.linalg.solve(jac_con_wrt_vars, -jac_con_wrt_param)

        # Extract the rows of dx/dtheta corresponding to the measured outputs y
        # to form the sensitivity matrix dy/dtheta used in the FIM.
        jac_measurements_wrt_param = np.zeros((self.num_measurements, self.num_params))
        for ind, m in enumerate(self.model.experiment_outputs.keys()):
            i = self.measurement_mapping[m]
            if i is None:
                jac_measurements_wrt_param[ind, :] = 0.0
            else:
                jac_measurements_wrt_param[ind, :] = jac_vars_wrt_param[i, :]

        self.jac_measurements_wrt_param = jac_measurements_wrt_param
        return jac_measurements_wrt_param

    def construct_sensitivity_constraints(self, model=None):
        """Add symbolic sensitivity variables and constraints to a Pyomo model.

        The added constraints encode the differentiated model equations
        dF/dx * dx/dtheta + dF/dtheta = 0, where F(x, u, theta) = 0 is the
        model, x is the vector of model variables, u is the
        vector of design variables, and theta is the vector of unknown
        parameters. This makes the local sensitivities dx/dtheta explicit
        inside the optimization model.
        """
        if self.jac_dict_sd is None:
            self._setup_differentiation()

        if model is None:
            model = self.model

        model.param_index = pyo.Set(initialize=self.param_index)
        model.constraint_index = pyo.Set(initialize=range(len(self.con_list)))
        model.var_index = pyo.Set(initialize=self.model_var_index)
        # Introduce Pyomo variables representing dx/dtheta so the local
        # sensitivity system can be written explicitly inside the optimization
        # model.
        model.jac_variables_wrt_param = pyo.Var(
            model.var_index, model.param_index, initialize=0
        )

        @model.Constraint(model.constraint_index, model.param_index)
        def jacobian_constraint(model, i, j):
            # Enforce dF/dx * dx/dtheta + dF/dtheta = 0 for each model equation
            # and parameter.
            return self.jac_dict_sd[(i, j)] == -sum(
                model.jac_variables_wrt_param[k, j] * self.jac_dict_sd[(i, k)]
                for k in model.var_index
            )
