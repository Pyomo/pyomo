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

import pyomo.environ as pyo

from pyomo.common.dependencies import numpy as np, numpy_available

from pyomo.core.base.param import ParamData
from pyomo.core.base.var import VarData

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
    E_opt : float
        log10(E-optimality) metric.
    ME_opt : float
        log10(Modified E-optimality) metric.
    """

    # Check whether the FIM is square, positive definite, and symmetric
    check_FIM(FIM)

    # Compute FIM metrics
    det_FIM = np.linalg.det(FIM)
    D_opt = np.log10(det_FIM)

    trace_FIM = np.trace(FIM)
    A_opt = np.log10(trace_FIM)

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

    ME_opt = np.log10(np.linalg.cond(FIM))

    return det_FIM, trace_FIM, E_vals, E_vecs, D_opt, A_opt, E_opt, ME_opt


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
    "log10(E-Optimality)" : float
        log10(E-optimality) metric
    "log10(Modified E-Optimality)" : float
        log10(Modified E-optimality) metric
    """

    det_FIM, trace_FIM, E_vals, E_vecs, D_opt, A_opt, E_opt, ME_opt = (
        compute_FIM_metrics(FIM)
    )

    return {
        "Determinant of FIM": det_FIM,
        "Trace of FIM": trace_FIM,
        "Eigenvalues": E_vals,
        "Eigenvectors": E_vecs,
        "log10(D-Optimality)": D_opt,
        "log10(A-Optimality)": A_opt,
        "log10(E-Optimality)": E_opt,
        "log10(Modified E-Optimality)": ME_opt,
    }
