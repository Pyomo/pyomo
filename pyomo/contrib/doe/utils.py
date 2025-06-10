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
                "param_vals should be a vector of dimensions: 1 by `n_params`. The shape you provided is {}.".format(
                    param_vals.shape
                )
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

    # TODO: Add swapping parameters for variables helper function
    # def get_parameters_from_suffix(suffix, fix_vars=False):
    #     """
    #     Finds the Params within the suffix provided. It will also check to see
    #     if there are Vars in the suffix provided. ``fix_vars`` will indicate
    #     if we should fix all the Vars in the set or not.
    #
    #     Parameters
    #     ----------
    #     suffix: pyomo Suffix object, contains the components to be checked
    #             as keys
    #     fix_vars: boolean, whether or not to fix the Vars, default = False
    #
    #     Returns
    #     -------
    #     param_list: list of Param
    #     """
    #     param_list = []
    #
    #     # FIX THE MODEL TREE ISSUE WHERE I GET base_model.<param> INSTEAD OF <param>
    #     # Check keys if they are Param or Var. Fix the vars if ``fix_vars`` is True
    #     for k, v in suffix.items():
    #         if isinstance(k, ParamData):
    #             param_list.append(k.name)
    #         elif isinstance(k, VarData):
    #             if fix_vars:
    #                 k.fix()
    #         else:
    #             pass  # ToDo: Write error for suffix keys that aren't ParamData or VarData
    #
    #     return param_list


def check_FIM(FIM):
    """Checks that the FIM is square, positive definite, and symmetric.

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
            "FIM provided is not positive definite. It has one or more negative eigenvalue(s) less than -{:.1e}".format(
                _SMALL_TOLERANCE_DEFINITENESS
            )
        )

    # Check if the FIM is symmetric
    if not np.allclose(FIM, FIM.T, atol=_SMALL_TOLERANCE_SYMMETRY):
        raise ValueError(
            "FIM provided is not symmetric using absolute tolerance {}".format(
                _SMALL_TOLERANCE_SYMMETRY
            )
        )
