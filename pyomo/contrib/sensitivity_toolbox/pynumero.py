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

from pyomo.common.dependencies import numpy as np
from pyomo.common.dependencies import scipy, attempt_import

import pyomo.environ as pyo

# Use attempt_import here due to unguarded NumPy import in this file
nlp = attempt_import('pyomo.contrib.pynumero.interfaces.pyomo_nlp')[0]
from pyomo.common.collections import ComponentSet, ComponentMap


def _coo_reorder_cols(mat, remap):
    """Change the order of columns in a COO matrix. The main use of this is
    to reorder variables in the Jacobian matrix. This changes the matrix in
    place.

    Parameters
    ----------
    mat: scipy.sparse.coo_matrix
        Reorder the columns of this matrix
    remap: dict
        dictionary where keys are old column and value is new column, if a column
        doesn't move, it doesn't need to be included.

    Returns
    -------
    NoneType
        None
    """
    for i in range(len(mat.data)):
        try:
            mat.col[i] = remap[mat.col[i]]
        except KeyError:
            pass  # it's fine if we don't move a col in remap


def get_dsdp_dfdp(model, theta):
    """Calculate the derivatives of the state variables (s) with respect to
    parameters (p) (ds/dp), and the derivative of the objective function (f)
    with respect to p (df/dp). The number of parameters in theta should be the
    same as the number of degrees of freedom.

    Parameters
    ----------
    model: pyomo.environ.Block | pyomo.contrib.pynumero.interfaces.PyomoNLP
        Model to calculate sensitivity on. To retain the cached objects in
        the pynumero interface, create a PyomoNLP first and pass it to this function.
    theta: list
        A list of parameters as pyomo.environ.VarData, the number of parameters
        should be equal to the degrees of freedom.

    Returns
    -------
    scipy.sparse.csc_matrix, csc_matrix, ComponentMap, ComponentMap
        ds/dp (ns by np), df/dp (1 by np), row map, column map.
        The column map maps Pyomo variables p to columns and the
        row map maps Pyomo variables s to rows.
    """
    # Create a Pynumero NLP and get Jacobian
    if isinstance(model, nlp.PyomoNLP):
        m2 = model
    else:
        m2 = nlp.PyomoNLP(model)
    J = m2.evaluate_jacobian_eq()
    v_list = m2.get_pyomo_variables()
    # Map variables to columns in J
    mv_map = {id(v): i for i, v in enumerate(v_list)}
    s_list = list(ComponentSet(v_list) - ComponentSet(theta))
    ns = len(s_list)
    np = len(theta)
    col_remap = {mv_map[id(v)]: i for i, v in enumerate(s_list + theta)}
    _coo_reorder_cols(J, remap=col_remap)
    J = J.tocsc()
    dB = -(
        J
        @ scipy.sparse.vstack(
            (scipy.sparse.coo_matrix((ns, np)), scipy.sparse.identity(np))
        ).tocsc()
    )
    # Calculate sensitivity matrix
    dsdp = scipy.sparse.linalg.spsolve(J[:, range(ns)], dB)
    # Get a map of state vars to columns
    s_map = {id(v): i for i, v in enumerate(s_list)}
    # Get the outputs we are interested in from the list of output vars
    column_map = ComponentMap([(v, i) for i, v in enumerate(theta)])
    row_map = ComponentMap([(v, i) for i, v in enumerate(s_list)])
    dfdx = scipy.sparse.coo_matrix(m2.evaluate_grad_objective())
    _coo_reorder_cols(dfdx, remap=col_remap)
    dfdx = dfdx.tocsc()
    dfdp = dfdx[0, :ns] @ dsdp + dfdx[0, ns:]
    # return sensitivity of the outputs to p and component maps
    return dsdp, dfdp, row_map, column_map


def get_dydp(y_list, dsdp, row_map):
    """Reduce the sensitivity matrix from get_dsdp_dfdp to only
    a specified set of state variables of interest.

    Parameters
    ----------
    y_list: list
        A list of state variables of interest (a subset of s)
    dsdp: csc_matrix
        A sensitivity matrix calculated by get_dsdp_dfdp
    row_map: ComponentMap
        A row map from get_dsdp_dfdp

    Returns
    -------
    csc_matrix, ComponentMap
        dy/dp and a new row map with only y variables

    """
    new_row_map = ComponentMap()
    for i, v in enumerate(y_list):
        new_row_map[v] = i
    rows = [row_map[v] for v in y_list]
    dydp = dsdp[rows, :]
    return dydp, new_row_map
