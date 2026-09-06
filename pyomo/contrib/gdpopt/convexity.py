# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Conservative convexity checks for outer approximation algorithms.

Outer approximation algorithms that linearize the nonlinear constraints at trial
points (GDPopt LOA, MindtPy OA, MindtPy ECP) only produce a valid relaxation of
the original problem when that problem is convex. Applied to a nonconvex problem,
the linearizations can cut off feasible points, so the resulting "dual bound" is
not a rigorous bound and must not be reported as certifying global optimality.

Algorithms that build their relaxation from McCormick envelopes instead (GDPopt
GLOA, MindtPy GOA) do produce a valid relaxation for nonconvex problems, so their
bounds are rigorous and are unaffected by this module.

The detection here is deliberately conservative: anything this module cannot
positively certify as convex is reported as not certified convex.
"""

from pyomo.common.dependencies import numpy as np
from pyomo.core import Block, Constraint, Objective, minimize, value
from pyomo.core.base.enums import SortComponents
from pyomo.gdp import Disjunct
from pyomo.repn.quadratic import QuadraticRepnVisitor
from pyomo.repn.util import OrderedVarRecorder


def _quadratic_matrix(repn):
    """Build the symmetric matrix Q of a quadratic repn.

    Returns None if any quadratic coefficient cannot be evaluated to a number.
    """
    var_to_idx = {}
    for var_ids in repn.quadratic:
        for var_id in var_ids:
            if var_id not in var_to_idx:
                var_to_idx[var_id] = len(var_to_idx)

    q_matrix = np.zeros((len(var_to_idx), len(var_to_idx)))
    for (var_id1, var_id2), coef in repn.quadratic.items():
        coef_val = value(coef, exception=False)
        if coef_val is None:
            return None
        idx1 = var_to_idx[var_id1]
        idx2 = var_to_idx[var_id2]
        if var_id1 == var_id2:
            q_matrix[idx1][idx1] += coef_val
        else:
            half_coef = 0.5 * coef_val
            q_matrix[idx1][idx2] += half_coef
            q_matrix[idx2][idx1] += half_coef

    return q_matrix


def quadratic_curvature(expr, eigenvalue_tolerance):
    """Classify the curvature of a quadratic expression.

    Returns 1 if the quadratic form is positive semidefinite (convex), -1 if it
    is negative semidefinite (concave), 0 if it has no quadratic terms or the
    quadratic form vanishes, and None if the curvature could not be determined.
    """
    recorder = OrderedVarRecorder({}, {}, SortComponents.deterministic)
    repn = QuadraticRepnVisitor({}, var_recorder=recorder).walk_expression(expr)
    if repn.nonlinear is not None:
        return None
    if repn.quadratic is None:
        return 0

    q_matrix = _quadratic_matrix(repn)
    if q_matrix is None:
        return None

    eigenvalues = np.linalg.eigvalsh(q_matrix)
    is_psd = all(eigenvalue >= -eigenvalue_tolerance for eigenvalue in eigenvalues)
    is_nsd = all(eigenvalue <= eigenvalue_tolerance for eigenvalue in eigenvalues)
    if is_psd and is_nsd:
        return 0
    if is_psd:
        return 1
    if is_nsd:
        return -1
    return None


def model_is_not_certified_convex(model, eigenvalue_tolerance):
    """Return True unless this model can be certified as convex.

    A True result means an outer approximation dual bound computed for the model
    must not be treated as rigorous.
    """
    for obj in model.component_data_objects(Objective, active=True, descend_into=True):
        curvature = quadratic_curvature(obj.expr, eigenvalue_tolerance)
        if obj.sense is minimize and curvature not in (0, 1):
            return True
        elif obj.sense is not minimize and curvature not in (0, -1):
            return True

    for constr in model.component_data_objects(
        Constraint, active=True, descend_into=(Block, Disjunct)
    ):
        curvature = quadratic_curvature(constr.body, eigenvalue_tolerance)
        if curvature == 0:
            continue
        if constr.equality:
            return True
        if constr.has_ub() and curvature not in (0, 1):
            return True
        if constr.has_lb() and curvature not in (0, -1):
            return True
    return False
