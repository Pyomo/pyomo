# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Detection of models whose outer approximation dual bounds may not be rigorous.

Outer approximation algorithms that linearize the nonlinear constraints at trial
points (GDPopt LOA, MindtPy OA, MindtPy ECP) only produce a valid relaxation of
the original problem when that problem is convex. Applied to a nonconvex problem,
the linearizations can cut off feasible points, so the resulting "dual bound" is
not a rigorous bound and must not be reported as certifying global optimality.

Algorithms that build their relaxation from McCormick envelopes instead (GDPopt
GLOA, MindtPy GOA) do produce a valid relaxation for nonconvex problems, so their
bounds are rigorous and are unaffected by this module.

The detection here is deliberately conservative: anything this module cannot
positively certify as convex is reported as possibly non-rigorous.
"""

from math import copysign, sqrt

from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.core import Block, Constraint, Objective, minimize, value
from pyomo.gdp import Disjunct
from pyomo.repn import generate_standard_repn


def _quadratic_matrix(repn):
    """Build the symmetric matrix Q of a quadratic standard repn.

    Returns None if any quadratic coefficient cannot be evaluated to a number.
    """
    var_to_idx = ComponentMap()
    for _coef, (v1, v2) in zip(repn.quadratic_coefs, repn.quadratic_vars):
        if v1 not in var_to_idx:
            var_to_idx[v1] = len(var_to_idx)
        if v2 not in var_to_idx:
            var_to_idx[v2] = len(var_to_idx)

    q_matrix = [[0.0 for _ in range(len(var_to_idx))] for _ in range(len(var_to_idx))]
    for coef, (v1, v2) in zip(repn.quadratic_coefs, repn.quadratic_vars):
        coef_val = value(coef, exception=False)
        if coef_val is None:
            return None
        idx1 = var_to_idx[v1]
        idx2 = var_to_idx[v2]
        if v1 is v2:
            q_matrix[idx1][idx1] += coef_val
        else:
            half_coef = 0.5 * coef_val
            q_matrix[idx1][idx2] += half_coef
            q_matrix[idx2][idx1] += half_coef

    return q_matrix


def symmetric_matrix_eigenvalues(matrix):
    """Return the eigenvalues of a real symmetric matrix.

    Uses NumPy when it is available. Otherwise falls back to a cyclic Jacobi
    iteration so that curvature classification still works in environments
    without NumPy. Returns None if the Jacobi iteration does not converge.
    """
    if numpy_available:
        return np.linalg.eigvalsh(matrix)

    order = len(matrix)
    if order <= 1:
        return [matrix[0][0]] if order else []

    matrix = [row[:] for row in matrix]
    scale = max(abs(val) for row in matrix for val in row)
    if scale == 0:
        return [0.0 for _ in range(order)]

    rotation_tolerance = 1e-12 * scale
    max_rotations = max(1, 50 * order * order)
    for _ in range(max_rotations):
        pivot_i, pivot_j, offdiag = 0, 1, abs(matrix[0][1])
        for i in range(order - 1):
            for j in range(i + 1, order):
                candidate = abs(matrix[i][j])
                if candidate > offdiag:
                    pivot_i, pivot_j, offdiag = i, j, candidate

        if offdiag <= rotation_tolerance:
            return [matrix[i][i] for i in range(order)]

        pivot = matrix[pivot_i][pivot_j]
        diag_i = matrix[pivot_i][pivot_i]
        diag_j = matrix[pivot_j][pivot_j]
        if diag_i == diag_j:
            tangent = copysign(1.0, pivot)
        else:
            tau = (diag_j - diag_i) / (2.0 * pivot)
            tangent = copysign(1.0, tau) / (abs(tau) + sqrt(1.0 + tau**2))
        cosine = 1.0 / sqrt(1.0 + tangent**2)
        sine = tangent * cosine

        matrix[pivot_i][pivot_i] = diag_i - tangent * pivot
        matrix[pivot_j][pivot_j] = diag_j + tangent * pivot
        matrix[pivot_i][pivot_j] = matrix[pivot_j][pivot_i] = 0.0

        for k in range(order):
            if k in (pivot_i, pivot_j):
                continue
            elem_i = matrix[k][pivot_i]
            elem_j = matrix[k][pivot_j]
            matrix[k][pivot_i] = matrix[pivot_i][k] = cosine * elem_i - sine * elem_j
            matrix[k][pivot_j] = matrix[pivot_j][k] = sine * elem_i + cosine * elem_j

    return None


def quadratic_curvature(expr):
    """Classify the curvature of a quadratic expression.

    Returns 1 if the quadratic form is positive semidefinite (convex), -1 if it
    is negative semidefinite (concave), 0 if it has no quadratic terms or the
    quadratic form vanishes, and None if the curvature could not be determined.
    """
    repn = generate_standard_repn(expr, quadratic=True)
    if not repn.quadratic_coefs:
        return 0

    q_matrix = _quadratic_matrix(repn)
    if q_matrix is None:
        return None

    eigenvalue_tolerance = 1e-10
    eigenvalues = symmetric_matrix_eigenvalues(q_matrix)
    if eigenvalues is None:
        return None
    is_psd = all(eigenvalue >= -eigenvalue_tolerance for eigenvalue in eigenvalues)
    is_nsd = all(eigenvalue <= eigenvalue_tolerance for eigenvalue in eigenvalues)
    if is_psd and is_nsd:
        return 0
    if is_psd:
        return 1
    if is_nsd:
        return -1
    return None


def model_may_have_nonrigorous_dual_bound(model):
    """Return True if linearization-based dual bounds for this model may be invalid.

    A True result means the model could not be certified as convex, so an outer
    approximation dual bound computed for it must not be treated as rigorous.
    """
    for obj in model.component_data_objects(Objective, active=True, descend_into=True):
        degree = obj.expr.polynomial_degree()
        if degree is None or degree not in (0, 1, 2):
            return True
        if degree == 2:
            curvature = quadratic_curvature(obj.expr)
            if obj.sense is minimize and curvature not in (0, 1):
                return True
            elif obj.sense is not minimize and curvature not in (0, -1):
                return True

    for constr in model.component_data_objects(
        Constraint, active=True, descend_into=(Block, Disjunct)
    ):
        degree = constr.body.polynomial_degree()
        if degree in (0, 1):
            continue
        if degree is None or degree not in (2,):
            return True
        if constr.equality:
            return True
        curvature = quadratic_curvature(constr.body)
        if constr.has_ub() and curvature not in (0, 1):
            return True
        if constr.has_lb() and curvature not in (0, -1):
            return True
    return False
