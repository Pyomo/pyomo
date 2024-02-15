#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# Source: https://docs.mosek.com/latest/pythonfusion/tutorial-sdo-shared.html#doc-tutorial-sdo

# This examples illustrates SDP formulations in Pyomo using
# the MOSEK interface. The following functions construct the
# same problem, but in the primal and dual forms respectively.
#
# Read more about SDP duality in the MOSEK modeling cookbook.

import numpy as np
import pyomo.kernel as pmo

from pyomo.core import sum_product

#   GENERAL SDP (PRIMAL FORM)
#   min      <barC, barX> + c*x
#   s.t.   <barA_i, barX> + a_i*x  = b_i    i = 1,...,m
#            x \in K    ;   barX \in PSD_CONE


def primal_sdo1():
    # Problem data
    d = 3
    n = int(d * (d + 1) / 2)

    # PSD matrices
    # NOTE: As the matrices are symmetric (required)
    #       we only specify lower-triangular part
    #       and the off-diagonal elements are doubled.
    barC = [2, 2, 0, 2, 2, 2]
    barA1 = [1, 0, 0, 1, 0, 1]
    barA2 = [1, 2, 2, 1, 2, 1]

    model = pmo.block()

    # VARIABLES
    model.x = pmo.variable_list(pmo.variable() for i in range(d))
    model.X = pmo.variable_list(pmo.variable() for i in range(n))

    # CONSTRAINTS
    # Linear
    model.c1 = pmo.constraint(
        sum_product(barA1, model.X, index=list(range(n))) + model.x[0] == 1
    )
    model.c2 = pmo.constraint(
        sum_product(barA2, model.X, index=list(range(n))) + model.x[1] + model.x[2]
        == 0.5
    )
    # Conic
    model.quad_cone = pmo.conic.quadratic(r=model.x[0], x=model.x[1:])
    # Off-diagonal elements need to be scaled by sqrt(2) in SVEC_PSD domain
    scale = [1, np.sqrt(2), np.sqrt(2), 1, np.sqrt(2), 1]
    model.psd_cone = pmo.conic.svec_psdcone.as_domain(
        x=[model.X[i] * scale[i] for i in range(n)]
    )

    # OBJECTIVE
    model.obj = pmo.objective(
        sum_product(barC, model.X, index=list(range(n))) + model.x[0]
    )

    msk = pmo.SolverFactory('mosek')
    results = msk.solve(model, tee=True)

    return results


# GENERAL SDP (DUAL FORM)
#
#   max.        b*y
#   s.t.     barC  -  sum(y_i, barA_i) \in PSD_CONE
#               c  -    A*y \in K
#
#   NOTE: the PSD constraint here is in the LMI (linear-matrix-inequality) form


def dual_sdo1():
    # Problem data
    d = 3
    n = int(d * (d + 1) / 2)

    c = [1, 0, 0]
    a_T = [[1, 0], [0, 1], [0, 1]]

    # PSD matrices
    barC = [2, np.sqrt(2), 0, 2, np.sqrt(2), 2]
    barA1 = [1, 0, 0, 1, 0, 1]
    barA2 = [1, np.sqrt(2), np.sqrt(2), 1, np.sqrt(2), 1]

    model = pmo.block()

    # VARIABLES
    model.y = pmo.variable_list(pmo.variable() for i in range(2))

    # CONSTRAINTS
    e1 = pmo.expression_list(
        pmo.expression(barC[i] - model.y[0] * barA1[i] - model.y[1] * barA2[i])
        for i in range(n)
    )
    model.psd_cone = pmo.conic.svec_psdcone.as_domain(x=e1)

    e2 = pmo.expression_list(
        pmo.expression(c[i] - sum_product(a_T[i], model.y, index=[0, 1]))
        for i in range(3)
    )
    model.quad_cone = pmo.conic.quadratic.as_domain(r=e2[0], x=e2[1:])

    # OBJECTIVE
    model.obj = pmo.objective(model.y[0] + 0.5 * model.y[1], sense=-1)

    msk = pmo.SolverFactory('mosek')
    results = msk.solve(model, tee=True)

    return results


if __name__ == '__main__':
    primal_sdo1()
    dual_sdo1()
