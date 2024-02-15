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

# Source: https://docs.mosek.com/9.0/pythonapi/tutorial-gp-shared.html

import pyomo.kernel as pmo


def solve_nonlinear(Aw, Af, alpha, beta, gamma, delta):
    m = pmo.block()

    m.h = pmo.variable(lb=0)
    m.w = pmo.variable(lb=0)
    m.d = pmo.variable(lb=0)

    m.c = pmo.constraint_tuple(
        [
            pmo.constraint(body=2 * (m.h * m.w + m.h * m.d), ub=Aw),
            pmo.constraint(body=m.w * m.d, ub=Af),
            pmo.constraint(lb=alpha, body=m.h / m.w, ub=beta),
            pmo.constraint(lb=gamma, body=m.d / m.w, ub=delta),
        ]
    )

    m.o = pmo.objective(m.h * m.w * m.d, sense=pmo.maximize)

    m.h.value, m.w.value, m.d.value = (1, 1, 1)
    ipopt = pmo.SolverFactory("ipopt")
    result = ipopt.solve(m)
    assert str(result.solver.termination_condition) == "optimal"
    print("nonlinear solution:")
    print("h: {0:.4f}, w: {1:.4f}, d: {2:.4f}".format(m.h(), m.w(), m.d()))
    print("volume: {0: .5f}".format(m.o()))
    print("")


def solve_conic(Aw, Af, alpha, beta, gamma, delta):
    m = pmo.block()

    m.x = pmo.variable()
    m.y = pmo.variable()
    m.z = pmo.variable()

    m.k = pmo.block_tuple(
        [
            pmo.conic.primal_exponential.as_domain(
                r=None, x1=1, x2=m.x + m.y + pmo.log(2.0 / Aw)
            ),
            pmo.conic.primal_exponential.as_domain(
                r=None, x1=1, x2=m.x + m.z + pmo.log(2.0 / Aw)
            ),
        ]
    )

    m.c = pmo.constraint_tuple(
        [
            pmo.constraint(body=m.k[0].r + m.k[1].r, ub=1),
            pmo.constraint(body=m.y + m.z, ub=pmo.log(Af)),
            pmo.constraint(lb=pmo.log(alpha), body=m.x - m.y, ub=pmo.log(beta)),
            pmo.constraint(lb=pmo.log(gamma), body=m.z - m.y, ub=pmo.log(delta)),
        ]
    )

    m.o = pmo.objective(m.x + m.y + m.z, sense=pmo.maximize)

    mosek = pmo.SolverFactory("mosek_direct")
    result = mosek.solve(m)
    assert str(result.solver.termination_condition) == "optimal"
    h, w, d = pmo.exp(m.x()), pmo.exp(m.y()), pmo.exp(m.z())
    print("conic solution:")
    print("h: {0:.4f}, w: {1:.4f}, d: {2:.4f}".format(h, w, d))
    print("volume: {0: .5f}".format(h * w * d))
    print("")


if __name__ == "__main__":
    Aw, Af, alpha, beta, gamma, delta = 200.0, 50.0, 2.0, 10.0, 2.0, 10.0
    solve_nonlinear(Aw, Af, alpha, beta, gamma, delta)
    solve_conic(Aw, Af, alpha, beta, gamma, delta)
