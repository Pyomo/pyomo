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

# Source: https://docs.mosek.com/9.0/pythonapi/tutorial-pow-shared.html

import pyomo.kernel as pmo


def solve_nonlinear():
    m = pmo.block()

    m.x = pmo.variable(lb=0)
    m.y = pmo.variable(lb=0)
    m.z = pmo.variable(lb=0)

    m.c = pmo.constraint(body=m.x + m.y + 0.5 * m.z, rhs=2)

    m.o = pmo.objective((m.x**0.2) * (m.y**0.8) + (m.z**0.4) - m.x, sense=pmo.maximize)

    m.x.value, m.y.value, m.z.value = (1, 1, 1)
    ipopt = pmo.SolverFactory("ipopt")
    result = ipopt.solve(m)
    assert str(result.solver.termination_condition) == "optimal"
    print("nonlinear solution:")
    print("x: {0:.4f}, y: {1:.4f}, z: {2:.4f}".format(m.x(), m.y(), m.z()))
    print("objective: {0: .5f}".format(m.o()))
    print("")


def solve_conic():
    m = pmo.block()

    m.x = pmo.variable(lb=0)
    m.y = pmo.variable(lb=0)
    m.z = pmo.variable(lb=0)

    m.p = pmo.variable()
    m.q = pmo.variable()
    m.r = pmo.variable(lb=0)

    m.k = pmo.block_tuple(
        [
            pmo.conic.primal_power.as_domain(r1=m.x, r2=m.y, x=[None], alpha=0.2),
            pmo.conic.primal_power.as_domain(r1=m.z, r2=1, x=[None], alpha=0.4),
        ]
    )

    m.c = pmo.constraint(body=m.x + m.y + 0.5 * m.z, rhs=2)

    m.o = pmo.objective(m.k[0].x[0] + m.k[1].x[0] - m.x, sense=pmo.maximize)

    mosek = pmo.SolverFactory("mosek_direct")
    result = mosek.solve(m)
    assert str(result.solver.termination_condition) == "optimal"
    print("conic solution:")
    print("x: {0:.4f}, y: {1:.4f}, z: {2:.4f}".format(m.x(), m.y(), m.z()))
    print("objective: {0: .5f}".format(m.o()))
    print("")


if __name__ == "__main__":
    solve_nonlinear()
    solve_conic()
