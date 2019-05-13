# Source: https://docs.mosek.com/modeling-cookbook/expo.html
#         (first example in Section 5.3.1)

import pyomo.kernel as pmo

def solve_nonlinear():

    m = pmo.block()

    m.x = pmo.variable()
    m.y = pmo.variable()
    m.z = pmo.variable()

    m.c = pmo.constraint_tuple([
        pmo.constraint(body=0.1*pmo.sqrt(m.x) + (2.0/m.y),
                       ub=1),
        pmo.constraint(body=(1.0/m.z) + (m.y/(m.x**2)),
                       ub=1)])

    m.o = pmo.objective(m.x + (m.y**2)*m.z,
                        sense=pmo.minimize)

    m.x.value, m.y.value, m.z.value = (1,1,1)
    ipopt = pmo.SolverFactory("ipopt")
    result = ipopt.solve(m)
    assert str(result.solver.termination_condition) == "optimal"
    print("nonlinear solution:")
    print("x: {0:.4f}, y: {1:.4f}, z: {2:.4f}".\
          format(m.x(), m.y(), m.z()))
    print("objective: {0: .5f}".\
          format(m.o()))
    print("")

def solve_conic():

    m = pmo.block()

    m.t = pmo.variable()
    m.u = pmo.variable()
    m.v = pmo.variable()
    m.w = pmo.variable()

    m.k = pmo.block_tuple([
        # exp(u-t) + exp(2v + w - t) <= 1
        pmo.conic.primal_exponential.\
            as_domain(r=None,
                      x1=1,
                      x2=m.u - m.t),
        pmo.conic.primal_exponential.\
            as_domain(r=None,
                      x1=1,
                      x2=2*m.v + m.w - m.t),
        # exp(0.5u + log(0.1)) + exp(-v + log(2)) <= 1
        pmo.conic.primal_exponential.\
            as_domain(r=None,
                      x1=1,
                      x2=0.5*m.u + pmo.log(0.1)),
        pmo.conic.primal_exponential.\
            as_domain(r=None,
                      x1=1,
                      x2=-m.v + pmo.log(2)),
        # exp(-w) + exp(v-2u) <= 1
        pmo.conic.primal_exponential.\
            as_domain(r=None,
                      x1=1,
                      x2=-m.w),
        pmo.conic.primal_exponential.\
            as_domain(r=None,
                      x1=1,
                      x2=m.v - 2*m.u)])

    m.c = pmo.constraint_tuple([
        pmo.constraint(body=m.k[0].r + m.k[1].r,
                       ub=1),
        pmo.constraint(body=m.k[2].r + m.k[3].r,
                       ub=1),
        pmo.constraint(body=m.k[4].r + m.k[5].r,
                       ub=1)])

    m.o = pmo.objective(m.t,
                        sense=pmo.minimize)

    mosek = pmo.SolverFactory("mosek")
    result = mosek.solve(m)
    assert str(result.solver.termination_condition) == "optimal"
    x, y, z = pmo.exp(m.u()), pmo.exp(m.v()), pmo.exp(m.w())
    print("conic solution:")
    print("x: {0:.4f}, y: {1:.4f}, z: {2:.4f}".\
          format(x, y, z))
    print("objective: {0: .5f}".\
          format(x + (y**2)*z))
    print("")

if __name__ == "__main__":
    solve_nonlinear()
    solve_conic()
