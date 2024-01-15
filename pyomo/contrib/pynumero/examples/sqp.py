from pyomo.contrib.pynumero.interfaces.nlp import NLP
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
from pyomo.contrib.pynumero.linalg.ma27_interface import MA27
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import numpy as np
from scipy.sparse import tril
import pyomo.environ as pe
from pyomo import dae
from pyomo.common.timing import TicTocTimer
import time
from pyomo.contrib.pynumero.linalg.base import LinearSolverInterface, LinearSolverStatus


def build_burgers_model(nfe_x=100, nfe_t=200, start_t=0, end_t=1):
    """
    Build a pyomo.dae model of an optimal control problem with Burgers' equation

    Parameters
    ----------
    nfe_x: int
        Number of finite elements for discretization of x
    nfe_t: int
        Number of finite elements for discretization of t
    start_t: float
        The start of the time horizon
    end_t: float
        The end of the time horizon

    Returns
    -------
    m: Pyomo model
    """
    dt = (end_t - start_t) / float(nfe_t)

    start_x = 0
    end_x = 1
    dx = (end_x - start_x) / float(nfe_x)

    m = pe.Block(concrete=True)
    m.omega = pe.Param(initialize=0.02)
    m.v = pe.Param(initialize=0.01)
    m.r = pe.Param(initialize=0)

    m.x = dae.ContinuousSet(bounds=(start_x, end_x))
    m.t = dae.ContinuousSet(bounds=(start_t, end_t))

    m.y = pe.Var(m.x, m.t)
    m.dydt = dae.DerivativeVar(m.y, wrt=m.t)
    m.dydx = dae.DerivativeVar(m.y, wrt=m.x)
    m.dydx2 = dae.DerivativeVar(m.y, wrt=(m.x, m.x))

    m.u = pe.Var(m.x, m.t)

    def _y_init_rule(m, x):
        if x <= 0.5 * end_x:
            return 1
        return 0

    m.y0 = pe.Param(m.x, default=_y_init_rule)

    def _upper_x_bound(m, t):
        return m.y[end_x, t] == 0

    m.upper_x_bound = pe.Constraint(m.t, rule=_upper_x_bound)

    def _lower_x_bound(m, t):
        return m.y[start_x, t] == 0

    m.lower_x_bound = pe.Constraint(m.t, rule=_lower_x_bound)

    def _upper_x_ubound(m, t):
        return m.u[end_x, t] == 0

    m.upper_x_ubound = pe.Constraint(m.t, rule=_upper_x_ubound)

    def _lower_x_ubound(m, t):
        return m.u[start_x, t] == 0

    m.lower_x_ubound = pe.Constraint(m.t, rule=_lower_x_ubound)

    def _lower_t_bound(m, x):
        if x == start_x or x == end_x:
            return pe.Constraint.Skip
        return m.y[x, start_t] == m.y0[x]

    def _lower_t_ubound(m, x):
        if x == start_x or x == end_x:
            return pe.Constraint.Skip
        return m.u[x, start_t] == 0

    m.lower_t_bound = pe.Constraint(m.x, rule=_lower_t_bound)
    m.lower_t_ubound = pe.Constraint(m.x, rule=_lower_t_ubound)

    # PDE
    def _pde(m, x, t):
        if t == start_t or x == end_x or x == start_x:
            e = pe.Constraint.Skip
        else:
            # print(foo.last_t, t-dt, abs(foo.last_t - (t-dt)))
            # assert math.isclose(foo.last_t, t - dt, abs_tol=1e-6)
            e = (
                m.dydt[x, t] - m.v * m.dydx2[x, t] + m.dydx[x, t] * m.y[x, t]
                == m.r + m.u[x, t]
            )
        return e

    m.pde = pe.Constraint(m.x, m.t, rule=_pde)

    # Discretize Model
    disc = pe.TransformationFactory('dae.finite_difference')
    disc.apply_to(m, nfe=nfe_t, wrt=m.t, scheme='BACKWARD')
    disc.apply_to(m, nfe=nfe_x, wrt=m.x, scheme='CENTRAL')

    # Solve control problem using Pyomo.DAE Integrals
    def _intX(m, x, t):
        return (m.y[x, t] - m.y0[x]) ** 2 + m.omega * m.u[x, t] ** 2

    m.intX = dae.Integral(m.x, m.t, wrt=m.x, rule=_intX)

    def _intT(m, t):
        return m.intX[t]

    m.intT = dae.Integral(m.t, wrt=m.t, rule=_intT)

    def _obj(m):
        e = 0.5 * m.intT
        for x in sorted(m.x):
            if x == start_x or x == end_x:
                pass
            else:
                e += 0.5 * 0.5 * dx * dt * m.omega * m.u[x, start_t] ** 2
        return e

    m.obj = pe.Objective(rule=_obj)

    return m


def sqp(
    nlp: NLP, linear_solver: LinearSolverInterface, max_iter=100, tol=1e-8, output=True
):
    """
    An example of a simple SQP algorithm for
    equality-constrained NLPs.

    Parameters
    ----------
    nlp: NLP
        A PyNumero NLP
    max_iter: int
        The maximum number of iterations
    tol: float
        The convergence tolerance
    """
    t0 = time.time()

    # setup KKT matrix
    kkt = BlockMatrix(2, 2)
    rhs = BlockVector(2)

    # create and initialize the iteration vector
    z = BlockVector(2)
    z.set_block(0, nlp.get_primals())
    z.set_block(1, nlp.get_duals())

    if output:
        print(
            f"{'Iter':<12}{'Objective':<12}{'Primal Infeasibility':<25}{'Dual Infeasibility':<25}{'Elapsed Time':<15}"
        )

    # main iteration loop
    for _iter in range(max_iter):
        nlp.set_primals(z.get_block(0))
        nlp.set_duals(z.get_block(1))

        grad_lag = (
            nlp.evaluate_grad_objective()
            + nlp.evaluate_jacobian_eq().transpose() * z.get_block(1)
        )
        residuals = nlp.evaluate_eq_constraints()

        if output:
            print(
                f"{_iter:<12}{nlp.evaluate_objective():<12.2e}{np.abs(residuals).max():<25.2e}{np.abs(grad_lag).max():<25.2e}{time.time()-t0:<15.2e}"
            )

        if np.abs(grad_lag).max() <= tol and np.abs(residuals).max() <= tol:
            break

        kkt.set_block(0, 0, nlp.evaluate_hessian_lag())
        kkt.set_block(1, 0, nlp.evaluate_jacobian_eq())
        kkt.set_block(0, 1, nlp.evaluate_jacobian_eq().transpose())

        rhs.set_block(0, grad_lag)
        rhs.set_block(1, residuals)

        delta, res = linear_solver.solve(kkt, -rhs)
        assert res.status == LinearSolverStatus.successful
        z += delta


def load_solution(m: pe.ConcreteModel(), nlp: PyomoNLP):
    primals = nlp.get_primals()
    pyomo_vars = nlp.get_pyomo_variables()
    for v, val in zip(pyomo_vars, primals):
        v.value = val


def main(linear_solver, nfe_x=100, nfe_t=200):
    m = build_burgers_model(nfe_x=nfe_x, nfe_t=nfe_t)
    nlp = PyomoNLP(m)
    sqp(nlp, linear_solver)
    load_solution(m, nlp)
    return pe.value(m.obj)


if __name__ == '__main__':
    # create the linear solver
    linear_solver = MA27()
    linear_solver.set_cntl(1, 1e-6)  # pivot tolerance

    optimal_obj = main(linear_solver)
    print(f'Optimal Objective: {optimal_obj}')
