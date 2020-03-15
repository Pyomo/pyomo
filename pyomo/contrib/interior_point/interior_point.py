from pyomo.contrib.interior_point.interface import InteriorPointInterface, BaseInteriorPointInterface
from pyomo.contrib.interior_point.linalg.mumps_interface import MumpsInterface
from scipy.sparse import tril
import numpy as np


def solve_interior_point(pyomo_model, max_iter=100):
    interface = InteriorPointInterface(pyomo_model)
    primals = interface.init_primals().copy()
    slacks = interface.init_slacks().copy()
    duals_eq = interface.init_duals_eq().copy()
    duals_ineq = interface.init_duals_ineq().copy()
    duals_primals_lb = interface.init_duals_primals_lb().copy()
    duals_primals_ub = interface.init_duals_primals_ub().copy()
    duals_slacks_lb = interface.init_duals_slacks_lb().copy()
    duals_slacks_ub = interface.init_duals_slacks_ub().copy()
    
    minimum_barrier_parameter = 1e-9
    barrier_parameter = 0.1
    interface.set_barrier_parameter(barrier_parameter)

    for _iter in range(max_iter):
        print('*********************************')
        print('primals: ', primals)
        print('slacks: ', slacks)
        print('duals_eq: ', duals_eq)
        print('duals_ineq: ', duals_ineq)
        print('duals_primals_lb: ', duals_primals_lb)
        print('duals_primals_ub: ', duals_primals_ub)
        print('duals_slacks_lb: ', duals_slacks_lb)
        print('duals_slacks_ub: ', duals_slacks_ub)
        interface.set_primals(primals)
        interface.set_slacks(slacks)
        interface.set_duals_eq(duals_eq)
        interface.set_duals_ineq(duals_ineq)
        interface.set_duals_primals_lb(duals_primals_lb)
        interface.set_duals_primals_ub(duals_primals_ub)
        interface.set_duals_slacks_lb(duals_slacks_lb)
        interface.set_duals_slacks_ub(duals_slacks_ub)
        interface.set_barrier_parameter(barrier_parameter)

        primal_inf, dual_inf, complimentarity_inf = check_convergence(interface=interface, barrier=0)
        print('primal_inf: ', primal_inf)
        print('dual_inf: ', dual_inf)
        print('complimentarity_inf: ', complimentarity_inf)

        kkt = interface.evaluate_primal_dual_kkt_matrix()
        print(kkt.toarray())
        kkt = tril(kkt.tocoo())
        rhs = interface.evaluate_primal_dual_kkt_rhs()
        print(rhs.flatten())
        linear_solver = MumpsInterface()  # icntl_options={1: 6, 2: 6, 3: 6, 4: 4})
        linear_solver.do_symbolic_factorization(kkt)
        linear_solver.do_numeric_factorization(kkt)
        delta = linear_solver.do_back_solve(rhs)

        interface.set_primal_dual_kkt_solution(delta)
        delta_primals = interface.get_delta_primals()
        delta_slacks = interface.get_delta_slacks()
        delta_duals_eq = interface.get_delta_duals_eq()
        delta_duals_ineq = interface.get_delta_duals_ineq()
        delta_duals_primals_lb = interface.get_delta_duals_primals_lb()
        delta_duals_primals_ub = interface.get_delta_duals_primals_ub()
        delta_duals_slacks_lb = interface.get_delta_duals_slacks_lb()
        delta_duals_slacks_ub = interface.get_delta_duals_slacks_ub()

        alpha = 1
        primals += alpha * delta_primals
        slacks += alpha * delta_slacks
        duals_eq += alpha * delta_duals_eq
        duals_ineq += alpha * delta_duals_ineq
        duals_primals_lb += alpha * delta_duals_primals_lb
        duals_primals_ub += alpha * delta_duals_primals_ub
        duals_slacks_lb += alpha * delta_duals_slacks_lb
        duals_slacks_ub += alpha * delta_duals_slacks_ub

        barrier_parameter = max(minimum_barrier_parameter, min(0.5*barrier_parameter, barrier_parameter**1.5))


def check_convergence(interface, barrier):
    """
    Parameters
    ----------
    interface: pyomo.contrib.interior_point.interface.BaseInteriorPointInterface
    barrier: float

    Returns
    -------
    primal_inf: float
    dual_inf: float
    complimentarity_inf: float
    """
    grad_obj = interface.evaluate_grad_objective()
    jac_eq = interface.evaluate_jacobian_eq()
    jac_ineq = interface.evaluate_jacobian_ineq()
    primals = interface.get_primals()
    slacks = interface.get_slacks()
    duals_eq = interface.get_duals_eq()
    duals_ineq = interface.get_duals_ineq()
    duals_primals_lb = interface.get_duals_primals_lb()
    duals_primals_ub = interface.get_duals_primals_ub()
    duals_slacks_lb = interface.get_duals_slacks_lb()
    duals_slacks_ub = interface.get_duals_slacks_ub()
    primals_lb_compression_matrix = interface.get_primals_lb_compression_matrix()
    primals_ub_compression_matrix = interface.get_primals_ub_compression_matrix()
    ineq_lb_compression_matrix = interface.get_ineq_lb_compression_matrix()
    ineq_ub_compression_matrix = interface.get_ineq_ub_compression_matrix()
    primals_lb_compressed = interface.get_primals_lb_compressed()
    primals_ub_compressed = interface.get_primals_ub_compressed()
    ineq_lb_compressed = interface.get_ineq_lb_compressed()
    ineq_ub_compressed = interface.get_ineq_ub_compressed()

    grad_lag_primals = (grad_obj +
                        jac_eq.transpose() * duals_eq +
                        jac_ineq.transpose() * duals_ineq -
                        primals_lb_compression_matrix.transpose() * duals_primals_lb +
                        primals_ub_compression_matrix.transpose() * duals_primals_ub)
    grad_lag_slacks = (-duals_ineq -
                       ineq_lb_compression_matrix.transpose() * duals_slacks_lb +
                       ineq_ub_compression_matrix.transpose() * duals_slacks_ub)
    eq_resid = interface.evaluate_eq_constraints()
    ineq_resid = interface.evaluate_ineq_constraints() - slacks
    primals_lb_resid = (primals_lb_compression_matrix * primals - primals_lb_compressed) * duals_primals_lb - barrier
    primals_ub_resid = (primals_ub_compressed - primals_ub_compression_matrix * primals) * duals_primals_ub - barrier
    slacks_lb_resid = (ineq_lb_compression_matrix * slacks - ineq_lb_compressed) * duals_slacks_lb - barrier
    slacks_ub_resid = (ineq_ub_compressed - ineq_ub_compression_matrix * slacks) * duals_slacks_ub - barrier

    if eq_resid.size == 0:
        max_eq_resid = 0
    else:
        max_eq_resid = np.max(np.abs(eq_resid))
    if ineq_resid.size == 0:
        max_ineq_resid = 0
    else:
        max_ineq_resid = np.max(np.abs(ineq_resid))
    primal_inf = max(max_eq_resid, max_ineq_resid)

    max_grad_lag_primals = np.max(np.abs(grad_lag_primals))
    if grad_lag_slacks.size == 0:
        max_grad_lag_slacks = 0
    else:
        max_grad_lag_slacks = np.max(np.abs(grad_lag_slacks))
    dual_inf = max(max_grad_lag_primals, max_grad_lag_slacks)

    if primals_lb_resid.size == 0:
        max_primals_lb_resid = 0
    else:
        max_primals_lb_resid = np.max(np.abs(primals_lb_resid))
    if primals_ub_resid.size == 0:
        max_primals_ub_resid = 0
    else:
        max_primals_ub_resid = np.max(np.abs(primals_ub_resid))
    if slacks_lb_resid.size == 0:
        max_slacks_lb_resid = 0
    else:
        max_slacks_lb_resid = np.max(np.abs(slacks_lb_resid))
    if slacks_ub_resid.size == 0:
        max_slacks_ub_resid = 0
    else:
        max_slacks_ub_resid = np.max(np.abs(slacks_ub_resid))
    complimentarity_inf = max(max_primals_lb_resid, max_primals_ub_resid, max_slacks_lb_resid, max_slacks_ub_resid)

    return primal_inf, dual_inf, complimentarity_inf
