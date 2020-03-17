from pyomo.contrib.interior_point.interface import InteriorPointInterface, BaseInteriorPointInterface
from pyomo.contrib.interior_point.linalg.mumps_interface import MumpsInterface
from pyomo.contrib.interior_point.linalg.scipy_interface import ScipyInterface
from pyomo.contrib.pynumero.interfaces.utils import build_bounds_mask, build_compression_matrix
from scipy.sparse import tril, coo_matrix, identity
import numpy as np
import logging
import time


logger = logging.getLogger('interior_point')


def solve_interior_point(pyomo_model, max_iter=100, tol=1e-8):
    t0 = time.time()
    interface = InteriorPointInterface(pyomo_model)
    primals = interface.init_primals().copy()
    slacks = interface.init_slacks().copy()
    duals_eq = interface.init_duals_eq().copy()
    duals_ineq = interface.init_duals_ineq().copy()
    duals_primals_lb = interface.init_duals_primals_lb().copy()
    duals_primals_ub = interface.init_duals_primals_ub().copy()
    duals_slacks_lb = interface.init_duals_slacks_lb().copy()
    duals_slacks_ub = interface.init_duals_slacks_ub().copy()

    _process_init(primals, interface.get_primals_lb(), interface.get_primals_ub())
    _process_init(slacks, interface.get_ineq_lb(), interface.get_ineq_ub())
    _process_init_duals(duals_primals_lb)
    _process_init_duals(duals_primals_ub)
    _process_init_duals(duals_slacks_lb)
    _process_init_duals(duals_slacks_ub)
    
    minimum_barrier_parameter = 1e-9
    barrier_parameter = 0.1
    interface.set_barrier_parameter(barrier_parameter)

    alpha_primal_max = 1
    alpha_dual_max = 1

    logger.info('{_iter:<10}{objective:<15}{primal_inf:<15}{dual_inf:<15}{compl_inf:<15}{barrier:<15}{alpha_p:<15}{alpha_d:<15}{time:<20}'.format(_iter='Iter',
                                                                                                                                                  objective='Objective',
                                                                                                                                                  primal_inf='Primal Inf',
                                                                                                                                                  dual_inf='Dual Inf',
                                                                                                                                                  compl_inf='Compl Inf',
                                                                                                                                                  barrier='Barrier',
                                                                                                                                                  alpha_p='Prim Step Size',
                                                                                                                                                  alpha_d='Dual Step Size',
                                                                                                                                                  time='Elapsed Time (s)'))

    for _iter in range(max_iter):
        interface.set_primals(primals)
        interface.set_slacks(slacks)
        interface.set_duals_eq(duals_eq)
        interface.set_duals_ineq(duals_ineq)
        interface.set_duals_primals_lb(duals_primals_lb)
        interface.set_duals_primals_ub(duals_primals_ub)
        interface.set_duals_slacks_lb(duals_slacks_lb)
        interface.set_duals_slacks_ub(duals_slacks_ub)
        
        primal_inf, dual_inf, complimentarity_inf = check_convergence(interface=interface, barrier=0)
        objective = interface.evaluate_objective()
        logger.info('{_iter:<10}{objective:<15.3e}{primal_inf:<15.3e}{dual_inf:<15.3e}{compl_inf:<15.3e}{barrier:<15.3e}{alpha_p:<15.3e}{alpha_d:<15.3e}{time:<20.2e}'.format(_iter=_iter,
                                                                                                                                                                              objective=objective,
                                                                                                                                                                              primal_inf=primal_inf,
                                                                                                                                                                              dual_inf=dual_inf,
                                                                                                                                                                              compl_inf=complimentarity_inf,
                                                                                                                                                                              barrier=barrier_parameter,
                                                                                                                                                                              alpha_p=alpha_primal_max,
                                                                                                                                                                              alpha_d=alpha_dual_max,
                                                                                                                                                                              time=time.time() - t0))
        if max(primal_inf, dual_inf, complimentarity_inf) <= tol:
            break
        primal_inf, dual_inf, complimentarity_inf = check_convergence(interface=interface, barrier=barrier_parameter)
        if max(primal_inf, dual_inf, complimentarity_inf) <= 0.1 * barrier_parameter:
            barrier_parameter = max(minimum_barrier_parameter, min(0.5*barrier_parameter, barrier_parameter**1.5))

        interface.set_barrier_parameter(barrier_parameter)
        kkt = interface.evaluate_primal_dual_kkt_matrix()
        rhs = interface.evaluate_primal_dual_kkt_rhs()
        linear_solver = MumpsInterface()  # icntl_options={1: 6, 2: 6, 3: 6, 4: 4})
        linear_solver.do_symbolic_factorization(kkt)
        linear_solver.do_numeric_factorization(kkt)
        delta = linear_solver.do_back_solve(rhs)

        interface.set_primal_dual_kkt_solution(delta)
        alpha_primal_max, alpha_dual_max = fraction_to_the_boundary(interface, 1-barrier_parameter)
        delta_primals = interface.get_delta_primals()
        delta_slacks = interface.get_delta_slacks()
        delta_duals_eq = interface.get_delta_duals_eq()
        delta_duals_ineq = interface.get_delta_duals_ineq()
        delta_duals_primals_lb = interface.get_delta_duals_primals_lb()
        delta_duals_primals_ub = interface.get_delta_duals_primals_ub()
        delta_duals_slacks_lb = interface.get_delta_duals_slacks_lb()
        delta_duals_slacks_ub = interface.get_delta_duals_slacks_ub()
        
        primals += alpha_primal_max * delta_primals
        slacks += alpha_primal_max * delta_slacks
        duals_eq += alpha_dual_max * delta_duals_eq
        duals_ineq += alpha_dual_max * delta_duals_ineq
        duals_primals_lb += alpha_dual_max * delta_duals_primals_lb
        duals_primals_ub += alpha_dual_max * delta_duals_primals_ub
        duals_slacks_lb += alpha_dual_max * delta_duals_slacks_lb
        duals_slacks_ub += alpha_dual_max * delta_duals_slacks_ub

    return primals, duals_eq, duals_ineq


def _process_init(x, lb, ub):
    if np.any((ub - lb) < 0):
        raise ValueError('Lower bounds for variables/inequalities should not be larger than upper bounds.')
    if np.any((ub - lb) == 0):
        raise ValueError('Variables and inequalities should not have equal lower and upper bounds.')

    lb_mask = build_bounds_mask(lb)
    ub_mask = build_bounds_mask(ub)

    lb_only = np.logical_and(lb_mask, np.logical_not(ub_mask))
    ub_only = np.logical_and(ub_mask, np.logical_not(lb_mask))
    lb_and_ub = np.logical_and(lb_mask, ub_mask)
    out_of_bounds = ((x >= ub) + (x <= lb))
    out_of_bounds_lb_only = np.logical_and(out_of_bounds, lb_only).nonzero()[0]
    out_of_bounds_ub_only = np.logical_and(out_of_bounds, ub_only).nonzero()[0]
    out_of_bounds_lb_and_ub = np.logical_and(out_of_bounds, lb_and_ub).nonzero()[0]

    np.put(x, out_of_bounds_lb_only, lb[out_of_bounds_lb_only] + 1)
    np.put(x, out_of_bounds_ub_only, ub[out_of_bounds_ub_only] - 1)

    cm = build_compression_matrix(lb_and_ub).tocsr()
    np.put(x, out_of_bounds_lb_and_ub, (0.5 * cm.transpose() * (cm*lb + cm*ub))[out_of_bounds_lb_and_ub])


def _process_init_duals(x):
    out_of_bounds = (x <= 0).nonzero()[0]
    np.put(x, out_of_bounds, 1)


def _fraction_to_the_boundary_helper_lb(tau, x, delta_x, xl_compressed, xl_compression_matrix):
    x_compressed = xl_compression_matrix * x
    delta_x_compressed = xl_compression_matrix * delta_x

    x = x_compressed
    delta_x = delta_x_compressed
    xl = xl_compressed

    negative_indices = (delta_x < 0).nonzero()[0]
    cols = negative_indices
    nnz = len(cols)
    rows = np.arange(nnz, dtype=np.int)
    data = np.ones(nnz)
    cm = coo_matrix((data, (rows, cols)), shape=(nnz, len(delta_x)))

    x = cm * x
    delta_x = cm * delta_x
    xl = cm * xl

    alpha = ((1 - tau) * (x - xl) + xl - x) / delta_x
    if len(alpha) == 0:
        return 1
    else:
        return alpha.min()


def _fraction_to_the_boundary_helper_ub(tau, x, delta_x, xu_compressed, xu_compression_matrix):
    x_compressed = xu_compression_matrix * x
    delta_x_compressed = xu_compression_matrix * delta_x

    x = x_compressed
    delta_x = delta_x_compressed
    xu = xu_compressed

    positive_indices = (delta_x > 0).nonzero()[0]
    cols = positive_indices
    nnz = len(cols)
    rows = np.arange(nnz, dtype=np.int)
    data = np.ones(nnz)
    cm = coo_matrix((data, (rows, cols)), shape=(nnz, len(delta_x)))

    x = cm * x
    delta_x = cm * delta_x
    xu = cm * xu

    alpha = (xu - (1 - tau) * (xu - x) - x) / delta_x
    if len(alpha) == 0:
        return 1
    else:
        return alpha.min()


def fraction_to_the_boundary(interface, tau):
    """
    Parameters
    ----------
    interface: pyomo.contrib.interior_point.interface.BaseInteriorPointInterface
    tau: float

    Returns
    -------
    alpha_primal_max: float
    alpha_dual_max: float
    """
    primals = interface.get_primals()
    slacks = interface.get_slacks()
    duals_eq = interface.get_duals_eq()
    duals_ineq = interface.get_duals_ineq()
    duals_primals_lb = interface.get_duals_primals_lb()
    duals_primals_ub = interface.get_duals_primals_ub()
    duals_slacks_lb = interface.get_duals_slacks_lb()
    duals_slacks_ub = interface.get_duals_slacks_ub()

    delta_primals = interface.get_delta_primals()
    delta_slacks = interface.get_delta_slacks()
    delta_duals_eq = interface.get_delta_duals_eq()
    delta_duals_ineq = interface.get_delta_duals_ineq()
    delta_duals_primals_lb = interface.get_delta_duals_primals_lb()
    delta_duals_primals_ub = interface.get_delta_duals_primals_ub()
    delta_duals_slacks_lb = interface.get_delta_duals_slacks_lb()
    delta_duals_slacks_ub = interface.get_delta_duals_slacks_ub()

    primals_lb_compressed = interface.get_primals_lb_compressed()
    primals_ub_compressed = interface.get_primals_ub_compressed()
    ineq_lb_compressed = interface.get_ineq_lb_compressed()
    ineq_ub_compressed = interface.get_ineq_ub_compressed()

    primals_lb_compression_matrix = interface.get_primals_lb_compression_matrix()
    primals_ub_compression_matrix = interface.get_primals_ub_compression_matrix()
    ineq_lb_compression_matrix = interface.get_ineq_lb_compression_matrix()
    ineq_ub_compression_matrix = interface.get_ineq_ub_compression_matrix()

    alpha_primal_max_a = _fraction_to_the_boundary_helper_lb(tau=tau,
                                                             x=primals,
                                                             delta_x=delta_primals,
                                                             xl_compressed=primals_lb_compressed,
                                                             xl_compression_matrix=primals_lb_compression_matrix)
    alpha_primal_max_b = _fraction_to_the_boundary_helper_ub(tau=tau,
                                                             x=primals,
                                                             delta_x=delta_primals,
                                                             xu_compressed=primals_ub_compressed,
                                                             xu_compression_matrix=primals_ub_compression_matrix)
    alpha_primal_max_c = _fraction_to_the_boundary_helper_lb(tau=tau,
                                                             x=slacks,
                                                             delta_x=delta_slacks,
                                                             xl_compressed=ineq_lb_compressed,
                                                             xl_compression_matrix=ineq_lb_compression_matrix)
    alpha_primal_max_d = _fraction_to_the_boundary_helper_ub(tau=tau,
                                                             x=slacks,
                                                             delta_x=delta_slacks,
                                                             xu_compressed=ineq_ub_compressed,
                                                             xu_compression_matrix=ineq_ub_compression_matrix)
    alpha_primal_max = min(alpha_primal_max_a, alpha_primal_max_b, alpha_primal_max_c, alpha_primal_max_d)

    alpha_dual_max_a = _fraction_to_the_boundary_helper_lb(tau=tau,
                                                           x=duals_primals_lb,
                                                           delta_x=delta_duals_primals_lb,
                                                           xl_compressed=np.zeros(len(duals_primals_lb)),
                                                           xl_compression_matrix=identity(len(duals_primals_lb), format='csr'))
    alpha_dual_max_b = _fraction_to_the_boundary_helper_lb(tau=tau,
                                                           x=duals_primals_ub,
                                                           delta_x=delta_duals_primals_ub,
                                                           xl_compressed=np.zeros(len(duals_primals_ub)),
                                                           xl_compression_matrix=identity(len(duals_primals_ub), format='csr'))
    alpha_dual_max_c = _fraction_to_the_boundary_helper_lb(tau=tau,
                                                           x=duals_slacks_lb,
                                                           delta_x=delta_duals_slacks_lb,
                                                           xl_compressed=np.zeros(len(duals_slacks_lb)),
                                                           xl_compression_matrix=identity(len(duals_slacks_lb), format='csr'))
    alpha_dual_max_d = _fraction_to_the_boundary_helper_lb(tau=tau,
                                                           x=duals_slacks_ub,
                                                           delta_x=delta_duals_slacks_ub,
                                                           xl_compressed=np.zeros(len(duals_slacks_ub)),
                                                           xl_compression_matrix=identity(len(duals_slacks_ub), format='csr'))
    alpha_dual_max = min(alpha_dual_max_a, alpha_dual_max_b, alpha_dual_max_c, alpha_dual_max_d)
    
    return alpha_primal_max, alpha_dual_max


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
