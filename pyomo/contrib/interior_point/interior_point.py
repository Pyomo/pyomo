#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.contrib.pynumero.interfaces.utils import build_bounds_mask, build_compression_matrix
import numpy as np
import logging
import time
from .linalg.results import LinearSolverStatus
from pyomo.common.timing import HierarchicalTimer
import enum


"""
Interface Requirements
----------------------
1) duals_primals_lb[i] must always be 0 if primals_lb[i] is -inf
2) duals_primals_ub[i] must always be 0 if primals_ub[i] is inf
3) duals_slacks_lb[i] must always be 0 if ineq_lb[i] is -inf
4) duals_slacks_ub[i] must always be 0 if ineq_ub[i] is inf
"""


ip_logger = logging.getLogger('interior_point')


class InteriorPointStatus(enum.Enum):
    optimal = 0
    error = 1


class LinearSolveContext(object):
    def __init__(self, 
            interior_point_logger, 
            linear_solver_logger,
            filename=None,
            level=logging.INFO):

        self.interior_point_logger = interior_point_logger
        self.linear_solver_logger = linear_solver_logger
        self.filename = filename

        if filename:
            self.handler = logging.FileHandler(filename)
            self.handler.setLevel(level)

    def __enter__(self):
        self.linear_solver_logger.propagate = False
        self.interior_point_logger.propagate = False
        if self.filename:
            self.linear_solver_logger.addHandler(self.handler)
            self.interior_point_logger.addHandler(self.handler)


    def __exit__(self, et, ev, tb):
        self.linear_solver_logger.propagate = True
        self.interior_point_logger.propagate = True
        if self.filename:
            self.linear_solver_logger.removeHandler(self.handler)
            self.interior_point_logger.removeHandler(self.handler)


# How should the RegContext work?
# TODO: in this class, use the linear_solver_context to ...
#       Use linear_solver_logger to write iter_no and reg_coef
#       
#       Define a method for logging IP_reg_info to the linear solver log
#       Method can be called within linear_solve_context
class FactorizationContext(object):
    def __init__(self, logger):
        # Any reason to pass in a logging level here?
        # ^ So the "regularization log" can have its own outlvl
        self.logger = logger

    def __enter__(self):
        self.logger.debug('Factorizing KKT')
        self.log_header()
        return self

    def __exit__(self, et, ev, tb):
        self.logger.debug('Finished factorizing KKT')
        # Will this swallow exceptions in this context?

    def log_header(self):
        self.logger.debug('{_iter:<10}'
                          '{reg_iter:<10}'
                          '{num_realloc:<10}'
                          '{reg_coef:<10}'
                          '{neg_eig:<10}'
                          '{status:<10}'.format(
            _iter='Iter',
            reg_iter='reg_iter',
            num_realloc='# realloc',
            reg_coef='reg_coef',
            neg_eig='neg_eig',
            status='status'))

    def log_info(self, _iter, reg_iter, num_realloc, coef, neg_eig, status):
        self.logger.debug('{_iter:<10}'
                          '{reg_iter:<10}'
                          '{num_realloc:<10}'
                          '{reg_coef:<10.2e}'
                          '{neg_eig:<10}'
                          '{status:<10}'.format(
            _iter=_iter,
            reg_iter=reg_iter,
            num_realloc=num_realloc,
            reg_coef=coef,
            neg_eig=str(neg_eig),
            status=status.name))


class InteriorPointSolver(object):
    """
    Class for creating interior point solvers with different options
    """
    def __init__(self,
                 linear_solver,
                 max_iter=100,
                 tol=1e-8,
                 linear_solver_log_filename=None,
                 max_reallocation_iterations=5,
                 reallocation_factor=2):
        self.linear_solver = linear_solver
        self.max_iter = max_iter
        self.tol = tol
        self.linear_solver_log_filename = linear_solver_log_filename
        self.max_reallocation_iterations = max_reallocation_iterations
        self.reallocation_factor = reallocation_factor
        self.base_eq_reg_coef = -1e-8
        self._barrier_parameter = 0.1
        self._minimum_barrier_parameter = 1e-9
        self.hess_reg_coef = 1e-4
        self.max_reg_iter = 6
        self.reg_factor_increase = 100

        self.logger = logging.getLogger('interior_point')
        self._iter = 0
        self.factorization_context = FactorizationContext(self.logger)

        if linear_solver_log_filename:
            with open(linear_solver_log_filename, 'w'):
                pass

        self.linear_solver_logger = self.linear_solver.getLogger()
        self.linear_solve_context = LinearSolveContext(self.logger,
                                        self.linear_solver_logger,
                                        self.linear_solver_log_filename)

    def update_barrier_parameter(self):
        self._barrier_parameter = max(self._minimum_barrier_parameter, 
                                    min(0.5 * self._barrier_parameter, 
                                        self._barrier_parameter ** 1.5))

    def set_linear_solver(self, linear_solver):
        """This method exists to hopefully make it easy to try the same IP
        algorithm with different linear solvers.
        Subclasses may have linear-solver specific methods, in which case
        this should not be called.

        Hopefully the linear solver interface can be standardized such that
        this is not a problem. (Need a generalized method for set_options)
        """
        self.linear_solver = linear_solver

    def set_interface(self, interface):
        self.interface = interface

    def solve(self, interface, timer=None, report_timing=False):
        """
        Parameters
        ----------
        interface: pyomo.contrib.interior_point.interface.BaseInteriorPointInterface
            The interior point interface. This object handles the function evaluation, 
            building the KKT matrix, and building the KKT right hand side.
        timer: HierarchicalTimer
        report_timing: bool
        """
        linear_solver = self.linear_solver
        max_iter = self.max_iter
        tol = self.tol
        if timer is None:
            timer = HierarchicalTimer()

        timer.start('IP solve')
        timer.start('init')

        self._barrier_parameter = 0.1

        self.set_interface(interface)

        t0 = time.time()
        primals = interface.init_primals().copy()
        slacks = interface.init_slacks().copy()
        duals_eq = interface.init_duals_eq().copy()
        duals_ineq = interface.init_duals_ineq().copy()
        duals_primals_lb = interface.init_duals_primals_lb().copy()
        duals_primals_ub = interface.init_duals_primals_ub().copy()
        duals_slacks_lb = interface.init_duals_slacks_lb().copy()
        duals_slacks_ub = interface.init_duals_slacks_ub().copy()

        self.process_init(primals, interface.primals_lb(), interface.primals_ub())
        self.process_init(slacks, interface.ineq_lb(), interface.ineq_ub())
        self.process_init_duals_lb(duals_primals_lb, self.interface.primals_lb())
        self.process_init_duals_ub(duals_primals_ub, self.interface.primals_ub())
        self.process_init_duals_lb(duals_slacks_lb, self.interface.ineq_lb())
        self.process_init_duals_ub(duals_slacks_ub, self.interface.ineq_ub())
        
        interface.set_barrier_parameter(self._barrier_parameter)

        alpha_primal_max = 1
        alpha_dual_max = 1

        self.logger.info('{_iter:<6}'
                         '{objective:<11}'
                         '{primal_inf:<11}'
                         '{dual_inf:<11}'
                         '{compl_inf:<11}'
                         '{barrier:<11}'
                         '{alpha_p:<11}'
                         '{alpha_d:<11}'
                         '{reg:<11}'
                         '{time:<7}'.format(_iter='Iter',
                                            objective='Objective',
                                            primal_inf='Prim Inf',
                                            dual_inf='Dual Inf',
                                            compl_inf='Comp Inf',
                                            barrier='Barrier',
                                            alpha_p='Prim Step',
                                            alpha_d='Dual Step',
                                            reg='Reg',
                                            time='Time'))

        reg_coef = 0

        timer.stop('init')
        status = InteriorPointStatus.error

        for _iter in range(max_iter):
            self._iter = _iter

            interface.set_primals(primals)
            interface.set_slacks(slacks)
            interface.set_duals_eq(duals_eq)
            interface.set_duals_ineq(duals_ineq)
            interface.set_duals_primals_lb(duals_primals_lb)
            interface.set_duals_primals_ub(duals_primals_ub)
            interface.set_duals_slacks_lb(duals_slacks_lb)
            interface.set_duals_slacks_ub(duals_slacks_ub)

            timer.start('convergence check')
            primal_inf, dual_inf, complimentarity_inf = \
                    self.check_convergence(barrier=0, timer=timer)
            timer.stop('convergence check')
            objective = interface.evaluate_objective()
            self.logger.info('{_iter:<6}'
                             '{objective:<11.2e}'
                             '{primal_inf:<11.2e}'
                             '{dual_inf:<11.2e}'
                             '{compl_inf:<11.2e}'
                             '{barrier:<11.2e}'
                             '{alpha_p:<11.2e}'
                             '{alpha_d:<11.2e}'
                             '{reg:<11.2e}'
                             '{time:<7.3f}'.format(_iter=_iter,
                                                   objective=objective,
                                                   primal_inf=primal_inf,
                                                   dual_inf=dual_inf,
                                                   compl_inf=complimentarity_inf,
                                                   barrier=self._barrier_parameter,
                                                   alpha_p=alpha_primal_max,
                                                   alpha_d=alpha_dual_max,
                                                   reg=reg_coef,
                                                   time=time.time() - t0))

            if max(primal_inf, dual_inf, complimentarity_inf) <= tol:
                status = InteriorPointStatus.optimal
                break
            timer.start('convergence check')
            primal_inf, dual_inf, complimentarity_inf = \
                    self.check_convergence(
                            barrier=self._barrier_parameter, 
                            timer=timer)
            timer.stop('convergence check')
            if max(primal_inf, dual_inf, complimentarity_inf) \
                    <= 0.1 * self._barrier_parameter:
                # This comparison is made with barrier problem infeasibility.
                # Sometimes have trouble getting dual infeasibility low enough
                self.update_barrier_parameter()

            interface.set_barrier_parameter(self._barrier_parameter)
            timer.start('eval')
            timer.start('eval kkt')
            kkt = interface.evaluate_primal_dual_kkt_matrix(timer=timer)
            timer.stop('eval kkt')
            timer.start('eval rhs')
            rhs = interface.evaluate_primal_dual_kkt_rhs(timer=timer)
            timer.stop('eval rhs')
            timer.stop('eval')

            # Factorize linear system
            timer.start('factorize')
            reg_coef = self.factorize(kkt=kkt, timer=timer)
            timer.stop('factorize')

            timer.start('back solve')
            with self.linear_solve_context:
                self.logger.info('Iter: %s' % self._iter)
                delta = linear_solver.do_back_solve(rhs)
            timer.stop('back solve')

            interface.set_primal_dual_kkt_solution(delta)
            timer.start('frac boundary')
            alpha_primal_max, alpha_dual_max = \
                    self.fraction_to_the_boundary()
            timer.stop('frac boundary')
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

        timer.stop('IP solve')
        if report_timing:
            print(timer)
        return status

    def factorize(self, kkt, timer=None):
        desired_n_neg_evals = (self.interface.n_eq_constraints() +
                               self.interface.n_ineq_constraints())
        reg_iter = 0
        with self.factorization_context as fact_con:
            status, num_realloc = try_factorization_and_reallocation(
                    kkt=kkt,
                    linear_solver=self.linear_solver,
                    reallocation_factor=self.reallocation_factor,
                    max_iter=self.max_reallocation_iterations,
                    timer=timer)
            if status not in {LinearSolverStatus.successful, 
                              LinearSolverStatus.singular}:
                raise RuntimeError(
                    'Could not factorize KKT system; linear solver status: ' 
                    + str(status))

            if status == LinearSolverStatus.successful:
                neg_eig = self.linear_solver.get_inertia()[1]
            else:
                neg_eig = None
            fact_con.log_info(
                    _iter=self._iter, 
                    reg_iter=reg_iter, 
                    num_realloc=num_realloc,
                    coef=0, 
                    neg_eig=neg_eig, 
                    status=status)
            reg_iter += 1

            if status == LinearSolverStatus.singular:
                constraint_reg_coef = (self.base_eq_reg_coef * 
                                       self._barrier_parameter**0.25)
                kkt = self.interface.regularize_equality_gradient(
                        kkt=kkt,
                        coef=constraint_reg_coef,
                        copy_kkt=False)

            total_hess_reg_coef = self.hess_reg_coef
            last_hess_reg_coef = 0

            while (neg_eig != desired_n_neg_evals or 
                    status == LinearSolverStatus.singular):
                kkt = self.interface.regularize_hessian(
                        kkt=kkt,
                        coef=total_hess_reg_coef - last_hess_reg_coef,
                        copy_kkt=False)
                status, num_realloc = try_factorization_and_reallocation(
                        kkt=kkt,
                        linear_solver=self.linear_solver,
                        reallocation_factor=self.reallocation_factor,
                        max_iter=self.max_reallocation_iterations,
                        timer=timer)
                if status != LinearSolverStatus.successful:
                    raise RuntimeError(
                    'Could not factorize KKT system; linear solver status: ' 
                    + str(status))
                neg_eig = self.linear_solver.get_inertia()[1]
                fact_con.log_info(
                        _iter=self._iter, 
                        reg_iter=reg_iter, 
                        num_realloc=num_realloc,
                        coef=total_hess_reg_coef, 
                        neg_eig=neg_eig, 
                        status=status)
                reg_iter += 1
                if reg_iter > self.max_reg_iter:
                    raise RuntimeError(
                    'Exceeded maximum number of regularization iterations.')
                last_hess_reg_coef = total_hess_reg_coef
                total_hess_reg_coef *= self.reg_factor_increase

        return last_hess_reg_coef

    def process_init(self, x, lb, ub):
        process_init(x, lb, ub)

    def process_init_duals_lb(self, x, lb):
        process_init_duals_lb(x, lb)

    def process_init_duals_ub(self, x, ub):
        process_init_duals_ub(x, ub)

    def check_convergence(self, barrier, timer=None):
        """
        Parameters
        ----------
        barrier: float
        timer: HierarchicalTimer
    
        Returns
        -------
        primal_inf: float
        dual_inf: float
        complimentarity_inf: float
        """
        if timer is None:
            timer = HierarchicalTimer()

        interface = self.interface
        slacks = interface.get_slacks()
        timer.start('grad obj')
        grad_obj = interface.get_obj_factor() * \
                interface.evaluate_grad_objective()
        timer.stop('grad obj')
        timer.start('jac eq')
        jac_eq = interface.evaluate_jacobian_eq()
        timer.stop('jac eq')
        timer.start('jac ineq')
        jac_ineq = interface.evaluate_jacobian_ineq()
        timer.stop('jac ineq')
        timer.start('eq cons')
        eq_resid = interface.evaluate_eq_constraints()
        timer.stop('eq cons')
        timer.start('ineq cons')
        ineq_resid = interface.evaluate_ineq_constraints() - slacks
        timer.stop('ineq cons')
        primals = interface.get_primals()
        duals_eq = interface.get_duals_eq()
        duals_ineq = interface.get_duals_ineq()
        duals_primals_lb = interface.get_duals_primals_lb()
        duals_primals_ub = interface.get_duals_primals_ub()
        duals_slacks_lb = interface.get_duals_slacks_lb()
        duals_slacks_ub = interface.get_duals_slacks_ub()

        primals_lb = interface.primals_lb()
        primals_ub = interface.primals_ub()
        primals_lb_mod = primals_lb.copy()
        primals_ub_mod = primals_ub.copy()
        primals_lb_mod[np.isneginf(primals_lb)] = 0  # these entries get multiplied by 0
        primals_ub_mod[np.isinf(primals_ub)] = 0  # these entries get multiplied by 0

        ineq_lb = interface.ineq_lb()
        ineq_ub = interface.ineq_ub()
        ineq_lb_mod = ineq_lb.copy()
        ineq_ub_mod = ineq_ub.copy()
        ineq_lb_mod[np.isneginf(ineq_lb)] = 0  # these entries get multiplied by 0
        ineq_ub_mod[np.isinf(ineq_ub)] = 0  # these entries get multiplied by 0

        timer.start('grad_lag_primals')
        grad_lag_primals = grad_obj + jac_eq.transpose() * duals_eq
        grad_lag_primals += jac_ineq.transpose() * duals_ineq
        grad_lag_primals -= duals_primals_lb
        grad_lag_primals += duals_primals_ub
        timer.stop('grad_lag_primals')
        timer.start('grad_lag_slacks')
        grad_lag_slacks = (-duals_ineq -
                           duals_slacks_lb +
                           duals_slacks_ub)
        timer.stop('grad_lag_slacks')
        timer.start('bound resids')
        primals_lb_resid = (primals - primals_lb_mod) * duals_primals_lb - barrier
        primals_ub_resid = (primals_ub_mod - primals) * duals_primals_ub - barrier
        primals_lb_resid[np.isneginf(primals_lb)] = 0
        primals_ub_resid[np.isinf(primals_ub)] = 0
        slacks_lb_resid = (slacks - ineq_lb_mod) * duals_slacks_lb - barrier
        slacks_ub_resid = (ineq_ub_mod - slacks) * duals_slacks_ub - barrier
        slacks_lb_resid[np.isneginf(ineq_lb)] = 0
        slacks_ub_resid[np.isinf(ineq_ub)] = 0
        timer.stop('bound resids')

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
        complimentarity_inf = max(max_primals_lb_resid, max_primals_ub_resid,
                                  max_slacks_lb_resid, max_slacks_ub_resid)
    
        return primal_inf, dual_inf, complimentarity_inf

    def fraction_to_the_boundary(self):
        return fraction_to_the_boundary(self.interface, 
                                        1 - self._barrier_parameter)


def try_factorization_and_reallocation(kkt, linear_solver, reallocation_factor,
        max_iter, timer=None):
    if timer is None:
        timer = HierarchicalTimer()

    assert max_iter >= 1
    for count in range(max_iter):
        timer.start('symbolic')
        """
        Performance could be improved significantly by only performing 
        symbolic factorization once.

        However, we first have to make sure the nonzero structure 
        (and ordering of row and column arrays) of the KKT matrix never 
        changes. We have not had time to test this thoroughly, yet. 
        """
        res = linear_solver.do_symbolic_factorization(
                matrix=kkt, 
                raise_on_error=False)
        timer.stop('symbolic')
        if res.status == LinearSolverStatus.successful:
            timer.start('numeric')
            res = linear_solver.do_numeric_factorization(
                    matrix=kkt, 
                    raise_on_error=False)
            timer.stop('numeric')
        status = res.status
        if status == LinearSolverStatus.not_enough_memory:
            linear_solver.increase_memory_allocation(reallocation_factor)
        else:
            break
    return status, count


def _fraction_to_the_boundary_helper_lb(tau, x, delta_x, xl):
    delta_x_mod = delta_x.copy()
    delta_x_mod[delta_x_mod == 0] = 1
    alpha = -tau * (x - xl) / delta_x_mod
    alpha[delta_x >= 0] = np.inf
    if alpha.size == 0:
        return 1
    else:
        return min(alpha.min(), 1)


def _fraction_to_the_boundary_helper_ub(tau, x, delta_x, xu):
    delta_x_mod = delta_x.copy()
    delta_x_mod[delta_x_mod == 0] = 1
    alpha = tau * (xu - x) / delta_x_mod
    alpha[delta_x <= 0] = np.inf
    if alpha.size == 0:
        return 1
    else:
        return min(alpha.min(), 1)


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
    duals_primals_lb = interface.get_duals_primals_lb()
    duals_primals_ub = interface.get_duals_primals_ub()
    duals_slacks_lb = interface.get_duals_slacks_lb()
    duals_slacks_ub = interface.get_duals_slacks_ub()

    delta_primals = interface.get_delta_primals()
    delta_slacks = interface.get_delta_slacks()
    delta_duals_primals_lb = interface.get_delta_duals_primals_lb()
    delta_duals_primals_ub = interface.get_delta_duals_primals_ub()
    delta_duals_slacks_lb = interface.get_delta_duals_slacks_lb()
    delta_duals_slacks_ub = interface.get_delta_duals_slacks_ub()

    primals_lb = interface.primals_lb()
    primals_ub = interface.primals_ub()
    ineq_lb = interface.ineq_lb()
    ineq_ub = interface.ineq_ub()

    alpha_primal_max_a = _fraction_to_the_boundary_helper_lb(
        tau=tau,
        x=primals,
        delta_x=delta_primals,
        xl=primals_lb)
    alpha_primal_max_b = _fraction_to_the_boundary_helper_ub(
        tau=tau,
        x=primals,
        delta_x=delta_primals,
        xu=primals_ub)
    alpha_primal_max_c = _fraction_to_the_boundary_helper_lb(
        tau=tau,
        x=slacks,
        delta_x=delta_slacks,
        xl=ineq_lb)
    alpha_primal_max_d = _fraction_to_the_boundary_helper_ub(
        tau=tau,
        x=slacks,
        delta_x=delta_slacks,
        xu=ineq_ub)
    alpha_primal_max = min(alpha_primal_max_a, alpha_primal_max_b,
                           alpha_primal_max_c, alpha_primal_max_d)

    alpha_dual_max_a = _fraction_to_the_boundary_helper_lb(
        tau=tau,
        x=duals_primals_lb,
        delta_x=delta_duals_primals_lb,
        xl=np.zeros(duals_primals_lb.size))
    alpha_dual_max_b = _fraction_to_the_boundary_helper_lb(
        tau=tau,
        x=duals_primals_ub,
        delta_x=delta_duals_primals_ub,
        xl=np.zeros(duals_primals_ub.size))
    alpha_dual_max_c = _fraction_to_the_boundary_helper_lb(
        tau=tau,
        x=duals_slacks_lb,
        delta_x=delta_duals_slacks_lb,
        xl=np.zeros(duals_slacks_lb.size))
    alpha_dual_max_d = _fraction_to_the_boundary_helper_lb(
        tau=tau,
        x=duals_slacks_ub,
        delta_x=delta_duals_slacks_ub,
        xl=np.zeros(duals_slacks_ub.size))
    alpha_dual_max = min(alpha_dual_max_a, alpha_dual_max_b,
                         alpha_dual_max_c, alpha_dual_max_d)

    return alpha_primal_max, alpha_dual_max


def process_init(x, lb, ub):
    if np.any((ub - lb) < 0):
        raise ValueError(
            'Lower bounds for variables/inequalities should not be larger '
            'than upper bounds.')
    if np.any((ub - lb) == 0):
        raise ValueError(
            'Variables and inequalities should not have equal lower and upper '
            'bounds.')

    lb_mask = build_bounds_mask(lb)
    ub_mask = build_bounds_mask(ub)

    lb_only = np.logical_and(lb_mask, np.logical_not(ub_mask))
    ub_only = np.logical_and(ub_mask, np.logical_not(lb_mask))
    lb_and_ub = np.logical_and(lb_mask, ub_mask)
    out_of_bounds = ((x >= ub) + (x <= lb))
    out_of_bounds_lb_only = np.logical_and(out_of_bounds, lb_only)
    out_of_bounds_ub_only = np.logical_and(out_of_bounds, ub_only)
    out_of_bounds_lb_and_ub = np.logical_and(out_of_bounds, lb_and_ub)

    cm = build_compression_matrix(out_of_bounds_lb_only)
    x[out_of_bounds_lb_only] = cm * (lb + 1)

    cm = build_compression_matrix(out_of_bounds_ub_only)
    x[out_of_bounds_ub_only] = cm * (ub - 1)

    del cm
    cm1 = build_compression_matrix(lb_and_ub)
    cm2 = build_compression_matrix(out_of_bounds_lb_and_ub)
    x[out_of_bounds_lb_and_ub] = cm2 * (0.5 * cm1.transpose() * (cm1 * lb + cm1 * ub))


def process_init_duals_lb(x, lb):
    x[x <= 0] = 1
    x[np.isneginf(lb)] = 0


def process_init_duals_ub(x, ub):
    x[x <= 0] = 1
    x[np.isinf(ub)] = 0
