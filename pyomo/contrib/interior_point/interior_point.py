from pyomo.contrib.interior_point.interface import InteriorPointInterface, BaseInteriorPointInterface
from pyomo.contrib.pynumero.interfaces.utils import build_bounds_mask, build_compression_matrix
from scipy.sparse import tril, coo_matrix, identity
import numpy as np
import logging
import time


ip_logger = logging.getLogger('interior_point')


class InteriorPointSolver(object):
    '''Class for creating interior point solvers with different options
    '''
    def __init__(self, linear_solver, max_iter=100, tol=1e-8, 
            regularize_kkt=False):
        self.linear_solver = linear_solver
        self.max_iter = max_iter
        self.tol = tol
        self.regularize_kkt = regularize_kkt

        self.logger = logging.getLogger('interior_point')


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


    def solve(self, interface, **kwargs):
        """
        Parameters
        ----------
        interface: pyomo.contrib.interior_point.interface.BaseInteriorPointInterface
            The interior point interface. This object handles the function evaluation, 
            building the KKT matrix, and building the KKT right hand side.
        linear_solver: pyomo.contrib.interior_point.linalg.base_linear_solver_interface.LinearSolverInterface
            A linear solver with the interface defined by LinearSolverInterface.
        max_iter: int
            The maximum number of iterations
        tol: float
            The tolerance for terminating the algorithm.
        """
        linear_solver = self.linear_solver
        max_iter = kwargs.pop('max_iter', self.max_iter)
        tol = kwargs.pop('tol', self.tol)
        regularize_kkt = kwargs.pop('regularize_kkt', self.regularize_kkt)
        max_reg_coef = kwargs.pop('max_reg_coef', 1e10)
        reg_factor_increase = kwargs.pop('reg_factor_increase', 1e2)

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

        self.process_init(primals, interface.get_primals_lb(), interface.get_primals_ub())
        self.process_init(slacks, interface.get_ineq_lb(), interface.get_ineq_ub())
        self.process_init_duals(duals_primals_lb)
        self.process_init_duals(duals_primals_ub)
        self.process_init_duals(duals_slacks_lb)
        self.process_init_duals(duals_slacks_ub)
        
        minimum_barrier_parameter = 1e-9
        barrier_parameter = 0.1
        interface.set_barrier_parameter(barrier_parameter)

        alpha_primal_max = 1
        alpha_dual_max = 1

        self.logger.info('{_iter:<10}{objective:<15}{primal_inf:<15}{dual_inf:<15}{compl_inf:<15}{barrier:<15}{alpha_p:<15}{alpha_d:<15}{time:<20}'.format(_iter='Iter',
                                        objective='Objective',
                                        primal_inf='Primal Inf',
                                        dual_inf='Dual Inf',
                                        compl_inf='Compl Inf',
                                        barrier='Barrier',
                                        alpha_p='Prim Step Size',
                                        alpha_d='Dual Step Size',
                                        time='Elapsed Time (s)'))
        
        # Write header line to linear solver log
        # Every linear_solver should /have/ a log_header method. Whether it
        # does anything is another question.
        linear_solver.log_header()

        for _iter in range(max_iter):
            interface.set_primals(primals)
            interface.set_slacks(slacks)
            interface.set_duals_eq(duals_eq)
            interface.set_duals_ineq(duals_ineq)
            interface.set_duals_primals_lb(duals_primals_lb)
            interface.set_duals_primals_ub(duals_primals_ub)
            interface.set_duals_slacks_lb(duals_slacks_lb)
            interface.set_duals_slacks_ub(duals_slacks_ub)
            
            primal_inf, dual_inf, complimentarity_inf = \
                    self.check_convergence(interface=interface, barrier=0)
            objective = interface.evaluate_objective()
            self.logger.info('{_iter:<10}{objective:<15.3e}{primal_inf:<15.3e}{dual_inf:<15.3e}{compl_inf:<15.3e}{barrier:<15.3e}{alpha_p:<15.3e}{alpha_d:<15.3e}{time:<20.2e}'.format(_iter=_iter,
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
            primal_inf, dual_inf, complimentarity_inf = \
                    self.check_convergence(interface=interface, 
                                           barrier=barrier_parameter)
            if max(primal_inf, dual_inf, complimentarity_inf) \
                    <= 0.1 * barrier_parameter:
                # This comparison is made with barrier problem infeasibility.
                # Sometimes have trouble getting dual infeasibility low enough
                barrier_parameter = max(minimum_barrier_parameter,
                                        min(0.5*barrier_parameter,
                                            barrier_parameter**1.5))

            interface.set_barrier_parameter(barrier_parameter)
            kkt = interface.evaluate_primal_dual_kkt_matrix()
            rhs = interface.evaluate_primal_dual_kkt_rhs()

            # Factorize linear system, with or without regularization:
            if not regularize_kkt:
                self.factorize_linear_system(kkt)
            else:
                self.factorize_with_regularization(kkt,
                        max_reg_coef=max_reg_coef,
                        factor_increase=reg_factor_increase)

            delta = linear_solver.do_back_solve(rhs)

            # Log some relevant info from linear solver
            linear_solver.log_info(_iter)

            interface.set_primal_dual_kkt_solution(delta)
            alpha_primal_max, alpha_dual_max = \
                    self.fraction_to_the_boundary(interface, 1-barrier_parameter)
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


    def factorize_linear_system(self, kkt):
        self.linear_solver.do_symbolic_factorization(kkt)
        self.linear_solver.do_numeric_factorization(kkt)
        # Should I return something here?


    def factorize_with_regularization(self, kkt,
                                      max_reg_coef=1e10,
                                      factor_increase=1e2):
        linear_solver = self.linear_solver
        desired_n_neg_evals = (self.interface._nlp.n_eq_constraints() + 
                               self.interface._nlp.n_ineq_constraints())

        reg_kkt_1 = kkt
        reg_coef = 1e-4

        err = linear_solver.try_factorization(kkt)
        if linear_solver.is_numerically_singular(err):
            self.logger.info(' KKT matrix is numerically singular. '
                             'Regularizing equality gradient...')
            reg_kkt_1 = self.interface.regularize_equality_gradient(kkt)
            err = linear_solver.try_factorization(reg_kkt_1)

        if (linear_solver.is_numerically_singular(err) or
                linear_solver.get_inertia()[1] != desired_n_neg_evals):

            self.logger.info(' KKT matrix has incorrect inertia.. '
                             'Regularizing Hessian...')

            # Every linear_solver should have an interface to a logger
            # like this:
            # Should probably use a context manager to properly log regularization
            # iterations...
            linear_solver.logger.info(' Regularizing Hessian')
            linear_solver.log_header(include_error=False,
                                     extra_fields=['Coefficient'])
            linear_solver.log_info(include_error=False)

            while reg_coef <= max_reg_coef:
                # Construct new regularized KKT matrix
                reg_kkt_2 = self.interface.regularize_hessian(reg_kkt_1, 
                                                              reg_coef)

                err = linear_solver.try_factorization(reg_kkt_2)
                linear_solver.log_info(include_error=False, 
                                       extra_fields=[reg_coef])
                if (linear_solver.is_numerically_singular(err) or
                        linear_solver.get_inertia()[1] != desired_n_neg_evals):
                    reg_coef = reg_coef * factor_increase
                else:
                    # Success
                    break

            if reg_coef > max_reg_coef:
                raise RuntimeError(
                    'Regularization coefficient has exceeded maximum. '
                    'At this point IPOPT would enter feasibility restoration.')

            linear_solver.logger.info(' Exiting regularization')

                # Should log more info about regularization:
                #   - null pivot tolerance
                #   - "how far" the factorization got before failing


    def process_init(self, x, lb, ub):
        if np.any((ub - lb) < 0):
            raise ValueError(
        'Lower bounds for variables/inequalities should not be larger than upper bounds.')
        if np.any((ub - lb) == 0):
            raise ValueError(
        'Variables and inequalities should not have equal lower and upper bounds.')
    
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
        np.put(x, out_of_bounds_lb_and_ub, 
                (0.5 * cm.transpose() * (cm*lb + cm*ub))[out_of_bounds_lb_and_ub])
    
    
    def process_init_duals(self, x):
        out_of_bounds = (x <= 0).nonzero()[0]
        np.put(x, out_of_bounds, 1)
    
    
    def _fraction_to_the_boundary_helper_lb(self, tau, x, delta_x, xl_compressed, 
                                            xl_compression_matrix):
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
    
        #alpha = ((1 - tau) * (x - xl) + xl - x) / delta_x
        # Why not reduce this?
        alpha = -tau * (x - xl) / delta_x
        if len(alpha) == 0:
            return 1
        else:
            return min(alpha.min(), 1)
    
    
    def _fraction_to_the_boundary_helper_ub(self, tau, x, delta_x, xu_compressed, 
                                            xu_compression_matrix):
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
    
        #alpha = (xu - (1 - tau) * (xu - x) - x) / delta_x
        alpha = tau * (xu - x) / delta_x
        if len(alpha) == 0:
            return 1
        else:
            return min(alpha.min(), 1)
    
    
    def fraction_to_the_boundary(self, interface, tau):
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
    
        alpha_primal_max_a = self._fraction_to_the_boundary_helper_lb(
                              tau=tau,
                              x=primals,
                              delta_x=delta_primals,
                              xl_compressed=primals_lb_compressed,
                              xl_compression_matrix=primals_lb_compression_matrix)
        alpha_primal_max_b = self._fraction_to_the_boundary_helper_ub(
                              tau=tau,
                              x=primals,
                              delta_x=delta_primals,
                              xu_compressed=primals_ub_compressed,
                              xu_compression_matrix=primals_ub_compression_matrix)
        alpha_primal_max_c = self._fraction_to_the_boundary_helper_lb(
                              tau=tau,
                              x=slacks,
                              delta_x=delta_slacks,
                              xl_compressed=ineq_lb_compressed,
                              xl_compression_matrix=ineq_lb_compression_matrix)
        alpha_primal_max_d = self._fraction_to_the_boundary_helper_ub(
                              tau=tau,
                              x=slacks,
                              delta_x=delta_slacks,
                              xu_compressed=ineq_ub_compressed,
                              xu_compression_matrix=ineq_ub_compression_matrix)
        alpha_primal_max = min(alpha_primal_max_a, alpha_primal_max_b, 
                               alpha_primal_max_c, alpha_primal_max_d)
    
        alpha_dual_max_a = self._fraction_to_the_boundary_helper_lb(
                            tau=tau,
                            x=duals_primals_lb,
                            delta_x=delta_duals_primals_lb,
                            xl_compressed=np.zeros(len(duals_primals_lb)),
                            xl_compression_matrix=identity(len(duals_primals_lb), 
                                                           format='csr'))
        alpha_dual_max_b = self._fraction_to_the_boundary_helper_lb(
                            tau=tau,
                            x=duals_primals_ub,
                            delta_x=delta_duals_primals_ub,
                            xl_compressed=np.zeros(len(duals_primals_ub)),
                            xl_compression_matrix=identity(len(duals_primals_ub), 
                                                           format='csr'))
        alpha_dual_max_c = self._fraction_to_the_boundary_helper_lb(
                            tau=tau,
                            x=duals_slacks_lb,
                            delta_x=delta_duals_slacks_lb,
                            xl_compressed=np.zeros(len(duals_slacks_lb)),
                            xl_compression_matrix=identity(len(duals_slacks_lb), 
                                                           format='csr'))
        alpha_dual_max_d = self._fraction_to_the_boundary_helper_lb(
                            tau=tau,
                            x=duals_slacks_ub,
                            delta_x=delta_duals_slacks_ub,
                            xl_compressed=np.zeros(len(duals_slacks_ub)),
                            xl_compression_matrix=identity(len(duals_slacks_ub), 
                                                           format='csr'))
        alpha_dual_max = min(alpha_dual_max_a, alpha_dual_max_b,
                             alpha_dual_max_c, alpha_dual_max_d)
        
        return alpha_primal_max, alpha_dual_max
    
    
    def check_convergence(self, interface, barrier):
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
                            primals_lb_compression_matrix.transpose() * 
                                                             duals_primals_lb +
                            primals_ub_compression_matrix.transpose() * 
                                                             duals_primals_ub)
        grad_lag_slacks = (-duals_ineq -
                           ineq_lb_compression_matrix.transpose() * duals_slacks_lb +
                           ineq_ub_compression_matrix.transpose() * duals_slacks_ub)
        eq_resid = interface.evaluate_eq_constraints()
        ineq_resid = interface.evaluate_ineq_constraints() - slacks
        primals_lb_resid = (primals_lb_compression_matrix * primals - 
                            primals_lb_compressed) * duals_primals_lb - barrier
        primals_ub_resid = (primals_ub_compressed - 
                            primals_ub_compression_matrix * primals) * \
                            duals_primals_ub - barrier
        slacks_lb_resid = (ineq_lb_compression_matrix * slacks - ineq_lb_compressed) \
                           * duals_slacks_lb - barrier
        slacks_ub_resid = (ineq_ub_compressed - ineq_ub_compression_matrix * slacks) \
                           * duals_slacks_ub - barrier
    
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
    

