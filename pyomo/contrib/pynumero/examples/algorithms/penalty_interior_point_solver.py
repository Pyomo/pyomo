#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.contrib.pynumero.examples.algorithms.print_utils import (print_nlp_info,
                                                                    print_summary)
from pyomo.contrib.pynumero.sparse import (BlockVector,
                                           BlockSymMatrix,
                                           diagonal_matrix)

from pyomo.contrib.pynumero.examples.algorithms.kkt_solver import FullKKTSolver

import math as pymath
import numpy as np
import logging
import json

from pyomo.contrib.pynumero.interfaces import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_compositions import CompositeNLP
from pyomo.contrib.pynumero.linalg.intrinsics import norm as pynumero_norm
from pyomo.contrib.pynumero.interfaces.nlp_state import BasicNLPState, NLPState
from pyomo.contrib.pynumero.sparse import empty_matrix
import pyomo.contrib.pynumero as pn
from scipy.sparse import identity
import sys


if not pn.mumps_available and not pn.ma27_available:
    raise ImportError('Need MA27 or MUMPS to run pynumero interior-point')

def optimality_error(nlp_state, scaled=True, smax=100.0, sc=1.0, sd=1.0):

    r = np.zeros(3)
    r[0] = nlp_state.primal_infeasibility()
    r[1] = nlp_state.dual_infeasibility()/sd
    r[2] = nlp_state.complementarity_infeasibility()/sc

    return np.max(r)

def optimality_error_penalty(nlp_state, scaled=True, smax=100.0, sc=1.0, sd=1.0):

    r = np.zeros(4)
    r[0] = pynumero_norm(nlp_state.grad_lag_bar_x(), ord=np.inf)
    r[1] = pynumero_norm(nlp_state.grad_lag_bar_s(), ord=np.inf)
    r[2] = pynumero_norm(nlp_state.evaluate_c() - nlp_state.correction_c(),
                         ord=np.inf)
    r[3] = pynumero_norm(nlp_state.residual_d() - nlp_state.correction_d(),
                         ord=np.inf)
    return np.max(r)

# state with interior point functions
class PIPNLPState(NLPState):

    def __init__(self, nlp, miu, rho, **kwargs):
        self._miu = miu
        self._rho = rho
        super(PIPNLPState, self).__init__(nlp, **kwargs)

    @property
    def miu(self):
        return self._miu

    @miu.setter
    def miu(self, other):
        self._miu = other

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, other):
        self._rho = other

    def grad_lag_bar_x(self):
        grad_x_lag_bar = self.grad_objective() + \
                         self.jacobian_c().T * self.yc + \
                         self.jacobian_d().T * self.yd + \
                         self.Pxu * (self._miu / self.slack_xu()) - \
                         self.Pxl * (self._miu / self.slack_xl())
        return grad_x_lag_bar

    def grad_lag_bar_s(self):
        grad_s_lag_bar = self.Pdu * (self._miu / self.slack_su()) - \
                         self.Pdl * (self._miu / self.slack_sl()) - \
                         self.yd
        return grad_s_lag_bar

    def barrier_objective(self):
        return self.objective() - self.barrier_term()

    def barrier_term(self):
        return self._miu * (np.sum(np.log(self.slack_xl())) + \
                           np.sum(np.log(self.slack_xu())) + \
                           np.sum(np.log(self.slack_sl())) + \
                           np.sum(np.log(self.slack_su())))

    def barrier_pentalty_objective(self):
        return self.barrier_objective() + self.penalty_term()

    def penalty_term(self):
        return self.rho * cd_infeasibility

    def cd_infeasibility(self):
        sum_c_square = np.sum(np.square(self.evaluate_c()))
        sum_d_square = np.sum(np.square(self.residual_d()))
        return np.sqrt(sum_d_square + sum_c_square)

    def complementarity_infeasibility(self):

        r = np.zeros(4)
        if self.zl.size > 0:
            r[0] = pynumero_norm(np.multiply(self.slack_xl(), self.zl) - \
                                  self._miu, ord=np.inf)
        if self.zu.size > 0:
            r[1] = pynumero_norm(np.multiply(self.slack_xu(), self.zu) - \
                                  self._miu, ord=np.inf)
        if self.vl.size > 0:
            r[2] = pynumero_norm(np.multiply(self.slack_sl(), self.vl) - \
                                  self._miu, ord=np.inf)
        if self.vu.size > 0:
            r[3] = pynumero_norm(np.multiply(self.slack_su(), self.vu) - \
                                  self._miu, ord=np.inf)
        return np.amax(r)

    def penalty_infeasibility(self):
        r = np.zeros(2)
        if self.yc.size > 0:
            r[0] = pynumero_norm(self.evaluate_c() - self.correction_c(),
                                 ord=np.inf)
        if self.yd.size > 0:
            r[0] = pynumero_norm(self.residual_d() - self.correction_d(),
                                 ord=np.inf)
        return np.amax(r)

    def correction_c(self):
        cd_inf = self.cd_infeasibility()
        return -cd_inf/self.rho * self.yc

    def correction_d(self):
        cd_inf = self.cd_infeasibility()
        return -cd_inf/self.rho * self.yd

    def grad_barrier_objective_x(self):
        df = self.grad_objective()
        xl_recip = np.reciprocal(self.slack_xl())
        xu_recip = np.reciprocal(self.slack_xu())
        return df + self._miu * (self.Pxu.dot(xu_recip) - self.Pxl.dot(xl_recip))

    def grad_barrier_objective_s(self):
        sl_recip = np.reciprocal(self.slack_sl())
        su_recip = np.reciprocal(self.slack_su())
        return self._miu * (self.Pdu.dot(su_recip) - self.Pdl.dot(sl_recip))

    def grad_barrier_penalty_objective_x(self):
        grad_box = self.grad_barrier_objective_x()
        cd_inf = self.cd_infeasibility()
        if cd_inf <= 1e-8:
            return grad_box
        grad_penalty = (self.jacobian_c().T * self.evaluate_c() + \
                        self.jacobian_d().T * self.residual_d) / cd_inf
        return grad_box + grad_penalty

    def grad_barrier_penalty_objective_s(self):
        grad_bos = self.grad_barrier_objective_s()
        cd_inf = self.cd_infeasibility()
        if cd_inf <= 1e-8:
            return grad_bos
        grad_penalty = self.residual_d() / cd_inf
        return grad_bos - grad_penalty


class PIPTrialNLPState(PIPNLPState):

    def __init__(self, nlp, miu, rho, **kwargs):
        super(PIPTrialNLPState, self).__init__(nlp, miu, rho, **kwargs)

        # free unnecessary caches
        self._jac_g = None
        self._jac_c = None
        self._jac_d = None
        self._hess_lag = None
        self._zl = None
        self._zu = None
        self._vl = None
        self._vu = None

    def _cache_nlp_functions(self):

        nlp = self._nlp
        x = self._x
        if self._need_recache_nlp_functions_x:
            self._f = nlp.objective(x)
            # evaluate gradient of the objective
            self._df = nlp.grad_objective(x, out=self._df)
            # evaluate residual constraints
            self._g = nlp.evaluate_g(x, out=self._g)
            # evaluate residual constraints
            self._res_c = nlp.evaluate_c(x, out=self._res_c, evaluated_g=self._g)
            # evaluate inequality constraints body
            self._body_d = nlp.evaluate_d(x, out=self._body_d, evaluated_g=self._g)

        self._need_recache_nlp_functions_x = False
        self._need_recache_nlp_functions_y = False

# for now have this here
class AugmentedSystem(object):

    def __init__(self, nlp_state):
        self._state = nlp_state
        nlp = nlp_state.nlp

        self._kkt = BlockSymMatrix(4)
        self._kkt[0, 0] = nlp_state.hessian_lag() + nlp_state.Dx_matrix()
        self._kkt[1, 1] = nlp_state.Ds_matrix()
        self._kkt[2, 0] = nlp_state.jacobian_c()
        self._kkt[3, 0] = nlp_state.jacobian_d()

        cd_inf = nlp_state.cd_infeasibility()
        if cd_inf >0:
            self._kkt[2, 2] = -cd_inf * identity(nlp.nc) / nlp_state.rho
            self._kkt[3, 3] = -cd_inf * identity(nlp.nd) / nlp_state.rho

        if nlp.nd == 0:
            if isinstance(nlp, CompositeNLP):
                d_vec = nlp.create_vector('d')
                mat = BlockSymMatrix(d_vec.nblocks)
                for j, v in enumerate(d_vec):
                    mat[j, j] = diagonal_matrix(v)
                self._kkt[3, 1] = mat

            else:
                self._kkt[3, 1] = empty_matrix(nlp.nd, nlp.nd)
        else:
            if isinstance(nlp, CompositeNLP):
                d_vec = nlp.create_vector('d')
                d_vec.fill(-1.0)
                mat = BlockSymMatrix(d_vec.nblocks)
                for j, v in enumerate(d_vec):
                    mat[j, j] = diagonal_matrix(v)
                self._kkt[3, 1] = mat
            else:
                self._kkt[3, 1] = - identity(nlp.nd)

        self._rhs = BlockVector([nlp_state.grad_lag_bar_x(),
                                 nlp_state.grad_lag_bar_s(),
                                 nlp_state.evaluate_c() - nlp_state.correction_c(),
                                 nlp_state.residual_d() - nlp_state.correction_d()])

    @property
    def matrix(self):
        return self._kkt

    @property
    def rhs(self):
        return self._rhs

    def update_system(self):

        nlp_state = self._state
        self._kkt[0, 0] = nlp_state.hessian_lag() + nlp_state.Dx_matrix()
        self._kkt[1, 1] = nlp_state.Ds_matrix()
        self._kkt[2, 0] = nlp_state.jacobian_c()
        self._kkt[3, 0] = nlp_state.jacobian_d()
        cd_inf = nlp_state.cd_infeasibility()
        if cd_inf >0:
            self._kkt[2, 2] = -cd_inf * identity(nlp.nc) / nlp_state.rho
            self._kkt[3, 3] = -cd_inf * identity(nlp.nd) / nlp_state.rho

        self._rhs[0] = nlp_state.grad_lag_bar_x()
        self._rhs[1] = nlp_state.grad_lag_bar_s()
        self._rhs[2] = nlp_state.evaluate_c() - nlp_state.correction_c()
        self._rhs[3] = nlp_state.residual_d() - nlp_state.correction_d()

class PenaltyInteriorPointWalker(object):

    def __init__(self, nlp_state, linear_solver, **kwargs):
        """

        Parameters
        ----------
        calculator: _InteriorPointCalculator
        """

        self._tau_min = kwargs.pop('tau_min', 0.99)
        self._max_backtrack = kwargs.pop('max_backtrack', 40)
        self._theta_min_factor = kwargs.pop('theta_min_fact', 1e-4)
        self._gamma_theta = kwargs.pop('gamma_theta', 1e-5)
        self._gamma_phi = kwargs.pop('gamma_phi', 1e-5)
        self._eta_phi = kwargs.pop('eta_phi', 1e-4)
        self._theta_max = 1e20
        self._s_theta = kwargs.pop('s_theta', 1.1)
        self._s_phi = kwargs.pop('s_phi', 2.3)
        self._kronecker = kwargs.pop('kronecker', 2.3)
        self._gamma_alpha = kwargs.pop('gamma_alpha', 0.05)

        # states
        self._state = nlp_state
        self._trial_state = PIPTrialNLPState(nlp_state.nlp,
                                             nlp_state.miu,
                                             nlp_state.rho)

        if linear_solver == 'ma27' or \
            linear_solver == 'mumps' or \
            linear_solver == 'ma57':
            self._lsolver = FullKKTSolver(linear_solver)
        else:
            assert isinstance(linear_solver, KKTSolver), \
                'Linear Solver must be subclass from KKTSolver'
            self._lsolver = linear_solver
            self._lsolver.reset_inertia_parameters()

        nlp = self._state.nlp

        self._linear_system = AugmentedSystem(self._state)

        # solve kkt system to get symbolic factorization
        dvars, info = self._lsolver.solve(self._linear_system.matrix,
                                          self._linear_system.rhs,
                                          nlp,
                                          do_symbolic=True)

        # ToDo: check if info did not give any errors

    def compute_step(self, max_iter_reg=40, max_iter_ref=10):

        # solve kkt system
        dvars, info = self._lsolver.solve(self._linear_system.matrix,
                                          self._linear_system.rhs,
                                          self._state.nlp,
                                          do_symbolic=False,  # done already in init
                                          max_iter_reg=max_iter_reg,
                                          max_iter_ref=max_iter_ref)

        val_reg = info['delta_reg']
        if info['status'] != 0:
            raise RuntimeError('Could not solve linear system')

        # update step vectors
        dx = -dvars[0]
        ds = -dvars[1]
        dyc = -dvars[2]
        dyd = -dvars[3]

        nlp_state = self._state
        ZlPxldx = np.multiply(nlp_state.zl, nlp_state.Pxl.T.dot(dx))
        dzl = np.divide(nlp_state.miu - ZlPxldx, \
                        nlp_state.slack_xl()) - nlp_state.zl

        ZuPxudx = np.multiply(nlp_state.zu, nlp_state.Pxu.T.dot(dx))
        dzu = np.divide((nlp_state.miu + ZuPxudx),
                         nlp_state.slack_xu()) - nlp_state.zu

        VlPdlds = np.multiply(nlp_state.vl, nlp_state.Pdl.T.dot(ds))
        dvl = np.divide((nlp_state.miu - VlPdlds), \
                         nlp_state.slack_sl()) - nlp_state.vl

        VuPduds = np.multiply(nlp_state.vu, nlp_state.Pdu.T.dot(ds))
        dvu = np.divide((nlp_state.miu + VuPduds), \
                         nlp_state.slack_su()) - nlp_state.vu

        return [dx, ds, dyc, dyd, dzl, dzu, dvl, dvu], val_reg

    def compute_step_size(self, dx, ds, dyc, dyd, dzl, dzu, dvl, dvu, wls=True):

        # constants
        max_backtrack = self._max_backtrack
        eta_phi = self._eta_phi
        gamma_theta = self._gamma_theta
        gamma_phi = self._gamma_phi
        kronecker = self._kronecker
        s_theta = self._s_theta
        s_phi = self._s_phi

        # current state
        nlp_state = self._state

        tau = max(self._tau_min, 1.0 - nlp_state.miu)
        alpha_primal = nlp_state.max_alpha_primal(dx, ds, tau)
        what_is_blocking = 'n'
        if wls:
            # update rho and miu
            raise NotImplementedError("TODO")
        else:
            num_backtracks = 1

        alpha_dual = nlp_state.max_alpha_dual(dzl, dzu, dvl, dvu, tau)
        return alpha_primal, alpha_dual, num_backtracks

    def update_linear_system(self):
        self._linear_system.update_system()

class PenaltyInteriorPointSolver(object):

    def __init__(self, nlp, **kwargs):

        tee = kwargs.pop('tee', True)
        self.__outer_max_iter = kwargs.pop('max_iter_outer', 1000)
        self.__inner_max_iter = kwargs.pop('max_iter_inner', 1000)
        self.__reg_max_iter = kwargs.pop('reg_max_iter', 40)
        self.__refine_max_iter = kwargs.pop('refine_max_iter', 0)  # need further testing
        self.__iter_limit = kwargs.pop('iter_limit', 100000)
        self.__wls = kwargs.pop('wls', True)
        self.__log_level = kwargs.pop('log_level', 0)
        self.__miu_init = kwargs.pop('miu_init', 0.1)
        self.__rho_init = kwargs.pop('rho_init', 100.0)
        self.__tee = kwargs.pop('tee', True)
        self.__kappa_miu = kwargs.pop('kappa_miu', 0.2)
        self.__kappa_eps = kwargs.pop('kappa_eps', 10.0)
        self.__theta_miu = kwargs.pop('theta_miu', 1.5)
        self.__tau_min = kwargs.pop('tau_min', 0.99)
        self.__epsilon_tol = kwargs.pop('epsilon_tol', 1e-8)
        self.__bound_push = kwargs.pop('bound_push', 1e-2)
        self.__disable_bound_push = kwargs.pop('disable_bound_push', False)

        bound_push = self.__bound_push
        disable_bound_push = self.__disable_bound_push
        self._state = PIPNLPState(nlp,
                                  self.__miu_init,
                                  self.__rho_init,
                                  bound_push=bound_push,
                                  disable_bound_push=disable_bound_push)

    @staticmethod
    def print_summary(iteration,
                      objective,
                      primal_inf,
                      dual_inf,
                      comp_inf,
                      miu,
                      norm_d,
                      val_reg,
                      alpha_dual,
                      alpha_primal,
                      num_ls):

        if val_reg == 0.0:
            formating = "{:>4d} {:>14.7e} {:>7.2e} {:>7.2e}  {:>7.2e}  {:.1f} {:>9.2e} {:>4} {:>10.2e} {:>8.2e} {:>3d}"
            regularization = "--"
        else:
            formating = "{:>4d} {:>14.7e} {:>7.2e} {:>7.2e}  {:>7.2e}  {:.1f} {:>9.2e}  {:>4.1f}  {:>7.2e} {:>7.2e} {:>3d}"
            regularization = pymath.log10(val_reg)

        line = formating.format(iteration,
                                objective,
                                primal_inf,
                                dual_inf,
                                comp_inf,
                                pymath.log10(miu),
                                norm_d,
                                regularization,
                                alpha_dual,
                                alpha_primal,
                                num_ls)
        print(line)

    @staticmethod
    def compute_new_miu(curr_miu, epsilon_tol, kappa_miu, theta_miu):
        return max(0.1 * epsilon_tol,min(kappa_miu * curr_miu, curr_miu ** theta_miu))

    def solve(self, **kwargs):

        tee = kwargs.pop('tee', self.__tee)
        outer_max_iter = kwargs.pop('max_iter_outer', self.__outer_max_iter)
        inner_max_iter = kwargs.pop('max_iter_inner', self.__outer_max_iter)
        reg_max_iter = kwargs.pop('reg_max_iter', self.__reg_max_iter)
        refine_max_iter = kwargs.pop('refine_max_iter', self.__refine_max_iter)
        iter_limit = kwargs.pop('max_iter', self.__iter_limit)
        wls = kwargs.pop('wls', self.__wls)
        log_level = kwargs.pop('log_level', self.__log_level)
        miu_init = kwargs.pop('miu_init', self.__miu_init)
        rho_init = kwargs.pop('rho_init', self.__rho_init)
        kappa_miu = kwargs.pop('kappa_miu', self.__kappa_miu)
        kappa_eps = kwargs.pop('kappa_eps', self.__kappa_eps)
        theta_miu = kwargs.pop('theta_miu', self.__theta_miu)
        tau_min = kwargs.pop('tau_min', self.__tau_min)
        epsilon_tol = kwargs.pop('epsilon_tol', self.__epsilon_tol)
        tiny_threshold = kwargs.pop('tiny_step_threshold', 5e-6)

        if pn.ma27_available:
            linear_solver = kwargs.pop('linear_solver', 'ma27')
        else:
            linear_solver = kwargs.pop('linear_solver', 'mumps')

        nlp_state = self._state
        if miu_init != self.__miu_init:
            nlp_state.miu = miu_init
            nlp_state.cache()

        if rho_init != self.__rho_init:
            nlp_state.rho = rho_init
            nlp_state.cache()

        # create walker object
        walker = PenaltyInteriorPointWalker(nlp_state,
                                            linear_solver,
                                            tau_min=tau_min)

        if tee:
            print_nlp_info(nlp_state.nlp, header='Penalty Interior-Point ', linear_solver=linear_solver)


        val_reg = 0.0
        counter_iter = 0
        counter_outer_iter = 0
        alpha_dual = 1.0
        alpha_primal = 1.0
        reached_limit = False
        n_ls = 0

        old_E_rho = 10 * optimality_error_penalty(nlp_state)
        for oi in range(outer_max_iter):

            counter_tiny_steps = 0
            for ii in range(inner_max_iter):

                # output to screen
                if counter_iter % 10 == 0 and tee:
                    print(
                        "iter    objective    inf_pr   inf_du   inf_cmp  lg(mu)  ||d||   lg(rg) alpha_du alpha_pr  ls")
                if tee:
                    self.print_summary(counter_iter,
                                       nlp_state.objective(),
                                       nlp_state.primal_infeasibility(),
                                       nlp_state.dual_infeasibility(),
                                       nlp_state.complementarity_infeasibility(),
                                       nlp_state.miu,
                                       nlp_state.norm_primal_step(),
                                       val_reg,
                                       alpha_dual,
                                       alpha_primal,
                                       n_ls)

                # compute step direction
                steps, val_reg = walker.compute_step(max_iter_reg=reg_max_iter,
                                                     max_iter_ref=refine_max_iter)

                # compute step size
                alpha_primal, alpha_dual, n_ls = walker.compute_step_size(*steps,
                                                                          wls=wls)

                # check for continuous tiny steps
                if alpha_primal < tiny_threshold:
                    counter_tiny_steps += 1
                else:
                    counter_tiny_steps = 0

                if counter_tiny_steps >=4:
                    counter_tiny_steps = 0
                    alpha_primal, alpha_dual, n_ls = walker.compute_step_size(*steps,
                                                                              wls=False)

                # update state
                nlp_state.update_state(x=nlp_state.x + steps[0] * alpha_primal,
                                       s=nlp_state.s + steps[1] * alpha_primal,
                                       yc=nlp_state.yc + steps[2] * alpha_primal,
                                       yd=nlp_state.yd + steps[3] * alpha_primal,
                                       zl=nlp_state.zl + steps[4] * alpha_dual,
                                       zu=nlp_state.zu + steps[5] * alpha_dual,
                                       vl=nlp_state.vl + steps[6] * alpha_dual,
                                       vu=nlp_state.vu + steps[7] * alpha_dual)
                # update kkt system
                walker.update_linear_system()

                # compute optimality error
                E_miu = optimality_error(nlp_state)
                E_rho = optimality_error_penalty(nlp_state)
                counter_iter += 1

                duals = BlockVector([nlp_state.yc, nlp_state.yd])
                if (E_rho < old_E_rho):
                    old_E_rho = E_rho
                    if pynumero_norm(duals) > 1e-2 * nlp_state.rho:
                        nlp_state.rho *= 10
                # print(nlp_state.rho)
                # ss = walker._linear_system.matrix.coo_data()
                # print(ss)
                if E_miu < kappa_eps * nlp_state.miu:
                    break

                if counter_iter >= iter_limit:
                    reached_limit = True
                    break

            if ii >= inner_max_iter-1:
                print("WARNING: Reached limit in number of inner iterations")

            if reached_limit:
                print("WARNING: Reached iteration limit")
                break

            # update barrier
            nlp_state.miu = self.compute_new_miu(nlp_state.miu,
                                                 epsilon_tol,
                                                 kappa_miu,
                                                 theta_miu)

            # heuristic to try skip inner loops
            for kk in range(4):
                sc, sd = nlp_state.scaling_factors_infeasibility()
                E_miu_j = nlp_state.complementarity_infeasibility()/sc
                if E_miu_j < kappa_eps * nlp_state.miu:
                    nlp_state.miu = self.compute_new_miu(nlp_state.miu,
                                                         epsilon_tol,
                                                         kappa_miu,
                                                         theta_miu)
                else:
                    break

            # check if it's done
            if nlp_state.miu <= epsilon_tol:
                sc, sd = nlp_state.scaling_factors_infeasibility()
                E0 = optimality_error(nlp_state, sc=sc, sd=sd)
                if E0 <= epsilon_tol:
                    break

        if tee:
            print_summary(counter_iter,
                          nlp_state.objective(),
                          nlp_state.primal_infeasibility(),
                          nlp_state.dual_infeasibility(),
                          nlp_state.complementarity_infeasibility())

        info = {}
        info['objective'] = nlp_state.objective()
        info['iterations'] = counter_iter
        info['x'] = nlp_state.x
        info['s'] = nlp_state.s
        info['g'] = nlp_state.evaluate_g()
        info['c'] = nlp_state.evaluate_c()
        info['d'] = nlp_state.evaluate_d()
        info['inf_pr'] = nlp_state.primal_infeasibility()
        info['inf_du'] = nlp_state.dual_infeasibility()
        info['opt_error'] = optimality_error(nlp_state)
        info['mult_c'] = nlp_state.yc
        info['mult_d'] = nlp_state.yd
        info['mult_zl'] = nlp_state.zl
        info['mult_vl'] = nlp_state.vl
        info['mult_zu'] = nlp_state.zu
        info['mult_vu'] = nlp_state.vu

        return nlp_state.x.copy(), info


import pyomo.environ as aml
from pyomo.contrib.pynumero.interfaces import PyomoNLP
if __name__ == "__main__":

    np.set_printoptions(linewidth=250, precision=3)

    m = aml.ConcreteModel()
    m._name = 'model1'
    m.x = aml.Var([1, 2, 3], initialize=4.0)
    m.c = aml.Constraint(expr=m.x[3] ** 2 + m.x[1] == 25)
    m.d = aml.Constraint(expr=m.x[2] ** 2 + m.x[1] <= 18.0)
    # m.d = aml.Constraint(expr=aml.inequality(-18, m.x[2] ** 2 + m.x[1],  28))
    m.o = aml.Objective(expr=m.x[1] ** 4 - 3 * m.x[1] * m.x[2] ** 3 + m.x[3] ** 2 - 8.0)
    m.x[1].setlb(0.0)
    m.x[2].setlb(0.0)

    # m.x.pprint()
    # solver = aml.SolverFactory('ipopt')
    # solver.options['mu_init'] = 1e-2
    # solver.solve(m, tee=True)
    # m.x.pprint()
    nlp = PyomoNLP(m)
    solver = PenaltyInteriorPointSolver(nlp)
    solver.solve(tee=True, wls=False, rho_init=1000, max_iter=50)
