#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from pyomo.contrib.pynumero.algorithms.solvers.print_utils import (print_nlp_info,
                                                                   print_summary)
from pyomo.contrib.pynumero.sparse import (BlockVector,
                                           BlockSymMatrix,
                                           diagonal_matrix)

try:
    from pyomo.contrib.pynumero.linalg.solvers import ma27_solver
    found_hsl = True
except ImportError as e:
    found_hsl = False

try:
    from pyomo.contrib.pynumero.linalg.solvers.mumps_solver import MUMPSSymLinearSolver
    found_mumps = True
except ImportError as e:
    found_mumps = False

from pyomo.contrib.pynumero.linalg.solvers.kkt_solver import FullKKTSolver

import math as pymath
import numpy as np
import logging
import json

from pyomo.contrib.pynumero.interfaces import PyomoNLP
from pyomo.contrib.pynumero.sparse import empty_matrix
import pyomo.contrib.pynumero as pn
from scipy.sparse import coo_matrix as scipy_coo_matrix
from scipy.sparse import identity
import sys

if not found_mumps and not found_hsl:
    raise ImportError('Need MA27 or MUMPS to run pynumero interior-point')

from pyomo.contrib.pynumero.linalg.solvers.kkt_solver import KKTSolver

def dict_matrix(matrix, with_offset=True):
    # Note: this is just for printing in the logger
    if with_offset:
        irows = np.copy(matrix.row)+1
        jcols = np.copy(matrix.col)+1
        data = np.copy(matrix.data)
        shape = (matrix.shape[0]+1, matrix.shape[1]+1)
        return scipy_coo_matrix((data, (irows, jcols)), shape=shape).todok()
    irows = np.copy(matrix.row)
    jcols = np.copy(matrix.col)
    data = np.copy(matrix.data)
    shape = matrix.shape
    return scipy_coo_matrix((data, (irows, jcols)), shape=shape).todok()


class _NlpData(object):

    def __init__(self, nlp, with_z=True, with_v=True):

        self.nlp = nlp

        # vector of primal variables
        self.x = nlp.x_init()
        # vector of slack variables
        self.s = nlp.evaluate_d(self.x)
        # vector of equality constraint multipliers
        self.yc = nlp.create_vector_y(subset='c')
        # vector of inequality constraint multipliers
        self.yd = nlp.create_vector_y(subset='d')

        self.zl = None
        self.zu = None
        if with_z:
            # vector for the multipliers of lower bounds on x
            self.zl = nlp.create_vector_x(subset='l')
            self.zl.fill(1.0)
            # vector for the multipliers of upper bounds on x
            self.zu = nlp.create_vector_x(subset='u')
            self.zu.fill(1.0)

        self.vl = None
        self.vu = None
        if with_v:
            # vector for the multipliers of lower bounds on s
            self.vl = nlp.create_vector_s(subset='l')
            self.vl.fill(1.0)
            # vector for the multipliers of upper bounds on s
            self.vu = nlp.create_vector_s(subset='u')
            self.vu.fill(1.0)

        # step vector of primal variables
        self.dx = nlp.create_vector_x()
        self.dx.fill(0.0)
        # step vector of slack variables
        self.ds = nlp.create_vector_y('d')
        self.ds.fill(0.0)
        # step vector of equality constraint multiplier variables
        self.dyc = nlp.create_vector_y('c')
        self.dyc.fill(0.0)
        # step vector of inequality constraint multiplier variables
        self.dyd = nlp.create_vector_y('d')
        self.dyd.fill(0.0)

        self.dzl = None
        self.dzu = None
        if with_z:
            # step vector of multipliers of lower bounds on x
            self.dzl = nlp.create_vector_x(subset='l')
            self.dzl.fill(0.0)
            # step vector of multipliers of upper bounds on x
            self.dzu = nlp.create_vector_x(subset='u')
            self.dzu.fill(0.0)

        self.dvl = None
        self.dvu = None
        if with_v:
            # step vector of multipliers of lower bounds on s
            self.dvl = nlp.create_vector_s(subset='l')
            self.dvl.fill(0.0)
            # step vector of multipliers of upper bounds on s
            self.dvu = nlp.create_vector_s(subset='u')
            self.dvu.fill(0.0)

        # vector of lower bounds on x
        self.condensed_xl = nlp.xl(condensed=True)
        # vector of upper bounds on x
        self.condensed_xu = nlp.xu(condensed=True)
        # vector of lower bounds on s
        self.condensed_dl = nlp.dl(condensed=True)
        # vector of upper bounds on s
        self.condensed_du = nlp.du(condensed=True)

        # initialize expansion matrices
        # expansion matrix from condensed xl to x
        self.Pxl = nlp.expansion_matrix_xl()
        # expansion matrix from condensed xu to x
        self.Pxu = nlp.expansion_matrix_xu()
        # expansion matrix from condensed dl to d
        self.Pdl = nlp.expansion_matrix_dl()
        # expansion matrix from condensed du to d
        self.Pdu = nlp.expansion_matrix_du()
        # expansion matrix from c to g
        self.Pc = nlp.expansion_matrix_c()
        # expansion matrix from c to g
        self.Pd = nlp.expansion_matrix_d()


class _InteriorPointData(_NlpData):

    def __init__(self, nlp, **kwargs):

        ip_params = kwargs.pop('options', None)

        super(_InteriorPointData, self).__init__(nlp)

        self.miu = 0.1
        self.kappa_miu = 0.2
        self.kappa_eps = 10.0
        self.theta_miu = 1.5
        self.tau_min = 0.99
        self.epsilon_tol = 1e-8
        self.bound_push = 1e-2
        if ip_params is not None:
            self.initialize_parameters(ip_params)

        # push to the interior
        self.x = self.push_variables_within_bounds(nlp.xl(),
                                                   self.x,
                                                   nlp.xu(),
                                                   self.bound_push)

        self.s = self.push_variables_within_bounds(nlp.dl(),
                                                   self.s,
                                                   nlp.du(),
                                                   self.bound_push)

        # ToDo: QP initialization of equality and inequality multipliers

    def initialize_parameters(self, options):
        for k, v in options.items():
            if not hasattr(self, k):
                raise RuntimeError("{} is not a parameter of InteriorSolver".format(k))
            else:
                self.__setattr__(k, v)

    @staticmethod
    def push_variables_within_bounds(vars_l, vars, vars_u, bound_push):

        # push primal variables within lower bounds
        xl = vars_l

        max_one_xl = pn.where(np.absolute(xl) <= 1.0, 1.0, xl)
        max_one_xl = pn.where(np.isinf(max_one_xl), 0.0, max_one_xl)

        lower_pushed_x = pn.where(vars <= xl,
                                  xl + bound_push * max_one_xl,
                                  vars)
        vars = lower_pushed_x

        # push primal variables within upper bound
        xu = vars_u
        max_one_xu = pn.where(np.absolute(xu) <= 1.0, 1.0, xu)
        max_one_xu = pn.where(np.isinf(max_one_xu), 0.0, max_one_xu)
        upper_pushed_x = pn.where(vars >= xu,
                                  xu - bound_push * max_one_xu,
                                  vars)
        vars = upper_pushed_x
        return vars

    # ToDo: add relax bounds method
    # ToDo: add QP initialization method for multipliers


class _InteriorPointCalculator(object):

    def __init__(self, data, **kwargs):

        # pointer to data
        self._data = data

        # objective function
        self._f = None
        # gradient of the objective
        self._df = None
        # general constraints
        self._g = None
        # residual equality constraints
        self._res_c = None
        # inequality constraints body
        self._body_d = None
        # jacobian general constraints
        self._jac_g = None
        # jacobian equality constraints
        self._jac_c = None
        # jacobian inequality constraints
        self._jac_d = None
        # hessian of the lagrangian
        self._hess_lag = None

        # slack vector to lower bound x
        self._slack_xl = None
        # slack vector to upper bound x
        self._slack_xu = None
        # slack vector to lower bound x
        self._slack_sl = None
        # slack vector to upper bound x
        self._slack_su = None

        # populate nlp caches
        self.cache()

    def _cache_nlp_functions(self):

        data = self._data
        nlp = data.nlp
        x = data.x
        self._f = nlp.objective(x)
        # evaluate gradient of the objective
        self._df = nlp.grad_objective(x, out=self._df)
        # evaluate residual constraints
        self._g = nlp.evaluate_g(x, out=self._g)
        # evaluate residual constraints
        self._res_c = nlp.evaluate_c(x, out=self._res_c, evaluated_g=self._g)
        # evaluate inequality constraints body
        self._body_d = nlp.evaluate_d(x, out=self._body_d, evaluated_g=self._g)
        # evaluate jacobian equality constraints
        self._jac_g = nlp.jacobian_g(x, out=self._jac_g)
        # evaluate jacobian equality constraints
        self._jac_c = nlp.jacobian_c(x, out=self._jac_c, evaluated_jac_g=self._jac_g)
        # evaluate jacobian inequality constraints
        self._jac_d = nlp.jacobian_d(x, out=self._jac_d, evaluated_jac_g=self._jac_g)
        # evaluate hessian of the lagrangian
        yd = data.Pd * data.yd
        yc = data.Pc * data.yc
        y = yd + yc
        self._hess_lag = nlp.hessian_lag(x, y, out=self._hess_lag, eval_f_c=False)

    def _cache_slack_vectors(self):
        safeguard = 1e-12
        data = self._data
        self._slack_xl = data.Pxl.transpose() * data.x - data.condensed_xl
        self._slack_xl = self._slack_xl.clip(min=safeguard)
        self._slack_xu = data.condensed_xu - data.Pxu.transpose() * data.x
        self._slack_xu = self._slack_xu.clip(min=safeguard)
        self._slack_sl = data.Pdl.transpose() * data.s - data.condensed_dl
        self._slack_sl = self._slack_sl.clip(min=safeguard)
        self._slack_su = data.condensed_du - data.Pdu.transpose() * data.s
        self._slack_su = self._slack_su.clip(min=safeguard)

    def cache(self):
        self._cache_nlp_functions()
        self._cache_slack_vectors()

    def objective(self):
        return self._f

    def grad_objective(self):
        return self._df

    def evaluate_g(self):
        return self._g

    def evaluate_c(self):
        return self._res_c

    def evaluate_d(self):
        return self._body_d

    def residual_d(self):
        return self._body_d - self._data.s

    def jacobian_g(self):
        return self._jac_g

    def jacobian_c(self):
        return self._jac_c

    def jacobian_d(self):
        return self._jac_d

    def hessian_lag(self):
        return self._hess_lag

    def slack_xl(self):
        return self._slack_xl

    def slack_xu(self):
        return self._slack_xu

    def slack_sl(self):
        return self._slack_sl

    def slack_su(self):
        return self._slack_su

    def grad_lag_x(self):
        data = self._data
        grad_x_lag = self.grad_objective() + \
                     self.jacobian_c().transpose() * data.yc + \
                     self.jacobian_d().transpose() * data.yd + \
                     data.Pxu * data.zu - \
                     data.Pxl * data.zl

        return grad_x_lag

    def grad_lag_s(self):
        data = self._data
        grad_s_lag = data.Pdu * data.vu - \
                     data.Pdl * data.vl - \
                     data.yd
        return grad_s_lag

    def grad_lag_bar_x(self):
        data = self._data
        grad_x_lag_bar = self.grad_objective() + \
                         self.jacobian_c().transpose() * data.yc + \
                         self.jacobian_d().transpose() * data.yd + \
                         data.Pxu * (data.miu / self.slack_xu()) - \
                         data.Pxl * (data.miu / self.slack_xl())
        return grad_x_lag_bar

    def grad_lag_bar_s(self):
        data = self._data
        grad_s_lag_bar = data.Pdu * (data.miu / self.slack_su()) - \
                         data.Pdl * (data.miu / self.slack_sl()) - \
                         data.yd
        return grad_s_lag_bar

    def barrier_term(self, miu):
        s_xl = self.slack_xl()
        s_sl = self.slack_sl()
        s_xu = self.slack_xu()
        s_su = self.slack_su()
        return miu * (np.sum(np.log(s_xl)) + np.sum(np.log(s_xu)) + np.sum(np.log(s_sl)) + np.sum(np.log(s_su)))

    def barrier_objective(self, miu):
        return self.objective() + self.barrier_term(miu)

    def Dx_matrix(self):

        """
        # ToDo: this diagonal matrices probably dont need to be computed ever. Use them for testing
        Zl = diagonal_matrix(self._zl)
        Zu = diagonal_matrix(self._zu)
        SLxl = diagonal_matrix(self._slack_xl())
        SLxu = diagonal_matrix(self._slack_xu())
        SLxl_inv = SLxl.inv()
        SLxu_inv = SLxu.inv()

        # compute diagonal addition matrices
        Dx1 = self._Pxl * SLxl_inv * Zl * self._Pxl.transpose() + \
             self._Pxu * SLxu_inv * Zu * self._Pxu.transpose()
        """
        data = self._data
        d_vec = data.Pxl * np.divide(data.zl, self.slack_xl()) + \
                data.Pxu * np.divide(data.zu, self.slack_xu())

        if not isinstance(d_vec, BlockVector):
            return diagonal_matrix(d_vec)
        dx_m = BlockSymMatrix(d_vec.nblocks)
        for j, v in enumerate(d_vec):
            dx_m[j, j] = diagonal_matrix(v)
        return dx_m

    def Ds_matrix(self):

        """
        # ToDo: this diagonal matrices probably dont need to be computed ever. Use them for testing
        Vl = diagonal_matrix(self._vl)
        Vu = diagonal_matrix(self._vu)
        SLdl = diagonal_matrix(self._slack_sl())
        SLdu = diagonal_matrix(self._slack_su())
        SLdl_inv = SLdl.inv()
        SLdu_inv = SLdu.inv()

        Ds1 = self._Pdl * SLdl_inv * Vl * self._Pdl.transpose() + \
             self._Pdu * SLdu_inv * Vu * self._Pdu.transpose()
        """
        data = self._data
        d_vec = data.Pdl * np.divide(data.vl, self.slack_sl()) + \
                data.Pdu * np.divide(data.vu, self.slack_su())

        if not isinstance(d_vec, BlockVector):
            return diagonal_matrix(d_vec)
        dx_m = BlockSymMatrix(d_vec.nblocks)
        for j, v in enumerate(d_vec):
            dx_m[j, j] = diagonal_matrix(v)
        return dx_m

    def new_miu(self):
        data = self._data
        return max(0.1 * data.epsilon_tol,
                   min(data.kappa_miu * data.miu, data.miu ** data.theta_miu))

    def optimality_error_scaling(self, smax=100.0):
        data = self._data
        suma = 0.0
        suma += np.absolute(data.zl).sum()
        suma += np.absolute(data.zu).sum()
        suma += np.absolute(data.vl).sum()
        suma += np.absolute(data.vu).sum()

        sc = suma
        n = data.zl.size + data.zu.size + data.vl.size + data.vu.size
        if n == 0:
            sc = 1.0
        else:
            sc = sc / n
            sc = max(smax, sc)/smax

        suma += np.absolute(data.yc).sum()
        suma += np.absolute(data.yd).sum()
        n = data.zl.size + data.zu.size + data.vl.size + data.vu.size + data.yc.size + data.yd.size
        sd = suma
        if n == 0:
            sd = 1.0
        else:
            sd = sd / n
            sd = max(smax, sd) / smax

        return sc, sd

    def optimality_error(self, miu, scaled=True):

        sc = 1.0
        sd = 1.0
        if scaled:
            sc, sd = self.optimality_error_scaling()

        r = np.zeros(3)
        r[0] = self.primal_infeasibility()
        r[1] = self.dual_infeasibility()/sd
        r[2] = self.complementarity_infeasibility(miu)/sc

        return np.max(r)

    def max_alpha_primal(self, delta_x, delta_s):

        data = self._data
        tau = max(data.tau_min, 1.0 - data.miu)
        alpha_l_x = 1.0
        alpha_u_x = 1.0
        alpha_l_s = 1.0
        alpha_u_s = 1.0

        if data.condensed_xl.size > 0:
            delta_xl = data.Pxl.transpose() * delta_x
            alphas = np.divide(delta_xl, -tau * self.slack_xl())
            alpha_l_x = alphas.max()
            if alpha_l_x > 0:
                alpha_l_x = min(1.0/alpha_l_x, 1.0)
            else:
                alpha_l_x = 1.0

        if data.condensed_xu.size > 0:
            delta_xu = data.Pxu.transpose() * delta_x
            alpha_u_x = np.divide(delta_xu, tau * self.slack_xu()).max()
            if alpha_u_x > 0:
                alpha_u_x = min(1.0/alpha_u_x, 1.0)
            else:
                alpha_u_x = 1.0

        if data.condensed_dl.size > 0:
            delta_sl = data.Pdl.transpose() * delta_s
            alpha_l_s = np.divide(delta_sl, -tau * self.slack_sl()).max()
            if alpha_l_s > 0:
                alpha_l_s = min(1.0/alpha_l_s, 1.0)
            else:
                alpha_l_s = 1.0

        if data.condensed_du.size > 0:
            delta_su = data.Pdu.transpose() * delta_s
            alpha_u_s = np.divide(delta_su, tau * self.slack_su()).max()
            if alpha_u_s > 0:
                alpha_u_s = min(1.0/alpha_u_s, 1.0)
            else:
                alpha_u_s = 1.0

        return min([alpha_l_x, alpha_u_x, alpha_l_s, alpha_u_s])

    def max_alpha_dual(self, delta_zl, delta_zu, delta_vl, delta_vu):

        data = self._data
        tau = max(data.tau_min, 1.0 - data.miu)
        alpha_l_z = 1.0
        alpha_u_z = 1.0
        alpha_l_v = 1.0
        alpha_u_v = 1.0

        if data.zl.size > 0:
            alphas = np.divide(delta_zl, -tau * data.zl)
            alpha_l_z = alphas.max()
            if alpha_l_z > 0:
                alpha_l_z = min(1.0/alpha_l_z, 1.0)
            else:
                alpha_l_z = 1.0

        if data.zu.size > 0:
            alphas = np.divide(delta_zu, tau * data.zu)
            alpha_u_z = alphas.max()
            if alpha_u_z > 0:
                alpha_u_z = min(1.0 / alpha_u_z, 1.0)
            else:
                alpha_u_z = 1.0


        if data.vl.size > 0:
            alphas = np.divide(delta_vl, -tau * data.vl)
            alpha_l_v = alphas.max()
            if alpha_l_v > 0:
                alpha_l_v = min(1.0 / alpha_l_v, 1.0)
            else:
                alpha_l_v = 1.0
        if data.vu.size > 0:
            alphas = np.divide(delta_vu, tau * data.vu)
            alpha_u_v = alphas.max()
            if alpha_u_v > 0:
                alpha_u_v = min(1.0 / alpha_u_v, 1.0)
            else:
                alpha_u_v = 1.0

        return min([alpha_l_z, alpha_u_z, alpha_l_v, alpha_u_v])

    def primal_infeasibility(self):
        norm_c = 0.0
        norm_d = 0.0
        data = self._data
        if data.yc.size > 0:
            norm_c = pn.linalg.norm(self.evaluate_c(), ord=np.inf)
        if data.yd.size > 0:
            d_val = self.evaluate_d()
            res_dl = data.condensed_dl - data.Pdl.T.dot(d_val)
            res_du = data.Pdu.T.dot(d_val) - data.condensed_du
            if res_dl.size > 0:
                res_dl = res_dl.clip(min=0.0)
                norm_d += pn.linalg.norm(res_dl, ord=np.inf)
            if res_du.size > 0:
                res_du = res_du.clip(min=0.0)
                norm_d += pn.linalg.norm(res_du, ord=np.inf)
            #norm_d = pn.linalg.norm(self.residual_d(), ord=np.inf)
        return max(norm_c, norm_d)

    def dual_infeasibility(self):
        norm_x = pn.linalg.norm(self.grad_lag_x(), ord=np.inf)
        norm_s = 0.0
        if self._data.s.size > 0:
            norm_s = pn.linalg.norm(self.grad_lag_s(), ord=np.inf)
        return max(norm_x, norm_s)

    def norm_primal_step(self):
        normx = pn.linalg.norm(self._data.dx, ord=np.inf)
        norms = 0.0
        if self._data.ds.size > 0:
            norms = pn.linalg.norm(self._data.ds, ord=np.inf)
        return max(normx, norms)

    def complementarity_infeasibility(self, miu):

        data = self._data
        r = np.zeros(4)
        if data.zl.size > 0:
            r[0] = pn.linalg.norm(np.multiply(self.slack_xl(), data.zl) - miu, ord=np.inf)
        if data.zu.size > 0:
            r[1] = pn.linalg.norm(np.multiply(self.slack_xu(), data.zu) - miu, ord=np.inf)
        if data.vl.size > 0:
            r[2] = pn.linalg.norm(np.multiply(self.slack_sl(), data.vl) - miu, ord=np.inf)
        if data.vu.size > 0:
            r[3] = pn.linalg.norm(np.multiply(self.slack_su(), data.vu) - miu, ord=np.inf)
        return np.amax(r)


class LineSearch(object):

    def __init__(self, nlp):
        self._data = _NlpData(nlp, with_z=False, with_v=False)
        self._calculator = _InteriorPointCalculator(self._data)
        self._filter = list()
        self._n_backtrack = 0
        self._theta_min = None
        self._theta_max = None

        # free unnecessary caches
        self._calculator._jac_g = None
        self._calculator._jac_c = None
        self._calculator._jac_d = None
        self._calculator._hess_lag = None


    @property
    def size_filter(self):
        return len(self._filter)

    @property
    def num_backtrack(self):
        return self._n_backtrack

    def evaluate_optimality(self, miu):
        return self._calculator.barrier_objective(miu)

    def evaluate_fesibility(self):
        return self._calculator.primal_infeasibility()

    def evaluate_grad_optimality_x(self, miu):
        calc = self._calculator
        df = self._calculator.grad_objective()
        xl_reciprocal = np.reciprocal(calc.slack_xl())
        xu_reciprocal = np.reciprocal(calc.slack_xu())
        dbarrier_x = self._data.Pxu * xu_reciprocal * miu
        dbarrier_x -= self._data.Pxl * xl_reciprocal * miu
        return df + dbarrier_x

    def evaluate_grad_optimality_s(self, miu):
        calc = self._calculator
        dbarrier_s = self._data.Pdu * np.reciprocal(calc.slack_su()) * miu
        dbarrier_s -= self._data.Pdl * np.reciprocal(calc.slack_sl()) * miu

        return dbarrier_s

    def clear_filter(self):
        self._filter = list()

    def add_to_filter(self,
                      optimality,
                      feasibility,
                      theta_min=1e-4,
                      theta_max=1e4):
        if self.size_filter == 0:
            self._theta_min = theta_min * max(1, feasibility)
            self._theta_max = theta_max * max(1, feasibility)
        self._filter.append((optimality, feasibility))

    def _cache_calculator(self):

        data = self._data
        calc = self._calculator
        nlp = data.nlp
        x = data.x
        calc._f = nlp.objective(x)
        calc._df = nlp.grad_objective(x, out=calc._df)
        calc._res_c = nlp.evaluate_c(x, out=calc._res_c)
        calc._body_d = nlp.evaluate_d(x, out=calc._body_d)
        calc._cache_slack_vectors()

    def search(self, x, s, dx, ds, miu, **kwargs):

        max_iter = kwargs.pop('max_backtrack', 40)
        rho = kwargs.pop('rho', 0.5)
        max_alpha = kwargs.pop('max_alpha', 1.0)
        s_phi = kwargs.pop('s_phi', 2.3)
        s_theta = kwargs.pop('s_theta', 1.1)
        kronecker = kwargs.pop('kronecker', 1.0)
        eta_phi = kwargs.pop('eta_phi', 1e-4)
        gamma_theta = kwargs.pop('gamma_theta', 1e-5)
        gamma_phi = kwargs.pop('gamma_phi', 1e-5)

        if self.size_filter == 0:
            raise RuntimeError('Filter must be initialized with at least one pair before calling search')

        alpha = max_alpha
        self._n_backtrack = 0

        # check for very small search directions
        # dxs = np.concatenate([dx.flatten(), ds.flatten()])
        # xs = np.concatenate([x.flatten(), s.flatten()])
        # dx_check = np.absolute(dxs)/(1+np.absolute(xs))
        dx_check = np.absolute(dx)/(1+np.absolute(x))
        if pn.linalg.norm(dx_check, ord=np.inf) < 10 * np.finfo(float).eps:
            return alpha

        self._data.x = x.copy()
        self._data.s = s.copy()
        self._data.dx = dx.copy()
        self._data.ds = ds.copy()

        # Cache the calculator
        self._cache_calculator()

        opt = self.evaluate_optimality(miu)
        feas = self.evaluate_fesibility()
        grad_opt_x = self.evaluate_grad_optimality_x(miu)
        mk = grad_opt_x.dot(dx)

        for i in range(max_iter):
            # trials
            self._data.x = x + alpha * dx
            self._data.s = s + alpha * ds

            # Cache calculator
            self._cache_calculator()

            opt_p = self.evaluate_optimality(miu)
            feas_p = self.evaluate_fesibility()

            # check if is larger than max infeasibility
            if feas_p >= self._theta_max:
                self._n_backtrack += 1
                alpha *= rho
                continue

            # check if trial is in the filter
            if feas > self._theta_min:
                for pair in self._filter:
                    if opt_p >= pair[0] and feas_p >= pair[1]:
                        self._n_backtrack += 1
                        alpha *= rho
                        continue

            # switching condition
            if feas <= self._theta_min:

                if mk < 0.0:
                    lhs = alpha * (-mk) ** s_phi
                    rhs = kronecker * feas ** s_theta
                    if lhs > rhs:

                        armijo_rhs = opt + eta_phi * alpha * mk
                        if opt_p <= armijo_rhs:
                            self._n_backtrack += 1
                            return alpha
                        if abs(armijo_rhs-armijo_rhs) <= 1e-4:
                            self._n_backtrack += 1
                            return alpha
                    else:
                        # check SDC or SDO
                        if feas_p <= (1 - gamma_theta) * feas or opt_p <= opt - gamma_phi * feas:
                            self.add_to_filter(opt_p, feas_p)
                            self._n_backtrack += 1
                            return alpha
                else:
                    # check SDC or SDO
                    if feas_p <= (1 - gamma_theta) * feas or opt_p <= opt - gamma_phi * feas:
                        self.add_to_filter(opt_p, feas_p)
                        self._n_backtrack += 1
                        return alpha
            else:
                # check SDC or SDO
                if feas_p <= (1 - gamma_theta) * feas or opt_p <= opt - gamma_phi * feas:
                    self.add_to_filter(opt_p, feas_p)
                    self._n_backtrack += 1
                    return alpha
            self._n_backtrack += 1
            alpha *= rho

        raise RuntimeError('Need restoration')

    def unconstrained_search(self, x, dx, max_alpha=1.0, **kwargs):

        eta1 = kwargs.pop('eta1', 1.0e-4)
        eta2 = kwargs.pop('eta2', 0.9)
        rho = kwargs.pop('rho', 0.5)
        max_iter = 40
        second_condition = kwargs.pop('cond', 0)
        alpha = max_alpha
        self._n_backtrack = 0
        opt = self._data.nlp.objective(x)
        grad = self._data.nlp.grad_objective(x)
        prod = grad.dot(dx)

        # check for very small search directions
        if pn.linalg.norm(np.absolute(dx) / (1 + np.absolute(x)), ord=np.inf) < 10 * np.finfo(float).eps:
            return alpha

        # only armijo
        if second_condition == 0:
            for i in range(max_iter):
                x_trial = x + alpha * dx
                opt_p = self._data.nlp.objective(x_trial)
                if opt_p <= opt + eta1 * alpha * grad.dot(dx):
                    return alpha
                alpha *= rho
                self._n_backtrack += 1

        # armijo and wolfe
        elif second_condition == 1:
            for i in range(max_iter):
                x_trial = x + alpha * dx
                opt_p = self._data.nlp.objective(x_trial)
                grad_p = self._data.nlp.grad_objective(x_trial)
                if opt_p <= opt + eta1 * alpha * prod and \
                                grad_p.dot(dx) >= eta2 * prod:
                    return alpha
                alpha *= rho
                self._n_backtrack += 1

        # armijo and strong wolfe
        elif second_condition == 2:
            for i in range(max_iter):
                x_trial = x + alpha * dx
                opt_p = self._data.nlp.objective(x_trial)
                grad_p = self._data.nlp.grad_objective(x_trial)
                if opt_p <= opt + eta1 * alpha * prod and \
                                abs(grad_p.dot(dx)) >= eta2 * abs(prod):
                    return alpha
                alpha *= rho
                self._n_backtrack += 1
        # armijo and goldstein
        elif second_condition == 3:
            for i in range(max_iter):
                x_trial = x + alpha * dx
                opt_p = self._data.nlp.objective(x_trial)
                if opt + (1-eta1)*alpha*prod <= opt_p <= opt + eta1 * alpha * prod:
                    return alpha
                alpha *= rho
                self._n_backtrack += 1
        else:
            raise NotImplementedError('line search conditions not implemented')
        raise RuntimeError('Line search failed')


class _InteriorPointWalker(object):

    def __init__(self, calculator, linear_solver):
        """

        Parameters
        ----------
        calculator: _InteriorPointCalculator
        """
        self._calculator = calculator
        self._kkt = None
        self._rhs = None
        if linear_solver == 'ma27' or linear_solver == 'mumps':
            self._lsolver = FullKKTSolver(linear_solver)
        else:
            assert isinstance(linear_solver, KKTSolver), 'Linear Solver must be subclass from KKTSolver'
            self._lsolver = linear_solver
            self._lsolver.reset_inertia_parameters()

        nlp = self._calculator._data.nlp
        self._line_search = LineSearch(nlp)
        self.setup()


    def setup(self):

        calc = self._calculator
        nlp = self._calculator._data.nlp

        calc.cache()
        # create KKT system
        self._kkt = BlockSymMatrix(4)
        self._kkt[0, 0] = calc.hessian_lag() + calc.Dx_matrix()
        self._kkt[1, 1] = calc.Ds_matrix()
        self._kkt[2, 0] = calc.jacobian_c()
        self._kkt[3, 0] = calc.jacobian_d()
        if nlp.nd == 0:
            self._kkt[3, 1] = empty_matrix(nlp.nd, nlp.nd)
        else:
            self._kkt[3, 1] = - identity(nlp.nd)

        self._rhs = BlockVector([calc.grad_lag_bar_x(),
                                 calc.grad_lag_bar_s(),
                                 calc.evaluate_c(),
                                 calc.residual_d()])

        # solve kkt system to get symbolic factorization
        dvars, info = self._lsolver.solve(self._kkt, self._rhs, nlp, do_symbolic=True)

    def cache(self):

        calc = self._calculator
        # update kkt
        self._kkt[0, 0] = calc.hessian_lag() + calc.Dx_matrix()
        self._kkt[1, 1] = calc.Ds_matrix()
        self._kkt[2, 0] = calc.jacobian_c()
        self._kkt[3, 0] = calc.jacobian_d()

        # update rhs
        self._rhs[0] = calc.grad_lag_bar_x()
        self._rhs[1] = calc.grad_lag_bar_s()
        self._rhs[2] = calc.evaluate_c()
        self._rhs[3] = calc.residual_d()

    def compute_step(self, max_iter_reg=40):

        calc = self._calculator
        data = calc._data
        nlp = data.nlp

        # solve kkt system
        dvars, info = self._lsolver.solve(self._kkt,
                                          self._rhs,
                                          nlp,
                                          do_symbolic=False,  # done already in setup
                                          max_iter_reg=max_iter_reg)

        val_reg = info['delta_reg']
        if info['status'] != 0:
            raise RuntimeError('Could not solve linear system')

        # update step vectors
        dx = -dvars[0]
        ds = -dvars[1]
        dyc = -dvars[2]
        dyd = -dvars[3]

        ZlPxldx = np.multiply(data.zl, data.Pxl.transpose() * dx)

        dzl = np.divide(data.miu - ZlPxldx, calc.slack_xl()) - data.zl
        ZuPxudx = np.multiply(data.zu, data.Pxu.transpose() * dx)
        dzu = np.divide((data.miu + ZuPxudx), calc.slack_xu()) - data.zu
        VlPdlds = np.multiply(data.vl, data.Pdl.transpose() * ds)
        dvl = np.divide((data.miu - VlPdlds), calc.slack_sl()) - data.vl
        VuPduds = np.multiply(data.vu, data.Pdu.transpose() * ds)
        dvu = np.divide((data.miu + VuPduds), calc.slack_su()) - data.vu

        return [dx, ds, dyc, dyd, dzl, dzu, dvl, dvu], val_reg

    def compute_step_size(self, steps, wls=True):

        dx = steps[0]
        ds = steps[1]
        dyc = steps[2]
        dyd = steps[3]
        dzl = steps[4]
        dzu = steps[5]
        dvl = steps[6]
        dvu = steps[7]

        calc = self._calculator
        data = self._calculator._data
        if ds.size + dyc.size + dyd.size + dzl.size + dzu.size == 0:
            alpha_primal = 1.0
            if wls:
                alpha_primal = self._line_search.unconstrained_search(data.x, dx)
            return alpha_primal, 1.0
        else:
            alpha_primal = calc.max_alpha_primal(dx, ds)
            if wls:
                alpha_primal = self._line_search.search(data.x,
                                                        data.s,
                                                        dx,
                                                        ds,
                                                        data.miu,
                                                        max_alpha=alpha_primal)

            alpha_dual = calc.max_alpha_dual(dzl, dzu, dvl, dvu)

        return alpha_primal, alpha_dual

    def initialize_filter(self, optimality, feasibility):
        self._line_search.add_to_filter(optimality, feasibility)

    def clear_filter(self):
        self._line_search.clear_filter()


class InteriorPointSolver(object):

    def __init__(self, nlp, options=None):

        self._nlp = nlp
        self._data = _InteriorPointData(nlp, options=options)
        self._calculator = _InteriorPointCalculator(self._data)
        # logger
        self.logger = self._setup_logger()

    @staticmethod
    def _setup_logger():
        logger = dict()
        logger['iterations'] = dict()
        logger['options'] = dict()
        return logger

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

    def _log(self, iteration, level=1):

        data = self._data
        calc = self._calculator
        self.logger['iterations'][iteration] = dict()
        self.logger['iterations'][iteration]['summary'] = dict()
        self.logger['iterations'][iteration]['summary']['objective'] = calc.objective()
        self.logger['iterations'][iteration]['summary']['inf_pr'] = calc.primal_infeasibility()
        self.logger['iterations'][iteration]['summary']['inf_du'] = calc.dual_infeasibility()
        self.logger['iterations'][iteration]['summary']['||d||'] = calc.norm_primal_step()
        self.logger['iterations'][iteration]['summary']['lg(rg)'] = 0.0
        self.logger['iterations'][iteration]['summary']['lg(mu)'] = pymath.log10(data.miu)
        self.logger['iterations'][iteration]['summary']['alpha_du'] = 1.0
        self.logger['iterations'][iteration]['summary']['alpha_pr'] = 1.0
        self.logger['iterations'][iteration]['summary']['ls'] = 0
        E_miu = calc.optimality_error(data.miu)
        self.logger['iterations'][iteration]['summary']['optimality_error'] = E_miu
        self.logger['iterations'][iteration]['summary']['miu'] = data.miu
        tau = max(data.tau_min, 1.0 - data.miu)
        self.logger['iterations'][iteration]['summary']['tau'] = tau


        if level > 1:
            self.logger['iterations'][iteration]['variables'] = dict()
            self.logger['iterations'][iteration]['variables']['||x||_inf'] = pn.linalg.norm(data.x,
                                                                                            ord=np.inf)
            if len(data.s) > 0:
                self.logger['iterations'][iteration]['variables']['||s||_inf'] = pn.linalg.norm(data.s,
                                                                                                ord=np.inf)
            else:
                self.logger['iterations'][iteration]['variables']['||s||_inf'] = 0.0
            if len(data.yc) > 0:
                self.logger['iterations'][iteration]['variables']['||yc||_inf'] = pn.linalg.norm(data.yc,
                                                                                                 ord=np.inf)
            else:
                self.logger['iterations'][iteration]['variables']['||yc||_inf'] = 0.0
            if len(data.yd) > 0:
                self.logger['iterations'][iteration]['variables']['||yd||_inf'] = pn.linalg.norm(data.yd,
                                                                                                 ord=np.inf)
            else:
                self.logger['iterations'][iteration]['variables']['||yd||_inf'] = 0.0
            if len(data.zl) > 0:
                self.logger['iterations'][iteration]['variables']['||zl||_inf'] = pn.linalg.norm(data.zl,
                                                                                                 ord=np.inf)
            else:
                self.logger['iterations'][iteration]['variables']['||zl||_inf'] = 0.0
            if len(data.zu) > 0:
                self.logger['iterations'][iteration]['variables']['||zu||_inf'] = pn.linalg.norm(data.zu,
                                                                                                 ord=np.inf)
            else:
                self.logger['iterations'][iteration]['variables']['||zu||_inf'] = 0.0
            if len(data.vl) > 0:
                self.logger['iterations'][iteration]['variables']['||vl||_inf'] = pn.linalg.norm(data.vl,
                                                                                                 ord=np.inf)
            else:
                self.logger['iterations'][iteration]['variables']['||vl||_inf'] = 0.0
            if len(data.vu) > 0:
                self.logger['iterations'][iteration]['variables']['||vu||_inf'] = pn.linalg.norm(data.vu,
                                                                                                 ord=np.inf)
            else:
                self.logger['iterations'][iteration]['variables']['||vu||_inf'] = 0.0

        if level > 2:
            self.logger['iterations'][iteration]['variables']['x'] = dict(enumerate(data.x))
            self.logger['iterations'][iteration]['variables']['s'] = dict(enumerate(data.s))
            self.logger['iterations'][iteration]['variables']['yc'] = dict(enumerate(data.yc))
            self.logger['iterations'][iteration]['variables']['yd'] = dict(enumerate(data.yd))
            self.logger['iterations'][iteration]['variables']['zl'] = dict(enumerate(data.zl))
            self.logger['iterations'][iteration]['variables']['zu'] = dict(enumerate(data.zu))
            self.logger['iterations'][iteration]['variables']['vl'] = dict(enumerate(data.vl))
            self.logger['iterations'][iteration]['variables']['vu'] = dict(enumerate(data.vu))

        if level > 3:
            self.logger['iterations'][iteration]['gradients'] = dict()
            self.logger['iterations'][iteration]['gradients']['grad_f'] = dict(enumerate(calc.grad_objective()))
            self.logger['iterations'][iteration]['gradients']['grad_lag_x'] = dict(enumerate(calc.grad_lag_x()))
            self.logger['iterations'][iteration]['gradients']['grad_lag_s'] = dict(enumerate(calc.grad_lag_s()))
            self.logger['iterations'][iteration]['gradients']['grad_lag_yc'] = dict(enumerate(calc.evaluate_c()))
            self.logger['iterations'][iteration]['gradients']['grad_lag_yd'] = dict(enumerate(calc.residual_d()))
            self.logger['iterations'][iteration]['gradients']['grad_lag_bar_x'] = dict(enumerate(calc.grad_lag_bar_x()))
            self.logger['iterations'][iteration]['gradients']['grad_lag_bar_s'] = dict(enumerate(calc.grad_lag_bar_s()))

        if level > 4:
            self.logger['iterations'][iteration]['slack_vectors'] = dict()
            self.logger['iterations'][iteration]['slack_vectors']['slack_xl'] = dict(enumerate(calc.slack_xl()))
            self.logger['iterations'][iteration]['slack_vectors']['slack_xu'] = dict(enumerate(calc.slack_xu()))
            self.logger['iterations'][iteration]['slack_vectors']['slack_sl'] = dict(enumerate(calc.slack_sl()))
            self.logger['iterations'][iteration]['slack_vectors']['slack_su'] = dict(enumerate(calc.slack_su()))

        if level > 5:
            self.logger['iterations'][iteration]['steps'] = dict()

            if len(data.ds) > 0:
                self.logger['iterations'][iteration]['steps']['||ds||_inf'] = pn.linalg.norm(data.ds,
                                                                                             ord=np.inf)
            else:
                self.logger['iterations'][iteration]['steps']['||ds||_inf'] = 0.0
            if len(data.dyc) > 0:
                self.logger['iterations'][iteration]['steps']['||dyc||_inf'] = pn.linalg.norm(data.dyc,
                                                                                              ord=np.inf)
            else:
                self.logger['iterations'][iteration]['steps']['||dyc||_inf'] = 0.0
            if len(data.dyd) > 0:
                self.logger['iterations'][iteration]['steps']['||dyd||_inf'] = pn.linalg.norm(data.dyd,
                                                                                              ord=np.inf)
            else:
                self.logger['iterations'][iteration]['steps']['||dyd||_inf'] = 0.0
            if len(data.dzl) > 0:
                self.logger['iterations'][iteration]['steps']['||dzl||_inf'] = pn.linalg.norm(data.dzl,
                                                                                              ord=np.inf)
            else:
                self.logger['iterations'][iteration]['steps']['||dzl||_inf'] = 0.0
            if len(data.dzu) > 0:
                self.logger['iterations'][iteration]['steps']['||dzu||_inf'] = pn.linalg.norm(data.dzu,
                                                                                              ord=np.inf)
            else:
                self.logger['iterations'][iteration]['steps']['||dzu||_inf'] = 0.0
            if len(data.dvl) > 0:
                self.logger['iterations'][iteration]['steps']['||dvl||_inf'] = pn.linalg.norm(data.dvl,
                                                                                              ord=np.inf)
            else:
                self.logger['iterations'][iteration]['steps']['||dvl||_inf'] = 0.0
            if len(data.dvu) > 0:
                self.logger['iterations'][iteration]['steps']['||dvu||_inf'] = pn.linalg.norm(data.dvu,
                                                                                              ord=np.inf)
            else:
                self.logger['iterations'][iteration]['steps']['||dvu||_inf'] = 0.0

        if level > 6:
            self.logger['iterations'][iteration]['steps']['dx'] = dict(enumerate(data.dx))
            self.logger['iterations'][iteration]['steps']['ds'] = dict(enumerate(data.ds))
            self.logger['iterations'][iteration]['steps']['dyc'] = dict(enumerate(data.dyc))
            self.logger['iterations'][iteration]['steps']['dyd'] = dict(enumerate(data.dyd))
            self.logger['iterations'][iteration]['steps']['dzl'] = dict(enumerate(data.dzl))
            self.logger['iterations'][iteration]['steps']['dzu'] = dict(enumerate(data.dzu))
            self.logger['iterations'][iteration]['steps']['dvl'] = dict(enumerate(data.dvl))
            self.logger['iterations'][iteration]['steps']['dvu'] = dict(enumerate(data.dvu))

        if level > 7:
            self.logger['iterations'][iteration]['matrices'] = dict()
            hess = dict_matrix(calc.hessian_lag())
            hess = {str(k):v for k, v in hess.items()}
            self.logger['iterations'][iteration]['matrices']['hessian_lag'] = hess
            jac_c = dict_matrix(calc.jacobian_c())
            jac_c = {str(k): v for k, v in jac_c.items()}
            self.logger['iterations'][iteration]['matrices']['jac_c'] = jac_c
            jac_d = dict_matrix(calc.jacobian_d())
            jac_d = {str(k): v for k, v in jac_d.items()}
            self.logger['iterations'][iteration]['matrices']['jac_d'] = jac_d
            dic_m = dict_matrix(calc.Dx_matrix())
            dic_m = {str(k): v for k, v in dic_m.items()}
            self.logger['iterations'][iteration]['matrices']['Dx'] = dic_m
            dic_m = dict_matrix(calc.Ds_matrix())
            dic_m = {str(k): v for k, v in dic_m.items()}
            self.logger['iterations'][iteration]['matrices']['Ds'] = dic_m

    def _log_linear_system(self, iteration, kkt, rhs, sol):

        self.logger['iterations'][iteration]['linear_system'] = dict()
        self.logger['iterations'][iteration]['linear_system']['rhs'] = dict()
        self.logger['iterations'][iteration]['linear_system']['sol'] = dict()
        self.logger['iterations'][iteration]['linear_system']['kkt'] = dict()

        for bid, v in enumerate(rhs):
            self.logger['iterations'][iteration]['linear_system']['rhs'][str(bid)] = dict(enumerate(v))

        for m in range(kkt.bshape[0]):
            for n in range(kkt.bshape[1]):
                if m >= n and kkt[m, n] is not None:
                    _str = "{},{}".format(m, n)
                    mat = dict_matrix(kkt[m, n])
                    mat = {str(k): v for k, v in mat.items()}
                    self.logger['iterations'][iteration]['linear_system']['kkt'][_str] = mat

        for bid, v in enumerate(sol):
            self.logger['iterations'][iteration]['linear_system']['sol'][str(bid)] = dict(enumerate(v))

    def solve(self, **kwargs):

        tee = kwargs.pop('tee', True)
        outer_max_iter = kwargs.pop('max_iter_outer', 1000)
        inner_max_iter = kwargs.pop('max_iter_inner', 1000)
        reg_max_iter = kwargs.pop('reg_max_iter', 40)
        iter_limit = kwargs.pop('iter_limit', 100000)
        wls = kwargs.pop('wls', True)
        log_level = kwargs.pop('log_level', 0)

        store_walker = kwargs.pop('store_walker', None)

        if found_hsl:
            linear_solver = kwargs.pop('linear_solver', 'ma27')
        else:
            linear_solver = kwargs.pop('linear_solver', 'mumps')

        # helper objects
        data = self._data
        calc = self._calculator
        nlp = self._nlp

        walker = _InteriorPointWalker(self._calculator,
                                      linear_solver=linear_solver)

        if tee:
            print_nlp_info(nlp, linearsolver=linear_solver)

        val_reg = 0.0
        counter_iter = 0
        counter_outer_iter = 0
        alpha_dual = 1.0
        alpha_primal = 1.0
        n_ls = 0

        # first iteration output
        if tee:
            print("iter    objective    inf_pr   inf_du   inf_cmp  lg(mu)  ||d||   lg(rg) alpha_du alpha_pr  ls")
            self.print_summary(counter_iter,
                               calc.objective(),
                               calc.primal_infeasibility(),
                               calc.dual_infeasibility(),
                               calc.complementarity_infeasibility(data.miu),
                               data.miu,
                               calc.norm_primal_step(),
                               val_reg,
                               alpha_dual,
                               alpha_primal,
                               n_ls)
        reached_limit = False
        E_0 = calc.optimality_error(0.0)
        while E_0 > data.epsilon_tol and counter_outer_iter < outer_max_iter:

            # compute optimality error
            E_miu = calc.optimality_error(data.miu)

            # initialize filter
            walker.initialize_filter(calc.barrier_objective(data.miu),
                                     calc.primal_infeasibility())

            counter_inner_iter = 0
            while E_miu > data.kappa_eps * data.miu and counter_inner_iter < inner_max_iter:

                # logger for debug
                if log_level > 0:
                    self._log(counter_iter, level=10)

                # compute step direction
                steps, val_reg = walker.compute_step(max_iter_reg=reg_max_iter)

                # compute step size (line search)
                alpha_primal, alpha_dual = walker.compute_step_size(steps, wls=wls)

                n_ls = walker._line_search.num_backtrack

                # update vectors
                for kk, dname in enumerate(['x', 's', 'yc', 'yd', 'zl', 'zu', 'vl', 'vu']):
                    # update variable vectors
                    data_d = getattr(data, dname)
                    if kk < 4:
                        data_d += steps[kk] * alpha_primal
                    else:
                        data_d += steps[kk] * alpha_dual
                    # update step vectors
                    setattr(data, 'd{}'.format(dname), steps[kk])

                # evaluate nlp functions at new point
                calc.cache()
                walker.cache()

                # compute optimality error
                E_miu = calc.optimality_error(data.miu)
                counter_iter += 1
                counter_inner_iter += 1

                # output to screen
                if counter_iter % 10 == 0 and tee:
                    print(
                        "iter    objective    inf_pr   inf_du   inf_cmp  lg(mu)  ||d||   lg(rg) alpha_du alpha_pr  ls")
                if tee:
                    self.print_summary(counter_iter,
                                       calc.objective(),
                                       calc.primal_infeasibility(),
                                       calc.dual_infeasibility(),
                                       calc.complementarity_infeasibility(data.miu),
                                       data.miu,
                                       calc.norm_primal_step(),
                                       val_reg,
                                       alpha_dual,
                                       alpha_primal,
                                       n_ls)

                if counter_iter >= iter_limit:
                    print('REACHED ITERATION LIMIT')
                    reached_limit = True
                    break


            # clear filter
            walker.clear_filter()

            # update barrier
            data.miu = calc.new_miu()
            for kk in range(4):
                sc, sd = calc.optimality_error_scaling()
                E_miu_j = calc.complementarity_infeasibility(data.miu)/sc
                if E_miu_j < data.kappa_eps * data.miu:
                    data.miu = calc.new_miu()
                else:
                    break

            E_0 = calc.optimality_error(0.0)
            counter_outer_iter += 1

            if reached_limit:
                break

        if tee:
            print_summary(counter_iter,
                          calc.objective(),
                          calc.primal_infeasibility(),
                          calc.dual_infeasibility(),
                          calc.complementarity_infeasibility(data.miu))

        if log_level>0:
            with open('data.txt', 'w') as outfile:
                json.dump(self.logger, outfile, sort_keys=True)

        info = {}
        info['objective'] = calc.objective()
        info['iterations'] = counter_iter
        info['x'] = data.x
        info['s'] = data.s
        info['g'] = calc.evaluate_g()
        info['c'] = calc.evaluate_c()
        info['d'] = calc.evaluate_d()
        info['inf_pr'] = calc.primal_infeasibility()
        info['inf_du'] = calc.dual_infeasibility()
        info['opt_error'] = calc.optimality_error(0.0)
        info['mult_c'] = data.yc
        info['mult_d'] = data.yd
        info['mult_zl'] = data.zl
        info['mult_vl'] = data.vl
        info['mult_zu'] = data.zu
        info['mult_vu'] = data.vu

        if store_walker is not None:
            assert isinstance(store_walker, list)
            store_walker.append(walker)

        return data.x.copy(), info


from pyomo.contrib.pynumero.algorithms.solvers.tests.cute_models import *
if __name__ == "__main__":

    model = create_model1()

    solver = aml.SolverFactory('ipopt')
    solver.options['nlp_scaling_method'] = 'none'
    solver.options['linear_system_scaling'] = 'none'
    #solver.solve(model, tee=True)
    #model.x.pprint()

    nlp = PyomoNLP(model)
    #print(nlp.x_init())
    #print(nlp.variable_order())
    opt = InteriorPointSolver(nlp)
    opt.solve(max_iter_outer=1000, max_iter_inner=1000, wls=True, tee=True)