from pyomo.contrib.pynumero.algorithms.print_utils import (print_nlp_info,
                                                           print_summary)
from pyomo.contrib.pynumero.sparse import (BlockVector,
                                           BlockSymMatrix,
                                           DiagonalMatrix,
                                           IdentityMatrix)

from pyomo.contrib.pynumero.algorithms import (InertiaCorrectionParams,
                                               BasicFilterLineSearch,
                                               newton_unconstrained)
from pyomo.contrib.pynumero.linalg.solvers import ma27_solver
from pyomo.contrib.pynumero.interfaces.utils import grad_x_lagrangian
import pyomo.environ as aml
import math as pymath
import numpy as np
import logging
import json

from pyomo.contrib.pynumero.interfaces import PyomoNLP
from scipy.sparse import coo_matrix as scipy_coo_matrix


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


def _fraction_to_boundary(vars, delta, tau):
    return

class InteriorPointSolver(object):

    def __init__(self, **kwargs):
        self.logger = logging.getLogger()
        pass

    def _initialize_containers(self, nlp, bound_push=1e-2):

        # ToDo: provide a way to pass initial guesses from outside world (not in nlp)

        # create vector of primal variables
        self._x = nlp.x_init()
        # bound push of primal variables
        self._x = self._push_variables_within_bounds(nlp.xl(),
                                                     self._x,
                                                     nlp.xu(),
                                                     bound_push=bound_push)

        # create vector of slack variables
        self._s = nlp.evaluate_d(self._x)
        # bound push of slack variables
        self._s = self._push_variables_within_bounds(nlp.dl(),
                                                     self._s,
                                                     nlp.du(),
                                                     bound_push=bound_push)

        # create vector of equality constraint multipliers
        self._yc = nlp.create_vector_y(subset='c')  # initialized at zero
        # create vector of inequality constraint multipliers
        self._yd = nlp.create_vector_y(subset='d')  # initialized at zero

        # ToDo: initialize from QP the multipliers of equality and inequality constraints

        # create vector for the multipliers of lower and upper bounds on x
        self._zl = nlp.create_vector_x(subset='l')
        self._zl.fill(1.0)  # initialized at one
        self._zu = nlp.create_vector_x(subset='u')
        self._zu.fill(1.0)  # initialized at one

        # create vector for the multipliers of lower and upper bounds on s
        self._vl = nlp.create_vector_s(subset='l')
        self._vl.fill(1.0)  # initialized at one
        self._vu = nlp.create_vector_s(subset='u')
        self._vu.fill(1.0)  # initialized at one

        # create vector of steps
        self._dx = np.zeros(self._x.size)
        self._ds = np.zeros(self._s.size)
        self._dyc = np.zeros(self._yc.size)
        self._dyd = np.zeros(self._yd.size)
        self._dzl = np.zeros(self._zl.size)
        self._dzu = np.zeros(self._zu.size)
        self._dvl = np.zeros(self._vl.size)
        self._dvu = np.zeros(self._vu.size)

        # condensed vectors
        self._condensed_xl = nlp.xl(condensed=True)
        self._condensed_xu = nlp.xu(condensed=True)
        self._condensed_dl = nlp.dl(condensed=True)
        self._condensed_du = nlp.du(condensed=True)

        # create expansion matrices
        self._Pxl = nlp.expansion_matrix_xl()
        self._Pxu = nlp.expansion_matrix_xu()
        self._Pdl = nlp.expansion_matrix_dl()
        self._Pdu = nlp.expansion_matrix_du()
        self._Pc = nlp.expansion_matrix_c()
        self._Pd = nlp.expansion_matrix_d()

        # initialize nlp evaluation vectors and matrices
        # evaluate objective function
        self._f = nlp.objective(self._x)
        # evaluate gradient of the objective
        self._df = nlp.grad_objective(self._x)
        # evaluate residual constraints
        self._res_c = nlp.evaluate_c(self._x)
        # evaluate inequality constraints body
        self._body_d = nlp.evaluate_d(self._x)
        # evaluate jacobian equality constraints
        self._jac_c = nlp.jacobian_c(self._x)
        # evaluate jacobian inequality constraints
        self._jac_d = nlp.jacobian_d(self._x)
        # evaluate hessian of the lagrangian
        y = self._Pd * self._yd + self._Pc * self._yc
        self._hess_lag = nlp.hessian_lag(self._x, y, eval_f_c=False)

        self._primal_inf = np.inf
        self._dual_inf = np.inf
        self._regularization = "--"
        self._norm_d = 0.0
        self._alpha_dual = 0.0
        self._alpha_primal = 0.0
        self._num_ls = 0

    @staticmethod
    def _push_variables_within_bounds(vars_l, vars, vars_u, bound_push=1e-2):

        # push primal variables within lower bounds
        xl = vars_l
        max_one_xl = np.where(np.absolute(xl) <= 1.0, 1.0, xl)
        max_one_xl[max_one_xl <= -np.inf] = 0.0
        lower_pushed_x = np.where(vars <= xl,
                                  xl + bound_push * max_one_xl,
                                  vars)
        vars = lower_pushed_x

        # push primal variables within upper bound
        xu = vars_u
        max_one_xu = np.where(np.absolute(xu) <= 1.0, 1.0, xu)
        max_one_xu[max_one_xu >= np.inf] = 0.0
        upper_pushed_x = np.where(vars >= xu,
                                  xu - bound_push * max_one_xu,
                                  vars)
        vars = upper_pushed_x
        return vars

    def _evaluate_nlp_functions(self, nlp, x, y):

        # function evaluations
        self._f = nlp.objective(x)
        # evaluate gradient of the objective
        nlp.grad_objective(x, out=self._df)
        # evaluate residual constraints
        nlp.evaluate_c(x, out=self._res_c)
        # evaluate inequality constraints body
        nlp.evaluate_d(x, out=self._body_d)
        # evaluate jacobian equality constraints
        nlp.jacobian_c(x, out=self._jac_c)
        # evaluate jacobian inequality constraints
        nlp.jacobian_d(x, out=self._jac_d)
        # evaluate hessian of the lagrangian
        nlp.hessian_lag(x, y, out=self._hess_lag, eval_f_c=False)

    def _grad_lag_x(self):
        grad_x_lag = self._df + \
                     self._jac_c.transpose() * self._yc + \
                     self._jac_d.transpose() * self._yd + \
                     self._Pxu * self._zu - \
                     self._Pxl * self._zl
        return grad_x_lag

    def _grad_lag_s(self):
        grad_s_lag = self._Pdu * self._vu - \
                     self._Pdl * self._vl - \
                     self._yd
        return grad_s_lag

    def _grad_lag_bar_x(self):
        grad_x_lag_bar = self._df + \
                         self._jac_c.transpose() * self._yc + \
                         self._jac_d.transpose() * self._yd + \
                         self._Pxu * (self._miu / self._slack_xu()) - \
                         self._Pxl * (self._miu / self._slack_xl())
        return grad_x_lag_bar

    def _grad_lag_bar_s(self):
        grad_s_lag_bar = self._Pdu * (self._miu / self._slack_su()) - \
                         self._Pdl * (self._miu / self._slack_sl()) - \
                         self._yd
        return grad_s_lag_bar

    def _update_mu(self, etol, kappa_mu, theta_mu):
        self._miu = max(0.1*etol*min(kappa_mu*self._miu, self._miu**theta_mu))

    def _Dx_matrix(self):

        """
        # ToDo: this diagonal matrices probably dont need to be computed ever. Use them for testing
        Zl = DiagonalMatrix(self._zl)
        Zu = DiagonalMatrix(self._zu)
        SLxl = DiagonalMatrix(self._slack_xl())
        SLxu = DiagonalMatrix(self._slack_xu())
        SLxl_inv = SLxl.inv()
        SLxu_inv = SLxu.inv()

        # compute diagonal addition matrices
        Dx1 = self._Pxl * SLxl_inv * Zl * self._Pxl.transpose() - \
             self._Pxu * SLxu_inv * Zu * self._Pxu.transpose()
        """
        d_vec = self._Pxl * np.divide(self._zl, self._slack_xl()) - \
                self._Pxu * np.divide(self._zu, self._slack_xu())
        Dx = DiagonalMatrix(d_vec)

        return Dx

    def _Ds_matrix(self):

        """
        # ToDo: this diagonal matrices probably dont need to be computed ever. Use them for testing
        Vl = DiagonalMatrix(self._vl)
        Vu = DiagonalMatrix(self._vu)
        SLdl = DiagonalMatrix(self._slack_sl())
        SLdu = DiagonalMatrix(self._slack_su())
        SLdl_inv = SLdl.inv()
        SLdu_inv = SLdu.inv()

        Ds1 = self._Pdl * SLdl_inv * Vl * self._Pdl.transpose() - \
             self._Pdu * SLdu_inv * Vu * self._Pdu.transpose()
        """

        d_vec = self._Pdl * np.divide(self._vl, self._slack_sl()) - \
                self._Pdu * np.divide(self._vu, self._slack_su())
        Ds = DiagonalMatrix(d_vec)
        return Ds

    def _slack_sl(self):
        return self._Pdl.transpose() * self._s - self._condensed_dl

    def _slack_su(self):
        return self._condensed_du - self._Pdu.transpose() * self._s

    def _slack_xl(self):
        return self._Pxl.transpose() * self._x - self._condensed_xl

    def _slack_xu(self):
        return self._condensed_xu - self._Pxu.transpose() * self._x

    def _initialize_logger(self):
        self._json_logger = dict()
        self._json_logger['iterations'] = dict()
        self._json_logger['options'] = dict()

    def _log(self, iteration, level=1):

        self._json_logger['iterations'][iteration] = dict()
        self._json_logger['iterations'][iteration]['summary'] = dict()
        self._json_logger['iterations'][iteration]['summary']['objective'] = self._f
        self._json_logger['iterations'][iteration]['summary']['inf_pr'] = self._primal_inf
        self._json_logger['iterations'][iteration]['summary']['inf_du'] = self._dual_inf
        self._json_logger['iterations'][iteration]['summary']['||d||'] = self._norm_d
        self._json_logger['iterations'][iteration]['summary']['lg(rg)'] = self._regularization
        self._json_logger['iterations'][iteration]['summary']['lg(mu)'] = pymath.log10(self._miu)
        self._json_logger['iterations'][iteration]['summary']['alpha_du'] = self._alpha_dual
        self._json_logger['iterations'][iteration]['summary']['alpha_pr'] = self._alpha_primal
        self._json_logger['iterations'][iteration]['summary']['ls'] = self._num_ls

        if level > 1:
            self._json_logger['iterations'][iteration]['variables'] = dict()
            self._json_logger['iterations'][iteration]['variables']['||x||_inf'] = np.linalg.norm(self._x,
                                                                                                  ord=np.inf)
            if len(self._s) > 0:
                self._json_logger['iterations'][iteration]['variables']['||s||_inf'] = np.linalg.norm(self._s,
                                                                                                      ord=np.inf)
            else:
                self._json_logger['iterations'][iteration]['variables']['||s||_inf'] = 0.0
            if len(self._yc) > 0:
                self._json_logger['iterations'][iteration]['variables']['||yc||_inf'] = np.linalg.norm(self._yc,
                                                                                                       ord=np.inf)
            else:
                self._json_logger['iterations'][iteration]['variables']['||yc||_inf'] = 0.0
            if len(self._yd) > 0:
                self._json_logger['iterations'][iteration]['variables']['||yd||_inf'] = np.linalg.norm(self._yd,
                                                                                                       ord=np.inf)
            else:
                self._json_logger['iterations'][iteration]['variables']['||yd||_inf'] = 0.0
            if len(self._zl) > 0:
                self._json_logger['iterations'][iteration]['variables']['||zl||_inf'] = np.linalg.norm(self._zl,
                                                                                                       ord=np.inf)
            else:
                self._json_logger['iterations'][iteration]['variables']['||zl||_inf'] = 0.0
            if len(self._zu) > 0:
                self._json_logger['iterations'][iteration]['variables']['||zu||_inf'] = np.linalg.norm(self._zu,
                                                                                                       ord=np.inf)
            else:
                self._json_logger['iterations'][iteration]['variables']['||zu||_inf'] = 0.0
            if len(self._vl) > 0:
                self._json_logger['iterations'][iteration]['variables']['||vl||_inf'] = np.linalg.norm(self._vl,
                                                                                                       ord=np.inf)
            else:
                self._json_logger['iterations'][iteration]['variables']['||vl||_inf'] = 0.0
            if len(self._vu) > 0:
                self._json_logger['iterations'][iteration]['variables']['||vu||_inf'] = np.linalg.norm(self._vu,
                                                                                                       ord=np.inf)
            else:
                self._json_logger['iterations'][iteration]['variables']['||vu||_inf'] = 0.0

        if level > 2:
            self._json_logger['iterations'][iteration]['variables']['x'] = dict(enumerate(self._x))
            self._json_logger['iterations'][iteration]['variables']['s'] = dict(enumerate(self._s))
            self._json_logger['iterations'][iteration]['variables']['yc'] = dict(enumerate(self._yc))
            self._json_logger['iterations'][iteration]['variables']['yd'] = dict(enumerate(self._yd))
            self._json_logger['iterations'][iteration]['variables']['zl'] = dict(enumerate(self._zl))
            self._json_logger['iterations'][iteration]['variables']['zu'] = dict(enumerate(self._zu))
            self._json_logger['iterations'][iteration]['variables']['vl'] = dict(enumerate(self._vl))
            self._json_logger['iterations'][iteration]['variables']['vu'] = dict(enumerate(self._vu))

        if level > 3:
            self._json_logger['iterations'][iteration]['gradients'] = dict()
            self._json_logger['iterations'][iteration]['gradients']['grad_f'] = dict(enumerate(self._df))
            self._json_logger['iterations'][iteration]['gradients']['grad_lag_x'] = dict(enumerate(self._grad_lag_x()))
            self._json_logger['iterations'][iteration]['gradients']['grad_lag_s'] = dict(enumerate(self._grad_lag_s()))
            self._json_logger['iterations'][iteration]['gradients']['grad_lag_yc'] = dict(enumerate(self._res_c))
            self._json_logger['iterations'][iteration]['gradients']['grad_lag_yd'] = dict(enumerate(self._body_d - self._s))
            self._json_logger['iterations'][iteration]['gradients']['grad_lag_bar_x'] = dict(enumerate(self._grad_lag_bar_x()))
            self._json_logger['iterations'][iteration]['gradients']['grad_lag_bar_s'] = dict(enumerate(self._grad_lag_bar_s()))

        if level > 4:
            self._json_logger['iterations'][iteration]['slack_vectors'] = dict()
            self._json_logger['iterations'][iteration]['slack_vectors']['slack_xl'] = dict(enumerate(self._slack_xl()))
            self._json_logger['iterations'][iteration]['slack_vectors']['slack_xu'] = dict(enumerate(self._slack_xu()))
            self._json_logger['iterations'][iteration]['slack_vectors']['slack_sl'] = dict(enumerate(self._slack_sl()))
            self._json_logger['iterations'][iteration]['slack_vectors']['slack_su'] = dict(enumerate(self._slack_su()))

        if level > 5:
            self._json_logger['iterations'][iteration]['steps'] = dict()

            if len(self._ds) > 0:
                self._json_logger['iterations'][iteration]['steps']['||ds||_inf'] = np.linalg.norm(self._ds,
                                                                                                      ord=np.inf)
            else:
                self._json_logger['iterations'][iteration]['steps']['||ds||_inf'] = 0.0
            if len(self._dyc) > 0:
                self._json_logger['iterations'][iteration]['steps']['||dyc||_inf'] = np.linalg.norm(self._dyc,
                                                                                                       ord=np.inf)
            else:
                self._json_logger['iterations'][iteration]['steps']['||dyc||_inf'] = 0.0
            if len(self._dyd) > 0:
                self._json_logger['iterations'][iteration]['steps']['||dyd||_inf'] = np.linalg.norm(self._dyd,
                                                                                                       ord=np.inf)
            else:
                self._json_logger['iterations'][iteration]['steps']['||dyd||_inf'] = 0.0
            if len(self._dzl) > 0:
                self._json_logger['iterations'][iteration]['steps']['||dzl||_inf'] = np.linalg.norm(self._dzl,
                                                                                                       ord=np.inf)
            else:
                self._json_logger['iterations'][iteration]['steps']['||dzl||_inf'] = 0.0
            if len(self._dzu) > 0:
                self._json_logger['iterations'][iteration]['steps']['||dzu||_inf'] = np.linalg.norm(self._dzu,
                                                                                                       ord=np.inf)
            else:
                self._json_logger['iterations'][iteration]['steps']['||dzu||_inf'] = 0.0
            if len(self._dvl) > 0:
                self._json_logger['iterations'][iteration]['steps']['||dvl||_inf'] = np.linalg.norm(self._dvl,
                                                                                                       ord=np.inf)
            else:
                self._json_logger['iterations'][iteration]['steps']['||dvl||_inf'] = 0.0
            if len(self._dvu) > 0:
                self._json_logger['iterations'][iteration]['steps']['||dvu||_inf'] = np.linalg.norm(self._dvu,
                                                                                                       ord=np.inf)
            else:
                self._json_logger['iterations'][iteration]['steps']['||dvu||_inf'] = 0.0

        if level > 6:
            self._json_logger['iterations'][iteration]['steps']['dx'] = dict(enumerate(self._dx))
            self._json_logger['iterations'][iteration]['steps']['ds'] = dict(enumerate(self._ds))
            self._json_logger['iterations'][iteration]['steps']['dyc'] = dict(enumerate(self._dyc))
            self._json_logger['iterations'][iteration]['steps']['dyd'] = dict(enumerate(self._dyd))
            self._json_logger['iterations'][iteration]['steps']['dzl'] = dict(enumerate(self._dzl))
            self._json_logger['iterations'][iteration]['steps']['dzu'] = dict(enumerate(self._dzu))
            self._json_logger['iterations'][iteration]['steps']['dvl'] = dict(enumerate(self._dvl))
            self._json_logger['iterations'][iteration]['steps']['dvu'] = dict(enumerate(self._dvu))

        if level > 7:
            self._json_logger['iterations'][iteration]['matrices'] = dict()
            hess = dict_matrix(self._hess_lag)
            hess = {str(k):v for k, v in hess.items()}
            self._json_logger['iterations'][iteration]['matrices']['hessian_lag'] = hess
            jac_c = dict_matrix(self._jac_c)
            jac_c = {str(k): v for k, v in jac_c.items()}
            self._json_logger['iterations'][iteration]['matrices']['jac_c'] = jac_c
            jac_d = dict_matrix(self._jac_d)
            jac_d = {str(k): v for k, v in jac_d.items()}
            self._json_logger['iterations'][iteration]['matrices']['jac_d'] = jac_d
            dic_m = dict_matrix(self._Dx_matrix())
            dic_m = {str(k): v for k, v in dic_m.items()}
            self._json_logger['iterations'][iteration]['matrices']['Dx'] = dic_m
            dic_m = dict_matrix(self._Ds_matrix())
            dic_m = {str(k): v for k, v in dic_m.items()}
            self._json_logger['iterations'][iteration]['matrices']['Ds'] = dic_m

    def _log_linear_system(self, iteration, kkt, rhs, sol):

        self._json_logger['iterations'][iteration]['linear_system'] = dict()
        self._json_logger['iterations'][iteration]['linear_system']['rhs'] = dict()
        self._json_logger['iterations'][iteration]['linear_system']['sol'] = dict()
        self._json_logger['iterations'][iteration]['linear_system']['kkt'] = dict()

        for bid, v in enumerate(rhs):
            self._json_logger['iterations'][iteration]['linear_system']['rhs'][str(bid)] = dict(enumerate(v))

        for m in range(kkt.bshape[0]):
            for n in range(kkt.bshape[1]):
                if m >= n and kkt[m, n] is not None:
                    _str = "{},{}".format(m, n)
                    mat = dict_matrix(kkt[m, n])
                    mat = {str(k): v for k, v in mat.items()}
                    self._json_logger['iterations'][iteration]['linear_system']['kkt'][_str] = mat

        for bid, v in enumerate(sol):
            self._json_logger['iterations'][iteration]['linear_system']['sol'][str(bid)] = dict(enumerate(v))

    def _print_summary(self, iteration, val_reg):

        if val_reg == 0.0:
            formating = "{:>4d} {:>14.7e} {:>7.2e} {:>7.2e}  {:.1f} {:>9.2e} {:>4} {:>10.2e} {:>8.2e} {:>3d}"
            self._regularization = "--"
        else:
            formating = "{:>4d} {:>14.7e} {:>7.2e} {:>7.2e}  {:.1f} {:>9.2e}  {:>4.1f}  {:>7.2e} {:>7.2e} {:>3d}"
            self._regularization = pymath.log10(val_reg)

        line = formating.format(iteration,
                                self._f,
                                self._primal_inf,
                                self._dual_inf,
                                pymath.log10(self._miu),
                                self._norm_d,
                                self._regularization,
                                self._alpha_dual,
                                self._alpha_primal,
                                self._num_ls)
        print(line)

    def _optimality_error(self,
                          grad_lag_x,
                          grad_lag_s,
                          res_c,
                          res_d):

        r = np.zeros(8)
        r[0] = np.linalg.norm(grad_lag_x, ord=np.inf)
        r[1] = np.linalg.norm(grad_lag_s, ord=np.inf)
        r[2] = np.linalg.norm(res_c, ord=np.inf)
        r[3] = np.linalg.norm(res_d, ord=np.inf)
        if self._zl.size > 0:
            r[4] = np.linalg.norm(np.multiply(self._slack_xl(), self._zl) - self._miu, ord=np.inf)
        if self._zu.size > 0:
            r[5] = np.linalg.norm(np.multiply(self._slack_xu(), self._zu) - self._miu, ord=np.inf)
        if self._vl.size > 0:
            r[6] = np.linalg.norm(np.multiply(self._slack_sl(), self._vl) - self._miu, ord=np.inf)
        if self._vu.size > 0:
            r[7] = np.linalg.norm(np.multiply(self._slack_su(), self._vu) - self._miu, ord=np.inf)
        print(r)
        return np.amax(r)

    def _fraction_to_boundary_primal(self, delta_x, delta_s, tau):

        alpha_l_x = 1.0
        alpha_u_x = 1.0
        alpha_l_s = 1.0
        alpha_u_s = 1.0

        if self._condensed_xl.size > 0:
            delta_xl = self._Pxl.transpose() * delta_x
            alphas = np.divide(-tau * self._slack_xl(), delta_xl)
            alpha_l_x = alphas.min()
            if alpha_l_x < 0:
                alpha_l_x = 1.0

        if self._condensed_xu.size > 0:
            delta_xu = self._Pxu.transpose() * delta_s
            alpha_u_x = np.divide(tau * self._slack_xu(), delta_xu).min()
            if alpha_u_x < 0:
                alpha_u_x = 1.0

        if self._condensed_dl.size > 0:
            delta_sl = self._Pdl.transpose() * delta_s
            alpha_l_s = np.divide(-tau * self._slack_sl(), delta_sl).min()
            if alpha_l_s < 0:
                alpha_l_s = 1.0

        if self._condensed_du.size > 0:
            delta_su = self._Pdu.transpose() * delta_s
            alpha_u_s = np.divide(tau * self._slack_su(), delta_su).min()
            if alpha_u_s < 0:
                alpha_u_s = 1.0

        return min([alpha_l_x, alpha_u_x, alpha_l_s, alpha_u_s])

    def _fraction_to_boundary_dual(self, delta_zl, delta_zu, delta_vl, delta_vu, tau):

        alpha_l_z = 1.0
        alpha_u_z = 1.0
        alpha_l_v = 1.0
        alpha_u_v = 1.0

        if self._zl.size > 0:
            alphas = np.divide(-tau * self._zl, delta_zl)
            #print("zl",alphas)
            alpha_l_z = alphas.min()
            if alpha_l_z < 0:
                alpha_l_z = 1.0

        if self._vl.size > 0:
            alphas = np.divide(-tau * self._vl, delta_vl)
            alpha_l_v = alphas.min()
            if alpha_l_v < 0:
                alpha_l_v = 1.0

        """
        if self._zu.size > 0:
            alphas = np.divide(tau * self._zu, delta_zu)
            print(alphas)
            alpha_u_z = alphas.min()
            if alpha_u_z < 0:
                alpha_u_z = 1.0

        if self._vu.size > 0:
            alphas = np.divide(tau * self._vu, delta_vu)
            print("vu", alphas)
            alpha_u_v = alphas.min()
            if alpha_u_v < 0:
                alpha_u_v = 1.0
        """
        return min([alpha_l_z, alpha_u_z, alpha_l_v, alpha_u_v])

    def solve(self, nlp, **kwargs):

        # parses optional inputs
        tee = kwargs.pop('tee', True)
        outer_max_iter = kwargs.pop('max_iter_outer', 100)
        inner_max_iter = kwargs.pop('max_iter_inner', 100)
        reg_max_iter = kwargs.pop('reg_max_iter', 100)
        wls = kwargs.pop('wls', True)
        debug_mode = kwargs.pop('debug_mode', False)

        # initialize barrier parameter
        self._miu = kwargs.pop('mu_init', 0.1)
        kappa_miu = kwargs.pop('kappa_miu', 0.2)
        theta_miu = kwargs.pop('theta_miu', 1.5)
        tao_min = kwargs.pop('tao_min', 0.99)
        tol = kwargs.pop('tol', 1e-8)

        # define vectors and matrices for ip-algorithm
        self._initialize_containers(nlp)
        self._initialize_logger()

        # build diagonal matrices
        Dx = self._Dx_matrix()
        Ds = self._Ds_matrix()

        # create KKT system
        kkt = BlockSymMatrix(4)
        kkt[0, 0] = self._hess_lag + Dx
        kkt[1, 1] = - Ds
        kkt[2, 0] = self._jac_c
        kkt[3, 0] = self._jac_d
        kkt[3, 1] = - IdentityMatrix(nlp.nd)

        # create RHS for newton step
        rhs = BlockVector([-self._grad_lag_bar_x(),
                           -self._grad_lag_bar_s(),
                           -self._res_c,
                           self._s-self._body_d])

        # create linear solver
        lsolver = ma27_solver.MA27LinearSolver()
        lsolver.do_symbolic_factorization(kkt, include_diagonal=False)
        diag_correction = np.zeros(kkt.shape[0])
        # ToDo: line search helper with fraction to the boundary rule

        print_nlp_info(nlp)
        val_reg = 0.0
        counter_iter = 0
        for i in range(1):
            for j in range(3):

                self._primal_inf = np.linalg.norm(self._res_c, ord=np.inf)
                self._dual_inf = np.linalg.norm(self._grad_lag_x(), ord=np.inf)

                if counter_iter % 10 == 0 and tee:
                    print("iter    objective    inf_pr   inf_du  lg(mu)  ||d||   lg(rg) alpha_du alpha_pr  ls")
                if tee:
                    self._print_summary(counter_iter, val_reg)


                # logger for debug
                self._log(counter_iter, level=10)

                # before going into linear solver check convergence
                #print(self._optimality_error(self._grad_lag_x(),self._grad_lag_s(), self._res_c, self._body_d - self._s))
                # compute search direction
                status = lsolver.do_numeric_factorization(kkt)

                # solve kkt
                dvars = lsolver.do_back_solve(rhs)

                self._log_linear_system(counter_iter, kkt, rhs, dvars)

                # update step vectors
                self._dx = dvars[0]
                self._ds = dvars[1]
                self._dyc = dvars[2]
                self._dyd = dvars[3]

                ZlPxldx = np.multiply(self._zl, self._Pxl.transpose() * self._dx)
                self._dzl = np.divide(self._miu - ZlPxldx, self._slack_xl()) - self._zl
                ZuPxudx = np.multiply(self._zu, self._Pxu.transpose() * self._dx)
                self._dzu = np.divide((self._miu - ZuPxudx), self._slack_xu()) + self._zu
                VlPdlds = np.multiply(self._vl, self._Pdl.transpose() * self._ds)
                self._dvl = np.divide((self._miu - VlPdlds), self._slack_sl()) - self._vl
                VuPduds = np.multiply(self._vu, self._Pdu.transpose() * self._ds)
                self._dvu = np.divide((self._miu - VuPduds), self._slack_su()) + self._vu

                self._norm_d = np.linalg.norm(self._dx, ord=np.inf)

                # update vectors of variables
                tao = max(tao_min, 1-self._miu)
                self._alpha_dual = self._fraction_to_boundary_dual(self._dzl, self._dzu, self._dvl, self._dvu, tao)
                self._alpha_primal = self._fraction_to_boundary_primal(self._dx, self._ds, tao)

                self._x += self._dx * self._alpha_primal
                self._s += self._ds * self._alpha_primal
                self._yc += self._dyc * self._alpha_primal
                self._yd += self._dyd * self._alpha_primal
                self._zl += self._dzl * self._alpha_dual
                self._zu += self._dzu * self._alpha_dual
                self._vl += self._dvl * self._alpha_dual
                self._vu += self._dvu * self._alpha_dual

                # evaluate nlp functions at new point
                y = self._Pd * self._yd + self._Pc * self._yc
                self._evaluate_nlp_functions(nlp, self._x, y)

                # compute diagonal addition matrices
                Dx = self._Dx_matrix()
                Ds = self._Ds_matrix()

                # update kkt
                kkt[0, 0] = self._hess_lag + Dx
                kkt[1, 1] = -Ds
                kkt[2, 0] = self._jac_c
                kkt[3, 0] = self._jac_d

                # update rhs
                rhs[0] = -self._grad_lag_bar_x()
                rhs[1] = -self._grad_lag_bar_s()
                rhs[2] = -self._res_c
                rhs[3] = self._s - self._body_d

                counter_iter += 1

        with open('data.txt', 'w') as outfile:
            json.dump(self._json_logger, outfile, sort_keys=True)


if __name__ == "__main__":

    m = aml.ConcreteModel()
    m.x = aml.Var([1, 2, 3], initialize=4.0)
    m.c = aml.Constraint(expr=m.x[3]**2+m.x[1] == 25)
    m.d = aml.Constraint(expr=m.x[2] ** 2 + m.x[1] <= 18.0)
    #m.d = aml.Constraint(expr=aml.inequality(-18, m.x[2] ** 2 + m.x[1],  28))
    m.o = aml.Objective(expr=m.x[1]**4 - 3*m.x[1]*m.x[2]**3 + m.x[3]**2 - 8.0)
    m.x[1].setlb(0.0)
    m.x[2].setlb(0.0)

    solver = aml.SolverFactory('ipopt')
    #solver.solve(m, tee=True)
    #m.x.pprint()

    nlp = PyomoNLP(m)
    print(nlp.variable_order())
    opt = InteriorPointSolver()
    opt.solve(nlp)


