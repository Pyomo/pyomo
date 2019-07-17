#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import numpy as np
from pyomo.contrib.pynumero.interfaces.nlp_compositions import CompositeNLP
from pyomo.contrib.pynumero.sparse import (BlockVector,
                                           BlockSymMatrix,
                                           diagonal_matrix)
from scipy.sparse import identity
import pyomo.contrib.pynumero as pn
from pyomo.contrib.pynumero.linalg.intrinsics import norm as pynumero_norm

# ToDo: think of keeping previous state values in t-1
# the state can potentially have the logger

class BasicNLPState(object):
    """
    Helper class for implementing linear and nonlinear optimization algorithms.
    The NLPState object provides a way for caching nlp evaluation functions. This
    is often beneficial when writing iterative algorithms for solving optimization
    problems

    Attributes
    ----------
    _nlp: NLP
        A subclass of the pyomo.contrib.pynumero.interfaces.nlp.NLP object
    _x: array_like
        Vector of primal variables. This is an instance of numpy.ndarray or a
        subclass of it. For instance, for composite NLPs this is a BlockVector
    _yc: array_like
        Vector of dual variables for equality constraints. This is an instance
        of numpy.ndarray or a subclass of it. For instance, for composite NLPs
        this is a BlockVector
    _yd: array_like
        Vector of dual variables for inequality constraints. This is an instance
        of numpy.ndarray or a subclass of it. For instance, for composite NLPs
        this is a BlockVector
    _s: array_like
        Vector of slack variables for inequality constraints. This is an instance
        of numpy.ndarray or a subclass of it. For instance, for composite NLPs
        this is a BlockVector
    _dx: array_like
        Step vector of primal variables. This is an instance of numpy.ndarray or a
        subclass of it. For instance, for composite NLPs this is a BlockVector
    _dyc: array_like
        Step vector of dual variables for equality constraints. This is an instance
        of numpy.ndarray or a subclass of it. For instance, for composite NLPs
        this is a BlockVector
    _dyd: array_like
        Step vector of dual variables for inequality constraints. This is an instance
        of numpy.ndarray or a subclass of it. For instance, for composite NLPs
        this is a BlockVector
    _ds: array_like
        Step vector of slack variables for inequality constraints. This is an instance
        of numpy.ndarray or a subclass of it. For instance, for composite NLPs
        this is a BlockVector
    condensed_xl: array_like
        Vector of lower bounds of primal variables (without -inf values). This is an instance
        of numpy.ndarray or a subclass of it. For instance, for composite NLPs
        this is a BlockVector
    condensed_xu: array_like
        Vector of upper bounds of primal variables (without inf values). This is an instance
        of numpy.ndarray or a subclass of it. For instance, for composite NLPs
        this is a BlockVector
    condensed_dl: array_like
        Vector of lower bounds for inequality constriants (without -inf values). This is an instance
        of numpy.ndarray or a subclass of it. For instance, for composite NLPs
        this is a BlockVector
    condensed_du: array_like
        Vector of upper bounds for inequality constriant (without inf values). This is an instance
        of numpy.ndarray or a subclass of it. For instance, for composite NLPs
        this is a BlockVector
    Pxl: Matrix_like
        projection matrix that brings a condensed vector from xl to x (Pxl.dot(xl))
    Pxu: Matrix_like
        projection matrix that brings a condensed vector from xu to x (Pxu.dot(xu))
    Pdl: Matrix_like
        projection matrix that brings a condensed vector from dl to s (Pdl.dot(s))
    Pdu: Matrix_like
        projection matrix that brings a condensed vector from du to s (Pdu.dot(s))
    Pc: Matrix_like
        projection matrix that brings a condensed vector from c to g (Pc.dot(c))
    Pd: Matrix_like
        projection matrix that brings a condensed vector from d to g (Pd.dot(d))
    _f: float64
        objective function evaluated at current state point
    _df: array_like
        gradient of objective function evaluated at current state point
    _g: array_like
        contraints evaluated at current state point
    _res_c: array_like
        equality contraints evaluated at current state point
    _body_d: array_like
        inequality contraints evaluated at current state point
    _jac_g: matrix_like
        jacobian of contraints evaluated at current state point
    _jac_c: matrix_like
        jacobian of equality contraints evaluated at current state point
    _jac_d: matrix_like
        jacobian of inequality contraints evaluated at current state point
    _hess_lag: matrix_like
        hessian of Lagrangian function evaluated at current state point
    _slack_xl: array_like
        vector with difference between current primals and lower bound xl
    _slack_xu: array_like
        vector with difference between current primals and upper bound xu
    _slack_sl: array_like
        vector with difference between current slacks and lower bound dl
    _slack_su: array_like
        vector with difference between current slacks and lower bound du

    Parameters
    ----------
    nlp: NLP object
        A subclass of the pyomo.contrib.pynumero.interfaces.nlp.NLP object
    """

    def __init__(self, nlp, **kwargs):
        bound_push = kwargs.pop('bound_push', 1e-2)
        push_bounds = not kwargs.pop('disable_bound_push', False)

        # nonlinear programm
        self._nlp = nlp

        # vector of primal variables
        self._x = nlp.x_init()
        if push_bounds:
            self._x = self.push_variables_within_bounds(nlp.xl(),
                                                        nlp.x_init(),
                                                        nlp.xu(),
                                                        bound_push)
        # vector of slack variables
        self._s = nlp.evaluate_d(self.x)
        if push_bounds:
            self._s = self.push_variables_within_bounds(nlp.dl(),
                                                       self._s,
                                                       nlp.du(),
                                                       bound_push)
        # vector of equality constraint multipliers
        self._yc = nlp.create_vector('c')
        # vector of inequality constraint multipliers
        self._yd = nlp.create_vector('d')

        # step vector of primal variables
        self._dx = nlp.create_vector('x')
        self._dx.fill(0.0)
        # step vector of slack variables
        self._ds = nlp.create_vector('d')
        self._ds.fill(0.0)
        # step vector of equality constraint multiplier variables
        self._dyc = nlp.create_vector('c')
        self._dyc.fill(0.0)
        # step vector of inequality constraint multiplier variables
        self._dyd = nlp.create_vector('d')
        self._dyd.fill(0.0)

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
        self.Pxl = nlp.projection_matrix_xl()
        # expansion matrix from condensed xu to x
        self.Pxu = nlp.projection_matrix_xu()
        # expansion matrix from condensed dl to d
        self.Pdl = nlp.projection_matrix_dl()
        # expansion matrix from condensed du to d
        self.Pdu = nlp.projection_matrix_du()
        # expansion matrix from c to g
        self.Pc = nlp.projection_matrix_c()
        # expansion matrix from c to g
        self.Pd = nlp.projection_matrix_d()

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

        # determines if recaching of functions is required
        self._need_recache_nlp_functions_x = True
        self._need_recache_nlp_functions_y = True
        self._need_recache_slack_vectors_x = True
        self._need_recache_slack_vectors_s = True

        # populate nlp caches (recaching flags must be declare before)
        self.cache()

    @property
    def x(self):
        """Returns current vector of primal variables"""
        return self._x

    @property
    def s(self):
        """Returns current vector of slack variables"""
        return self._s

    @property
    def yc(self):
        """Returns current vector of dual variables of equality constraints"""
        return self._yc

    @property
    def yd(self):
        """Returns current vector of dual variables of inequality constraints"""
        return self._yd

    @property
    def dx(self):
        """Returns last step vector of primal variables"""
        return self._dx

    @property
    def ds(self):
        """Returns last step vector of slack variables"""
        return self._ds

    @property
    def dyc(self):
        """Returns last step vector of dual variables of equality constriants"""
        return self._dyc

    @property
    def dyd(self):
        """Returns last step vector of dual variables of equality constriants"""
        return self._dyd

    @property
    def nlp(self):
        """Returns nlp object"""
        return self._nlp

    def update_state(self, **kwargs):
        """
        Update variables and function evaluations to a new state point

        Other Parameters
        ----------------
        x: array_like
            new vector of primal variables
        s: array_like
            new vector of primal variables
        yc: array_like
            new vector of dual variables of equality constraints
        yd: array_like
            new vector of dual variables of equality constraints

        """
        x = kwargs.pop('x', None)
        s = kwargs.pop('s', None)
        yc = kwargs.pop('yc', None)
        yd = kwargs.pop('yd', None)

        if x is not None:
            assert isinstance(x, np.ndarray)
            assert x.size == self.nlp.nx
            # should we check is in the interior?
            self._dx = x-self._x
            self._x = x
            self._need_recache_nlp_functions_x = True
            self._need_recache_slack_vectors_x = True

        if s is not None:
            assert isinstance(s, np.ndarray)
            assert s.size == self.nlp.nd
            # should we check is in the interior?
            self._ds = s-self._s
            self._s = s
            self._need_recache_slack_vectors_s = True

        if yc is not None:
            assert isinstance(yc, np.ndarray)
            assert yc.size == self.nlp.nc
            self._dyc = yc-self._yc
            self._yc = yc
            self._need_recache_nlp_functions_y = True

        if yd is not None:
            assert isinstance(yd, np.ndarray)
            assert yd.size == self.nlp.nd
            self._dyd = yd-self._yd
            self._yd = yd
            self._need_recache_nlp_functions_y = True

        self.cache()

    def cache(self):
        """Evaluates and store all functions at current state point"""
        self._cache_nlp_functions()
        self._cache_slack_vectors()

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
            # evaluate jacobian equality constraints
            self._jac_g = nlp.jacobian_g(x, out=self._jac_g)
            # evaluate jacobian equality constraints
            self._jac_c = nlp.jacobian_c(x, out=self._jac_c, evaluated_jac_g=self._jac_g)
            # evaluate jacobian inequality constraints
            self._jac_d = nlp.jacobian_d(x, out=self._jac_d, evaluated_jac_g=self._jac_g)

        if self._need_recache_nlp_functions_x or \
            self._need_recache_nlp_functions_y:
            # evaluate hessian of the lagrangian
            yd = self.Pd * self.yd
            yc = self.Pc * self.yc
            y = yd + yc
            self._hess_lag = nlp.hessian_lag(x, y, out=self._hess_lag, eval_f_c=False)

        self._need_recache_nlp_functions_x = False
        self._need_recache_nlp_functions_y = False

    def _cache_slack_vectors(self):
        # ToDo: remove safeguard (if fraction to the boundary is working
        # appropiately the safeguard should never be needed)
        # intead raise a warning "to close to boundary"
        safeguard = 1e-12
        if self._need_recache_slack_vectors_x:
            self._slack_xl = self.Pxl.transpose() * self._x - self.condensed_xl
            self._slack_xl = self._slack_xl.clip(min=safeguard)
            self._slack_xu = self.condensed_xu - self.Pxu.transpose() * self._x
            self._slack_xu = self._slack_xu.clip(min=safeguard)
        if self._need_recache_slack_vectors_s:
            self._slack_sl = self.Pdl.transpose() * self._s - self.condensed_dl
            self._slack_sl = self._slack_sl.clip(min=safeguard)
            self._slack_su = self.condensed_du - self.Pdu.transpose() * self._s
            self._slack_su = self._slack_su.clip(min=safeguard)

        self._need_recache_slack_vectors_x = False
        self._need_recache_slack_vectors_s = False

    def objective(self):
        """Returns objective function at current state point"""
        return self._f

    def grad_objective(self):
        """Returns vector with gradient of objective function at current state point"""
        return self._df

    def evaluate_g(self):
        """Returns vector of constraints evaluated at current state point"""
        return self._g

    def evaluate_c(self):
        """Returns vector of equality constraints evaluated at current state point"""
        return self._res_c

    def evaluate_d(self):
        """Returns vector of inequality constraints evaluated at current state point"""
        return self._body_d

    def residual_d(self):
        """Returns vector of residuals inequality constraints evaluated at current state point"""
        return self._body_d - self.s

    def jacobian_g(self):
        """Returns jacobian matrix of constraints evaluated at current state point"""
        return self._jac_g

    def jacobian_c(self):
        """Returns jacobian matrix of equality constraints evaluated at current state point"""
        return self._jac_c

    def jacobian_d(self):
        """Returns jacobian matrix of inequality constraints evaluated at current state point"""
        return self._jac_d

    def hessian_lag(self):
        """Returns hessian of lagrangian function evaluated at current state point"""
        return self._hess_lag

    def slack_xl(self):
        """Returns vector with difference between current primal variables and lower bound xl"""
        return self._slack_xl

    def slack_xu(self):
        """Returns vector with difference between current primal variables and upper bound xu"""
        return self._slack_xu

    def slack_sl(self):
        """Returns vector with difference between current slack variables and lower bound dl"""
        return self._slack_sl

    def slack_su(self):
        """Returns vector with difference between current slack variables and upper bound du"""
        return self._slack_su

    def max_alpha_primal(self, delta_x, delta_s, tau):
        """Return primal step size from fraction to the boundary rule"""
        alpha_l_x = 1.0
        alpha_u_x = 1.0
        alpha_l_s = 1.0
        alpha_u_s = 1.0

        if self.condensed_xl.size > 0:

            delta_xl = self.Pxl.T * delta_x
            mask = delta_xl < 0.0
            if mask.any():
                subset_delta_xl = delta_xl.compress(mask)
                subset_sli = self.slack_xl().compress(mask)
                alpha_l_x = np.divide(subset_delta_xl,
                                      -tau * subset_sli).max()
                alpha_l_x = min(1.0/alpha_l_x, 1.0)
            else:
                alpha_l_x = 1.0

        if self.condensed_xu.size > 0:

            delta_xu = self.Pxu.T * delta_x
            mask = delta_xu > 0.0
            if mask.any():
                subset_delta_xu = delta_xu.compress(mask)
                subset_sui = self.slack_xu().compress(mask)
                alpha_u_x = np.divide(subset_delta_xu,
                                      tau * subset_sui).max()
                alpha_u_x = min(1.0/alpha_u_x, 1.0)
            else:
                alpha_u_x = 1.0

        if self.condensed_dl.size > 0:

            delta_sl = self.Pdl.T * delta_s
            mask = delta_sl < 0.0
            if mask.any():
                subset_delta_sl = delta_sl.compress(mask)
                subset_sli = self.slack_sl().compress(mask)
                alpha_l_s = np.divide(subset_delta_sl,
                                      -tau * subset_sli).max()
                alpha_l_s = min(1.0/alpha_l_s, 1.0)
            else:
                alpha_l_s = 1.0

        if self.condensed_du.size > 0:

            delta_su = self.Pdu.transpose() * delta_s
            mask = delta_su > 0.0
            if mask.any():
                subset_delta_su = delta_su.compress(mask)
                subset_sui = self.slack_su().compress(mask)
                alpha_u_s = np.divide(subset_delta_su,
                                      tau * subset_sui).max()
                alpha_u_s = min(1.0/alpha_u_s, 1.0)
            else:
                alpha_u_s = 1.0

        return min([alpha_l_x, alpha_u_x, alpha_l_s, alpha_u_s])

    def primal_infeasibility(self):
        """Returns infinity norm of g(x)=0"""
        norm_c = 0.0
        norm_d = 0.0
        if self.yc.size > 0:
            norm_c = pynumero_norm(self.evaluate_c(), ord=np.inf)
        if self.yd.size > 0:
            d_val = self.evaluate_d()
            res_dl = self.condensed_dl - self.Pdl.T.dot(d_val)
            res_du = self.Pdu.T.dot(d_val) - self.condensed_du
            if res_dl.size > 0:
                res_dl = res_dl.clip(min=0.0)
                norm_d += pynumero_norm(res_dl, ord=np.inf)
            if res_du.size > 0:
                res_du = res_du.clip(min=0.0)
                norm_d += pynumero_norm(res_du, ord=np.inf)
        return max(norm_c, norm_d)

    def norm_primal_step(self):
        """Returns infinity norm of primal step vector (including slacks s)"""
        normx = pynumero_norm(self.dx, ord=np.inf)
        norms = 0.0
        if self.ds.size > 0:
            norms = pynumero_norm(self.ds, ord=np.inf)
        return max(normx, norms)

    @staticmethod
    def push_variables_within_bounds(vars_l, vars, vars_u, bound_push):
        """
        Brings vector to interior of upper and lower bounds

        Parameters
        ----------
        vars_l: array_like
            vector of lower bounds
        vars_u: array_like
            vector of upper bounds
        vars: array_like
            vector of variables

        Returns
        -------
        array_like

        """
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


class NLPState(BasicNLPState):
    """
    Helper class for implementing linear and nonlinear optimization algorithms.
    The NLPState object provides a way for caching nlp evaluation functions. This
    is often beneficial when writing iterative algorithms for solving optimization
    problems

    Attributes
    ----------
    _zl: array_like
        Vector of multiplier variables of lower bounds on x. This is an instance
        of numpy.ndarray or a subclass of it. For instance, for composite NLPs
        this is a BlockVector
    _zu: array_like
        Vector of multiplier variables of upper bounds on x. This is an instance
        of numpy.ndarray or a subclass of it. For instance, for composite NLPs
        this is a BlockVector
    _vl: array_like
        Vector of multiplier variables of lower bounds on s. This is an instance
        of numpy.ndarray or a subclass of it. For instance, for composite NLPs
        this is a BlockVector
    _vu: array_like
        Vector of multiplier variables of upper bounds on s. This is an instance
        of numpy.ndarray or a subclass of it. For instance, for composite NLPs
        this is a BlockVector
    _dvl: array_like
        Step vector of multiplier variables of lower bounds on x. This is an
        instance of numpy.ndarray or a subclass of it. For instance, for
        composite NLPs this is a BlockVector
    _dsl: array_like
        Step vector of multiplier variables of lower bounds on s. This is an
        instance of numpy.ndarray or a subclass of it. For instance, for
        composite NLPs this is a BlockVector
    _dvu: array_like
        Step vector of multiplier variables of upper bounds on x. This is an
        instance of numpy.ndarray or a subclass of it. For instance, for
        composite NLPs this is a BlockVector
    _dsu: array_like
        Step vector of multiplier variables of upper bounds on s. This is an
        instance of numpy.ndarray or a subclass of it. For instance, for
        composite NLPs this is a BlockVector
    """
    def __init__(self, nlp, **kwargs):

        super(NLPState, self).__init__(nlp, **kwargs)

        # vector for the multipliers of lower bounds on x
        self._zl = nlp.create_vector('xl')
        self._zl.fill(1.0)
        # vector for the multipliers of upper bounds on x
        self._zu = nlp.create_vector('xu')
        self._zu.fill(1.0)

        # vector for the multipliers of lower bounds on s
        self._vl = nlp.create_vector('dl')
        self._vl.fill(1.0)
        # vector for the multipliers of upper bounds on s
        self._vu = nlp.create_vector('du')
        self._vu.fill(1.0)

        # step vector of multipliers of lower bounds on x
        self._dzl = nlp.create_vector('xl')
        self._dzl.fill(0.0)
        # step vector of multipliers of upper bounds on x
        self._dzu = nlp.create_vector('xu')
        self._dzu.fill(0.0)

        # step vector of multipliers of lower bounds on s
        self._dvl = nlp.create_vector('dl')
        self._dvl.fill(0.0)
        # step vector of multipliers of upper bounds on s
        self._dvu = nlp.create_vector('du')
        self._dvu.fill(0.0)

        self.cache()

    @property
    def zl(self):
        """Returns current vector of multiplier variables of lower bounds on x"""
        return self._zl

    @property
    def zu(self):
        """Returns current vector of multiplier variables of upper bounds on x"""
        return self._zu

    @property
    def vl(self):
        """Returns current vector of multiplier variables of lower bounds on s"""
        return self._vl

    @property
    def vu(self):
        """Returns current vector of multiplier variables of upper bounds on x"""
        return self._vu

    @property
    def dzl(self):
        """Returns last step vector of multiplier variables of lower bounds on x"""
        return self._dzl

    @property
    def dzu(self):
        """Returns last step vector of multiplier variables of upper bounds on x"""
        return self._dzu

    @property
    def dvl(self):
        """Returns last step vector of multiplier variables of lower bounds on s"""
        return self._dvl

    @property
    def dvu(self):
        """Returns last step vector of multiplier variables of upper bounds on s"""
        return self._dvu

    def update_state(self, **kwargs):
        """
        Update variables and function evaluations to a new state point

        Other Parameters
        ----------------
        x: array_like
            new vector of primal variables
        s: array_like
            new vector of primal variables
        yc: array_like
            new vector of dual variables of equality constraints
        yd: array_like
            new vector of dual variables of equality constraints
        zl: array_like
            new vector of dual variables of lower bounds on x
        vl: array_like
            new vector of dual variables of lower bounds on s
        zu: array_like
            new vector of dual variables of upper bounds on x
        vu: array_like
            new vector of dual variables of upper bounds on s
        """
        # call parent class
        super(NLPState, self).update_state(**kwargs)

        zl = kwargs.pop('zl', None)
        zu = kwargs.pop('zu', None)
        vl = kwargs.pop('vl', None)
        vu = kwargs.pop('vu', None)

        if zl is not None:
            assert isinstance(zl, np.ndarray)
            assert zl.size == self._zl.size
            self._dzl = zl-self._zl
            self._zl = zl

        if zu is not None:
            assert isinstance(zu, np.ndarray)
            assert zu.size == self._zu.size
            self._dzu = zu-self._zu
            self._zu = zu

        if vl is not None:
            assert isinstance(vl, np.ndarray)
            assert vl.size == self._vl.size
            self._dvl = vl-self._vl
            self._vl = vl

        if vu is not None:
            assert isinstance(vu, np.ndarray)
            assert vu.size == self._vu.size
            self._dvu = vu-self._vu
            self._vu = vu

    def grad_lag_x(self):
        """Returns gradient of the Lagrangian function with respect to x
        evaluated at current state point"""
        grad_x_lag = self.grad_objective() + \
                     self.jacobian_c().T * self.yc + \
                     self.jacobian_d().T * self.yd + \
                     self.Pxu * self.zu - \
                     self.Pxl * self.zl

        return grad_x_lag

    def grad_lag_s(self):
        """Returns gradient of the Lagrangian function with respect to s
        evaluated at current state point"""
        grad_s_lag = self.Pdu * self.vu - \
                     self.Pdl * self.vl - \
                     self.yd
        return grad_s_lag

    def grad_lag_bar_x(self):
        grad_x_lag_bar = self.grad_objective() + \
                         self.jacobian_c().T * self.yc + \
                         self.jacobian_d().T * self.yd
        return grad_x_lag_bar

    def grad_lag_bar_s(self):
        grad_s_lag_bar = self.yd
        return grad_s_lag_bar

    def Dx_matrix(self):

        d_vec = self.Pxl * np.divide(self.zl, self.slack_xl()) + \
                self.Pxu * np.divide(self.zu, self.slack_xu())

        if type(d_vec) == np.ndarray:
            return diagonal_matrix(d_vec)

        if isinstance(d_vec, BlockVector):
            dx_m = BlockSymMatrix(d_vec.nblocks)
            for j, v in enumerate(d_vec):
                dx_m[j, j] = diagonal_matrix(v)
            return dx_m

        msg = '{} not supported yet'.format(type(d_vec))
        raise NotImplementedError(msg)

    def Ds_matrix(self):

        d_vec = self.Pdl * np.divide(self.vl, self.slack_sl()) + \
                self.Pdu * np.divide(self.vu, self.slack_su())

        if type(d_vec) == np.ndarray:
            return diagonal_matrix(d_vec)
        if isinstance(d_vec, BlockVector):
            dx_m = BlockSymMatrix(d_vec.nblocks)
            for j, v in enumerate(d_vec):
                dx_m[j, j] = diagonal_matrix(v)
            return dx_m
        msg = '{} not supported yet'.format(type(d_vec))
        raise NotImplementedError(msg)

    def max_alpha_dual(self, delta_zl, delta_zu, delta_vl, delta_vu, tau):
        """Return dual step size from fraction to the boundary rule"""
        alpha_l_z = 1.0
        alpha_u_z = 1.0
        alpha_l_v = 1.0
        alpha_u_v = 1.0

        if self.zl.size > 0:

            mask = delta_zl < 0.0
            if mask.any():
                subset_delta_l_z = delta_zl.compress(mask)
                subset_zli = self.zl.compress(mask)
                alpha_l_z = np.divide(subset_delta_l_z, -tau * subset_zli).max()
                alpha_l_z = min(1.0/alpha_l_z, 1.0)
            else:
                alpha_l_z = 1.0

        if self.zu.size > 0:

            mask = delta_zu < 0.0
            if mask.any():
                subset_delta_u_z = delta_zu.compress(mask)
                subset_zui = self.zu.compress(mask)
                alpha_u_z = np.divide(subset_delta_u_z,
                                      -tau * subset_zui).max()
                alpha_u_z = min(1.0/alpha_u_z, 1.0)
            else:
                alpha_u_z = 1.0


        if self.vl.size > 0:

            mask = delta_vl < 0.0
            if mask.any():
                subset_delta_l_v = delta_vl.compress(mask)
                subset_vli = self.vl.compress(mask)
                alpha_l_v = np.divide(subset_delta_l_v,
                                      -tau * subset_vli).max()
                alpha_l_v = min(1.0/alpha_l_v, 1.0)
            else:
                alpha_l_v = 1.0

        if self.vu.size > 0:

            mask = delta_vu < 0.0
            if mask.any():
                subset_delta_u_v = delta_vu.compress(mask)
                subset_vui = self.vu.compress(mask)
                alpha_u_v = np.divide(subset_delta_u_v,
                                      -tau * subset_vui).max()
                alpha_u_v = min(1.0/alpha_u_v, 1.0)
            else:
                alpha_u_v = 1.0

        return min([alpha_l_z, alpha_u_z, alpha_l_v, alpha_u_v])

    def dual_infeasibility(self):
        norm_x = pynumero_norm(self.grad_lag_x(), ord=np.inf)
        norm_s = 0.0
        if self.s.size > 0:
            norm_s = pynumero_norm(self.grad_lag_s(), ord=np.inf)
        return max(norm_x, norm_s)

    def complementarity_infeasibility(self):

        r = np.zeros(4)
        if self.zl.size > 0:
            r[0] = pynumero_norm(np.multiply(self.slack_xl(), self.zl), ord=np.inf)
        if self.zu.size > 0:
            r[1] = pynumero_norm(np.multiply(self.slack_xu(), self.zu), ord=np.inf)
        if self.vl.size > 0:
            r[2] = pynumero_norm(np.multiply(self.slack_sl(), self.vl), ord=np.inf)
        if self.vu.size > 0:
            r[3] = pynumero_norm(np.multiply(self.slack_su(), self.vu), ord=np.inf)
        return np.amax(r)

    def scaling_factors_infeasibility(self, smax=100.0):

        sc = 1.0
        sd = 1.0

        suma = 0.0
        suma += np.absolute(self.zl).sum()
        suma += np.absolute(self.zu).sum()
        suma += np.absolute(self.vl).sum()
        suma += np.absolute(self.vu).sum()

        sc = suma
        n = self.zl.size + self.zu.size + self.vl.size + self.vu.size
        if n == 0:
            sc = 1.0
        else:
            sc = sc / n
            sc = max(smax, sc)/smax

        suma += np.absolute(self.yc).sum()
        suma += np.absolute(self.yd).sum()
        n = self.zl.size + self.zu.size + self.vl.size + \
            self.vu.size + self.yc.size + self.yd.size
        sd = suma
        if n == 0:
            sd = 1.0
        else:
            sd = sd / n
            sd = max(smax, sd) / smax

        return sc, sd
