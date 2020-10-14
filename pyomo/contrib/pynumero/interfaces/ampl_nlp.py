#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
"""
This module defines the classes that provide an NLP interface based on
the Ampl Solver Library (ASL) implementation
"""
try:
    import pyomo.contrib.pynumero.asl as _asl
except ImportError as e:
    print('{}'.format(e))
    raise ImportError('Error importing asl.'
                      'Make sure libpynumero_ASL is installed and added to path.')

from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP

__all__ = ['AslNLP', 'AmplNLP']

# ToDo: need to add support for modifying bounds.
# support for changing variable bounds seems possible.
# support for changing inequality bounds would require more work. (this is less frequent?)
# TODO: check performance impacts of cacheing - memory and computational time.
# TODO: only create and cache data for ExtendedNLP methods if they are ever asked for
# TODO: There are todos in the code below
class AslNLP(ExtendedNLP):
    def __init__(self, nl_file):
        """
        Base class for NLP classes based on the Ampl Solver Library and 
        NL files.

        Parameters
        ----------
        nl_file : string
            filename of the NL-file containing the model
        """
        super(AslNLP, self).__init__()

        # nl file
        self._nl_file = nl_file

        # initialize the ampl interface
        self._asl = _asl.AmplInterface(self._nl_file)

        # collect the NLP structure and key data
        self._collect_nlp_structure()

        # create vectors to store the values for the primals and the duals
        # TODO: Check if we should initialize these to zero or from the init values
        self._primals = self._init_primals.copy()
        self._duals_full = self._init_duals_full.copy()
        self._duals_eq = self._init_duals_eq.copy()
        self._duals_ineq = self._init_duals_ineq.copy()
        self._obj_factor = 1.0
        self._cached_objective = None
        self._cached_grad_objective = self.create_new_vector('primals')
        self._cached_con_full = np.zeros(self._n_con_full, dtype=np.float64)
        self._cached_jac_full = coo_matrix((np.zeros(self._nnz_jac_full, dtype=np.float64),
                                               (self._irows_jac_full, self._jcols_jac_full)),
                                              shape=(self._n_con_full, self._n_primals))
        # these are only being cached for quicker copy of the matrix with the nonzero structure
        # TODO: only create these caches if the ExtendedNLP methods are asked for?
        self._cached_jac_eq = coo_matrix((np.zeros(self._nnz_jac_eq, dtype=np.float64),
                                               (self._irows_jac_eq, self._jcols_jac_eq)),
                                               shape=(self._n_con_eq, self._n_primals))
        self._cached_jac_ineq = coo_matrix((np.zeros(self._nnz_jac_ineq),
                                                 (self._irows_jac_ineq, self._jcols_jac_ineq)),
                                                shape=(self._n_con_ineq, self._n_primals))
        self._cached_hessian_lag = coo_matrix((np.zeros(self._nnz_hessian_lag, dtype=np.float64),
                                               (self._irows_hess, self._jcols_hess)),
                                              shape=(self._n_primals, self._n_primals))

        self._invalidate_primals_cache()
        self._invalidate_duals_cache()
        self._invalidate_obj_factor_cache()

    def _invalidate_primals_cache(self):
        self._objective_is_cached = False
        self._grad_objective_is_cached = False
        self._con_full_is_cached = False
        self._jac_full_is_cached = False
        self._hessian_lag_is_cached = False

    def _invalidate_duals_cache(self):
        self._hessian_lag_is_cached = False

    def _invalidate_obj_factor_cache(self):
        self._hessian_lag_is_cached = False

    def _collect_nlp_structure(self):
        """
        Collect characteristics of the NLP from the ASL interface
        """
        # get the problem dimensions
        self._n_primals = self._asl.get_n_vars()
        self._n_con_full = self._asl.get_n_constraints()
        self._nnz_jac_full = self._asl.get_nnz_jac_g()
        self._nnz_hess_lag_lower = self._asl.get_nnz_hessian_lag()

        # get the initial values for the primals 
        self._init_primals = np.zeros(self._n_primals, dtype=np.float64)
        self._init_duals_full = np.zeros(self._n_con_full, dtype=np.float64)
        self._asl.get_init_x(self._init_primals)
        self._asl.get_init_multipliers(self._init_duals_full)
        self._init_primals.flags.writeable = False
        self._init_duals_full.flags.writeable = False

        # get the bounds on the primal variables
        self._primals_lb = np.zeros(self._n_primals, dtype=np.float64)
        self._primals_ub = np.zeros(self._n_primals, dtype=np.float64)
        self._asl.get_x_lower_bounds(self._primals_lb)
        self._asl.get_x_upper_bounds(self._primals_ub)
        self._primals_lb.flags.writeable = False
        self._primals_ub.flags.writeable = False

        # get the bounds on the constraints (equality and
        # inequality are mixed in the ampl solver library)
        self._con_full_lb = np.zeros(self._n_con_full, dtype=np.float64)
        self._con_full_ub = np.zeros(self._n_con_full, dtype=np.float64)
        self._asl.get_g_lower_bounds(self._con_full_lb)
        self._asl.get_g_upper_bounds(self._con_full_ub)

        # check to make sure there are no fixed variables or crossed bounds
        # TODO: this tolerance should somehow be linked to the algorithm tolerance?
        # TODO: is the "fixed" check necessary?
        tolerance_fixed_bounds = 1e-8
        bounds_difference = self._primals_ub - self._primals_lb
        abs_bounds_difference = np.absolute(bounds_difference)
        fixed_vars = np.any(abs_bounds_difference < tolerance_fixed_bounds)
        if fixed_vars:
            print(np.where(abs_bounds_difference<tolerance_fixed_bounds))
            raise RuntimeError("Variables fixed using bounds is not currently supported.")

        inconsistent_bounds = np.any(bounds_difference < 0.0)
        if inconsistent_bounds:
            # TODO: improve error message
            raise RuntimeError("Variables found with upper bounds set below the lower bounds.")

        # Build the maps for converting from the full constraint
        # vector (which includes all equality and inequality constraints)
        # to separate vectors of equality and inequality constraints.
        self._build_constraint_maps()

        # get the values for the lower and upper bounds on the
        # inequalities (extracted from con_full)
        self._con_ineq_lb = np.compress(self._con_full_ineq_mask, self._con_full_lb)
        self._con_ineq_ub = np.compress(self._con_full_ineq_mask, self._con_full_ub)
        self._con_ineq_lb.flags.writeable = False
        self._con_ineq_ub.flags.writeable = False

        # get the initial values for the dual variables
        self._init_duals_eq = np.compress(self._con_full_eq_mask, self._init_duals_full)
        self._init_duals_ineq = np.compress(self._con_full_ineq_mask, self._init_duals_full)
        self._init_duals_eq.flags.writeable = False
        self._init_duals_ineq.flags.writeable = False

        # TODO: Should we be doing this or not?
        # adjust the rhs to be 0 for equality constraints (in both full and eq)
        self._con_full_rhs = self._con_full_ub.copy()
        # set the rhs to zero for the inequality constraints (will use lb, ub)
        self._con_full_rhs[~self._con_full_eq_mask] = 0.0
        # change the upper and lower bounds to zero for equality constraints
        self._con_full_lb[self._con_full_eq_mask] = 0.0
        self._con_full_ub[self._con_full_eq_mask] = 0.0
        self._con_full_lb.flags.writeable = False
        self._con_full_ub.flags.writeable = False
        
        # set number of equatity and inequality constraints from maps
        self._n_con_eq = len(self._con_eq_full_map)
        self._n_con_ineq = len(self._con_ineq_full_map)

        # populate jacobian structure
        self._irows_jac_full = np.zeros(self._nnz_jac_full, dtype=np.intc)
        self._jcols_jac_full = np.zeros(self._nnz_jac_full, dtype=np.intc)
        self._asl.struct_jac_g(self._irows_jac_full, self._jcols_jac_full)
        self._irows_jac_full -= 1
        self._jcols_jac_full -= 1
        self._irows_jac_full.flags.writeable = False
        self._jcols_jac_full.flags.writeable = False

        self._nz_con_full_eq_mask = np.isin(self._irows_jac_full, self._con_eq_full_map)
        self._nz_con_full_ineq_mask = np.logical_not(self._nz_con_full_eq_mask)
        self._irows_jac_eq = np.compress(self._nz_con_full_eq_mask, self._irows_jac_full)
        self._jcols_jac_eq = np.compress(self._nz_con_full_eq_mask, self._jcols_jac_full)
        self._irows_jac_ineq = np.compress(self._nz_con_full_ineq_mask, self._irows_jac_full)
        self._jcols_jac_ineq = np.compress(self._nz_con_full_ineq_mask, self._jcols_jac_full)
        self._nnz_jac_eq = len(self._irows_jac_eq)
        self._nnz_jac_ineq = len(self._irows_jac_ineq)

        # this is expensive but only done once - can we do this with numpy somehow?
        self._con_full_eq_map = full_eq_map = {self._con_eq_full_map[i]: i for i in range(self._n_con_eq)}
        for i, v in enumerate(self._irows_jac_eq):
            self._irows_jac_eq[i] = full_eq_map[v]

        self._con_full_ineq_map = full_ineq_map = {self._con_ineq_full_map[i]: i for i in range(self._n_con_ineq)}
        for i, v in enumerate(self._irows_jac_ineq):
            self._irows_jac_ineq[i] = full_ineq_map[v]

        self._irows_jac_eq.flags.writeable = False
        self._jcols_jac_eq.flags.writeable = False
        self._irows_jac_ineq.flags.writeable = False
        self._jcols_jac_ineq.flags.writeable = False

        # set nnz for equality and inequality jacobian
        self._nnz_jac_eq = len(self._jcols_jac_eq)
        self._nnz_jac_ineq = len(self._jcols_jac_ineq)

        # populate hessian structure (lower triangular)
        self._irows_hess = np.zeros(self._nnz_hess_lag_lower, dtype=np.intc)
        self._jcols_hess = np.zeros(self._nnz_hess_lag_lower, dtype=np.intc)
        self._asl.struct_hes_lag(self._irows_hess, self._jcols_hess)
        self._irows_hess -= 1
        self._jcols_hess -= 1

        # rework hessian to full matrix (lower and upper)
        diff = self._irows_hess - self._jcols_hess
        self._lower_hess_mask = np.where(diff != 0)
        lower = self._lower_hess_mask
        self._irows_hess = np.concatenate((self._irows_hess, self._jcols_hess[lower]))
        self._jcols_hess = np.concatenate((self._jcols_hess, self._irows_hess[lower]))
        self._nnz_hessian_lag = self._irows_hess.size

        self._irows_hess.flags.writeable = False
        self._jcols_hess.flags.writeable = False

    def _build_constraint_maps(self):
        """Creates internal maps and masks that convert from the full
        vector of constraints (the vector that includes all equality
        and inequality constraints combined) to separate vectors that
        include the equality and inequality constraints only.
        """
        # check the bounds on the constraints for crossing
        bounds_difference = self._con_full_ub - self._con_full_lb
        inconsistent_bounds = np.any(bounds_difference < 0.0)
        if inconsistent_bounds:
            raise RuntimeError("Bounds on range constraints found with upper bounds set below the lower bounds.")

        # build maps from con_full to con_eq and con_ineq
        abs_bounds_difference = np.absolute(bounds_difference)
        tolerance_equalities = 1e-8
        self._con_full_eq_mask = abs_bounds_difference < tolerance_equalities
        self._con_eq_full_map = self._con_full_eq_mask.nonzero()[0]
        self._con_full_ineq_mask = abs_bounds_difference >= tolerance_equalities
        self._con_ineq_full_map = self._con_full_ineq_mask.nonzero()[0]
        self._con_full_eq_mask.flags.writeable = False
        self._con_eq_full_map.flags.writeable = False
        self._con_full_ineq_mask.flags.writeable = False
        self._con_ineq_full_map.flags.writeable = False

        # these do not appear to be used anywhere - keeping the logic for now
        """
        #TODO: Can we simplify this logic?
        con_full_fulllb_mask = np.isfinite(self._con_full_lb) * self._con_full_ineq_mask + self._con_full_eq_mask
        con_fulllb_full_map = con_full_fulllb_mask.nonzero()[0]
        con_full_fullub_mask = np.isfinite(self._con_full_ub) * self._con_full_ineq_mask + self._con_full_eq_mask
        con_fullub_full_map = con_full_fullub_mask.nonzero()[0]

        self._ineq_lb_mask = np.isin(self._ineq_g_map, lb_g_map)
        self._lb_ineq_map = np.where(self._ineq_lb_mask)[0]
        self._ineq_ub_mask = np.isin(self._ineq_g_map, ub_g_map)
        self._ub_ineq_map = np.where(self._ineq_ub_mask)[0]
        self._ineq_lb_mask.flags.writeable = False
        self._lb_ineq_map.flags.writeable = False
        self._ineq_ub_mask.flags.writeable = False
        self._ub_ineq_map.flags.writeable = False
        """

    # overloaded from NLP
    def n_primals(self):
        return self._n_primals

    # overloaded from NLP
    def n_constraints(self):
        return self._n_con_full

    # overloaded from ExtendedNLP
    def n_eq_constraints(self):
        return self._n_con_eq

    # overloaded from ExtendedNLP
    def n_ineq_constraints(self):
        return self._n_con_ineq

    # overloaded from NLP
    def nnz_jacobian(self):
        return self._nnz_jac_full
    
    # overloaded from ExtendedNLP
    def nnz_jacobian_eq(self):
        return self._nnz_jac_eq

    # overloaded from ExtendedNLP
    def nnz_jacobian_ineq(self):
        return self._nnz_jac_ineq

    # overloaded from NLP
    def nnz_hessian_lag(self):
        return self._nnz_hessian_lag

    # overloaded from NLP
    def primals_lb(self):
        return self._primals_lb

    # overloaded from NLP
    def primals_ub(self):
        return self._primals_ub

    # overloaded from NLP
    def constraints_lb(self):
        return self._con_full_lb
    
    # overloaded from NLP
    def constraints_ub(self):
        return self._con_full_ub
    
    # overloaded from ExtendedNLP
    def ineq_lb(self):
        return self._con_ineq_lb

    # overloaded from ExtendedNLP
    def ineq_ub(self):
        return self._con_ineq_ub

    # overloaded from NLP
    def init_primals(self):
        return self._init_primals

    # overloaded from NLP
    def init_duals(self):
        return self._init_duals_full

    # overloaded from ExtendedNLP
    def init_duals_eq(self):
        return self._init_duals_eq

    # overloaded from ExtendedNLP
    def init_duals_ineq(self):
        return self._init_duals_ineq

    # overloaded from NLP / Extended NLP
    def create_new_vector(self, vector_type):
        """
        Creates a vector of the appropriate length and structure as 
        requested

        Parameters
        ----------
        vector_type: {'primals', 'constraints', 'eq_constraints', 'ineq_constraints',
                      'duals', 'duals_eq', 'duals_ineq'}
            String identifying the appropriate  vector  to create.

        Returns
        -------
        numpy.ndarray
        """
        if vector_type == 'primals':
            return np.zeros(self.n_primals(), dtype=np.float64)
        elif vector_type == 'constraints' or vector_type == 'duals':
            return np.zeros(self.n_constraints(), dtype=np.float64)
        elif vector_type == 'eq_constraints' or vector_type == 'duals_eq':
            return np.zeros(self.n_eq_constraints(), dtype=np.float64)
        elif vector_type == 'ineq_constraints' or vector_type == 'duals_ineq':
            return np.zeros(self.n_ineq_constraints(), dtype=np.float64)
        else:
            raise RuntimeError('Called create_new_vector with an unknown vector_type')

    # overloaded from NLP
    def set_primals(self, primals):
        self._invalidate_primals_cache()
        np.copyto(self._primals, primals)

    # overloaded from NLP
    def get_primals(self):
        return  self._primals.copy()

    # overloaded from NLP
    def set_duals(self, duals):
        self._invalidate_duals_cache()
        np.copyto(self._duals_full, duals)
        # keep the separated duals up to date just in case
        np.compress(self._con_full_eq_mask, self._duals_full, out=self._duals_eq)
        np.compress(self._con_full_ineq_mask, self._duals_full, out=self._duals_ineq)

    # overloaded from NLP
    def get_duals(self):
        return self._duals_full.copy()

    # overloaded from NLP
    def set_obj_factor(self, obj_factor):
        self._invalidate_obj_factor_cache()
        self._obj_factor = obj_factor

    # overloaded from NLP
    def get_obj_factor(self):
        return self._obj_factor
    
    # overloaded from ExtendedNLP
    def set_duals_eq(self, duals_eq):
        self._invalidate_duals_cache()
        np.copyto(self._duals_eq, duals_eq)
        # keep duals_full up to date just in case
        self._duals_full[self._con_full_eq_mask] = self._duals_eq

    # overloaded from ExtendedNLP
    def get_duals_eq(self):
        return self._duals_eq.copy()

    # overloaded from ExtendedNLP
    def set_duals_ineq(self, duals_ineq):
        self._invalidate_duals_cache()
        np.copyto(self._duals_ineq, duals_ineq)
        # keep duals_full up to date just in case
        self._duals_full[self._con_full_ineq_mask] = self._duals_ineq

    # overloaded from ExtendedNLP
    def get_duals_ineq(self):
        return self._duals_ineq.copy()

    # overloaded from NLP
    def get_obj_scaling(self):
        return None

    # overloaded from NLP - derived classes may implement
    def get_primals_scaling(self):
        return None

    # overloaded from NLP - derived classes may implement
    def get_constraints_scaling(self):
        return None

    # overloaded from ExtendedNLP
    def get_eq_constraints_scaling(self):
        constraints_scaling = self.get_constraints_scaling()
        if constraints_scaling is not None:
            return np.compress(self._con_full_eq_mask,
                               constraints_scaling)
        return None

    # overloaded from ExtendedNLP
    def get_ineq_constraints_scaling(self):
        constraints_scaling = self.get_constraints_scaling()
        if constraints_scaling is not None:
            return np.compress(self._con_full_ineq_mask,
                               constraints_scaling)
        return None

    def _evaluate_objective_and_cache_if_necessary(self):
        if not self._objective_is_cached:
            self._cached_objective = self._asl.eval_f(self._primals)
            self._objective_is_cached = True

    # overloaded from NLP
    def evaluate_objective(self):
        self._evaluate_objective_and_cache_if_necessary()
        return self._cached_objective

    # overloaded from NLP
    def evaluate_grad_objective(self, out=None):
        if not self._grad_objective_is_cached:
            self._asl.eval_deriv_f(self._primals, self._cached_grad_objective)
            self._grad_objective_is_cached = True

        if out is not None:
            if not isinstance(out, np.ndarray) or out.size != self._n_primals:
                raise RuntimeError('Called evaluate_grad_objective with an invalid "out" argument - should take an ndarray of size {}'.format(self._n_primals))
            np.copyto(out, self._cached_grad_objective)
            return out
        else:
            return self._cached_grad_objective.copy()

    def _evaluate_constraints_and_cache_if_necessary(self):
        # ASL computes the full constraint vector, therefore, we merge
        # this computation into one
        if not self._con_full_is_cached:
            self._asl.eval_g(self._primals, self._cached_con_full)
            self._cached_con_full -= self._con_full_rhs
            self._con_full_is_cached = True

    # overloaded from NLP
    def evaluate_constraints(self, out=None):
        self._evaluate_constraints_and_cache_if_necessary()

        if out is not None:
            if not isinstance(out, np.ndarray) or out.size != self._n_con_full:
                raise RuntimeError('Called evaluate_constraints with an invalid'
                                   ' "out" argument - should take an ndarray of '
                                   'size {}'.format(self._n_con_full))
            np.copyto(out, self._cached_con_full)
            return out
        else:
            return self._cached_con_full.copy()

    # overloaded from ExtendedNLP
    def evaluate_eq_constraints(self, out=None):
        self._evaluate_constraints_and_cache_if_necessary()

        if out is not None:
            if not isinstance(out, np.ndarray) or out.size != self._n_con_eq:
                raise RuntimeError('Called evaluate_eq_constraints with an invalid'
                                   ' "out" argument - should take an ndarray of '
                                   'size {}'.format(self._n_con_eq))
            self._cached_con_full.compress(self._con_full_eq_mask, out=out)
            return  out
        else:
            return self._cached_con_full.compress(self._con_full_eq_mask)

    # overloaded from ExtendedNLP
    def evaluate_ineq_constraints(self, out=None):
        self._evaluate_constraints_and_cache_if_necessary()

        if out is not None:
            if not isinstance(out, np.ndarray) or out.size != self._n_con_ineq:
                raise RuntimeError('Called evaluate_ineq_constraints with an invalid'
                                   ' "out" argument - should take an ndarray of '
                                   'size {}'.format(self._n_con_ineq))
            self._cached_con_full.compress(self._con_full_ineq_mask, out=out)
            return out
        else:
            return self._cached_con_full.compress(self._con_full_ineq_mask)

    def _evaluate_jacobians_and_cache_if_necessary(self):
        # ASL computes the jacobian for the full constraints, therefore, we merge
        # this computation into one
        if not self._jac_full_is_cached:
            self._asl.eval_jac_g(self._primals, self._cached_jac_full.data)
            self._jac_full_is_cached = True

    # overloaded from NLP
    def evaluate_jacobian(self, out=None):
        self._evaluate_jacobians_and_cache_if_necessary()

        if out is not None:
            if not isinstance(out, coo_matrix) \
               or out.shape[0] != self._n_con_full \
               or out.shape[1] != self._n_primals \
               or out.nnz != self._nnz_jac_full:
                raise RuntimeError('evaluate_jacobian called with an "out" argument'
                                   ' that is invalid. This should be a coo_matrix with'
                                   ' shape=({},{}) and nnz={}'
                                   .format(self._n_con_full, self._n_primals, self._nnz_jac_full))
            np.copyto(out.data, self._cached_jac_full.data)
            return out
        else:
            return self._cached_jac_full.copy()

    # overloaded from ExtendedNLP
    def evaluate_jacobian_eq(self, out=None):
        self._evaluate_jacobians_and_cache_if_necessary()

        if out is not None:
            if not isinstance(out, coo_matrix) \
               or out.shape[0] != self._n_con_eq \
               or out.shape[1] != self._n_primals \
               or out.nnz != self._nnz_jac_eq:
                raise RuntimeError('evaluate_jacobian_eq called with an "out" argument'
                                   ' that is invalid. This should be a coo_matrix with'
                                   ' shape=({},{}) and nnz={}'
                                   .format(self._n_con_eq, self._n_primals, self._nnz_jac_eq))
            
            self._cached_jac_full.data.compress(self._nz_con_full_eq_mask, out=out.data)
            return out
        else:
            self._cached_jac_full.data.compress(self._nz_con_full_eq_mask, out=self._cached_jac_eq.data)
            return self._cached_jac_eq.copy()

    # overloaded from NLP
    def evaluate_jacobian_ineq(self, out=None):
        self._evaluate_jacobians_and_cache_if_necessary()

        if out is not None:
            if not isinstance(out, coo_matrix) \
               or out.shape[0] != self._n_con_ineq \
               or out.shape[1] != self._n_primals \
               or out.nnz != self._nnz_jac_ineq:
                raise RuntimeError('evaluate_jacobian_ineq called with an "out" argument'
                                   ' that is invalid. This should be a coo_matrix with'
                                   ' shape=({},{}) and nnz={}'
                                   .format(self._n_con_ineq, self._n_primals, self._nnz_jac_ineq))
            
            self._cached_jac_full.data.compress(self._nz_con_full_ineq_mask, out=out.data)
            return out
        else:
            self._cached_jac_full.data.compress(self._nz_con_full_ineq_mask, out=self._cached_jac_ineq.data)
            return self._cached_jac_ineq.copy()

    def evaluate_hessian_lag(self, out=None):
        if not self._hessian_lag_is_cached:
            # evaluating the hessian requires that we have first
            # evaluated the objective and the constraints
            self._evaluate_objective_and_cache_if_necessary()
            self._evaluate_constraints_and_cache_if_necessary()

            # get the hessian
            data = np.zeros(self._nnz_hess_lag_lower, np.float64)
            self._asl.eval_hes_lag(self._primals, self._duals_full,
                                   data, obj_factor=self._obj_factor)
            values = np.concatenate((data, data[self._lower_hess_mask]))
            #TODO: find out why this is done
            values += 1e-16 # this is to deal with scipy bug temporarily
            np.copyto(self._cached_hessian_lag.data, values)
            self._hessian_lag_is_cached = True

        if out is not None:
            if not isinstance(out, coo_matrix) or out.shape[0] != self._n_primals or \
               out.shape[1] != self._n_primals or out.nnz != self._nnz_hessian_lag:
                raise RuntimeError('evaluate_hessian_lag called with an "out" argument'
                                   ' that is invalid. This should be a coo_matrix with'
                                   ' shape=({},{}) adn nnz={}'
                                   .format(self._n_primals, self._n_primals, self._nnz_hessian_lag))
            np.copyto(out.data, self._cached_hessian_lag.data)
            return out
        else:
            return self._cached_hessian_lag.copy()

    def report_solver_status(self, status_code, status_message):
        self._asl.finalize_solution(status_code, status_message, self._primals, self._duals)

class AmplNLP(AslNLP):
    def __init__(self, nl_file, row_filename=None, col_filename=None):
        """
        AMPL nonlinear program interface.
        If row_filename and col_filename are not provided, the interface
        will see if files exist (with same name as nl_file but the .row 
        and .col extensions)

        Parameters
        ----------
        nl_file: str
            filename of the NL-file containing the model
        row_filename: str, optional
            filename of .row file with identity of constraints
        col_filename: str, optional
            filename of .col file with identity of variables

        """
        # call parent class to set the nl file name and load the model
        super(AmplNLP, self).__init__(nl_file)

        # check for the existence of the row / col files
        if row_filename is None:
            tmp_filename = os.path.splitext(nl_file)[0] + '.row'
            if os.path.isfile(tmp_filename):
                row_filename = tmp_filename

        if col_filename is None:
            tmp_filename = os.path.splitext(nl_file)[0] + '.col'
            if os.path.isfile(tmp_filename):
                col_filename = tmp_filename

        self._rowfile = row_filename
        self._colfile = col_filename

        # create containers with names of variables
        self._vidx_to_name = None
        self._name_to_vidx = None
        if col_filename is not None:
            self._vidx_to_name = self._build_component_names_list(col_filename)
            self._name_to_vidx = {self._vidx_to_name[vidx]: vidx for vidx in range(self._n_primals)}

        # create containers with names of constraints and objective
        self._con_full_idx_to_name = None
        self._name_to_con_full_idx = None
        self._obj_name = None
        if row_filename is not None:
            all_names = self._build_component_names_list(row_filename)
            # objective is the last one in the list
            # TODO: what happens with multiple objectives?
            self._obj_name = all_names[-1]
            del all_names[-1]
            self._con_full_idx_to_name = all_names
            self._con_eq_idx_to_name = [all_names[self._con_eq_full_map[i]] for i in range(self._n_con_eq)]
            self._con_ineq_idx_to_name = [all_names[self._con_ineq_full_map[i]] for i in range(self._n_con_ineq)]
            self._name_to_con_full_idx = {all_names[cidx]: cidx for cidx in range(self._n_con_full)}
            self._name_to_con_eq_idx = {name:idx for idx,name in enumerate(self._con_eq_idx_to_name)}
            self._name_to_con_ineq_idx = {name:idx for idx,name in enumerate(self._con_ineq_idx_to_name)}

            
    def variable_names(self):
        """Returns ordered list with names of primal variables"""
        return list(self._vidx_to_name)

    def constraint_names(self):
        """Returns an ordered list with the names of all the constraints
        (corresponding to evaluate_constraints)"""
        return list(self._con_full_idx_to_name)

    def eq_constraint_names(self):
        """Returns ordered list with names of equality constraints only
        (corresponding to evaluate_eq_constraints)"""
        return list(self._con_eq_idx_to_name)

    def ineq_constraint_names(self):
        """Returns ordered list with names of inequality constraints only
        (corresponding to evaluate_ineq_constraints)"""
        return list(self._con_ineq_idx_to_name)

    def variable_idx(self, var_name):
        """
        Returns the index of the variable named var_name

        Parameters
        ----------
        var_name: str
            Name of variable

        Returns
        -------
        int

        """
        return self._name_to_vidx[var_name]

    def constraint_idx(self, con_name):
        """
        Returns the index of the constraint named con_name
        (corresponding to the order returned by evaluate_constraints)

        Parameters
        ----------
        con_name: str
            Name of constraint

        Returns
        -------
        int
        """
        return self._name_to_con_full_idx[con_name]

    def eq_constraint_idx(self, con_name):
        """
        Returns the index of the equality constraint named con_name
        (corresponding to the order returned by evaluate_eq_constraints)

        Parameters
        ----------
        con_name: str
            Name of constraint

        Returns
        -------
        int

        """
        return self._name_to_con_eq_idx[con_name]

    def ineq_constraint_idx(self, con_name):
        """
        Returns the index of the inequality constraint named con_name
        (corresponding to the order returned by evaluate_ineq_constraints)

        Parameters
        ----------
        con_name: str
            Name of constraint

        Returns
        -------
        int

        """
        return self._name_to_con_ineq_idx[con_name]

    @staticmethod
    def _build_component_names_list(filename):
        """ Builds an ordered list of strings from a file 
        containing strings on separate lines (e.g., the row
        and col files """
        ordered_names = list()
        with open(filename, 'r') as f:
            for line in f:
                ordered_names.append(line.strip('\n'))
        return ordered_names
