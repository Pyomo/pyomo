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

import os
import numpy as np
import six

from scipy.sparse import coo_matrix
from pyomo.common.deprecation import deprecated
import pyomo.core.base as pyo
from pyomo.common.collections import ComponentMap
from ..sparse.block_matrix import BlockMatrix
from pyomo.contrib.pynumero.interfaces.nlp import NLP
from .external_grey_box import ExternalGreyBoxBlock


class PyomoGreyBoxNLP(NLP):
    def __init__(self, pyomo_model):
        # store all the greybox custom block data objects
        greybox_components = []
        try:
            # We support Pynumero's ExternalGreyBoxBlock modeling
            # objects.  We need to find them and convert them to Blocks
            # before calling the NL writer so that the attached Vars get
            # picked up by the writer.
            for greybox in pyomo_model.component_objects(
                    ExternalGreyBoxBlock, descend_into=True):
                greybox.parent_block().reclassify_component_type(
                    greybox, pyo.Block)
                greybox_components.append(greybox)

            self._pyomo_model = pyomo_model
            self._pyomo_nlp = PyomoNLP(pyomo_model)

        finally:
            # Restore the ctypes of the ExternalGreyBoxBlock components
            for greybox in greybox_components:
                greybox.parent_block().reclassify_component_type(
                    greybox, ExternalGreyBoxBlock)

        # get the greybox block data objects
        greybox_data = []
        for greybox in greybox_components:
            greybox_data.extend(data for data in greybox.values()
                                if data.active)

        if len(greybox_data) > 1:
            raise NotImplementedError("The PyomoGreyBoxModel interface has not"
                                      " been tested with Pyomo models that contain"
                                      " more than one ExternalGreyBoxBlock. Currently,"
                                      " only a single block is supported.")

        if self._pyomo_nlp.n_primals() == 0:
            raise ValueError(
                "No variables were found in the Pyomo part of the model."
                " PyomoGreyBoxModel requires at least one variable"
                " to be active in a Pyomo objective or constraint")

        # number of additional variables required - they are in the
        # greybox models but not included in the NL file
        self._n_greybox_primals = 0

        # number of residuals (equality constraints + output constraints
        # coming from the grey box models
        self._greybox_primals_names = []
        self._greybox_constraints_names = []

        # Update the primal index map with any variables in the
        # greybox models that do not otherwise appear in the NL
        # and capture some other book keeping items
        n_primals = self._pyomo_nlp.n_primals()
        greybox_primals = []
        self._vardata_to_idx = ComponentMap(self._pyomo_nlp._vardata_to_idx)
        for data in greybox_data:
            # check that none of the inputs / outputs are fixed
            for v in six.itervalues(data.inputs):
                if v.fixed:
                    raise NotImplementedError('Found a grey box model input that is fixed: {}.'
                                              ' This interface does not currently support fixed'
                                              ' variables. Please add a constraint instead.'
                                              ''.format(v.getname(fully_qualified=True)))
            for v in six.itervalues(data.outputs):
                if v.fixed:
                    raise NotImplementedError('Found a grey box model output that is fixed: {}.'
                                              ' This interface does not currently support fixed'
                                              ' variables. Please add a constraint instead.'
                                              ''.format(v.getname(fully_qualified=True)))

            block_name = data.getname()
            for nm in data._ex_model.equality_constraint_names():
                self._greybox_constraints_names.append('{}.{}'.format(block_name, nm))
            for nm in data._ex_model.output_names():
                self._greybox_constraints_names.append('{}.{}_con'.format(block_name, nm))

            for var in data.component_data_objects(pyo.Var):
                if var not in self._vardata_to_idx:
                    # there is a variable in the greybox block that
                    # is not in the NL - append this to the end
                    self._vardata_to_idx[var] = n_primals
                    n_primals += 1
                    greybox_primals.append(var)
                    self._greybox_primals_names.append(var.getname(fully_qualified=True))
        self._n_greybox_primals = len(greybox_primals)
        self._greybox_primal_variables = greybox_primals

        # Configure the primal and dual data caches
        self._greybox_primals_lb = np.zeros(self._n_greybox_primals)
        self._greybox_primals_ub = np.zeros(self._n_greybox_primals)
        self._init_greybox_primals = np.zeros(self._n_greybox_primals)
        for i, var in enumerate(greybox_primals):
            if var.value is not None:
                self._init_greybox_primals[i] = var.value
            self._greybox_primals_lb[i] = -np.inf if var.lb is None else var.lb
            self._greybox_primals_ub[i] = np.inf if var.ub is None else var.ub
        self._greybox_primals_lb.flags.writeable = False
        self._greybox_primals_ub.flags.writeable = False

        self._greybox_primals = self._init_greybox_primals.copy()

        # data member to store the cached greybox constraints and jacobian
        self._cached_greybox_constraints = None
        self._cached_greybox_jac = None

        # Now that we know the total number of columns, create the
        # necessary greybox helper objects
        self._external_greybox_helpers = \
            [_ExternalGreyBoxModelHelper(data, self._vardata_to_idx, self.init_primals()) for data in greybox_data]

        # make sure the primal values get to the greybox models
        self.set_primals(self.get_primals())

        self._n_greybox_constraints = 0
        for h in self._external_greybox_helpers:
            self._n_greybox_constraints += h.n_residuals()
        assert len(self._greybox_constraints_names) == self._n_greybox_constraints

        # If any part of the problem is scaled (i.e., obj, primals,
        # or any of the constraints for any of the external grey boxes),
        # then we want scaling factors for everything (defaulting to
        # ones for any of the missing factors).
        # This code builds all the scaling factors, with the defaults,
        # but then sets them all to None if *no* scaling factors are provided
        # for any of the pieces. (inefficient, but only done once)
        need_scaling = False
        self._obj_scaling = self._pyomo_nlp.get_obj_scaling()
        if self._obj_scaling is None:
            self._obj_scaling = 1.0
        else:
            need_scaling = True

        self._primals_scaling = np.ones(self.n_primals())
        scaling_suffix = self._pyomo_nlp._pyomo_model.component('scaling_factor')
        if scaling_suffix and scaling_suffix.ctype is pyo.Suffix:
            need_scaling = True
            for i,v in enumerate(self.get_pyomo_variables()):
                if v in scaling_suffix:
                    self._primals_scaling[i] = scaling_suffix[v]

        self._constraints_scaling = []
        pyomo_nlp_scaling = self._pyomo_nlp.get_constraints_scaling()
        if pyomo_nlp_scaling is None:
            pyomo_nlp_scaling = np.ones(self._pyomo_nlp.n_constraints())
        else:
            need_scaling = True
        self._constraints_scaling.append(pyomo_nlp_scaling)

        for h in self._external_greybox_helpers:
            tmp_scaling = h.get_residual_scaling()
            if tmp_scaling is None:
                tmp_scaling = np.ones(h.n_residuals())
            else:
                need_scaling = True
            self._constraints_scaling.append(tmp_scaling)

        if need_scaling:
            self._constraints_scaling = np.concatenate(self._constraints_scaling)
        else:
            self._obj_scaling = None
            self._primals_scaling = None
            self._constraints_scaling = None

        # might want the user to be able to specify these at some point
        self._init_greybox_duals = np.ones(self._n_greybox_constraints)
        self._init_greybox_primals.flags.writeable = False
        self._init_greybox_duals.flags.writeable = False
        self._greybox_duals = self._init_greybox_duals.copy()

        # compute the jacobian for the external greybox models
        # to get some of the statistics
        self._evaluate_greybox_jacobians_and_cache_if_necessary()
        self._nnz_greybox_jac = len(self._cached_greybox_jac.data)

    def _invalidate_greybox_primals_cache(self):
        self._greybox_constraints_cached = False
        self._greybox_jac_cached = False

    # overloaded from NLP
    def n_primals(self):
        return self._pyomo_nlp.n_primals() + self._n_greybox_primals

    # overloaded from NLP
    def n_constraints(self):
        return self._pyomo_nlp.n_constraints() + self._n_greybox_constraints

    # overloaded from ExtendedNLP
    def n_eq_constraints(self):
        return self._pyomo_nlp.n_eq_constraints() + self._n_greybox_constraints

    # overloaded from ExtendedNLP
    def n_ineq_constraints(self):
        return self._pyomo_nlp.n_ineq_constraints()

    # overloaded from NLP
    def nnz_jacobian(self):
        return self._pyomo_nlp.nnz_jacobian() + self._nnz_greybox_jac

    # overloaded from AslNLP
    def nnz_jacobian_eq(self):
        return self._pyomo_nlp.nnz_jacobian_eq() + self._nnz_greybox_jac

    # overloaded from NLP
    def nnz_hessian_lag(self):
        raise NotImplementedError(
            "PyomoGreyBoxNLP does not currently support Hessians")

    # overloaded from NLP
    def primals_lb(self):
        return np.concatenate((self._pyomo_nlp.primals_lb(),
            self._greybox_primals_lb,
        ))

    # overloaded from NLP
    def primals_ub(self):
        return np.concatenate((
            self._pyomo_nlp.primals_ub(),
            self._greybox_primals_ub,
        ))

    # overloaded from NLP
    def constraints_lb(self):
        return np.concatenate((
            self._pyomo_nlp.constraints_lb(),
            np.zeros(self._n_greybox_constraints, dtype=np.float64),
        ))

    # overloaded from NLP
    def constraints_ub(self):
        return np.concatenate((
            self._pyomo_nlp.constraints_ub(),
            np.zeros(self._n_greybox_constraints, dtype=np.float64),
        ))

    # overloaded from NLP
    def init_primals(self):
        return np.concatenate((
            self._pyomo_nlp.init_primals(),
            self._init_greybox_primals,
        ))

    # overloaded from NLP
    def init_duals(self):
        return np.concatenate((
            self._pyomo_nlp.init_duals(),
            self._init_greybox_duals,
        ))

    # overloaded from ExtendedNLP
    def init_duals_eq(self):
        return np.concatenate((
            self._pyomo_nlp.init_duals_eq(),
            self._init_greybox_duals,
        ))

    # overloaded from NLP / Extended NLP
    def create_new_vector(self, vector_type):
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
        self._invalidate_greybox_primals_cache()

        # set the primals on the "pyomo" part of the nlp
        self._pyomo_nlp.set_primals(
            primals[:self._pyomo_nlp.n_primals()])

        # copy the values for the greybox primals
        np.copyto(self._greybox_primals, primals[self._pyomo_nlp.n_primals():])

        for external in self._external_greybox_helpers:
            external.set_primals(primals)

    # overloaded from AslNLP
    def get_primals(self):
        # return the value of the primals that the pyomo
        # part knows about as well as any extra values that
        # are only in the greybox part
        return np.concatenate((
            self._pyomo_nlp.get_primals(),
            self._greybox_primals,
        ))

    # overloaded from NLP
    def set_duals(self, duals):
        #self._invalidate_greybox_duals_cache()

        # set the duals for the pyomo part of the nlp
        self._pyomo_nlp.set_duals(
            duals[:-self._n_greybox_constraints])

        # set the duals for the greybox part of the nlp
        np.copyto(self._greybox_duals, duals[-self._n_greybox_constraints:])

    # overloaded from NLP
    def get_duals(self):
        # return the duals for the pyomo part of the nlp
        # concatenated with the greybox part
        return np.concatenate((
            self._pyomo_nlp.get_duals(),
            self._greybox_duals,
        ))

    # overloaded from ExtendedNLP
    def set_duals_eq(self, duals):
        #self._invalidate_greybox_duals_cache()

        # set the duals for the pyomo part of the nlp
        self._pyomo_nlp.set_duals_eq(
            duals[:-self._n_greybox_constraints])

        # set the duals for the greybox part of the nlp
        np.copyto(self._greybox_duals, duals[-self._n_greybox_constraints:])

    # overloaded from NLP
    def get_duals_eq(self):
        # return the duals for the pyomo part of the nlp
        # concatenated with the greybox part
        return np.concatenate((
            self._pyomo_nlp.get_duals_eq(),
            self._greybox_duals,
        ))

    # overloaded from NLP
    def set_obj_factor(self, obj_factor):
        # objective is owned by the pyomo model
        self._pyomo_nlp.set_obj_factor(obj_factor)

    # overloaded from NLP
    def get_obj_factor(self):
        # objective is owned by the pyomo model
        return self._pyomo_nlp.get_obj_factor()

    # overloaded from NLP
    def get_obj_scaling(self):
        return self._obj_scaling

    # overloaded from NLP
    def get_primals_scaling(self):
        return self._primals_scaling

    # overloaded from NLP
    def get_constraints_scaling(self):
        return self._constraints_scaling

    # overloaded from NLP
    def evaluate_objective(self):
        # objective is owned by the pyomo model
        return self._pyomo_nlp.evaluate_objective()

    # overloaded from NLP
    def evaluate_grad_objective(self, out=None):
        # objective is owned by the pyomo model
        return np.concatenate((
            self._pyomo_nlp.evaluate_grad_objective(out),
            np.zeros(self._n_greybox_primals)))

    def _evaluate_greybox_constraints_and_cache_if_necessary(self):
        if self._greybox_constraints_cached:
            return

        self._cached_greybox_constraints = np.concatenate(tuple(
            external.evaluate_residuals()
            for external in self._external_greybox_helpers))
        self._greybox_constraints_cached = True

    # overloaded from NLP
    def evaluate_constraints(self, out=None):
        self._evaluate_greybox_constraints_and_cache_if_necessary()

        if out is not None:
            if not isinstance(out, np.ndarray) \
               or out.size != self.n_constraints():
                raise RuntimeError(
                    'Called evaluate_constraints with an invalid'
                    ' "out" argument - should take an ndarray of '
                    'size {}'.format(self.n_constraints()))

            # call on the pyomo part of the nlp
            self._pyomo_nlp.evaluate_constraints(
                out[:-self._n_greybox_constraints])

            # call on the greybox part of the nlp
            np.copyto(out[-self._n_greybox_constraints:],
                      self._cached_greybox_constraints)
            return out

        else:
            # concatenate the pyomo and external constraint residuals
            return np.concatenate((
                self._pyomo_nlp.evaluate_constraints(),
                self._cached_greybox_constraints,
            ))

    # overloaded from ExtendedNLP
    def evaluate_eq_constraints(self, out=None):
        self._evaluate_greybox_constraints_and_cache_if_necessary()

        if out is not None:
            if not isinstance(out, np.ndarray) \
               or out.size != self.n_eq_constraints():
                raise RuntimeError(
                    'Called evaluate_eq_constraints with an invalid'
                    ' "out" argument - should take an ndarray of '
                    'size {}'.format(self.n_eq_constraints()))
            self._pyomo_nlp.evaluate_eq_constraints(
                out[:-self._n_greybox_constraints])
            np.copyto(out[-self._n_greybox_constraints:], self._cached_greybox_constraints)
            return out
        else:
            return np.concatenate((
                self._pyomo_nlp.evaluate_eq_constraints(),
                self._cached_greybox_constraints,
            ))

    def _evaluate_greybox_jacobians_and_cache_if_necessary(self):
        if self._greybox_jac_cached:
            return

        jac = BlockMatrix(len(self._external_greybox_helpers), 1)
        for i, external in enumerate(self._external_greybox_helpers):
            jac.set_block(i, 0, external.evaluate_jacobian())
        self._cached_greybox_jac = jac.tocoo()
        self._greybox_jac_cached = True

    # overloaded from NLP
    def evaluate_jacobian(self, out=None):
        self._evaluate_greybox_jacobians_and_cache_if_necessary()

        if out is not None:
            if ( not isinstance(out, coo_matrix)
                 or out.shape[0] != self.n_constraints()
                 or out.shape[1] != self.n_primals()
                 or out.nnz != self.nnz_jacobian() ):
                raise RuntimeError(
                    'evaluate_jacobian called with an "out" argument'
                    ' that is invalid. This should be a coo_matrix with'
                    ' shape=({},{}) and nnz={}'
                    .format(self.n_constraints(), self.n_primals(),
                            self.nnz_jacobian()))
            n_pyomo_constraints = self.n_constraints() - self._n_greybox_constraints
            self._pyomo_nlp.evaluate_jacobian(
                out=coo_matrix((out.data[:-self._nnz_greybox_jac],
                                (out.row[:-self._nnz_greybox_jac],
                                 out.col[:-self._nnz_greybox_jac])),
                               shape=(n_pyomo_constraints, self._pyomo_nlp.n_primals())))
            np.copyto(out.data[-self._nnz_greybox_jac:],
                      self._cached_greybox_jac.data)
            return out
        else:
            base = self._pyomo_nlp.evaluate_jacobian()
            base = coo_matrix((base.data, (base.row, base.col)),
                              shape=(base.shape[0], self.n_primals()))

            jac = BlockMatrix(2,1)
            jac.set_block(0, 0, base)
            jac.set_block(1, 0, self._cached_greybox_jac)
            return jac.tocoo()

            # TODO: Doesn't this need a "shape" specification?
            #return coo_matrix((
            #    np.concatenate((base.data, self._cached_greybox_jac.data)),
            #    ( np.concatenate((base.row, self._cached_greybox_jac.row)),
            #      np.concatenate((base.col, self._cached_greybox_jac.col)) )
            #))

    # overloaded from ExtendedNLP
    """
    def evaluate_jacobian_eq(self, out=None):
        raise NotImplementedError()
        self._evaluate_greybox_jacobians_and_cache_if_necessary()

        if out is not None:
            if ( not isinstance(out, coo_matrix)
                 or out.shape[0] != self.n_eq_constraints()
                 or out.shape[1] != self.n_primals()
                 or out.nnz != self.nnz_jacobian_eq() ):
                raise RuntimeError(
                    'evaluate_jacobian called with an "out" argument'
                    ' that is invalid. This should be a coo_matrix with'
                    ' shape=({},{}) and nnz={}'
                    .format(self.n_eq_constraints(), self.n_primals(),
                            self.nnz_jacobian_eq()))
            self._pyomo_nlp.evaluate_jacobian_eq(
                coo_matrix((out.data[:-self._nnz_greybox_jac],
                            (out.row[:-self._nnz_greybox_jac],
                             out.col[:-self._nnz_greybox_jac])))
            )
            np.copyto(out.data[-self._nnz_greybox_jac],
                      self._cached_greybox_jac.data)
            return out
        else:
            base = self._pyomo_nlp.evaluate_jacobian_eq()
            # TODO: Doesn't this need a "shape" specification?
            return coo_matrix((
                np.concatenate((base.data, self._cached_greybox_jac.data)),
                ( np.concatenate((base.row, self._cached_greybox_jac.row)),
                  np.concatenate((base.col, self._cached_greybox_jac.col)) )
            ))
    """
    # overloaded from NLP
    def evaluate_hessian_lag(self, out=None):
        # return coo_matrix(([], ([],[])), shape=(self.n_primals(), self.n_primals()))
        raise NotImplementedError(
            "PyomoGreyBoxNLP does not currently support Hessians")

    # overloaded from NLP
    def report_solver_status(self, status_code, status_message):
        raise NotImplementedError('Todo: implement this')

    @deprecated(msg='This method has been replaced with primals_names', version='', remove_in='6.0')
    def variable_names(self):
        return self.primals_names()

    def primals_names(self):
        names = list(self._pyomo_nlp.variable_names())
        names.extend(self._greybox_primals_names)
        return names

    def constraint_names(self):
        names = list(self._pyomo_nlp.constraint_names())
        names.extend(self._greybox_constraints_names)
        return names

    def pyomo_model(self):
        """
        Return optimization model
        """
        return self._pyomo_model

    def get_pyomo_objective(self):
        """
        Return an instance of the active objective function on the Pyomo model.
        (there can be only one)
        """
        return self._pyomo_nlp.get_pyomo_objective()

    def get_pyomo_variables(self):
        """
        Return an ordered list of the Pyomo VarData objects in
        the order corresponding to the primals
        """
        return self._pyomo_nlp.get_pyomo_variables() + \
            self._greybox_primal_variables

    def get_pyomo_constraints(self):
        """
        Return an ordered list of the Pyomo ConData objects in
        the order corresponding to the primals
        """
        # FIXME: what do we return for the external block constraints?
        # return self._pyomo_nlp.get_pyomo_constraints()
        raise NotImplementedError(
            "returning list of all constraints when using an external "
            "model is TBD")

    def load_state_into_pyomo(self, bound_multipliers=None):
        primals = self.get_primals()
        variables = self.get_pyomo_variables()
        for var, val in zip(variables, primals):
            var.set_value(val)
        m = self.pyomo_model()
        model_suffixes = dict(
            pyo.suffix.active_import_suffix_generator(m))
        if 'dual' in model_suffixes:
            model_suffixes['dual'].clear()
            # Until we sort out how to return the duals for the external
            # block (implied) constraints, I am disabling *all* duals
            #
            # duals = self.get_duals()
            # constraints = self.get_pyomo_constraints()
            # model_suffixes['dual'].update(
            #     zip(constraints, duals))
        if 'ipopt_zL_out' in model_suffixes:
            model_suffixes['ipopt_zL_out'].clear()
            if bound_multipliers is not None:
                model_suffixes['ipopt_zL_out'].update(
                    zip(variables, bound_multipliers[0]))
        if 'ipopt_zU_out' in model_suffixes:
            model_suffixes['ipopt_zU_out'].clear()
            if bound_multipliers is not None:
                model_suffixes['ipopt_zU_out'].update(
                    zip(variables, bound_multipliers[1]))


class _ExternalGreyBoxAsNLP(NLP):
    """
    This class takes an ExternalGreyBoxModel and makes it look
    like an NLP so it can be used with other interfaces. Currently,
    the ExternalGreyBoxModel supports constraints only (no objective),
    so some of the methods are not appropriate and raise exceptions
    """
    def __init__(self, external_grey_box_block):
        self._block = external_grey_box_block
        self._ex_model = external_grey_box_block.get_external_model()
        n_inputs = len(self._block.inputs)
        assert n_inputs == self._ex_model.n_inputs()
        n_eq_constraints = self._ex_model.n_equality_constraints()
        n_outputs = len(self._block.outputs)
        assert n_outputs == self._ex_model.n_outputs()

        if self._ex_model.n_outputs() == 0 and \
           self._ex_model.n_equality_constraints() == 0:
            raise ValueError(
                'ExternalGreyBoxModel has no equality constraints '
                'or outputs. To use _ExternalGreyBoxAsNLP, it must'
                ' have at least one or both.')

        # create the list of primals and constraint names
        # primals will be ordered inputs, followed by outputs
        self._primals_names = \
            [self._block.inputs[k].getname(fully_qualified=True) \
             for k in self._block.inputs]
        self._primals_names.extend(
            [self._block.outputs[k].getname(fully_qualified=True) \
             for k in self._block.outputs]
        )

        prefix = self._block.getname(fully_qualified=True)
        self._constraint_names = \
            ['{}.{}'.format(prefix, nm) \
             for nm in self._ex_model.equality_constraint_names()])
        output_var_names = \
            [self._block.outputs[k].getname(fully_qualified=False) \
             for k in self._block.outputs]
        self._constraint_names.extend(
            ['{}.output_con_{}'.format(prefix, nm) \
             for nm in output_var_names])

        # create the numpy arrays of bounds on the primals
        self._primals_lb = [self._block.inputs[k].lb \
                            for k in self._block.inputs]
        self._primals_lb.extend(
            [self._block.outputs[k].lb for k in self._block.outputs]
        )
        self._primals_lb = np.asarray(self._primals_lb)
        self._primals_ub = \
            [self._block.inputs[k].ub for k in self._block.inputs]
        self._primals_ub.extend(
            [self._block.outputs[k].ub for k in self._block.outputs]
        )
        self._primals_ub = np.asarray(self.primals_ub)

        # create the numpy arrays for the initial values
        self._init_primals = \
            [pyo.value(self._block.inputs[k]) for k in self._block.inputs]
        self._init_primals.extend(
            [pyo.value(self._block.outputs[k]) for k in self._block.outputs]
        )
        self._init_primals = np.asarray(self._init_primals)

        # create a numpy array to store the values of the primals
        self._primal_values = np.copy(self._init_primals)
        # make sure the values are passed through to other objects
        self.set_primals(self._init_primals)

        # create the numpy arrays for bounds on the constraints
        # for now all of these are equalities
        self._constraints_lb = np.zeros(self.n_constraints())
        self._constraints_ub = np.zeros(self.n_constraints())

        # create the numpy arrays to store the dual variables
        self._dual_values = np.copy(self._init_duals)
        # make sure the values are passed through to other objects
        self.set_duals(self._init_duals)
        
        self._nnz_jacobian = None
        self._nnz_hessian_lag = None

    def n_primals(self):
        return len(self._primals_names)

    def primals_names(self):
        return list(self._primals_names)

    def n_constraints(self):
        return len(self._constraint_names)

    def constraint_names(self):
        return list(self._constraint_names)

    def nnz_jacobian(self):
        if self._nnz_jacobian is None:
            J = self.evaluate_jacobian()
            self._nnz_jacobian = len(J.data)
        return self._nnz_jacobian

    def nnz_hessian_lag(self):
        if self._nnz_hessian_lag is None:
            H = self.evaluate_hessian_lag()
            self._nnz_hessian_lag = len(H.data)
        return self._nnz_hessian_lag

    def primals_lb(self):
        return self._primals_lb
    
    def primals_ub(self):
        return self._primals_ub

    def constraints_lb(self):
        return self._constraints_lb

    def constraints_ub(self):
        return self._constraints_ub

    def init_primals(self):
        return self._init_primals

    def init_duals(self):
        # for now, these are initialized to 1.0
        return np.ones(self.n_constraints)

    def create_new_vector(self, vector_type):
        if vector_type == 'primals':
            return np.zeros(self.n_primals(), dtype=np.float64)
        elif vector_type == 'constraints' or vector_type == 'duals':
            return np.zeros(self.n_constraints(), dtype=np.float64)

    def set_primals(self, primals):
        assert len(primals) == self.n_primals()
        np.copyto(self._primal_values, primals)
        self._ex_model.set_input_values(primals[:self._ex_model.n_inputs()])

    def get_primals(self):
        return np.copy(self._primal_values)

    def set_duals(self, duals):
        assert len(duals) == self.n_constraints()
        np.copyto(self._dual_values, duals)
        if self._ex_model.n_equality_constraints() > 0:
            self._ex_model.set_equality_constraint_multipliers(
                self._dual_values[:self._ex_model.n_equality_constraints()]
                )
        if self._ex_model.n_outputs() > 0:
            self._ex_model.set_output_constraint_multipliers(
                self._dual_values[self._ex_model.n_equality_constraints():]
                )

    def get_duals(self):
        return np.copy(self._dual_values)

    def set_obj_factor(self, obj_factor):
        raise NotImplementedError('_ExternalGreyBoxAsNLP does not support objectives')

    def get_obj_factor(self):
        raise NotImplementedError('_ExternalGreyBoxAsNLP does not support objectives')

    def get_obj_scaling(self):
        raise NotImplementedError('_ExternalGreyBoxAsNLP does not support objectives')

    def get_primals_scaling(self):
        raise NotImplementedError(
            '_ExternalGreyBoxAsNLP does not support scaling of primals '
            'directly. This should be handled at a higher level using '
            'suffixes on the Pyomo variables.'
            )

    def get_constraints_scaling(self):
        scaling = np.ones(self.n_constraints())
        if self._ex_model.n_equality_constraints() > 0:
            eq_scaling = self._ex_model.get_equality_constraint_scaling_factors()
            if eq_scaling is not None:
                scaling[:self._ex_model.n_equality_constraints] = eq_scaling
        if self._ex_model.n_outputs() > 0:
            output_scaling = self._ex_model.get_output_constraint_scaling_factors()
            if output_scaling is not None:
                scaling[:self._ex_model.n_output_constraints] = output_scaling

    CRASH HERE Not IMPLEMENTED...
                
    @abc.abstractmethod
    def evaluate_objective(self):
        """Returns value of objective function evaluated at the 
        values given for the primal variables in set_primals

        Returns
        -------
        float
        """
        pass

    @abc.abstractmethod
    def evaluate_grad_objective(self, out=None):
        """Returns gradient of the objective function evaluated at the 
        values given for the primal variables in set_primals

        Parameters
        ----------
        out: vector_like, optional
            Output vector. Its type is preserved and it
            must be of the right shape to hold the output.

        Returns
        -------
        vector_like
        """
        pass

    @abc.abstractmethod
    def evaluate_constraints(self, out=None):
        """Returns the values for the constraints evaluated at
        the values given for the primal variales in set_primals

        Parameters
        ----------
        out: array_like, optional
            Output array. Its type is preserved and it
            must be of the right shape to hold the output.

        Returns
        -------
        vector_like
        """
        pass

    @abc.abstractmethod
    def evaluate_jacobian(self, out=None):
        """Returns the Jacobian of the constraints evaluated
        at the values given for the primal variables in set_primals

        Parameters
        ----------
        out : matrix_like (e.g., coo_matrix), optional
            Output matrix with the structure of the jacobian already defined.

        Returns
        -------
        matrix_like
        """
        pass

    @abc.abstractmethod
    def evaluate_hessian_lag(self, out=None):
        """Return the Hessian of the Lagrangian function evaluated
        at the values given for the primal variables in set_primals and
        the dual variables in set_duals

        Parameters
        ----------
        out : matrix_like (e.g., coo_matrix), optional
            Output matrix with the structure of the hessian already defined. Optional

        Returns
        -------
        matrix_like
        """
        pass

    @abc.abstractmethod
    def report_solver_status(self, status_code, status_message):
        """Report the solver status to NLP class using the values for the 
        primals and duals defined in the set methods"""
        pass
