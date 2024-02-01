#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
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
import logging

from scipy.sparse import coo_matrix, identity
from pyomo.common.deprecation import deprecated
import pyomo.core.base as pyo
from pyomo.common.collections import ComponentMap
from pyomo.contrib.pynumero.sparse.block_matrix import BlockMatrix
from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from pyomo.contrib.pynumero.interfaces.nlp import NLP
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.utils import (
    make_lower_triangular_full,
    CondensedSparseSummation,
)
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.interfaces.nlp_projections import ProjectedNLP


# Todo: make some of the numpy arrays not writable from __init__
class PyomoNLPWithGreyBoxBlocks(NLP):
    def __init__(self, pyomo_model):
        super(PyomoNLPWithGreyBoxBlocks, self).__init__()

        # get the list of all grey box blocks and build _ExternalGreyBoxAsNLP objects
        greybox_components = []
        # build a map from the names to the variable data objects
        # this is done over *all* variables in active blocks, even
        # if they are not included in this model
        self._pyomo_model_var_names_to_datas = None
        try:
            # We support Pynumero's ExternalGreyBoxBlock modeling
            # objects that are provided through ExternalGreyBoxBlock objects
            # We reclassify these as Pyomo Block objects before building the
            # PyomoNLP object to expose any variables on the block to
            # the underlying Pyomo machinery
            for greybox in pyomo_model.component_objects(
                ExternalGreyBoxBlock, descend_into=True
            ):
                greybox.parent_block().reclassify_component_type(greybox, pyo.Block)
                greybox_components.append(greybox)

            # store the pyomo model
            self._pyomo_model = pyomo_model
            # build a PyomoNLP object (will include the "pyomo"
            # part of the model only)
            self._pyomo_nlp = PyomoNLP(pyomo_model)
            self._pyomo_model_var_names_to_datas = {
                v.getname(fully_qualified=True): v
                for v in pyomo_model.component_data_objects(
                    ctype=pyo.Var, descend_into=True
                )
            }
            self._pyomo_model_constraint_names_to_datas = {
                c.getname(fully_qualified=True): c
                for c in pyomo_model.component_data_objects(
                    ctype=pyo.Constraint, descend_into=True
                )
            }

        finally:
            # Restore the ctypes of the ExternalGreyBoxBlock components
            for greybox in greybox_components:
                greybox.parent_block().reclassify_component_type(
                    greybox, ExternalGreyBoxBlock
                )

        if self._pyomo_nlp.n_primals() == 0:
            raise ValueError(
                "No variables were found in the Pyomo part of the model."
                " PyomoGreyBoxModel requires at least one variable"
                " to be active in a Pyomo objective or constraint"
            )

        # build the list of NLP wrappers for the greybox objects
        greybox_nlps = []
        fixed_vars = []
        for greybox in greybox_components:
            # iterate through the data objects if component is indexed
            for data in greybox.values():
                if data.active:
                    # check that no variables are fixed
                    fixed_vars.extend(v for v in data.inputs.values() if v.fixed)
                    fixed_vars.extend(v for v in data.outputs.values() if v.fixed)
                    greybox_nlp = _ExternalGreyBoxAsNLP(data)
                    greybox_nlps.append(greybox_nlp)

        if fixed_vars:
            logging.getLogger(__name__).error(
                'PyomoNLPWithGreyBoxBlocks found fixed variables for the'
                ' inputs and/or outputs of an ExternalGreyBoxBlock. This'
                ' is not currently supported. The fixed variables were:\n\t'
                + '\n\t'.join(f.getname(fully_qualified=True) for f in fixed_vars)
            )
            raise NotImplementedError(
                'PyomoNLPWithGreyBoxBlocks does not support fixed inputs or outputs'
            )

        # let's build up the union of all the primal variables names
        # RBP: Why use names here? Why not just ComponentSet of all
        # data objects?
        primals_names = set(self._pyomo_nlp.primals_names())
        for gbnlp in greybox_nlps:
            primals_names.update(gbnlp.primals_names())

        # sort the names for consistency run to run
        self._n_primals = len(primals_names)
        self._primals_names = primals_names = sorted(primals_names)
        self._pyomo_model_var_datas = [
            self._pyomo_model_var_names_to_datas[nm] for nm in self._primals_names
        ]

        # get the names of all the constraints
        self._constraint_names = list(self._pyomo_nlp.constraint_names())
        self._constraint_datas = [
            self._pyomo_model_constraint_names_to_datas.get(nm)
            for nm in self._constraint_names
        ]
        for gbnlp in greybox_nlps:
            self._constraint_names.extend(gbnlp.constraint_names())
            self._constraint_datas.extend(
                [(gbnlp._block, nm) for nm in gbnlp.constraint_names()]
            )
        self._n_constraints = len(self._constraint_names)

        self._has_hessian_support = True
        for nlp in greybox_nlps:
            if not nlp.has_hessian_support():
                self._has_hessian_support = False

        # wrap all the nlp objects with projected nlp objects
        self._pyomo_nlp = ProjectedNLP(self._pyomo_nlp, primals_names)
        for i, gbnlp in enumerate(greybox_nlps):
            greybox_nlps[i] = ProjectedNLP(greybox_nlps[i], primals_names)

        # build a list of all the nlps in order
        self._nlps = nlps = [self._pyomo_nlp]
        nlps.extend(greybox_nlps)

        # build the primal and dual inits and lb, ub vectors
        self._init_primals = self._pyomo_nlp.init_primals()
        self._primals_lb = self._pyomo_nlp.primals_lb()
        self._primals_ub = self._pyomo_nlp.primals_ub()
        for gbnlp in greybox_nlps:
            local = gbnlp.init_primals()
            mask = ~np.isnan(local)
            self._init_primals[mask] = local[mask]

            local = gbnlp.primals_lb()
            mask = ~np.isnan(local)
            self._primals_lb[mask] = np.maximum(self._primals_lb[mask], local[mask])

            local = gbnlp.primals_ub()
            mask = ~np.isnan(local)
            self._primals_ub[mask] = np.minimum(self._primals_ub[mask], local[mask])

        # all the nan's should be gone (every primal should be initialized)
        if (
            np.any(np.isnan(self._init_primals))
            or np.any(np.isnan(self._primals_lb))
            or np.any(np.isnan(self._primals_ub))
        ):
            raise ValueError(
                'NaN values found in initialization of primals or'
                ' primals_lb or primals_ub in _PyomoNLPWithGreyBoxBlocks.'
            )

        self._init_duals = BlockVector(len(nlps))
        self._dual_values_blockvector = BlockVector(len(nlps))
        self._constraints_lb = BlockVector(len(nlps))
        self._constraints_ub = BlockVector(len(nlps))
        for i, nlp in enumerate(nlps):
            self._init_duals.set_block(i, nlp.init_duals())
            self._constraints_lb.set_block(i, nlp.constraints_lb())
            self._constraints_ub.set_block(i, nlp.constraints_ub())
            self._dual_values_blockvector.set_block(
                i, np.nan * np.zeros(nlp.n_constraints())
            )
        self._init_duals = self._init_duals.flatten()
        self._constraints_lb = self._constraints_lb.flatten()
        self._constraints_ub = self._constraints_ub.flatten()
        # verify that there are no nans in the init_duals
        if (
            np.any(np.isnan(self._init_duals))
            or np.any(np.isnan(self._constraints_lb))
            or np.any(np.isnan(self._constraints_ub))
        ):
            raise ValueError(
                'NaN values found in initialization of duals or'
                ' constraints_lb or constraints_ub in'
                ' _PyomoNLPWithGreyBoxBlocks.'
            )

        self._primal_values = np.nan * np.ones(self._n_primals)
        # set the values of the primals and duals to make sure initial
        # values get all the way through to the underlying models
        self.set_primals(self._init_primals)
        self.set_duals(self._init_duals)
        assert not np.any(np.isnan(self._primal_values))
        assert not np.any(np.isnan(self._dual_values_blockvector))

        # if any of the problem is scaled (i.e., one or more of primals,
        # constraints, or objective), then we want scaling factors for
        # all of them (defaulted to 1)
        need_scaling = False
        # objective is owned by self._pyomo_nlp, not in any of the greybox models
        self._obj_scaling = self._pyomo_nlp.get_obj_scaling()
        if self._obj_scaling is None:
            self._obj_scaling = 1.0
        else:
            need_scaling = True

        self._primals_scaling = np.ones(self.n_primals())
        scaling_suffix = pyomo_model.component('scaling_factor')
        if scaling_suffix and scaling_suffix.ctype is pyo.Suffix:
            need_scaling = True
            for i, v in enumerate(self._pyomo_model_var_datas):
                if v in scaling_suffix:
                    self._primals_scaling[i] = scaling_suffix[v]

        self._constraints_scaling = BlockVector(len(nlps))
        for i, nlp in enumerate(nlps):
            local_constraints_scaling = nlp.get_constraints_scaling()
            if local_constraints_scaling is None:
                self._constraints_scaling.set_block(i, np.ones(nlp.n_constraints()))
            else:
                self._constraints_scaling.set_block(i, local_constraints_scaling)
                need_scaling = True
        if need_scaling:
            self._constraints_scaling = self._constraints_scaling.flatten()
        else:
            self._obj_scaling = None
            self._primals_scaling = None
            self._constraints_scaling = None

        # compute the jacobian and the hessian to get nnz
        jac = self.evaluate_jacobian()
        self._nnz_jacobian = len(jac.data)

        self._sparse_hessian_summation = None
        self._nnz_hessian_lag = None
        if self._has_hessian_support:
            hess = self.evaluate_hessian_lag()
            self._nnz_hessian_lag = len(hess.data)

    # overloaded from NLP
    def n_primals(self):
        return self._n_primals

    # overloaded from NLP
    def primals_names(self):
        return self._primals_names

    # overloaded from NLP
    def n_constraints(self):
        return self._n_constraints

    # overloaded from NLP
    def constraint_names(self):
        return self._constraint_names

    # overloaded from NLP
    def nnz_jacobian(self):
        return self._nnz_jacobian

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
        return self._constraints_lb

    # overloaded from NLP
    def constraints_ub(self):
        return self._constraints_ub

    # overloaded from NLP
    def init_primals(self):
        return self._init_primals

    # overloaded from NLP
    def init_duals(self):
        return self._init_duals

    # overloaded from NLP / Extended NLP
    def create_new_vector(self, vector_type):
        if vector_type == 'primals':
            return np.zeros(self.n_primals(), dtype=np.float64)
        elif vector_type == 'constraints' or vector_type == 'duals':
            return np.zeros(self.n_constraints(), dtype=np.float64)
        else:
            raise RuntimeError('Called create_new_vector with an unknown vector_type')

    # overloaded from NLP
    def set_primals(self, primals):
        np.copyto(self._primal_values, primals)
        for nlp in self._nlps:
            nlp.set_primals(primals)

    # overloaded from AslNLP
    def get_primals(self):
        return np.copy(self._primal_values)

    # overloaded from NLP
    def set_duals(self, duals):
        self._dual_values_blockvector.copyfrom(duals)
        for i, nlp in enumerate(self._nlps):
            nlp.set_duals(self._dual_values_blockvector.get_block(i))

    # overloaded from NLP
    def get_duals(self):
        return self._dual_values_blockvector.flatten()

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
        return self._pyomo_nlp.evaluate_grad_objective(out=out)

    # overloaded from NLP
    def evaluate_constraints(self, out=None):
        # todo: implement the "out" version more efficiently
        ret = BlockVector(len(self._nlps))
        for i, nlp in enumerate(self._nlps):
            ret.set_block(i, nlp.evaluate_constraints())

        if out is not None:
            ret.copyto(out)
            return out

        return ret.flatten()

    # overloaded from NLP
    def evaluate_jacobian(self, out=None):
        ret = BlockMatrix(len(self._nlps), 1)
        for i, nlp in enumerate(self._nlps):
            ret.set_block(i, 0, nlp.evaluate_jacobian())
        ret = ret.tocoo()

        if out is not None:
            assert np.array_equal(ret.row, out.row)
            assert np.array_equal(ret.col, out.col)
            np.copyto(out.data, ret.data)
            return out
        return ret

    def evaluate_hessian_lag(self, out=None):
        list_of_hessians = [nlp.evaluate_hessian_lag() for nlp in self._nlps]
        if self._sparse_hessian_summation is None:
            # This is assuming that the nonzero structures of Hessians
            # do not change
            self._sparse_hessian_summation = CondensedSparseSummation(list_of_hessians)
        ret = self._sparse_hessian_summation.sum(list_of_hessians)

        if out is not None:
            assert np.array_equal(ret.row, out.row)
            assert np.array_equal(ret.col, out.col)
            np.copyto(out.data, ret.data)
            return out
        return ret

    def report_solver_status(self, status_code, status_message):
        raise NotImplementedError('This is not yet implemented.')

    def load_state_into_pyomo(self, bound_multipliers=None):
        # load the values of the primals into the pyomo
        primals = self.get_primals()
        for value, vardata in zip(primals, self._pyomo_model_var_datas):
            vardata.set_value(value)

        # get the active suffixes
        m = self._pyomo_model
        model_suffixes = dict(pyo.suffix.active_import_suffix_generator(m))

        # we need to correct the sign of the multipliers based on whether or
        # not we are minimizing or maximizing - this is done in the ASL interface
        # for ipopt, but does not appear to be done in cyipopt.
        obj_sign = 1.0
        # since we will assert the number of objective functions,
        # we only focus on active objective function.
        objs = list(
            m.component_data_objects(
                ctype=pyo.Objective, active=True, descend_into=True
            )
        )
        assert len(objs) == 1
        if objs[0].sense == pyo.maximize:
            obj_sign = -1.0

        if 'dual' in model_suffixes:
            model_suffixes['dual'].clear()
            dual_values = self._dual_values_blockvector.flatten()
            for value, t in zip(dual_values, self._constraint_datas):
                if type(t) is tuple:
                    model_suffixes['dual'].setdefault(t[0], {})[t[1]] = (
                        -obj_sign * value
                    )
                else:
                    # t is a constraint data
                    model_suffixes['dual'][t] = -obj_sign * value

        if 'ipopt_zL_out' in model_suffixes:
            model_suffixes['ipopt_zL_out'].clear()
            if bound_multipliers is not None:
                model_suffixes['ipopt_zL_out'].update(
                    zip(self._pyomo_model_var_datas, obj_sign * bound_multipliers[0])
                )
        if 'ipopt_zU_out' in model_suffixes:
            model_suffixes['ipopt_zU_out'].clear()
            if bound_multipliers is not None:
                model_suffixes['ipopt_zU_out'].update(
                    zip(self._pyomo_model_var_datas, -obj_sign * bound_multipliers[1])
                )


def _default_if_none(value, default):
    if value is None:
        return default
    return value


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

        if (
            self._ex_model.n_outputs() == 0
            and self._ex_model.n_equality_constraints() == 0
        ):
            raise ValueError(
                'ExternalGreyBoxModel has no equality constraints '
                'or outputs. To use _ExternalGreyBoxAsNLP, it must'
                ' have at least one or both.'
            )

        # create the list of primals and constraint names
        # primals will be ordered inputs, followed by outputs
        self._primals_names = [
            self._block.inputs[k].getname(fully_qualified=True)
            for k in self._block.inputs
        ]
        self._primals_names.extend(
            self._block.outputs[k].getname(fully_qualified=True)
            for k in self._block.outputs
        )
        n_primals = len(self._primals_names)

        prefix = self._block.getname(fully_qualified=True)
        self._constraint_names = [
            '{}.{}'.format(prefix, nm)
            for nm in self._ex_model.equality_constraint_names()
        ]
        output_var_names = [
            self._block.outputs[k].getname(fully_qualified=False)
            for k in self._block.outputs
        ]
        self._constraint_names.extend(
            [
                '{}.output_constraints[{}]'.format(prefix, nm)
                for nm in self._ex_model.output_names()
            ]
        )

        # create the numpy arrays of bounds on the primals
        self._primals_lb = BlockVector(2)
        self._primals_ub = BlockVector(2)
        self._init_primals = BlockVector(2)
        lb = np.nan * np.zeros(n_inputs)
        ub = np.nan * np.zeros(n_inputs)
        init_primals = np.nan * np.zeros(n_inputs)
        for i, k in enumerate(self._block.inputs):
            lb[i] = _default_if_none(self._block.inputs[k].lb, -np.inf)
            ub[i] = _default_if_none(self._block.inputs[k].ub, np.inf)
            init_primals[i] = _default_if_none(self._block.inputs[k].value, 0.0)
        self._primals_lb.set_block(0, lb)
        self._primals_ub.set_block(0, ub)
        self._init_primals.set_block(0, init_primals)

        lb = np.nan * np.zeros(n_outputs)
        ub = np.nan * np.zeros(n_outputs)
        init_primals = np.nan * np.zeros(n_outputs)
        for i, k in enumerate(self._block.outputs):
            lb[i] = _default_if_none(self._block.outputs[k].lb, -np.inf)
            ub[i] = _default_if_none(self._block.outputs[k].ub, np.inf)
            init_primals[i] = _default_if_none(self._block.outputs[k].value, 0.0)
        self._primals_lb.set_block(1, lb)
        self._primals_ub.set_block(1, ub)
        self._init_primals.set_block(1, init_primals)
        self._primals_lb = self._primals_lb.flatten()
        self._primals_ub = self._primals_ub.flatten()
        self._init_primals = self._init_primals.flatten()

        # create a numpy array to store the values of the primals
        self._primal_values = np.copy(self._init_primals)
        # make sure the values are passed through to other objects
        self.set_primals(self._init_primals)

        # create the numpy arrays for the duals and initial values
        # for now, initialize the duals to zero
        self._init_duals = np.zeros(self.n_constraints(), dtype=np.float64)
        # create the numpy arrays to store the dual variables
        self._dual_values = np.copy(self._init_duals)
        # make sure the values are passed through to other objects
        self.set_duals(self._init_duals)

        # create the numpy arrays for bounds on the constraints
        # for now all of these are equalities
        self._constraints_lb = np.zeros(self.n_constraints(), dtype=np.float64)
        self._constraints_ub = np.zeros(self.n_constraints(), dtype=np.float64)

        # do we have hessian support
        self._has_hessian_support = True
        if self._ex_model.n_equality_constraints() > 0 and not hasattr(
            self._ex_model, 'evaluate_hessian_equality_constraints'
        ):
            self._has_hessian_support = False
        if self._ex_model.n_outputs() > 0 and not hasattr(
            self._ex_model, 'evaluate_hessian_outputs'
        ):
            self._has_hessian_support = False

        self._nnz_jacobian = None
        self._nnz_hessian_lag = None
        self._cached_constraint_residuals = None
        self._cached_jacobian = None
        self._cached_hessian = None

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
        return np.copy(self._primals_lb)

    def primals_ub(self):
        return np.copy(self._primals_ub)

    def constraints_lb(self):
        return np.copy(self._constraints_lb)

    def constraints_ub(self):
        return np.copy(self._constraints_ub)

    def init_primals(self):
        return np.copy(self._init_primals)

    def init_duals(self):
        return np.copy(self._init_duals)

    def create_new_vector(self, vector_type):
        if vector_type == 'primals':
            return np.zeros(self.n_primals(), dtype=np.float64)
        elif vector_type == 'constraints' or vector_type == 'duals':
            return np.zeros(self.n_constraints(), dtype=np.float64)

    def _cache_invalidate_primals(self):
        self._cached_constraint_residuals = None
        self._cached_jacobian = None
        self._cached_hessian = None

    def set_primals(self, primals):
        self._cache_invalidate_primals()
        assert len(primals) == self.n_primals()
        np.copyto(self._primal_values, primals)
        self._ex_model.set_input_values(primals[: self._ex_model.n_inputs()])

    def get_primals(self):
        return np.copy(self._primal_values)

    def _cache_invalidate_duals(self):
        self._cached_hessian = None

    def set_duals(self, duals):
        self._cache_invalidate_duals()
        assert len(duals) == self.n_constraints()
        np.copyto(self._dual_values, duals)
        if self._ex_model.n_equality_constraints() > 0:
            self._ex_model.set_equality_constraint_multipliers(
                self._dual_values[: self._ex_model.n_equality_constraints()]
            )
        if self._ex_model.n_outputs() > 0:
            self._ex_model.set_output_constraint_multipliers(
                self._dual_values[self._ex_model.n_equality_constraints() :]
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
        # todo: would this be better with block vectors
        scaling = np.ones(self.n_constraints(), dtype=np.float64)
        scaled = False
        if self._ex_model.n_equality_constraints() > 0:
            eq_scaling = self._ex_model.get_equality_constraint_scaling_factors()
            if eq_scaling is not None:
                scaling[: self._ex_model.n_equality_constraints()] = eq_scaling
                scaled = True
        if self._ex_model.n_outputs() > 0:
            output_scaling = self._ex_model.get_output_constraint_scaling_factors()
            if output_scaling is not None:
                scaling[self._ex_model.n_equality_constraints() :] = output_scaling
                scaled = True
        if scaled:
            return scaling
        return None

    def evaluate_objective(self):
        # todo: Should we return 0 here?
        raise NotImplementedError('_ExternalGreyBoxNLP does not support objectives')

    def evaluate_grad_objective(self, out=None):
        # todo: Should we return 0 here?
        raise NotImplementedError('_ExternalGreyBoxNLP does not support objectives')

    def _evaluate_constraints_if_necessary_and_cache(self):
        if self._cached_constraint_residuals is None:
            c = BlockVector(2)
            if self._ex_model.n_equality_constraints() > 0:
                c.set_block(0, self._ex_model.evaluate_equality_constraints())
            else:
                c.set_block(0, np.zeros(0, dtype=np.float64))
            if self._ex_model.n_outputs() > 0:
                output_values = self._primal_values[self._ex_model.n_inputs() :]
                c.set_block(1, self._ex_model.evaluate_outputs() - output_values)
            else:
                c.set_block(1, np.zeros(0, dtype=np.float64))
            self._cached_constraint_residuals = c.flatten()

    def evaluate_constraints(self, out=None):
        self._evaluate_constraints_if_necessary_and_cache()
        if out is not None:
            assert len(out) == self.n_constraints()
            np.copyto(out, self._cached_constraint_residuals)
            return out

        return np.copy(self._cached_constraint_residuals)

    def _evaluate_jacobian_if_necessary_and_cache(self):
        if self._cached_jacobian is None:
            jac = BlockMatrix(2, 2)
            jac.set_row_size(0, self._ex_model.n_equality_constraints())
            jac.set_row_size(1, self._ex_model.n_outputs())
            jac.set_col_size(0, self._ex_model.n_inputs())
            jac.set_col_size(1, self._ex_model.n_outputs())

            if self._ex_model.n_equality_constraints() > 0:
                jac.set_block(
                    0, 0, self._ex_model.evaluate_jacobian_equality_constraints()
                )
            if self._ex_model.n_outputs() > 0:
                jac.set_block(1, 0, self._ex_model.evaluate_jacobian_outputs())
                jac.set_block(1, 1, -1.0 * identity(self._ex_model.n_outputs()))

            self._cached_jacobian = jac.tocoo()

    def evaluate_jacobian(self, out=None):
        self._evaluate_jacobian_if_necessary_and_cache()
        if out is not None:
            jac = self._cached_jacobian
            assert np.array_equal(jac.row, out.row)
            assert np.array_equal(jac.col, out.col)
            np.copyto(out.data, jac.data)
            return out

        return self._cached_jacobian.copy()

    def _evaluate_hessian_if_necessary_and_cache(self):
        if self._cached_hessian is None:
            hess = BlockMatrix(2, 2)
            hess.set_row_size(0, self._ex_model.n_inputs())
            hess.set_row_size(1, self._ex_model.n_outputs())
            hess.set_col_size(0, self._ex_model.n_inputs())
            hess.set_col_size(1, self._ex_model.n_outputs())

            # get the hessian w.r.t. the equality constraints
            eq_hess = None
            if self._ex_model.n_equality_constraints() > 0:
                eq_hess = self._ex_model.evaluate_hessian_equality_constraints()
                # let's check that it is lower triangular
                if np.any(eq_hess.row < eq_hess.col):
                    raise ValueError(
                        'ExternalGreyBoxModel must return lower '
                        'triangular portion of the Hessian only'
                    )

                eq_hess = make_lower_triangular_full(eq_hess)

            output_hess = None
            if self._ex_model.n_outputs() > 0:
                output_hess = self._ex_model.evaluate_hessian_outputs()
                # let's check that it is lower triangular
                if np.any(output_hess.row < output_hess.col):
                    raise ValueError(
                        'ExternalGreyBoxModel must return lower '
                        'triangular portion of the Hessian only'
                    )

                output_hess = make_lower_triangular_full(output_hess)

            input_hess = None
            if eq_hess is not None and output_hess is not None:
                # we may want to make this more efficient
                row = np.concatenate((eq_hess.row, output_hess.row))
                col = np.concatenate((eq_hess.col, output_hess.col))
                data = np.concatenate((eq_hess.data, output_hess.data))

                assert eq_hess.shape == output_hess.shape
                input_hess = coo_matrix((data, (row, col)), shape=eq_hess.shape)
            elif eq_hess is not None:
                input_hess = eq_hess
            elif output_hess is not None:
                input_hess = output_hess
            assert input_hess is not None  # need equality or outputs or both

            hess.set_block(0, 0, input_hess)
            self._cached_hessian = hess.tocoo()

    def has_hessian_support(self):
        return self._has_hessian_support

    def evaluate_hessian_lag(self, out=None):
        if not self._has_hessian_support:
            raise NotImplementedError(
                'Hessians not supported for all of the external grey box'
                ' models. Therefore, Hessians are not supported overall.'
            )

        self._evaluate_hessian_if_necessary_and_cache()
        if out is not None:
            hess = self._cached_hessian
            assert np.array_equal(hess.row, out.row)
            assert np.array_equal(hess.col, out.col)
            np.copyto(out.data, hess.data)
            return out

        return self._cached_hessian.copy()

    def report_solver_status(self, status_code, status_message):
        raise NotImplementedError('report_solver_status not implemented')
