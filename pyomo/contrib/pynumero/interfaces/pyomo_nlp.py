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

from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import WriterFactory
import pyomo.core.base as pyo
from pyomo.common.collections import ComponentMap
from pyomo.common.env import CtypesEnviron
from ..sparse.block_matrix import BlockMatrix
from pyomo.contrib.pynumero.interfaces.ampl_nlp import AslNLP
from pyomo.contrib.pynumero.interfaces.nlp import NLP
from .external_grey_box import ExternalGreyBoxBlock, _ExternalGreyBoxModelHelper


__all__ = ['PyomoNLP']

# TODO: There are todos in the code below
class PyomoNLP(AslNLP):
    def __init__(self, pyomo_model):
        """
        Pyomo nonlinear program interface

        Parameters
        ----------
        pyomo_model: pyomo.environ.ConcreteModel
            Pyomo concrete model
        """
        TempfileManager.push()
        try:
            # get the temp file names for the nl file
            nl_file = TempfileManager.create_tempfile(
                suffix='pynumero.nl')

            # The current AmplInterface code only supports a single
            # objective function Therefore, we throw an error if there
            # is not one (and only one) active objective function. This
            # is better than adding a dummy objective that the user does
            # not know about (since we do not have a good place to
            # remove this objective later)
            #
            # TODO: extend the AmplInterface and the AslNLP to correctly
            # handle this
            #
            # This currently addresses issue #1217
            objectives = list(pyomo_model.component_data_objects(
                ctype=pyo.Objective, active=True, descend_into=True))
            if len(objectives) != 1:
                raise NotImplementedError(
                    'The ASL interface and PyomoNLP in PyNumero currently '
                    'only support single objective problems. Deactivate '
                    'any extra objectives you may have, or add a dummy '
                    'objective (f(x)=0) if you have a square problem.')
            self._objective = objectives[0]

            # write the nl file for the Pyomo model and get the symbolMap
            fname, symbolMap = WriterFactory('nl')(
                pyomo_model, nl_file, lambda x:True, {})

            # create component maps from vardata to idx and condata to idx
            self._vardata_to_idx = vdidx = ComponentMap()
            self._condata_to_idx = cdidx = ComponentMap()

            # TODO: Are these names totally consistent?
            for name, obj in six.iteritems(symbolMap.bySymbol):
                if name[0] == 'v':
                    vdidx[obj()] = int(name[1:])
                elif name[0] == 'c':
                    cdidx[obj()] = int(name[1:])

            # The NL writer advertises the external function libraries
            # through the PYOMO_AMPLFUNC environment variable; merge it
            # with any preexisting AMPLFUNC definitions
            amplfunc = "\n".join(
                val for val in (
                    os.environ.get('AMPLFUNC', ''),
                    os.environ.get('PYOMO_AMPLFUNC', ''),
                ) if val)
            with CtypesEnviron(AMPLFUNC=amplfunc):
                super(PyomoNLP, self).__init__(nl_file)

            # keep pyomo model in cache
            self._pyomo_model = pyomo_model

            # Create ComponentMap corresponding to equality constraint indices
            # This must be done after the call to super-init.
            full_to_equality = self._con_full_eq_map
            equality_mask = self._con_full_eq_mask
            self._condata_to_eq_idx = ComponentMap(
                    (con, full_to_equality[i])
                    for con, i in six.iteritems(self._condata_to_idx)
                    if equality_mask[i]
                    )
            full_to_inequality = self._con_full_ineq_map
            inequality_mask = self._con_full_ineq_mask
            self._condata_to_ineq_idx = ComponentMap(
                    (con, full_to_inequality[i])
                    for con, i in six.iteritems(self._condata_to_idx)
                    if inequality_mask[i]
                    )

        finally:
            # delete the nl file
            TempfileManager.pop()


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
        return self._objective

    def get_pyomo_variables(self):
        """
        Return an ordered list of the Pyomo VarData objects in
        the order corresponding to the primals
        """
        # ToDo: is there a more efficient way to do this
        idx_to_vardata = {i:v for v,i in six.iteritems(self._vardata_to_idx)}
        return [idx_to_vardata[i] for i in range(len(idx_to_vardata))]

    def get_pyomo_constraints(self):
        """
        Return an ordered list of the Pyomo ConData objects in
        the order corresponding to the primals
        """
        # ToDo: is there a more efficient way to do this
        idx_to_condata = {i:v for v,i in six.iteritems(self._condata_to_idx)}
        return [idx_to_condata[i] for i in range(len(idx_to_condata))]

    def get_pyomo_equality_constraints(self):
        """
        Return an ordered list of the Pyomo ConData objects in
        the order corresponding to the equality constraints.
        """
        idx_to_condata = {i: c for c, i in
                six.iteritems(self._condata_to_eq_idx)}
        return [idx_to_condata[i] for i in range(len(idx_to_condata))]

    def get_pyomo_inequality_constraints(self):
        """
        Return an ordered list of the Pyomo ConData objects in
        the order corresponding to the inequality constraints.
        """
        idx_to_condata = {i: c for c, i in
                six.iteritems(self._condata_to_ineq_idx)}
        return [idx_to_condata[i] for i in range(len(idx_to_condata))]

    def variable_names(self):
        """
        Return an ordered list of the Pyomo variable
        names in the order corresponding to the primals
        """
        pyomo_variables = self.get_pyomo_variables()
        return [v.getname(fully_qualified=True) for v in pyomo_variables]

    def constraint_names(self):
        """
        Return an ordered list of the Pyomo constraint
        names in the order corresponding to internal constraint order
        """
        pyomo_constraints = self.get_pyomo_constraints()
        return [v.getname(fully_qualified=True) for v in pyomo_constraints]

    def equality_constraint_names(self):
        """
        Return an ordered list of the Pyomo ConData names in
        the order corresponding to the equality constraints.
        """
        equality_constraints = self.get_pyomo_equality_constraints()
        return [v.getname(fully_qualified=True) for v in equality_constraints]

    def inequality_constraint_names(self):
        """
        Return an ordered list of the Pyomo ConData names in
        the order corresponding to the inequality constraints.
        """
        inequality_constraints = self.get_pyomo_inequality_constraints()
        return [v.getname(fully_qualified=True) for v in inequality_constraints]

    def get_primal_indices(self, pyomo_variables):
        """
        Return the list of indices for the primals
        corresponding to the list of Pyomo variables provided

        Parameters
        ----------
        pyomo_variables : list of Pyomo Var or VarData objects
        """
        assert isinstance(pyomo_variables, list)
        var_indices = []
        for v in pyomo_variables:
            if v.is_indexed():
                for vd in v.values():
                    var_id = self._vardata_to_idx[vd]
                    var_indices.append(var_id)
            else:
                var_id = self._vardata_to_idx[v]
                var_indices.append(var_id)
        return var_indices

    def get_constraint_indices(self, pyomo_constraints):
        """
        Return the list of indices for the constraints
        corresponding to the list of Pyomo constraints provided

        Parameters
        ----------
        pyomo_constraints : list of Pyomo Constraint or ConstraintData objects
        """
        assert isinstance(pyomo_constraints, list)
        con_indices = []
        for c in pyomo_constraints:
            if c.is_indexed():
                for cd in c.values():
                    con_id = self._condata_to_idx[cd]
                    con_indices.append(con_id)
            else:
                con_id = self._condata_to_idx[c]
                con_indices.append(con_id)
        return con_indices

    def get_equality_constraint_indices(self, constraints):
        """
        Return the list of equality indices for the constraints
        corresponding to the list of Pyomo constraints provided.

        Parameters
        ----------
        constraints : list of Pyomo Constraints or ConstraintData objects
        """
        indices = []
        for c in constraints:
            if c.is_indexed():
                for cd in c.values():
                    con_eq_idx = self._condata_to_eq_idx[cd]
                    indices.append(con_eq_idx)
            else:
                con_eq_idx = self._condata_to_eq_idx[c]
                indices.append(con_eq_idx)
        return indices

    def get_inequality_constraint_indices(self, constraints):
        """
        Return the list of inequality indices for the constraints
        corresponding to the list of Pyomo constraints provided.

        Parameters
        ----------
        constraints : list of Pyomo Constraints or ConstraintData objects
        """
        indices = []
        for c in constraints:
            if c.is_indexed():
                for cd in c.values():
                    con_ineq_idx = self._condata_to_ineq_idx[cd]
                    indices.append(con_ineq_idx)
            else:
                con_ineq_idx = self._condata_to_ineq_idx[c]
                indices.append(con_ineq_idx)
        return indices

    # overloaded from NLP
    def get_obj_scaling(self):
        obj = self.get_pyomo_objective()
        scaling_suffix = self._pyomo_model.component('scaling_factor')
        if scaling_suffix and scaling_suffix.ctype is pyo.Suffix:
            if obj in scaling_suffix:
                return scaling_suffix[obj]
            return 1.0
        return None

    # overloaded from NLP
    def get_primals_scaling(self):
        scaling_suffix = self._pyomo_model.component('scaling_factor')
        if scaling_suffix and scaling_suffix.ctype is pyo.Suffix:
            primals_scaling = np.ones(self.n_primals())
            for i,v in enumerate(self.get_pyomo_variables()):
                if v in scaling_suffix:
                    primals_scaling[i] = scaling_suffix[v]
            return primals_scaling
        return None

    # overloaded from NLP
    def get_constraints_scaling(self):
        scaling_suffix = self._pyomo_model.component('scaling_factor')
        if scaling_suffix and scaling_suffix.ctype is pyo.Suffix:
            constraints_scaling = np.ones(self.n_constraints())
            for i,c in enumerate(self.get_pyomo_constraints()):
                if c in scaling_suffix:
                    constraints_scaling[i] = scaling_suffix[c]
            return constraints_scaling
        return None

    def extract_subvector_grad_objective(self, pyomo_variables):
        """Compute the gradient of the objective and return the entries
        corresponding to the given Pyomo variables

        Parameters
        ----------
        pyomo_variables : list of Pyomo Var or VarData objects
        """
        grad_obj = self.evaluate_grad_objective()
        return grad_obj[self.get_primal_indices(pyomo_variables)]

    def extract_subvector_constraints(self, pyomo_constraints):
        """
        Return the values of the constraints
        corresponding to the list of Pyomo constraints provided

        Parameters
        ----------
        pyomo_constraints : list of Pyomo Constraint or ConstraintData objects
        """
        residuals = self.evaluate_constraints()
        return residuals[self.get_constraint_indices(pyomo_constraints)]

    def extract_submatrix_jacobian(self, pyomo_variables, pyomo_constraints):
        """
        Return the submatrix of the jacobian that corresponds to the list
        of Pyomo variables and list of Pyomo constraints provided

        Parameters
        ----------
        pyomo_variables : list of Pyomo Var or VarData objects
        pyomo_constraints : list of Pyomo Constraint or ConstraintData objects
        """
        jac = self.evaluate_jacobian()
        primal_indices = self.get_primal_indices(pyomo_variables)
        constraint_indices = self.get_constraint_indices(pyomo_constraints)
        row_mask = np.isin(jac.row, constraint_indices)
        col_mask = np.isin(jac.col, primal_indices)
        submatrix_mask = row_mask & col_mask
        submatrix_irows = np.compress(submatrix_mask, jac.row)
        submatrix_jcols = np.compress(submatrix_mask, jac.col)
        submatrix_data = np.compress(submatrix_mask, jac.data)

        # ToDo: this is expensive - have to think about how to do this with numpy
        row_submatrix_map = {j:i for i,j in enumerate(constraint_indices)}
        for i, v in enumerate(submatrix_irows):
            submatrix_irows[i] = row_submatrix_map[v]

        col_submatrix_map = {j:i for i,j in enumerate(primal_indices)}
        for i, v in enumerate(submatrix_jcols):
            submatrix_jcols[i] = col_submatrix_map[v]

        return coo_matrix((submatrix_data, (submatrix_irows, submatrix_jcols)), shape=(len(constraint_indices), len(primal_indices)))

    def extract_submatrix_hessian_lag(self, pyomo_variables_rows, pyomo_variables_cols):
        """
        Return the submatrix of the hessian of the lagrangian that
        corresponds to the list of Pyomo variables provided

        Parameters
        ----------
        pyomo_variables_rows : list of Pyomo Var or VarData objects
            List of Pyomo Var or VarData objects corresponding to the desired rows
        pyomo_variables_cols : list of Pyomo Var or VarData objects
            List of Pyomo Var or VarData objects corresponding to the desired columns
        """
        hess_lag = self.evaluate_hessian_lag()
        primal_indices_rows = self.get_primal_indices(pyomo_variables_rows)
        primal_indices_cols = self.get_primal_indices(pyomo_variables_cols)
        row_mask = np.isin(hess_lag.row, primal_indices_rows)
        col_mask = np.isin(hess_lag.col, primal_indices_cols)
        submatrix_mask = row_mask & col_mask
        submatrix_irows = np.compress(submatrix_mask, hess_lag.row)
        submatrix_jcols = np.compress(submatrix_mask, hess_lag.col)
        submatrix_data = np.compress(submatrix_mask, hess_lag.data)

        # ToDo: this is expensive - have to think about how to do this with numpy
        submatrix_map = {j:i for i,j in enumerate(primal_indices_rows)}
        for i, v in enumerate(submatrix_irows):
            submatrix_irows[i] = submatrix_map[v]

        submatrix_map = {j:i for i,j in enumerate(primal_indices_cols)}
        for i, v in enumerate(submatrix_jcols):
            submatrix_jcols[i] = submatrix_map[v]

        return coo_matrix((submatrix_data, (submatrix_irows, submatrix_jcols)), shape=(len(primal_indices_rows), len(primal_indices_cols)))

    def load_state_into_pyomo(self, bound_multipliers=None):
        primals = self.get_primals()
        variables = self.get_pyomo_variables()
        for var, val in zip(variables, primals):
            var.set_value(val)
        m = self.pyomo_model()
        model_suffixes = dict(
            pyo.suffix.active_import_suffix_generator(m))
        if 'dual' in model_suffixes:
            duals = self.get_duals()
            constraints = self.get_pyomo_constraints()
            model_suffixes['dual'].clear()
            model_suffixes['dual'].update(
                zip(constraints, duals))
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

    def variable_names(self):
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
