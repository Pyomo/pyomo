#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
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

from scipy.sparse import coo_matrix
from pyomo.common.deprecation import deprecated
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import WriterFactory
import pyomo.core.base as pyo
from pyomo.common.collections import ComponentMap
from pyomo.common.env import CtypesEnviron
from pyomo.solvers.amplfunc_merge import amplfunc_merge
from ..sparse.block_matrix import BlockMatrix
from pyomo.contrib.pynumero.interfaces.ampl_nlp import AslNLP
from pyomo.contrib.pynumero.interfaces.nlp import NLP
from pyomo.core.base.suffix import SuffixFinder
from .external_grey_box import ExternalGreyBoxBlock


# TODO: There are todos in the code below
class PyomoNLP(AslNLP):
    def __init__(self, pyomo_model, nl_file_options=None):
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
            nl_file = TempfileManager.create_tempfile(suffix='pynumero.nl')

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
            objectives = list(
                pyomo_model.component_data_objects(
                    ctype=pyo.Objective, active=True, descend_into=True
                )
            )
            if len(objectives) != 1:
                raise NotImplementedError(
                    'The ASL interface and PyomoNLP in PyNumero currently '
                    'only support single objective problems. Deactivate '
                    'any extra objectives you may have, or add a dummy '
                    'objective (f(x)=0) if you have a square problem '
                    '(found %s objectives).' % (len(objectives),)
                )
            self._objective = objectives[0]

            # write the nl file for the Pyomo model and get the symbolMap
            if nl_file_options is None:
                nl_file_options = dict()
            fname, symbolMap = WriterFactory('nl')(
                pyomo_model, nl_file, lambda x: True, nl_file_options
            )
            self._symbol_map = symbolMap

            # create component maps from vardata to idx and condata to idx
            self._vardata_to_idx = vdidx = ComponentMap()
            self._condata_to_idx = cdidx = ComponentMap()

            # TODO: Are these names totally consistent?
            for name, obj in symbolMap.bySymbol.items():
                if name[0] == 'v':
                    vdidx[obj] = int(name[1:])
                elif name[0] == 'c':
                    cdidx[obj] = int(name[1:])

            # The NL writer advertises the external function libraries
            # through the PYOMO_AMPLFUNC environment variable; merge it
            # with any preexisting AMPLFUNC definitions
            amplfunc = amplfunc_merge(os.environ)

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
                for con, i in self._condata_to_idx.items()
                if equality_mask[i]
            )
            full_to_inequality = self._con_full_ineq_map
            inequality_mask = self._con_full_ineq_mask
            self._condata_to_ineq_idx = ComponentMap(
                (con, full_to_inequality[i])
                for con, i in self._condata_to_idx.items()
                if inequality_mask[i]
            )

        finally:
            # delete the nl file
            TempfileManager.pop()

    @property
    def symbol_map(self):
        return self._symbol_map

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
        idx_to_vardata = {i: v for v, i in self._vardata_to_idx.items()}
        return [idx_to_vardata[i] for i in range(len(idx_to_vardata))]

    def get_pyomo_constraints(self):
        """
        Return an ordered list of the Pyomo ConData objects in
        the order corresponding to the primals
        """
        # ToDo: is there a more efficient way to do this
        idx_to_condata = {i: v for v, i in self._condata_to_idx.items()}
        return [idx_to_condata[i] for i in range(len(idx_to_condata))]

    def get_pyomo_equality_constraints(self):
        """
        Return an ordered list of the Pyomo ConData objects in
        the order corresponding to the equality constraints.
        """
        idx_to_condata = {i: c for c, i in self._condata_to_eq_idx.items()}
        return [idx_to_condata[i] for i in range(len(idx_to_condata))]

    def get_pyomo_inequality_constraints(self):
        """
        Return an ordered list of the Pyomo ConData objects in
        the order corresponding to the inequality constraints.
        """
        idx_to_condata = {i: c for c, i in self._condata_to_ineq_idx.items()}
        return [idx_to_condata[i] for i in range(len(idx_to_condata))]

    @deprecated(
        msg='This method has been replaced with primals_names',
        version='6.0.0',
        remove_in='6.0',
    )
    def variable_names(self):
        return self.primals_names()

    def primals_names(self):
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
        scaling_finder = SuffixFinder(
            'scaling_factor', default=1.0, context=self._pyomo_model
        )
        val = scaling_finder.find(self.get_pyomo_objective())
        if not scaling_finder.all_suffixes:
            return None
        return val

    # overloaded from NLP
    def get_primals_scaling(self):
        scaling_finder = SuffixFinder(
            'scaling_factor', default=1.0, context=self._pyomo_model
        )
        primals_scaling = np.fromiter(
            (scaling_finder.find(v) for v in self.get_pyomo_variables()),
            count=self.n_primals(),
            dtype=float,
        )
        if not scaling_finder.all_suffixes:
            return None
        return primals_scaling

    # overloaded from NLP
    def get_constraints_scaling(self):
        scaling_finder = SuffixFinder(
            'scaling_factor', default=1.0, context=self._pyomo_model
        )
        constraints_scaling = np.fromiter(
            (scaling_finder.find(v) for v in self.get_pyomo_constraints()),
            count=self.n_constraints(),
            dtype=float,
        )
        if not scaling_finder.all_suffixes:
            return None
        return constraints_scaling

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
        row_submatrix_map = {j: i for i, j in enumerate(constraint_indices)}
        for i, v in enumerate(submatrix_irows):
            submatrix_irows[i] = row_submatrix_map[v]

        col_submatrix_map = {j: i for i, j in enumerate(primal_indices)}
        for i, v in enumerate(submatrix_jcols):
            submatrix_jcols[i] = col_submatrix_map[v]

        return coo_matrix(
            (submatrix_data, (submatrix_irows, submatrix_jcols)),
            shape=(len(constraint_indices), len(primal_indices)),
        )

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
        submatrix_map = {j: i for i, j in enumerate(primal_indices_rows)}
        for i, v in enumerate(submatrix_irows):
            submatrix_irows[i] = submatrix_map[v]

        submatrix_map = {j: i for i, j in enumerate(primal_indices_cols)}
        for i, v in enumerate(submatrix_jcols):
            submatrix_jcols[i] = submatrix_map[v]

        return coo_matrix(
            (submatrix_data, (submatrix_irows, submatrix_jcols)),
            shape=(len(primal_indices_rows), len(primal_indices_cols)),
        )

    def load_state_into_pyomo(self, bound_multipliers=None):
        primals = self.get_primals()
        variables = self.get_pyomo_variables()
        for var, val in zip(variables, primals):
            var.set_value(val)
        m = self.pyomo_model()
        model_suffixes = dict(pyo.suffix.active_import_suffix_generator(m))
        if 'dual' in model_suffixes:
            duals = self.get_duals()
            constraints = self.get_pyomo_constraints()
            model_suffixes['dual'].clear()
            model_suffixes['dual'].update(zip(constraints, duals))
        if 'ipopt_zL_out' in model_suffixes:
            model_suffixes['ipopt_zL_out'].clear()
            if bound_multipliers is not None:
                model_suffixes['ipopt_zL_out'].update(
                    zip(variables, bound_multipliers[0])
                )
        if 'ipopt_zU_out' in model_suffixes:
            model_suffixes['ipopt_zU_out'].clear()
            if bound_multipliers is not None:
                model_suffixes['ipopt_zU_out'].update(
                    zip(variables, bound_multipliers[1])
                )


# TODO: look for the [:-i] when i might be zero
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
                ExternalGreyBoxBlock, descend_into=True
            ):
                greybox.parent_block().reclassify_component_type(greybox, pyo.Block)
                greybox_components.append(greybox)

            self._pyomo_model = pyomo_model
            self._pyomo_nlp = PyomoNLP(pyomo_model)

        finally:
            # Restore the ctypes of the ExternalGreyBoxBlock components
            for greybox in greybox_components:
                greybox.parent_block().reclassify_component_type(
                    greybox, ExternalGreyBoxBlock
                )

        # get the greybox block data objects
        greybox_data = []
        for greybox in greybox_components:
            greybox_data.extend(data for data in greybox.values() if data.active)

        if len(greybox_data) > 1:
            raise NotImplementedError(
                "The PyomoGreyBoxModel interface has not"
                " been tested with Pyomo models that contain"
                " more than one ExternalGreyBoxBlock. Currently,"
                " only a single block is supported."
            )

        if self._pyomo_nlp.n_primals() == 0:
            raise ValueError(
                "No variables were found in the Pyomo part of the model."
                " PyomoGreyBoxModel requires at least one variable"
                " to be active in a Pyomo objective or constraint"
            )

        # check that the greybox model supports what we would expect
        # TODO: add support for models that do not provide jacobians
        """
        for data in greybox_data:
            c = data._ex_model.model_capabilities()
            if (c.n_equality_constraints() > 0 \
               and not c.supports_jacobian_equality_constraints) \
               or (c.n_equality_constraints() > 0 \
               and not c.supports_jacobian_equality_constraints)
                raise NotImplementedError('PyomoGreyBoxNLP does not support models'
                                          ' without explicit Jacobian support')
        """

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
            for v in data.inputs.values():
                if v.fixed:
                    raise NotImplementedError(
                        'Found a grey box model input that is fixed: {}.'
                        ' This interface does not currently support fixed'
                        ' variables. Please add a constraint instead.'
                        ''.format(v.getname(fully_qualified=True))
                    )
            for v in data.outputs.values():
                if v.fixed:
                    raise NotImplementedError(
                        'Found a grey box model output that is fixed: {}.'
                        ' This interface does not currently support fixed'
                        ' variables. Please add a constraint instead.'
                        ''.format(v.getname(fully_qualified=True))
                    )

            block_name = data.getname()
            for nm in data._ex_model.equality_constraint_names():
                self._greybox_constraints_names.append('{}.{}'.format(block_name, nm))
            for nm in data._ex_model.output_names():
                self._greybox_constraints_names.append(
                    '{}.{}_con'.format(block_name, nm)
                )

            for var in data.component_data_objects(pyo.Var):
                if var not in self._vardata_to_idx:
                    # there is a variable in the greybox block that
                    # is not in the NL - append this to the end
                    self._vardata_to_idx[var] = n_primals
                    n_primals += 1
                    greybox_primals.append(var)
                    self._greybox_primals_names.append(
                        var.getname(fully_qualified=True)
                    )
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

        # create the helper objects
        con_offset = self._pyomo_nlp.n_constraints()
        self._external_greybox_helpers = []
        for data in greybox_data:
            h = _ExternalGreyBoxModelHelper(data, self._vardata_to_idx, con_offset)
            self._external_greybox_helpers.append(h)
            con_offset += h.n_residuals()

        self._n_greybox_constraints = con_offset - self._pyomo_nlp.n_constraints()
        assert len(self._greybox_constraints_names) == self._n_greybox_constraints

        # make sure the primal values get to the greybox models
        self.set_primals(self.get_primals())

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

        scaling_finder = SuffixFinder(
            'scaling_factor', default=1.0, context=self._pyomo_model
        )
        self._primals_scaling = np.fromiter(
            (scaling_finder.find(v) for v in self.get_pyomo_variables()),
            count=self.n_primals(),
            dtype=float,
        )
        need_scaling = bool(scaling_finder.all_suffixes)

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
        self._init_greybox_duals = np.zeros(self._n_greybox_constraints)
        self._init_greybox_primals.flags.writeable = False
        self._init_greybox_duals.flags.writeable = False
        self._greybox_duals = self._init_greybox_duals.copy()

        # compute the jacobian for the external greybox models
        # to get some of the statistics
        self._evaluate_greybox_jacobians_and_cache_if_necessary()
        self._nnz_greybox_jac = len(self._cached_greybox_jac.data)

        # make sure the duals get to the external greybox models
        self.set_duals(self.get_duals())

        # compute the hessian for the external greybox models
        # to get some of the statistics
        try:
            self._evaluate_greybox_hessians_and_cache_if_necessary()
            self._nnz_greybox_hess = len(self._cached_greybox_hess.data)
        except (AttributeError, NotImplementedError):
            self._nnz_greybox_hess = None

    def _invalidate_greybox_primals_cache(self):
        self._greybox_constraints_cached = False
        self._greybox_jac_cached = False
        self._greybox_hess_cached = False

    def _invalidate_greybox_duals_cache(self):
        self._greybox_hess_cached = False

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
        return self._pyomo_nlp.nnz_hessian_lag() + self._nnz_greybox_hess

    # overloaded from NLP
    def primals_lb(self):
        return np.concatenate((self._pyomo_nlp.primals_lb(), self._greybox_primals_lb))

    # overloaded from NLP
    def primals_ub(self):
        return np.concatenate((self._pyomo_nlp.primals_ub(), self._greybox_primals_ub))

    # overloaded from NLP
    def constraints_lb(self):
        return np.concatenate(
            (
                self._pyomo_nlp.constraints_lb(),
                np.zeros(self._n_greybox_constraints, dtype=np.float64),
            )
        )

    # overloaded from NLP
    def constraints_ub(self):
        return np.concatenate(
            (
                self._pyomo_nlp.constraints_ub(),
                np.zeros(self._n_greybox_constraints, dtype=np.float64),
            )
        )

    # overloaded from NLP
    def init_primals(self):
        return np.concatenate(
            (self._pyomo_nlp.init_primals(), self._init_greybox_primals)
        )

    # overloaded from NLP
    def init_duals(self):
        return np.concatenate((self._pyomo_nlp.init_duals(), self._init_greybox_duals))

    # overloaded from ExtendedNLP
    def init_duals_eq(self):
        return np.concatenate(
            (self._pyomo_nlp.init_duals_eq(), self._init_greybox_duals)
        )

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
        self._pyomo_nlp.set_primals(primals[: self._pyomo_nlp.n_primals()])

        # copy the values for the greybox primals
        np.copyto(self._greybox_primals, primals[self._pyomo_nlp.n_primals() :])

        for external in self._external_greybox_helpers:
            external.set_primals(primals)

    # overloaded from AslNLP
    def get_primals(self):
        # return the value of the primals that the pyomo
        # part knows about as well as any extra values that
        # are only in the greybox part
        return np.concatenate((self._pyomo_nlp.get_primals(), self._greybox_primals))

    # overloaded from NLP
    def set_duals(self, duals):
        self._invalidate_greybox_duals_cache()

        # set the duals for the pyomo part of the nlp
        self._pyomo_nlp.set_duals(duals[: self._pyomo_nlp.n_constraints()])

        # set the duals for the greybox part of the nlp
        np.copyto(self._greybox_duals, duals[self._pyomo_nlp.n_constraints() :])

        # set the duals in the helpers for the hessian computation
        for h in self._external_greybox_helpers:
            h.set_duals(duals)

    # overloaded from NLP
    def get_duals(self):
        # return the duals for the pyomo part of the nlp
        # concatenated with the greybox part
        return np.concatenate((self._pyomo_nlp.get_duals(), self._greybox_duals))

    # overloaded from ExtendedNLP
    def set_duals_eq(self, duals):
        raise NotImplementedError('set_duals_eq not implemented for PyomoGreyBoxNLP')
        # we think the code below is correct, but it has not yet been tested
        """
        #self._invalidate_greybox_duals_cache()

        # set the duals for the pyomo part of the nlp
        self._pyomo_nlp.set_duals_eq(
            duals[:self._pyomo_nlp.n_equality_constraints()])

        # set the duals for the greybox part of the nlp
        np.copyto(self._greybox_duals, duals[self._pyomo_nlp.n_equality_constraints():])
        # set the duals in the helpers for the hessian computation
        for h in self._external_greybox_helpers:
            h.set_duals_eq(duals)
        """

    # TODO: Implement set_duals_ineq

    # overloaded from NLP
    def get_duals_eq(self):
        raise NotImplementedError('get_duals_eq not implemented for PyomoGreyBoxNLP')
        """
        # return the duals for the pyomo part of the nlp
        # concatenated with the greybox part
        return np.concatenate((
            self._pyomo_nlp.get_duals_eq(),
            self._greybox_duals,
        ))
        """

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
        return np.concatenate(
            (
                self._pyomo_nlp.evaluate_grad_objective(out),
                np.zeros(self._n_greybox_primals),
            )
        )

    def _evaluate_greybox_constraints_and_cache_if_necessary(self):
        if self._greybox_constraints_cached:
            return

        self._cached_greybox_constraints = np.concatenate(
            tuple(
                external.evaluate_residuals()
                for external in self._external_greybox_helpers
            )
        )
        self._greybox_constraints_cached = True

    # overloaded from NLP
    def evaluate_constraints(self, out=None):
        self._evaluate_greybox_constraints_and_cache_if_necessary()

        if out is not None:
            if not isinstance(out, np.ndarray) or out.size != self.n_constraints():
                raise RuntimeError(
                    'Called evaluate_constraints with an invalid'
                    ' "out" argument - should take an ndarray of '
                    'size {}'.format(self.n_constraints())
                )

            # call on the pyomo part of the nlp
            self._pyomo_nlp.evaluate_constraints(out[: -self._n_greybox_constraints])

            # call on the greybox part of the nlp
            np.copyto(
                out[-self._n_greybox_constraints :], self._cached_greybox_constraints
            )
            return out

        else:
            # concatenate the pyomo and external constraint residuals
            return np.concatenate(
                (
                    self._pyomo_nlp.evaluate_constraints(),
                    self._cached_greybox_constraints,
                )
            )

    # overloaded from ExtendedNLP
    def evaluate_eq_constraints(self, out=None):
        raise NotImplementedError('Not yet implemented for PyomoGreyBoxNLP')
        """
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
        """

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
            if (
                not isinstance(out, coo_matrix)
                or out.shape[0] != self.n_constraints()
                or out.shape[1] != self.n_primals()
                or out.nnz != self.nnz_jacobian()
            ):
                raise RuntimeError(
                    'evaluate_jacobian called with an "out" argument'
                    ' that is invalid. This should be a coo_matrix with'
                    ' shape=({},{}) and nnz={}'.format(
                        self.n_constraints(), self.n_primals(), self.nnz_jacobian()
                    )
                )
            n_pyomo_constraints = self.n_constraints() - self._n_greybox_constraints

            # to avoid an additional copy, we pass in a slice (numpy view) of the underlying
            # data, row, and col that we were handed to be populated in evaluate_jacobian
            self._pyomo_nlp.evaluate_jacobian(
                out=coo_matrix(
                    (
                        out.data[: -self._nnz_greybox_jac],
                        (
                            out.row[: -self._nnz_greybox_jac],
                            out.col[: -self._nnz_greybox_jac],
                        ),
                    ),
                    shape=(n_pyomo_constraints, self._pyomo_nlp.n_primals()),
                )
            )
            np.copyto(out.data[-self._nnz_greybox_jac :], self._cached_greybox_jac.data)
            return out
        else:
            base = self._pyomo_nlp.evaluate_jacobian()
            base = coo_matrix(
                (base.data, (base.row, base.col)),
                shape=(base.shape[0], self.n_primals()),
            )

            jac = BlockMatrix(2, 1)
            jac.set_block(0, 0, base)
            jac.set_block(1, 0, self._cached_greybox_jac)
            return jac.tocoo()

            # TODO: Doesn't this need a "shape" specification?
            # return coo_matrix((
            #    np.concatenate((base.data, self._cached_greybox_jac.data)),
            #    ( np.concatenate((base.row, self._cached_greybox_jac.row)),
            #      np.concatenate((base.col, self._cached_greybox_jac.col)) )
            # ))

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
    def _evaluate_greybox_hessians_and_cache_if_necessary(self):
        if self._greybox_hess_cached:
            return

        data = list()
        irow = list()
        jcol = list()
        for external in self._external_greybox_helpers:
            hess = external.evaluate_hessian()
            data.append(hess.data)
            irow.append(hess.row)
            jcol.append(hess.col)

        data = np.concatenate(data)
        irow = np.concatenate(irow)
        jcol = np.concatenate(jcol)

        self._cached_greybox_hess = coo_matrix(
            (data, (irow, jcol)), shape=(self.n_primals(), self.n_primals())
        )
        self._greybox_hess_cached = True

    def evaluate_hessian_lag(self, out=None):
        self._evaluate_greybox_hessians_and_cache_if_necessary()
        if out is not None:
            if (
                not isinstance(out, coo_matrix)
                or out.shape[0] != self.n_primals()
                or out.shape[1] != self.n_primals()
                or out.nnz != self.nnz_hessian_lag()
            ):
                raise RuntimeError(
                    'evaluate_hessian_lag called with an "out" argument'
                    ' that is invalid. This should be a coo_matrix with'
                    ' shape=({},{}) and nnz={}'.format(
                        self.n_primals(), self.n_primals(), self.nnz_hessian()
                    )
                )
            # to avoid an additional copy, we pass in a slice (numpy view) of the underlying
            # data, row, and col that we were handed to be populated in evaluate_hessian_lag
            # the coo_matrix is simply a holder of the data, row, and col structures
            self._pyomo_nlp.evaluate_hessian_lag(
                out=coo_matrix(
                    (
                        out.data[: -self._nnz_greybox_hess],
                        (
                            out.row[: -self._nnz_greybox_hess],
                            out.col[: -self._nnz_greybox_hess],
                        ),
                    ),
                    shape=(self._pyomo_nlp.n_primals(), self._pyomo_nlp.n_primals()),
                )
            )
            np.copyto(
                out.data[-self._nnz_greybox_hess :], self._cached_greybox_hess.data
            )
            return out
        else:
            hess = self._pyomo_nlp.evaluate_hessian_lag()
            data = np.concatenate((hess.data, self._cached_greybox_hess.data))
            row = np.concatenate((hess.row, self._cached_greybox_hess.row))
            col = np.concatenate((hess.col, self._cached_greybox_hess.col))
            hess = coo_matrix(
                (data, (row, col)), shape=(self.n_primals(), self.n_primals())
            )
            return hess

    # overloaded from NLP
    def report_solver_status(self, status_code, status_message):
        raise NotImplementedError('Todo: implement this')

    @deprecated(
        msg='This method has been replaced with primals_names',
        version='6.0.0',
        remove_in='6.0',
    )
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
        return self._pyomo_nlp.get_pyomo_variables() + self._greybox_primal_variables

    def get_pyomo_constraints(self):
        """
        Return an ordered list of the Pyomo ConData objects in
        the order corresponding to the primals
        """
        # FIXME: what do we return for the external block constraints?
        # return self._pyomo_nlp.get_pyomo_constraints()
        raise NotImplementedError(
            "returning list of all constraints when using an external model is TBD"
        )

    def load_state_into_pyomo(self, bound_multipliers=None):
        primals = self.get_primals()
        variables = self.get_pyomo_variables()
        for var, val in zip(variables, primals):
            var.set_value(val)
        m = self.pyomo_model()
        model_suffixes = dict(pyo.suffix.active_import_suffix_generator(m))
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
                    zip(variables, bound_multipliers[0])
                )
        if 'ipopt_zU_out' in model_suffixes:
            model_suffixes['ipopt_zU_out'].clear()
            if bound_multipliers is not None:
                model_suffixes['ipopt_zU_out'].update(
                    zip(variables, bound_multipliers[1])
                )


class _ExternalGreyBoxModelHelper(object):
    def __init__(self, ex_grey_box_block, vardata_to_idx, con_offset):
        """This helper takes an ExternalGreyBoxModel and provides the residual,
        Jacobian, and Hessian computation mapped to the correct variable space.

        The ExternalGreyBoxModel provides an interface that supports
        equality constraints (pure residuals) and output equations. Let
        u be the inputs, o be the outputs, and x be the full set of
        primal variables from the entire pyomo_nlp.

        With this, the ExternalGreyBoxModel provides the residual
        computations w_eq(u), and w_o(u), as well as the Jacobians,
        Jw_eq(u), and Jw_o(u). This helper provides h(x)=0, where h(x) =
        [h_eq(x); h_o(x)-o] and h_eq(x)=w_eq(Pu*x), and
        h_o(x)=w_o(Pu*x), and Pu is a mapping from the full primal
        variables "x" to the inputs "u".

        It also provides the Jacobian of h w.r.t. x.
           J_h(x) = [Jw_eq(Pu*x); Jw_o(Pu*x)-Po*x]
        where Po is a mapping from the full primal variables "x" to the
        outputs "o".

        """
        self._block = ex_grey_box_block
        self._ex_model = ex_grey_box_block.get_external_model()
        self._n_primals = len(vardata_to_idx)
        n_inputs = len(self._block.inputs)
        assert n_inputs == self._ex_model.n_inputs()
        n_eq_constraints = self._ex_model.n_equality_constraints()
        n_outputs = len(self._block.outputs)
        assert n_outputs == self._ex_model.n_outputs()

        # store the map of input indices (0 .. n_inputs) to
        # the indices in the full primals vector
        self._inputs_to_primals_map = np.fromiter(
            (vardata_to_idx[v] for v in self._block.inputs.values()),
            dtype=np.int64,
            count=n_inputs,
        )

        # store the map of output indices (0 .. n_outputs) to
        # the indices in the full primals vector
        self._outputs_to_primals_map = np.fromiter(
            (vardata_to_idx[v] for v in self._block.outputs.values()),
            dtype=np.int64,
            count=n_outputs,
        )

        if (
            self._ex_model.n_outputs() == 0
            and self._ex_model.n_equality_constraints() == 0
        ):
            raise ValueError(
                'ExternalGreyBoxModel has no equality constraints '
                'or outputs. It must have at least one or both.'
            )

        self._ex_eq_duals_to_full_map = None
        if n_eq_constraints > 0:
            self._ex_eq_duals_to_full_map = list(
                range(con_offset, con_offset + n_eq_constraints)
            )

        self._ex_output_duals_to_full_map = None
        if n_outputs > 0:
            self._ex_output_duals_to_full_map = list(
                range(
                    con_offset + n_eq_constraints,
                    con_offset + n_eq_constraints + n_outputs,
                )
            )

        # we need to change the column indices in the jacobian
        # from the 0..n_inputs provided by the external model
        # to the indices corresponding to the full Pyomo model
        # so we create that here
        self._eq_jac_primal_jcol = None
        self._outputs_jac_primal_jcol = None
        self._additional_output_entries_irow = None
        self._additional_output_entries_jcol = None
        self._additional_output_entries_data = None
        self._con_offset = con_offset
        self._eq_hess_jcol = None
        self._eq_hess_irow = None
        self._output_hess_jcol = None
        self._output_hess_irow = None

    def set_primals(self, primals):
        # map the full primals "x" to the inputs "u" and set
        # the values on the external model
        input_values = primals[self._inputs_to_primals_map]
        self._ex_model.set_input_values(input_values)

        # map the full primals "x" to the outputs "o" and
        # store a vector of the current output values to
        # use when evaluating residuals
        self._output_values = primals[self._outputs_to_primals_map]

    def set_duals(self, duals):
        # map the full duals to the duals for the equality
        # and the output constraints
        if self._ex_eq_duals_to_full_map is not None:
            duals_eq = duals[self._ex_eq_duals_to_full_map]
            self._ex_model.set_equality_constraint_multipliers(duals_eq)

        if self._ex_output_duals_to_full_map is not None:
            duals_outputs = duals[self._ex_output_duals_to_full_map]
            self._ex_model.set_output_constraint_multipliers(duals_outputs)

    def n_residuals(self):
        return self._ex_model.n_equality_constraints() + self._ex_model.n_outputs()

    def get_residual_scaling(self):
        eq_scaling = self._ex_model.get_equality_constraint_scaling_factors()
        output_con_scaling = self._ex_model.get_output_constraint_scaling_factors()
        if eq_scaling is None and output_con_scaling is None:
            return None
        if eq_scaling is None:
            eq_scaling = np.ones(self._ex_model.n_equality_constraints())
        if output_con_scaling is None:
            output_con_scaling = np.ones(self._ex_model.n_outputs())

        return np.concatenate((eq_scaling, output_con_scaling))

    def evaluate_residuals(self):
        # evaluate the equality constraints and the output equations
        # and return a single vector of residuals
        # returns residual for h(x)=0, where h(x) = [h_eq(x); h_o(x)-o]
        resid_list = []
        if self._ex_model.n_equality_constraints() > 0:
            resid_list.append(self._ex_model.evaluate_equality_constraints())

        if self._ex_model.n_outputs() > 0:
            computed_output_values = self._ex_model.evaluate_outputs()
            output_resid = computed_output_values - self._output_values
            resid_list.append(output_resid)

        return np.concatenate(resid_list)

    def evaluate_jacobian(self):
        # compute the jacobian of h(x) w.r.t. x
        # J_h(x) = [Jw_eq(Pu*x); Jw_o(Pu*x)-Po*x]

        # Jw_eq(x)
        eq_jac = None
        if self._ex_model.n_equality_constraints() > 0:
            eq_jac = self._ex_model.evaluate_jacobian_equality_constraints()
            if self._eq_jac_primal_jcol is None:
                # The first time through, we won't have created the
                # mapping of external primals ('u') to the full space
                # primals ('x')
                self._eq_jac_primal_jcol = self._inputs_to_primals_map[eq_jac.col]
            # map the columns from the inputs "u" back to the full primals "x"
            eq_jac = coo_matrix(
                (eq_jac.data, (eq_jac.row, self._eq_jac_primal_jcol)),
                (eq_jac.shape[0], self._n_primals),
            )

        outputs_jac = None
        if self._ex_model.n_outputs() > 0:
            outputs_jac = self._ex_model.evaluate_jacobian_outputs()

            row = outputs_jac.row
            # map the columns from the inputs "u" back to the full primals "x"
            if self._outputs_jac_primal_jcol is None:
                # The first time through, we won't have created the
                # mapping of external outputs ('o') to the full space
                # primals ('x')
                self._outputs_jac_primal_jcol = self._inputs_to_primals_map[
                    outputs_jac.col
                ]

                # We also need tocreate the irow, jcol, nnz structure for the
                # output variable portion of h(u)-o=0
                self._additional_output_entries_irow = np.asarray(
                    range(self._ex_model.n_outputs())
                )
                self._additional_output_entries_jcol = self._outputs_to_primals_map
                self._additional_output_entries_data = -1.0 * np.ones(
                    self._ex_model.n_outputs()
                )

            col = self._outputs_jac_primal_jcol
            data = outputs_jac.data

            # add the additional entries for the -Po*x portion of the jacobian
            row = np.concatenate((row, self._additional_output_entries_irow))
            col = np.concatenate((col, self._additional_output_entries_jcol))
            data = np.concatenate((data, self._additional_output_entries_data))
            outputs_jac = coo_matrix(
                (data, (row, col)), shape=(outputs_jac.shape[0], self._n_primals)
            )

        jac = None
        if eq_jac is not None:
            if outputs_jac is not None:
                # create a jacobian with both Jw_eq and Jw_o
                jac = BlockMatrix(2, 1)
                jac.name = 'external model jacobian'
                jac.set_block(0, 0, eq_jac)
                jac.set_block(1, 0, outputs_jac)
            else:
                assert self._ex_model.n_outputs() == 0
                assert self._ex_model.n_equality_constraints() > 0
                # only need the Jacobian with Jw_eq (there are not
                # output equations)
                jac = eq_jac
        else:
            assert outputs_jac is not None
            assert self._ex_model.n_outputs() > 0
            assert self._ex_model.n_equality_constraints() == 0
            # only need the Jacobian with Jw_o (there are no equalities)
            jac = outputs_jac

        return jac

    def evaluate_hessian(self):
        # compute the portion of the Hessian of the Lagrangian
        # of h(x) w.r.t. x
        # H_h(x) = sum_i y_eq_i * Hw_eq(Pu*x) +_ sum_k y_o_j * Hw_o(Pu*x)]

        data_list = []
        irow_list = []
        jcol_list = []

        # Hw_eq(x)
        eq_hess = None
        if self._ex_model.n_equality_constraints() > 0:
            eq_hess = self._ex_model.evaluate_hessian_equality_constraints()
            if self._eq_hess_jcol is None:
                # first time through, let's also check that it is lower triangular
                if np.any(eq_hess.row < eq_hess.col):
                    raise ValueError(
                        'ExternalGreyBoxModel must return lower '
                        'triangular portion of the Hessian only'
                    )

                # The first time through, we won't have created the
                # mapping of external primals ('u') to the full space
                # primals ('x')
                self._eq_hess_irow = row = self._inputs_to_primals_map[eq_hess.row]
                self._eq_hess_jcol = col = self._inputs_to_primals_map[eq_hess.col]

                # mapping may have made this not lower triangular
                mask = col > row
                row[mask], col[mask] = col[mask], row[mask]

            data_list.append(eq_hess.data)
            irow_list.append(self._eq_hess_irow)
            jcol_list.append(self._eq_hess_jcol)

        if self._ex_model.n_outputs() > 0:
            output_hess = self._ex_model.evaluate_hessian_outputs()
            if self._output_hess_irow is None:
                # first time through, let's also check that it is lower triangular
                if np.any(output_hess.row < output_hess.col):
                    raise ValueError(
                        'ExternalGreyBoxModel must return lower '
                        'triangular portion of the Hessian only'
                    )

                # The first time through, we won't have created the
                # mapping of external outputs ('o') to the full space
                # primals ('x')
                self._output_hess_irow = row = self._inputs_to_primals_map[
                    output_hess.row
                ]
                self._output_hess_jcol = col = self._inputs_to_primals_map[
                    output_hess.col
                ]

                # mapping may have made this not lower triangular
                mask = col > row
                row[mask], col[mask] = col[mask], row[mask]

            data_list.append(output_hess.data)
            irow_list.append(self._output_hess_irow)
            jcol_list.append(self._output_hess_jcol)

        data = np.concatenate(data_list)
        irow = np.concatenate(irow_list)
        jcol = np.concatenate(jcol_list)
        hess = coo_matrix((data, (irow, jcol)), (self._n_primals, self._n_primals))

        return hess
