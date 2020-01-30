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
import pyomo
import pyomo.environ as aml
from pyomo.contrib.pynumero.interfaces.ampl_nlp import AslNLP
import pyutilib

from scipy.sparse import coo_matrix
import numpy as np
import six

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
        pyutilib.services.TempfileManager.push()
        try:
            # get the temp file names for the nl file
            nl_file = pyutilib.services.TempfileManager.create_tempfile(suffix='pynumero.nl')

            
            # The current AmplInterface code only supports a single objective function
            # Therefore, we throw an error if there is not one (and only one) active
            # objective function. This is better than adding a dummy objective that the
            # user does not know about (since we do nnot have a good place to remove
            # this objective later
            # TODO: extend the AmplInterface and the AslNLP to correctly handle this
            # This currently addresses issue #1217
            objectives = list(pyomo_model.component_data_objects(ctype=aml.Objective, active=True, descend_into=True))
            if len(objectives) != 1:
                raise NotImplementedError('The ASL interface and PyomoNLP in PyNumero currently only support single objective'
                                          ' problems. Deactivate any extra objectives you may have, or add a dummy objective'
                                          ' (f(x)=0) if you have a square problem.')

            # write the nl file for the Pyomo model and get the symbolMap
            fname, symbolMap = pyomo.opt.WriterFactory('nl')(pyomo_model, nl_file, lambda x:True, {})
            
            # create component maps from vardata to idx and condata to idx
            self._vardata_to_idx = vdidx = pyomo.core.kernel.component_map.ComponentMap()
            self._condata_to_idx = cdidx = pyomo.core.kernel.component_map.ComponentMap()

            # TODO: Are these names totally consistent?
            for name, obj in six.iteritems(symbolMap.bySymbol):
                if name[0] == 'v':
                    vdidx[obj()] = int(name[1:])
                elif name[0] == 'c':
                    cdidx[obj()] = int(name[1:])

            # now call the AslNLP with the newly created nl_file
            super(PyomoNLP, self).__init__(nl_file)

            # keep pyomo model in cache
            self._pyomo_model = pyomo_model

        finally:
            # delete the nl file
            pyutilib.services.TempfileManager.pop()

    def pyomo_model(self):
        """
        Return optimization model
        """
        return self._pyomo_model

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

    def variable_names(self):
        """
        Return an ordered list of the Pyomo variable
        names in the order corresponding to the primals
        """
        pyomo_variables = self.get_pyomo_variables()
        return [v.getname() for v in pyomo_variables]

    def constraint_names(self):
        """
        Return an ordered list of the Pyomo constraint
        names in the order corresponding to internal constraint order
        """
        pyomo_constraints = self.get_pyomo_constraints()
        return [v.getname() for v in pyomo_constraints]

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
    

