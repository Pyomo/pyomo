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
from pyomo.contrib.pynumero.interfaces.ampl_nlp import AslNLP
import pyutilib

from scipy.sparse import coo_matrix
import numpy as np
import six

try:
    import poek as pk
    poek_available=True
    __all__ = ['PoekNLP_NLP']
except:
    poek_available=False
    __all__ = []


class PoekNL_NLP(AslNLP):

    def __init__(self, poek_nlpmodel):
        """
        Poek nonlinear program interface that uses NL files

        Parameters
        ----------
        poek_nlpmodel: 
            Poek nlp_model
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
            if poek_nlpmodel.num_objectives() != 1:
                raise NotImplementedError('The ASL interface and PoekNL_NLP in PyNumero currently only support single objective problems.')

            # write the nl file for the model
            vtmp = pk.STLMapIntInt()     # NL index -> POEK varid
            ctmp = pk.STLMapIntInt()     # NL index -> POEK conid
            poek_nlpmodel.write(nl_file, vtmp, ctmp)
            #print("vtmp", vtmp)
            #print("ctmp", ctmp)

            nlpvarmap = {}
            for i in range(poek_nlpmodel.num_variables()):
                nlpvarmap[poek_nlpmodel.get_variable(i).id] = i
            #print("nlpvarmap",nlpvarmap)
            self.varmap = {}            # NL index -> POEK NLP variable no.
            self.invvarmap = {}         # POEK varid -> NL index
            for nli, ndx in vtmp.items():
                self.varmap[nli] = nlpvarmap[ndx]
                self.invvarmap[ndx] = nli
            #print("varmap", self.varmap)
            #print("invvarmap", self.invvarmap)

            nlpconmap = {}
            for i in range(poek_nlpmodel.num_constraints()):
                nlpconmap[poek_nlpmodel.get_constraint(i).id] = i
            #print("nlpconmap",nlpconmap)
            self.conmap = {}            # NL index -> POEK NLP constraint no.
            self.invconmap = {}         # POEK conid -> NL index
            for nli, ndx in ctmp.items():
                self.conmap[nli] = nlpconmap[ndx]
                self.invconmap[ndx] = nli
            #print("conmap", self.conmap)
            #print("invconmap", self.invconmap)

            # now call the AslNLP with the newly created nl_file
            super(PoekNL_NLP, self).__init__(nl_file)

            # keep model in cache
            self._poek_nlpmodel = poek_nlpmodel

        finally:
            # delete the nl file
            pyutilib.services.TempfileManager.pop()

    def poek_nlpmodel(self):
        """
        Return optimization model
        """
        return self._poek_nlpmodel

    def get_poek_variables(self):
        """
        Return an ordered list of the Poek variable objects in
        the order corresponding to the primals
        """
        return [self._poek_nlpmodel.get_variable(self.varmap[i]) for i in range(len(self.varmap))]

    def get_poek_constraints(self):
        """
        Return an ordered list of the Poek constraint objects in
        the order corresponding to the duals
        """
        return [self._poek_nlpmodel.get_constraint(self.conmap[i]) for i in range(len(self.conmap))]

    def variable_names(self):
        """
        Return an ordered list of the variable
        names in the order corresponding to the primals
        """
        return [v.name for v in self.get_poek_variables()]

    def constraint_names(self):
        """
        Return an ordered list of the constraint
        names in the order corresponding to duals
        """
        return [c.name for c in self.get_poek_constraints()]

    def get_primal_indices(self, poek_variables):
        """
        Return the list of indices for the primals
        corresponding to the list of Poek variables provided

        Parameters
        ----------
        poek_variables : list of Poek variable objects
        """
        assert isinstance(poek_variables, list)
        return [self.invvarmap[v.id] for v in poek_variables]

    def get_constraint_indices(self, poek_constraints):
        """
        Return the list of indices for the constraints
        corresponding to the list of Poek constraints provided

        Parameters
        ----------
        poek_constraints : list of Poek constraints
        """
        assert isinstance(poek_constraints, list)
        return [self.invconmap[c.id] for c in poek_constraints]

    def extract_subvector_grad_objective(self, poek_variables):
        """Compute the gradient of the objective and return the entries
        corresponding to the given Poek variables

        Parameters
        ----------
        poek_variables : list of Poek variable objects
        """
        grad_obj = self.evaluate_grad_objective()
        #print(grad_obj)
        return grad_obj[self.get_primal_indices(poek_variables)]

    def extract_subvector_constraints(self, poek_constraints):
        """
        Return the values of the constraints
        corresponding to the list of Poek constraints provided

        Parameters
        ----------
        poek_constraints : list of Poek constraint objects
        """
        residuals = self.evaluate_constraints()
        return residuals[self.get_constraint_indices(poek_constraints)]

    def extract_submatrix_jacobian(self, poek_variables, poek_constraints):
        """
        Return the submatrix of the jacobian that corresponds to the list
        of Poek variables and list of Poek constraints provided

        Parameters
        ----------
        poek_variables : list of Poek variable objects
        poek_constraints : list of Poek constraint objects
        """
        jac = self.evaluate_jacobian()
        primal_indices = self.get_primal_indices(poek_variables)
        constraint_indices = self.get_constraint_indices(poek_constraints)
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

    def extract_submatrix_hessian_lag(self, poek_variables_rows, poek_variables_cols):
        """
        Return the submatrix of the hessian of the lagrangian that 
        corresponds to the list of Poek variables provided

        Parameters
        ----------
        poek_variables_rows : list of Poek variable objects
            List of poek variable objects corresponding to the desired rows
        poek_variables_cols : list of Poek variable objects
            List of Poek variable objects corresponding to the desired columns
        """
        hess_lag = self.evaluate_hessian_lag()
        primal_indices_rows = self.get_primal_indices(poek_variables_rows)
        primal_indices_cols = self.get_primal_indices(poek_variables_cols)
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
    

