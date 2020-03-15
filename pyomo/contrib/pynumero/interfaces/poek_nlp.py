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
This module defines the classes that provide an NLP interface via POEK
"""

#import pyomo
import pyomo.environ as aml
from pyomo.contrib.pynumero.interfaces.nlp import NLP
#from pyomo.contrib.pynumero.interfaces.ampl_nlp import AslNLP
#import pyutilib

#from scipy.sparse import coo_matrix
#import numpy as np
#import six

try:
    import poek
    poek_available=True
    __all__ = ['PoekNLP']
except:
    poek_available=False
    __all__ = []


class PoekNLP(NLP):

    def __init__(self, model):
        """
        POEK nonlinear program interface

        Parameters
        ----------
        model: peok.ConcreteModel
            POEK model
        """
        super(NLP, self).__init__()
        self._model = model

    def n_primals(self):
        """
        Returns number of primal variables
        """
        return self._model.num_variables()

    def n_constraints(self):
        """
        Returns number of constraints
        """
        return self._model.num_constraints()

    def nnz_jacobian(self):
        """
        Returns number of nonzero values in jacobian of equality constraints
        """
        return self._model.nnz_jacobian()

    def nnz_hessian_lag(self):
        """
        Returns number of nonzero values in hessian of the lagrangian function
        """
        return self._model.nnz_hessian_lag()

    def primals_lb(self):
        """
        Returns vector of lower bounds for the primal variables

        Returns
        -------
        vector-like

        """
        pass

    def primals_ub(self):
        """
        Returns vector of upper bounds for the primal variables

        Returns
        -------
        vector-like

        """
        pass

    def constraints_lb(self):
        """
        Returns vector of lower bounds for the constraints

        Returns
        -------
        vector-like

        """
        pass

    def constraints_ub(self):
        """
        Returns vector of upper bounds for the constraints

        Returns
        -------
        vector-like

        """
        pass

    def init_primals(self):
        """
        Returns vector with initial values for the primal variables
        """
        pass

    def init_duals(self):
        """
        Returns vector with initial values for the dual variables 
        of the constraints
        """
        pass

    def create_new_vector(self, vector_type):
        """
        Creates a vector of the appropriate length and structure as 
        requested

        Parameters
        ----------
        vector_type: {'primals', 'constraints', 'duals'}
            String identifying the appropriate  vector  to create.

        Returns
        -------
        vector-like
        """
        pass


    def set_primals(self, primals):
        """Set the value of the primal variables to be used
        in calls to the evaluation methods

        Parameters
        ----------
        primals: vector_like
            Vector with the values of primal variables.
        """
        pass

    def get_primals(self):
        """Get a copy of the values of the primal variables as
        provided in set_primals. These are the values that will
        be used in calls to the evaluation methods
        """
        pass

    def set_duals(self, duals):
        """Set the value of the dual variables for the constraints
        to be used in calls to the evaluation methods (hessian_lag)

        Parameters
        ----------
        duals: vector_like
            Vector with the values of dual variables for the equality constraints
        """
        pass

    def get_duals(self):
        """Get a copy of the values of the dual variables as
        provided in set_duals. These are the values that will
        be used in calls to the evaluation methods.
        """
        pass

    def set_obj_factor(self, obj_factor):
        """Set the value of the objective function factor
        to be used in calls to the evaluation of the hessian
        of the lagrangian (evaluate_hessian_lag)

        Parameters
        ----------
        obj_factor: float
            Value of the objective function factor used
            in the evaluation of the hessian of the lagrangian
        """
        pass

    def get_obj_factor(self):
        """Get the value of the objective function factor as 
        set by set_obj_factor. This is the value that will
        be used in calls to the evaluation of the hessian
        of the lagrangian (evaluate_hessian_lag)
        """
        pass

    def evaluate_objective(self):
        """Returns value of objective function evaluated at the 
        values given for the primal variables in set_primals

        Returns
        -------
        float
        """
        pass

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

    def report_solver_status(self, status_code, status_message):
        """Report the solver status to NLP class using the values for the 
        primals and duals defined in the set methods"""
        pass


    def model(self):
        """
        Return optimization model
        """
        return self._model
