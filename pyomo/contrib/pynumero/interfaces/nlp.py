#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
"""The pyomo.contrib.pynumero.interfaces.nlp module includes abstract
classes to represent nonlinear programming problems. There are two
classes that provide different representations for the NLP.

The first interface (NLP) presents the NLP in the following form
(where all equality and inequality constaints are combined)

minimize             f(x)
subject to    g_L <= g(x) <= g_U
              x_L <=  x   <= x_U

where x \in R^{n_x} are the primal variables,
      x_L \in R^{n_x} are the lower bounds of the primal variables,
      x_U \in R^{n_x} are the uppper bounds of the primal variables,
      g: R^{n_x} \rightarrow R^{n_c} are constraints (combined 
         equality and inequality)
      
The second interface (ExtendedNLP) extends the definition above and
presents the NLP in the following form where the equality and
inequality constraints are separated.

minimize             f(x)
subject to           h(x) = 0
              q_L <= q(x) <= q_U
              x_L <=  x   <= x_U

where x \in R^{n_x} are the primal variables,
      x_L \in R^{n_x} are the lower bounds of the primal variables,
      x_U \in R^{n_x} are the uppper bounds of the primal variables,
      h: R^{n_x} \rightarrow R^{n_eq} are the equality constraints
      q: R^{n_x} \rightarrow R^{n_ineq} are the inequality constraints

Note: In the case of the ExtendedNLP, it is generally assumed that
both the NLP and the ExtendedNLP interfaces are supported and
consistent (ExtendedNLP inherits from NLP). For example, a user should
be able to call set_duals or set_duals_eq and set_duals_ineq, or
mix-and-match between evaluate_jacobian or evaluate_jacobian_eq and
evaluate_jacobian_ineq.

.. rubric:: Contents

"""
import six
import abc

__all__ = ['NLP']

@six.add_metaclass(abc.ABCMeta)
class NLP(object):
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def n_primals(self):
        """
        Returns number of primal variables
        """
        pass

    @abc.abstractmethod
    def n_constraints(self):
        """
        Returns number of constraints
        """
        pass
    
    @abc.abstractmethod
    def nnz_jacobian(self):
        """
        Returns number of nonzero values in jacobian of equality constraints
        """
        pass

    @abc.abstractmethod
    def nnz_hessian_lag(self):
        """
        Returns number of nonzero values in hessian of the lagrangian function
        """
        pass

    @abc.abstractmethod
    def primals_lb(self):
        """
        Returns vector of lower bounds for the primal variables

        Returns
        -------
        vector-like

        """
        pass

    @abc.abstractmethod
    def primals_ub(self):
        """
        Returns vector of upper bounds for the primal variables

        Returns
        -------
        vector-like

        """
        pass

    @abc.abstractmethod
    def constraints_lb(self):
        """
        Returns vector of lower bounds for the constraints

        Returns
        -------
        vector-like

        """
        pass

    @abc.abstractmethod
    def constraints_ub(self):
        """
        Returns vector of upper bounds for the constraints

        Returns
        -------
        vector-like

        """
        pass

    @abc.abstractmethod
    def init_primals(self):
        """
        Returns vector with initial values for the primal variables
        """
        pass

    @abc.abstractmethod
    def init_duals(self):
        """
        Returns vector with initial values for the dual variables 
        of the constraints
        """
        pass

    @abc.abstractmethod
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

    @abc.abstractmethod
    def set_primals(self, primals):
        """Set the value of the primal variables to be used
        in calls to the evaluation methods

        Parameters
        ----------
        primals: vector_like
            Vector with the values of primal variables.
        """
        pass

    @abc.abstractmethod
    def get_primals(self):
        """Get a copy of the values of the primal variables as
        provided in set_primals. These are the values that will
        be used in calls to the evaluation methods
        """
        pass

    @abc.abstractmethod
    def set_duals(self, duals):
        """Set the value of the dual variables for the constraints
        to be used in calls to the evaluation methods (hessian_lag)

        Parameters
        ----------
        duals: vector_like
            Vector with the values of dual variables for the equality constraints
        """
        pass

    @abc.abstractmethod
    def get_duals(self):
        """Get a copy of the values of the dual variables as
        provided in set_duals. These are the values that will
        be used in calls to the evaluation methods.
        """
        pass

    @abc.abstractmethod
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

    @abc.abstractmethod
    def get_obj_factor(self):
        """Get the value of the objective function factor as 
        set by set_obj_factor. This is the value that will
        be used in calls to the evaluation of the hessian
        of the lagrangian (evaluate_hessian_lag)
        """
        pass

    @abc.abstractmethod
    def get_obj_scaling(self):
        """ Return the desired scaling factor to use for the
        for the objective function. None indicates no scaling.
        This indicates potential scaling for the model, but the
        evaluation methods should return *unscaled* values

        Returns
        -------
        float or None
        """
        pass

    @abc.abstractmethod
    def get_primals_scaling(self):
        """ Return the desired scaling factors to use for the
        for the primals. None indicates no scaling.
        This indicates potential scaling for the model, but the
        evaluation methods should return *unscaled* values

        Returns
        -------
        array-like or None
        """
        pass

    @abc.abstractmethod
    def get_constraints_scaling(self):
        """ Return the desired scaling factors to use for the
        for the constraints. None indicates no scaling.
        This indicates potential scaling for the model, but the
        evaluation methods should return *unscaled* values

        Returns
        -------
        array-like or None
        """
        pass

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

@six.add_metaclass(abc.ABCMeta)
class ExtendedNLP(NLP):
    """ This interface extends the NLP interface to support a presentation
    of the problem that separates equality and inequality constraints
    """
    def __init__(self):
        super(ExtendedNLP, self).__init__()
        pass
    
    @abc.abstractmethod
    def n_eq_constraints(self):
        """
        Returns number of equality constraints
        """
        pass
    
    @abc.abstractmethod
    def n_ineq_constraints(self):
        """
        Returns number of inequality constraints
        """
        pass
    
    @abc.abstractmethod
    def nnz_jacobian_eq(self):
        """
        Returns number of nonzero values in jacobian of equality constraints
        """
        pass

    @abc.abstractmethod
    def nnz_jacobian_ineq(self):
        """
        Returns number of nonzero values in jacobian of inequality constraints
        """
        pass

    @abc.abstractmethod
    def ineq_lb(self):
        """
        Returns vector of lower bounds for inequality constraints

        Returns
        -------
        vector-like

        """
        pass

    @abc.abstractmethod
    def ineq_ub(self):
        """
        Returns vector of upper bounds for inequality constraints

        Returns
        -------
        vector-like

        """
        pass

    @abc.abstractmethod
    def init_duals_eq(self):
        """
        Returns vector with initial values for the dual variables of the
        equality constraints
        """
        pass

    @abc.abstractmethod
    def init_duals_ineq(self):
        """
        Returns vector with initial values for the dual variables of the
        inequality constraints
        """
        pass

    @abc.abstractmethod
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
        vector-like
        """
        pass

    @abc.abstractmethod
    def set_duals_eq(self, duals_eq):
        """Set the value of the dual variables for the equality constraints
        to be used in calls to the evaluation methods (hessian_lag)

        Parameters
        ----------
        duals_eq: vector_like
            Vector with the values of dual variables for the equality constraints
        """
        pass

    @abc.abstractmethod
    def get_duals_eq(self):
        """Get a copy of the values of the dual variables of the equality
        constraints as provided in set_duals_eq. These are the values
        that will be used in calls to the evaluation methods.
        """
        pass

    @abc.abstractmethod
    def set_duals_ineq(self, duals_ineq):
        """Set the value of the dual variables for the inequality constraints
        to be used in calls to the evaluation methods (hessian_lag)

        Parameters
        ----------
        duals_ineq: vector_like
            Vector with the values of dual variables for the inequality constraints
        """
        pass

    @abc.abstractmethod
    def get_duals_ineq(self):
        """Get a copy of the values of the dual variables of the inequality
        constraints as provided in set_duals_eq. These are the values
        that will be used in calls to the evaluation methods.
        """
        pass

    @abc.abstractmethod
    def get_eq_constraints_scaling(self):
        """ Return the desired scaling factors to use for the
        for the equality constraints. None indicates no scaling.
        This indicates potential scaling for the model, but the
        evaluation methods should return *unscaled* values

        Returns
        -------
        array-like or None
        """
        pass

    @abc.abstractmethod
    def get_ineq_constraints_scaling(self):
        """ Return the desired scaling factors to use for the
        for the inequality constraints. None indicates no scaling.
        This indicates potential scaling for the model, but the
        evaluation methods should return *unscaled* values

        Returns
        -------
        array-like or None
        """
        pass

    @abc.abstractmethod
    def evaluate_eq_constraints(self, out=None):
        """Returns the values for the equality constraints evaluated at
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
    def evaluate_ineq_constraints(self, out=None):
        """Returns the values of the inequality constraints evaluated at
        the values given for the primal variables in set_primals

        Parameters
        ----------
        out : array_like, optional
            Output array. Its type is preserved and it
            must be of the right shape to hold the output.

        Returns
        -------
        vector_like
        """
        pass

    @abc.abstractmethod
    def evaluate_jacobian_eq(self, out=None):
        """Returns the Jacobian of the equality constraints evaluated
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
    def evaluate_jacobian_ineq(self, out=None):
        """Returns the Jacobian of the inequality constraints evaluated
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

