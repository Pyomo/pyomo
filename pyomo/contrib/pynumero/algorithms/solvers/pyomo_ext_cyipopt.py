import numpy as np
import six
import abc
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptProblemInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from pyomo.environ import Var, Constraint
from pyomo.core.base.var import _VarData

"""
This module is used for interfacing a multi-input / multi-output external
evaluation code with a Pyomo model and then solve the coupled model
with CyIpopt.

To use this interface:
   * inherit from ExternalInputOutputModel and implement the necessary methods
     (This provides methods to set the input values, evaluate the output values,
     and evaluate the jacobian of the outputs with respect to the inputs.)
   * create a PyomoExternalCyIpoptProblem object, giving it your pyomo model, an
     instance of the derived ExternalInputOutputModel, a list of the Pyomo variables
     that map to the inputs of the external model, and a list of the Pyomo variables
     that map to the outputs from the external model.
   * The standard CyIpopt solver interface can be called using the PyomoExternalCyIpoptProblem

See the PyNumero examples to see the module in use.
You can also look at the tests for this interface to see an example of use.

Todo:
   * Currently, you cannot "fix" a pyomo variable that corresponds to an input or output
     and you must use a constraint instead (this is because Pyomo removes fixed variables
     before sending them to the solver)
   * Remove the dummy variable and constraint once Pyomo supports non-removal of certain
     variables
"""
@six.add_metaclass(abc.ABCMeta)
class ExternalInputOutputModel(object):
    """
    This is the base class for building external input output models
    for use with Pyomo and CyIpopt
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def set_inputs(self, input_values):
        """
        This method is called by the solver to set the current values
        for the input variables. The derived class must cache these if
        necessary for any subsequent calls to evalute_outputs or
        evaluate_derivatives.
        """
        pass

    @abc.abstractmethod
    def evaluate_outputs(self):
        """
        Compute the outputs from the model (using the values
        set in input_values) and return as a numpy array
        """
        pass

    @abc.abstractmethod
    def evaluate_derivatives(self):
        """
        Compute the derivatives of the outputs with respect
        to the inputs (using the values set in input_values).
        This should be a dense matrix with the rows in
        the order of the output variables and the cols in
        the order of the input variables.
        """
        pass

    # ToDo: Hessians not yet handled

class PyomoExternalCyIpoptProblem(CyIpoptProblemInterface):
    def __init__(self, pyomo_model, ex_input_output_model, inputs, outputs):
        """
        Create an instance of this class to pass as a problem to CyIpopt.

        Parameters
        ----------
        pyomo_model : ConcreteModel
           The ConcreteModel representing the Pyomo part of the problem. This
           model must contain Pyomo variables for the inputs and the outputs.

        ex_input_output_model : ExternalInputOutputModel
           An instance of a derived class (from ExternalInputOutputModel) that provides
           the methods to compute the outputs and the derivatives.

        inputs : list of Pyomo variables (_VarData)
           The Pyomo model needs to have variables to represent the inputs to the 
           external model. This is the list of those input variables in the order 
           that corresponds to the input_values vector provided in the set_inputs call.

        outputs : list of Pyomo variables (_VarData)
          The Pyomo model needs to have variables to represent the outputs from the
          external model. This is the list of those output variables in the order
          that corresponds to the numpy array returned from the evaluate_outputs call.
        """
        self._pyomo_model = pyomo_model
        self._ex_io_model = ex_input_output_model

        # verify that the inputs and outputs were passed correctly
        self._inputs = [v for v in inputs]
        for v in self._inputs:
            if not isinstance(v, _VarData):
                raise RuntimeError('Argument inputs passed to PyomoExternalCyIpoptProblem must be'
                                   ' a list of VarData objects. Note: if you have an indexed variable, pass'
                                   ' each index as a separate entry in the list (e.g., inputs=[m.x[1], m.x[2]]).')

        self._outputs = [v for v in outputs]
        for v in self._outputs:
            if not isinstance(v, _VarData):
                raise RuntimeError('Argument outputs passed to PyomoExternalCyIpoptProblem must be'
                                   ' a list of VarData objects. Note: if you have an indexed variable, pass'
                                   ' each index as a separate entry in the list (e.g., inputs=[m.x[1], m.x[2]]).')

        # we need to add a dummy variable and constraint to the pyomo_nlp
        # to make sure it does not remove variables that do not
        # appear in the pyomo part of the model
        # ToDo: Find a better way to do this
        if hasattr(self._pyomo_model, '_dummy_constraint_CyIpoptPyomoExNLP'):
            del self._pyomo_model._dummy_constraint_CyIPoptPyomoExNLP
        if hasattr(self._pyomo_model, '_dummy_variable_CyIpoptPyomoExNLP'):
            del self._pyomo_model._dummy_variable_CyIPoptPyomoExNLP
        
        self._pyomo_model._dummy_variable_CyIpoptPyomoExNLP = Var()
        self._pyomo_model._dummy_constraint_CyIPoptPyomoExNLP = Constraint(
            expr = 0 == sum(v for v in self._pyomo_model.component_data_objects(ctype=Var,
                                                                                 descend_into=True,
                                                                                 active=True)
                           )
           )

        # make an nlp interface from the pyomo model
        self._pyomo_nlp = PyomoNLP(self._pyomo_model)
        
        # create initial value vectors for primals and duals
        init_primals = self._pyomo_nlp.init_primals()
        init_duals_pyomo = self._pyomo_nlp.init_duals()
        if np.any(np.isnan(init_duals_pyomo)):
            # set initial values to 1 if we did not get
            # any from Pyomo
            init_duals_pyomo.fill(1.0)
        init_duals_ex = np.ones(len(self._outputs), dtype=np.float64)
        init_duals = BlockVector(2)
        init_duals.set_block(0, init_duals_pyomo)
        init_duals.set_block(1, init_duals_ex)

        # build the map from inputs and outputs to the full x vector
        self._input_columns = self._pyomo_nlp.get_primal_indices(self._inputs)
        self._input_x_mask = np.zeros(self._pyomo_nlp.n_primals(), dtype=np.float64)
        self._input_x_mask[self._input_columns] = 1.0
        self._output_columns = self._pyomo_nlp.get_primal_indices(self._outputs)
        self._output_x_mask = np.zeros(self._pyomo_nlp.n_primals(), dtype=np.float64)
        self._output_x_mask[self._output_columns] = 1.0
        
        # create caches for primals and duals
        self._cached_primals = init_primals.copy()
        self._cached_duals = init_duals.clone(copy=True)
        self._cached_obj_factor = 1.0

        # set the initial values for the pyomo primals and duals
        self._pyomo_nlp.set_primals(self._cached_primals)
        self._pyomo_nlp.set_duals(self._cached_duals.get_block(0))
        # set the initial values for the external inputs
        ex_inputs = self._ex_io_inputs_from_full_primals(self._cached_primals)
        self._ex_io_model.set_inputs(ex_inputs)

        # create the lower and upper bounds for the complete problem
        pyomo_nlp_con_lb = self._pyomo_nlp.constraints_lb()
        ex_con_lb = np.zeros(len(self._outputs), dtype=np.float64)
        self._gL = np.concatenate((pyomo_nlp_con_lb, ex_con_lb))
        pyomo_nlp_con_ub = self._pyomo_nlp.constraints_ub()
        ex_con_ub = np.zeros(len(self._outputs), dtype=np.float64)
        self._gU = np.concatenate((pyomo_nlp_con_ub, ex_con_ub))

        ### setup the jacobian structures
        self._jac_pyomo = self._pyomo_nlp.evaluate_jacobian()

        # We will be mapping the dense external jacobian (doutputs/dinputs)
        # to the correct columns from the full x vector
        ex_start_row = self._pyomo_nlp.n_constraints()
        jac_ex = self._ex_io_model.evaluate_derivatives()
        jac_ex_irows = list()
        jac_ex_jcols = list()
        jac_ex_data = list()
        for i in range(len(self._outputs)):
            for j in range(len(self._inputs)):
                jac_ex_irows.append(ex_start_row + i)
                jac_ex_jcols.append(self._input_columns[j])
                jac_ex_data.append(jac_ex[i,j])
        # add the jac for output variables from the extra equations
        for i in range(len(self._outputs)):
           jac_ex_irows.append(ex_start_row + i)
           jac_ex_jcols.append(self._output_columns[i])
           jac_ex_data.append(-1.0)

        self._full_jac_irows = np.concatenate((self._jac_pyomo.row, jac_ex_irows))
        self._full_jac_jcols = np.concatenate((self._jac_pyomo.col, jac_ex_jcols))
        self._full_jac_data = np.concatenate((self._jac_pyomo.data, jac_ex_data))

        # currently, this interface does not do anything with Hessians

    def load_x_into_pyomo(self, primals):
        """
        Use this method to load a numpy array of values into the corresponding
        Pyomo variables (e.g., the solution from CyIpopt)

        Parameters
        ----------
        primals : numpy array
           The array of values that will be given to the Pyomo variables. The
           order of this array is the same as the order in the PyomoNLP created
           internally.
        """
        pyomo_variables = self._pyomo_nlp.get_pyomo_variables()
        for i,v in enumerate(primals):
            pyomo_variables[i].set_value(v)

    def _set_primals_if_necessary(self, primals):
        if not np.array_equal(primals, self._cached_primals):
            self._pyomo_nlp.set_primals(primals)
            ex_inputs = self._ex_io_inputs_from_full_primals(primals)
            self._ex_io_model.set_inputs(ex_inputs)
            self._cached_primals = primals.copy()

    def _set_duals_if_necessary(self, duals):
        if not np.array_equal(duals, self._cached_duals):
            self._cached_duals.copy_from(duals)
            self._pyomo_nlp.set_duals(self._cached_duals.get_block(0))

    def _set_obj_factor_if_necessary(self, obj_factor):
        if obj_factor != self._cached_obj_factor:
            self._pyomo_nlp.set_obj_factor(obj_factor)
            self._cached_obj_factor = obj_factor

    def x_init(self):
        return self._pyomo_nlp.init_primals()

    def x_lb(self):
        return self._pyomo_nlp.primals_lb()
    
    def x_ub(self):
        return self._pyomo_nlp.primals_ub()

    def g_lb(self):
        return self._gL.copy()

    def g_ub(self):
        return self._gU.copy()
    
    def objective(self, primals):
        self._set_primals_if_necessary(primals)
        return self._pyomo_nlp.evaluate_objective()

    def gradient(self, primals):
        self._set_primals_if_necessary(primals)
        return self._pyomo_nlp.evaluate_grad_objective()

    def constraints(self, primals):
        self._set_primals_if_necessary(primals)
        pyomo_constraints = self._pyomo_nlp.evaluate_constraints()
        ex_io_outputs = self._ex_io_model.evaluate_outputs()
        ex_io_constraints = ex_io_outputs - self._ex_io_outputs_from_full_primals(primals)
        constraints = BlockVector(2)
        constraints.set_block(0, pyomo_constraints)
        constraints.set_block(1, ex_io_constraints)
        return constraints.flatten()

    def jacobianstructure(self):
        return self._full_jac_irows, self._full_jac_jcols
        
    def jacobian(self, primals):
        self._set_primals_if_necessary(primals)
        self._pyomo_nlp.evaluate_jacobian(out=self._jac_pyomo)
        pyomo_data = self._jac_pyomo.data
        ex_io_deriv = self._ex_io_model.evaluate_derivatives().flatten('C')
        self._full_jac_data[0:len(pyomo_data)] = pyomo_data
        self._full_jac_data[len(pyomo_data):len(pyomo_data)+len(ex_io_deriv)] = ex_io_deriv

        # the -1s for the output variables should still be  here
        return self._full_jac_data

    def hessianstructure(self):
        return np.zeros(0), np.zeros(0)
        #raise NotImplementedError('No Hessians for now')

    def hessian(self, x, y, obj_factor):
        raise NotImplementedError('No Hessians for now')

    def _ex_io_inputs_from_full_primals(self, primals):
        return np.compress(self._input_x_mask, primals)

    def _ex_io_outputs_from_full_primals(self, primals):
        return  np.compress(self._output_x_mask, primals)

    
        
            
