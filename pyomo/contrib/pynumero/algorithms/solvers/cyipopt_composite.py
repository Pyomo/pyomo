#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
import numpy as np
import six
import abc
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptProblemInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from pyomo.environ import Var, Constraint, value
from pyomo.core.base.var import _VarData
from pyomo.common.modeling import unique_component_name
from pyomo.core.base.block import _BlockData, declare_custom_block
import pyomo.environ as pyo

"""
This module is used for interfacing an external model (e.g., compiled code)
with a Pyomo model and then solve the composite model with CyIpopt.

To use this interface:
   * inherit from ExternalGreyBoxModel and implement the necessary methods
   * create a CyIpoptCompositeExtProblemInterface object, giving it your pyomo model, an
     instance of the derived ExternalGreyBoxModel, a list of the Pyomo variables
     that map to the inputs of the external model, and a list of the Pyomo variables
     that map to any outputs from the external model.
   * The standard CyIpopt solver interface can be called using the 
     CyIpoptCompositeExtProblemInterface.

See the PyNumero tests for this interface to see an example of use.

Todo:
   * Currently, you cannot "fix" a pyomo variable that corresponds to an input or output
     and you must use a constraint instead (this is because Pyomo removes fixed variables
     before sending them to the solver)
   * Remove the dummy variable and constraint if possible
"""

@six.add_metaclass(abc.ABCMeta)
class ExternalGreyBoxModel(object):
    """
    This is the base class for building external input output models
    for use with Pyomo and CyIpopt
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def input_names(self):
        """
        Provide the list of string names to corresponding to the inputs
        of this external model. These should be returned in the order
        that they are to be used in set_input_values.
        """
        pass

    @abc.abstractmethod
    def equality_constraint_names(self):
        """
        Provide the list of string names corresponding to any residuals 
        for this external model. These should be in the order corresponding
        to values returned from evaluate_residuals.
        """
        pass

    @abc.abstractmethod
    def output_names(self):
        """
        Provide the list of string names corresponding to the outputs
        of this external model. These should be in the order corresponding
        to values returned from evaluate_outputs.

        If there are no outputs, return None
        """
        pass

    @abc.abstractmethod
    def set_input_values(self, input_values):
        """
        This method is called by the solver to set the current values
        for the input variables. The derived class must cache these if
        necessary for any subsequent calls to evalute_outputs or
        evaluate_derivatives.
        """
        pass

    def get_equality_constraint_scaling_factors(self):
        """
        This method is called by the solver interface to get desired
        values for scaling the equality constraints. None means no
        scaling is desired. Note that, depending on the solver,
        one may need to set solver options so these factors are used
        """
        return None
    
    def get_output_constraint_scaling_factors(self):
        """
        This method is called by the solver interface to get desired
        values for scaling the constraints with output variables. Returning
        None means that no scaling of the output constraints is desired.
        Note that, depending on the solver, one may need to set solver options
        so these factors are used
        """
        return None

    @abc.abstractmethod
    def evaluate_equality_constraints(self):
        """
        Compute the residuals from the model (using the values
        set in input_values) and return as a numpy array
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
    def evaluate_jacobian_equality_constraints(self):
        """
        Compute the derivatives of the residuals with respect
        to the inputs (using the values set in input_values).
        This should be a scipy matrix with the rows in
        the order of the residual names and the cols in
        the order of the input variables.
        """
        pass

    @abc.abstractmethod
    def evaluate_jacobian_outputs(self):
        """
        Compute the derivatives of the outputs with respect
        to the inputs (using the values set in input_values).
        This should be a scipy matrix with the rows in
        the order of the output variables and the cols in
        the order of the input variables.
        """
        pass

    # ToDo: Hessians not yet handled


@declare_custom_block(name='ExternalGreyBoxBlock', new_ctype=True)
class ExternalGreyBoxBlockData(_BlockData):

    # TODO CHANGE THE CTYPE
    
    def set_external_model(self, external_grey_box_model):
        self._ex_model = ex_model = external_grey_box_model

        self._input_names = ex_model.input_names()
        if self._input_names is None:
            raise ValueError('input_names() returned None on external_grey_box_model.'
                             ' Must specify at least one input.')
        self.inputs = pyo.Var(self._input_names)

        self._equality_constraint_names = ex_model.equality_constraint_names()
        
        self._output_names = ex_model.output_names()
        # note, output_names may be None, indicating no output variables
        # in the model
        if self._output_names is not None:
            self.outputs = pyo.Var(self._output_names)

    def get_external_model(self):
        return self._ex_model

    def has_equality_constraints(self):
        return self._equality_constraint_names is not None

    def has_outputs(self):
        return self._output_names is not None

    def get_nlp_interface_helper(self, pyomo_nlp):
        return _ExternalGreyBoxHelper(self, pyomo_nlp)


class _ExternalGreyBoxModelHelper(object):
    def __init__(self, ex_grey_box_block, pyomo_nlp):
        """
        This helper takes an ExternalGreyBoxModel and provides the residual
        and Jacobian computation.

        The ExternalGreyBoxModel provides an interface that supports equality
        constraints (pure residuals) and output equations. Let u be the inputs,
        o be the outputs, and x be the full set of primal variables
        from the entire pyomo_nlp.

        With this, the ExternalGreyBoxModel provides the residual computations
        w_eq(u), and w_o(u), as well as the Jacobians, Jw_eq(u), and Jw_o(u). This
        helper provides h(x)=0, where h(x) = [h_eq(x); h_o(x)-o] and 
        h_eq(x)=w_eq(Pu*x), and h_o(x)=w_o(Pu*x), and Pu is a mapping from the full primal 
        variables "x" to the inputs "u".

        It also provides the Jacobian of h w.r.t. x.
           J_h(x) = [Jw_eq(Pu*x); Jw_o(Pu*x)-Po*x]
        where Po is a mapping from the full primal variables "x" to the outputs "o".
        """
        self._block = ex_grey_box_block
        self._ex_model = ex_grey_box_block.get_external_model()
        n_inputs = len(self._block.inputs)
        n_outputs = len(self._block.outputs)

        # store the map of input indices (0 .. n_inputs) to
        # the indices in the full primals vector
        self._inputs_to_primals_map = \
            pyomo_nlp.get_primals_indices(self._block.inputs)

        # store the map of output indices (0 .. n_outputs) to
        # the indices in the full primals vector
        self._outputs_to_primals_map = \
            pyomo_nlp.get_primals_indices(self._block.outputs)
        
        # setup some structures for the jacobians
        primal_values = pyomo_nlp.get_primals()
        input_values = primal_values[self._inputs_to_primals_map]
        self._ex_model.set_input_values(input_values)

        if not self._ex_model.has_outputs() and \
           not self._ex_model.has_equality_constraints():
            raise ValueError('ExternalGreyBoxModel has no equality constraints or outputs. It must have one or both.')

        # we need to change the column indices in the jacobian
        # from the 0..n_inputs provided by the external model
        # to the indices corresponding to the full Pyomo model
        # so we create that here
        self._eq_jac_jcol_for_primals = None
        if self._ex_model.has_equality_constraints():
            jac = self._ex_model.evaluate_jacobian_equality_constraints()
            self._eq_jac_jcol_for_primals = [self._inputs_to_primals_map[j] for j in jac.col]

        self._outputs_jac_jcol_for_primals = None
        if self._ex_model.has_outputs():
            jac = self._ex_model.evaluate_jacobian_outputs()
            self._outputs_jac_jcol_for_primals = [self._inputs_to_primals_map[j] for j in jac.col]

        # create the irow, jcol, nnz structure for the
        # output variable portion of h(u)-o=0
        self._additional_output_entries_irow = \
            np.asarray([i for i in range(n_outputs)])
        self._additional_output_entries_jcol = \
            np.asarray([self._outputs_to_primals_map[j] for j in range(n_outputs)])
        self._additional_output_entries_data = \
            -1.0*np.ones(n_outputs)
        
    def set_primals(self, primals):
        # map the full primals "x" to the inputs "u" and set
        # the values on the external model 
        input_values = primals[self._inputs_to_primals_map]
        self._ex_model.set_input_values(input_values)

        # map the full primals "x" to the outputs "o" and
        # store a vector of the current output values to
        # use when evaluating residuals
        self._output_values = primals[self._outputs_to_primals_map]

    def evaluate_residuals(self):
        # evalute the equality constraints and the output equations
        # and return a single vector of residuals
        # returns residual for h(x)=0, where h(x) = [h_eq(x); h_o(x)-o]
        eq_con_resid = self._ex_model.evaluate_equality_constraints()
        computed_output_values = self._ex_model.evaluate_outputs()
        output_resid = computed_output_values - self._output_values
        return np.concatenate(eq_con_resid, output_resid)
        
    def evaluate_jacobian(self):
        # compute the jacobian of h(x) w.r.t. x
        # J_h(x) = [Jw_eq(Pu*x); Jw_o(Pu*x)-Po*x]

        # Jw_eq(x)
        eq_jac = None
        if self._ex_model.has_equality_constraints():
            eq_jac = self._ex_model.evaluate_jacobian_equality_constraints()
            # map the columns from the inputs "u" back to the full primals "x"
            eq_jac.col = self._eq_jac_jcol_for_primals

        outputs_jac = None
        if self._ex_model.has_outputs():
            outputs_jac = evaluate_jacobian_outputs()
            row = outputs_jac.row
            # map the columns from the inputs "u" back to the full primals "x"
            col = self._outputs_jac_jcol_for_primals
            data = outputs_jac.data
            # add the additional entries for the -Po*x portion of the jacobian
            row = np.concatenate(row, self._additional_output_entries_irow)
            col = np.concatenate(col, self._additional_output_entries_jcol)
            data  = np.concatenate(data, self._additional_output_entries_data)

        jac = None
        if eq_jac is not None:
            if outputs_jac is not None:
                # create a jacobian with both Jw_eq and Jw_o
                jac = BlockMatrix(2,1)
                jac.name = 'external model jacobian'
                jac.set_block(0,0,eq_jac)
                jac.set_block(1,0,outputs_jac)
            else:
                assert not self._ex_model.has_outputs()
                assert self._ex_model.has_equality_constraints()
                # only need the Jacobian with Jw_eq (there are not
                # output equations)
                jac = eq_jac
        else:
            assert outputs_jac is not None
            assert self._ex_model.has_outputs()
            assert not self._ex_model.has_equality_constraints()
            # only need the Jacobian with Jw_o (there are no equalities)
            jac = outputs_jac

        return jac

'''
class PyomoExternalCyIpoptCompositeProblem(CyIpoptProblemInterface):
    def __init__(self, pyomo_model):
        """
        Use this class to pass a Pyomo Model to CyIpopt when
        that model contains an ExternalGreyBoxBlock

        Parameters
        ----------
        pyomo_model : ConcreteModel
           The ConcreteModel that contains the Pyomo variables and the ExternalGreyBoxBlock
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
        # appear in the pyomo part of the model - also ensure unique name in case model
        # is used in more than one instance of this class
        # ToDo: Improve this by convincing Pyomo not to remove the inputs and outputs
        dummy_var_name = unique_component_name(self._pyomo_model, '_dummy_variable_CyIpoptPyomoExNLP')
        dummy_var = Var()
        setattr(self._pyomo_model, dummy_var_name, dummy_var)
        dummy_con_name = unique_component_name(self._pyomo_model, '_dummy_constraint_CyIpoptPyomoExNLP')
        dummy_con = Constraint(
            expr = getattr(self._pyomo_model, dummy_var_name) == \
               sum(v for v in self._inputs) + sum(v for v in self._outputs)
            )
        setattr(self._pyomo_model, dummy_con_name, dummy_con)

        # initialize the dummy var to the right hand side
        dummy_var_value = 0
        for v in self._inputs:
            if v.value is not None:
                dummy_var_value += value(v)
        for v in self._outputs:
            if v.value is not None:
                dummy_var_value += value(v)
        dummy_var.value = dummy_var_value

        # make an nlp interface from the pyomo model
        self._pyomo_nlp = PyomoNLP(self._pyomo_model)
        
        # create initial value vectors for primals and duals
        init_primals = self._pyomo_nlp.init_primals()
        init_duals_pyomo = self._pyomo_nlp.init_duals()
        if np.any(np.isnan(init_duals_pyomo)):
            # set initial values to 1 for any entries that we don't get
            # (typically, all are set, or none are set)
            init_duals_pyomo[np.isnan(init_duals_pyomo)] = 1.0
        init_duals_ex = np.ones(len(self._outputs), dtype=np.float64)
        init_duals = BlockVector(2)
        init_duals.set_block(0, init_duals_pyomo)
        init_duals.set_block(1, init_duals_ex)

        # build the map from inputs and outputs to the full x vector
        self._input_columns = self._pyomo_nlp.get_primal_indices(self._inputs)
        #self._input_x_mask = np.zeros(self._pyomo_nlp.n_primals(), dtype=np.float64)
        #self._input_x_mask[self._input_columns] = 1.0
        self._output_columns = self._pyomo_nlp.get_primal_indices(self._outputs)
        #self._output_x_mask = np.zeros(self._pyomo_nlp.n_primals(), dtype=np.float64)
        #self._output_x_mask[self._output_columns] = 1.0
        
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

        # create the scaling parameters if they are provided
        self._obj_scaling = self._pyomo_nlp.get_obj_scaling()
        self._primals_scaling = self._pyomo_nlp.get_primals_scaling()
        pyomo_constraints_scaling = self._pyomo_nlp.get_constraints_scaling()
        self._constraints_scaling = None
        # check if we need constraint scaling, and if so, add in the
        # outputs_eqn_scaling
        if pyomo_constraints_scaling is not None or outputs_eqn_scaling is not None:
            if pyomo_constraints_scaling is None:
                pyomo_constraints_scaling = np.ones(self._pyomo_nlp.n_primals(), dtype=np.float64)
            if outputs_eqn_scaling is None:
                outputs_eqn_scaling = np.ones(len(self._outputs), dtype=np.float64)
            if type(outputs_eqn_scaling) is list:
                outputs_eqn_scaling = np.asarray(outputs_eqn_scaling, dtype=np.float64)
            self._constraints_scaling = np.concatenate((pyomo_constraints_scaling,
                                                       outputs_eqn_scaling))

        ### setup the jacobian structures
        self._jac_pyomo = self._pyomo_nlp.evaluate_jacobian()

        # We will be mapping the dense external jacobian (doutputs/dinputs)
        # to the correct columns from the full x vector
        ex_start_row = self._pyomo_nlp.n_constraints()

        jac_ex = self._ex_io_model.evaluate_derivatives()

        # the jacobian returned from the external model is in the
        # space of the external model only. We need to shift
        # the rows down and shift the columns appropriately
        jac_ex_irows = np.copy(jac_ex.row)
        jac_ex_irows += ex_start_row
        jac_ex_jcols = np.copy(jac_ex.col)
        for z,col in enumerate(jac_ex_jcols):
            jac_ex_jcols[z] = self._input_columns[col]
        jac_ex_data = np.copy(jac_ex.data)

        # CDL: this code was for the dense version of evaluate_derivatives
        # for i in range(len(self._outputs)):
        #     for j in range(len(self._inputs)):
        #         jac_ex_irows.append(ex_start_row + i)
        #         jac_ex_jcols.append(self._input_columns[j])
        #         jac_ex_data.append(jac_ex[i,j])

        jac_ex_output_irows = list()
        jac_ex_output_jcols = list()
        jac_ex_output_data = list()

        # add the jac for output variables from the extra equations
        for i in range(len(self._outputs)):
           jac_ex_output_irows.append(ex_start_row + i)
           jac_ex_output_jcols.append(self._output_columns[i])
           jac_ex_output_data.append(-1.0)

        self._full_jac_irows = np.concatenate((self._jac_pyomo.row, jac_ex_irows, jac_ex_output_irows))
        self._full_jac_jcols = np.concatenate((self._jac_pyomo.col, jac_ex_jcols, jac_ex_output_jcols))
        self._full_jac_data = np.concatenate((self._jac_pyomo.data, jac_ex_data, jac_ex_output_data))

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

    def scaling_factors(self):
        return self._obj_scaling, self._primals_scaling, self._constraints_scaling

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
        ex_io_deriv = self._ex_io_model.evaluate_derivatives()
        # CDL: dense version: ex_io_deriv = self._ex_io_model.evaluate_derivatives().flatten('C')
        self._full_jac_data[0:len(pyomo_data)] = pyomo_data
        self._full_jac_data[len(pyomo_data):len(pyomo_data)+len(ex_io_deriv.data)] = ex_io_deriv.data
        # CDL: dense version: self._full_jac_data[len(pyomo_data):len(pyomo_data)+len(ex_io_deriv)] = ex_io_deriv

        # the -1s for the output variables should still be  here
        return self._full_jac_data

    def hessianstructure(self):
        return np.zeros(0), np.zeros(0)
        #raise NotImplementedError('No Hessians for now')

    def hessian(self, x, y, obj_factor):
        raise NotImplementedError('No Hessians for now')

    def _ex_io_inputs_from_full_primals(self, primals):
        return primals[self._input_columns]
        #return np.compress(self._input_x_mask, primals)

    def _ex_io_outputs_from_full_primals(self, primals):
        return primals[self._output_columns]
        #return  np.compress(self._output_x_mask, primals)
'''
