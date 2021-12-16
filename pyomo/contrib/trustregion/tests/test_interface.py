#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging

import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.environ import (
    Var, VarList, ConcreteModel, Reals, ExternalFunction, value,
    Objective, Constraint, sqrt, sin, cos, SolverFactory
    )
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.visitor import identify_variables
from pyomo.contrib.trustregion.interface import TRFInterface
from pyomo.contrib.trustregion.TRF import _trf_config

logger = logging.getLogger('pyomo.contrib.trustregion')


class TestTrustRegionInterface(unittest.TestCase):

    def setUp(self):
        self.m = ConcreteModel()
        self.m.z = Var(range(3), domain=Reals, initialize=2.)
        self.m.x = Var(range(2), initialize=2.)
        self.m.x[1] = 1.0

        def blackbox(a, b):
            return sin(a - b)
        def grad_blackbox(args, fixed):
            a, b = args[:2]
            return [ cos(a - b), -cos(a - b) ]

        self.m.bb = ExternalFunction(blackbox, grad_blackbox)

        self.m.obj = Objective(
            expr=(self.m.z[0]-1.0)**2 + (self.m.z[0]-self.m.z[1])**2
            + (self.m.z[2]-1.0)**2 + (self.m.x[0]-1.0)**4
            + (self.m.x[1]-1.0)**6
        )
        self.m.c1 = Constraint(
            expr=(self.m.x[0] * self.m.z[0]**2
                  + self.m.bb(self.m.x[0], self.m.x[1])
                  == 2*sqrt(2.0))
            )
        self.m.c2 = Constraint(
            expr=self.m.z[2]**4 * self.m.z[1]**2 + self.m.z[1] == 8+sqrt(2.0))
        self.config = _trf_config()
        self.ext_fcn_surrogate_map_rule = lambda comp,ef: 0
        self.interface = TRFInterface(self.m,
                                      self.ext_fcn_surrogate_map_rule,
                                      self.config)

    def test_initializeInterface(self):
        self.assertEqual(self.m, self.interface.original_model)
        self.assertEqual(self.config, self.interface.config)
        self.assertEqual(self.interface.basis_expression_rule,
                         self.ext_fcn_surrogate_map_rule)
        self.assertEqual('ipopt', self.interface.solver.name)

    def test_replaceRF(self):
        # These data objects are normally initialized by
        # replaceExternalFunctionsWithVariables
        self.interface.data.all_variables = ComponentSet()
        self.interface.data.truth_models = ComponentMap()
        self.interface.data.ef_outputs = VarList()
        # The objective function has no EF.
        # Therefore, replaceEF should do nothing
        expr = self.interface.model.obj.expr
        new_expr = self.interface.replaceEF(expr)
        self.assertEqual(expr, new_expr)
        # The first contraint has one EF.
        # Therefore, replaceEF should do a substitution
        expr = self.interface.model.c1.expr
        new_expr = self.interface.replaceEF(expr)
        self.assertIsNot(expr, new_expr)
        self.assertEquals(str(new_expr),
                          'x[0]*z[0]**2 + trf_data.ef_outputs[1]  ==  2.8284271247461903')

    def test_remove_ef_from_expr(self):
        # These data objects are normally initialized by
        # replaceExternalFunctionsWithVariables
        self.interface.data.all_variables = ComponentSet()
        self.interface.data.truth_models = ComponentMap()
        self.interface.data.ef_outputs = VarList()
        self.interface.data.basis_expressions = ComponentMap()
        # The objective function has no EF.
        # Therefore, remove_ef_from_expr should do nothing
        component = self.interface.model.obj
        self.interface._remove_ef_from_expr(component)
        self.assertEqual(str(self.interface.model.obj.expr),
                         '(z[0] - 1.0)**2 + (z[0] - z[1])**2 + (z[2] - 1.0)**2 + (x[0] - 1.0)**4 + (x[1] - 1.0)**6')
        # The first contraint has one EF.
        # Therefore, remove_ef_from_expr should do something
        component = self.interface.model.c1
        str_expr = str(component.expr)
        self.interface._remove_ef_from_expr(component)
        self.assertNotEqual(str_expr, str(component.expr))
        self.assertEqual(str(component.expr),
                         'x[0]*z[0]**2 + trf_data.ef_outputs[1]  ==  2.8284271247461903')

    def test_replaceExternalFunctionsWithVariables(self):
        # In running this method, we not only replace EFs
        # with 'holder' vars; we also get useful information
        # about inputs, outputs, basis expressions, etc.
        self.interface.replaceExternalFunctionsWithVariables()
        # Check the directly defined model vars against all_variables
        for var in self.interface.model.component_data_objects(Var):
            self.assertIn(var, ComponentSet(self.interface.data.all_variables))
        # Check the output vars against all_variables
        for i in self.interface.data.ef_outputs:
            self.assertIn(self.interface.data.ef_outputs[i],
                          ComponentSet(self.interface.data.all_variables))
        # The truth models should be a mapping from the EF to
        # the replacement
        for i, k in self.interface.data.truth_models.items():
            self.assertIsInstance(k, ExternalFunctionExpression)
            self.assertIn(str(self.interface.model.x[0]), str(k))
            self.assertIn(str(self.interface.model.x[1]), str(k))
            self.assertIsInstance(i, _GeneralVarData)
            self.assertEqual(i, self.interface.data.ef_outputs[1])
        for i, k in self.interface.data.basis_expressions.items():
            self.assertEqual(k, 0)
            self.assertEqual(i, self.interface.data.ef_outputs[1])
        self.assertEqual(1, list(self.interface.data.ef_inputs.keys())[0])
        self.assertEqual(self.interface.data.ef_inputs[1],
                         [self.interface.model.x[0],
                          self.interface.model.x[1]])
        # TRF only supports one active Objective
        # Make sure that it fails if there are multiple objs
        self.m.obj2 = Objective(
            expr=(self.m.x[0]**2 - (self.m.z[1] - 3)**3))
        interface = TRFInterface(self.m,
                                 self.ext_fcn_surrogate_map_rule,
                                 self.config)
        with self.assertRaises(ValueError):
            interface.replaceExternalFunctionsWithVariables()

    def test_createConstraints(self):
        # replaceExternalFunctionsWithVariables sets up some
        # necessary items in the block
        self.interface.replaceExternalFunctionsWithVariables()
        self.interface.createConstraints()
        # The constraints should have been created and deactivated
        self.assertFalse(self.interface.data.basis_constraint.active)
        self.assertFalse(self.interface.data.sm_constraint_basis.active)
        # The size of each constraint should be 1
        self.assertEqual(len(self.interface.data.basis_constraint), 1)
        self.assertEqual(len(self.interface.data.sm_constraint_basis), 1)
        # Because they are size 1, they should have one key
        self.assertEqual(list(self.interface.data.basis_constraint.keys()), [1])
        cs = ComponentSet(identify_variables(self.interface.data.basis_constraint[1].expr))
        # The basis constraint only has the EF variable
        self.assertEqual(len(cs), 1)
        self.assertIn(self.interface.data.ef_outputs[1], cs)
        cs = ComponentSet(identify_variables(self.interface.data.sm_constraint_basis[1].expr))
        # The surrogate model constraint has the EF var, with inputs
        # of x[0] and x[1], as seen in self.m.c1
        self.assertEqual(len(cs), 3)
        self.assertIn(self.interface.model.x[0], cs)
        self.assertIn(self.interface.model.x[1], cs)
        self.assertIn(self.interface.data.ef_outputs[1], cs)

    def test_updateSurrogateModel(self):
        # Set up necessary data objects and Params
        self.interface.replaceExternalFunctionsWithVariables()
        self.interface.createConstraints()
        # Set starter values on the model
        self.interface.model.x[0] = 2.0
        self.interface.model.z.set_values({0: 5.0, 1: 2.5, 2: -1.0})
        self.interface.data.basis_model_output[:] = 0
        self.interface.data.grad_basis_model_output[...] = 0
        self.interface.data.truth_model_output[:] = 0
        self.interface.data.grad_truth_model_output[...] = 0
        self.interface.data.value_of_ef_inputs[...] = 0
        self.interface.updateSurrogateModel()
        # The basis model here is 0, so all basis values should be 0
        for key, val in self.interface.data.basis_model_output.items():
            self.assertEqual(value(val), 0)
        for key, val in self.interface.data.grad_basis_model_output.items():
            self.assertEqual(value(val), 0)
        # We initialized the parameters here to 0, and we do not update
        # this param as part of updateSurrogateModel
        for key, val in self.interface.data.truth_model_output.items():
            self.assertEqual(value(val), 0)
        # The truth gradients should equal the output of [cos(2-1), -cos(2-1)]
        truth_grads = []
        for key, val in self.interface.data.grad_truth_model_output.items():
            truth_grads.append(value(val))
        self.assertEqual(truth_grads, [cos(1), -cos(1)])
        # The inputs should be the values of x[0] and x[1]
        for key, val in self.interface.data.value_of_ef_inputs.items():
            self.assertEqual(value(self.interface.model.x[key[1]]), value(val))
        # Change the model values to something else and try again
        self.interface.model.x.set_values({0 : 0, 1 : 0})
        self.interface.updateSurrogateModel()
        # The basis values should still all be 0
        for key, val in self.interface.data.basis_model_output.items():
            self.assertEqual(value(val), 0)
        for key, val in self.interface.data.grad_basis_model_output.items():
            self.assertEqual(value(val), 0)
        # We still have not updated this value, so the value should be 0
        for key, val in self.interface.data.truth_model_output.items():
            self.assertEqual(value(val), 0)
        # The truth gradients should equal the output of [cos(0-0), -cos(0-0)]
        truth_grads = []
        for key, val in self.interface.data.grad_truth_model_output.items():
            truth_grads.append(value(val))
        self.assertEqual(truth_grads, [cos(0), -cos(0)])
        # The inputs should be the values of x[0] and x[1]
        for key, val in self.interface.data.value_of_ef_inputs.items():
            self.assertEqual(value(self.interface.model.x[key[1]]), value(val))

    def test_getCurrentEFValues(self):
        # Set up necessary data objects
        self.interface.replaceExternalFunctionsWithVariables()
        self.interface.createConstraints()
        # Initialize variable values
        self.interface.model.x.set_values({0 : 2, 1 : 1})
        result = self.interface.getCurrentEFValues()
        # The initialized value for the output, calculated in line 228,
        self.assertEqual(result, { 1 : ([value(self.interface.model.x[0]),
                                         value(self.interface.model.x[1])],
                                        sin(1))})
        self.interface.model.x[0] = self.interface.model.x[1] = 0
        result = self.interface.getCurrentEFValues()
        self.assertEqual(result, { 1 : ([value(self.interface.model.x[0]),
                                         value(self.interface.model.x[1])],
                                        sin(1))})

    def test_getCurrentModelState(self):
        # Set up necessary data objects
        self.interface.replaceExternalFunctionsWithVariables()
        # Set starter values on the model
        self.interface.model.x[0] = 2.0
        self.interface.model.z.set_values({0: 5.0, 1: 2.5, 2: -1.0})
        result = self.interface.getCurrentModelState()
        self.assertEqual(len(result), len(self.interface.data.all_variables))
        for var in self.interface.data.all_variables:
            self.assertIn(value(var), result)

    def test_calculateFeasibility(self):
        # Set up necessary data objects
        self.interface.replaceExternalFunctionsWithVariables()
        self.interface.createConstraints()
        # Set starter values on the model
        self.interface.model.x[0] = 2.0
        self.interface.model.z.set_values({0: 5.0, 1: 2.5, 2: -1.0})
        self.interface.data.basis_model_output[:] = 0
        self.interface.data.grad_basis_model_output[...] = 0
        self.interface.data.truth_model_output[:] = 0
        self.interface.data.grad_truth_model_output[...] = 0
        self.interface.data.value_of_ef_inputs[...] = 0
        feasibility = self.interface.calculateFeasibility()
        # The initial feasibility should be 0 because we haven't
        # solved anything, so the truth model and ef_outputs are the same
        self.assertEqual(feasibility, 0)
        # We update the surrogate model to get real parameters
        self.interface.updateSurrogateModel()
        feasibility = self.interface.calculateFeasibility()
        # Because we have not solved the model, the output and truth model
        # should currently match
        self.assertEqual(feasibility, 0)
        # TODO: Add a test for after the model is solved (12/15/2021)

    def test_calculateStepSizeInfNorm(self):
        # Set up necessary data objects
        self.interface.replaceExternalFunctionsWithVariables()
        self.interface.createConstraints()
        # Set starter values on the model
        self.interface.model.x[0] = 2.0
        self.interface.model.z.set_values({0: 5.0, 1: 2.5, 2: -1.0})
        self.interface.data.basis_model_output[:] = 0
        self.interface.data.grad_basis_model_output[...] = 0
        self.interface.data.truth_model_output[:] = 0
        self.interface.data.grad_truth_model_output[...] = 0
        self.interface.data.value_of_ef_inputs[...] = 0
        # Get original and new values
        original_values = self.interface.getCurrentEFValues()
        self.interface.updateSurrogateModel()
        new_values = self.interface.getCurrentEFValues()
        stepnorm = self.interface.calculateStepSizeInfNorm(original_values,
                                                           new_values)
        # Currently, we have taken NO step.
        # Therefore, the norm should be 0.
        self.assertEqual(stepnorm, 0)
        # TODO: Add a test for after the model is solved (12/15/2021)

    @unittest.skipIf(not SolverFactory('ipopt').available(False),
                     "The IPOPT solver is not available")
    def test_solveModel(self):
        # Set up initial data objects and Params
        self.interface.replaceExternalFunctionsWithVariables()
        self.interface.createConstraints()
        self.interface.data.basis_constraint.activate()
        # Set starter values on the model
        self.interface.model.x[0] = 2.0
        self.interface.model.z.set_values({0: 5.0, 1: 2.5, 2: -1.0})
        self.interface.data.basis_model_output[:] = 0
        self.interface.data.grad_basis_model_output[...] = 0
        self.interface.data.truth_model_output[:] = 0
        self.interface.data.grad_truth_model_output[...] = 0
        self.interface.data.value_of_ef_inputs[...] = 0
        objective, step_norm, feasibility = self.interface.solveModel()
        self.assertEqual(objective, 5.150744273013601)
        self.assertEqual(step_norm, 0.9041534948101395)
        self.assertEqual(feasibility, 0.09569982275514467)
        self.interface.data.basis_constraint.deactivate()
        self.interface.updateSurrogateModel()
        self.interface.data.sm_constraint_basis.activate()
        objective, step_norm, feasibility = self.interface.solveModel()
        self.assertEqual(objective, 5.150674356202409)
        self.assertEqual(step_norm, 0.13254909994294217)
        self.assertEqual(feasibility, 0.09509735734117808)


if __name__ == '__main__':
    unittest.main()