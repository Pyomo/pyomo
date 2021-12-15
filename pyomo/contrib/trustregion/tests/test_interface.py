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
from pyomo.common.dependencies import numpy_available
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.environ import (
    Var, VarList, ConcreteModel, Reals, ExternalFunction,
    Objective, Constraint, sqrt, sin, SolverFactory, Block
    )
from pyomo.core.expr.visitor import identify_variables
from pyomo.contrib.trustregion.interface import TRFInterface
from pyomo.contrib.trustregion.TRF import _trf_config

logger = logging.getLogger('pyomo.contrib.trustregion')


@unittest.skipIf(not numpy_available,
                 "Cannot test the trustregion solver without numpy")
class TestTrustRegionInterface(unittest.TestCase):

    def setUp(self):
        self.m = ConcreteModel()
        self.m.z = Var(range(3), domain=Reals, initialize=2.)
        self.m.x = Var(range(2), initialize=2.)
        self.m.x[1] = 1.0

        def blackbox(a,b):
            return sin(a-b)

        self.m.bb = ExternalFunction(blackbox)

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
        self.interface = TRFInterface(self.m, self.ext_fcn_surrogate_map_rule,
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
            self.assertIn(var, self.interface.data.all_variables)
        # Check the output vars against all_variables
        for i in self.interface.data.ef_outputs:
            self.assertIn(self.interface.data.ef_outputs[i], self.interface.data.all_variables)
        # The truth models should be a mapping from the EF to
        # the replacement
        # TODO: (12/14/2021) Finish this test suite
        for i, k in self.interface.data.truth_models.items():
            print(i, k)
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
        self.interface.replaceExternalFunctionsWithVariables()
        self.interface.createConstraints()
        self.interface.data.basis_model_output[:] = 0
        self.interface.data.grad_basis_model_output[...] = 0
        self.interface.data.truth_model_output[:] = 0
        self.interface.data.grad_truth_model_output[...] = 0
        self.interface.data.value_of_ef_inputs[...] = 0
        self.interface.model.x[0] = 2.0
        self.interface.model.z.set_values({0: 5.0, 1: 2.5, 2: -1.0})
        # self.interface.updateSurrogateModel()



if __name__ == '__main__':
    unittest.main()