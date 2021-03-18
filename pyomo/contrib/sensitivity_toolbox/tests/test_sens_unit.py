# ____________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and
# Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
# rights in this software.
# This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________

"""
Unit Tests for interfacing with sIPOPT and k_aug
"""

import pyutilib.th as unittest
from six import StringIO
import logging

from pyomo.environ import (
        ConcreteModel,
        Objective,
        Param,
        Var,
        Block,
        Suffix,
        value,
        Constraint,
        inequality,
        NonNegativeReals,
        )
from pyomo.core.base.component import ComponentData
from pyomo.common.dependencies import scipy_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.core.expr.current import identify_variables
from pyomo.core.expr.visitor import identify_mutable_parameters
from pyomo.contrib.sensitivity_toolbox.sens import (
        SensitivityInterface,
        _NotAnIndex,
        )

import pyomo.contrib.sensitivity_toolbox.examples.parameter as param_example

def make_indexed_model():
    """
    Creates the model used in the "parameter.py" example, but with indexed
    variables, parameters, and constraints.
    """
    m = ConcreteModel()

    m.x = Var([1, 2, 3], initialize={1: 0.15, 2: 0.15, 3: 0.0},
            domain=NonNegativeReals)

    m.eta = Param([1, 2], initialize={1: 4.5, 2: 1.0}, mutable=True)

    m.const = Constraint([1, 2], rule={
        1: 6*m.x[1] + 3*m.x[2] + 2*m.x[3] - m.eta[1] == 0,
        2: m.eta[2]*m.x[1] + m.x[2] - m.x[3] - 1 == 0,
        })

    m.cost = Objective(expr=m.x[1]**2 + m.x[2]**2 + m.x[3]**2)

    return m

def make_model_with_inequalities():
    """
    Creates a modified version of the model used in the "parameter.py"
    example, now with simple (one-sided) inequalities.
    """
    m = ConcreteModel()

    m.x = Var([1, 2, 3], initialize={1: 0.15, 2: 0.15, 3: 0.0},
            domain=NonNegativeReals)

    m.eta = Param([1, 2], initialize={1: 4.5, 2: 1.0}, mutable=True)

    m.const = Constraint([1, 2], rule={
        1: 6*m.x[1] + 3*m.x[2] + 2*m.x[3] >= m.eta[1],
        2: m.eta[2]*m.x[1] + m.x[2] - m.x[3] - 1 <= 0,
        })

    m.cost = Objective(expr=m.x[1]**2 + m.x[2]**2 + m.x[3]**2)

    return m

def make_model_with_ranged_inequalities():
    """
    Creates a modified version of the model used in the "parameter.py"
    example, now with simple (one-sided) inequalities.
    """
    m = ConcreteModel()

    m.x = Var([1, 2, 3], initialize={1: 0.15, 2: 0.15, 3: 0.0},
            domain=NonNegativeReals)

    m.p = Param(initialize=10.0, mutable=True)

    m.eta = Param([1, 2], initialize={1: 4.5, 2: 1.0}, mutable=True)

    m.const = Constraint([1, 2], rule={
        1: inequality(
            lower=-m.eta[1],
            body=6*m.x[1] + 3*m.x[2] + 2*m.x[3],
            upper=m.p + m.eta[1],
            ),
        2: m.eta[2]*m.x[1] + m.x[2] - m.x[3] - 1 <= 0,
        })

    m.cost = Objective(expr=m.x[1]**2 + m.x[2]**2 + m.x[3]**2)

    return m

class TestSensitivityInterface(unittest.TestCase):

    def assertIsSubset(self, s1, s2):
        for item in s1:
            self.assertIn(item, s2)
    
    def test_get_names(self):
        block_name = SensitivityInterface.get_default_block_name()
        self.assertEqual(block_name, "_SENSITIVITY_TOOLBOX_DATA")

        var_name = 'var'
        sens_var_name = SensitivityInterface.get_default_var_name(var_name)
        self.assertEqual(sens_var_name, var_name)

        param_name = 'param'
        sens_param_name = SensitivityInterface.get_default_param_name(param_name)
        self.assertEqual(sens_param_name, param_name)

    def test_constructor_clone(self):
        model = param_example.create_model()
        sens = SensitivityInterface(model)
        self.assertIs(sens._original_model, model)
        self.assertIsNot(sens.model_instance, model)

    def test_constructor_no_clone(self):
        model = param_example.create_model()
        sens = SensitivityInterface(model, clone_model=False)
        self.assertIs(sens._original_model, model)
        self.assertIs(sens.model_instance, model)

    def test_add_data_block(self):
        model = param_example.create_model()
        sens = SensitivityInterface(model, clone_model=False)

        block = sens._add_data_block()
        self.assertIs(sens.block.parent_block(), sens.model_instance)
        self.assertIs(sens.block.ctype, Block)
        self.assertEqual(sens.block.local_name, sens.get_default_block_name())

        with self.assertRaises(RuntimeError) as ex:
            sens._add_data_block()
        # We just tried adding the same block twice.
        self.assertIn("Cannot add component", str(ex.exception))

        # Try re-adding the same block, but this time we are prepared
        # for it to already exist.
        new_block = sens._add_data_block(existing_block=block)
        self.assertIsNot(block, new_block)

        new_block._has_replaced_expressions = True
        with self.assertRaises(RuntimeError) as ex:
            sens._add_data_block(existing_block=new_block)
        # Cannot remove and re-add sensitivity block if expressions
        # were replaced.
        self.assertIn("Re-using sensitivity interface", str(ex.exception))

    def test_process_param_list(self):
        model = make_indexed_model()
        sens = SensitivityInterface(model, clone_model=False)

        param_list = [model.x[1], model.eta]
        new_param_list = sens._process_param_list(param_list)
        self.assertIs(param_list, new_param_list)

        sens = SensitivityInterface(model, clone_model=True)
        new_param_list = sens._process_param_list(param_list)
        # The new param list contains the "same" variables in the
        # cloned model.
        self.assertIs(new_param_list[0], sens.model_instance.x[1])
        self.assertIs(new_param_list[1], sens.model_instance.eta)

    def test_add_sensitivity_data(self):
        model = make_indexed_model()
        sens = SensitivityInterface(model, clone_model=False)
        sens._add_data_block()

        param_list = [model.x, model.eta]
        with self.assertRaises(ValueError) as exc:
            sens._add_sensitivity_data(param_list)
        self.assertIn("variables must be fixed", str(exc.exception))

        sens.model_instance.x.fix()
        # NOTE: sending the same component twice probably shouldn't
        # be supported. It is convenient for testing, however.
        param_list = [model.x, model.x[1], model.eta, model.eta[1]]
        sens._add_sensitivity_data(param_list)

        block_param_list = list(sens.block.component_data_objects(Param))
        block_var_list = list(sens.block.component_data_objects(Var))
        self.assertEqual(len(block_param_list), 4)
        self.assertEqual(len(block_var_list), 3)
        self.assertEqual(len(sens.block._sens_data_list), 7)

        pred_sens_data_list = [
                (model.x[1], Param, 0, 1),
                (model.x[2], Param, 0, 2),
                (model.x[3], Param, 0, 3),
                (model.x[1], Param, 1, _NotAnIndex),
                (Var, model.eta[1], 2, 1),
                (Var, model.eta[2], 2, 2),
                (Var, model.eta[1], 3, _NotAnIndex),
                ]

        for data, pred in zip(sens.block._sens_data_list, pred_sens_data_list):
            if isinstance(pred[0], ComponentData):
                self.assertIs(data[0], pred[0])
                self.assertIs(data[1].ctype, pred[1])
                name = data[0].parent_component().local_name
                self.assertTrue(
                        data[1].parent_component().local_name.startswith(name))
            else:
                self.assertIs(data[0].ctype, pred[0])
                self.assertIs(data[1], pred[1])
                name = data[1].parent_component().local_name
                self.assertTrue(
                        data[0].parent_component().local_name.startswith(name))
            self.assertEqual(data[2], pred[2])
            self.assertEqual(data[3], pred[3])

    def test_expression_replacement_no_replacement(self):
        model = make_indexed_model()
        sens = SensitivityInterface(model, clone_model=False)
        sens._add_data_block()
        instance = sens.model_instance
        block = sens.block
        instance.x.fix()
        param_list = [instance.x[1], instance.x[2], instance.x[3]]
        sens._add_sensitivity_data(param_list)

        self.assertEqual(len(block.constList), 0)
        variable_sub_map = {}
        sens._replace_parameters_in_constraints(variable_sub_map)
        self.assertEqual(len(block.constList), 2)

        # Rely on order of constraints here... Fine as long as
        # component_data_objects iteration is deterministic
        pred_const_list = [instance.const[1], instance.const[2]]
        for orig, replaced in zip(pred_const_list, block.constList.values()):
            self.assertEqual(orig.expr.to_string(), replaced.expr.to_string())
            self.assertFalse(orig.active)
            self.assertTrue(replaced.active)

    def test_expression_replacement_equality(self):
        model = make_indexed_model()
        sens = SensitivityInterface(model, clone_model=False)
        sens._add_data_block()
        instance = sens.model_instance
        block = sens.block
        instance.x.fix()
        param_list = [instance.eta[1], instance.eta[2]]
        sens._add_sensitivity_data(param_list)

        orig_components = (list(instance.component_data_objects(Constraint,
            active=True)) + list(instance.component_data_objects(Objective,
                active=True)))
        orig_expr = [con.expr for con in orig_components]

        # These will be modified to account for expected replacements
        expected_variables = ComponentMap((con,
            ComponentSet(identify_variables(con.expr)))
            for con in orig_components)
        expected_parameters = ComponentMap((con,
            ComponentSet(identify_mutable_parameters(con.expr)))
            for con in orig_components)

        # As constructed by the `setup_sensitivity` method:
        variable_sub_map = dict((id(param), var)
                for var, param, list_idx, _ in block._sens_data_list
                if param_list[list_idx].ctype is Param)
        # Sanity check
        self.assertEqual(len(variable_sub_map), 2)

        # Map each param to the var that should replace it
        param_var_map = ComponentMap((param, var)
                for var, param, _, _ in block._sens_data_list)

        # Remove parameters we expect to replace and add vars
        # we expect to replace with.
        for con in orig_components:
            for param in param_var_map:
                if param in expected_parameters[con]:
                    expected_variables[con].add(param_var_map[param])
                    expected_parameters[con].remove(param)

        # We check that the new components (Constraints and Objectives) contain
        # the expected parameters and variables.
        replaced = sens._replace_parameters_in_constraints(variable_sub_map)
        self.assertEqual(len(block.constList), 2)
        for con in block.constList.values():
            self.assertTrue(con.active)
            param_set = ComponentSet(identify_mutable_parameters(con.expr))
            var_set = ComponentSet(identify_variables(con.expr))
            orig_con = replaced[con]
            self.assertIsNot(orig_con, con)
            self.assertEqual(param_set, expected_parameters[orig_con])
            self.assertEqual(var_set, expected_variables[orig_con])

        self.assertIs(block.cost.ctype, Objective)
        obj = block.cost
        param_set = ComponentSet(identify_mutable_parameters(obj.expr))
        var_set = ComponentSet(identify_variables(obj.expr))
        orig_obj = replaced[obj]
        self.assertIsNot(orig_obj, obj)
        self.assertEqual(param_set, expected_parameters[orig_obj])
        self.assertEqual(var_set, expected_variables[orig_obj])

        # Original components were deactivated but otherwise not altered
        for con, expr in zip(orig_components, orig_expr):
            self.assertFalse(con.active)
            #self.assertIs(con.expr, expr)
            # ^Why does this fail?
            self.assertEqual(con.expr.to_string(), expr.to_string())

    def test_expression_replacement_inequality(self):
        model = make_model_with_inequalities()
        sens = SensitivityInterface(model, clone_model=False)
        sens._add_data_block()
        instance = sens.model_instance
        block = sens.block
        instance.x.fix()
        param_list = [instance.eta[1], instance.eta[2]]
        sens._add_sensitivity_data(param_list)

        orig_components = (list(instance.component_data_objects(Constraint,
            active=True)) + list(instance.component_data_objects(Objective,
                active=True)))
        orig_expr = [con.expr for con in orig_components]

        # These will be modified to account for expected replacements
        expected_variables = ComponentMap((con,
            ComponentSet(identify_variables(con.expr)))
            for con in orig_components)
        expected_parameters = ComponentMap((con,
            ComponentSet(identify_mutable_parameters(con.expr)))
            for con in orig_components)

        # As constructed by the `setup_sensitivity` method:
        variable_sub_map = dict((id(param), var)
                for var, param, list_idx, _ in block._sens_data_list
                if param_list[list_idx].ctype is Param)
        # Sanity check
        self.assertEqual(len(variable_sub_map), 2)

        # Map each param to the var that should replace it
        param_var_map = ComponentMap((param, var)
                for var, param, _, _ in block._sens_data_list)

        # Remove parameters we expect to replace and add vars
        # we expect to replace with.
        for con in orig_components:
            for param in param_var_map:
                if param in expected_parameters[con]:
                    expected_variables[con].add(param_var_map[param])
                    expected_parameters[con].remove(param)

        # We check that the new components (Constraints and Objectives) contain
        # the expected parameters and variables.
        replaced = sens._replace_parameters_in_constraints(variable_sub_map)
        self.assertEqual(len(block.constList), 2)
        for con in block.constList.values():
            self.assertTrue(con.active)
            param_set = ComponentSet(identify_mutable_parameters(con.expr))
            var_set = ComponentSet(identify_variables(con.expr))
            orig_con = replaced[con]
            self.assertIsNot(orig_con, con)
            self.assertEqual(param_set, expected_parameters[orig_con])
            self.assertEqual(var_set, expected_variables[orig_con])

        self.assertIs(block.cost.ctype, Objective)
        obj = block.cost
        param_set = ComponentSet(identify_mutable_parameters(obj.expr))
        var_set = ComponentSet(identify_variables(obj.expr))
        orig_obj = replaced[obj]
        self.assertIsNot(orig_obj, obj)
        self.assertEqual(param_set, expected_parameters[orig_obj])
        self.assertEqual(var_set, expected_variables[orig_obj])

        # Original components were deactivated but otherwise not altered
        for con, expr in zip(orig_components, orig_expr):
            self.assertFalse(con.active)
            #self.assertIs(con.expr, expr)
            # ^Why does this fail?
            self.assertEqual(con.expr.to_string(), expr.to_string())

    def test_expression_replacement_ranged_inequality(self):
        model = make_model_with_ranged_inequalities()
        sens = SensitivityInterface(model, clone_model=False)
        sens._add_data_block()
        instance = sens.model_instance
        block = sens.block
        instance.x.fix()
        param_list = [instance.eta[1], instance.eta[2]]
        sens._add_sensitivity_data(param_list)

        orig_components = (list(instance.component_data_objects(Constraint,
            active=True)) + list(instance.component_data_objects(Objective,
                active=True)))
        orig_expr = [con.expr for con in orig_components]

        # These will be modified to account for expected replacements
        expected_variables = ComponentMap((con,
            ComponentSet(identify_variables(con.expr)))
            for con in orig_components)
        expected_parameters = ComponentMap((con,
            ComponentSet(identify_mutable_parameters(con.expr)))
            for con in orig_components)

        # As constructed by the `setup_sensitivity` method:
        variable_sub_map = dict((id(param), var)
                for var, param, list_idx, _ in block._sens_data_list
                if param_list[list_idx].ctype is Param)
        # Sanity check
        self.assertEqual(len(variable_sub_map), 2)

        # Map each param to the var that should replace it
        param_var_map = ComponentMap((param, var)
                for var, param, _, _ in block._sens_data_list)

        # Remove parameters we expect to replace and add vars
        # we expect to replace with.
        for con in orig_components:
            for param in param_var_map:
                if param in expected_parameters[con]:
                    expected_variables[con].add(param_var_map[param])
                    expected_parameters[con].remove(param)

        # We check that the new components (Constraints and Objectives) contain
        # the expected parameters and variables.
        replaced = sens._replace_parameters_in_constraints(variable_sub_map)
        # With ranged inequalities, we end up with more constraints than we
        # started with:
        self.assertEqual(len(block.constList), 3)

        for con in block.constList.values():
            self.assertTrue(con.active)
            param_set = ComponentSet(identify_mutable_parameters(con.expr))
            var_set = ComponentSet(identify_variables(con.expr))
            orig_con = replaced[con]
            self.assertIsNot(orig_con, con)
            # Note that for ranged inequalities, it is not valid to check
            # that the two sets are equal as a mutable parameter could be
            # contained in only one "sub-inequality"
            self.assertIsSubset(param_set, expected_parameters[orig_con])
            self.assertEqual(var_set, expected_variables[orig_con])

        self.assertIs(block.cost.ctype, Objective)
        obj = block.cost
        param_set = ComponentSet(identify_mutable_parameters(obj.expr))
        var_set = ComponentSet(identify_variables(obj.expr))
        orig_obj = replaced[obj]
        self.assertIsNot(orig_obj, obj)
        self.assertEqual(param_set, expected_parameters[orig_obj])
        self.assertEqual(var_set, expected_variables[orig_obj])

        # Original components were deactivated but otherwise not altered
        for con, expr in zip(orig_components, orig_expr):
            self.assertFalse(con.active)
            #self.assertIs(con.expr, expr)
            # ^Why does this fail?
            self.assertEqual(con.expr.to_string(), expr.to_string())


if __name__=="__main__":
    unittest.main()
