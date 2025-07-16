#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common import unittest

from pyomo.common.dependencies import numpy as numpy, numpy_available

import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet

import pyomo.contrib.alternative_solutions.aos_utils as au


class TestAOSUtilsUnit(unittest.TestCase):

    def get_multiple_objective_model(self):
        """Create a simple model with three objectives."""
        m = pyo.ConcreteModel()
        m.b1 = pyo.Block()
        m.b2 = pyo.Block()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.b1.o = pyo.Objective(expr=m.x)
        m.b2.o = pyo.Objective([0, 1])
        m.b2.o[0] = pyo.Objective(expr=m.y)
        m.b2.o[1] = pyo.Objective(expr=m.x + m.y)
        return m

    def test_multiple_objectives(self):
        """Check that an error is thrown with multiple objectives."""
        m = self.get_multiple_objective_model()
        assert_text = (
            "Model has 3 active objective functions, exactly one " "is required."
        )
        with self.assertRaisesRegex(AssertionError, assert_text):
            au.get_active_objective(m)

    def test_no_objectives(self):
        """Check that an error is thrown with no objectives."""
        m = self.get_multiple_objective_model()
        m.b1.o.deactivate()
        m.b2.o.deactivate()
        assert_text = (
            "Model has 0 active objective functions, exactly one " "is required."
        )
        with self.assertRaisesRegex(AssertionError, assert_text):
            au.get_active_objective(m)

    def test_one_objective(self):
        """
        Check that the active objective is returned, when there is just one
        objective.
        """
        m = self.get_multiple_objective_model()
        m.b1.o.deactivate()
        m.b2.o[0].deactivate()
        self.assertEqual(m.b2.o[1], au.get_active_objective(m))

    def test_aos_block(self):
        """Ensure that an alternative solution block is added."""
        m = self.get_multiple_objective_model()
        block_name = "test_block"
        b = au._add_aos_block(m, block_name)
        self.assertEqual(b.name, block_name)
        self.assertEqual(b.ctype, pyo.Block)

    def get_simple_model(self, sense=pyo.minimize):
        """Create a simple 2d linear program with an objective."""
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.o = pyo.Objective(expr=m.x + m.y, sense=sense)
        return m

    def test_no_obj_constraint(self):
        """Ensure that no objective constraints are added."""
        m = self.get_simple_model()
        cons = au._add_objective_constraint(m, m.o, 2, None, None)
        self.assertEqual(cons, [])
        self.assertEqual(m.find_component("optimality_tol_rel"), None)
        self.assertEqual(m.find_component("optimality_tol_abs"), None)

    def test_min_rel_obj_constraint(self):
        """Ensure that the correct relative objective constraint is added."""
        m = self.get_simple_model()
        cons = au._add_objective_constraint(m, m.o, 2, 0.1, None)
        self.assertEqual(len(cons), 1)
        self.assertEqual(m.find_component("optimality_tol_rel"), cons[0])
        self.assertEqual(m.find_component("optimality_tol_abs"), None)
        self.assertEqual(2.2, cons[0].upper)
        self.assertEqual(None, cons[0].lower)

    def test_min_abs_obj_constraint(self):
        """Ensure that the correct absolute objective constraint is added."""
        m = self.get_simple_model()
        cons = au._add_objective_constraint(m, m.o, 2, None, 1)
        self.assertEqual(len(cons), 1)
        self.assertEqual(m.find_component("optimality_tol_rel"), None)
        self.assertEqual(m.find_component("optimality_tol_abs"), cons[0])
        self.assertEqual(3, cons[0].upper)
        self.assertEqual(None, cons[0].lower)

    def test_min_both_obj_constraint(self):
        m = self.get_simple_model()
        cons = au._add_objective_constraint(m, m.o, -10, 0.3, 5)
        self.assertEqual(len(cons), 2)
        self.assertEqual(m.find_component("optimality_tol_rel"), cons[0])
        self.assertEqual(m.find_component("optimality_tol_abs"), cons[1])
        self.assertEqual(-7, cons[0].upper)
        self.assertEqual(None, cons[0].lower)
        self.assertEqual(-5, cons[1].upper)
        self.assertEqual(None, cons[1].lower)

    def test_max_both_obj_constraint(self):
        """
        Ensure that the correct relative and absolute objective constraints are
        added.
        """
        m = self.get_simple_model(sense=pyo.maximize)
        cons = au._add_objective_constraint(m, m.o, -1, 0.3, 1)
        self.assertEqual(len(cons), 2)
        self.assertEqual(m.find_component("optimality_tol_rel"), cons[0])
        self.assertEqual(m.find_component("optimality_tol_abs"), cons[1])
        self.assertEqual(None, cons[0].upper)
        self.assertEqual(-1.3, cons[0].lower)
        self.assertEqual(None, cons[1].upper)
        self.assertEqual(-2, cons[1].lower)

    def test_max_both_obj_constraint2(self):
        """
        Ensure that the correct relative and absolute objective constraints are
        added.
        """
        m = self.get_simple_model(sense=pyo.maximize)
        cons = au._add_objective_constraint(m, m.o, 20, 0.5, 11)
        self.assertEqual(len(cons), 2)
        self.assertEqual(m.find_component("optimality_tol_rel"), cons[0])
        self.assertEqual(m.find_component("optimality_tol_abs"), cons[1])
        self.assertEqual(None, cons[0].upper)
        self.assertEqual(10, cons[0].lower)
        self.assertEqual(None, cons[1].upper)
        self.assertEqual(9, cons[1].lower)

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_random_direction(self):
        """
        Ensure that _get_random_direction returns a normal vector.
        """
        from numpy.linalg import norm

        vector = au._get_random_direction(10)
        self.assertAlmostEqual(1.0, norm(vector))

    def get_var_model(self):
        """
        Create a model with multiple variables that are nested over several
        layers of blocks.
        """

        indices = [0, 1, 2, 3]

        m = pyo.ConcreteModel()

        m.b1 = pyo.Block()
        m.b2 = pyo.Block()
        m.b1.sb1 = pyo.Block()
        m.b2.sb2 = pyo.Block()

        m.x = pyo.Var(domain=pyo.Reals)
        m.b1.y = pyo.Var(domain=pyo.Binary)
        m.b2.z = pyo.Var(domain=pyo.Integers)

        m.x_f = pyo.Var(domain=pyo.Reals)
        m.b1.y_f = pyo.Var(domain=pyo.Binary)
        m.b2.z_f = pyo.Var(domain=pyo.Integers)
        m.x_f.fix(0)
        m.b1.y_f.fix(0)
        m.b2.z_f.fix(0)

        m.b1.sb1.x_l = pyo.Var(indices, domain=pyo.Reals)
        m.b1.sb1.y_l = pyo.Var(indices, domain=pyo.Binary)
        m.b2.sb2.z_l = pyo.Var(indices, domain=pyo.Integers)

        m.b1.sb1.x_l[3].fix(0)
        m.b1.sb1.y_l[3].fix(0)
        m.b2.sb2.z_l[3].fix(0)

        vars_minus_x = (
            [m.b1.y, m.b2.z, m.x_f, m.b1.y_f, m.b2.z_f]
            + [m.b1.sb1.x_l[i] for i in indices]
            + [m.b1.sb1.y_l[i] for i in indices]
            + [m.b2.sb2.z_l[i] for i in indices]
        )

        m.con = pyo.Constraint(expr=sum(v for v in vars_minus_x) <= 1)
        m.b1.con = pyo.Constraint(expr=m.b1.y <= 1)
        m.b1.sb1.con = pyo.Constraint(expr=m.b1.sb1.y_l[0] <= 1)
        m.obj = pyo.Objective(expr=m.x)

        m.all_vars = ComponentSet([m.x] + vars_minus_x)
        m.unfixed_vars = ComponentSet([var for var in m.all_vars if not var.is_fixed()])

        return m

    def test_get_all_variables_unfixed(self):
        """Check that all unfixed variables are gathered."""
        m = self.get_var_model()
        var = au.get_model_variables(m)
        self.assertEqual(var, m.unfixed_vars)

    def test_get_all_variables(self):
        """Check that all fixed and unfixed variables are gathered."""
        m = self.get_var_model()
        var = au.get_model_variables(m, include_fixed=True)
        self.assertEqual(var, m.all_vars)

    def test_get_all_continuous(self):
        """Check that all continuous variables are gathered."""
        m = self.get_var_model()
        var = au.get_model_variables(
            m, include_continuous=True, include_binary=False, include_integer=False
        )
        continuous_vars = ComponentSet(
            var for var in m.unfixed_vars if var.is_continuous()
        )
        self.assertEqual(var, continuous_vars)

    def test_get_all_binary(self):
        """Check that all binary variables are gathered."""
        m = self.get_var_model()
        var = au.get_model_variables(
            m, include_continuous=False, include_binary=True, include_integer=False
        )
        binary_vars = ComponentSet(var for var in m.unfixed_vars if var.is_binary())
        self.assertEqual(var, binary_vars)

    def test_get_all_integer(self):
        """Check that all integer variables are gathered."""
        m = self.get_var_model()
        var = au.get_model_variables(
            m, include_continuous=False, include_binary=False, include_integer=True
        )
        continuous_vars = ComponentSet(
            var for var in m.unfixed_vars if var.is_integer()
        )
        self.assertEqual(var, continuous_vars)

    def test_get_specific_vars(self):
        """Check that all variables from a list are gathered."""
        m = self.get_var_model()
        components = [m.x, m.b1.sb1.y_l[0], m.b2.sb2.z_l]
        var = au.get_model_variables(m, components=components)
        specific_vars = ComponentSet(
            [m.x, m.b1.sb1.y_l[0], m.b2.sb2.z_l[0], m.b2.sb2.z_l[1], m.b2.sb2.z_l[2]]
        )
        self.assertEqual(var, specific_vars)

    def test_get_block_vars1(self):
        """
        Check that all variables from block are gathered (without
        descending into subblocks).
        """
        m = self.get_var_model()
        components = [m.b2.sb2.z_l, (m.b1, False)]
        var = au.get_model_variables(m, components=components)
        specific_vars = ComponentSet(
            [m.b1.y, m.b2.sb2.z_l[0], m.b2.sb2.z_l[1], m.b2.sb2.z_l[2]]
        )
        self.assertEqual(var, specific_vars)

    def test_get_block_vars2(self):
        """
        Check that all variables from block are gathered (without
        descending into subblocks).
        """
        m = self.get_var_model()
        components = [m.b1]
        var = au.get_model_variables(m, components=components)
        specific_vars = ComponentSet([m.b1.y, m.b1.sb1.y_l[0]])
        self.assertEqual(var, specific_vars)

    def test_get_constraint_vars(self):
        """Check that all variables constraints and objectives are gathered."""
        m = self.get_var_model()
        components = [m.con, m.obj]
        var = au.get_model_variables(m, components=components)
        self.assertEqual(var, m.unfixed_vars)


if __name__ == "__main__":
    unittest.main()
