# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import logging

import pyomo.common.unittest as unittest
from pyomo.common.collections import Bunch
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available
from pyomo.contrib.mindtpy.algorithm_base_class import _MindtPyAlgorithm
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_OA_config
from pyomo.contrib.mindtpy.cut_generation import add_affine_cuts
from pyomo.contrib.mindtpy.tests.MINLP5_simple import SimpleMINLP5
from pyomo.contrib.mindtpy.util import add_var_bound, set_var_valid_value
from pyomo.environ import (
    Block,
    ConcreteModel,
    Constraint,
    ConstraintList,
    Integers,
    Var,
    value,
)


class UnitTestMindtPy(unittest.TestCase):
    def test_set_var_valid_value(self):
        m = ConcreteModel()
        m.x1 = Var(within=Integers, bounds=(-1, 4), initialize=0)

        set_var_valid_value(
            m.x1,
            var_val=5,
            integer_tolerance=1e-6,
            zero_tolerance=1e-6,
            ignore_integrality=False,
        )
        self.assertEqual(m.x1.value, 4)

        set_var_valid_value(
            m.x1,
            var_val=-2,
            integer_tolerance=1e-6,
            zero_tolerance=1e-6,
            ignore_integrality=False,
        )
        self.assertEqual(m.x1.value, -1)

        set_var_valid_value(
            m.x1,
            var_val=1.1,
            integer_tolerance=1e-6,
            zero_tolerance=1e-6,
            ignore_integrality=True,
        )
        self.assertEqual(m.x1.value, 1.1)

        set_var_valid_value(
            m.x1,
            var_val=2.00000001,
            integer_tolerance=1e-6,
            zero_tolerance=1e-6,
            ignore_integrality=False,
        )
        self.assertEqual(m.x1.value, 2)

        set_var_valid_value(
            m.x1,
            var_val=0.0000001,
            integer_tolerance=1e-9,
            zero_tolerance=1e-6,
            ignore_integrality=False,
        )
        self.assertEqual(m.x1.value, 0)

    def test_add_var_bound(self):
        m = SimpleMINLP5().clone()
        m.x.lb = None
        m.x.ub = None
        m.y.lb = None
        m.y.ub = None
        solver_object = _MindtPyAlgorithm()
        solver_object.config = _get_MindtPy_OA_config()
        solver_object.set_up_solve_data(m)
        solver_object.create_utility_block(solver_object.working_model, 'MindtPy_utils')
        add_var_bound(solver_object.working_model, solver_object.config)
        self.assertEqual(
            solver_object.working_model.x.lower,
            -solver_object.config.continuous_var_bound - 1,
        )
        self.assertEqual(
            solver_object.working_model.x.upper,
            solver_object.config.continuous_var_bound,
        )
        self.assertEqual(
            solver_object.working_model.y.lower,
            -solver_object.config.integer_var_bound - 1,
        )
        self.assertEqual(
            solver_object.working_model.y.upper, solver_object.config.integer_var_bound
        )

    @unittest.skipIf(not mcpp_available(), "MC++ is not available")
    def test_goa_affine_cut_uses_convex_slope_for_upper_cut(self):
        m = ConcreteModel()
        m.P = Var(bounds=(0, 10), initialize=0.05)
        m.Q = Var(bounds=(1, 10), initialize=1.000002190520329)
        m.F = Var(bounds=(0, 10), initialize=0.050000453446344)
        m.QP = Var(bounds=(0, 10), initialize=1.0)
        m.c = Constraint(expr=m.P * m.Q - m.F * m.QP == 0)

        m.MindtPy_utils = Block()
        m.MindtPy_utils.nonlinear_constraint_list = [m.c]
        m.MindtPy_utils.cuts = Block()
        m.MindtPy_utils.cuts.aff_cuts = ConstraintList()

        config = Bunch(logger=logging.getLogger('pyomo.contrib.mindtpy.tests'))
        add_affine_cuts(m, config, Bunch())

        feasible_point = [(m.P, 0.05), (m.Q, 1.1), (m.F, 0.055), (m.QP, 1.0)]
        for var, val in feasible_point:
            var.set_value(val)

        aff_cuts = list(m.MindtPy_utils.cuts.aff_cuts.values())
        self.assertEqual(len(aff_cuts), 2)
        convex_cut = aff_cuts[1]
        self.assertLessEqual(value(convex_cut.body), value(convex_cut.upper) + 1e-12)
        self.assertAlmostEqual(value(convex_cut.body), -0.5)


if __name__ == '__main__':
    unittest.main()
