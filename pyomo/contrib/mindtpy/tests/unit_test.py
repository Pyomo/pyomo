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

import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.util import set_var_valid_value

from pyomo.environ import Var, Integers, ConcreteModel, Integers
from pyomo.contrib.mindtpy.algorithm_base_class import _MindtPyAlgorithm
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_OA_config
from pyomo.contrib.mindtpy.tests.MINLP5_simple import SimpleMINLP5
from pyomo.contrib.mindtpy.util import add_var_bound


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


if __name__ == '__main__':
    unittest.main()
