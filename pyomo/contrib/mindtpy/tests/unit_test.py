# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Unit tests for MindtPy utility helpers."""

import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.util import set_var_valid_value

from pyomo.environ import Var, Integers, ConcreteModel
from pyomo.contrib.mindtpy.algorithm_base_class import _MindtPyAlgorithm
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_OA_config
from pyomo.contrib.mindtpy.tests.minlp5_simple import Minlp5Simple
from pyomo.contrib.mindtpy.util import add_var_bound


class UnitTestMindtPy(unittest.TestCase):
    """Unit tests for selected MindtPy helper functions."""

    def test_set_var_valid_value(self):
        """Verify value coercion and bound handling for integer variables."""
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
        """Verify default bounds are added when variable bounds are missing."""
        m = Minlp5Simple().clone()
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

    def test_legacy_test_model_imports(self):
        """Verify legacy MindtPy test-model import paths remain available."""
        from pyomo.contrib.mindtpy.tests.MINLP_simple import SimpleMINLP
        from pyomo.contrib.mindtpy.tests.MINLP2_simple import (
            SimpleMINLP as SimpleMINLP2,
        )
        from pyomo.contrib.mindtpy.tests.MINLP3_simple import (
            SimpleMINLP as SimpleMINLP3,
        )
        from pyomo.contrib.mindtpy.tests.MINLP4_simple import SimpleMINLP4
        from pyomo.contrib.mindtpy.tests.MINLP5_simple import SimpleMINLP5
        from pyomo.contrib.mindtpy.tests.feasibility_pump1 import FeasPump1
        from pyomo.contrib.mindtpy.tests.feasibility_pump2 import FeasPump2
        from pyomo.contrib.mindtpy.tests.from_proposal import ProposalModel
        from pyomo.contrib.mindtpy.tests.minlp_simple import MinlpSimple
        from pyomo.contrib.mindtpy.tests.minlp2_simple import Minlp2Simple
        from pyomo.contrib.mindtpy.tests.minlp3_simple import Minlp3Simple
        from pyomo.contrib.mindtpy.tests.minlp4_simple import Minlp4Simple
        from pyomo.contrib.mindtpy.tests.minlp5_simple import Minlp5Simple
        from pyomo.contrib.mindtpy.tests.feasibility_pump1 import FeasibilityPump1
        from pyomo.contrib.mindtpy.tests.feasibility_pump2 import FeasibilityPump2
        from pyomo.contrib.mindtpy.tests.from_proposal import FromProposalModel

        self.assertIs(SimpleMINLP, MinlpSimple)
        self.assertIs(SimpleMINLP2, Minlp2Simple)
        self.assertIs(SimpleMINLP3, Minlp3Simple)
        self.assertIs(SimpleMINLP4, Minlp4Simple)
        self.assertIs(SimpleMINLP5, Minlp5Simple)
        self.assertIs(FeasPump1, FeasibilityPump1)
        self.assertIs(FeasPump2, FeasibilityPump2)
        self.assertIs(ProposalModel, FromProposalModel)


if __name__ == '__main__':
    unittest.main()
