#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.util.subsystems import (
        create_subsystem_block,
        SubsystemManager,
        ParamSweeper,
        )


class TestSubsystemBlock(unittest.TestCase):
    def _make_simple_model(self):
        m = pyo.ConcreteModel()

        m.v1 = pyo.Var(bounds=(0, None))
        m.v2 = pyo.Var(bounds=(0, None))
        m.v3 = pyo.Var()
        m.v4 = pyo.Var()

        m.con1 = pyo.Constraint(expr=m.v1*m.v2*m.v3 == m.v4)
        m.con2 = pyo.Constraint(expr=m.v1 + m.v2**2 == 2*m.v4)
        m.con3 = pyo.Constraint(expr=m.v1**2 - m.v3 == 3*m.v4)

        return m

    def test_square_subsystem(self):
        m = self._make_simple_model()

        cons = [m.con2, m.con3]
        vars = [m.v1, m.v2]
        # With m.v3 and m.v4 fixed, m.con2 and m.con3 form a square subsystem
        block = create_subsystem_block(cons, vars)

        self.assertEqual(len(block.vars), 2)
        self.assertEqual(len(block.cons), 2)
        self.assertEqual(len(block.input_vars), 2)
        self.assertEqual(len([v for v in block.component_data_objects(pyo.Var)
            if not v.fixed]), 4)

        block.input_vars.fix()
        self.assertEqual(len([v for v in block.component_data_objects(pyo.Var)
            if not v.fixed]), 2)

        self.assertIs(block.cons[0], m.con2)
        self.assertIs(block.cons[1], m.con3)
        self.assertIs(block.vars[0], m.v1)
        self.assertIs(block.vars[1], m.v2)
        self.assertIs(block.input_vars[0], m.v4)
        self.assertIs(block.input_vars[1], m.v3)

        # Make sure block is not part of the original model's tree. We
        # don't want to alter the user's model at all.
        self.assertIsNot(block.model(), m)

        # Components on the block are references to components on the
        # original model
        for comp in block.component_objects((pyo.Var, pyo.Constraint)):
            self.assertTrue(comp.is_reference())
            for data in comp.values():
                self.assertIs(data.model(), m)

    def test_subsystem_inputs_only(self):
        m = self._make_simple_model()

        cons = [m.con2, m.con3]
        block = create_subsystem_block(cons)

        self.assertEqual(len(block.vars), 0)
        self.assertEqual(len(block.input_vars), 4)
        self.assertEqual(len(block.cons), 2)

        self.assertEqual(len([v for v in block.component_data_objects(pyo.Var)
            if not v.fixed]), 4)

        block.input_vars.fix()
        self.assertEqual(len([v for v in block.component_data_objects(pyo.Var)
            if not v.fixed]), 0)

        var_set = ComponentSet([m.v1, m.v2, m.v3, m.v4])
        self.assertIs(block.cons[0], m.con2)
        self.assertIs(block.cons[1], m.con3)
        self.assertIn(block.input_vars[0], var_set)
        self.assertIn(block.input_vars[1], var_set)
        self.assertIn(block.input_vars[2], var_set)
        self.assertIn(block.input_vars[3], var_set)

        # Make sure block is not part of the original model's tree. We
        # don't want to alter the user's model at all.
        self.assertIsNot(block.model(), m)

        # Components on the block are references to components on the
        # original model
        for comp in block.component_objects((pyo.Var, pyo.Constraint)):
            self.assertTrue(comp.is_reference())
            for data in comp.values():
                self.assertIs(data.model(), m)

    @unittest.skipUnless(pyo.SolverFactory("ipopt").available(),
            "Ipopt is not available")
    def test_solve_subsystem(self):
        # This is a test of this function's intended use. We extract a
        # subsystem then solve it without altering the rest of the model.
        m = self._make_simple_model()
        ipopt = pyo.SolverFactory("ipopt")

        m.v5 = pyo.Var(initialize=1.0)
        m.c4 = pyo.Constraint(expr=m.v5 == 5.0)

        cons = [m.con2, m.con3]
        vars = [m.v1, m.v2]
        block = create_subsystem_block(cons, vars)

        m.v3.fix(1.0)
        m.v4.fix(2.0)

        # Initialize to avoid converging infeasible due to bad pivots
        m.v1.set_value(1.0)
        m.v2.set_value(1.0)
        ipopt.solve(block)

        # Have solved model to expected values
        self.assertAlmostEqual(m.v1.value, pyo.sqrt(7.0), delta=1e-8)
        self.assertAlmostEqual(m.v2.value, pyo.sqrt(4.0-pyo.sqrt(7.0)),
                delta=1e-8)

        # Rest of model has not changed
        self.assertEqual(m.v5.value, 1.0)


if __name__ == '__main__':
    unittest.main()
