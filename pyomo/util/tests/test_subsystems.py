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

import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.subsystems import (
    create_subsystem_block,
    generate_subsystem_blocks,
    TemporarySubsystemManager,
    ParamSweeper,
    identify_external_functions,
    add_local_external_functions,
)
from pyomo.common.gsl import find_GSL


def _make_simple_model():
    m = pyo.ConcreteModel()

    m.v1 = pyo.Var(bounds=(0, None))
    m.v2 = pyo.Var(bounds=(0, None))
    m.v3 = pyo.Var()
    m.v4 = pyo.Var()

    m.con1 = pyo.Constraint(expr=m.v1 * m.v2 * m.v3 == m.v4)
    m.con2 = pyo.Constraint(expr=m.v1 + m.v2**2 == 2 * m.v4)
    m.con3 = pyo.Constraint(expr=m.v1**2 - m.v3 == 3 * m.v4)

    return m


class TestSubsystemBlock(unittest.TestCase):
    def test_square_subsystem(self):
        m = _make_simple_model()

        cons = [m.con2, m.con3]
        vars = [m.v1, m.v2]
        # With m.v3 and m.v4 fixed, m.con2 and m.con3 form a square subsystem
        block = create_subsystem_block(cons, vars)

        self.assertEqual(len(block.vars), 2)
        self.assertEqual(len(block.cons), 2)
        self.assertEqual(len(block.input_vars), 2)
        self.assertEqual(
            len([v for v in block.component_data_objects(pyo.Var) if not v.fixed]), 4
        )

        block.input_vars.fix()
        self.assertEqual(
            len([v for v in block.component_data_objects(pyo.Var) if not v.fixed]), 2
        )

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
        m = _make_simple_model()

        cons = [m.con2, m.con3]
        block = create_subsystem_block(cons)

        self.assertEqual(len(block.vars), 0)
        self.assertEqual(len(block.input_vars), 4)
        self.assertEqual(len(block.cons), 2)

        self.assertEqual(
            len([v for v in block.component_data_objects(pyo.Var) if not v.fixed]), 4
        )

        block.input_vars.fix()
        self.assertEqual(
            len([v for v in block.component_data_objects(pyo.Var) if not v.fixed]), 0
        )

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

    @unittest.skipUnless(
        pyo.SolverFactory("ipopt").available(), "Ipopt is not available"
    )
    def test_solve_subsystem(self):
        # This is a test of this function's intended use. We extract a
        # subsystem then solve it without altering the rest of the model.
        m = _make_simple_model()
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
        self.assertAlmostEqual(m.v2.value, pyo.sqrt(4.0 - pyo.sqrt(7.0)), delta=1e-8)

        # Rest of model has not changed
        self.assertEqual(m.v5.value, 1.0)

    def test_generate_subsystems_without_fixed_var(self):
        m = _make_simple_model()
        subs = [([m.con1], [m.v1, m.v4]), ([m.con2, m.con3], [m.v2, m.v3])]
        other_vars = [[m.v2, m.v3], [m.v1, m.v4]]
        for i, (block, inputs) in enumerate(generate_subsystem_blocks(subs)):
            with TemporarySubsystemManager(to_fix=inputs):
                self.assertIs(block.model(), block)
                var_set = ComponentSet(subs[i][1])
                con_set = ComponentSet(subs[i][0])
                input_set = ComponentSet(other_vars[i])

                self.assertEqual(len(var_set), len(block.vars))
                self.assertEqual(len(con_set), len(block.cons))
                self.assertEqual(len(input_set), len(block.input_vars))
                self.assertTrue(all(var in var_set for var in block.vars[:]))
                self.assertTrue(all(con in con_set for con in block.cons[:]))
                self.assertTrue(all(var in input_set for var in inputs))
                self.assertTrue(all(var.fixed for var in inputs))
                self.assertFalse(any(var.fixed for var in block.vars[:]))

        # Test that we have properly unfixed variables
        self.assertFalse(any(var.fixed for var in m.component_data_objects(pyo.Var)))

    def test_generate_subsystems_with_exception(self):
        m = _make_simple_model()
        subs = [([m.con1], [m.v1, m.v4]), ([m.con2, m.con3], [m.v2, m.v3])]
        other_vars = [[m.v2, m.v3], [m.v1, m.v4]]
        with self.assertRaises(RuntimeError):
            for i, (block, inputs) in enumerate(generate_subsystem_blocks(subs)):
                with TemporarySubsystemManager(to_fix=inputs):
                    self.assertTrue(all(var.fixed for var in inputs))
                    self.assertFalse(any(var.fixed for var in block.vars[:]))
                    if i == 1:
                        raise RuntimeError()

        # Test that we have properly unfixed variables
        self.assertFalse(any(var.fixed for var in m.component_data_objects(pyo.Var)))

    def test_generate_subsystems_with_fixed_var(self):
        m = _make_simple_model()
        m.v4.fix()
        subs = [([m.con1], [m.v1]), ([m.con2, m.con3], [m.v2, m.v3])]
        other_vars = [[m.v2, m.v3], [m.v1]]
        for i, (block, inputs) in enumerate(generate_subsystem_blocks(subs)):
            inputs = list(block.input_vars.values())
            with TemporarySubsystemManager(to_fix=inputs):
                self.assertIs(block.model(), block)
                var_set = ComponentSet(subs[i][1])
                con_set = ComponentSet(subs[i][0])
                input_set = ComponentSet(other_vars[i])

                self.assertEqual(len(var_set), len(block.vars))
                self.assertEqual(len(con_set), len(block.cons))
                self.assertEqual(len(input_set), len(inputs))
                self.assertTrue(all(var in var_set for var in block.vars[:]))
                self.assertTrue(all(con in con_set for con in block.cons[:]))
                self.assertTrue(all(var in input_set for var in inputs))
                self.assertTrue(all(var.fixed for var in inputs))
                self.assertFalse(any(var.fixed for var in block.vars[:]))

        # Test that we have properly unfixed variables, except variables
        # that were already fixed.
        self.assertFalse(m.v1.fixed)
        self.assertFalse(m.v2.fixed)
        self.assertFalse(m.v3.fixed)
        self.assertTrue(m.v4.fixed)

    def test_generate_subsystems_include_fixed_var(self):
        m = _make_simple_model()
        m.v4.fix()
        subsystems = [([m.con1], [m.v1]), ([m.con2, m.con3], [m.v2, m.v3])]
        other_vars = [[m.v2, m.v3, m.v4], [m.v1, m.v4]]
        for i, (block, inputs) in enumerate(
            generate_subsystem_blocks(subsystems, include_fixed=True)
        ):
            with TemporarySubsystemManager(to_fix=inputs):
                self.assertIs(block.model(), block)
                var_set = ComponentSet(subsystems[i][1])
                con_set = ComponentSet(subsystems[i][0])
                input_set = ComponentSet(other_vars[i])

                self.assertEqual(len(var_set), len(block.vars))
                self.assertEqual(len(con_set), len(block.cons))
                self.assertEqual(len(input_set), len(block.input_vars))
                self.assertTrue(all(var in var_set for var in block.vars[:]))
                self.assertTrue(all(con in con_set for con in block.cons[:]))
                self.assertTrue(all(var in input_set for var in inputs))
                self.assertTrue(all(var.fixed for var in inputs))
                self.assertFalse(any(var.fixed for var in block.vars[:]))

        self.assertFalse(m.v1.fixed)
        self.assertFalse(m.v2.fixed)
        self.assertFalse(m.v3.fixed)
        self.assertTrue(m.v4.fixed)

    def test_generate_subsystems_dont_fix_inputs(self):
        m = _make_simple_model()
        subs = [([m.con1], [m.v1]), ([m.con2, m.con3], [m.v2, m.v3])]
        other_vars = [[m.v2, m.v3, m.v4], [m.v1, m.v4]]
        for i, (block, inputs) in enumerate(generate_subsystem_blocks(subs)):
            self.assertIs(block.model(), block)
            var_set = ComponentSet(subs[i][1])
            con_set = ComponentSet(subs[i][0])
            input_set = ComponentSet(other_vars[i])

            self.assertEqual(len(var_set), len(block.vars))
            self.assertEqual(len(con_set), len(block.cons))
            self.assertEqual(len(input_set), len(inputs))
            self.assertTrue(all(var in var_set for var in block.vars[:]))
            self.assertTrue(all(con in con_set for con in block.cons[:]))
            self.assertTrue(all(var in input_set for var in inputs))
            self.assertFalse(any(var.fixed for var in inputs))
            self.assertFalse(any(var.fixed for var in block.vars[:]))

        self.assertFalse(m.v1.fixed)
        self.assertFalse(m.v2.fixed)
        self.assertFalse(m.v3.fixed)
        self.assertFalse(m.v4.fixed)

    def test_generate_dont_fix_inputs_with_fixed_var(self):
        m = _make_simple_model()
        m.v4.fix()
        subs = [([m.con1], [m.v1]), ([m.con2, m.con3], [m.v2, m.v3])]
        other_vars = [[m.v2, m.v3], [m.v1]]
        for i, (block, inputs) in enumerate(generate_subsystem_blocks(subs)):
            self.assertIs(block.model(), block)
            var_set = ComponentSet(subs[i][1])
            con_set = ComponentSet(subs[i][0])
            input_set = ComponentSet(other_vars[i])

            self.assertEqual(len(var_set), len(block.vars))
            self.assertEqual(len(con_set), len(block.cons))
            self.assertEqual(len(input_set), len(inputs))
            self.assertTrue(all(var in var_set for var in block.vars[:]))
            self.assertTrue(all(con in con_set for con in block.cons[:]))
            self.assertTrue(all(var in input_set for var in inputs))
            self.assertFalse(m.v1.fixed)
            self.assertFalse(m.v2.fixed)
            self.assertFalse(m.v3.fixed)
            self.assertTrue(m.v4.fixed)

        self.assertFalse(m.v1.fixed)
        self.assertFalse(m.v2.fixed)
        self.assertFalse(m.v3.fixed)
        self.assertTrue(m.v4.fixed)

    def _make_model_with_external_functions(self, named_expressions=False):
        m = pyo.ConcreteModel()
        gsl = find_GSL()
        m.bessel = pyo.ExternalFunction(library=gsl, function="gsl_sf_bessel_J0")
        m.fermi = pyo.ExternalFunction(library=gsl, function="gsl_sf_fermi_dirac_m1")
        m.v1 = pyo.Var(initialize=1.0)
        m.v2 = pyo.Var(initialize=2.0)
        m.v3 = pyo.Var(initialize=3.0)
        if named_expressions:
            m.subexpr = pyo.Expression(pyo.PositiveIntegers)
            m.subexpr[1] = 2 * m.fermi(m.v1)
            m.subexpr[2] = m.bessel(m.v1) - m.bessel(m.v2)
            m.subexpr[3] = m.subexpr[2] + m.v3**2
            subexpr1 = m.subexpr[1]
            subexpr2 = m.subexpr[2]
            subexpr3 = m.subexpr[3]
        else:
            subexpr1 = 2 * m.fermi(m.v1)
            subexpr2 = m.bessel(m.v1) - m.bessel(m.v2)
            subexpr3 = subexpr2 + m.v3**2
        m.con1 = pyo.Constraint(expr=m.v1 == 0.5)
        m.con2 = pyo.Constraint(expr=subexpr1 + m.v2**2 - m.v3 == 1.0)
        m.con3 = pyo.Constraint(expr=subexpr3 == 2.0)
        return m

    @unittest.skipUnless(find_GSL(), "Could not find the AMPL GSL library")
    def test_identify_external_functions(self):
        m = self._make_model_with_external_functions()
        m._con = pyo.Constraint(expr=2 * m.fermi(m.bessel(m.v1**2) + 0.1) == 1.0)

        gsl = find_GSL()

        fcns = list(identify_external_functions(m.con2.expr))
        self.assertEqual(len(fcns), 1)
        self.assertEqual(fcns[0]._fcn._library, gsl)
        self.assertEqual(fcns[0]._fcn._function, "gsl_sf_fermi_dirac_m1")

        fcns = list(identify_external_functions(m.con3.expr))
        fcn_data = set((fcn._fcn._library, fcn._fcn._function) for fcn in fcns)
        self.assertEqual(len(fcns), 2)
        pred_fcn_data = {(gsl, "gsl_sf_bessel_J0")}
        self.assertEqual(fcn_data, pred_fcn_data)

        fcns = list(identify_external_functions(m._con.expr))
        fcn_data = set((fcn._fcn._library, fcn._fcn._function) for fcn in fcns)
        self.assertEqual(len(fcns), 2)
        pred_fcn_data = {(gsl, "gsl_sf_bessel_J0"), (gsl, "gsl_sf_fermi_dirac_m1")}
        self.assertEqual(fcn_data, pred_fcn_data)

    @unittest.skipUnless(find_GSL(), "Could not find the AMPL GSL library")
    def test_local_external_functions_with_named_expressions(self):
        m = self._make_model_with_external_functions(named_expressions=True)
        variables = list(m.component_data_objects(pyo.Var))
        constraints = list(m.component_data_objects(pyo.Constraint, active=True))
        b = create_subsystem_block(constraints, variables)
        self.assertTrue(isinstance(b._gsl_sf_bessel_J0, pyo.ExternalFunction))
        self.assertTrue(isinstance(b._gsl_sf_fermi_dirac_m1, pyo.ExternalFunction))

    def _solve_ef_model_with_ipopt(self):
        m = self._make_model_with_external_functions()
        ipopt = pyo.SolverFactory("ipopt")
        ipopt.solve(m)
        return m

    @unittest.skipUnless(find_GSL(), "Could not find the AMPL GSL library")
    @unittest.skipUnless(
        pyo.SolverFactory("ipopt").available(), "ipopt is not available"
    )
    def test_with_external_function(self):
        m = self._make_model_with_external_functions()
        subsystem = ([m.con2, m.con3], [m.v2, m.v3])

        m.v1.set_value(0.5)
        block = create_subsystem_block(*subsystem)
        ipopt = pyo.SolverFactory("ipopt")
        with TemporarySubsystemManager(to_fix=list(block.input_vars.values())):
            ipopt.solve(block)

        # Correct values obtained by solving with Ipopt directly
        # in another script.
        self.assertEqual(m.v1.value, 0.5)
        self.assertFalse(m.v1.fixed)
        self.assertAlmostEqual(m.v2.value, 1.04816, delta=1e-5)
        self.assertAlmostEqual(m.v3.value, 1.34356, delta=1e-5)

        # Result obtained by solving the full system
        m_full = self._solve_ef_model_with_ipopt()
        self.assertAlmostEqual(m.v1.value, m_full.v1.value)
        self.assertAlmostEqual(m.v2.value, m_full.v2.value)
        self.assertAlmostEqual(m.v3.value, m_full.v3.value)

    @unittest.skipUnless(find_GSL(), "Could not find the AMPL GSL library")
    @unittest.skipUnless(
        pyo.SolverFactory("ipopt").available(), "ipopt is not available"
    )
    def test_with_external_function_in_named_expression(self):
        m = self._make_model_with_external_functions(named_expressions=True)
        subsystem = ([m.con2, m.con3], [m.v2, m.v3])

        m.v1.set_value(0.5)
        block = create_subsystem_block(*subsystem)
        ipopt = pyo.SolverFactory("ipopt")
        with TemporarySubsystemManager(to_fix=list(block.input_vars.values())):
            ipopt.solve(block)

        # Correct values obtained by solving with Ipopt directly
        # in another script.
        self.assertEqual(m.v1.value, 0.5)
        self.assertFalse(m.v1.fixed)
        self.assertAlmostEqual(m.v2.value, 1.04816, delta=1e-5)
        self.assertAlmostEqual(m.v3.value, 1.34356, delta=1e-5)

        # Result obtained by solving the full system
        m_full = self._solve_ef_model_with_ipopt()
        self.assertAlmostEqual(m.v1.value, m_full.v1.value)
        self.assertAlmostEqual(m.v2.value, m_full.v2.value)
        self.assertAlmostEqual(m.v3.value, m_full.v3.value)

    @unittest.skipUnless(find_GSL(), "Could not find the AMPL GSL library")
    def test_external_function_with_potential_name_collision(self):
        m = self._make_model_with_external_functions()
        m.b = pyo.Block()
        m.b._gsl_sf_bessel_J0 = pyo.Var()
        m.b.con = pyo.Constraint(expr=m.b._gsl_sf_bessel_J0 == m.bessel(m.v1))
        add_local_external_functions(m.b)
        self.assertTrue(isinstance(m.b._gsl_sf_bessel_J0, pyo.Var))
        ex_fcns = list(m.b.component_objects(pyo.ExternalFunction))
        self.assertEqual(len(ex_fcns), 1)
        fcn = ex_fcns[0]
        self.assertEqual(fcn._function, "gsl_sf_bessel_J0")


class TestTemporarySubsystemManager(unittest.TestCase):
    def test_context(self):
        m = _make_simple_model()

        to_fix = [m.v4]
        to_deactivate = [m.con1]
        to_reset = [m.v1]

        m.v1.set_value(1.5)

        with TemporarySubsystemManager(to_fix, to_deactivate, to_reset):
            self.assertEqual(m.v1.value, 1.5)
            self.assertTrue(m.v4.fixed)
            self.assertFalse(m.con1.active)

            m.v1.set_value(2.0)
            m.v4.set_value(3.0)

        self.assertEqual(m.v1.value, 1.5)
        self.assertEqual(m.v4.value, 3.0)
        self.assertFalse(m.v4.fixed)
        self.assertTrue(m.con1.active)

    def test_context_some_redundant(self):
        m = _make_simple_model()

        to_fix = [m.v2, m.v4]
        to_deactivate = [m.con1, m.con2]
        to_reset = [m.v1]

        m.v1.set_value(1.5)
        m.v2.fix()
        m.con1.deactivate()

        with TemporarySubsystemManager(to_fix, to_deactivate, to_reset):
            self.assertEqual(m.v1.value, 1.5)
            self.assertTrue(m.v2.fixed)
            self.assertTrue(m.v4.fixed)
            self.assertFalse(m.con1.active)
            self.assertFalse(m.con2.active)

            m.v1.set_value(2.0)
            m.v2.set_value(3.0)

        self.assertEqual(m.v1.value, 1.5)
        self.assertEqual(m.v2.value, 3.0)
        self.assertTrue(m.v2.fixed)
        self.assertFalse(m.v4.fixed)
        self.assertTrue(m.con2.active)
        self.assertFalse(m.con1.active)

    @unittest.skipUnless(
        pyo.SolverFactory("ipopt").available(), "Ipopt is not available"
    )
    def test_fix_then_solve(self):
        # This is a test of the expected use case. We have a (square)
        # subsystem that we can solve easily after fixing and deactivating
        # certain variables and constraints.

        m = _make_simple_model()
        ipopt = pyo.SolverFactory("ipopt")

        # Initialize to avoid converging infeasible due to bad pivots
        m.v1.set_value(1.0)
        m.v2.set_value(1.0)
        m.v3.set_value(1.0)
        m.v4.set_value(2.0)

        with TemporarySubsystemManager(to_fix=[m.v3, m.v4], to_deactivate=[m.con1]):
            # Solve the subsystem with m.v1, m.v2 unfixed and
            # m.con2, m.con3 inactive.
            ipopt.solve(m)

        # Have solved model to expected values
        self.assertAlmostEqual(m.v1.value, pyo.sqrt(7.0), delta=1e-8)
        self.assertAlmostEqual(m.v2.value, pyo.sqrt(4.0 - pyo.sqrt(7.0)), delta=1e-8)

    def test_generate_subsystems_with_exception(self):
        m = _make_simple_model()
        subsystems = [([m.con1], [m.v1, m.v4]), ([m.con2, m.con3], [m.v2, m.v3])]
        other_vars = [[m.v2, m.v3], [m.v1, m.v4]]
        block = create_subsystem_block(*subsystems[0])
        with self.assertRaises(RuntimeError):
            inputs = list(block.input_vars[:])
            with TemporarySubsystemManager(to_fix=inputs):
                self.assertTrue(all(var.fixed for var in inputs))
                self.assertFalse(any(var.fixed for var in block.vars[:]))
                raise RuntimeError()

        # Test that we have properly unfixed variables
        self.assertFalse(any(var.fixed for var in m.component_data_objects(pyo.Var)))

    def test_to_unfix(self):
        m = _make_simple_model()
        m.v1.fix()
        m.v3.fix()
        with TemporarySubsystemManager(to_unfix=[m.v3]):
            self.assertTrue(m.v1.fixed)
            self.assertFalse(m.v2.fixed)
            self.assertFalse(m.v3.fixed)
            self.assertFalse(m.v4.fixed)

        self.assertTrue(m.v1.fixed)
        self.assertFalse(m.v2.fixed)
        self.assertTrue(m.v3.fixed)
        self.assertFalse(m.v4.fixed)


class TestParamSweeper(unittest.TestCase):
    def test_set_values(self):
        m = _make_simple_model()

        n_scenario = 2
        input_values = ComponentMap([(m.v3, [1.3, 2.3]), (m.v4, [1.4, 2.4])])

        to_fix = [m.v3, m.v4]
        to_deactivate = [m.con1]

        with ParamSweeper(
            2, input_values, to_fix=to_fix, to_deactivate=to_deactivate
        ) as sweeper:
            self.assertFalse(m.v1.fixed)
            self.assertFalse(m.v2.fixed)
            self.assertTrue(m.v3.fixed)
            self.assertTrue(m.v4.fixed)
            self.assertFalse(m.con1.active)
            self.assertTrue(m.con2.active)
            self.assertTrue(m.con3.active)
            for i, (inputs, outputs) in enumerate(sweeper):
                self.assertEqual(len(inputs), 2)
                self.assertEqual(len(outputs), 0)
                self.assertIn(m.v3, inputs)
                self.assertIn(m.v4, inputs)
                for var, val in inputs.items():
                    self.assertEqual(var.value, val)
                    self.assertEqual(var.value, input_values[var][i])

        # Values have been reset after exit.
        self.assertIs(m.v3.value, None)
        self.assertIs(m.v4.value, None)

    def test_mutable_parameter(self):
        m = _make_simple_model()
        m.p1 = pyo.Param(mutable=True, initialize=7.0)

        n_scenario = 2
        input_values = ComponentMap(
            [(m.v3, [1.3, 2.3]), (m.v4, [1.4, 2.4]), (m.p1, [1.5, 2.5])]
        )

        to_fix = [m.v3, m.v4]
        to_deactivate = [m.con1]

        with ParamSweeper(
            2, input_values, to_fix=to_fix, to_deactivate=to_deactivate
        ) as sweeper:
            self.assertFalse(m.v1.fixed)
            self.assertFalse(m.v2.fixed)
            self.assertTrue(m.v3.fixed)
            self.assertTrue(m.v4.fixed)
            self.assertFalse(m.con1.active)
            self.assertTrue(m.con2.active)
            self.assertTrue(m.con3.active)
            for i, (inputs, outputs) in enumerate(sweeper):
                self.assertIn(m.v3, inputs)
                self.assertIn(m.v4, inputs)
                self.assertIn(m.p1, inputs)
                self.assertEqual(len(inputs), 3)
                for var, val in inputs.items():
                    self.assertEqual(var.value, val)
                    self.assertEqual(var.value, input_values[var][i])

        # Values have been reset after exit.
        self.assertIs(m.v3.value, None)
        self.assertIs(m.v4.value, None)
        self.assertEqual(m.p1.value, 7.0)

    def test_output_values(self):
        m = _make_simple_model()

        n_scenario = 2
        input_values = ComponentMap([(m.v3, [1.3, 2.3]), (m.v4, [1.4, 2.4])])

        output_values = ComponentMap([(m.v1, [1.1, 2.1]), (m.v2, [1.2, 2.2])])

        to_fix = [m.v3, m.v4]
        to_deactivate = [m.con1]

        with ParamSweeper(
            2, input_values, output_values, to_fix=to_fix, to_deactivate=to_deactivate
        ) as sweeper:
            self.assertFalse(m.v1.fixed)
            self.assertFalse(m.v2.fixed)
            self.assertTrue(m.v3.fixed)
            self.assertTrue(m.v4.fixed)
            self.assertFalse(m.con1.active)
            self.assertTrue(m.con2.active)
            self.assertTrue(m.con3.active)
            for i, (inputs, outputs) in enumerate(sweeper):
                self.assertEqual(len(inputs), 2)
                self.assertEqual(len(outputs), 2)
                self.assertIn(m.v3, inputs)
                self.assertIn(m.v4, inputs)
                self.assertIn(m.v1, outputs)
                self.assertIn(m.v2, outputs)
                for var, val in inputs.items():
                    self.assertEqual(var.value, val)
                    self.assertEqual(var.value, input_values[var][i])

                for var, val in outputs.items():
                    self.assertEqual(val, output_values[var][i])

        # Values have been reset after exit.
        self.assertIs(m.v1.value, None)
        self.assertIs(m.v2.value, None)
        self.assertIs(m.v3.value, None)
        self.assertIs(m.v4.value, None)

    @unittest.skipUnless(
        pyo.SolverFactory("ipopt").available(), "Ipopt is not available"
    )
    def test_with_solve(self):
        m = _make_simple_model()
        ipopt = pyo.SolverFactory("ipopt")

        n_scenario = 2
        input_values = ComponentMap([(m.v3, [1.3, 2.3]), (m.v4, [1.4, 2.4])])

        _v1_val_1 = pyo.sqrt(3 * 1.4 + 1.3)
        _v1_val_2 = pyo.sqrt(3 * 2.4 + 2.3)
        _v2_val_1 = pyo.sqrt(2 * 1.4 - _v1_val_1)
        _v2_val_2 = pyo.sqrt(2 * 2.4 - _v1_val_2)
        output_values = ComponentMap(
            [(m.v1, [_v1_val_1, _v1_val_2]), (m.v2, [_v2_val_1, _v2_val_2])]
        )

        to_fix = [m.v3, m.v4]
        to_deactivate = [m.con1]
        to_reset = [m.v1, m.v2]

        # Initialize values so we don't fail due to bad initialization
        m.v1.set_value(1.0)
        m.v2.set_value(1.0)

        with ParamSweeper(
            n_scenario,
            input_values,
            output_values,
            to_fix=to_fix,
            to_deactivate=to_deactivate,
            to_reset=to_reset,
        ) as sweeper:
            self.assertFalse(m.v1.fixed)
            self.assertFalse(m.v2.fixed)
            self.assertTrue(m.v3.fixed)
            self.assertTrue(m.v4.fixed)
            self.assertFalse(m.con1.active)
            self.assertTrue(m.con2.active)
            self.assertTrue(m.con3.active)
            for i, (inputs, outputs) in enumerate(sweeper):
                ipopt.solve(m)

                for var, val in inputs.items():
                    # These values should not have been altered.
                    # I believe exact equality should be appropriate here.
                    self.assertEqual(var.value, val)
                    self.assertEqual(var.value, input_values[var][i])

                for var, val in outputs.items():
                    self.assertAlmostEqual(var.value, val, delta=1e-8)
                    self.assertAlmostEqual(var.value, output_values[var][i], delta=1e-8)

        # Values have been reset after exit.
        self.assertIs(m.v1.value, 1.0)
        self.assertIs(m.v2.value, 1.0)
        self.assertIs(m.v3.value, None)
        self.assertIs(m.v4.value, None)

    def test_with_exception(self):
        m = _make_simple_model()

        n_scenario = 2
        input_values = ComponentMap([(m.v3, [1.3, 2.3]), (m.v4, [1.4, 2.4])])

        output_values = ComponentMap([(m.v1, [1.1, 2.1]), (m.v2, [1.2, 2.2])])

        to_fix = [m.v3, m.v4]
        to_deactivate = [m.con1]

        with self.assertRaises(RuntimeError):
            with ParamSweeper(
                2,
                input_values,
                output_values,
                to_fix=to_fix,
                to_deactivate=to_deactivate,
            ) as sweeper:
                self.assertFalse(m.v1.fixed)
                self.assertFalse(m.v2.fixed)
                self.assertTrue(m.v3.fixed)
                self.assertTrue(m.v4.fixed)
                self.assertFalse(m.con1.active)
                self.assertTrue(m.con2.active)
                self.assertTrue(m.con3.active)
                for i, (inputs, outputs) in enumerate(sweeper):
                    self.assertEqual(len(inputs), 2)
                    self.assertEqual(len(outputs), 2)
                    self.assertIn(m.v3, inputs)
                    self.assertIn(m.v4, inputs)
                    self.assertIn(m.v1, outputs)
                    self.assertIn(m.v2, outputs)
                    for var, val in inputs.items():
                        self.assertEqual(var.value, val)
                        self.assertEqual(var.value, input_values[var][i])

                    for var, val in outputs.items():
                        self.assertEqual(val, output_values[var][i])

                    if i == 0:
                        raise RuntimeError()

        # Values have been reset after exit.
        self.assertIs(m.v1.value, None)
        self.assertIs(m.v2.value, None)
        self.assertIs(m.v3.value, None)
        self.assertIs(m.v4.value, None)
        self.assertFalse(m.v1.fixed)
        self.assertFalse(m.v2.fixed)
        self.assertFalse(m.v3.fixed)
        self.assertFalse(m.v4.fixed)


if __name__ == '__main__':
    unittest.main()
