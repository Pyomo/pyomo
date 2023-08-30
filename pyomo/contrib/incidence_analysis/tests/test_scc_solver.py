#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.dependencies import networkx_available
from pyomo.common.dependencies import scipy_available
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.scc_solver import (
    TemporarySubsystemManager,
    generate_strongly_connected_components,
    solve_strongly_connected_components,
)
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
    make_gas_expansion_model,
    make_dynamic_model,
)
import pyomo.common.unittest as unittest


@unittest.skipUnless(scipy_available, "SciPy is not available")
@unittest.skipUnless(networkx_available, "NetworkX is not available")
class TestGenerateSCC(unittest.TestCase):
    def test_gas_expansion(self):
        N = 5
        m = make_gas_expansion_model(N)
        m.rho[0].fix()
        m.F[0].fix()
        m.T[0].fix()

        constraints = list(m.component_data_objects(pyo.Constraint))
        self.assertEqual(
            len(list(generate_strongly_connected_components(constraints))), N + 1
        )
        for i, (block, inputs) in enumerate(
            generate_strongly_connected_components(constraints)
        ):
            with TemporarySubsystemManager(to_fix=inputs):
                if i == 0:
                    # P[0], ideal_gas[0]
                    self.assertEqual(len(block.vars), 1)
                    self.assertEqual(len(block.cons), 1)

                    var_set = ComponentSet([m.P[i]])
                    con_set = ComponentSet([m.ideal_gas[i]])
                    for var, con in zip(block.vars[:], block.cons[:]):
                        self.assertIn(var, var_set)
                        self.assertIn(con, con_set)

                    # Other variables are fixed; not included
                    self.assertEqual(len(block.input_vars), 0)

                elif i == 1:
                    # P[1], rho[1], F[1], T[1], etc.
                    self.assertEqual(len(block.vars), 4)
                    self.assertEqual(len(block.cons), 4)

                    var_set = ComponentSet([m.P[i], m.rho[i], m.F[i], m.T[i]])
                    con_set = ComponentSet(
                        [m.ideal_gas[i], m.mbal[i], m.ebal[i], m.expansion[i]]
                    )
                    for var, con in zip(block.vars[:], block.cons[:]):
                        self.assertIn(var, var_set)
                        self.assertIn(con, con_set)

                    # P[0] is in expansion[1]
                    other_var_set = ComponentSet([m.P[i - 1]])
                    self.assertEqual(len(block.input_vars), 1)
                    for var in block.input_vars[:]:
                        self.assertIn(var, other_var_set)

                else:
                    # P[i], rho[i], F[i], T[i], etc.
                    self.assertEqual(len(block.vars), 4)
                    self.assertEqual(len(block.cons), 4)

                    var_set = ComponentSet([m.P[i], m.rho[i], m.F[i], m.T[i]])
                    con_set = ComponentSet(
                        [m.ideal_gas[i], m.mbal[i], m.ebal[i], m.expansion[i]]
                    )
                    for var, con in zip(block.vars[:], block.cons[:]):
                        self.assertIn(var, var_set)
                        self.assertIn(con, con_set)

                    # P[i-1], rho[i-1], F[i-1], T[i-1], etc.
                    other_var_set = ComponentSet(
                        [m.P[i - 1], m.rho[i - 1], m.F[i - 1], m.T[i - 1]]
                    )
                    self.assertEqual(len(block.input_vars), 4)
                    for var in block.input_vars[:]:
                        self.assertIn(var, other_var_set)

    def test_dynamic_backward_disc_with_initial_conditions(self):
        nfe = 5
        m = make_dynamic_model(nfe=nfe, scheme="BACKWARD")
        time = m.time
        t0 = m.time.first()
        t1 = m.time.next(t0)

        m.flow_in.fix()
        m.height[t0].fix()

        constraints = list(m.component_data_objects(pyo.Constraint))
        self.assertEqual(
            len(list(generate_strongly_connected_components(constraints))),
            nfe + 2,
            # The "initial constraints" have two SCCs because they
            # decompose into the algebraic equation and differential
            # equation. This decomposition is because the discretization
            # equation is not present.
            #
            # This is actually quite troublesome for testing because
            # it means that the topological order of strongly connected
            # components is not unique (alternatively, the initial
            # conditions and rest of the model are independent, or the
            # bipartite graph of variables and equations is disconnected).
        )
        t_scc_map = {}
        for i, (block, inputs) in enumerate(
            generate_strongly_connected_components(constraints)
        ):
            with TemporarySubsystemManager(to_fix=inputs):
                t = block.vars[0].index()
                t_scc_map[t] = i
                if t == t0:
                    continue
                else:
                    t_prev = m.time.prev(t)

                    con_set = ComponentSet(
                        [m.diff_eqn[t], m.flow_out_eqn[t], m.dhdt_disc_eq[t]]
                    )
                    var_set = ComponentSet([m.height[t], m.dhdt[t], m.flow_out[t]])
                    self.assertEqual(len(con_set), len(block.cons))
                    self.assertEqual(len(var_set), len(block.vars))
                    for var, con in zip(block.vars[:], block.cons[:]):
                        self.assertIn(var, var_set)
                        self.assertIn(con, con_set)
                        self.assertFalse(var.fixed)

                    other_var_set = (
                        ComponentSet([m.height[t_prev]]) if t != t1 else ComponentSet()
                    )
                    # At t1, "input var" height[t0] is fixed, so
                    # it is not included here.
                    self.assertEqual(len(inputs), len(other_var_set))
                    for var in block.input_vars[:]:
                        self.assertIn(var, other_var_set)
                        self.assertTrue(var.fixed)

        scc = -1
        for t in m.time:
            if t == t0:
                self.assertTrue(m.height[t].fixed)
            else:
                self.assertFalse(m.height[t].fixed)

                # Make sure "finite element blocks" in the SCC DAG are
                # in a valid topological order
                self.assertGreater(t_scc_map[t], scc)
                scc = t_scc_map[t]

            self.assertFalse(m.flow_out[t].fixed)
            self.assertFalse(m.dhdt[t].fixed)
            self.assertTrue(m.flow_in[t].fixed)

    def test_dynamic_backward_disc_without_initial_conditions(self):
        nfe = 5
        m = make_dynamic_model(nfe=nfe, scheme="BACKWARD")
        time = m.time
        t0 = m.time.first()
        t1 = m.time.next(t0)

        m.flow_in.fix()
        m.height[t0].fix()
        m.flow_out[t0].fix()
        m.dhdt[t0].fix()
        m.diff_eqn[t0].deactivate()
        m.flow_out_eqn[t0].deactivate()

        constraints = list(m.component_data_objects(pyo.Constraint, active=True))
        self.assertEqual(
            len(list(generate_strongly_connected_components(constraints))), nfe
        )
        for i, (block, inputs) in enumerate(
            generate_strongly_connected_components(constraints)
        ):
            with TemporarySubsystemManager(to_fix=inputs):
                # We have a much easier time testing the SCCs generated
                # in this test.
                t = m.time[i + 2]
                t_prev = m.time.prev(t)

                con_set = ComponentSet(
                    [m.diff_eqn[t], m.flow_out_eqn[t], m.dhdt_disc_eq[t]]
                )
                var_set = ComponentSet([m.height[t], m.dhdt[t], m.flow_out[t]])
                self.assertEqual(len(con_set), len(block.cons))
                self.assertEqual(len(var_set), len(block.vars))
                for var, con in zip(block.vars[:], block.cons[:]):
                    self.assertIn(var, var_set)
                    self.assertIn(con, con_set)
                    self.assertFalse(var.fixed)

                other_var_set = (
                    ComponentSet([m.height[t_prev]]) if t != t1 else ComponentSet()
                )
                # At t1, "input var" height[t0] is fixed, so
                # it is not included here.
                self.assertEqual(len(inputs), len(other_var_set))
                for var in block.input_vars[:]:
                    self.assertIn(var, other_var_set)
                    self.assertTrue(var.fixed)

        for t in time:
            if t == t0:
                self.assertTrue(m.height[t].fixed)
                self.assertTrue(m.flow_out[t].fixed)
                self.assertTrue(m.dhdt[t].fixed)
            else:
                self.assertFalse(m.height[t].fixed)
                self.assertFalse(m.flow_out[t].fixed)
                self.assertFalse(m.dhdt[t].fixed)

    def test_dynamic_backward_with_inputs(self):
        nfe = 5
        m = make_dynamic_model(nfe=nfe, scheme="BACKWARD")
        time = m.time
        t0 = m.time.first()
        t1 = m.time.next(t0)

        # Initial conditions are still fixed
        m.height[t0].fix()
        m.flow_out[t0].fix()
        m.dhdt[t0].fix()
        m.diff_eqn[t0].deactivate()
        m.flow_out_eqn[t0].deactivate()

        # Variables that we want in our SCCs:
        # Here we exclude "dynamic inputs" (flow_in) instead of fixing them
        variables = [
            var
            for var in m.component_data_objects(pyo.Var)
            if not var.fixed and var.parent_component() is not m.flow_in
        ]
        constraints = list(m.component_data_objects(pyo.Constraint, active=True))
        self.assertEqual(
            len(list(generate_strongly_connected_components(constraints, variables))),
            nfe,
        )

        # The result of the generator is the same as in the previous
        # test, but we are using the more general API
        for i, (block, inputs) in enumerate(
            generate_strongly_connected_components(constraints, variables)
        ):
            with TemporarySubsystemManager(to_fix=inputs):
                t = m.time[i + 2]
                t_prev = m.time.prev(t)

                con_set = ComponentSet(
                    [m.diff_eqn[t], m.flow_out_eqn[t], m.dhdt_disc_eq[t]]
                )
                var_set = ComponentSet([m.height[t], m.dhdt[t], m.flow_out[t]])
                self.assertEqual(len(con_set), len(block.cons))
                self.assertEqual(len(var_set), len(block.vars))
                for var, con in zip(block.vars[:], block.cons[:]):
                    self.assertIn(var, var_set)
                    self.assertIn(con, con_set)
                    self.assertFalse(var.fixed)

                other_var_set = ComponentSet([m.flow_in[t]])
                if t != t1:
                    other_var_set.add(m.height[t_prev])
                    # At t1, "input var" height[t0] is fixed, so
                    # it is not included here.
                self.assertEqual(len(inputs), len(other_var_set))
                for var in block.input_vars[:]:
                    self.assertIn(var, other_var_set)
                    self.assertTrue(var.fixed)

        for t in time:
            if t == t0:
                self.assertTrue(m.height[t].fixed)
                self.assertTrue(m.flow_out[t].fixed)
                self.assertTrue(m.dhdt[t].fixed)
            else:
                self.assertFalse(m.height[t].fixed)
                self.assertFalse(m.flow_out[t].fixed)
                self.assertFalse(m.dhdt[t].fixed)

    def test_dynamic_forward_disc(self):
        nfe = 5
        m = make_dynamic_model(nfe=nfe, scheme="FORWARD")
        time = m.time
        t0 = m.time.first()
        t1 = m.time.next(t0)

        m.flow_in.fix()
        m.height[t0].fix()

        constraints = list(m.component_data_objects(pyo.Constraint))
        # For a forward discretization, the entire model decomposes
        self.assertEqual(
            len(list(generate_strongly_connected_components(constraints))),
            len(list(m.component_data_objects(pyo.Constraint))),
        )
        self.assertEqual(
            len(list(generate_strongly_connected_components(constraints))),
            3 * nfe + 2,
            # "Initial constraints" only add two variables/equations
        )
        for i, (block, inputs) in enumerate(
            generate_strongly_connected_components(constraints)
        ):
            with TemporarySubsystemManager(to_fix=inputs):
                # The order is:
                #   algebraic -> derivative -> differential -> algebraic -> ...
                idx = i // 3
                mod = i % 3
                t = m.time[idx + 1]
                if t != time.last():
                    t_next = m.time.next(t)

                self.assertEqual(len(block.vars), 1)
                self.assertEqual(len(block.cons), 1)

                if mod == 0:
                    self.assertIs(block.vars[0], m.flow_out[t])
                    self.assertIs(block.cons[0], m.flow_out_eqn[t])
                elif mod == 1:
                    self.assertIs(block.vars[0], m.dhdt[t])
                    self.assertIs(block.cons[0], m.diff_eqn[t])
                elif mod == 2:
                    # Never get to mod == 2 when t == time.last()
                    self.assertIs(block.vars[0], m.height[t_next])
                    self.assertIs(block.cons[0], m.dhdt_disc_eq[t])

    def test_with_zero_coefficients(self):
        """Test where the blocks we identify are incorrect if we don't filter
        out variables with coefficients of zero
        """
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=1.0)
        m.eq1 = pyo.Constraint(expr=m.x[1] + 2 * m.x[2] + 0 * m.x[3] == 7)
        m.eq2 = pyo.Constraint(expr=m.x[1] + pyo.log(m.x[1]) == 0)
        blocks = generate_strongly_connected_components([m.eq1, m.eq2])
        blocks = [bl for (bl, _) in blocks]

        self.assertEqual(len(blocks[0].vars), 1)
        self.assertIs(blocks[0].vars[0], m.x[1])
        self.assertIs(blocks[0].cons[0], m.eq2)

        self.assertEqual(len(blocks[1].vars), 1)
        self.assertIs(blocks[1].vars[0], m.x[2])
        self.assertIs(blocks[1].cons[0], m.eq1)


@unittest.skipUnless(scipy_available, "SciPy is not available")
@unittest.skipUnless(networkx_available, "NetworkX is not available")
class TestSolveSCC(unittest.TestCase):
    def test_dynamic_backward_no_solver(self):
        nfe = 5
        m = make_dynamic_model(nfe=nfe, scheme="BACKWARD")
        time = m.time
        t0 = time.first()
        m.flow_in.fix()
        m.height[t0].fix()

        with self.assertRaisesRegex(RuntimeError, "An external solver is required*"):
            solve_strongly_connected_components(m)

        for t in time:
            if t == t0:
                self.assertTrue(m.height[t].fixed)
            else:
                self.assertFalse(m.height[t].fixed)
            self.assertFalse(m.flow_out[t].fixed)
            self.assertFalse(m.dhdt[t].fixed)
            self.assertTrue(m.flow_in[t].fixed)

    @unittest.skipUnless(
        pyo.SolverFactory("ipopt").available(), "IPOPT is not available"
    )
    def test_dynamic_backward(self):
        nfe = 5
        m = make_dynamic_model(nfe=nfe, scheme="BACKWARD")
        time = m.time
        t0 = time.first()
        m.flow_in.fix()
        m.height[t0].fix()

        solver = pyo.SolverFactory("ipopt")
        solve_kwds = {"tee": False}
        solve_strongly_connected_components(m, solver=solver, solve_kwds=solve_kwds)

        for con in m.component_data_objects(pyo.Constraint):
            # Sanity check that this is an equality constraint...
            self.assertEqual(pyo.value(con.upper), pyo.value(con.lower))

            # Assert that the constraint is satisfied within tolerance
            self.assertAlmostEqual(
                pyo.value(con.body), pyo.value(con.upper), delta=1e-7
            )

        for t in time:
            if t == t0:
                self.assertTrue(m.height[t].fixed)
            else:
                self.assertFalse(m.height[t].fixed)
            self.assertFalse(m.flow_out[t].fixed)
            self.assertFalse(m.dhdt[t].fixed)
            self.assertTrue(m.flow_in[t].fixed)

    def test_dynamic_forward(self):
        nfe = 5
        m = make_dynamic_model(nfe=nfe, scheme="FORWARD")
        time = m.time
        t0 = time.first()
        m.flow_in.fix()
        m.height[t0].fix()

        solve_strongly_connected_components(m)

        for con in m.component_data_objects(pyo.Constraint):
            # Sanity check that this is an equality constraint...
            self.assertEqual(pyo.value(con.upper), pyo.value(con.lower))

            # Assert that the constraint is satisfied within tolerance
            self.assertAlmostEqual(
                pyo.value(con.body), pyo.value(con.upper), delta=1e-7
            )

        for t in time:
            if t == t0:
                self.assertTrue(m.height[t].fixed)
            else:
                self.assertFalse(m.height[t].fixed)
            self.assertFalse(m.flow_out[t].fixed)
            self.assertFalse(m.dhdt[t].fixed)
            self.assertTrue(m.flow_in[t].fixed)

    def test_with_calc_var_kwds(self):
        m = pyo.ConcreteModel()
        m.v0 = pyo.Var()
        m.v1 = pyo.Var()
        m.v2 = pyo.Var(initialize=79703634.05074187)
        m.v2.fix()
        m.p0 = pyo.Param(initialize=3e5)
        m.p1 = pyo.Param(initialize=1.296e12)
        m.con0 = pyo.Constraint(expr=m.v0 == m.p0)
        m.con1 = pyo.Constraint(expr=0.0 == m.p1 * m.v1 / m.v0 + m.v2)
        calc_var_kwds = {"eps": 1e-7}
        # This solve fails to converge without raising the tolerance.
        # The secant method used to converge a linear variable-constraint
        # pair (such as con0, v0 and con1, v1) leaves us with a residual of
        # 1.48e-8 due to roundoff error.
        results = solve_strongly_connected_components(m, calc_var_kwds=calc_var_kwds)
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(m.v0.value, m.p0.value)
        self.assertAlmostEqual(m.v1.value, -18.4499152895)
        # -18.4499 ~= -v2*v0/p1

    def test_with_zero_coefficients(self):
        """Test where the blocks we identify are incorrect if we don't filter
        out variables with coefficients of zero
        """
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=1.0)
        m.eq1 = pyo.Constraint(expr=m.x[1] + 2 * m.x[2] + 0 * m.x[3] == 7)
        m.eq2 = pyo.Constraint(expr=m.x[1] + pyo.log(m.x[1]) == 0)
        results = solve_strongly_connected_components(m)

        self.assertAlmostEqual(m.x[1].value, 0.56714329)
        self.assertAlmostEqual(m.x[2].value, 3.21642835)
        self.assertEqual(m.x[3].value, 1.0)

    def test_with_inequalities(self):
        """Test that we correctly ignore inequalities"""
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=1.0)
        m.eq1 = pyo.Constraint(expr=m.x[1] + 2 * m.x[2] + 0 * m.x[3] == 7)
        m.eq2 = pyo.Constraint(expr=m.x[1] + pyo.log(m.x[1]) == 0)

        # Solving the system violates this inequality. That is fine. We happily
        # ignore it for the purpose of this equation solve.
        m.ineq1 = pyo.Constraint(expr=m.x[1] + 2 * m.x[2] + m.x[3] <= 3)

        results = solve_strongly_connected_components(m)

        self.assertAlmostEqual(m.x[1].value, 0.56714329)
        self.assertAlmostEqual(m.x[2].value, 3.21642835)
        self.assertEqual(m.x[3].value, 1.0)


if __name__ == "__main__":
    unittest.main()
