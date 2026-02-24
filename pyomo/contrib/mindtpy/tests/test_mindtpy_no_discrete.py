# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________


from pyomo.opt import TerminationCondition
import pyomo.common.unittest as unittest

from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    SolverFactory,
    Var,
    minimize,
    value,
)

required_nlp_solvers = 'ipopt'
# Open-source (or generally available) solver pair used by MindtPy tests that
# don't require persistent commercial solvers.
if SolverFactory('appsi_highs').available(exception_flag=False) and SolverFactory(
    'appsi_highs'
).version() >= (1, 7, 0):
    short_circuit_required_solvers = ('ipopt', 'appsi_highs')
else:
    short_circuit_required_solvers = ('ipopt', 'glpk')

short_circuit_subsolvers_available = all(
    SolverFactory(s).available(exception_flag=False)
    for s in short_circuit_required_solvers
)


@unittest.skipIf(
    not short_circuit_subsolvers_available,
    'Required subsolvers %s are not available' % (short_circuit_required_solvers,),
)
class TestMindtPyShortCircuitNoDiscrete(unittest.TestCase):
    def test_no_discrete_decisions_short_circuit_loads_values(self):
        """Regression test for MindtPy short-circuit with no discrete decisions.

        If all discrete variables are fixed, MindtPy should directly solve the
        original model (LP/NLP) and still return a valid SolverResults and load
        primal values onto the provided model.
        """
        m = ConcreteModel()
        m.x = Var(domain=NonNegativeReals)
        m.y = Var(domain=Binary)
        m.y.fix(0)

        # Nonlinear constraint forces the NLP short-circuit branch
        m.c = Constraint(expr=m.x**2 >= 1 + m.y)
        m.objective = Objective(expr=m.x, sense=minimize)

        with SolverFactory('mindtpy') as opt:
            results = opt.solve(
                m,
                strategy='OA',
                mip_solver=short_circuit_required_solvers[1],
                nlp_solver=short_circuit_required_solvers[0],
            )

        self.assertIsNotNone(results)
        self.assertIn(
            results.solver.termination_condition,
            [
                TerminationCondition.optimal,
                TerminationCondition.locallyOptimal,
                TerminationCondition.feasible,
            ],
        )
        # Core regression: primal values must be loaded onto the model.
        self.assertIsNotNone(
            m.x.value,
            "x.value is None; MindtPy did not populate primal values in the short-circuit path",
        )
        obj_val = value(m.objective.expr, exception=False)
        self.assertIsNotNone(
            obj_val, "Objective evaluates to None; model variables were not populated"
        )
        # Sanity check on the solution (y is fixed to 0, so x >= 1)
        self.assertGreaterEqual(m.x.value, 1.0 - 1e-6)
        self.assertAlmostEqual(m.x.value, 1.0, places=4)
        self.assertAlmostEqual(obj_val, 1.0, places=4)

    def test_short_circuit_infeasible_nlp_returns_valid_results(self):
        """Infeasible NLP short-circuit should return results, not load bad data."""
        m = ConcreteModel()
        m.x = Var(domain=NonNegativeReals)
        m.y = Var(domain=Binary)
        m.y.fix(0)

        # Infeasible: x >= 0 AND x^2 <= -1
        m.c1 = Constraint(expr=m.x >= 0)
        m.c2 = Constraint(expr=m.x**2 <= -1)
        m.objective = Objective(expr=m.x, sense=minimize)

        with SolverFactory('mindtpy') as opt:
            results = opt.solve(
                m,
                strategy='OA',
                mip_solver=short_circuit_required_solvers[1],
                nlp_solver=short_circuit_required_solvers[0],
            )

        self.assertIsNotNone(results)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.infeasible
        )

    def test_short_circuit_linear_model_uses_lp_path(self):
        """Linear model with fixed discrete should use LP short-circuit."""
        m = ConcreteModel()
        m.x = Var(domain=NonNegativeReals)
        m.y = Var(domain=Binary)
        m.y.fix(0)

        # Pure LP (polynomial degree 1)
        m.c = Constraint(expr=m.x >= 1)
        m.objective = Objective(expr=m.x, sense=minimize)

        with SolverFactory('mindtpy') as opt:
            results = opt.solve(
                m,
                strategy='OA',
                mip_solver=short_circuit_required_solvers[1],
                nlp_solver=short_circuit_required_solvers[0],
            )

        self.assertIsNotNone(results)
        self.assertAlmostEqual(m.x.value, 1.0, places=4)
