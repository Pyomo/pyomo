# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
#

import pyomo.common.unittest as unittest

import pyomo.environ as pyo

from pyomo.common.dependencies import numpy as np, scipy_available, numpy_available
from pyomo.common.log import LoggingIntercept
from pyomo.repn.plugins.standard_form import LinearStandardFormCompiler

import pyomo.core.base.constraint as constraint
import pyomo.core.base.objective as objective

for sol in ['glpk', 'cbc', 'gurobi', 'cplex', 'xpress']:
    linear_solver = pyo.SolverFactory(sol)
    if linear_solver.available(exception_flag=False):
        break
else:
    linear_solver = None


@unittest.skipUnless(
    scipy_available & numpy_available, "standard_form requires scipy and numpy"
)
class TestLinearStandardFormCompiler(unittest.TestCase):
    def push_templatization(self, mode):
        self.templatize_stack.append(
            (constraint.TEMPLATIZE_CONSTRAINTS, objective.TEMPLATIZE_OBJECTIVES)
        )
        constraint.TEMPLATIZE_CONSTRAINTS = mode
        objective.TEMPLATIZE_OBJECTIVES = mode

    def pop_templatization(self):
        constraint.TEMPLATIZE_CONSTRAINTS, objective.TEMPLATIZE_OBJECTIVES = (
            self.templatize_stack.pop()
        )

    def setUp(self):
        self.templatize_stack = []

    def test_linear_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var([1, 2, 3])
        m.c = pyo.Constraint(expr=m.x + 2 * m.y[1] >= 3)
        m.d = pyo.Constraint(expr=m.y[1] + 4 * m.y[3] <= 5)

        repn = LinearStandardFormCompiler().write(m)

        self.assertTrue(np.all(repn.c == np.array([0, 0, 0])))
        self.assertTrue(np.all(repn.A == np.array([[-1, -2, 0], [0, 1, 4]])))
        self.assertTrue(np.all(repn.rhs == np.array([-3, 5])))
        self.assertEqual(repn.rows, [(m.c, -1), (m.d, 1)])
        self.assertEqual(repn.columns, [m.x, m.y[1], m.y[3]])

    def test_almost_dense_linear_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var([1, 2, 3])
        m.c = pyo.Constraint(expr=m.x + 2 * m.y[1] + 4 * m.y[3] >= 10)
        m.d = pyo.Constraint(expr=5 * m.x + 6 * m.y[1] + 8 * m.y[3] <= 20)

        repn = LinearStandardFormCompiler().write(m)

        self.assertTrue(np.all(repn.c == np.array([0, 0, 0])))
        self.assertTrue(np.all(repn.A == np.array([[-1, -2, -4], [5, 6, 8]])))
        self.assertTrue(np.all(repn.rhs == np.array([-10, 20])))
        self.assertEqual(repn.rows, [(m.c, -1), (m.d, 1)])
        self.assertEqual(repn.columns, [m.x, m.y[1], m.y[3]])

    def test_linear_model_row_col_order(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var([1, 2, 3])
        m.c = pyo.Constraint(expr=m.x + 2 * m.y[1] >= 3)
        m.d = pyo.Constraint(expr=m.y[1] + 4 * m.y[3] <= 5)

        repn = LinearStandardFormCompiler().write(
            m, column_order=[m.y[3], m.y[2], m.x, m.y[1]], row_order=[m.d, m.c]
        )

        self.assertTrue(np.all(repn.c == np.array([0, 0, 0])))
        self.assertTrue(np.all(repn.A == np.array([[4, 0, 1], [0, -1, -2]])))
        self.assertTrue(np.all(repn.rhs == np.array([5, -3])))
        self.assertEqual(repn.rows, [(m.d, 1), (m.c, -1)])
        self.assertEqual(repn.columns, [m.y[3], m.x, m.y[1]])

    def test_linear_model_fixed_vars(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var([1, 2, 3, 4, 5])
        m.o = pyo.Objective(expr=5 * m.x + 3 * m.y[5] + 1)
        m.c = pyo.Constraint(expr=m.x + 2 * m.y[1] >= 3)
        m.d = pyo.Constraint(expr=m.y[1] + 4 * m.y[3] <= 5)

        m.y[3].fix(10)
        m.y[4].fix(100)
        m.y[5].fix(1000)

        repn = LinearStandardFormCompiler().write(m)

        self.assertTrue(np.all(repn.c == np.array([5, 0])))
        self.assertEqual(repn.c_offset, 3001)
        self.assertTrue(np.all(repn.A == np.array([[-1, -2], [0, 1]])))
        self.assertTrue(np.all(repn.rhs == np.array([-3, -35])))
        self.assertEqual(repn.rows, [(m.c, -1), (m.d, 1)])
        self.assertEqual(repn.columns, [m.x, m.y[1]])

    def test_suffix_warning(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var([1, 2, 3])
        m.c = pyo.Constraint(expr=m.x + 2 * m.y[1] >= 3)
        m.d = pyo.Constraint(expr=m.y[1] + 4 * m.y[3] <= 5)

        m.dual = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.b = pyo.Block()
        m.b.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

        with LoggingIntercept() as LOG:
            repn = LinearStandardFormCompiler().write(m)
        self.assertEqual(LOG.getvalue(), "")

        m.dual[m.c] = 5
        with LoggingIntercept() as LOG:
            repn = LinearStandardFormCompiler().write(m)
        self.assertEqual(
            LOG.getvalue(),
            "EXPORT Suffix 'dual' found on 1 block:\n"
            "    dual\n"
            "Standard Form compiler ignores export suffixes.  Skipping.\n",
        )

        m.b.dual[m.d] = 1
        with LoggingIntercept() as LOG:
            repn = LinearStandardFormCompiler().write(m)
        self.assertEqual(
            LOG.getvalue(),
            "EXPORT Suffix 'dual' found on 2 blocks:\n"
            "    dual\n"
            "    b.dual\n"
            "Standard Form compiler ignores export suffixes.  Skipping.\n",
        )

    def _verify_solution(self, soln, repn, eq):
        # clear out any old solution
        for v, val in soln:
            v.value = None
        for v in repn.x:
            v.value = None

        x = np.array(repn.x, dtype=object)
        ax = repn.A.todense() @ x

        def c_rule(m, i):
            if eq:
                return ax[i] == repn.b[i]
            else:
                return ax[i] <= repn.b[i]

        try:
            self.push_templatization(False)
            test_model = pyo.ConcreteModel()
            test_model.o = pyo.Objective(expr=repn.c[[1], :].todense()[0] @ x)
            test_model.c = pyo.Constraint(range(len(repn.b)), rule=c_rule)
        finally:
            self.pop_templatization()
        linear_solver.solve(test_model, tee=True)

        # Propagate any solution back to the original variables
        for v, expr in repn.eliminated_vars:
            v.value = pyo.value(expr)
        self.assertEqual(*zip(*((v.value, val) for v, val in soln)))

    @unittest.skipIf(
        linear_solver is None, 'verifying results requires a linear solver'
    )
    def test_alternative_forms(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var(
            [0, 1, 3], bounds=lambda m, i: (-1 * (i % 2) * 5, 10 - 12 * (i // 2))
        )
        m.c = pyo.Constraint(expr=m.x + 2 * m.y[1] >= 3)
        m.d = pyo.Constraint(expr=m.y[1] + 4 * m.y[3] <= 5)
        m.e = pyo.Constraint(expr=pyo.inequality(-2, m.y[0] + 1 + 6 * m.y[1], 7))
        m.f = pyo.Constraint(expr=m.x + m.y[0] + 2 == 10)
        m.o = pyo.Objective([1, 3], rule=lambda m, i: m.x + i * 5 * m.y[i])
        m.o[1].sense = pyo.maximize

        col_order = [m.x, m.y[0], m.y[1], m.y[3]]

        m.o[1].deactivate()
        linear_solver.solve(m)
        m.o[1].activate()
        soln = [(v, v.value) for v in col_order]

        repn = LinearStandardFormCompiler().write(m, column_order=col_order)

        self.assertEqual(
            repn.rows, [(m.c, -1), (m.d, 1), (m.e, 1), (m.e, -1), (m.f, 1), (m.f, -1)]
        )
        self.assertEqual(repn.x, [m.x, m.y[0], m.y[1], m.y[3]])
        ref = np.array(
            [
                [-1, 0, -2, 0],
                [0, 0, 1, 4],
                [0, 1, 6, 0],
                [0, -1, -6, 0],
                [1, 1, 0, 0],
                [-1, -1, 0, 0],
            ]
        )
        self.assertTrue(np.all(repn.A == ref))
        self.assertTrue(np.all(repn.b == np.array([-3, 5, 6, 3, 8, -8])))
        self.assertTrue(np.all(repn.c == np.array([[-1, 0, -5, 0], [1, 0, 0, 15]])))
        self._verify_solution(soln, repn, False)

        repn = LinearStandardFormCompiler().write(
            m, nonnegative_vars=True, column_order=col_order
        )

        self.assertEqual(
            repn.rows, [(m.c, -1), (m.d, 1), (m.e, 1), (m.e, -1), (m.f, 1), (m.f, -1)]
        )
        self.assertEqual(
            list(map(str, repn.x)),
            ['_neg_0', '_pos_0', 'y[0]', '_neg_2', '_pos_2', '_neg_3'],
        )
        ref = np.array(
            [
                [1, -1, 0, 2, -2, 0],
                [0, 0, 0, -1, 1, -4],
                [0, 0, 1, -6, 6, 0],
                [0, 0, -1, 6, -6, 0],
                [-1, 1, 1, 0, 0, 0],
                [1, -1, -1, 0, 0, 0],
            ]
        )
        self.assertTrue(np.all(repn.A == ref))
        self.assertTrue(np.all(repn.b == np.array([-3, 5, 6, 3, 8, -8])))
        self.assertTrue(
            np.all(repn.c == np.array([[1, -1, 0, 5, -5, 0], [-1, 1, 0, 0, 0, -15]]))
        )
        self._verify_solution(soln, repn, False)

        repn = LinearStandardFormCompiler().write(
            m, slack_form=True, column_order=col_order
        )

        self.assertEqual(repn.rows, [(m.c, 1), (m.d, 1), (m.e, 1), (m.f, 1)])
        self.assertEqual(
            list(map(str, repn.x)),
            ['x', 'y[0]', 'y[1]', 'y[3]', '_slack_0', '_slack_1', '_slack_2'],
        )
        self.assertEqual(
            list(v.bounds for v in repn.x),
            [(None, None), (0, 10), (-5, 10), (-5, -2), (None, 0), (0, None), (-9, 0)],
        )
        ref = np.array(
            [
                [1, 0, 2, 0, 1, 0, 0],
                [0, 0, 1, 4, 0, 1, 0],
                [0, 1, 6, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0],
            ]
        )
        self.assertTrue(np.all(repn.A == ref))
        self.assertTrue(np.all(repn.b == np.array([3, 5, -3, 8])))
        self.assertTrue(
            np.all(
                repn.c == np.array([[-1, 0, -5, 0, 0, 0, 0], [1, 0, 0, 15, 0, 0, 0]])
            )
        )
        self._verify_solution(soln, repn, True)

        repn = LinearStandardFormCompiler().write(
            m, mixed_form=True, column_order=col_order
        )

        self.assertEqual(
            repn.rows, [(m.c, -1), (m.d, 1), (m.e, 1), (m.e, -1), (m.f, 0)]
        )
        self.assertEqual(list(map(str, repn.x)), ['x', 'y[0]', 'y[1]', 'y[3]'])
        self.assertEqual(
            list(v.bounds for v in repn.x), [(None, None), (0, 10), (-5, 10), (-5, -2)]
        )
        ref = np.array(
            [[1, 0, 2, 0], [0, 0, 1, 4], [0, 1, 6, 0], [0, 1, 6, 0], [1, 1, 0, 0]]
        )
        self.assertTrue(np.all(repn.A == ref))
        self.assertTrue(np.all(repn.b == np.array([3, 5, 6, -3, 8])))
        self.assertTrue(np.all(repn.c == np.array([[-1, 0, -5, 0], [1, 0, 0, 15]])))
        # Note that the mixed_form solution is a mix of inequality and
        # equality constraints, so we cannot (easily) reuse the
        # _verify_solutions helper (as in the above cases):
        # self._verify_solution(soln, repn, False)

        repn = LinearStandardFormCompiler().write(
            m, slack_form=True, nonnegative_vars=True, column_order=col_order
        )

        self.assertEqual(repn.rows, [(m.c, 1), (m.d, 1), (m.e, 1), (m.f, 1)])
        self.assertEqual(
            list(map(str, repn.x)),
            [
                '_neg_0',
                '_pos_0',
                'y[0]',
                '_neg_2',
                '_pos_2',
                '_neg_3',
                '_neg_4',
                '_slack_1',
                '_neg_6',
            ],
        )
        self.assertEqual(
            list(v.bounds for v in repn.x),
            [
                (0, None),
                (0, None),
                (0, 10),
                (0, 5),
                (0, 10),
                (2, 5),
                (0, None),
                (0, None),
                (0, 9),
            ],
        )
        ref = np.array(
            [
                [-1, 1, 0, -2, 2, 0, -1, 0, 0],
                [0, 0, 0, -1, 1, -4, 0, 1, 0],
                [0, 0, 1, -6, 6, 0, 0, 0, -1],
                [-1, 1, 1, 0, 0, 0, 0, 0, 0],
            ]
        )
        self.assertTrue(np.all(repn.A == ref))
        self.assertTrue(np.all(repn.b == np.array([3, 5, -3, 8])))
        ref = np.array([[1, -1, 0, 5, -5, 0, 0, 0, 0], [-1, 1, 0, 0, 0, -15, 0, 0, 0]])
        self.assertTrue(np.all(repn.c == ref))
        self._verify_solution(soln, repn, True)

    def test_keep_range_constraints(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var([0, 1, 3], bounds=lambda m, i: (0, 10))
        # Pure lower-bound constraint
        m.c = pyo.Constraint(expr=m.x + 2 * m.y[1] >= 3)
        # Pure upper-bound constraint
        m.d = pyo.Constraint(expr=m.y[1] + 4 * m.y[3] <= 5)
        # Range constraint: -2 <= y[0] + 1 + 6*y[1] <= 7  →  -3 <= y[0] + 6*y[1] <= 6
        m.e = pyo.Constraint(expr=pyo.inequality(-2, m.y[0] + 1 + 6 * m.y[1], 7))
        # Equality
        m.f = pyo.Constraint(expr=m.x + m.y[0] == 8)
        m.o = pyo.Objective(expr=5 * m.x)

        col_order = [m.x, m.y[0], m.y[1], m.y[3]]

        # --- mixed_form + keep_range_constraints ---
        repn = LinearStandardFormCompiler().write(
            m, mixed_form=True, keep_range_constraints=True, column_order=col_order
        )
        # m.e: single range row (bound_type=2); all others are normal mixed rows
        self.assertEqual(repn.rows, [(m.c, -1), (m.d, 1), (m.e, 2), (m.f, 0)])
        ref_A = np.array([[1, 0, 2, 0], [0, 0, 1, 4], [0, 1, 6, 0], [1, 1, 0, 0]])
        self.assertTrue(np.all(repn.A.toarray() == ref_A))
        # m.e: rhs = ub - offset = 7 - 1 = 6
        self.assertTrue(np.all(repn.rhs == np.array([3, 5, 6, 8])))
        # rhs_range: only m.e is a range row; range = 7 - (-2) = 9
        self.assertTrue(np.all(repn.rhs_range == np.array([0.0, 0.0, 9.0, 0.0])))

        # --- default form + keep_range_constraints ---
        repn2 = LinearStandardFormCompiler().write(
            m, keep_range_constraints=True, column_order=col_order
        )
        # lb-only (m.c) → negated ≤ row; ub-only (m.d) → ≤ row;
        # range (m.e) → single row; equality (m.f) → two rows (ub + negated lb)
        self.assertEqual(
            repn2.rows, [(m.c, -1), (m.d, 1), (m.e, 2), (m.f, 1), (m.f, -1)]
        )
        self.assertTrue(np.all(repn2.rhs_range == np.array([0.0, 0.0, 9.0, 0.0, 0.0])))

        # --- without keep_range_constraints m.e still splits into two rows ---
        repn3 = LinearStandardFormCompiler().write(
            m, mixed_form=True, column_order=col_order
        )
        e_rows = [
            (r.constraint, r.bound_type) for r in repn3.rows if r.constraint is m.e
        ]
        self.assertEqual(e_rows, [(m.e, 1), (m.e, -1)])
        # rhs_range is None when keep_range_constraints=False
        self.assertIsNone(repn3.rhs_range)

        # --- slack_form + keep_range_constraints must raise ---
        with self.assertRaises(ValueError):
            LinearStandardFormCompiler().write(
                m, slack_form=True, keep_range_constraints=True
            )

    def test_allow_nonlinear_constraints(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.c_lin = pyo.Constraint(expr=m.x + m.y <= 5)
        m.c_nl = pyo.Constraint(expr=m.x**2 + m.y <= 3)
        m.o = pyo.Objective(expr=m.x + m.y)

        # Default (allow_nonlinear=False) must raise on the nonlinear constraint.
        with self.assertRaises(Exception):
            LinearStandardFormCompiler().write(m, mixed_form=True)

        # allow_nonlinear=True: nonlinear constraint is collected separately;
        # the linear constraint still appears in A.
        repn = LinearStandardFormCompiler().write(
            m, mixed_form=True, allow_nonlinear=True
        )
        self.assertEqual(repn.nonlinear_constraints, [m.c_nl])
        self.assertEqual(repn.nonlinear_objectives, [])
        # Only the linear constraint appears in A.
        self.assertEqual(len(repn.rows), 1)
        self.assertEqual(repn.rows[0].constraint, m.c_lin)
        # Linear objective is still compiled into c.
        self.assertEqual(repn.objectives, [m.o])
        self.assertTrue(np.all(repn.c.toarray() != 0))

    def test_allow_nonlinear_objective(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.c_lin = pyo.Constraint(expr=m.x + m.y <= 5)
        m.o_nl = pyo.Objective(expr=m.x**2 + m.y)

        # Default must raise on the nonlinear objective.
        with self.assertRaises(Exception):
            LinearStandardFormCompiler().write(m, mixed_form=True)

        repn = LinearStandardFormCompiler().write(
            m, mixed_form=True, allow_nonlinear=True
        )
        # Nonlinear objective is NOT compiled into c; it appears in nonlinear_objectives.
        self.assertEqual(repn.nonlinear_objectives, [m.o_nl])
        self.assertEqual(repn.objectives, [])
        # c is empty (no linear objectives).
        self.assertEqual(repn.c.shape[0], 0)
        # The linear constraint is still compiled normally.
        self.assertEqual(len(repn.rows), 1)
        self.assertEqual(repn.rows[0].constraint, m.c_lin)

    def test_allow_nonlinear_mixed(self):
        """Linear constraints/objectives compiled; nonlinear ones passed through."""
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.c_lin = pyo.Constraint(expr=m.x + 2 * m.y >= 1)
        m.c_nl = pyo.Constraint(expr=m.x * m.y <= 4)
        m.o_lin = pyo.Objective(expr=m.x + m.y)

        repn = LinearStandardFormCompiler().write(
            m, mixed_form=True, allow_nonlinear=True
        )

        # Exactly one linear row, one nonlinear constraint.
        self.assertEqual(len(repn.rows), 1)
        self.assertEqual(repn.rows[0].constraint, m.c_lin)
        self.assertEqual(repn.nonlinear_constraints, [m.c_nl])
        # Linear objective compiles normally.
        self.assertEqual(repn.objectives, [m.o_lin])
        self.assertEqual(repn.nonlinear_objectives, [])
        # Both variables appear as columns (referenced by the linear constraint).
        col_ids = {id(v) for v in repn.columns}
        self.assertIn(id(m.x), col_ids)
        self.assertIn(id(m.y), col_ids)

    def test_inf_bounds_normalized(self):
        """Constraints returning ±inf bounds are treated as unbounded (None).

        Both pyomo.kernel and AML constraints can return ±inf from
        to_bounded_expression() when the user explicitly passes float('inf').
        LinearStandardFormCompiler must normalize these so that a one-sided
        constraint is not misclassified as a range constraint, and a fully
        unbounded constraint is skipped rather than emitted as a range row.
        """
        import pyomo.kernel as pmo

        # --- kernel ---
        mk = pmo.block()
        mk.x = pmo.variable()
        # lb=-inf (unbounded below) → should become a pure ≤ row, not a range row
        mk.c_ub = pmo.constraint(ub=2.0, body=mk.x)
        # ub=+inf (unbounded above) → should become a pure ≥ row, not a range row
        mk.c_lb = pmo.constraint(lb=-3.0, body=mk.x)
        # Explicit finite range → should still be a range row
        mk.c_rng = pmo.constraint((-1.0, mk.x, 4.0))

        repn = LinearStandardFormCompiler().write(
            mk, mixed_form=True, keep_range_constraints=True
        )
        by_con = {r.constraint: r.bound_type for r in repn.rows}
        self.assertEqual(by_con[mk.c_ub], 1)  # ≤, not range
        self.assertEqual(by_con[mk.c_lb], -1)  # ≥, not range
        self.assertEqual(by_con[mk.c_rng], 2)  # finite range
        rhs_map = {r.constraint: repn.rhs[i] for i, r in enumerate(repn.rows)}
        self.assertEqual(rhs_map[mk.c_ub], 2.0)
        self.assertEqual(rhs_map[mk.c_lb], -3.0)
        self.assertEqual(rhs_map[mk.c_rng], 4.0)  # ub of range row
        rr_map = {r.constraint: repn.rhs_range[i] for i, r in enumerate(repn.rows)}
        self.assertEqual(rr_map[mk.c_ub], 0.0)
        self.assertEqual(rr_map[mk.c_lb], 0.0)
        self.assertEqual(rr_map[mk.c_rng], 5.0)  # 4 - (-1)

        # --- AML: explicit float('inf') in RangedExpression ---
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        # One-sided: (-inf, x, 5) — lb should be treated as unbounded
        m.c_ub = pyo.Constraint(rule=lambda m: (float('-inf'), m.x, 5))
        # Fully unbounded: (-inf, x, inf) — should be skipped entirely
        m.c_skip = pyo.Constraint(rule=lambda m: (float('-inf'), m.x, float('inf')))

        repn2 = LinearStandardFormCompiler().write(
            m, mixed_form=True, keep_range_constraints=True
        )
        by_con2 = {r.constraint: r.bound_type for r in repn2.rows}
        self.assertEqual(by_con2[m.c_ub], 1)  # ≤, not range
        self.assertNotIn(m.c_skip, by_con2)  # fully unbounded: skipped

    def test_extra_valid_ctypes(self):
        """Component types in extra_valid_ctypes are permitted but not compiled."""
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.y)
        m.sos = pyo.SOSConstraint(var=m.x, sos=1)

        # Without extra_valid_ctypes, LSFC raises on the SOSConstraint.
        with self.assertRaises(ValueError):
            LinearStandardFormCompiler().write(m, mixed_form=True)

        # With extra_valid_ctypes, the SOSConstraint is silently skipped.
        repn = LinearStandardFormCompiler().write(
            m, mixed_form=True, extra_valid_ctypes=[pyo.SOSConstraint]
        )
        # Only m.y appears in the objective; m.x[i] are unreferenced by
        # linear constraints/objectives so not included in repn.columns.
        self.assertEqual(len(repn.rows), 0)
        col_ids = {id(v) for v in repn.columns}
        self.assertIn(id(m.y), col_ids)
        self.assertNotIn(id(m.x[1]), col_ids)


class TestTemplatedLinearStandardFormCompiler(TestLinearStandardFormCompiler):
    def setUp(self):
        super().setUp()
        self.push_templatization(True)

    def tearDown(self):
        self.pop_templatization()
