# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from pyomo.environ import (
    SolverFactory,
    ConcreteModel,
    Block,
    Var,
    Constraint,
    Objective,
    NonNegativeReals,
    Suffix,
    value,
    minimize,
    maximize,
    Set,
    Binary,
)
import pytest
from pyomo.opt import TerminationCondition
import pyomo.common.unittest as unittest
from pyomo.solvers.plugins.solvers.cuopt_direct import cuopt_available, CUOPTDirect


def _cuopt_supports_qc():
    return cuopt_available and CUOPTDirect._supports_quadratic_constraint


def _cuopt_at_least(*required):
    """True iff cuOpt is available and at least the given (major, minor[, patch]) version."""
    if not cuopt_available:
        return False
    try:
        version = tuple(int(p) for p in CUOPTDirect._version[: len(required)])
    except (AttributeError, TypeError, ValueError):
        return False
    return version >= required


@unittest.pytest.mark.solver("cuopt")
class CUOPTTests(unittest.TestCase):
    @unittest.skipIf(not cuopt_available, "The CuOpt solver is not available")
    def test_values_and_rc(self):
        m = ConcreteModel()

        m.dual = Suffix(direction=Suffix.IMPORT)
        m.rc = Suffix(direction=Suffix.IMPORT)

        m.x = Var(domain=NonNegativeReals)
        m.top_con = Constraint(expr=m.x >= 0)

        m.b1 = Block()
        m.b1.y = Var(domain=NonNegativeReals)
        m.b1.con1 = Constraint(expr=m.x + m.b1.y <= 10)

        m.b1.subb = Block()
        m.b1.subb.z = Var(domain=NonNegativeReals)
        m.b1.subb.con2 = Constraint(expr=2 * m.b1.y + m.b1.subb.z >= 8)

        m.b2 = Block()
        m.b2.w = Var(domain=NonNegativeReals)
        m.b2.con3 = Constraint(expr=m.b1.subb.z - m.b2.w == 2)

        # Minimize cost = 1*x + 2*y + 3*z + 0.5*w
        m.obj = Objective(
            expr=1.0 * m.x + 2.0 * m.b1.y + 3.0 * m.b1.subb.z + 0.5 * m.b2.w,
            sense=minimize,
        )

        opt = SolverFactory('cuopt')
        res = opt.solve(m, save_results=False)

        expected_rc = [1.0, 0.0, 0.0, 2.5]
        expected_val = [0.0, 3.0, 2.0, 0.0]
        expected_dual = [0.0, 0.0, 1.0, 2.0]

        for i, v in enumerate([m.x, m.b1.y, m.b1.subb.z, m.b2.w]):
            self.assertAlmostEqual(m.rc[v], expected_rc[i], places=5)
            self.assertAlmostEqual(value(v), expected_val[i], places=5)

        for i, c in enumerate([m.top_con, m.b1.con1, m.b1.subb.con2, m.b2.con3]):
            self.assertAlmostEqual(m.dual[c], expected_dual[i], places=5)

        # Set max iteration = 1 to abort problem due to iteration limit
        opt.options["iteration_limit"] = 1
        res = opt.solve(m)
        self.assertEqual(res.solver.status, "aborted")

    @unittest.skipIf(not cuopt_available, "The CuOpt solver is not available")
    def test_errors_exceptions(self):
        items = ["a", "b", "c", "d", "e"]
        value = {"a": 10, "b": 7, "c": 25, "d": 15, "e": 6}
        weight = {"a": 4, "b": 3, "c": 9, "d": 6, "e": 2}
        capacity = 12

        m = ConcreteModel()
        m.I = Set(initialize=items)
        m.x = Var(m.I, within=Binary)
        m.obj1 = Objective(expr=sum(value[i] * m.x[i] for i in m.I), sense=maximize)
        m.cap = Constraint(expr=sum(weight[i] * m.x[i] for i in m.I) <= capacity)

        opt = SolverFactory('cuopt')
        res = opt.solve(m)

        # Add second dummy objective
        m.obj2 = Objective(expr=sum(m.x[i] for i in m.I), sense=minimize)
        # Raise error due to multiple objectives
        with pytest.raises(
            ValueError, match=r"Solver interface does not support multiple objectives."
        ):
            res = opt.solve(m)
        m.obj2.deactivate()

        # Raise error due to unsupported suffix
        m.slack = Suffix(direction=Suffix.IMPORT)
        with pytest.raises(RuntimeError, match=r"cannot extract solution suffix=slack"):
            res = opt.solve(m)

    @unittest.skipIf(not cuopt_available, "The CuOpt solver is not available")
    def test_infeasible_trivial_constraint(self):
        m = ConcreteModel()
        m.x = Var(domain=NonNegativeReals)
        m.fixed_var = Var()
        m.fixed_var.fix(5)
        m.obj = Objective(expr=m.x, sense=minimize)
        # trivial constraint that is infeasible: 5 <= 3
        m.bad_con = Constraint(expr=m.fixed_var <= 3)

        opt = SolverFactory('cuopt')
        with pytest.raises(ValueError, match=r"Trivial constraint.*infeasible"):
            opt.solve(m, skip_trivial_constraints=True)

    @unittest.skipUnless(
        _cuopt_at_least(26, 4),
        "cuOpt UnboundedOrInfeasible status (11) requires cuOpt 26.04 or later",
    )
    def test_unbounded_or_infeasible_status(self):
        # An LP with no variable bounds and an unbounded objective triggers
        # cuOpt's presolver to return UnboundedOrInfeasible (status 11), which
        # the plugin maps to TerminationCondition.infeasibleOrUnbounded.
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.obj = Objective(expr=m.x + m.y, sense=minimize)

        opt = SolverFactory('cuopt')
        res = opt.solve(m, load_solutions=False)

        self.assertEqual(res.solver.termination_condition, "infeasibleOrUnbounded")
        self.assertEqual(res.solver.status, "warning")
        self.assertEqual(res.solution[0].status, "unsure")

    @unittest.skipIf(not cuopt_available, "The CuOpt solver is not available")
    def test_nonlinear_constraint_rejected(self):
        m = ConcreteModel()
        m.x = Var(domain=NonNegativeReals)
        m.y = Var(domain=NonNegativeReals)
        m.obj = Objective(expr=m.x + m.y, sense=minimize)
        # nonlinear constraint (degree > 2)
        m.nonlin_con = Constraint(expr=m.x**3 <= 10)

        opt = SolverFactory('cuopt')
        with pytest.raises(ValueError, match=r"contains nonlinear terms"):
            opt.solve(m)

    @unittest.skipIf(not cuopt_available, "The CuOpt solver is not available")
    def test_quadratic_objective_matrix(self):
        m = ConcreteModel()
        m.x = Var(domain=NonNegativeReals)
        m.y = Var(domain=NonNegativeReals)
        m.obj = Objective(expr=m.x**2 + 2 * m.x + 4 * m.y**2 + 3.0, sense=minimize)
        m.c = Constraint(expr=m.x + m.y >= 1)

        opt = SolverFactory('cuopt')
        opt._set_instance(m)

        q_values = opt._solver_model.get_quadratic_objective_values()
        q_indices = opt._solver_model.get_quadratic_objective_indices()
        q_offsets = opt._solver_model.get_quadratic_objective_offsets()
        self.assertEqual(list(q_values), [1.0, 4.0])
        self.assertEqual(list(q_indices), [0, 1])
        self.assertEqual(list(q_offsets), [0, 1, 2])
        self.assertAlmostEqual(opt._solver_model.get_objective_offset(), 3.0)
        c = opt._solver_model.get_objective_coefficients()
        self.assertAlmostEqual(c[0], 2.0)
        self.assertAlmostEqual(c[1], 0.0)

    @unittest.skipUnless(
        _cuopt_supports_qc(),
        "cuOpt quadratic constraint support requires add_quadratic_constraint",
    )
    def test_quadratic_constraint_soc(self):
        m = ConcreteModel()
        m.x0 = Var(bounds=(None, None))
        m.x1 = Var(bounds=(None, None))
        m.x2 = Var(bounds=(None, None))
        m.y = Var(bounds=(0, 5))
        m.obj = Objective(expr=3 * m.x0 + 2 * m.x1 + m.x2, sense=minimize)
        m.soc = Constraint(
            expr=m.x0 * m.x0 + m.x1 * m.x1 + m.x2 * m.x2 - m.y * m.y <= 0
        )
        m.lin = Constraint(expr=m.x0 + m.x1 + 3 * m.x2 >= 1)

        opt = SolverFactory('cuopt')
        opt._set_instance(m)

        qcs = opt._solver_model.get_quadratic_constraints()
        self.assertEqual(len(qcs), 1)
        qc = qcs[0]
        self.assertEqual(qc["constraint_row_type"], "L")
        self.assertAlmostEqual(qc["rhs_value"], 0.0)
        self.assertEqual(list(qc["vals"]), [1.0, 1.0, 1.0, -1.0])
        self.assertEqual(list(qc["rows"]), [0, 1, 2, 3])
        self.assertEqual(list(qc["cols"]), [0, 1, 2, 3])
        self.assertTrue(opt._has_quadratic_content)

        res = opt.solve(m)
        self.assertEqual(res.solver.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(value(m.obj), -13.548638872057857, places=4)
        self.assertAlmostEqual(value(m.x0), -3.874621914903146, places=4)
        self.assertAlmostEqual(value(m.x1), -2.129788270186565, places=4)
        self.assertAlmostEqual(value(m.x2), 2.3348034130247104, places=4)
        self.assertAlmostEqual(value(m.y), 5.0, places=4)

    @unittest.skipUnless(
        _cuopt_supports_qc(),
        "cuOpt quadratic constraint support requires add_quadratic_constraint",
    )
    def test_mip_with_quadratic_constraint_rejected(self):
        m = ConcreteModel()
        m.x = Var(within=Binary)
        m.y = Var(bounds=(0, None))
        m.z = Var()
        m.obj = Objective(expr=m.x + m.z, sense=minimize)
        m.soc = Constraint(expr=m.z * m.z - m.y * m.y <= 0)

        opt = SolverFactory('cuopt')
        with pytest.raises(
            ValueError, match=r"does not support mixed-integer problems with quadratic"
        ):
            opt.solve(m)


if __name__ == "__main__":
    unittest.main()
