# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import os
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
from pyomo.common.dependencies import attempt_import
from pyomo.opt import check_available_solvers
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.unittest as unittest
from pyomo.solvers.plugins.solvers.cuopt_direct import cuopt_available


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

    @unittest.skipIf(not cuopt_available, "The CuOpt solver is not available")
    def test_nonlinear_constraint_rejected(self):
        m = ConcreteModel()
        m.x = Var(domain=NonNegativeReals)
        m.y = Var(domain=NonNegativeReals)
        m.obj = Objective(expr=m.x + m.y, sense=minimize)
        # nonlinear constraint: x * y <= 10
        m.nonlin_con = Constraint(expr=m.x * m.y <= 10)

        opt = SolverFactory('cuopt')
        with pytest.raises(ValueError, match=r"contains nonlinear terms"):
            opt.solve(m)


if __name__ == "__main__":
    unittest.main()
